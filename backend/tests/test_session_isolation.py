"""Multi-user isolation (external review F-08, F-09, F-11, F-21, F-22, F-23).

Skills honour ownership; uploads and reports are scoped per session; per-user bearer tokens give each user a
distinct owner; persisted datasets load lazily into a bounded LRU.
"""
from __future__ import annotations

import hashlib
import itertools
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.config import settings
from app.core.llm import MockLLM
from app.core.orchestrator import Orchestrator
from app.core.store import SessionStore
from app.tools.base import ToolContext
from app.tools.data_tools import load_dataset

SAMPLES = Path(__file__).parent.parent / "sample_data"
ORAL = SAMPLES / "oral_pk.csv"
BE = SAMPLES / "theoph_pk.csv"
ALICE = {"Authorization": "Bearer alice-secret"}
BOB = {"Authorization": "Bearer bob-secret"}


def _clock():
    c = itertools.count()
    return lambda: f"t{next(c)}"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(settings, "api_token", None)
    monkeypatch.setattr(settings, "api_tokens", "alice=alice-secret,bob=bob-secret", raising=False)
    main.orch = Orchestrator(llm=MockLLM(), clock=_clock(), store=SessionStore(":memory:"))
    return TestClient(main.app, raise_server_exceptions=False)


def _session(client, headers) -> str:
    r = client.post("/api/sessions", headers=headers)
    assert r.status_code == 200, r.text
    return r.json()["id"]


def _upload(client, sid, headers, name="a.csv", data=None):
    return client.post(f"/api/sessions/{sid}/upload", headers=headers,
                       files={"file": (name, data if data is not None else ORAL.read_bytes(), "text/csv")})


# ── F-11 per-user tokens ─────────────────────────────────────────────────────────
def test_user_tokens_give_distinct_owners_and_isolate_sessions(client):
    a, b = _session(client, ALICE), _session(client, BOB)
    assert main.orch.sessions[a].owner == "alice" and main.orch.sessions[b].owner == "bob"
    assert client.get(f"/api/sessions/{a}/state", headers=ALICE).status_code == 200
    assert client.get(f"/api/sessions/{a}/state", headers=BOB).status_code == 403


def test_unknown_or_missing_token_is_401_when_user_tokens_set(client):
    assert client.post("/api/sessions").status_code == 401
    assert client.post("/api/sessions", headers={"Authorization": "Bearer nope"}).status_code == 401


def test_user_token_actor_is_the_user_name_never_the_token(client):
    a = _session(client, ALICE)
    r = client.get(f"/api/sessions/{a}/audit", headers=ALICE)
    assert r.status_code == 200
    assert "alice-secret" not in r.text
    assert main.orch.sessions[a].audit.to_list()[0]["actor"] == "alice"


def test_legacy_single_token_and_open_mode_unchanged(client, monkeypatch):
    monkeypatch.setattr(settings, "api_tokens", None, raising=False)
    monkeypatch.setattr(settings, "api_token", "legacy")
    expected = "token:" + hashlib.sha256(b"legacy").hexdigest()[:16]
    assert main.current_owner("Bearer legacy") == expected
    assert client.post("/api/sessions").status_code == 401
    monkeypatch.setattr(settings, "api_token", None)
    assert main.current_owner(None) is None


def test_legacy_token_still_accepted_alongside_user_tokens(client, monkeypatch):
    monkeypatch.setattr(settings, "api_token", "legacy")
    assert main.current_owner("Bearer legacy").startswith("token:")
    assert main.current_owner("Bearer alice-secret") == "alice"


@pytest.mark.parametrize("raw", ["alice", "alice=", "=x", "a=1,b=1", "token:x=1", "anonymous=1"])
def test_malformed_user_tokens_are_rejected(raw):
    from app.config import Settings
    with pytest.raises(ValueError):
        Settings(_env_file=None, api_tokens=raw)


# ── F-08 skill ownership ─────────────────────────────────────────────────────────
def _capture(client, headers, name="sk") -> int:
    sid = _session(client, headers)
    main.orch.sessions[sid].commands = [{"agent": "nca", "tool": "compute_nca", "args": {}}]
    return client.post(f"/api/sessions/{sid}/capture-skill", headers=headers, json={"name": name}).status_code


def test_skill_of_alice_is_invisible_and_immutable_for_bob(client):
    assert _capture(client, ALICE) == 200
    assert client.get("/api/skills/sk", headers=ALICE).status_code == 200
    assert client.get("/api/skills/sk", headers=BOB).status_code == 403
    assert client.get("/api/skills/sk/markdown", headers=BOB).status_code == 403
    assert client.post("/api/skills/sk/run", headers=BOB, json={"dataset_path": str(ORAL)}).status_code == 403
    assert client.delete("/api/skills/sk", headers=BOB).status_code == 403
    assert main.orch.skills.get("sk") is not None


def test_bob_cannot_overwrite_alices_skill_by_capturing_same_name(client):
    assert _capture(client, ALICE) == 200
    assert _capture(client, BOB) in (403, 409)
    sk = main.orch.skills.get("sk")
    assert sk.owner == "alice" and sk.version == 1


def test_owner_may_update_and_delete_own_skill(client):
    assert _capture(client, ALICE) == 200
    assert _capture(client, ALICE) == 200
    assert main.orch.skills.get("sk").version == 2
    assert client.delete("/api/skills/sk", headers=ALICE).status_code == 200


def test_open_mode_skills_still_work(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(settings, "api_token", None)
    monkeypatch.setattr(settings, "api_tokens", None, raising=False)
    main.orch = Orchestrator(llm=MockLLM(), clock=_clock(), store=SessionStore(":memory:"))
    c = TestClient(main.app, raise_server_exceptions=False)
    assert _capture(c, {}) == 200
    assert c.get("/api/skills/sk").status_code == 200
    assert c.delete("/api/skills/sk").status_code == 200


# ── F-09 uploads ─────────────────────────────────────────────────────────────────
def test_uploads_land_in_a_per_session_directory(client, tmp_path):
    a, b = _session(client, ALICE), _session(client, BOB)
    pa = Path(_upload(client, a, ALICE).json()["metadata"]["dataset_path"])
    pb = Path(_upload(client, b, BOB).json()["metadata"]["dataset_path"])
    assert pa != pb and pa.exists() and pb.exists()
    assert pa.parent == tmp_path / "sessions" / a
    assert pb.parent == tmp_path / "sessions" / b


def test_session_cannot_load_another_sessions_upload(client):
    a, b = _session(client, ALICE), _session(client, BOB)
    pa = _upload(client, a, ALICE).json()["metadata"]["dataset_path"]
    sess_b = main.orch.sessions[b]
    with pytest.raises(ValueError, match="not permitted"):
        load_dataset(sess_b.state, sess_b.ctx, {"path": pa})
    sess_a = main.orch.sessions[a]
    assert load_dataset(sess_a.state, sess_a.ctx, {"path": pa}).writes["dataset_id"]
    assert load_dataset(sess_b.state, sess_b.ctx, {"path": str(ORAL)}).writes["dataset_id"]  # sample_data ok


def test_upload_appends_dataset_uploaded_audit_entry_with_raw_sha(client):
    a = _session(client, ALICE)
    data = ORAL.read_bytes()
    assert _upload(client, a, ALICE, data=data).status_code == 200
    entries = [e for e in main.orch.sessions[a].audit.to_list() if e["action"] == "dataset_uploaded"]
    assert len(entries) == 1
    assert hashlib.sha256(data).hexdigest() in entries[0]["reason"]
    assert entries[0]["actor"] == "alice"
    assert main.orch.sessions[a].audit.verify()


def test_skill_replay_on_own_upload_from_another_session_is_allowed_other_users_refused(client):
    a1 = _session(client, ALICE)
    path = _upload(client, a1, ALICE).json()["metadata"]["dataset_path"]
    assert _capture(client, ALICE) == 200
    ok = client.post("/api/skills/sk/run", headers=ALICE, json={"dataset_path": path})
    assert ok.status_code == 200, ok.text
    assert main.orch.skills.get("sk")  # still there
    # bob cannot even run it (F-08); and bob cannot replay on alice's file via his own skill
    _capture(client, BOB, name="bobsk")
    bad = client.post("/api/skills/bobsk/run", headers=BOB, json={"dataset_path": path})
    assert bad.status_code == 400


# ── F-22 bounded upload read ─────────────────────────────────────────────────────
def test_upload_reads_in_bounded_chunks_and_rejects_oversize_with_413(client, monkeypatch):
    from starlette.datastructures import UploadFile
    sizes: list[int] = []
    original = UploadFile.read

    async def spy(self, size=-1):
        sizes.append(size)
        return await original(self, size)

    monkeypatch.setattr(UploadFile, "read", spy)
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 64)
    a = _session(client, ALICE)
    r = _upload(client, a, ALICE, data=b"a,b\n" + b"1,2\n" * 100)
    assert r.status_code == 413
    assert sizes and all(0 < s <= (1 << 20) for s in sizes)


# ── F-21 per-session reports ─────────────────────────────────────────────────────
def _report(client, headers, csv: Path):
    sid = _session(client, headers)
    path = _upload(client, sid, headers, data=csv.read_bytes()).json()["metadata"]["dataset_path"]
    o = main.orch
    o.run_tool(sid, "load_dataset", "data_manager", {"path": path, "dataset_id": "shared_ds"})
    o.run_tool(sid, "compute_nca", "nca", {})
    r = client.post(f"/api/sessions/{sid}/report", headers=headers)
    assert r.status_code == 200, r.text
    return sid, r.json()


def test_two_sessions_same_dataset_id_write_distinct_reports(client, tmp_path):
    a, ra = _report(client, ALICE, ORAL)
    b, rb = _report(client, BOB, BE)
    pa, pb = Path(main.orch.sessions[a].state.report_path), Path(main.orch.sessions[b].state.report_path)
    assert pa != pb and pa.exists() and pb.exists()
    assert pa.parent == tmp_path / "reports" / a and pb.parent == tmp_path / "reports" / b
    assert pa.name == pb.name == "report_shared_ds.docx"
    assert hashlib.sha256(pa.read_bytes()).hexdigest() != hashlib.sha256(pb.read_bytes()).hexdigest()
    da = client.get(f"/api/sessions/{a}/report/{pa.name}", headers=ALICE)
    db = client.get(f"/api/sessions/{b}/report/{pb.name}", headers=BOB)
    assert da.content == pa.read_bytes() and db.content == pb.read_bytes()
    ma = client.get(f"/api/sessions/{a}/provenance/report", headers=ALICE).json()
    assert ma["session_id"] == a
    assert hashlib.sha256(pa.read_bytes()).hexdigest() in str(ma)


def test_download_refuses_a_report_path_outside_the_session_directory(client, tmp_path):
    a, _ = _report(client, ALICE, ORAL)
    stray = tmp_path / "report_shared_ds.docx"
    stray.write_bytes(b"not mine")
    sess = main.orch.sessions[a]
    sess.state = sess.state.model_copy(update={"report_path": str(stray)})
    assert client.get(f"/api/sessions/{a}/report/{stray.name}", headers=ALICE).status_code == 404
    assert client.get(f"/api/sessions/{a}/provenance/report", headers=ALICE).status_code == 404


# ── F-23 lazy dataset loading ────────────────────────────────────────────────────
def _persisted_orch(tmp_path, n=3, **kw):
    db = str(tmp_path / "s.db")
    first = Orchestrator(llm=MockLLM(), clock=_clock(), store=SessionStore(db))
    ids = []
    for i in range(n):
        s = first.create_session(owner=f"u{i}")
        first.run_tool(s.id, "load_dataset", "data_manager", {"path": str(ORAL), "dataset_id": f"ds{i}"})
        ids.append(s.id)
    second = Orchestrator(llm=MockLLM(), clock=_clock(), store=SessionStore(db), **kw)
    return second, ids


def test_restart_does_not_load_every_dataframe(tmp_path):
    o, ids = _persisted_orch(tmp_path)
    assert set(o.sessions) == set(ids)
    assert all(s.ctx.dataset_store.loaded_ids() == [] for s in o.sessions.values())


def test_first_access_loads_and_tools_still_see_the_dataset(tmp_path):
    o, ids = _persisted_orch(tmp_path)
    s = o.sessions[ids[0]]
    dsid = s.state.dataset_id
    assert dsid in s.ctx.dataset_store
    assert len(s.ctx.dataset_store[dsid]) > 0
    assert s.ctx.dataset_store.loaded_ids() == [dsid]
    assert o.sessions[ids[1]].ctx.dataset_store.loaded_ids() == []


def test_lru_evicts_oldest_and_reloads_on_demand(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "max_loaded_datasets", 2, raising=False)
    o, ids = _persisted_orch(tmp_path)
    stores = [o.sessions[i].ctx.dataset_store for i in ids]
    dsids = [o.sessions[i].state.dataset_id for i in ids]
    for st, d in zip(stores, dsids):
        st[d]
    assert stores[0].loaded_ids() == []                       # evicted
    assert stores[1].loaded_ids() == [dsids[1]] and stores[2].loaded_ids() == [dsids[2]]
    assert len(stores[0][dsids[0]]) > 0                       # transparently reloaded
    assert stores[1].loaded_ids() == []                       # now the oldest, evicted


def test_lazy_load_refuses_a_file_changed_since_startup(tmp_path):
    import shutil
    src = tmp_path / "up.csv"
    shutil.copy(ORAL, src)
    db = str(tmp_path / "s.db")
    first = Orchestrator(llm=MockLLM(), clock=_clock(), store=SessionStore(db))
    s = first.create_session()
    first.run_tool(s.id, "load_dataset", "data_manager", {"path": str(ORAL), "dataset_id": "d"})
    second = Orchestrator(llm=MockLLM(), clock=_clock(), store=SessionStore(db))
    st = second.sessions[s.id].ctx.dataset_store
    st._paths["d"] = (str(src), "0" * 64)                     # simulate a sha mismatch at first access
    assert st.get("d") is None


def test_toolcontext_default_store_is_still_a_plain_mapping():
    ctx = ToolContext()
    ctx.dataset_store["x"] = 1
    assert ctx.dataset_store.get("x") == 1 and "x" in ctx.dataset_store
