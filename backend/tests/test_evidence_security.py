"""Evidence security suite in one place: every upload and download negative, session isolation,
the human-only decision tools, the keyless prompt-injection check (the paper_d footnote changes
nothing and nothing is auto-approved) and the LLM-path proof that document text only ever
reaches the model inside <document> blocks, never in the system prompt."""
from __future__ import annotations

import itertools
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.config import settings
from app.core.llm import MockLLM
from app.core.orchestrator import Orchestrator
from app.core.pharmstate import PharmState
from app.core.provenance import file_sha256
from app.core.store import SessionStore
from app.evidence import extract_tools, ingest, llm_access, screen_tools
from app.evidence import store as evstore
from app.evidence.config_tools import validate_criteria, validate_template
from app.evidence.prompts import EVIDENCE_SYSTEM
from app.tools.base import ToolContext
from app.tools.builtins import default_registry
from tests import evidence_fixtures as fx
from tests.test_evidence_export import assert_download_refused
from tests.test_evidence_extract import TEMPLATE
from tests.test_evidence_llm import FakeLLM
from tests.test_evidence_records_api import _setup
from tests.test_evidence_screen import CRITERIA

EV = "/api/sessions/{sid}/evidence"
TRAVERSAL = ["../pharmagent.db", "%2e%2e%2fpharmagent.db", "..%2F..%2Fpharmagent.db", "/etc/passwd",
             "%2Fetc%2Fpasswd", "pharmagent.db", "evidence_x.csv", "..\\pharmagent.db"]
_DOC_RE = re.compile(r"<document\b[^>]*>.*?</document>", re.DOTALL)


@pytest.fixture(scope="module")
def papers(tmp_path_factory) -> dict[str, Path]:
    d = tmp_path_factory.mktemp("papers")
    paths = fx.build_all(d)
    paths["paper_d_clean"] = fx.build_pdf(d / "clean", "paper_d", injection=False)
    paths["encrypted"] = fx.encrypted_pdf(d)
    paths["no_text"] = fx.no_text_pdf(d)
    paths["many_pages"] = fx.many_pages_pdf(d, 301)
    return paths


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    evstore.configure(":memory:")
    main.orch = Orchestrator(llm=MockLLM(), clock=lambda c=itertools.count(): f"t{next(c)}",
                             store=SessionStore(":memory:"))
    token = settings.api_token
    yield TestClient(main.app)
    settings.api_token = token
    llm_access.set_llm(None)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    st = evstore.configure(":memory:")
    llm_access.set_llm(None)
    yield SimpleNamespace(store=st, ctx=ToolContext(data_dir=str(tmp_path)), sid="sess_sec", dir=tmp_path)
    llm_access.set_llm(None)


def _sid(client, h=None) -> str:
    return client.post("/api/sessions", headers=h or {}).json()["id"]


def _upload(client, sid, blob, name="paper.pdf", h=None):
    data = blob.read_bytes() if isinstance(blob, Path) else blob
    return client.post(EV.format(sid=sid) + "/upload", headers=h or {},
                       files={"file": (name, data, "application/pdf")})


def _err(r) -> str:
    return r.json()["error"]["message"]


def _files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix != ".db"]


# ------------------------------------------------------------- upload negatives --

def test_upload_rejects_non_pdf_suffix(client, papers):
    r = _upload(client, _sid(client), papers["paper_a"], name="paper.csv")
    assert r.status_code == 400 and "PDF only" in _err(r)


def test_upload_rejects_bad_magic_and_leaves_no_file(client, tmp_path):
    r = _upload(client, _sid(client), b"ID,TIME,DV\n1,0,0\n", name="data.pdf")
    assert r.status_code == 400 and "not a PDF" in _err(r) and _files(tmp_path) == []


def test_upload_rejects_oversize_and_leaves_no_file(client, tmp_path):
    r = _upload(client, _sid(client), b"%PDF-1.4\n" + b"\0" * (26 * 1024 * 1024), name="big.pdf")
    assert r.status_code == 413 and _files(tmp_path) == []


def test_upload_rejects_encrypted_and_page_cap(client, tmp_path, papers):
    sid = _sid(client)
    r = _upload(client, sid, papers["encrypted"], name="enc.pdf")
    assert r.status_code == 422 and "encrypted" in _err(r).lower()
    r = _upload(client, sid, papers["many_pages"], name="long.pdf")
    assert r.status_code == 422 and "page" in _err(r).lower()
    assert _files(tmp_path) == []


def test_upload_filename_traversal_is_sanitised_and_stored_by_hash(client, tmp_path, papers):
    r = _upload(client, _sid(client), papers["paper_a"], name="../../etc/x.pdf")
    assert r.status_code == 200
    src = r.json()["source"]
    assert src["filename"] == "x.pdf" and "pdf_path" not in src and "parse_path" not in src
    stored = [p.name for p in _files(tmp_path)]
    assert f"{src['sha256']}.pdf" in stored and not any(".." in n for n in stored)
    assert all(p.resolve().is_relative_to(tmp_path.resolve()) for p in _files(tmp_path))


def test_no_text_pdf_is_recorded_but_never_screened(client, papers):
    sid = _sid(client)
    r = _upload(client, sid, papers["no_text"], name="scan.pdf")
    assert r.status_code == 200 and r.json()["source"]["parse_status"] == "no_text"
    client.post(EV.format(sid=sid) + "/criteria", json={"criteria": CRITERIA})
    r = client.post(EV.format(sid=sid) + "/screen", json={})
    assert r.status_code == 400 and "no parsed publications" in _err(r)


def test_ingest_tool_refuses_paths_outside_the_evidence_dir_and_sha_mismatch(client, tmp_path, papers):
    sid = _sid(client)
    outside = tmp_path / "outside.pdf"
    outside.write_bytes(papers["paper_a"].read_bytes())
    sha = file_sha256(outside)
    with pytest.raises(ValueError, match="evidence directory"):
        main.orch.run_tool(sid, "ingest_evidence_source", "evidence",
                           {"path": str(outside), "sha256": sha, "actor": "x"}, actor="x")
    inside = ingest.evidence_dir(tmp_path) / f"{sha}.pdf"
    inside.write_bytes(outside.read_bytes())
    with pytest.raises(ValueError, match="sha256"):
        main.orch.run_tool(sid, "ingest_evidence_source", "evidence",
                           {"path": str(inside), "sha256": "0" * 64, "actor": "x"}, actor="x")


# ----------------------------------------------------------- session isolation --

def test_other_sessions_sources_pages_and_pdf_are_404(client, papers):
    s1, s2 = _sid(client), _sid(client)
    src = _upload(client, s1, papers["paper_a"]).json()["source"]
    assert client.get(EV.format(sid=s2) + f"/sources/{src['id']}/pdf").status_code == 404
    assert client.get(EV.format(sid=s2) + f"/sources/{src['id']}/pages/1").status_code == 404
    assert client.get(EV.format(sid=s1) + f"/sources/{src['id']}/pages/1").status_code == 200
    snap = client.get(EV.format(sid=s2)).json()
    assert snap["sources"] == [] and snap["counts"]["sources"] == 0


def test_other_sessions_screens_and_records_cannot_be_decided(client, papers):
    s1 = _setup(client, papers, names=("paper_a",))
    client.post(EV.format(sid=s1) + "/extract", json={})
    screen = client.get(EV.format(sid=s1)).json()["screens"][0]
    record = client.get(EV.format(sid=s1)).json()["records"][0]
    s2 = _sid(client)
    r = client.post(EV.format(sid=s2) + "/screen/decide",
                    json={"screen_id": screen["id"], "decision": "reject", "reason": "x"})
    assert r.status_code == 404
    r = client.post(EV.format(sid=s2) + "/records/decide",
                    json={"record_id": record["id"], "decision": "approve", "reason": "x"})
    assert r.status_code == 404
    snap = client.get(EV.format(sid=s1)).json()
    assert all(r["status"] == "proposed" for r in snap["records"])


def test_bearer_token_guards_every_evidence_route(client, papers):
    settings.api_token = "secret-token"
    h = {"Authorization": "Bearer secret-token"}
    sid = _sid(client, h)
    for method, path in (("GET", ""), ("POST", "/criteria"), ("POST", "/template"), ("POST", "/screen"),
                         ("POST", "/screen/decide"), ("POST", "/extract"), ("POST", "/records/verify"),
                         ("POST", "/records/decide"), ("POST", "/export"), ("GET", "/dataset/x.csv"),
                         ("GET", "/sources/src_x/pages/1"), ("GET", "/sources/src_x/pdf")):
        r = client.request(method, EV.format(sid=sid) + path, json={} if method == "POST" else None)
        assert r.status_code == 401, (method, path, r.status_code)
    assert _upload(client, sid, papers["paper_a"]).status_code == 401
    assert client.get(EV.format(sid=sid), headers=h).status_code == 200


# --------------------------------------------------------------- decisions --

def test_decision_tools_are_human_only(client, papers):
    """A model-issued decide_* call (no confirm) is refused; the HTTP route supplies confirm itself."""
    sid = _setup(client, papers, names=("paper_a",))
    client.post(EV.format(sid=sid) + "/extract", json={})
    screen = client.get(EV.format(sid=sid)).json()["screens"][0]
    record = client.get(EV.format(sid=sid)).json()["records"][0]
    for tool, args in (("decide_screening", {"screen_id": screen["id"]}),
                       ("decide_records", {"record_id": record["id"]})):
        with pytest.raises(ValueError, match="confirm"):
            main.orch.run_tool(sid, tool, "evidence", {**args, "decision": "approve", "reason": "x"}, actor="llm")
    reg = default_registry()
    assert not {t.name for t in reg.for_agent("evidence")} & {"decide_screening", "decide_records"}
    assert client.get(EV.format(sid=sid)).json()["records"][0]["status"] == "proposed"


def test_failed_verification_can_never_be_approved(client, papers, tmp_path):
    sid = _setup(client, papers, names=("paper_a",))
    client.post(EV.format(sid=sid) + "/extract", json={})
    rec = client.get(EV.format(sid=sid)).json()["records"][0]
    src = client.get(EV.format(sid=sid)).json()["sources"][0]
    (ingest.evidence_dir(tmp_path) / f"{src['sha256']}.pdf").write_bytes(b"%PDF-1.4\n%changed\n")
    client.post(EV.format(sid=sid) + "/records/verify", json={})
    r = client.post(EV.format(sid=sid) + "/records/decide",
                    json={"record_id": rec["id"], "decision": "approve", "reason": "please"})
    assert r.status_code == 400 and "failed literal verification" in _err(r)
    assert client.get(EV.format(sid=sid), params={"status": "approved"}).json()["records"] == []


# --------------------------------------------------------- download negatives --

def _exported(client, papers) -> tuple[str, dict]:
    sid = _setup(client, papers, names=("paper_a", "paper_d"))
    client.post(EV.format(sid=sid) + "/extract", json={})
    recs = client.get(EV.format(sid=sid)).json()["records"]
    client.post(EV.format(sid=sid) + "/records/decide", json={"decisions": [
        {"record_id": r["id"], "decision": "approve", "reason": "ok"} for r in recs if r["verify_status"] == "pass"]})
    out = client.post(EV.format(sid=sid) + "/export", json={}).json()["result"]
    return sid, out


@pytest.mark.parametrize("name", TRAVERSAL)
def test_download_refuses_traversal_and_foreign_names(client, papers, name):
    sid, _ = _exported(client, papers)
    assert_download_refused(client.get(EV.format(sid=sid) + f"/dataset/{name}"), name)


def test_download_before_export_and_from_another_session_is_404(client, papers):
    s1 = _setup(client, papers, names=("paper_a",))
    assert client.get(EV.format(sid=s1) + f"/dataset/evidence_{s1}_20260101T000000Z.csv").status_code == 404
    s2, out = _exported(client, papers)
    name = Path(out["long_path"]).name
    assert client.get(EV.format(sid=s1) + f"/dataset/{name}").status_code == 404
    assert client.get(EV.format(sid=s2) + f"/dataset/{name}").status_code == 200


def test_download_never_serves_a_file_outside_data_dir(client, papers, tmp_path, monkeypatch):
    sid, out = _exported(client, papers)
    outside = tmp_path.parent / f"{sid}_outside.csv"
    outside.write_text("secret\n")
    try:
        state = main.orch.get_session(sid).state
        monkeypatch.setattr(state, "evidence_dataset_path", str(outside))      # a corrupted mirror
        assert client.get(EV.format(sid=sid) + f"/dataset/{outside.name}").status_code == 404
        assert client.get(EV.format(sid=sid) + f"/dataset/{Path(out['wide_path']).name}").status_code == 200
    finally:
        outside.unlink(missing_ok=True)


# ------------------------------------------------------- prompt injection (keyless) --

def _screen_rows(env, names: tuple[str, ...], papers) -> dict[str, dict]:
    env.store.save_config(env.sid, "criteria", validate_criteria(CRITERIA), "alice")
    srcs = {n: ingest.ingest_pdf(papers[n].read_bytes(), f"{n}.pdf", env.sid, "alice", data_dir=env.dir,
                                 store=env.store) for n in names}
    res = screen_tools.screen_publications(PharmState(session_id=env.sid), env.ctx, {})
    by_src = {s["source_id"]: s for s in res.result["screens"]}
    return {n: by_src[s.id] for n, s in srcs.items()}


def _strip(row: dict) -> tuple:
    return (row["verdict"], row["score"], row["status"],
            tuple((c["criterion_id"], c["met"], len(c["evidence"])) for c in row["criteria"]))


def test_keyless_injection_footnote_changes_no_verdict_and_nothing_is_auto_approved(env, papers):
    dirty = _screen_rows(env, ("paper_a", "paper_b", "paper_c", "paper_d"), papers)
    env.store = evstore.configure(":memory:")
    clean = _screen_rows(env, ("paper_a", "paper_b", "paper_c", "paper_d_clean"), papers)
    for n in ("paper_a", "paper_b", "paper_c"):
        assert _strip(dirty[n]) == _strip(clean[n])
    assert _strip(dirty["paper_d"]) == _strip(clean["paper_d_clean"])
    assert all(r["status"] == "proposed" for r in [*dirty.values(), *clean.values()])
    assert env.store.list_screens(env.sid, status="approved") == []
    words = " ".join(ev["term"] for r in dirty.values() for c in r["criteria"] for ev in c["evidence"]).lower()
    assert "approve" not in words and "ignore" not in words and "instruction" not in words


def test_keyless_extraction_treats_the_injection_as_text_and_approves_nothing(env, papers):
    _screen_rows(env, ("paper_d",), papers)
    for s in env.store.list_screens(env.sid):
        env.store.decide_screen(s.id, "approved", "alice", "agree")
    env.store.save_config(env.sid, "template", validate_template({"fields": [
        *TEMPLATE["fields"], {"name": "notes", "type": "text"}]}), "alice")
    extract_tools.extract_evidence(PharmState(session_id=env.sid), env.ctx, {})
    rows = env.store.list_records(env.sid)
    assert rows and all(r.status == "proposed" for r in rows)
    assert not any(fx.INJECTION.lower() in r.value.lower() for r in rows)
    assert not any(r.value == "2.41" for r in rows)
    assert env.store.approved_record_hashes(env.sid) == []


# ---------------------------------------------------------- LLM path: wrapping --

def _outside_documents(user: str) -> str:
    return _DOC_RE.sub("", user)


def _page_texts(env, names) -> list[str]:
    texts = []
    for src in env.store.list_sources(env.sid):
        if src.filename in {f"{n}.pdf" for n in names}:
            texts.extend((p.get("text") or "") for p in ingest.load_pages(src)["pages"])
    return texts


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"[.\n]", text) if len(s.strip()) >= 25]


def test_llm_prompts_carry_page_text_only_inside_document_blocks(env, papers):
    names = ("paper_a", "paper_d")
    _screen_rows(env, names, papers)
    llm = FakeLLM(None)                                  # keyed-looking, every answer invalid -> fallback
    llm_access.set_llm(llm)
    assert llm_access.llm_available()
    screen_tools.screen_publications(PharmState(session_id=env.sid), env.ctx, {"source_ids": [
        s.id for s in env.store.list_sources(env.sid)]})
    for s in env.store.list_screens(env.sid):
        env.store.decide_screen(s.id, "approved", "alice", "agree")
    env.store.save_config(env.sid, "template", validate_template(TEMPLATE), "alice")
    extract_tools.extract_evidence(PharmState(session_id=env.sid), env.ctx, {})
    assert len(llm.calls) >= 2 + 2                       # 2 screens + >= 1 page per included source
    sentences = [s for t in _page_texts(env, names) for s in _sentences(t)]
    assert sentences and any(fx.INJECTION in s for s in sentences)
    for call in llm.calls:
        assert call["system"] == EVIDENCE_SYSTEM
        assert not any(s in call["system"] for s in sentences)
        user = call["user"]
        assert "<document" in user and "</document>" in user
        outside = _outside_documents(user)
        assert not any(s in outside for s in sentences), outside[:300]
        assert fx.INJECTION not in outside
        assert user.rindex("</document>") < len(user) and outside.strip()      # the task follows the block
    assert all(r.status == "proposed" for r in env.store.list_records(env.sid))
