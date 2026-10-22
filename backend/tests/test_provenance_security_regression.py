"""Owner gate 5: focused security regression for the provenance routes and the rewritten upload path."""
from __future__ import annotations

import hashlib
import itertools
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.datastructures import FormData

import app.main as main
from app.config import settings
from app.core.llm import MockLLM
from app.core.orchestrator import Orchestrator
from app.core.store import SessionStore

SAMPLE = Path(__file__).parent.parent / "sample_data" / "oral_pk.csv"
ROUTES = ["/provenance/export/nca", "/provenance/export/control/nonmem", "/provenance/cdisc",
          "/provenance/report", "/provenance/report/272", "/provenance/memo"]
AUTH = {"Authorization": "Bearer secret-token"}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    token = settings.api_token
    main.orch = Orchestrator(llm=MockLLM(), clock=lambda c=itertools.count(): f"t{next(c)}",
                             store=SessionStore(":memory:"))
    yield TestClient(main.app)
    settings.api_token = token


# ── authorization of every /provenance route; no cross-session access ─────────────────────────
@pytest.mark.parametrize("route", ROUTES)
def test_provenance_routes_require_the_token(client, route):
    settings.api_token = "secret-token"
    sid = client.post("/api/sessions", headers=AUTH).json()["id"]
    assert client.get(f"/api/sessions/{sid}{route}").status_code == 401
    assert client.get(f"/api/sessions/{sid}{route}", headers={"Authorization": "Bearer nope"}).status_code == 401


@pytest.mark.parametrize("route", ROUTES)
def test_provenance_routes_refuse_another_owners_session(client, route):
    settings.api_token = "secret-token"
    other = main.orch.create_session(owner="token:someone-else").id
    assert client.get(f"/api/sessions/{other}{route}", headers=AUTH).status_code == 403


@pytest.mark.parametrize("route", ROUTES)
def test_provenance_routes_404_for_an_unknown_session(client, route):
    assert client.get(f"/api/sessions/sess_nope{route}").status_code == 404


# ── manifests bind the exact downloaded bytes (2.7.2 and memo; others in test_export_provenance) ──
def test_272_and_memo_manifests_bind_the_exact_downloaded_bytes(client):
    from tests.memo_fixtures import full_chain, full_state

    sid = client.post("/api/sessions").json()["id"]
    sess = main.orch.get_session(sid)
    sess.state = full_state().model_copy(update={"session_id": sid, "run_provenance": sess.state.run_provenance})
    sess.audit = full_chain(genesis=(sid, dict(sess.state.run_provenance)))
    main.orch.run_tool(sid, "build_briefing_memo", "report", {}, actor="alice")
    main.orch.run_tool(sid, "generate_272", "report", {})
    st = main.orch.get_session(sid).state
    for path, dl, man in ((st.memo_path, "memo", "/provenance/memo"),
                          (st.regulatory_report_path, "report/272", "/provenance/report/272")):
        body = client.get(f"/api/sessions/{sid}/{dl}/{Path(path).name}")
        assert body.status_code == 200
        m = client.get(f"/api/sessions/{sid}{man}").json()
        assert m["export"]["sha256"] == hashlib.sha256(body.content).hexdigest()
        assert body.headers["X-PharmAgent-Run-Provenance-Status"] == m["run_provenance_status"] == "verified"


# ── the rewritten upload path ───────────────────────────────────────────────────────────────────
@pytest.fixture
def form_closes(monkeypatch):
    closed = []
    original = FormData.close

    async def spy(self):
        closed.append(True)
        await original(self)

    monkeypatch.setattr(FormData, "close", spy)
    return closed


def _csvs(d: Path) -> list[Path]:
    return [p for p in d.iterdir() if p.suffix == ".csv"]


def test_a_malformed_multipart_body_is_a_client_error_not_a_crash(client):
    sid = client.post("/api/sessions").json()["id"]
    r = client.post(f"/api/sessions/{sid}/upload", content=b"--garbage\r\nnot multipart",
                    headers={"Content-Type": "multipart/form-data; boundary=zzz"})
    assert 400 <= r.status_code < 500


def test_an_oversized_upload_is_413_and_leaves_no_file(client, tmp_path, monkeypatch, form_closes):
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 64)
    sid = client.post("/api/sessions").json()["id"]
    r = client.post(f"/api/sessions/{sid}/upload", files={"file": ("big.csv", b"a,b\n" + b"1,2\n" * 100, "text/csv")})
    assert r.status_code == 413 and _csvs(tmp_path) == [] and form_closes


def test_a_csv_parse_failure_removes_the_stored_file(client, tmp_path, form_closes):
    sid = client.post("/api/sessions").json()["id"]
    r = client.post(f"/api/sessions/{sid}/upload", files={"file": ("bad.csv", b"\x00\x01\x02\xff\xfe", "text/csv")})
    assert r.status_code in (200, 400)
    if r.status_code == 400:
        assert _csvs(tmp_path) == []
    assert form_closes


@pytest.mark.parametrize("files,data,code", [
    ({"other": ("x.csv", b"a\n1\n", "text/csv")}, None, 422),                     # no 'file' field
    ({"file": ("x.txt", b"a\n1\n", "text/plain")}, None, 400),                    # wrong suffix
])
def test_bad_upload_forms_are_refused_and_the_form_is_closed(client, tmp_path, form_closes, files, data, code):
    sid = client.post("/api/sessions").json()["id"]
    r = client.post(f"/api/sessions/{sid}/upload", files=files, data=data)
    assert r.status_code == code and _csvs(tmp_path) == [] and form_closes


def test_more_than_one_file_is_refused(client, tmp_path):
    sid = client.post("/api/sessions").json()["id"]
    r = client.post(f"/api/sessions/{sid}/upload",
                    files=[("file", ("a.csv", b"a\n1\n", "text/csv")), ("file", ("b.csv", b"a\n1\n", "text/csv"))])
    assert 400 <= r.status_code < 500 and _csvs(tmp_path) == []


def test_a_successful_upload_closes_the_form(client, tmp_path, form_closes):
    sid = client.post("/api/sessions").json()["id"]
    r = client.post(f"/api/sessions/{sid}/upload", files={"file": ("oral_pk.csv", SAMPLE.read_bytes(), "text/csv")})
    assert r.status_code == 200 and form_closes and len(_csvs(tmp_path)) == 1
