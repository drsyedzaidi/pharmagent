"""HTTP layer for the briefing memo: POST /memo and GET /memo/{filename} mirror the
/report endpoints -- same bearer auth, same filename-must-match path safety, and the
run goes through the registry choke point (audit entry, actor recorded)."""
import itertools
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.config import settings
from app.core.llm import MockLLM
from app.core.orchestrator import Orchestrator
from app.core.store import SessionStore

THEOPH = str(Path(__file__).parent.parent / "sample_data" / "theoph_pk.csv")
DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    main.orch = Orchestrator(llm=MockLLM(), clock=lambda c=itertools.count(): f"t{next(c)}",
                             store=SessionStore(":memory:"))
    token = settings.api_token
    yield TestClient(main.app)
    settings.api_token = token


def _analysed_session(client, headers=None) -> str:
    sid = client.post("/api/sessions", headers=headers).json()["id"]
    r = client.post(f"/api/sessions/{sid}/workflow/start", headers=headers,
                    json={"workflow": "nca_full", "params": {"path": THEOPH}})
    assert r.json()["status"] == "awaiting_review"
    return sid


def test_post_builds_the_memo_and_get_downloads_the_same_file(client):
    sid = _analysed_session(client)
    r = client.post(f"/api/sessions/{sid}/memo", json={"title": "Exposure briefing"})
    assert r.status_code == 200
    body = r.json()
    res = body["result"]
    assert body["tool"] == "build_briefing_memo" and body["agent"] == "report"
    assert res["status"] == "ok" and res["untraced_numbers"] == [] and res["title"] == "Exposure briefing"
    assert body["audit_ok"] is True and body["state"]["memo_path"] == res["memo_path"]

    d = client.get(f"/api/sessions/{sid}/memo/{res['filename']}")
    assert d.status_code == 200 and d.headers["content-type"] == DOCX
    assert d.content[:2] == b"PK" and d.content == Path(res["memo_path"]).read_bytes()


def test_body_is_optional(client):
    sid = _analysed_session(client)
    assert client.post(f"/api/sessions/{sid}/memo").status_code == 200


def test_the_run_is_audited_with_the_actor(client):
    sid = _analysed_session(client)
    client.post(f"/api/sessions/{sid}/memo")
    audit = client.get(f"/api/sessions/{sid}/audit").json()
    entry = audit["entries"][-1]
    assert (entry["tool"], entry["agent"], entry["actor"]) == ("build_briefing_memo", "report", "anonymous")
    assert audit["integrity"]["chain_ok"] is True


def test_no_analysis_yet_is_a_400_not_a_500(client):
    sid = client.post("/api/sessions").json()["id"]
    r = client.post(f"/api/sessions/{sid}/memo")
    assert r.status_code == 400 and "nothing to brief" in r.json()["error"]["message"]


def test_digit_title_is_a_400(client):
    sid = _analysed_session(client)
    r = client.post(f"/api/sessions/{sid}/memo", json={"title": "Study 101"})
    assert r.status_code == 400 and "digits" in r.json()["error"]["message"]


def _assert_serves_nothing_outside(resp) -> None:
    """A traversal URL never reaches the memo endpoint's file. Without a frontend build it
    is a 404; with one (frontend/dist present, e.g. after a desktop build), the SPA
    fallback answers unknown paths with the app's own index.html. Anything else -- a
    DOCX or a file from outside the app -- fails."""
    if resp.status_code == 404:
        return
    import app.main as main_mod
    dist = main_mod._frontend_dist()
    assert dist is not None, f"unexpected {resp.status_code} without a frontend build"
    assert resp.status_code == 200 and resp.content == (dist / "index.html").read_bytes()


def test_download_path_safety_mirrors_the_report_endpoint(client):
    sid = _analysed_session(client)
    # nothing built yet
    assert client.get(f"/api/sessions/{sid}/memo/memo_x.docx").status_code == 404
    res = client.post(f"/api/sessions/{sid}/memo").json()["result"]
    base = f"/api/sessions/{sid}/memo/"
    assert client.get(base + "other.docx").status_code == 404                  # name must match
    for probe in ("..%2F..%2Fetc%2Fpasswd", "%2Fetc%2Fpasswd"):               # traversal
        _assert_serves_nothing_outside(client.get(base + probe))
    Path(res["memo_path"]).unlink()
    assert client.get(base + res["filename"]).status_code == 404               # file missing


def test_unknown_session_is_404(client):
    assert client.post("/api/sessions/nope/memo").status_code == 404
    assert client.get("/api/sessions/nope/memo/x.docx").status_code == 404


def test_bearer_auth_is_enforced_like_the_other_endpoints(client):
    settings.api_token = "secret-token"
    h = {"Authorization": "Bearer secret-token"}
    sid = _analysed_session(client, h)
    assert client.post(f"/api/sessions/{sid}/memo").status_code == 401
    assert client.post(f"/api/sessions/{sid}/memo", headers={"Authorization": "Bearer nope"}).status_code == 401
    ok = client.post(f"/api/sessions/{sid}/memo", headers=h)
    assert ok.status_code == 200
    name = ok.json()["result"]["filename"]
    assert client.get(f"/api/sessions/{sid}/memo/{name}").status_code == 401
    assert client.get(f"/api/sessions/{sid}/memo/{name}", headers=h).status_code == 200
    # the audited actor is the non-secret principal, never the raw token
    actor = client.get(f"/api/sessions/{sid}/audit", headers=h).json()["entries"][-1]["actor"]
    assert actor.startswith("token:") and "secret-token" not in actor


def test_a_session_cannot_download_another_sessions_memo(client):
    a = _analysed_session(client)
    name = client.post(f"/api/sessions/{a}/memo").json()["result"]["filename"]
    b = client.post("/api/sessions").json()["id"]
    assert client.get(f"/api/sessions/{b}/memo/{name}").status_code == 404


def test_overlong_title_is_rejected_by_validation(client):
    sid = _analysed_session(client)
    assert client.post(f"/api/sessions/{sid}/memo", json={"title": "x" * 300}).status_code == 422
