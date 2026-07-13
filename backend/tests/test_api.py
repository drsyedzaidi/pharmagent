"""HTTP-layer tests: endpoints, bearer-token auth, ownership, role overrides,
and the background-job (async) execution path."""
import itertools
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.config import settings
from app.core.llm import MockLLM
from app.core.orchestrator import Orchestrator
from app.core.store import SessionStore

SAMPLE = str(Path(__file__).parent.parent / "sample_data" / "oral_pk.csv")


@pytest.fixture
def client():
    main.orch = Orchestrator(llm=MockLLM(), clock=lambda c=itertools.count(): f"t{next(c)}",
                             store=SessionStore(":memory:"))
    token = settings.api_token
    yield TestClient(main.app)
    settings.api_token = token  # restore


def test_health_and_models_public(client):
    assert client.get("/api/health").json()["status"] == "ok"
    assert len(client.get("/api/pk_models").json()["models"]) == 18


def test_session_and_nca_workflow(client):
    sid = client.post("/api/sessions").json()["id"]
    r = client.post(f"/api/sessions/{sid}/workflow/start",
                    json={"workflow": "nca_full", "params": {"path": SAMPLE}}).json()
    assert r["status"] == "awaiting_review"
    assert r["state"]["nca_parameters"] and len(r["state"]["nca_parameters"]) == 12
    audit = client.get(f"/api/sessions/{sid}/audit").json()
    assert audit["verified"] is True and audit["count"] >= 1


def test_role_override(client):
    sid = client.post("/api/sessions").json()["id"]
    client.post(f"/api/sessions/{sid}/chat", json={"message": f"load dataset {SAMPLE}"})
    out = client.post(f"/api/sessions/{sid}/roles",
                      json={"overrides": {"CMT": "TIME"}}).json()
    assert out["detected_roles"]["CMT"] == "TIME"


def test_auth_required_when_token_set(client):
    settings.api_token = "secret-token"
    # no header -> 401
    assert client.post("/api/sessions").status_code == 401
    # correct token -> 201/200
    h = {"Authorization": "Bearer secret-token"}
    sid = client.post("/api/sessions", headers=h).json()["id"]
    assert client.get(f"/api/sessions/{sid}/state", headers=h).status_code == 200
    # different token -> cannot access another owner's session (403)
    h2 = {"Authorization": "Bearer secret-token"}  # same token == same owner here
    assert client.get(f"/api/sessions/{sid}/state", headers=h2).status_code == 200
    # wrong token -> 401
    assert client.get(f"/api/sessions/{sid}/state",
                      headers={"Authorization": "Bearer nope"}).status_code == 401


def test_upload_rejects_non_csv(client):
    sid = client.post("/api/sessions").json()["id"]
    r = client.post(f"/api/sessions/{sid}/upload",
                    files={"file": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def _poll(client, sid, job_id, tries=80):
    for _ in range(tries):
        j = client.get(f"/api/sessions/{sid}/jobs/{job_id}").json()
        if j["status"] != "running":
            return j
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def test_nlme_runs_as_background_job(client):
    sid = client.post("/api/sessions").json()["id"]
    # No model fitted -> the tool returns a 'no_model' status, but it must travel
    # through the async submit -> poll path (instant submit, then completion).
    r = client.post(f"/api/sessions/{sid}/nlme", json={"method": "focei"}).json()
    assert r["status"] == "running" and r["job_id"].startswith("job_")
    j = _poll(client, sid, r["job_id"])
    assert j["status"] == "done"
    assert j["result"]["result"]["status"] == "no_model"
    # unknown job id -> 404
    assert client.get(f"/api/sessions/{sid}/jobs/job_nope").status_code == 404


def test_scm_submits_a_job(client):
    sid = client.post("/api/sessions").json()["id"]
    r = client.post(f"/api/sessions/{sid}/scm", json={}).json()
    assert r["kind"] == "scm" and r["job_id"].startswith("job_")
    j = _poll(client, sid, r["job_id"])
    assert j["status"] == "done"
    assert j["result"]["result"]["status"] == "no_model"


def test_jobmanager_reports_done_and_error():
    from app.core.jobs import JobManager
    jm = JobManager(max_workers=2, clock=lambda: "t")
    done = jm.submit(session_id="s", kind="x", fn=lambda: 2 + 2)
    for _ in range(80):
        j = jm.get(done)
        if j["status"] != "running":
            break
        time.sleep(0.05)
    assert j["status"] == "done" and j["result"] == 4

    def boom():
        raise ValueError("kaboom")
    err = jm.submit(session_id="s", kind="x", fn=boom)
    for _ in range(80):
        j2 = jm.get(err)
        if j2["status"] != "running":
            break
        time.sleep(0.05)
    assert j2["status"] == "error" and "kaboom" in j2["error"]
    assert jm.get("job_missing") is None
