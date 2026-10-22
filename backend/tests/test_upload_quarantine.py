"""Owner decision: a quarantined session is entirely read-only, including associated file storage. Uploads return 423
before any multipart parsing or storage."""
from __future__ import annotations

import itertools
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

import app.main as main
from app.config import settings
from app.core.llm import MockLLM
from app.core.orchestrator import Orchestrator
from app.core.provenance import collect_provenance
from app.core.store import SessionStore

SAMPLE = Path(__file__).parent.parent / "sample_data" / "oral_pk.csv"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    main.orch = Orchestrator(llm=MockLLM(), clock=lambda c=itertools.count(): f"t{next(c)}",
                             store=SessionStore(":memory:"))
    return TestClient(main.app)


def _quarantine(sid):
    sess = main.orch.sessions[sid]
    forged = tuple({**collect_provenance(), "git_sha": "forged"}.items())
    sess.state = sess.state.model_copy(update={"run_provenance": forged})


def test_upload_to_a_quarantined_session_is_refused_before_parsing_or_storage(client, tmp_path, monkeypatch):
    sid = client.post("/api/sessions").json()["id"]
    _quarantine(sid)
    parsed = []
    original = Request.form

    def spy(self, *a, **k):
        parsed.append(True)
        return original(self, *a, **k)

    monkeypatch.setattr(Request, "form", spy)
    before = set(tmp_path.iterdir())
    r = client.post(f"/api/sessions/{sid}/upload", files={"file": ("x.csv", SAMPLE.read_bytes(), "text/csv")})
    assert r.status_code == 423 and "mismatch" in r.text
    assert parsed == []                                              # multipart never parsed
    assert set(tmp_path.iterdir()) == before                         # nothing stored


def test_a_malformed_body_on_a_quarantined_session_is_still_423_not_a_parse_error(client):
    sid = client.post("/api/sessions").json()["id"]
    _quarantine(sid)
    r = client.post(f"/api/sessions/{sid}/upload", content=b"--garbage\r\nnot multipart",
                    headers={"Content-Type": "multipart/form-data; boundary=zzz"})
    assert r.status_code == 423


def test_a_normal_session_can_still_upload(client, tmp_path):
    sid = client.post("/api/sessions").json()["id"]
    r = client.post(f"/api/sessions/{sid}/upload", files={"file": ("oral_pk.csv", SAMPLE.read_bytes(), "text/csv")})
    assert r.status_code == 200 and r.json()["dataset_id"]
    assert any(p.suffix == ".csv" for p in tmp_path.iterdir())


def test_ownership_is_checked_before_quarantine(client):
    r = client.post("/api/sessions/sess_nope/upload", files={"file": ("x.csv", b"a,b\n1,2\n", "text/csv")})
    assert r.status_code == 404
