"""Admission order for an upload: authenticate and own the session, reserve quota, and only THEN read the multipart body.

The endpoint has no ``UploadFile`` parameter, so FastAPI cannot parse or spool the body before authentication or before
the quota is taken. A refused request consumes none of its body and creates no temporary file."""
from __future__ import annotations

import asyncio
import itertools
import tempfile

import pytest
from fastapi.testclient import TestClient
from starlette import formparsers, requests

import app.main as main
from app.config import settings
from app.core.llm import MockLLM
from app.core.orchestrator import Orchestrator
from app.core.store import SessionStore
from app.pageimage import documents, limits
from tests.pdf_fixtures import make_pdf, unenforced_limits_off_linux

_weak = pytest.fixture(autouse=True)(unenforced_limits_off_linux)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(settings, "pageimage_enabled", True)
    main.orch = Orchestrator(llm=MockLLM(), clock=lambda c=itertools.count(): f"t{next(c)}",
                             store=SessionStore(":memory:"))
    token = settings.api_token
    yield TestClient(main.app)
    settings.api_token = token


def _forbid_reading_the_body(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("the request body was read or spooled")
    monkeypatch.setattr(requests.Request, "_get_form", boom)
    monkeypatch.setattr(formparsers.MultiPartParser, "parse", boom)
    monkeypatch.setattr(formparsers, "SpooledTemporaryFile", boom)
    monkeypatch.setattr(tempfile, "SpooledTemporaryFile", boom)


def _post_asgi(path: str, headers: dict[str, str], n_chunks: int = 8):
    """POST a multipart body straight to the app and report (status, receive calls)."""
    chunk = b"x" * (1 << 20)
    head = (b'--b\r\nContent-Disposition: form-data; name="file"; filename="a.pdf"\r\n'
            b"Content-Type: application/pdf\r\n\r\n%PDF-")
    state = {"reads": 0, "sent": []}

    async def receive():
        state["reads"] += 1
        if state["reads"] > n_chunks:
            return {"type": "http.request", "body": b"\r\n--b--\r\n", "more_body": False}
        return {"type": "http.request", "body": head if state["reads"] == 1 else chunk, "more_body": True}

    async def send(msg):
        state["sent"].append(msg)

    hdrs = [(b"content-type", b"multipart/form-data; boundary=b")] + [(k.encode(), v.encode()) for k, v in headers.items()]
    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST", "scheme": "http",
             "path": path, "raw_path": path.encode(), "query_string": b"", "headers": hdrs, "server": ("t", 80),
             "client": ("c", 1), "root_path": ""}
    asyncio.run(main.app(scope, receive, send))
    return next(m["status"] for m in state["sent"] if m["type"] == "http.response.start"), state["reads"]


def test_unauthenticated_upload_reads_no_body_and_creates_nothing(client, monkeypatch, tmp_path):
    sid = client.post("/api/sessions").json()["id"]
    monkeypatch.setattr(settings, "api_token", "secret")
    _forbid_reading_the_body(monkeypatch)
    status, reads = _post_asgi(f"/api/sessions/{sid}/documents", {})
    assert status in (401, 403)
    assert reads == 0
    assert not (tmp_path / "data" / "documents").exists()


def test_wrong_owner_upload_reads_no_body_and_creates_nothing(client, monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "api_token", "tok-a")
    theirs = main.orch.create_session(owner="token:someone-else")
    _forbid_reading_the_body(monkeypatch)
    status, reads = _post_asgi(f"/api/sessions/{theirs.id}/documents", {"authorization": "Bearer tok-a"})
    assert status == 403 and reads == 0
    assert not (tmp_path / "data" / "documents").exists()


def test_quota_refused_upload_reads_no_body_and_creates_nothing(client, monkeypatch, tmp_path):
    sid = client.post("/api/sessions").json()["id"]
    for _ in range(limits.MAX_DOCS_PER_SESSION):
        documents.new_slot(sid, 10)
    before = sorted(p.name for p in (tmp_path / "data" / "documents" / sid).iterdir())
    _forbid_reading_the_body(monkeypatch)
    status, reads = _post_asgi(f"/api/sessions/{sid}/documents", {})
    assert status == 409 and reads == 0
    assert sorted(p.name for p in (tmp_path / "data" / "documents" / sid).iterdir()) == before


def test_global_quota_refused_upload_reads_no_body(client, monkeypatch):
    sid = client.post("/api/sessions").json()["id"]
    monkeypatch.setattr(limits, "MAX_GLOBAL_BYTES", 10)
    _forbid_reading_the_body(monkeypatch)
    status, reads = _post_asgi(f"/api/sessions/{sid}/documents", {"content-length": "5000"})
    assert status == 503 and reads == 0


def test_the_body_is_read_only_after_admission(client):
    """The positive control for the tests above: with a session and quota, the same request DOES read its body."""
    sid = client.post("/api/sessions").json()["id"]
    status, reads = _post_asgi(f"/api/sessions/{sid}/documents", {})
    assert reads > 0 and status in (201, 400, 413)


def test_a_reservation_exists_while_the_form_is_parsed(client, monkeypatch, tmp_path):
    """Concurrent request bodies are bounded by quota because the reservation is taken first."""
    sid = client.post("/api/sessions").json()["id"]
    seen = {}
    real_form = requests.Request._get_form

    async def spy(self, *a, **k):
        d = tmp_path / "data" / "documents" / sid
        seen["resv"] = sorted(p.suffix for p in d.iterdir())
        return await real_form(self, *a, **k)
    monkeypatch.setattr(requests.Request, "_get_form", spy)
    r = client.post(f"/api/sessions/{sid}/documents", files={"file": ("a.pdf", make_pdf(), "application/pdf")})
    assert r.status_code == 201
    assert seen["resv"] == [".resv"]
