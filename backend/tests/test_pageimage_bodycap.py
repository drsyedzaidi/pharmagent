"""Step-5 hardening, item 1: the upload body is capped BEFORE auth and BEFORE multipart parsing/spooling.

FastAPI parses (and spools to disk) the multipart body before it resolves dependencies, so without a gate an
unauthenticated or non-owner client can make the server buffer an arbitrarily large body. The gate sits in the route's
handler wrapper: a declared Content-Length over the cap is refused outright, and an undeclared or understated length is
cut off as soon as the bytes actually received pass the cap."""
from __future__ import annotations

import asyncio
import itertools

import pytest
from fastapi.testclient import TestClient
from starlette import formparsers, requests

import app.main as main
from app.config import settings
from app.core.llm import MockLLM
from app.core.orchestrator import Orchestrator
from app.core.store import SessionStore
from app.pageimage import limits
from tests.pdf_fixtures import make_pdf, unenforced_limits_off_linux

_weak = pytest.fixture(autouse=True)(unenforced_limits_off_linux)

CAP = limits.MAX_PDF_BYTES + limits.MULTIPART_OVERHEAD_BYTES


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(settings, "pageimage_enabled", True)
    main.orch = Orchestrator(llm=MockLLM(), clock=lambda c=itertools.count(): f"t{next(c)}",
                             store=SessionStore(":memory:"))
    token = settings.api_token
    yield TestClient(main.app)
    settings.api_token = token


def _no_parse(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("the body was parsed although it is over the cap")
    monkeypatch.setattr(requests.Request, "_get_form", boom)
    monkeypatch.setattr(formparsers.MultiPartParser, "parse", boom)


def test_cap_is_the_pdf_cap_plus_bounded_overhead():
    assert limits.MULTIPART_OVERHEAD_BYTES <= 1024 * 1024
    assert CAP > limits.MAX_PDF_BYTES


def test_declared_oversize_is_refused_without_parsing(client, monkeypatch):
    _no_parse(monkeypatch)
    sid = client.post("/api/sessions").json()["id"]
    r = client.post(f"/api/sessions/{sid}/documents", content=b"x" * 10,
                    headers={"content-type": "multipart/form-data; boundary=b", "content-length": str(CAP + 1)})
    assert r.status_code == 413


def test_oversize_is_refused_before_auth(client, monkeypatch):
    """A caller that would fail auth gets 413, not a spooled body then 401: the cap is the cheaper, earlier check."""
    _no_parse(monkeypatch)
    monkeypatch.setattr(settings, "api_token", "secret")
    r = client.post("/api/sessions/anything/documents", content=b"x",
                    headers={"content-type": "multipart/form-data; boundary=b", "content-length": str(CAP + 1)})
    assert r.status_code == 413


def test_unauthenticated_small_upload_is_still_refused_by_auth(client, monkeypatch):
    sid = client.post("/api/sessions").json()["id"]
    monkeypatch.setattr(settings, "api_token", "secret")
    r = client.post(f"/api/sessions/{sid}/documents", files={"file": ("a.pdf", make_pdf(), "application/pdf")})
    assert r.status_code in (401, 403)


def test_bad_content_length_is_a_400(client):
    sid = client.post("/api/sessions").json()["id"]
    r = client.post(f"/api/sessions/{sid}/documents", content=b"x",
                    headers={"content-type": "multipart/form-data; boundary=b", "content-length": "abc"})
    assert r.status_code in (400, 413)


def test_normal_upload_still_works(client):
    sid = client.post("/api/sessions").json()["id"]
    r = client.post(f"/api/sessions/{sid}/documents", files={"file": ("a.pdf", make_pdf(pages=2), "application/pdf")})
    assert r.status_code == 201


def test_understated_or_chunked_body_is_cut_off_at_the_cap():
    """Drive the ASGI app directly with a body that never ends and no Content-Length: the server must stop reading near
    the cap and answer 413, not keep consuming (and spooling) the stream."""
    chunk = b"x" * (1 << 20)
    head = (b'--b\r\nContent-Disposition: form-data; name="file"; filename="a.pdf"\r\n'
            b"Content-Type: application/pdf\r\n\r\n%PDF-")                     # a real file part that never ends
    state = {"reads": 0, "sent": []}

    async def receive():
        state["reads"] += 1
        if state["reads"] > 400:                      # 400 MiB: far past the cap; a runaway server would get here
            return {"type": "http.request", "body": b"", "more_body": False}
        return {"type": "http.request", "body": head if state["reads"] == 1 else chunk, "more_body": True}

    async def send(msg):
        state["sent"].append(msg)

    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST", "scheme": "http",
             "path": "/api/sessions/s1/documents", "raw_path": b"/api/sessions/s1/documents", "query_string": b"",
             "headers": [(b"content-type", b"multipart/form-data; boundary=b")], "server": ("t", 80),
             "client": ("c", 1), "root_path": ""}
    settings.pageimage_enabled = True
    try:
        asyncio.run(main.app(scope, receive, send))
    finally:
        settings.pageimage_enabled = False
    status = next(m["status"] for m in state["sent"] if m["type"] == "http.response.start")
    assert status == 413
    assert state["reads"] * len(chunk) <= CAP + 2 * len(chunk)
