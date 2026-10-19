"""How the upload endpoint maps storage failures, end to end through the API.

A reservation reclaimed while its body is still arriving -> 409 "upload expired"; the per-chunk heartbeat; and a storage
lock or accounting failure at the heartbeat or at commit -> 503. In every case the upload leaves nothing behind."""
from __future__ import annotations

import asyncio
import itertools
import os
import time

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.config import settings
from app.core.llm import MockLLM
from app.core.orchestrator import Orchestrator
from app.core.store import SessionStore
from app.pageimage import documents, limits, quota
from tests.pdf_fixtures import make_pdf, unenforced_limits_off_linux

_weak = pytest.fixture(autouse=True)(unenforced_limits_off_linux)

EXPIRED = "upload expired; please retry"
UNAVAILABLE = "document storage is unavailable; retry later"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "pageimage_enabled", True)
    main.orch = Orchestrator(llm=MockLLM(), clock=lambda c=itertools.count(): f"t{next(c)}",
                             store=SessionStore(":memory:"))
    token = settings.api_token
    yield TestClient(main.app, raise_server_exceptions=False)
    settings.api_token = token


def _root():
    return settings.data_dir / "documents"


def _pieces(sid):
    d = _root() / sid
    return sorted(p.name for p in d.iterdir() if not p.name.startswith(".render-")) if d.exists() else []


def _up(client, sid, data=None):
    return client.post(f"/api/sessions/{sid}/documents",
                       files={"file": ("a.pdf", data if data is not None else make_pdf(), "application/pdf")})


def test_a_reservation_reclaimed_mid_body_is_409_upload_expired(client):
    """The client stalls half-way through its body; meanwhile its reservation goes stale and another upload's cleanup
    reclaims it. When the body finally arrives the upload is refused with 409 and leaves nothing behind."""
    sid = client.post("/api/sessions").json()["id"]
    pdf = make_pdf()
    head = (b'--b\r\nContent-Disposition: form-data; name="file"; filename="a.pdf"\r\n'
            b"Content-Type: application/pdf\r\n\r\n")
    state = {"reads": 0, "status": None, "body": b""}

    async def go():
        stalled, resume = asyncio.Event(), asyncio.Event()

        async def receive():
            state["reads"] += 1
            if state["reads"] == 1:
                return {"type": "http.request", "body": head + pdf[:64], "more_body": True}
            stalled.set()
            await resume.wait()
            return {"type": "http.request", "body": pdf[64:] + b"\r\n--b--\r\n", "more_body": False}

        async def send(msg):
            if msg["type"] == "http.response.start":
                state["status"] = msg["status"]
            elif msg["type"] == "http.response.body":
                state["body"] += msg.get("body", b"")
        path = f"/api/sessions/{sid}/documents"
        scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST", "scheme": "http",
                 "path": path, "raw_path": path.encode(), "query_string": b"", "server": ("t", 80),
                 "client": ("c", 1), "root_path": "", "headers": [(b"content-type", b"multipart/form-data; boundary=b")]}
        task = asyncio.create_task(main.app(scope, receive, send))
        await asyncio.wait_for(stalled.wait(), 20)
        (resv,) = (_root() / sid).glob("*.resv")
        old = time.time() - limits.STALE_SECONDS - 60
        os.utime(resv, (old, old))                                       # no heartbeat for over an hour
        await asyncio.to_thread(documents.new_slot, "someone-else", 10)  # their reservation runs cleanup
        assert not resv.exists()
        resume.set()
        await asyncio.wait_for(task, 30)
    asyncio.run(go())
    assert state["status"] == 409 and EXPIRED.encode() in state["body"]
    assert _pieces(sid) == []


def test_the_copy_loop_heartbeats_once_per_chunk(client, monkeypatch):
    """A 2.5 MiB upload is copied in three 1 MiB chunks: three heartbeats, not one per upload."""
    from app.pageimage import router
    calls = []
    real = documents.touch
    monkeypatch.setattr(documents, "touch", lambda part: (calls.append(part), real(part))[1])
    monkeypatch.setattr(router, "probe_pdf", lambda path, **k: 1)       # the bytes need not be a renderable PDF here
    sid = client.post("/api/sessions").json()["id"]
    body = b"%PDF-1.7\n" + b"x" * (5 * (1 << 19))
    assert _up(client, sid, body).status_code == 201
    assert len(calls) == 3


def test_a_failing_cleanup_never_replaces_the_original_error(client, monkeypatch):
    def boom(part):
        raise RuntimeError("cleanup broke")
    monkeypatch.setattr(documents, "discard", boom)
    sid = client.post("/api/sessions").json()["id"]
    r = _up(client, sid, b"not a pdf")
    assert r.status_code == 400 and r.json()["error"]["message"] == "file is not a PDF"


@pytest.mark.parametrize("where,error,status,message", [
    ("touch", documents.ReservationLostError(), 409, EXPIRED),
    ("touch", quota.QuotaUnavailable("lock busy"), 503, UNAVAILABLE),
    ("commit", documents.ReservationLostError(), 409, EXPIRED),
    ("commit", quota.QuotaUnavailable("lock busy"), 503, UNAVAILABLE),
])
def test_storage_failures_map_to_fixed_statuses_and_leave_nothing(client, monkeypatch, where, error, status, message):
    def boom(*a, **k):
        raise error
    monkeypatch.setattr(documents, where, boom)
    sid = client.post("/api/sessions").json()["id"]
    r = _up(client, sid)
    assert r.status_code == status and r.json()["error"]["message"] == message
    assert _pieces(sid) == [] and quota.usage(_root())[1] == 0
