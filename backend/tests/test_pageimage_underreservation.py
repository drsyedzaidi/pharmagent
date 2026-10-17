"""The reservation never trusts the client's Content-Length.

A client that declares ``Content-Length: 0`` and then streams ~50 MiB must still hold a worst-case (50 MiB) reservation
for as long as its form is being read, so concurrent bodies cannot exceed the byte budgets. These tests hold several
uploads open in flight (their receive channel blocks after the multipart head) and probe what a further request sees."""
from __future__ import annotations

import asyncio
import itertools

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

MIB = 1024 * 1024
HEAD = (b'--b\r\nContent-Disposition: form-data; name="file"; filename="a.pdf"\r\n'
        b"Content-Type: application/pdf\r\n\r\n%PDF-")


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(settings, "pageimage_enabled", True)
    main.orch = Orchestrator(llm=MockLLM(), clock=lambda c=itertools.count(): f"t{next(c)}",
                             store=SessionStore(":memory:"))
    token = settings.api_token
    yield TestClient(main.app)
    settings.api_token = token


def _scope(sid: str, content_length: str) -> dict:
    path = f"/api/sessions/{sid}/documents"
    return {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST", "scheme": "http",
            "path": path, "raw_path": path.encode(), "query_string": b"", "server": ("t", 80), "client": ("c", 1),
            "root_path": "", "headers": [(b"content-type", b"multipart/form-data; boundary=b"),
                                         (b"content-length", content_length.encode())]}


class Upload:
    """One ASGI request whose body is held open after the multipart head until ``release`` is set."""

    def __init__(self, sid: str, content_length: str = "0", hold: bool = True) -> None:
        self.sid, self.content_length, self.hold = sid, content_length, hold
        self.reads, self.status = 0, None
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def run(self) -> None:
        async def receive():
            self.reads += 1
            if self.reads == 1:
                return {"type": "http.request", "body": HEAD, "more_body": True}
            self.started.set()
            if self.hold:
                await self.release.wait()
            return {"type": "http.request", "body": b"\r\n--b--\r\n", "more_body": False}

        async def send(msg):
            if msg["type"] == "http.response.start":
                self.status = msg["status"]
        await main.app(_scope(self.sid, self.content_length), receive, send)


async def _hold(uploads: list[Upload]) -> list[asyncio.Task]:
    tasks = [asyncio.create_task(u.run()) for u in uploads]
    await asyncio.wait_for(asyncio.gather(*(u.started.wait() for u in uploads)), 30)
    return tasks


def _sessions(client, n):
    return [client.post("/api/sessions").json()["id"] for _ in range(n)]


def test_the_worst_case_reservation_is_independent_of_the_declared_length(client, tmp_path):
    sid, = _sessions(client, 1)

    async def go():
        up = Upload(sid, content_length="0")
        task = asyncio.create_task(up.run())
        await asyncio.wait_for(up.started.wait(), 30)
        usage = quota.usage(tmp_path / "data" / "documents")
        up.release.set()
        await task
        return usage
    (per, total) = asyncio.run(go())
    assert per[sid][0] == 1 and total == limits.MAX_PDF_BYTES == 50 * MIB        # declared 0, reserved 50 MiB


def test_four_declared_zero_uploads_fill_the_session_budget_and_the_fifth_reads_nothing(client, tmp_path):
    sid, = _sessions(client, 1)
    assert limits.MAX_SESSION_BYTES == 4 * limits.MAX_PDF_BYTES

    async def go():
        held = [Upload(sid) for _ in range(4)]
        tasks = await _hold(held)
        used = quota.usage(tmp_path / "data" / "documents")[0][sid][1]
        fifth = Upload(sid, hold=False)
        await fifth.run()
        for u in held:
            u.release.set()
        await asyncio.gather(*tasks)
        return used, fifth, held
    used, fifth, held = asyncio.run(go())
    assert used == 4 * 50 * MIB
    assert fifth.status == 409 and fifth.reads == 0                              # refused before any body byte
    assert all(u.status == 400 for u in held)                                   # their (empty) files are rejected
    assert quota.usage(tmp_path / "data" / "documents") == ({sid: (0, 0)}, 0)    # and every reservation released


def test_the_global_budget_holds_across_sessions_for_declared_zero_uploads(client, tmp_path, monkeypatch):
    monkeypatch.setattr(limits, "MAX_GLOBAL_BYTES", 3 * limits.MAX_PDF_BYTES)
    sids = _sessions(client, 4)

    async def go():
        held = [Upload(s) for s in sids[:3]]
        tasks = await _hold(held)
        total = quota.usage(tmp_path / "data" / "documents")[1]
        fourth = Upload(sids[3], hold=False)
        await fourth.run()
        for u in held:
            u.release.set()
        await asyncio.gather(*tasks)
        return total, fourth
    total, fourth = asyncio.run(go())
    assert total == 3 * 50 * MIB
    assert fourth.status == 503 and fourth.reads == 0


def test_a_failed_upload_releases_its_worst_case_reservation(client, tmp_path):
    sid, = _sessions(client, 1)
    r = client.post(f"/api/sessions/{sid}/documents", files={"file": ("a.pdf", b"not a pdf", "application/pdf")})
    assert r.status_code == 400
    assert quota.usage(tmp_path / "data" / "documents") == ({sid: (0, 0)}, 0)


def test_a_small_completed_upload_costs_its_size_not_the_worst_case(client, tmp_path):
    sid, = _sessions(client, 1)
    pdf = make_pdf(pages=2)
    r = client.post(f"/api/sessions/{sid}/documents", files={"file": ("a.pdf", pdf, "application/pdf")})
    assert r.status_code == 201
    assert quota.usage(tmp_path / "data" / "documents") == ({sid: (1, len(pdf))}, len(pdf))
    assert not any(p.suffix == ".resv" for p in (tmp_path / "data" / "documents" / sid).iterdir())
    # the freed worst-case reservation is usable again: four more declared-zero uploads still fit under 200 MiB - 1 doc
    for _ in range(3):
        documents.new_slot(sid, limits.MAX_PDF_BYTES)
    with pytest.raises(documents.QuotaExceeded):
        documents.new_slot(sid, limits.MAX_PDF_BYTES)
