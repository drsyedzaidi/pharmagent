"""H2: an upload's reservation is kept alive while its multipart body is still arriving, and a reservation that was
reclaimed never gets a byte written to its ``.part``.

``upload_document`` reserves quota, then receives and parses the whole body, and only then copies it to the ``.part``.
While the body arrives, every message received refreshes the reservation through the same locked heartbeat the copy
uses (in a worker thread, at most once per ``limits.HEARTBEAT_SECONDS``); a body that sends nothing gets no heartbeat
and goes stale as before. The copy refreshes and confirms the reservation before every write, the first included, so a
reclaimed upload is refused with 409 before it creates its ``.part``.

Every test drives the ASGI app directly, waits only for events and observable conditions (never a fixed sleep), and
checks that every task it created and every thread started meanwhile has finished."""
from __future__ import annotations

import asyncio
import itertools
import logging
import os
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette import formparsers
from starlette.requests import Request

import app.main as main
from app.config import settings
from app.core.llm import MockLLM
from app.core.orchestrator import Orchestrator
from app.core.store import SessionStore
from app.pageimage import documents, limits, quota, router
from tests.pdf_fixtures import make_pdf, unenforced_limits_off_linux

_weak = pytest.fixture(autouse=True)(unenforced_limits_off_linux)

EXPIRED = "upload expired; please retry"
WAIT = 20                    # upper bound on every wait for an event or a task; nothing here sleeps
FILE_HEAD = (b'--b\r\nContent-Disposition: form-data; name="file"; filename="a.pdf"\r\n'
             b"Content-Type: application/pdf\r\n\r\n")
TAIL = b"\r\n--b--\r\n"
PDF = make_pdf()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "pageimage_enabled", True)
    main.orch = Orchestrator(llm=MockLLM(), clock=lambda c=itertools.count(): f"t{next(c)}",
                             store=SessionStore(":memory:"))
    token = settings.api_token
    yield TestClient(main.app, raise_server_exceptions=False)
    settings.api_token = token


@pytest.fixture
def spools(monkeypatch):
    """Every temp file the multipart parser spools to, so a test can check each one was closed."""
    made = []
    real = formparsers.SpooledTemporaryFile

    def tracking(*a, **k):
        f = real(*a, **k)
        made.append(f)
        return f
    monkeypatch.setattr(formparsers, "SpooledTemporaryFile", tracking)
    return made


@pytest.fixture
def part_opens(monkeypatch):
    """Every ``.part`` anything opens. The upload creates and writes its ``.part`` only through ``Path.open``."""
    opened = []
    real = Path.open

    def tracking(self, *a, **k):
        if self.suffix == ".part":
            opened.append(self)
        return real(self, *a, **k)
    monkeypatch.setattr(Path, "open", tracking)
    return opened


def _root():
    return settings.data_dir / "documents"


def _pieces(sid):
    d = _root() / sid
    return sorted(p.name for p in d.iterdir() if not p.name.startswith(".render-")) if d.exists() else []


def _resv(sid):
    (resv,) = (_root() / sid).glob("*.resv")
    return resv


def _age(path):
    """As if the upload had sent nothing, and refreshed nothing, for over an hour."""
    old = time.time() - limits.STALE_SECONDS - 60
    os.utime(path, (old, old))


def _three_parts():
    return [FILE_HEAD + PDF[:64], PDF[64:128], PDF[128:] + TAIL]


class Upload:
    """One upload over ASGI whose body arrives in ``parts``: part 0 at once, each later part only once the test
    releases it. ``asked[i]`` is set when the app asks for part i, which also means it has finished with part i - 1,
    that part's receive-time heartbeat included."""

    def __init__(self, sid, parts):
        self.sid, self.parts = sid, parts
        self.asked = [asyncio.Event() for _ in parts]
        self.released = [asyncio.Event() for _ in parts]
        self.released[0].set()
        self.reads, self.status, self.body, self.task = 0, None, b"", None

    async def _receive(self):
        i = self.reads
        self.reads += 1
        if i >= len(self.parts):
            return {"type": "http.disconnect"}
        self.asked[i].set()
        await self.released[i].wait()
        return {"type": "http.request", "body": self.parts[i], "more_body": i < len(self.parts) - 1}

    async def _send(self, msg):
        if msg["type"] == "http.response.start":
            self.status = msg["status"]
        elif msg["type"] == "http.response.body":
            self.body += msg.get("body", b"")

    def start(self):
        path = f"/api/sessions/{self.sid}/documents"
        scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST", "scheme": "http",
                 "path": path, "raw_path": path.encode(), "query_string": b"", "server": ("t", 80),
                 "client": ("c", 1), "root_path": "", "headers": [(b"content-type", b"multipart/form-data; boundary=b")]}
        self.task = asyncio.create_task(main.app(scope, self._receive, self._send))
        return self

    async def finish(self):
        """Let the rest of the body through and wait for the response; returns its status."""
        for e in self.released:
            e.set()
        await asyncio.wait_for(self.task, WAIT)
        return self.status


async def _wait(event):
    await asyncio.wait_for(event.wait(), WAIT)


async def _first(*things):
    """Wait (bounded) until the first of these events is set or tasks is done. The caller then asserts which."""
    waiters = [asyncio.create_task(t.wait()) if isinstance(t, asyncio.Event) else t for t in things]
    await asyncio.wait(waiters, timeout=WAIT, return_when=asyncio.FIRST_COMPLETED)
    helpers = [w for w, t in zip(waiters, things) if w is not t]
    for w in helpers:
        w.cancel()
    await asyncio.gather(*helpers, return_exceptions=True)


def _run(scenario):
    """Run ``scenario`` on a fresh event loop. Afterwards no task it created, and no thread started meanwhile (worker
    threads included), may still be running."""
    before = set(threading.enumerate())

    async def top():
        await asyncio.wait_for(scenario(), 60)
        left = asyncio.all_tasks() - {asyncio.current_task()}
        assert not left, f"tasks left running: {left}"
    asyncio.run(top())
    started = [t for t in threading.enumerate() if t not in before]
    for t in started:
        t.join(WAIT)
    assert not [t for t in started if t.is_alive()], "threads left running"


def _lock_holder():
    """A thread holding the storage lock (as another worker process's cleanup or commit would) until released."""
    held, release = threading.Event(), threading.Event()

    def hold():
        with quota.locked(_root()):
            held.set()
            release.wait(WAIT)
    t = threading.Thread(target=hold, daemon=True)
    t.start()
    return t, held, release


# ───────────────────── pacing ─────────────────────

def test_the_receive_heartbeat_is_paced_by_time_not_by_bytes(monkeypatch):
    """One message every 10 s for five minutes refreshes the reservation once a minute; a burst of a thousand
    messages within the same second adds nothing."""
    monkeypatch.setattr(limits, "HEARTBEAT_SECONDS", 60, raising=False)
    now = [0.0]
    beats = []
    monkeypatch.setattr(documents, "touch", lambda part: beats.append(now[0]))

    async def receive():
        return {"type": "http.request", "body": b"x", "more_body": True}

    async def scenario():
        req = router._heartbeating(Request({"type": "http", "headers": []}, receive), Path("x.part"),
                                   clock=lambda: now[0])
        for t in range(0, 301, 10):
            now[0] = float(t)
            await req.receive()
        for _ in range(1000):
            await req.receive()
    _run(scenario)
    assert beats == [60.0, 120.0, 180.0, 240.0, 300.0]


# ───────────────────── slow, stalled, cancelled, concurrent ─────────────────────

def test_a_slow_body_keeps_its_reservation_through_another_uploads_cleanup(client, monkeypatch):
    """The body is quiet for over an hour, then more of it arrives. That message refreshes the reservation, so the
    cleanup another upload runs right after it does not reclaim this live upload, which then completes."""
    monkeypatch.setattr(limits, "HEARTBEAT_SECONDS", 0, raising=False)       # every message is due a heartbeat
    sid = client.post("/api/sessions").json()["id"]

    async def scenario():
        up = Upload(sid, _three_parts()).start()
        await _wait(up.asked[1])
        resv = _resv(sid)
        _age(resv)
        up.released[1].set()
        await _wait(up.asked[2])                                              # that part and its heartbeat are done
        assert time.time() - resv.stat().st_mtime < limits.STALE_SECONDS
        await asyncio.to_thread(documents.new_slot, "someone-else", 10)       # their reservation runs cleanup
        assert resv.exists(), "a live upload's reservation was reclaimed"
        assert await up.finish() == 201
    _run(scenario)
    docs = client.get(f"/api/sessions/{sid}/documents").json()["documents"]
    assert [d["size_bytes"] for d in docs] == [len(PDF)]


def test_a_stalled_body_that_was_reclaimed_is_refused_before_any_part_write(client, monkeypatch, part_opens, spools,
                                                                             caplog):
    """The body stalls while the wall clock moves on but the monotonic clock pacing the receive heartbeat does not (a
    laptop asleep, say), so no receive-time heartbeat is due when the rest arrives. The reservation went stale and
    another upload's cleanup reclaimed it: the check before the copy's first write must refuse the upload."""
    monkeypatch.setattr(limits, "HEARTBEAT_SECONDS", float("inf"), raising=False)
    sid = client.post("/api/sessions").json()["id"]

    async def scenario():
        up = Upload(sid, [FILE_HEAD + PDF[:64], PDF[64:] + TAIL]).start()
        await _wait(up.asked[1])
        resv = _resv(sid)
        _age(resv)
        await asyncio.to_thread(documents.new_slot, "someone-else", 10)
        assert not resv.exists()
        with caplog.at_level(logging.ERROR):
            assert await up.finish() == 409
        assert EXPIRED.encode() in up.body
    _run(scenario)
    assert part_opens == [], "a reclaimed upload wrote to its .part"
    assert _pieces(sid) == [] and quota.usage(_root())[0][sid] == (0, 0)
    assert spools and all(f.closed for f in spools)
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_a_reclaimed_reservation_is_refused_while_the_body_is_still_arriving(client, monkeypatch, part_opens, spools,
                                                                             caplog):
    """Reclaimed during a stall; the next message's heartbeat finds the reservation gone and refuses at once: the app
    stops reading the body (it never asks for the last part) and nothing reaches the .part."""
    monkeypatch.setattr(limits, "HEARTBEAT_SECONDS", 0, raising=False)
    sid = client.post("/api/sessions").json()["id"]

    async def scenario():
        up = Upload(sid, _three_parts()).start()
        await _wait(up.asked[1])
        resv = _resv(sid)
        _age(resv)
        await asyncio.to_thread(documents.new_slot, "someone-else", 10)
        assert not resv.exists()
        with caplog.at_level(logging.ERROR):
            up.released[1].set()
            await _first(up.task, up.asked[2])
            assert up.task.done() and not up.asked[2].is_set(), "the body kept being read after its reservation went"
        assert up.status == 409 and EXPIRED.encode() in up.body
    _run(scenario)
    assert part_opens == []
    assert _pieces(sid) == [] and quota.usage(_root())[0][sid] == (0, 0)
    assert spools and all(f.closed for f in spools)
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_the_receive_heartbeat_waits_for_the_storage_lock_off_the_event_loop(client, monkeypatch):
    """While the storage lock is held elsewhere, a receive-time heartbeat waits for it in a worker thread: the event
    loop keeps running (this coroutine observes the heartbeat still blocked), and the upload completes once the lock
    is free."""
    monkeypatch.setattr(limits, "HEARTBEAT_SECONDS", 0, raising=False)
    sid = client.post("/api/sessions").json()["id"]
    real = documents.touch

    async def scenario():
        loop, entered, returned = asyncio.get_running_loop(), asyncio.Event(), threading.Event()

        def spy(part):
            loop.call_soon_threadsafe(entered.set)
            try:
                real(part)
            finally:
                returned.set()
        up = Upload(sid, _three_parts()).start()
        await _wait(up.asked[1])
        holder, held, release = _lock_holder()
        assert await asyncio.to_thread(held.wait, WAIT)
        monkeypatch.setattr(documents, "touch", spy)
        up.released[1].set()
        await _first(entered, up.asked[2])
        assert entered.is_set() and not up.asked[2].is_set(), "no heartbeat while the body was arriving"
        assert not returned.is_set(), "the heartbeat ran on the event loop thread"
        release.set()
        await asyncio.to_thread(holder.join, WAIT)
        assert await up.finish() == 201
        assert returned.is_set()
    _run(scenario)
    assert len(client.get(f"/api/sessions/{sid}/documents").json()["documents"]) == 1


def test_an_upload_cancelled_during_its_receive_heartbeat_still_releases(client, monkeypatch, spools):
    """The request is cancelled while its receive-time heartbeat waits for the storage lock in a worker thread. The
    upload still releases its reservation and pieces once the lock is free, and the heartbeat's thread finishes."""
    monkeypatch.setattr(limits, "HEARTBEAT_SECONDS", 0, raising=False)
    sid = client.post("/api/sessions").json()["id"]
    real = documents.touch

    async def scenario():
        loop, entered = asyncio.get_running_loop(), asyncio.Event()

        def spy(part):
            loop.call_soon_threadsafe(entered.set)
            real(part)
        up = Upload(sid, _three_parts()).start()
        await _wait(up.asked[1])
        holder, held, release = _lock_holder()
        assert await asyncio.to_thread(held.wait, WAIT)
        monkeypatch.setattr(documents, "touch", spy)
        up.released[1].set()
        await _first(entered, up.asked[2])
        assert entered.is_set() and not up.asked[2].is_set()
        up.task.cancel()
        release.set()
        await asyncio.to_thread(holder.join, WAIT)
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(up.task, WAIT)
    _run(scenario)
    assert _pieces(sid) == [] and quota.usage(_root())[1] == 0
    assert spools and all(f.closed for f in spools)


def test_concurrent_uploads_cleanup_spares_the_slow_live_one_and_reclaims_the_stalled_one(client, monkeypatch,
                                                                                         part_opens):
    """Two uploads have been quiet for over an hour; then one of them (slow) sends more and the other (stalled) does
    not. A third upload starts meanwhile and its cleanup must reclaim only the stalled one: the slow and the third
    upload both complete, the stalled one is refused with 409 without ever writing its .part."""
    monkeypatch.setattr(limits, "HEARTBEAT_SECONDS", 0, raising=False)
    slow_sid, stalled_sid, third_sid = (client.post("/api/sessions").json()["id"] for _ in range(3))

    async def scenario():
        slow, stalled = Upload(slow_sid, _three_parts()).start(), Upload(stalled_sid, _three_parts()).start()
        await _wait(slow.asked[1])
        await _wait(stalled.asked[1])
        slow_resv, stalled_resv = _resv(slow_sid), _resv(stalled_sid)
        _age(slow_resv)
        _age(stalled_resv)
        slow.released[1].set()
        await _wait(slow.asked[2])
        third = Upload(third_sid, [FILE_HEAD + PDF + TAIL]).start()           # its reservation runs cleanup
        assert await third.finish() == 201
        assert slow_resv.exists(), "the third upload's cleanup reclaimed a live upload"
        assert not stalled_resv.exists()
        assert await slow.finish() == 201
        assert await stalled.finish() == 409 and EXPIRED.encode() in stalled.body
    _run(scenario)
    for sid in (slow_sid, third_sid):
        assert len(client.get(f"/api/sessions/{sid}/documents").json()["documents"]) == 1
    assert _pieces(stalled_sid) == []
    assert not [p for p in part_opens if p.parent.name == stalled_sid]
    assert quota.usage(_root())[1] == 2 * len(PDF)
