"""Parse slots are scheduled fairly: a session holds at most one slot, waiting parses queue FIFO with a
bounded wait and a bounded queue, grants rotate round-robin across sessions (a session with many queued
uploads cannot starve the others), and the refusal is a 503 with a Retry-After header. A fake slow parse
stands in for the child process; the session id travels from ingest into run_isolated."""
from __future__ import annotations

import itertools
import threading
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.config import settings
from app.core.llm import MockLLM
from app.core.orchestrator import Orchestrator
from app.core.store import SessionStore
from app.evidence import ingest, parse_sandbox, upload_limits
from app.evidence import store as evstore
from app.evidence.parse_queue import FairSlots, ParseBusyError
from app.evidence.routes import http_error
from tests import evidence_rawpdf as raw

BOX = [0, 0, 612, 792]


class Probe:
    """Records who holds a slot and when, with a fake slow parse in place of the child process."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.running: dict[str, int] = {}
        self.peak: dict[str, int] = {}
        self.total_peak = 0
        self.order: list[str] = []

    def hold(self, session: str, seconds: float) -> None:
        with self.lock:
            self.running[session] = self.running.get(session, 0) + 1
            self.peak[session] = max(self.peak.get(session, 0), self.running[session])
            self.total_peak = max(self.total_peak, sum(self.running.values()))
            self.order.append(session)
        time.sleep(seconds)
        with self.lock:
            self.running[session] -= 1


def _fake_child(monkeypatch, probe: Probe, seconds: float) -> None:
    """Replace the child process by a sleep, attributed to the session that asked."""
    current = threading.local()
    real_run = parse_sandbox.run_isolated

    def run(path, *, target, wall_s, max_rss, session_id=None):
        current.sid = session_id
        return real_run(path, target=target, wall_s=wall_s, max_rss=max_rss, session_id=session_id)

    monkeypatch.setattr(parse_sandbox, "_start", lambda args: SimpleNamespace(is_alive=lambda: False, exitcode=0))
    monkeypatch.setattr(parse_sandbox, "_watch", lambda proc, wall_s, max_rss: probe.hold(current.sid, seconds))
    monkeypatch.setattr(parse_sandbox, "_answer", lambda out, code: {})
    monkeypatch.setattr(parse_sandbox, "run_isolated", run)


def _parse_async(sid: str, results: dict, key: str, started: dict | None = None) -> threading.Thread:
    def go():
        t0 = time.monotonic()
        try:
            parse_sandbox.run_isolated("x.pdf", target="t:f", wall_s=5, max_rss=1, session_id=sid)
            results[key] = ("ok", time.monotonic() - t0)
        except ParseBusyError as exc:
            results[key] = ("busy", time.monotonic() - t0, exc.retry_after)

    th = threading.Thread(target=go, daemon=True)
    th.start()
    return th


def _wait_queued(slots: FairSlots, n: int) -> None:
    deadline = time.monotonic() + 3
    while slots.queued < n and time.monotonic() < deadline:
        time.sleep(0.005)
    assert slots.queued >= n


# --------------------------------------------------------------------------- slots ---

def test_one_session_cannot_hold_two_slots(monkeypatch):
    probe = Probe()
    _fake_child(monkeypatch, probe, 0.15)
    results: dict = {}
    threads = [_parse_async("attacker", results, f"a{i}") for i in range(3)]
    for th in threads:
        th.join(5)
    assert probe.peak["attacker"] == 1
    assert all(results[k][0] == "ok" for k in ("a0", "a1"))      # queued behind itself, not refused


def test_session_with_a_held_slot_leaves_the_free_slot_to_another_session(monkeypatch):
    probe = Probe()
    _fake_child(monkeypatch, probe, 0.5)
    results: dict = {}
    first = _parse_async("attacker", results, "a0")
    time.sleep(0.05)
    more = [_parse_async("attacker", results, f"a{i}") for i in (1, 2)]
    time.sleep(0.05)
    victim = _parse_async("victim", results, "v")
    victim.join(3)
    assert results["v"][0] == "ok" and results["v"][1] < 0.8      # its own 0.5 s parse; not the attacker's queue too
    for th in [first, *more]:
        th.join(5)


def test_attacker_sessions_cannot_block_a_victim_beyond_the_bounded_wait(monkeypatch):
    monkeypatch.setattr(parse_sandbox, "QUEUE_WAIT_S", 2.0)
    probe = Probe()
    _fake_child(monkeypatch, probe, 0.3)
    results: dict = {}
    threads = []
    for i in range(6):                                              # six attacker sessions
        threads.append(_parse_async(f"evil{i}", results, f"e{i}"))
        time.sleep(0.01)
    victim = _parse_async("victim", results, "v")
    victim.join(5)
    assert results["v"][0] == "ok", results["v"]
    assert results["v"][1] < 2.0 + 0.5
    for th in threads:
        th.join(5)


def test_wait_is_bounded_and_the_refusal_carries_retry_after(monkeypatch):
    monkeypatch.setattr(parse_sandbox, "QUEUE_WAIT_S", 0.3)
    probe = Probe()
    _fake_child(monkeypatch, probe, 1.5)
    results: dict = {}
    hogs = [_parse_async(f"hog{i}", results, f"h{i}") for i in range(2)]
    time.sleep(0.05)
    t0 = time.monotonic()
    victim = _parse_async("victim", results, "v")
    victim.join(3)
    assert results["v"][0] == "busy" and results["v"][2] > 0
    assert 0.25 <= results["v"][1] < 1.0, results["v"]
    assert time.monotonic() - t0 < 1.0
    for th in hogs:
        th.join(5)


# --------------------------------------------------------------------- fairness order ---

def _grant_order(slots: FairSlots, arrivals: list[str]) -> list[str]:
    """Hold the only slot as 'X', queue *arrivals* in order, release, and return the grant order."""
    slots.acquire("X", wait_s=1)
    order: list[str] = []
    done = threading.Event()
    threads = []

    def go(sid: str) -> None:
        slots.acquire(sid, wait_s=5)
        order.append(sid)
        time.sleep(0.02)
        slots.release(sid)

    for i, sid in enumerate(arrivals):
        th = threading.Thread(target=go, args=(sid,), daemon=True)
        th.start()
        threads.append(th)
        _wait_queued(slots, i + 1)
    slots.release("X")
    for th in threads:
        th.join(5)
    done.set()
    return order


def test_grants_rotate_round_robin_across_sessions():
    slots = FairSlots(capacity=1, max_queue=16, max_queued_per_session=4)
    assert _grant_order(slots, ["A", "A", "A", "B", "C"]) == ["A", "B", "C", "A", "A"]


def test_first_come_first_served_between_sessions():
    slots = FairSlots(capacity=1, max_queue=16, max_queued_per_session=4)
    assert _grant_order(slots, ["C", "B", "A"]) == ["C", "B", "A"]


def test_queue_is_bounded_per_session_and_in_total():
    slots = FairSlots(capacity=1, max_queue=3, max_queued_per_session=2)
    slots.acquire("X", wait_s=1)
    def queue_up(session: str) -> None:
        try:
            slots.acquire(session, wait_s=0.6)
            slots.release(session)
        except ParseBusyError:
            pass

    threads = [threading.Thread(target=queue_up, args=(s,), daemon=True) for s in ("A", "A", "B")]
    for i, th in enumerate(threads):
        th.start()
        _wait_queued(slots, i + 1)
    with pytest.raises(ParseBusyError) as per_session:           # A already has two queued
        slots.acquire("A", wait_s=0.5)
    assert per_session.value.retry_after > 0
    with pytest.raises(ParseBusyError):                          # the whole queue is full
        slots.acquire("D", wait_s=0.5)
    slots.release("X")
    for th in threads:
        th.join(3)


def test_expired_waiter_leaves_the_queue():
    slots = FairSlots(capacity=1, max_queue=4, max_queued_per_session=2)
    slots.acquire("X", wait_s=1)
    with pytest.raises(ParseBusyError):
        slots.acquire("A", wait_s=0.1)
    assert slots.queued == 0
    slots.release("X")
    slots.acquire("A", wait_s=0.1)
    slots.release("A")


# -------------------------------------------------------------------------- plumbing ---

def test_session_id_reaches_the_slot_scheduler_from_ingest(tmp_path, monkeypatch):
    evstore.configure(":memory:")
    seen: list[str] = []

    class Recording(FairSlots):
        def acquire(self, session, wait_s):
            seen.append(session)
            return super().acquire(session, wait_s)

    monkeypatch.setattr(parse_sandbox, "_slots", Recording(capacity=2, max_queue=16, max_queued_per_session=2))
    pdf = raw.build([(BOX, raw.text(raw.BODY))])
    src = ingest.ingest_pdf(pdf, "a.pdf", "sess-42", "alice", data_dir=tmp_path)
    assert src.parse_status == "parsed"
    assert seen and set(seen) == {"sess-42"}, seen        # the validation child and the parse child


def test_unattributed_calls_do_not_share_a_session_slot():
    slots = FairSlots(capacity=2, max_queue=4, max_queued_per_session=1)
    with slots.slot(None, wait_s=0.2), slots.slot(None, wait_s=0.2):
        pass


def test_busy_is_503_with_retry_after():
    exc = http_error(ParseBusyError("busy", retry_after=7))
    assert exc.status_code == 503 and exc.headers == {"Retry-After": "7"}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    evstore.configure(":memory:")
    upload_limits.reset()
    main.orch = Orchestrator(llm=MockLLM(), clock=lambda c=itertools.count(): f"t{next(c)}",
                             store=SessionStore(":memory:"))
    yield TestClient(main.app)
    upload_limits.reset()


def test_upload_refused_by_the_scheduler_is_503_with_header_and_records_nothing(client, monkeypatch):
    monkeypatch.setattr(parse_sandbox, "_slots", FairSlots(capacity=1, max_queue=1, max_queued_per_session=1))
    monkeypatch.setattr(parse_sandbox, "QUEUE_WAIT_S", 0.2)
    sid = client.post("/api/sessions").json()["id"]
    parse_sandbox._slots.acquire("someone-else", wait_s=1)      # another session holds the only slot
    try:
        pdf = raw.build([(BOX, raw.text(raw.BODY))])
        res = client.post(f"/api/sessions/{sid}/evidence/upload",
                          files={"file": ("a.pdf", pdf, "application/pdf")})
    finally:
        parse_sandbox._slots.release("someone-else")
    assert res.status_code == 503
    assert int(res.headers["Retry-After"]) > 0
    assert evstore.get_store().list_sources(sid) == []
