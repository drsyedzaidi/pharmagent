"""Render-queue isolation: per-session and global render slots shared across processes (flock), a bounded number of
waiters, and fail-closed behaviour. One session cannot use more than its share; a flood from one session cannot block
another; and the global cap holds across server workers, not just inside one process."""
from __future__ import annotations

import multiprocessing
import os
import signal
import threading
import time

import pytest

from app.config import settings
from app.pageimage import limits, slots

_RELEASES: list[threading.Event] = []


@pytest.fixture(autouse=True)
def _release_every_holder():
    """A failing assertion must never leave a holder thread blocked: that would hang the run instead of failing it."""
    _RELEASES.clear()
    yield
    for e in _RELEASES:
        e.set()


@pytest.fixture
def locks(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(limits, "QUEUE_WAIT_SECONDS", 0.2)
    return tmp_path / "data" / ".pageimage-locks"


def _hold(session, started, release, results):
    """Take a slot in a thread, report, and keep it until released."""
    try:
        with slots.render_slot(session):
            results.append(("ok", session))
            started.set()
            release.wait(10)
    except slots.SlotBusy:
        results.append(("busy", session))
        started.set()


def _in_thread(session):
    started, release, results = threading.Event(), threading.Event(), []
    _RELEASES.append(release)
    t = threading.Thread(daemon=True, target=_hold, args=(session, started, release, results))
    t.start()
    assert started.wait(10)
    return t, release, results


def test_the_global_cap_is_never_exceeded_under_concurrency(locks):
    active, peak, lock = 0, 0, threading.Lock()

    def work(i):
        nonlocal active, peak
        try:
            with slots.render_slot(f"s{i}", wait=5):
                with lock:
                    active += 1
                    peak = max(peak, active)
                time.sleep(0.05)
                with lock:
                    active -= 1
        except slots.SlotBusy:
            pass
    threads = [threading.Thread(daemon=True, target=work, args=(i,)) for i in range(12)]
    [t.start() for t in threads]
    [t.join(30) for t in threads]
    assert 1 <= peak <= limits.MAX_CONCURRENT_RENDERS


def test_one_session_cannot_hold_more_than_its_share_and_cannot_block_another(locks):
    t, release, res = _in_thread("alice")                       # alice holds her one slot
    assert res == [("ok", "alice")]
    started = time.monotonic()
    with pytest.raises(slots.SlotBusy):                         # her second render waits, then is refused
        with slots.render_slot("alice", wait=0.1):
            pass
    assert time.monotonic() - started >= 0.09
    t2, release2, res2 = _in_thread("bob")                      # bob is unaffected: a global slot is still free
    assert res2 == [("ok", "bob")]
    release.set(), release2.set()
    t.join(), t2.join()


def test_a_session_waiting_does_not_consume_a_global_slot(locks):
    """alice's second render blocks at the PER-SESSION gate, so it must not be holding one of the two global slots while
    it waits: bob and carol can both still run."""
    t, release, _ = _in_thread("alice")
    waiter_done = threading.Event()

    def alice_again():
        try:
            with slots.render_slot("alice", wait=1.0):
                pass
        except slots.SlotBusy:
            pass
        waiter_done.set()
    w = threading.Thread(daemon=True, target=alice_again)
    w.start()
    time.sleep(0.15)
    with slots.render_slot("bob", wait=0.2):                    # global slot 2 free for bob
        pass
    release.set()
    t.join(), w.join()
    assert waiter_done.is_set()


def test_the_number_of_waiting_threads_is_bounded_and_refusal_is_immediate(locks, monkeypatch):
    monkeypatch.setattr(limits, "MAX_WAITING_RENDERS", 2)
    monkeypatch.setattr(limits, "MAX_WAITING_PER_SESSION", 5)
    monkeypatch.setattr(limits, "MAX_CONCURRENT_RENDERS", 1)
    t, release, _ = _in_thread("holder")
    waiters = []
    for i in range(2):
        th = threading.Thread(daemon=True, target=lambda i=i: _swallow(f"w{i}", 2.0))
        th.start()
        waiters.append(th)
    time.sleep(0.2)                                             # both are now blocked in the queue
    started = time.monotonic()
    with pytest.raises(slots.SlotBusy):
        with slots.render_slot("late", wait=5):
            pass
    assert time.monotonic() - started < 0.5                     # refused at once, not after the 5 s wait
    release.set()
    [w.join(10) for w in waiters]
    t.join()


def test_one_session_cannot_fill_the_waiting_room(locks, monkeypatch):
    monkeypatch.setattr(limits, "MAX_WAITING_PER_SESSION", 1)
    monkeypatch.setattr(limits, "MAX_RENDERS_PER_SESSION", 1)
    t, release, _ = _in_thread("alice")
    th = threading.Thread(daemon=True, target=lambda: _swallow("alice", 2.0))
    th.start()
    time.sleep(0.2)
    started = time.monotonic()
    with pytest.raises(slots.SlotBusy):
        with slots.render_slot("alice", wait=5):
            pass
    assert time.monotonic() - started < 0.5
    with slots.render_slot("bob", wait=1):                      # bob still gets in: the room is not alice's to fill
        pass
    release.set()
    th.join(10), t.join()


def _swallow(session, wait):
    try:
        with slots.render_slot(session, wait=wait):
            pass
    except slots.SlotBusy:
        pass


def test_counters_do_not_leak_after_refusals(locks):
    t, release, _ = _in_thread("alice")
    for _ in range(20):
        with pytest.raises(slots.SlotBusy):
            with slots.render_slot("alice", wait=0.01):
                pass
    release.set()
    t.join()
    for _ in range(5):                                          # every counter went back to zero: nothing stays refused
        with slots.render_slot("alice", wait=1):
            pass
    assert slots._waiting_total == 0 and not slots._waiting_by_session


def test_lock_files_are_bounded_whatever_the_number_of_sessions(locks):
    for i in range(1000):
        with slots.render_slot(f"session-{i}", wait=1):
            pass
    names = [p.name for p in locks.iterdir()]
    assert len(names) <= limits.SESSION_BUCKETS * limits.MAX_RENDERS_PER_SESSION + limits.MAX_CONCURRENT_RENDERS


def test_a_killed_holder_frees_its_slot(locks, tmp_path):
    ctx = multiprocessing.get_context("spawn")
    from tests import _slot_workers
    ready = ctx.Event()
    p = ctx.Process(target=_slot_workers.hold_forever, args=(str(tmp_path / "data"), "alice", ready))
    p.start()
    assert ready.wait(60)
    with pytest.raises(slots.SlotBusy):
        with slots.render_slot("alice", wait=0.1):
            pass
    os.kill(p.pid, signal.SIGKILL)
    p.join(30)
    with slots.render_slot("alice", wait=2):                    # the OS released the flock when the process died
        pass


def _race(tmp_path, sessions, bypass, overrides=None):
    ctx = multiprocessing.get_context("spawn")
    from tests import _slot_workers
    n = len(sessions)
    attempted, queue = ctx.Barrier(n), ctx.Queue()
    procs = [ctx.Process(target=_slot_workers.try_and_hold,
                         args=(str(tmp_path / "data"), s, bypass, overrides or {}, attempted, queue))
             for s in sessions]
    [p.start() for p in procs]
    out = [queue.get(timeout=120) for _ in procs]
    [p.join(60) for p in procs]
    return out


def test_the_global_cap_holds_across_processes(locks, tmp_path):
    out = _race(tmp_path, ["a", "b", "c", "d"], bypass=False)
    assert sorted(out) == ["busy", "busy", "ok", "ok"]


def test_canary_without_flock_every_process_gets_in(locks, tmp_path):
    out = _race(tmp_path, ["a", "b", "c", "d"], bypass=True)
    assert out.count("ok") == 4, "the harness can no longer see a missing lock"


def test_the_per_session_cap_holds_across_processes(locks, tmp_path):
    out = _race(tmp_path, ["same"] * 3, bypass=False)
    assert sorted(out) == ["busy", "busy", "ok"]


def test_an_unusable_lock_directory_fails_closed(tmp_path, monkeypatch):
    blocker = tmp_path / "data"
    blocker.write_text("a file where the data directory should be")
    monkeypatch.setattr(settings, "data_dir", blocker)
    with pytest.raises(slots.SlotUnavailable):
        with slots.render_slot("alice", wait=0.1):
            pass
