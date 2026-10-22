"""Security review LOW 5: the provenance-status refresh writes sess.state, so it must run under the per-session lock
(a concurrent job holding the lock could otherwise lose an update) and must not deadlock re-entrant callers."""
from __future__ import annotations

import threading
import time

import pytest

from app.core.llm import MockLLM
from app.core.orchestrator import Orchestrator
from app.core.store import SessionStore


@pytest.fixture
def orch(tmp_path):
    return Orchestrator(llm=MockLLM(), store=SessionStore(str(tmp_path / "s.db")))


def test_the_refresh_runs_while_holding_the_session_lock(orch, monkeypatch):
    sid = orch.create_session().id
    seen = []
    original = orch._refresh_provenance_status

    def spy(sess):
        seen.append(orch.session_lock(sess.id)._is_owned())       # RLock: held by this thread?
        original(sess)

    monkeypatch.setattr(orch, "_refresh_provenance_status", spy)
    orch.get_session(sid)
    assert seen == [True]


def test_a_caller_already_holding_the_lock_does_not_deadlock(orch):
    sid = orch.create_session().id
    done = threading.Event()

    def worker():
        with orch.session_lock(sid):
            orch.get_session(sid)
            orch.get_session(sid)
        done.set()

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    assert done.wait(5), "get_session deadlocked while the caller held the session lock"


def test_a_refresh_waits_for_another_thread_holding_the_lock_then_proceeds(orch):
    sid = orch.create_session().id
    held = threading.Event()

    def holder():
        with orch.session_lock(sid):
            held.set()
            time.sleep(0.4)

    t = threading.Thread(target=holder, daemon=True)
    t.start()
    held.wait(2)
    start = time.monotonic()
    orch.get_session(sid)
    assert time.monotonic() - start >= 0.3                          # serialised behind the holder
    t.join(2)
    assert not t.is_alive()
