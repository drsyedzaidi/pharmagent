"""Owner gate 6: full-chain verification under the session lock must not create an avoidable availability problem.
Full verification at creation and restore; afterwards each access verifies only entries appended since the last check
(linked to the remembered head). A shorter chain or a changed head forces a full re-verify."""
from __future__ import annotations

import time

import pytest

from app.core.llm import MockLLM
from app.core.orchestrator import Orchestrator
from app.core.store import SessionStore


@pytest.fixture
def orch():
    return Orchestrator(llm=MockLLM(), store=SessionStore(":memory:"))


def _grow(sess, n):
    while len(sess.audit.entries) < n:
        sess.audit.append(agent="nca", tool="compute_nca", action="compute_nca(ds)", inputs={"i": len(sess.audit.entries)},
                          outputs={"summary": "x" * 200}, timestamp="2026-10-22T00:00:00+00:00", actor="alice")


def test_repeated_access_to_a_large_session_is_cheap(orch):
    sess = orch.create_session()
    _grow(sess, 20_000)
    orch.get_session(sess.id)                                       # pays for the new entries once
    t = time.perf_counter()
    for _ in range(20):
        orch.get_session(sess.id)
    per_call = (time.perf_counter() - t) / 20
    assert per_call < 0.01, f"{per_call * 1000:.1f} ms per access"     # was ~170 ms at this size


def test_an_appended_broken_entry_is_still_detected_incrementally(orch):
    sess = orch.create_session()
    _grow(sess, 50)
    assert orch.get_session(sess.id).state.run_provenance_status == "verified"
    _grow(sess, 51)
    sess.audit.entries[-1].outputs_hash = "0" * 64                    # tampered after the last check
    assert orch.get_session(sess.id).state.run_provenance_status == "mismatch"


def test_a_shorter_chain_or_changed_head_forces_a_full_reverify(orch):
    sess = orch.create_session()
    _grow(sess, 50)
    orch.get_session(sess.id)
    sess.audit.entries[10].entry_hash = "f" * 64                      # inside the verified prefix ...
    del sess.audit.entries[30:]                                       # ... and the chain is now shorter
    assert orch.get_session(sess.id).state.run_provenance_status == "mismatch"


def test_restore_always_verifies_the_full_chain(tmp_path):
    import json
    import sqlite3

    db = str(tmp_path / "s.db")
    o = Orchestrator(llm=MockLLM(), store=SessionStore(db))
    sid = o.create_session().id
    _grow(o.get_session(sid), 30)
    o._persist(o.get_session(sid))
    con = sqlite3.connect(db)
    (aj,) = con.execute("SELECT audit_json FROM sessions WHERE id = ?", (sid,)).fetchone()
    au = json.loads(aj)
    au[5]["outputs_hash"] = "0" * 64                                  # deep inside the chain
    con.execute("UPDATE sessions SET audit_json = ? WHERE id = ?", (json.dumps(au), sid))
    con.commit()
    con.close()
    assert Orchestrator(llm=MockLLM(), store=SessionStore(db)).get_session(sid).state.run_provenance_status == "mismatch"


@pytest.mark.parametrize("tamper", [None, "hash", "prev", "version", "truncate_mid"])
def test_incremental_and_full_checks_agree(orch, tamper):
    from app.core.provenance import chain_status

    sess = orch.create_session()
    _grow(sess, 40)
    cache: dict = {}
    chain_status(sess.audit, cache)                                   # warm the cache on the intact chain
    _grow(sess, 60)
    e = sess.audit.entries[50]
    if tamper == "hash":
        e.outputs_hash = "0" * 64
    elif tamper == "prev":
        e.prev_hash = "1" * 64
    elif tamper == "version":
        e.hash_version = 99
    elif tamper == "truncate_mid":
        del sess.audit.entries[45:55]
    full = sess.audit.verify_status()
    inc = chain_status(sess.audit, cache)
    assert (inc["ok"], inc["degraded"]) == (full["ok"], full["degraded"])
