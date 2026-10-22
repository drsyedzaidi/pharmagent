"""Review of db3afb3, LOW 2 and LOW 3: the incremental-verification cache must be bound to the exact in-memory chain it
verified, must recompute (not just compare) its head entry before resuming, and must record the length it checked."""
from __future__ import annotations

import json

import pytest

from app.core.audit import AuditChain
from app.core.llm import MockLLM
from app.core.orchestrator import Orchestrator
from app.core.store import SessionStore


@pytest.fixture
def orch():
    return Orchestrator(llm=MockLLM(), store=SessionStore(":memory:"))


def _grow(sess, n):
    while len(sess.audit.entries) < n:
        sess.audit.append(agent="nca", tool="t", action="a", inputs={"i": len(sess.audit.entries)}, outputs={},
                          timestamp="t", actor="a")


def test_a_wholesale_replaced_chain_is_fully_reverified(orch):
    sess = orch.create_session()
    _grow(sess, 5)
    assert orch.get_session(sess.id).state.run_provenance_status == "verified"
    other = AuditChain.from_list(json.loads(json.dumps(sess.audit.to_list())))
    other.entries[1].tool = "FORGED"                                   # content changed, stored hash left alone
    sess.audit = other
    assert orch.get_session(sess.id).state.run_provenance_status == "mismatch"


def test_head_entry_content_changed_without_its_hash_is_detected(orch):
    sess = orch.create_session()
    _grow(sess, 5)
    orch.get_session(sess.id)
    sess.audit.entries[-1].tool = "FORGED2"                           # the cached head itself
    assert orch.get_session(sess.id).state.run_provenance_status == "mismatch"


def test_the_cache_records_the_length_it_actually_checked(orch):
    from app.core.provenance import chain_status

    sess = orch.create_session()
    _grow(sess, 7)
    cache: dict = {}
    chain_status(sess.audit, cache)
    assert cache["n"] == 7 and cache["head"] == sess.audit.entries[6].entry_hash
