"""Owner gate before push: the incremental-verification cache must not trust a verified prefix after any in-place
change to it. Recomputing the head does not cover ancestors (the head binds its stored prev_hash, not the recomputed
content of every earlier entry), so the chain carries a version that every modification changes."""
from __future__ import annotations

import pytest

from app.core.audit import AuditEntry
from app.core.llm import MockLLM
from app.core.orchestrator import Orchestrator
from app.core.store import SessionStore


@pytest.fixture
def orch():
    return Orchestrator(llm=MockLLM(), store=SessionStore(":memory:"))


def _verified_long(orch, n=400):
    sess = orch.create_session()
    while len(sess.audit.entries) < n:
        sess.audit.append(agent="nca", tool="t", action="a", inputs={"i": len(sess.audit.entries)}, outputs={},
                          timestamp="t", actor="a")
    assert orch.get_session(sess.id).state.run_provenance_status == "verified"   # cache now populated
    return sess


def _forged_copy(e):
    f = AuditEntry(**{**e.to_dict(), "tool": "FORGED"})                # stored hashes kept, content changed
    return f


MUTATIONS = {
    "attribute of a middle entry": lambda es: setattr(es[200], "tool", "FORGED"),
    "attribute of the genesis entry": lambda es: setattr(es[0], "reason", "FORGED"),
    "replace a middle entry": lambda es: es.__setitem__(200, _forged_copy(es[200])),
    "slice-assign a middle entry": lambda es: es.__setitem__(slice(200, 201), [_forged_copy(es[200])]),
}


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_an_in_place_change_inside_the_verified_prefix_is_a_mismatch(orch, name):
    sess = _verified_long(orch)
    entries, n, head = sess.audit.entries, len(sess.audit.entries), sess.audit.entries[-1]
    MUTATIONS[name](sess.audit.entries)
    assert sess.audit.entries is entries and len(entries) == n and entries[-1] is head   # identity/length/head kept
    assert orch.get_session(sess.id).state.run_provenance_status == "mismatch"


def test_a_plain_list_assigned_as_entries_is_still_tracked(orch):
    sess = _verified_long(orch, 50)
    sess.audit.entries = list(sess.audit.entries)
    assert orch.get_session(sess.id).state.run_provenance_status == "verified"
    sess.audit.entries[20].tool = "FORGED"
    assert orch.get_session(sess.id).state.run_provenance_status == "mismatch"


def test_appends_still_resume_incrementally(orch, monkeypatch):
    sess = _verified_long(orch, 300)
    calls = []
    orig = AuditEntry.compute_hash
    monkeypatch.setattr(AuditEntry, "compute_hash", lambda self: calls.append(1) or orig(self))
    sess.audit.append(agent="nca", tool="t", action="a", inputs={}, outputs={}, timestamp="t", actor="a")
    assert orch.get_session(sess.id).state.run_provenance_status == "verified"
    assert len(calls) < 10                                              # head + new entry, not 300
