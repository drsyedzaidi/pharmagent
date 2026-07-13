"""Identity-aware audit + provenance: actor/reason are tamper-evident, the chain
stays backward-compatible with pre-identity entries, session creation stamps
provenance, and the human-review approval is a signed audit entry."""
from __future__ import annotations

import itertools
from pathlib import Path

from app.core.audit import GENESIS, AuditChain, AuditEntry
from app.core.llm import MockLLM
from app.core.orchestrator import Orchestrator
from app.core.provenance import collect_provenance, file_sha256
from app.core.store import SessionStore

SAMPLE = str(Path(__file__).parent.parent / "sample_data" / "oral_pk.csv")


def _orch():
    counter = itertools.count()
    return Orchestrator(llm=MockLLM(), clock=lambda: f"t{next(counter)}",
                        store=SessionStore(":memory:"))


# ── audit identity ────────────────────────────────────────────────────────────

def test_actor_and_reason_recorded_and_chain_verifies():
    chain = AuditChain()
    chain.append(agent="qc", tool="human_review", action="approved",
                 inputs={"approve": True}, outputs={}, timestamp="t0",
                 actor="alice@lab", reason="trough within target")
    e = chain.entries[-1]
    assert e.actor == "alice@lab" and e.reason == "trough within target"
    assert chain.verify() is True


def test_tampering_with_actor_breaks_the_chain():
    chain = AuditChain()
    chain.append(agent="qc", tool="human_review", action="approved", inputs={},
                 outputs={}, timestamp="t0", actor="alice")
    chain.entries[-1].actor = "mallory"      # forge the signer post-hoc
    assert chain.verify() is False           # hash no longer matches


def test_backward_compatible_with_pre_identity_entries():
    """An entry persisted before actor/reason existed (no such keys) must still
    rebuild and verify — empty-string defaults hash identically to the old form."""
    legacy = AuditChain()
    legacy.append(agent="nca", tool="compute_nca", action="x",
                  inputs={"a": 1}, outputs={"b": 2}, timestamp="t0")
    d = legacy.entries[0].to_dict()
    d.pop("actor"); d.pop("reason")          # simulate an old persisted dict
    rebuilt = AuditChain.from_list([d])
    assert rebuilt.verify() is True
    # and a legacy-style hash equals one computed without the identity fields.
    e = AuditEntry(index=0, timestamp="t0", agent="nca", tool="compute_nca",
                   action="x", inputs_hash=legacy.entries[0].inputs_hash,
                   outputs_hash=legacy.entries[0].outputs_hash, prev_hash=GENESIS)
    assert e.compute_hash() == legacy.entries[0].entry_hash


# ── provenance ────────────────────────────────────────────────────────────────

def test_provenance_has_versions_and_is_constant():
    p = collect_provenance()
    for key in ("app_version", "python", "platform", "numpy", "scipy", "pandas", "git_sha"):
        assert key in p and p[key]
    assert collect_provenance() == p          # cached / constant per process


def test_file_sha256_stable_and_missing_safe():
    h1 = file_sha256(SAMPLE)
    assert len(h1) == 64 and h1 == file_sha256(SAMPLE)
    assert file_sha256("/no/such/file.csv") == "n/a"


# ── orchestrator-level identity wiring ────────────────────────────────────────

def test_session_creation_stamps_provenance_genesis():
    orch = _orch()
    sess = orch.create_session(owner="alice")
    first = sess.audit.entries[0]
    assert first.tool == "session" and first.action == "session_created"
    assert first.actor == "alice"
    assert sess.audit.verify() is True


def test_human_review_is_a_signed_audit_entry():
    orch = _orch()
    sid = orch.create_session(owner="alice").id
    orch.start_workflow(sid, "nca_full", {"path": SAMPLE})   # runs to the review gate
    n_before = len(orch.get_session(sid).audit.entries)
    orch.resume_workflow(sid, approve=True, actor="bob@lab", reason="looks good")
    entries = orch.get_session(sid).audit.entries
    hr = [e for e in entries if e.tool == "human_review"]
    assert hr and hr[-1].actor == "bob@lab" and hr[-1].reason == "looks good"
    assert "approved" in hr[-1].action
    assert len(entries) > n_before
    assert orch.get_session(sid).audit.verify() is True


def test_set_roles_is_audited_with_actor():
    orch = _orch()
    sid = orch.create_session(owner="alice").id
    orch.chat(sid, f"load dataset {SAMPLE}", actor="alice")
    orch.set_roles(sid, {"CMT": "TIME"}, actor="alice", reason="fix mapping")
    roles_entries = [e for e in orch.get_session(sid).audit.entries if e.tool == "set_roles"]
    assert roles_entries and roles_entries[-1].actor == "alice"
    assert roles_entries[-1].reason == "fix mapping"
    assert orch.get_session(sid).audit.verify() is True
