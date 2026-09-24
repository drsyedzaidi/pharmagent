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


def test_tampering_with_action_breaks_the_chain():
    # The human-review e-signature lives in `action`; flipping approved<->rejected
    # must be tamper-evident (v2 binds `action` into the hash).
    chain = AuditChain()
    chain.append(agent="qc", tool="human_review", action="approved", inputs={},
                 outputs={}, timestamp="t0", actor="alice")
    chain.entries[-1].action = "rejected"
    assert chain.verify() is False


def test_tampering_with_index_breaks_the_chain():
    chain = AuditChain()
    chain.append(agent="nca", tool="a", action="x", inputs={}, outputs={}, timestamp="t0")
    chain.append(agent="nca", tool="b", action="y", inputs={}, outputs={}, timestamp="t1")
    chain.entries[-1].index = 0              # renumber / reorder
    assert chain.verify() is False


def test_field_boundaries_cannot_be_shifted():
    """v2 delimits every field, so re-splitting two adjacent variable-length fields
    must NOT reproduce the same digest. Without delimiters an unsigned entry
    (actor='', reason='alice approved') could be rewritten to look signed
    (actor='alice', reason=' approved') with a byte-identical hash."""
    chain = AuditChain()
    chain.append(agent="qc", tool="human_review", action="approved", inputs={},
                 outputs={}, timestamp="t0", actor="", reason="alice approved")
    forged = chain.entries[0]
    before = forged.entry_hash
    forged.actor, forged.reason = "alice", " approved"
    assert forged.compute_hash() != before
    assert chain.verify() is False
    # same for the agent/tool boundary
    c2 = AuditChain()
    c2.append(agent="qc", tool="human_review", action="x", inputs={}, outputs={},
              timestamp="t0")
    c2.entries[0].agent, c2.entries[0].tool = "q", "chuman_review"
    assert c2.verify() is False


def test_field_boundaries_survive_injected_separators():
    """A delimiter join is only injective if the delimiter cannot appear in the
    fields — and `actor`/`reason` are user-supplied through the API. Shifting
    content across the boundary must not collide."""
    def mk(actor, reason):
        return AuditEntry(index=0, timestamp="t", agent="g", tool="T", action="A",
                          inputs_hash="i", outputs_hash="o", prev_hash=GENESIS,
                          actor=actor, reason=reason, hash_version=2)
    for sep in ("\x1f", "\x00", "|", "\n"):
        assert mk("a", f"b{sep}c").compute_hash() != mk(f"a{sep}b", "c").compute_hash(), sep


def test_unknown_hash_version_is_not_a_pass():
    """Relabelling the version must not preserve the digest, and a version this
    build cannot recompute must fail closed rather than be assumed valid."""
    chain = AuditChain()
    chain.append(agent="qc", tool="human_review", action="approved", inputs={},
                 outputs={}, timestamp="t0", actor="alice")
    chain.entries[0].hash_version = 999
    assert chain.verify() is False
    assert chain.verify_status()["unverifiable_entries"] == 1


def test_legacy_chain_reports_degraded_not_clean():
    """v1 cannot bind `action`, so a legacy approval can still be flipped. That
    weakness must be visible instead of reported as a clean verification."""
    legacy = AuditEntry(index=0, timestamp="t", agent="qc", tool="human_review",
                        action="approved", inputs_hash="i", outputs_hash="o",
                        prev_hash=GENESIS, hash_version=1)
    legacy.entry_hash = legacy.compute_hash()
    chain = AuditChain(); chain.entries = [legacy]
    status = chain.verify_status()
    assert status["ok"] is True and status["degraded"] is True
    assert status["legacy_entries"] == 1
    # a v2 chain is NOT degraded
    fresh = AuditChain()
    fresh.append(agent="qc", tool="human_review", action="approved", inputs={},
                 outputs={}, timestamp="t0")
    assert fresh.verify_status()["degraded"] is False


def test_from_list_survives_malformed_persisted_rows():
    """Persisted audit JSON is untrusted and is read for every session at startup:
    a bad row must not raise out of the constructor, and must not vanish."""
    good = AuditChain()
    good.append(agent="a", tool="b", action="c", inputs={}, outputs={}, timestamp="t0")
    d = good.entries[0].to_dict()
    # unknown key (e.g. written by a newer version) is dropped, entry still verifies
    assert AuditChain.from_list([{**d, "surprise": 1}]).verify() is True
    # a non-int hash_version is coerced rather than exploding inside compute_hash
    chain = AuditChain.from_list([{**d, "hash_version": "2"}])
    assert isinstance(chain.entries[0].hash_version, int)
    assert chain.verify() in (True, False)           # must not raise
    # an unrebuildable row is KEPT (dropping it would forge a shorter valid chain)
    for bad in ([{"index": 0}], ["bad"], [None], [42]):
        broken = AuditChain.from_list(bad)
        assert len(broken.entries) == 1, bad
        assert broken.entries[0].action == "unreadable_entry", bad
        assert broken.verify() is False, bad
    # a null semantic field must be coerced, not blow up inside compute_hash
    nulled = AuditChain.from_list([{**d, "action": None, "actor": None,
                                    "hash_version": 2}])
    assert nulled.verify() is False        # and must not raise


def test_backward_compatible_with_pre_identity_entries():
    """New entries are v2 (bind index + action); a genuinely legacy v1 entry —
    hashed under the old format, no actor/reason/hash_version keys — must still
    rebuild and verify unchanged."""
    chain = AuditChain()
    chain.append(agent="nca", tool="compute_nca", action="x",
                 inputs={"a": 1}, outputs={"b": 2}, timestamp="t0")
    assert chain.entries[0].hash_version == 2 and chain.verify() is True
    # a genuinely old persisted entry: v1 format, hashed without action/index/identity
    legacy = AuditEntry(index=0, timestamp="t0", agent="nca", tool="compute_nca",
                        action="x", inputs_hash="ih", outputs_hash="oh",
                        prev_hash=GENESIS, hash_version=1)
    legacy.entry_hash = legacy.compute_hash()
    d = legacy.to_dict()
    d.pop("actor"); d.pop("reason"); d.pop("hash_version")   # old persisted dict shape
    rebuilt = AuditChain.from_list([d])
    assert rebuilt.entries[0].hash_version == 1 and rebuilt.verify() is True
    # v1 and v2 of identical fields hash DIFFERENTLY — that is the fix.
    v2 = AuditEntry(**{**legacy.to_dict(), "hash_version": 2})
    assert v2.compute_hash() != legacy.entry_hash


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
