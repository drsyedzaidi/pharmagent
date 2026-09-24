"""Regression tests for #57: DB-only audit rewrites and truncation.

The attacker in these tests can replace any value in the primary SQLite
``sessions`` row, including recomputing the public v1/v2 SHA-256 chain.  They
cannot read the audit MAC keys or modify the independently stored anchor.
"""
from __future__ import annotations

import base64
import json
import sqlite3
import stat

import pytest
from pydantic import SecretStr

from app.config import Settings
from app.core.audit import AuditChain, AuditEntry
from app.core.audit_seal import (
    AuditConfigurationError,
    AuditEnrollmentRequired,
    AuditIntegrityError,
    AuditKeyring,
    AuditSecurity,
    FileAnchorStore,
    MemoryAnchorStore,
    audit_security_from_settings,
    enroll_legacy_session,
)
from app.core.llm import MockLLM
from app.core.orchestrator import Orchestrator
from app.core.store import SessionStore

KEY_ID = "test-2026-07"
KEY = b"k" * 32


def _security(anchor: MemoryAnchorStore | None = None) -> AuditSecurity:
    return AuditSecurity(
        keyring=AuditKeyring(active_key_id=KEY_ID, keys={KEY_ID: KEY}),
        anchor=anchor or MemoryAnchorStore(installation_id="install-test"),
    )


def _orch(db: str, security: AuditSecurity) -> Orchestrator:
    return Orchestrator(
        llm=MockLLM(),
        clock=iter(("t0", "t1", "t2", "t3", "t4", "t5", "t6")).__next__,
        store=SessionStore(db),
        audit_security=security,
    )


def _row(db: str, sid: str) -> tuple[str, str]:
    with sqlite3.connect(db) as conn:
        row = conn.execute(
            "SELECT audit_json, audit_seal_json FROM sessions WHERE id=?", (sid,)
        ).fetchone()
    assert row is not None
    return row[0], row[1]


def _replace_row(db: str, sid: str, audit_json: str, seal_json: str) -> None:
    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE sessions SET audit_json=?, audit_seal_json=? WHERE id=?",
            (audit_json, seal_json, sid),
        )
        conn.commit()


def _publicly_rechain(entries: list[dict]) -> list[dict]:
    """What a DB writer can do today without possessing any secret."""
    previous = "0" * 64
    rebuilt: list[dict] = []
    for index, raw in enumerate(entries):
        candidate = {
            **raw,
            "index": index,
            "prev_hash": previous,
            "hash_version": 2,
        }
        entry = AuditEntry(**candidate)
        entry.entry_hash = entry.compute_hash()
        rebuilt.append(entry.to_dict())
        previous = entry.entry_hash
    return rebuilt


def test_keyed_seal_binds_full_semantic_audit_snapshot():
    security = _security()
    chain = AuditChain()
    chain.append(
        agent="qc",
        tool="human_review",
        action="approved",
        inputs={},
        outputs={},
        timestamp="t0",
        actor="alice",
        reason="clinically reviewed",
    )
    seal = security.create_seal(
        session_id="sess_a",
        owner="alice",
        created_at="t0",
        entries=chain.to_list(),
        baseline="fresh",
    )

    status = security.inspect("sess_a", chain.to_list(), seal.to_dict())
    assert status["mac_ok"] is True
    assert status["anchor_ok"] is False
    assert status["verified"] is False

    forged = _publicly_rechain(
        [{**chain.to_list()[0], "actor": "mallory", "reason": "forged"}]
    )
    forged_status = security.inspect("sess_a", forged, seal.to_dict())
    assert forged_status["mac_ok"] is False
    assert forged_status["verified"] is False


def test_key_id_substitution_does_not_validate_under_another_known_key():
    anchor = MemoryAnchorStore(installation_id="install-test")
    security = AuditSecurity(
        keyring=AuditKeyring(
            active_key_id=KEY_ID,
            keys={KEY_ID: KEY, "also-known": b"z" * 32},
        ),
        anchor=anchor,
    )
    chain = AuditChain()
    chain.append(
        agent="system",
        tool="session",
        action="session_created",
        inputs={},
        outputs={},
        timestamp="t0",
    )
    seal = security.create_seal(
        session_id="sess_a",
        owner=None,
        created_at="t0",
        entries=chain.to_list(),
    ).to_dict()
    seal["key_id"] = "also-known"
    assert security.inspect("sess_a", chain.to_list(), seal)["mac_ok"] is False


def test_db_writer_cannot_rewrite_and_publicly_rechain(tmp_path):
    db = str(tmp_path / "rewrite.db")
    security = _security()
    orch = _orch(db, security)
    sid = orch.create_session(owner="alice").id
    audit_json, seal_json = _row(db, sid)

    forged = json.loads(audit_json)
    forged[0]["actor"] = "mallory"
    forged[0]["reason"] = "rewritten directly in sqlite"
    forged = _publicly_rechain(forged)
    _replace_row(db, sid, json.dumps(forged), seal_json)

    with pytest.raises(AuditIntegrityError):
        _orch(db, security)


def test_db_writer_cannot_replay_valid_older_seal_to_truncate_tail(tmp_path):
    db = str(tmp_path / "truncate.db")
    security = _security()
    orch = _orch(db, security)
    sess = orch.create_session(owner="alice")
    old_audit, old_seal = _row(db, sess.id)

    sess.audit.append(
        agent="qc",
        tool="human_review",
        action="approved",
        inputs={"approve": True},
        outputs={},
        timestamp="t4",
        actor="alice",
    )
    orch._persist(sess)
    assert len(json.loads(_row(db, sess.id)[0])) == 2

    # Both values are genuine and were once current.  The external monotonic
    # anchor, not the public chain, is what makes this rollback detectable.
    _replace_row(db, sess.id, old_audit, old_seal)
    with pytest.raises(AuditIntegrityError):
        _orch(db, security)


def test_cross_session_transplant_fails_session_bound_mac(tmp_path):
    db = str(tmp_path / "transplant.db")
    security = _security()
    orch = _orch(db, security)
    first = orch.create_session(owner="alice")
    second = orch.create_session(owner="alice")
    first_audit, first_seal = _row(db, first.id)

    _replace_row(db, second.id, first_audit, first_seal)
    with pytest.raises(AuditIntegrityError):
        _orch(db, security)


def test_owner_identity_is_bound_into_the_seal(tmp_path):
    db = str(tmp_path / "owner.db")
    security = _security()
    orch = _orch(db, security)
    sid = orch.create_session(owner="alice").id
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE sessions SET owner=? WHERE id=?", ("mallory", sid))
        conn.commit()

    with pytest.raises(AuditIntegrityError):
        _orch(db, security)


def test_whole_session_row_deletion_is_detected_by_orphan_anchor(tmp_path):
    db = str(tmp_path / "deleted-row.db")
    security = _security()
    orch = _orch(db, security)
    sid = orch.create_session(owner="alice").id
    with sqlite3.connect(db) as conn:
        conn.execute("DELETE FROM sessions WHERE id=?", (sid,))
        conn.commit()

    with pytest.raises(AuditIntegrityError, match="missing database session"):
        _orch(db, security)


def test_unknown_historical_key_fails_closed(tmp_path):
    db = str(tmp_path / "missing-key.db")
    anchor = MemoryAnchorStore(installation_id="install-test")
    orch = _orch(db, _security(anchor))
    orch.create_session(owner="alice")

    wrong_keys = AuditKeyring(active_key_id="replacement", keys={"replacement": b"r" * 32})
    with pytest.raises(AuditIntegrityError):
        _orch(db, AuditSecurity(keyring=wrong_keys, anchor=anchor))


def test_key_rotation_keeps_old_seals_verifiable(tmp_path):
    db = str(tmp_path / "rotation.db")
    anchor = MemoryAnchorStore(installation_id="install-test")
    original = _security(anchor)
    first = _orch(db, original)
    sid = first.create_session(owner="alice").id

    rotated_keys = AuditKeyring(
        active_key_id="replacement",
        keys={KEY_ID: KEY, "replacement": b"r" * 32},
    )
    rotated_security = AuditSecurity(keyring=rotated_keys, anchor=anchor)
    rotated = _orch(db, rotated_security)
    rotated._persist(rotated.get_session(sid))  # unchanged snapshot, new-key successor
    assert rotated.audit_status(sid)["generation"] == 2

    # Retaining both keys verifies the complete anchor history.
    assert _orch(db, rotated_security).audit_status(sid)["verified"] is True
    # Removing the historical key is a verification failure, not a clean pass.
    new_only = AuditSecurity(
        keyring=AuditKeyring(
            active_key_id="replacement", keys={"replacement": b"r" * 32}
        ),
        anchor=anchor,
    )
    with pytest.raises(AuditIntegrityError):
        _orch(db, new_only)


class _FailOnceAnchor(MemoryAnchorStore):
    fail_next = False

    def advance(self, expected_mac, seal):
        if self.fail_next:
            self.fail_next = False
            raise OSError("simulated anchor outage")
        return super().advance(expected_mac, seal)


class _AdvanceThenFailOnceAnchor(MemoryAnchorStore):
    fail_after_next = False

    def advance(self, expected_mac, seal):
        result = super().advance(expected_mac, seal)
        if self.fail_after_next:
            self.fail_after_next = False
            raise OSError("simulated lost anchor response")
        return result


def test_signed_one_generation_db_ahead_recovers_after_anchor_outage(tmp_path):
    db = str(tmp_path / "recover.db")
    anchor = _FailOnceAnchor(installation_id="install-test")
    security = _security(anchor)
    orch = _orch(db, security)
    sess = orch.create_session(owner="alice")
    sess.audit.append(
        agent="qc",
        tool="human_review",
        action="approved",
        inputs={},
        outputs={},
        timestamp="t4",
        actor="alice",
    )

    anchor.fail_next = True
    with pytest.raises(AuditIntegrityError, match="anchor is unavailable"):
        orch._persist(sess)

    # The request did not report success, but SQLite may already contain the
    # signed successor.  Restart accepts only this exact one-generation,
    # previous-MAC-linked successor and completes the idempotent anchor advance.
    recovered = _orch(db, security)
    status = recovered.audit_status(sess.id)
    assert status["verified"] is True
    assert status["generation"] == 2


def test_anchor_advance_is_idempotent_after_lost_success_response(tmp_path):
    db = str(tmp_path / "idempotent.db")
    anchor = _AdvanceThenFailOnceAnchor(installation_id="install-test")
    security = _security(anchor)
    orch = _orch(db, security)
    sess = orch.create_session(owner="alice")
    sess.audit.append(
        agent="qc",
        tool="human_review",
        action="approved",
        inputs={},
        outputs={},
        timestamp="t4",
        actor="alice",
    )
    anchor.fail_after_next = True
    with pytest.raises(AuditIntegrityError, match="anchor is unavailable"):
        orch._persist(sess)

    recovered = _orch(db, security)
    assert recovered.audit_status(sess.id)["verified"] is True
    assert len(anchor.history(sess.id)) == 2


def test_state_only_persistence_does_not_advance_audit_anchor(tmp_path):
    db = str(tmp_path / "state-only.db")
    anchor = MemoryAnchorStore(installation_id="install-test")
    security = _security(anchor)
    orch = _orch(db, security)
    sess = orch.create_session(owner="alice")
    before = orch.audit_status(sess.id)

    orch._persist(sess)
    after = orch.audit_status(sess.id)
    assert after["verified"] is True
    assert after["generation"] == before["generation"] == 1
    assert len(anchor.history(sess.id)) == 1


def test_sqlite_cas_rejects_edit_between_verification_and_write(tmp_path, monkeypatch):
    db = str(tmp_path / "cas.db")
    security = _security()
    orch = _orch(db, security)
    sess = orch.create_session(owner="alice")
    sess.audit.append(
        agent="qc",
        tool="human_review",
        action="approved",
        inputs={},
        outputs={},
        timestamp="t4",
        actor="alice",
    )
    original = orch.store.save_secured

    def race(**kwargs):
        audit_json, seal_json = _row(db, sess.id)
        forged = _publicly_rechain(
            [{**entry, "actor": "racing-writer"} for entry in json.loads(audit_json)]
        )
        _replace_row(db, sess.id, json.dumps(forged), seal_json)
        return original(**kwargs)

    monkeypatch.setattr(orch.store, "save_secured", race)
    with pytest.raises(AuditIntegrityError, match="during secured save"):
        orch._persist(sess)


def test_enforced_mode_refuses_unsealed_legacy_rows(tmp_path):
    db = str(tmp_path / "legacy.db")
    plain = Orchestrator(llm=MockLLM(), store=SessionStore(db))
    plain.create_session(owner="alice")

    with pytest.raises(AuditEnrollmentRequired):
        _orch(db, _security())


def test_unprotected_development_mode_is_never_reported_as_verified():
    orch = Orchestrator(llm=MockLLM(), store=SessionStore(":memory:"))
    sess = orch.create_session()
    status = orch.audit_status(sess.id)
    assert status["chain_ok"] is True
    assert status["mode"] == "hash_only"
    assert status["verified"] is False


def test_explicit_legacy_enrollment_attests_snapshot_without_retroactive_claim(tmp_path):
    db = str(tmp_path / "enroll.db")
    plain = Orchestrator(llm=MockLLM(), store=SessionStore(db))
    sid = plain.create_session(owner="alice").id
    security = _security()

    enrolled = enroll_legacy_session(
        SessionStore(db),
        security,
        sid,
        operator="security-admin",
        timestamp="migration-t0",
    )
    assert enrolled["verified"] is True
    assert enrolled["baseline"] == "operator_attested_legacy"
    assert enrolled["trusted_since_index"] == 1

    protected = _orch(db, security)
    status = protected.audit_status(sid)
    assert status["verified"] is True
    assert status["trusted_since_index"] == 1
    assert protected.get_session(sid).audit.entries[-1].action == (
        "legacy_snapshot_attested"
    )


def test_enrollment_seal_binds_v1_action_that_legacy_hash_omitted(tmp_path):
    db = str(tmp_path / "legacy-v1.db")
    plain = Orchestrator(llm=MockLLM(), store=SessionStore(db))
    sid = plain.create_session(owner="alice").id
    audit_json, _ = _row(db, sid)
    raw = json.loads(audit_json)[0]
    legacy = AuditEntry(**{**raw, "hash_version": 1})
    legacy.entry_hash = legacy.compute_hash()
    _replace_row(db, sid, json.dumps([legacy.to_dict()]), None)

    security = _security()
    enroll_legacy_session(
        SessionStore(db),
        security,
        sid,
        operator="security-admin",
        timestamp="migration-t0",
    )
    enrolled_audit, enrolled_seal = _row(db, sid)
    tampered = json.loads(enrolled_audit)
    tampered[0]["action"] = "forged-v1-action"
    # v1 structural verification still passes because action was never hashed.
    assert AuditChain.from_list(tampered).verify() is True
    _replace_row(db, sid, json.dumps(tampered), enrolled_seal)
    with pytest.raises(AuditIntegrityError):
        _orch(db, security)


def test_file_anchor_is_persistent_owner_only_and_monotonic(tmp_path):
    path = tmp_path / "protected" / "audit-anchor.jsonl"
    first = FileAnchorStore(path)
    security = AuditSecurity(
        keyring=AuditKeyring(active_key_id=KEY_ID, keys={KEY_ID: KEY}),
        anchor=first,
    )
    chain = AuditChain()
    chain.append(
        agent="system",
        tool="session",
        action="session_created",
        inputs={},
        outputs={},
        timestamp="t0",
    )
    seal = security.create_seal(
        session_id="sess_file",
        owner=None,
        created_at="t0",
        entries=chain.to_list(),
    )
    first.advance("0" * 64, seal)

    second = FileAnchorStore(path)
    assert second.installation_id == first.installation_id
    assert second.head("sess_file").seal_mac == seal.seal_mac
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_file_anchor_rejects_group_or_world_writable_parent(tmp_path):
    parent = tmp_path / "shared"
    parent.mkdir()
    parent.chmod(0o777)
    try:
        with pytest.raises(AuditConfigurationError, match="directory"):
            FileAnchorStore(parent / "audit-anchor.jsonl")
    finally:
        parent.chmod(0o700)


def test_corrupt_file_anchor_tail_fails_closed(tmp_path):
    path = tmp_path / "anchor.jsonl"
    FileAnchorStore(path)
    with path.open("ab") as handle:
        handle.write(b"{partial")
    with pytest.raises(AuditIntegrityError, match="journal is corrupt"):
        FileAnchorStore(path)


def test_malformed_seal_field_types_fail_closed_before_verification():
    security = _security()
    chain = AuditChain()
    chain.append(
        agent="system",
        tool="session",
        action="session_created",
        inputs={},
        outputs={},
        timestamp="t0",
    )
    seal = security.create_seal(
        session_id="sess_typed",
        owner=None,
        created_at="t0",
        entries=chain.to_list(),
    ).to_dict()
    seal["seal_mac"] = 123

    with pytest.raises(AuditIntegrityError, match="unsupported values"):
        security.assert_current("sess_typed", chain.to_list(), seal)


def test_enforced_configuration_requires_distinct_complete_strong_keys(tmp_path):
    encoded = base64.b64encode(KEY).decode()
    config = Settings(
        _env_file=None,
        audit_mode="enforce",
        audit_keys_json=SecretStr(json.dumps({KEY_ID: encoded})),
        audit_active_key_id=KEY_ID,
        audit_anchor_path=tmp_path / "anchor.jsonl",
    )
    security = audit_security_from_settings(config)
    assert security is not None
    assert security.keyring.active_key_id == KEY_ID
    assert encoded not in repr(security.keyring)

    with pytest.raises(AuditConfigurationError):
        audit_security_from_settings(
            Settings(
                _env_file=None,
                audit_mode="enforce",
                audit_keys_json=SecretStr(
                    json.dumps({KEY_ID: base64.b64encode(b"short").decode()})
                ),
                audit_active_key_id=KEY_ID,
                audit_anchor_path=tmp_path / "weak.jsonl",
            )
        )
    with pytest.raises(AuditConfigurationError):
        audit_security_from_settings(
            Settings(
                _env_file=None,
                audit_mode="enforce",
                audit_keys_json=SecretStr(json.dumps({KEY_ID: "not-base64***"})),
                audit_active_key_id=KEY_ID,
                audit_anchor_path=tmp_path / "malformed.jsonl",
            )
        )
    with pytest.raises(AuditConfigurationError):
        audit_security_from_settings(
            Settings(_env_file=None, api_token="production-token", audit_mode="off")
        )
