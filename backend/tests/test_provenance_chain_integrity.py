"""Review of 5d099fd..6aceff6, HIGH 1 (+ LOW 4): run-provenance status must require a fully valid audit chain, not just
matching genesis fields; repeated keys are a mismatch; the assurance level (hash_only / keyed_seal) is reported."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from app.core.audit import AuditChain, hash_payload
from app.core.llm import MockLLM
from app.core.orchestrator import Orchestrator
from app.core.provenance import verify_run_provenance
from app.core.store import SessionStore

SAMPLE = str(Path(__file__).parent.parent / "sample_data" / "oral_pk.csv")


@pytest.fixture
def orch(tmp_path):
    return Orchestrator(llm=MockLLM(), store=SessionStore(str(tmp_path / "s.db")))


def _copy_chain(chain: AuditChain) -> AuditChain:
    return AuditChain.from_list(json.loads(json.dumps(chain.to_list())))


def _rehash_from(chain: AuditChain, start: int) -> None:
    """Recompute entry hashes and prev links from ``start`` on (what an attacker with DB access would do)."""
    for i in range(start, len(chain.entries)):
        e = chain.entries[i]
        if i > 0:
            e.prev_hash = chain.entries[i - 1].entry_hash
        e.entry_hash = e.compute_hash()


def test_a_broken_genesis_hash_is_a_mismatch(orch):
    sess = orch.create_session()
    chain = _copy_chain(sess.audit)
    chain.entries[0].entry_hash = "0" * 64
    assert verify_run_provenance(sess.state.run_provenance, chain, sess.id) == "mismatch"


def test_a_broken_link_anywhere_in_the_chain_is_a_mismatch(orch):
    sess = orch.create_session()
    orch.run_tool(sess.id, "load_dataset", "data_manager", {"path": SAMPLE})
    chain = _copy_chain(orch.get_session(sess.id).audit)
    assert len(chain.entries) >= 2
    chain.entries[-1].prev_hash = "f" * 64
    assert verify_run_provenance(sess.state.run_provenance, chain, sess.id) == "mismatch"


def test_changed_provenance_with_changed_stored_hashes_is_rejected_unless_the_whole_chain_verifies(orch):
    sess = orch.create_session()
    orch.run_tool(sess.id, "load_dataset", "data_manager", {"path": SAMPLE})
    forged = tuple((k, "EVIL" if k == "git_sha" else v) for k, v in sess.state.run_provenance)
    chain = _copy_chain(orch.get_session(sess.id).audit)
    chain.entries[0].inputs_hash = hash_payload(dict(forged))            # reviewer probe B: field changed only
    assert verify_run_provenance(forged, chain, sess.id) == "mismatch"
    chain.entries[0].entry_hash = chain.entries[0].compute_hash()          # genesis rehashed, later link now broken
    assert verify_run_provenance(forged, chain, sess.id) == "mismatch"
    _rehash_from(chain, 1)                                                 # whole chain rebuilt consistently
    # Hash-only mode cannot tell a complete consistent rebuild from the original; that is exactly why the
    # assurance level is reported as hash_only (and keyed-seal mode rejects it on restore: see below).
    assert verify_run_provenance(forged, chain, sess.id) == "verified"


def test_only_a_chain_valid_pre_marker_genesis_is_legacy(orch):
    sess = orch.create_session()
    chain = _copy_chain(sess.audit)
    chain.entries[0].outputs_hash = hash_payload({"session_id": sess.id})  # reviewer probe A, not rehashed
    assert verify_run_provenance(None, chain, sess.id) == "mismatch"
    chain.entries[0].entry_hash = chain.entries[0].compute_hash()          # now a consistent pre-marker chain
    assert verify_run_provenance(None, chain, sess.id) == "legacy_unverified"


def test_repeated_keys_in_the_record_are_a_mismatch(orch):
    sess = orch.create_session()
    rec = sess.state.run_provenance
    assert verify_run_provenance((("git_sha", "EVIL"),) + rec, sess.audit, sess.id) == "mismatch"


def test_a_db_downgrade_without_rehash_quarantines_the_session(tmp_path):
    db = str(tmp_path / "s.db")
    sid = Orchestrator(llm=MockLLM(), store=SessionStore(db)).create_session().id
    con = sqlite3.connect(db)
    sj, aj = con.execute("SELECT state_json, audit_json FROM sessions WHERE id = ?", (sid,)).fetchone()
    st, au = json.loads(sj), json.loads(aj)
    st["run_provenance"] = None
    au[0]["outputs_hash"] = hash_payload({"session_id": sid})
    con.execute("UPDATE sessions SET state_json = ?, audit_json = ? WHERE id = ?", (json.dumps(st), json.dumps(au), sid))
    con.commit()
    con.close()
    sess = Orchestrator(llm=MockLLM(), store=SessionStore(db)).get_session(sid)
    assert sess.state.run_provenance_status == "mismatch"


# ── assurance level ─────────────────────────────────────────────────────────────────────────────
def test_hash_only_mode_reports_hash_only_assurance(orch):
    sid = orch.create_session().id
    assert orch.get_session(sid).state.run_provenance_assurance == "hash_only"
    assert orch.audit_status(sid)["run_provenance_assurance"] == "hash_only"


def test_keyed_seal_mode_reports_keyed_seal_assurance_and_rejects_a_full_rebuild(tmp_path):
    from app.core.audit_seal import AuditIntegrityError
    from tests.test_audit_seal import _orch, _security

    db = str(tmp_path / "k.db")
    security = _security()
    orch = _orch(db, security)
    sid = orch.create_session().id
    assert orch.get_session(sid).state.run_provenance_status == "verified"
    assert orch.get_session(sid).state.run_provenance_assurance == "keyed_seal"
    assert orch.audit_status(sid)["run_provenance_assurance"] == "keyed_seal"
    # full consistent rebuild of the row: the keyed seal no longer matches, so restore is refused
    con = sqlite3.connect(db)
    sj, aj = con.execute("SELECT state_json, audit_json FROM sessions WHERE id = ?", (sid,)).fetchone()
    st, chain = json.loads(sj), AuditChain.from_list(json.loads(aj))
    st["run_provenance"] = [[k, "EVIL" if k == "git_sha" else v] for k, v in st["run_provenance"]]
    chain.entries[0].inputs_hash = hash_payload({k: v for k, v in st["run_provenance"]})
    _rehash_from(chain, 0)
    con.execute("UPDATE sessions SET state_json = ?, audit_json = ? WHERE id = ?",
                (json.dumps(st), json.dumps(chain.to_list()), sid))
    con.commit()
    con.close()
    with pytest.raises(AuditIntegrityError):
        _orch(db, security)


def test_reports_state_the_assurance_level(orch, tmp_path):
    from docx import Document

    from app.tools.base import ToolContext
    from app.tools.report_tools import generate_report

    state = orch.create_session().state
    path = generate_report(state, ToolContext(data_dir=str(tmp_path)), {}).result["report_path"]
    text = "\n".join(p.text for p in Document(path).paragraphs)
    assert "assurance: hash_only" in text and "not cryptographically authenticated" in text
