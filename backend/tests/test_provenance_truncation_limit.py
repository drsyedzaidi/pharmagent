"""Re-review MEDIUM 1: in hash_only mode, removing trailing audit entries leaves a valid prefix, so it is not
detectable without a keyed seal or external anchor. This is a disclosed limit (like a consistent rebuild); keyed-seal
mode rejects it on restore."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from app.core.llm import MockLLM
from app.core.orchestrator import Orchestrator
from app.core.store import SessionStore

SAMPLE = str(Path(__file__).parent.parent / "sample_data" / "oral_pk.csv")


def _truncate(db: str, sid: str, keep: int) -> None:
    con = sqlite3.connect(db)
    (aj,) = con.execute("SELECT audit_json FROM sessions WHERE id = ?", (sid,)).fetchone()
    con.execute("UPDATE sessions SET audit_json = ? WHERE id = ?", (json.dumps(json.loads(aj)[:keep]), sid))
    con.commit()
    con.close()


def test_hash_only_truncation_is_a_disclosed_limit_labelled_hash_only(tmp_path):
    from app.core.provenance import _assurance_text, verify_run_provenance

    db = str(tmp_path / "s.db")
    orch = Orchestrator(llm=MockLLM(), store=SessionStore(db))
    sid = orch.create_session().id
    orch.run_tool(sid, "load_dataset", "data_manager", {"path": SAMPLE})
    _truncate(db, sid, 1)
    sess = Orchestrator(llm=MockLLM(), store=SessionStore(db)).get_session(sid)
    assert len(sess.audit.entries) == 1
    assert sess.state.run_provenance_status == "verified" and sess.state.run_provenance_assurance == "hash_only"
    assert "removal of trailing entries is not detectable" in _assurance_text(sess.state)
    assert "trailing entries" in verify_run_provenance.__doc__


def test_keyed_seal_mode_rejects_a_truncated_chain_on_restore(tmp_path):
    from app.core.audit_seal import AuditIntegrityError
    from tests.test_audit_seal import _orch, _security

    db = str(tmp_path / "k.db")
    security = _security()
    orch = _orch(db, security)
    sid = orch.create_session().id
    orch.run_tool(sid, "load_dataset", "data_manager", {"path": SAMPLE})
    _truncate(db, sid, 1)
    with pytest.raises(AuditIntegrityError):
        _orch(db, security)
