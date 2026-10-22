"""Re-review MEDIUM 2: the assurance label must reflect what THIS session's seal covers, not just that keyed mode is on.
A legacy session enrolled later is sealed from an operator-attested baseline, not since creation."""
from __future__ import annotations

from app.core.audit_seal import enroll_legacy_session
from app.core.llm import MockLLM
from app.core.orchestrator import Orchestrator
from app.core.store import SessionStore
from tests.test_audit_seal import _orch, _security


def test_a_session_sealed_since_creation_is_keyed_seal(tmp_path):
    db = str(tmp_path / "k.db")
    security = _security()
    sid = _orch(db, security).create_session().id
    assert _orch(db, security).get_session(sid).state.run_provenance_assurance == "keyed_seal"


def test_a_later_enrolled_legacy_session_is_labelled_attested_not_sealed_since_creation(tmp_path):
    db = str(tmp_path / "e.db")
    sid = Orchestrator(llm=MockLLM(), store=SessionStore(db)).create_session(owner="alice").id
    security = _security()
    enroll_legacy_session(SessionStore(db), security, sid, operator="ops", timestamp="t9")
    orch = _orch(db, security)
    sess = orch.get_session(sid)
    assert sess.state.run_provenance_assurance == "keyed_seal_attested_legacy"
    assert orch.audit_status(sid)["run_provenance_assurance"] == "keyed_seal_attested_legacy"


def test_report_text_for_an_attested_session_does_not_claim_sealing_since_creation():
    from app.core.pharmstate import PharmState
    from app.core.provenance import _assurance_text

    text = _assurance_text(PharmState(run_provenance_assurance="keyed_seal_attested_legacy"))
    assert "operator-attested" in text and "since creation" not in text.replace("not since creation", "")
