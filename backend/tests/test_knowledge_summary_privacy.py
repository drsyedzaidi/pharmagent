"""The mikebrain critic must never be shown a per-subject finding id.

``Agent._state_summary`` lists the ids of unresolved blocking findings, and the adversarial reviewer names
per-subject findings after the subject (``recompute-cmax-1771``, ``nca-clf-sign-S07``). ``compact_state_summary``
fed those ids to the critic, so they reached the LLM prompt, the retrieval query and the persisted
``knowledge_results[].state_summary`` (returned in API state). With a hosted LLM configured, a subject id left
the machine. The summary now carries the finding KIND (``recompute-cmax``) only.

Every behavioural test seeds the review BEFORE consulting: a test that consults first, with no review in state,
cannot see this leak.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.core.audit import AuditChain
from app.core.llm import MockLLM
from app.core.pharmstate import PharmState
from app.knowledge import manifest as mf
from app.knowledge.extract import extract_text
from app.knowledge.index import NoEmbedder
from app.tools import knowledge_tools as kt
from app.tools.base import ToolContext
from app.tools.builtins import default_registry

FIXTURES = Path(__file__).parent / "fixtures" / "knowledge"
SUBJECT_IDS = ("recompute-cmax-1771", "nca-clf-sign-S07", "nca-aucorder-12")
SUBJECT_TOKENS = ("1771", "S07", "-12")
QUESTION = "Is covariate screening on EBEs valid when eta shrinkage is 45 percent?"


class _Spy:
    model = "spy"

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete(self, system, user, *, max_tokens=1024):
        self.prompts.append(user)
        return json.dumps({"verdict": "no_evidence", "rationale": "nothing relevant", "citations": []})


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch):
    monkeypatch.setattr(kt, "get_llm", MockLLM)
    monkeypatch.setattr(kt, "default_embedder", lambda: NoEmbedder())


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    root = tmp_path / "knowledge"
    for p in sorted(FIXTURES.glob("*.txt")):
        mf.add_document(root, mf.DocumentRecord(str(p), mf.file_sha256(p), p.stem, 2026, "paper", "t",
                                                datetime.now(UTC).isoformat()), extract_text(p))
    return tmp_path


def _state_with_review(*ids: str) -> PharmState:
    """A session that has ALREADY been reviewed and has unresolved blocking findings with these ids."""
    return PharmState(
        session_id="s1",
        nlme_results={"status": "ok", "converged": False, "condition_number": 5000.0,
                      "shrinkage_pct": {"V": 45.0}, "theta_rse_pct": {"CL": 9.0}},
        review_results={
            "status": "FINDINGS BLOCK GOAL", "goal_met": False,
            "counts": {"CRITICAL": len(ids), "HIGH": 0, "MEDIUM": 0, "LOW": 0},
            "findings": [{"id": i, "severity": "CRITICAL", "resolved": False} for i in ids]})


def test_blocking_ids_in_the_critic_summary_are_reduced_to_their_kind():
    summary = kt.compact_state_summary(_state_with_review(*SUBJECT_IDS, "nlme-converge"))
    ids = summary["review_diagnostics"]["blocking_ids"]
    assert ids == ["nca-aucorder", "nca-clf-sign", "nlme-converge", "recompute-cmax"]
    text = json.dumps(summary)
    assert not [t for t in SUBJECT_TOKENS if t in text], text


@pytest.mark.parametrize("other", ["nlme-converge", "nlme-cond", "nlme-shrink-V", "pkmodel-tie", "scm-postselection",
                                   "engine-single", "memo-stale", "nca-monotonic-100-200"])
def test_ids_that_carry_no_subject_are_left_exactly_as_they_were(other):
    got = kt.compact_state_summary(_state_with_review(other))["review_diagnostics"]["blocking_ids"]
    assert got == [other]


def test_a_review_that_already_holds_subject_findings_never_reaches_the_critic(data_dir, monkeypatch):
    spy = _Spy()
    monkeypatch.setattr(kt, "get_llm", lambda: spy)
    state, res = default_registry().execute(
        "consult_mikebrain", state=_state_with_review(*SUBJECT_IDS), ctx=ToolContext(data_dir=str(data_dir)),
        args={"question": QUESTION}, audit=AuditChain(), timestamp="t", actor="tester")
    assert res.result["status"] == "ok" and len(spy.prompts) == 1
    prompt = spy.prompts[0]
    assert "recompute-cmax" in prompt and "nca-clf-sign" in prompt           # the kind is still there to retrieve on
    for token in SUBJECT_TOKENS:
        assert token not in prompt, f"{token!r} reached the LLM prompt"
        assert token not in json.dumps(res.result), f"{token!r} reached the audited result"
        assert token not in json.dumps(state.knowledge_results), f"{token!r} reached the persisted knowledge_results"
        assert token not in kt.build_query(QUESTION, kt.compact_state_summary(_state_with_review(*SUBJECT_IDS)))


def test_the_advisory_path_of_adversarial_review_inherits_the_fix(data_dir, monkeypatch):
    """adversarial_review(consult_mikebrain=true) consults once per finding, from the state as it stands: an
    EARLIER review's subject-level findings are in that state."""
    spy = _Spy()
    monkeypatch.setattr(kt, "get_llm", lambda: spy)
    state, res = default_registry().execute(
        "adversarial_review", state=_state_with_review(*SUBJECT_IDS), ctx=ToolContext(data_dir=str(data_dir)),
        args={"consult_mikebrain": True}, audit=AuditChain(), timestamp="t", actor="tester")
    assert res.result["advisory_status"] == "ok" and spy.prompts
    for token in SUBJECT_TOKENS:
        assert all(token not in p for p in spy.prompts), token
        assert token not in json.dumps(res.result["advisories"]), token


@pytest.mark.parametrize("finding_id, kind", [
    ("nca-clf-sign-1771", "nca-clf-sign"), ("recompute-cmax-S07", "recompute-cmax"), ("nca-aucorder-12", "nca-aucorder"),
    ("nlme-converge", "nlme-converge"), ("nca-monotonic-100-200", "nca-monotonic-100-200"),
    ("NCA-derived", "NCA-derived"), ("recompute-from-raw-data", "recompute-from-raw-data")])
def test_finding_kind(finding_id, kind):
    assert kt.finding_kind(finding_id) == kind
