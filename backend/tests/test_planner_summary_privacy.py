"""The state summary the LLMs see must not carry subject ids.

The adversarial reviewer names per-subject findings ``<kind>-<subject id>``
(``recompute-cmax-1771``, ``nca-clf-sign-ABC``). ``Agent._state_summary`` listed
those ids verbatim as ``review_diagnostics.blocking_ids``, and that summary is
what both the chat tool-selection model and the goal planner receive. With a
hosted LLM a subject id would leave the machine. The summary now carries the
finding KIND only (``recompute-cmax``); the counts already say how many.

The ids are reduced by a prefix allow-list that mirrors what the review engine
emits, so a subject id of any shape (digits, letters, hyphens) is removed, and
a drift guard fails if the engine gains a per-subject family the list lacks.
Keyless: no LLM, no fit.
"""
from __future__ import annotations

import inspect
import itertools
import json
import re
from pathlib import Path

import pytest

from app.agents.base import Agent
from app.compute import adversarial
from app.core import finding_ids
from app.core.orchestrator import Orchestrator
from app.core.pharmstate import PharmState
from app.core.store import SessionStore
from app.workflows import WORKFLOWS

SAMPLE = str(Path(__file__).parent.parent / "sample_data" / "oral_pk.csv")

#: Subject ids of every shape seen in practice, including ones with no digit and with hyphens.
SUBJECTS = ["1771", "S07", "ABC", "S-A12", "001.5", "P_09"]

PER_SUBJECT_KINDS = [
    "nca-aucorder", "nca-extrap", "nca-clf-sign", "nca-clf-mag", "nca-vzf", "nca-thalf",
    "nca-clf-consist", "recompute-cmax", "recompute-aucband",
]
SUBJECT_FREE_IDS = [
    "nlme-converge", "nlme-cond", "nlme-iiv-CL", "nlme-shrink-V", "nlme-rse-ka",
    "nca-monotonic-100-300", "scm-postselection", "sim-uncertainty-omitted",
    "sim-covariate-range-unverifiable-standard_adult", "INCOMPLETE", "memo-stale",
]


# ── the helper ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("kind", PER_SUBJECT_KINDS)
@pytest.mark.parametrize("subject", SUBJECTS)
def test_a_per_subject_finding_id_is_reduced_to_its_kind_whatever_the_subject_id_looks_like(kind, subject):
    assert finding_ids.finding_kind(f"{kind}-{subject}") == kind


@pytest.mark.parametrize("fid", SUBJECT_FREE_IDS)
def test_ids_that_carry_no_subject_are_unchanged(fid):
    assert finding_ids.finding_kind(fid) == fid


@pytest.mark.parametrize("fid,expected", [
    ("nca-newcheck-1771", "nca-*"),
    ("nca-newcheck-ABC", "nca-*"),
    ("recompute-newkind-S-07", "recompute-*"),
])
def test_an_unknown_subject_capable_family_fails_closed(fid, expected):
    # An id in the nca-/recompute- namespace that the list does not know may end in a subject id.
    assert finding_ids.finding_kind(fid) == expected


def test_reduction_is_idempotent():
    for fid in [f"{k}-{s}" for k in PER_SUBJECT_KINDS for s in SUBJECTS] + SUBJECT_FREE_IDS + ["nca-x-1"]:
        once = finding_ids.finding_kind(fid)
        assert finding_ids.finding_kind(once) == once


def test_every_per_subject_family_the_review_engine_emits_is_on_the_list():
    """Drift guard. adversarial.py builds per-subject ids as f"<prefix>{sid}"; each prefix must be known."""
    src = inspect.getsource(adversarial)
    emitted = set(re.findall(r'f"((?:nca|recompute)-[a-z-]+-)\{sid\}"', src))
    assert emitted, "the engine's per-subject id pattern changed: update this guard"
    assert emitted <= set(finding_ids.SUBJECT_FINDING_PREFIXES), emitted - set(finding_ids.SUBJECT_FINDING_PREFIXES)


# ── the summary ───────────────────────────────────────────────────────────────

def _review(*ids: str, severity: str = "CRITICAL", resolved: bool = False) -> dict:
    return {
        "status": "FINDINGS BLOCK GOAL", "goal_met": False,
        "counts": {"CRITICAL": len(ids), "HIGH": 0, "MEDIUM": 0, "LOW": 0},
        "findings": [{"id": i, "severity": severity, "resolved": resolved,
                      "target": f"subject {i.rsplit('-', 1)[-1]} Cmax"} for i in ids],
    }


def _summary(review: dict) -> dict:
    return Agent(name="reviewer", system_prompt="")._state_summary(PharmState(review_results=review))


def test_blocking_ids_carry_the_kind_and_not_the_subject():
    ids = [f"recompute-cmax-{s}" for s in SUBJECTS] + [f"nca-clf-sign-{s}" for s in SUBJECTS]
    summary = _summary(_review(*ids))
    assert summary["review_diagnostics"]["blocking_ids"] == ["nca-clf-sign", "recompute-cmax"]
    blob = json.dumps(summary)
    for subject in SUBJECTS:
        assert f"-{subject}" not in blob and f'"{subject}"' not in blob, subject


def test_blocking_ids_are_deduplicated_and_sorted():
    summary = _summary(_review("recompute-cmax-3", "nca-clf-sign-1", "recompute-cmax-2", "nca-clf-sign-1"))
    assert summary["review_diagnostics"]["blocking_ids"] == ["nca-clf-sign", "recompute-cmax"]


def test_non_subject_blocking_ids_still_reach_the_model_so_it_can_reason_about_them():
    summary = _summary(_review("nlme-converge", "nca-monotonic-100-300", "recompute-cmax-1771"))
    assert summary["review_diagnostics"]["blocking_ids"] == ["nca-monotonic-100-300", "nlme-converge", "recompute-cmax"]


def test_resolved_and_lower_severity_findings_are_still_not_listed():
    assert _summary(_review("recompute-cmax-1771", resolved=True))["review_diagnostics"]["blocking_ids"] == []
    assert _summary(_review("recompute-cmax-1771", severity="MEDIUM"))["review_diagnostics"]["blocking_ids"] == []


def test_the_counts_and_status_the_model_relies_on_are_unchanged():
    rev = _review("recompute-cmax-1771", "recompute-cmax-1772")
    d = _summary(rev)["review_diagnostics"]
    assert d["status"] == rev["status"] and d["goal_met"] is False and d["counts"] == rev["counts"]


def test_a_review_without_findings_gives_an_empty_list():
    assert _summary({"status": "GOAL MET", "goal_met": True, "counts": {}, "findings": []}
                    )["review_diagnostics"]["blocking_ids"] == []


# ── both LLM paths ────────────────────────────────────────────────────────────

class _Recording:
    """Records what each LLM entry point is shown; stops at once."""

    def __init__(self) -> None:
        self.summaries: list[dict] = []

    def plan(self, goal, state_summary, observations, history, tools):
        self.summaries.append(state_summary)
        return {"tool": None, "args": {}, "rationale": "done", "stop_reason": "goal met"}

    def select_tool(self, agent, message, tools, state_summary):
        self.summaries.append(state_summary)
        return None

    def classify(self, message, options, descriptions):
        return options[0]


def _orch(llm) -> Orchestrator:
    return Orchestrator(llm=llm, clock=lambda c=itertools.count(): f"t{next(c)}", store=SessionStore(":memory:"))


def _seed_review(orch: Orchestrator, sid: str) -> None:
    from app.core.pharmstate import apply_writes
    sess = orch.get_session(sid)
    sess.state = apply_writes(sess.state, "reviewer", {"review_results": _review(
        *[f"recompute-cmax-{s}" for s in SUBJECTS], *[f"nca-clf-sign-{s}" for s in SUBJECTS])})
    orch._persist(sess)


def _assert_no_subject(summary: dict) -> None:
    assert summary["review_diagnostics"]["blocking_ids"] == ["nca-clf-sign", "recompute-cmax"]
    blob = json.dumps(summary, default=str)
    for subject in SUBJECTS:
        assert f"-{subject}" not in blob, subject


def test_the_goal_planner_never_receives_a_subject_id(monkeypatch):
    monkeypatch.setitem(WORKFLOWS, "goal_privacy", {
        "name": "goal_privacy", "kind": "goal", "goal": "privacy test", "allowed_tools": ["load_dataset"]})
    llm = _Recording()
    orch = _orch(llm)
    sid = orch.create_session().id
    _seed_review(orch, sid)
    orch.start_workflow(sid, "goal_privacy", {"path": SAMPLE})
    assert llm.summaries, "the planner was never consulted"
    for summary in llm.summaries:
        _assert_no_subject(summary)


def test_the_chat_tool_selection_model_never_receives_a_subject_id():
    llm = _Recording()
    orch = _orch(llm)
    sid = orch.create_session().id
    _seed_review(orch, sid)
    orch.chat(sid, "please run the nca review")
    assert llm.summaries, "select_tool was never consulted"
    for summary in llm.summaries:
        _assert_no_subject(summary)
