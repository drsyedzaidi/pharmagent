"""The planner policy rejects proposals that name models outside the PK model library.

Found by the replicate-divergence harness: gemma4:31b proposed
``fit_pk_model(models=['1cmt_oral', '2cmt_oral'])`` and later
``['1cmt_abs', '2cmt_abs']``; the registry keys are ``oral_1cmt`` /
``oral_2cmt``. Those proposals reached the human gate, and ``fit_pk_model``
silently drops unknown names from ``models``, so approving one would run a
different comparison than the one shown. The policy now refuses them first,
naming the valid keys so the planner can correct itself.
"""
from __future__ import annotations

import pytest

from app.compute.pk_models import REGISTRY
from app.core import policy
from app.core.pharmstate import PharmState
from app.tools.builtins import default_registry

_NLME_READY = PharmState(pk_model_results={"status": "ok"})
_SCM_READY = PharmState(pk_model_results={"status": "ok"}, nlme_results={"status": "ok"})


def _roster(*names: str) -> dict:
    reg = default_registry()
    return {n: reg.get(n) for n in names}


def _decide(tool: str, args: dict, state: PharmState = PharmState(), history: list | None = None):
    roster = _roster("fit_pk_model", "run_nlme", "run_scm", "run_dose_sweep",
                     "simulate_pk_profile", "fit_compartmental")
    return policy.decide(tool, args, state, roster=roster, history=history or [],
                         budget=policy.Budget())


@pytest.mark.parametrize("models", [["1cmt_oral", "2cmt_oral"], ["1cmt_abs", "2cmt_abs"]])
def test_invented_models_are_rejected_before_the_gate(models: list[str]) -> None:
    d = _decide("fit_pk_model", {"compare": True, "models": models})
    assert d.allow is False
    assert d.mode == policy.MODE_REJECTED
    assert d.stop is False                       # the planner may correct and re-propose
    for name in models:
        assert repr(name) in d.reason
    assert "oral_1cmt" in d.reason and "oral_2cmt" in d.reason   # valid keys are named


def test_only_the_unknown_names_are_reported() -> None:
    d = _decide("fit_pk_model", {"compare": True, "models": ["oral_1cmt", "2cmt_oral"]})
    assert d.allow is False
    assert "'2cmt_oral'" in d.reason
    assert "'oral_1cmt'" not in d.reason


def test_non_string_model_names_are_rejected() -> None:
    d = _decide("fit_pk_model", {"compare": True, "models": ["oral_1cmt", 2]})
    assert d.allow is False and d.mode == policy.MODE_REJECTED


@pytest.mark.parametrize("tool, state", [
    ("fit_pk_model", PharmState()),
    ("run_nlme", _NLME_READY),
    ("run_scm", _SCM_READY),
    ("run_dose_sweep", _SCM_READY),
    ("simulate_pk_profile", PharmState()),
])
def test_unknown_model_key_is_rejected_on_every_model_library_tool(tool: str, state: PharmState) -> None:
    d = _decide(tool, {"model_key": "2cmt_oral"}, state)
    assert d.allow is False and d.mode == policy.MODE_REJECTED
    assert "'2cmt_oral'" in d.reason


def test_model_key_is_checked_before_state_prerequisites() -> None:
    """The bad name is the real blocker even when the fit is not ready yet."""
    d = _decide("run_nlme", {"model_key": "2cmt_oral"}, PharmState())
    assert "'2cmt_oral'" in d.reason
    assert "requires" not in d.reason


def test_valid_model_names_still_reach_the_gate() -> None:
    d = _decide("fit_pk_model", {"compare": True, "models": ["oral_1cmt", "oral_2cmt"]})
    assert d.allow is True and d.mode == policy.MODE_GATED
    d = _decide("run_nlme", {"model_key": "oral_2cmt"}, _NLME_READY)
    assert d.allow is True and d.mode == policy.MODE_GATED


def test_proposals_without_model_arguments_are_unchanged() -> None:
    d = _decide("fit_pk_model", {"compare": True})
    assert d.allow is True and d.mode == policy.MODE_AUTONOMOUS


def test_fit_compartmental_models_are_a_different_namespace() -> None:
    """``fit_compartmental`` takes '1cmt' / '2cmt', not model-library keys."""
    d = _decide("fit_compartmental", {"models": ["1cmt", "2cmt"]})
    assert "model library" not in d.reason


def test_an_identical_re_proposal_after_a_model_key_rejection_stops_the_leg() -> None:
    args = {"compare": True, "models": ["1cmt_oral"]}
    first = _decide("fit_pk_model", args)
    history = [{"tool": "fit_pk_model", "args": args, "allow": first.allow,
                "mode": first.mode, "reason": first.reason}]
    again = _decide("fit_pk_model", args, history=history)
    assert again.allow is False and again.stop is True


def test_every_library_key_is_accepted() -> None:
    for key in REGISTRY:
        d = _decide("simulate_pk_profile", {"model_key": key})
        assert "model library" not in d.reason, key


# ── through the real goal leg ────────────────────────────────────────────────

def test_goal_leg_never_parks_an_invented_model_and_the_planner_can_correct_itself(monkeypatch) -> None:
    """The harness finding end to end: the invented proposal is rejected with the valid
    keys in its history, nothing is parked for approval, and the corrected proposal
    (same tool, library keys) is what reaches the human gate. No fit runs."""
    from pathlib import Path

    from app.core.llm import MockLLM
    from app.core.orchestrator import Orchestrator
    from app.core.store import SessionStore
    from app.workflows import WORKFLOWS

    sample = str(Path(__file__).parent.parent / "sample_data" / "oral_pk.csv")
    monkeypatch.setitem(WORKFLOWS, "goal_models", {
        "name": "goal_models", "kind": "goal", "goal": "compare structural models",
        "allowed_tools": ["load_dataset", "fit_pk_model"]})

    class Planner:
        plans = [
            {"tool": "load_dataset", "args": {}, "rationale": "load"},
            {"tool": "fit_pk_model", "args": {"compare": True, "models": ["1cmt_oral", "2cmt_oral"]},
             "rationale": "compare 1- and 2-compartment oral models"},
            {"tool": "fit_pk_model", "args": {"compare": True, "models": ["oral_1cmt", "oral_2cmt"]},
             "rationale": "same comparison with library keys"},
        ]

        def __init__(self) -> None:
            self.seen_history: list[list[dict]] = []

        def plan(self, goal, state_summary, observations, history, tools):
            self.seen_history.append([dict(h) for h in history])
            return self.plans.pop(0)

    llm = Planner()
    orch = Orchestrator(llm=MockLLM(), clock=lambda c=iter(range(10_000)): f"t{next(c)}",
                        store=SessionStore(":memory:"))
    orch.llm = llm
    sid = orch.create_session().id
    out = orch.start_workflow(sid, "goal_models", {"path": sample})

    assert out["status"] == "awaiting_review"
    pending = out["state"]["pending_tool"]
    assert pending["tool"] == "fit_pk_model"
    assert pending["args"]["models"] == ["oral_1cmt", "oral_2cmt"]     # the corrected call, not the invented one
    hist = out["state"]["planner_history"]
    rejected = [h for h in hist if not h["allow"]]
    assert len(rejected) == 1 and "'1cmt_oral'" in rejected[0]["reason"]
    # the planner saw that rejection, with the valid keys, before its third proposal
    last_seen = llm.seen_history[-1]
    assert any("oral_1cmt" in h["reason"] and not h["allow"] for h in last_seen)
    assert out["state"]["pk_model_results"] is None                    # nothing was fitted
