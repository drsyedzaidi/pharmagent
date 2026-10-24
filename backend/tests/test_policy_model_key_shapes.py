"""Planner policy: model arguments must have the shape the tools actually read.

``model_key`` is ONE name (a string); ``models`` is a LIST of names. The
name check alone accepts two shapes that pass it but break the tool:

* ``models="oral_1cmt"`` (a bare string, a valid key): ``fit_pk_model`` iterates
  the string character by character, drops every one, and reports
  "Compared 0 PK models; best by AIC: none." after the human approved a
  comparison that fits nothing.
* ``model_key=["oral_1cmt"]`` (a list): the tool does ``model_key in REGISTRY``
  on a list and raises ``TypeError: unhashable type`` after approval.

Both are now refused before the gate, with a reason that says which shape each
argument needs. Keyless: policy decisions only, plus one scripted goal loop.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from app.core import policy
from app.core.orchestrator import Orchestrator
from app.core.pharmstate import PharmState
from app.core.store import SessionStore
from app.tools.builtins import default_registry
from app.workflows import WORKFLOWS

SAMPLE = str(Path(__file__).parent.parent / "sample_data" / "oral_pk.csv")
VALID = "oral_1cmt"


def _decide(tool: str, args: dict) -> policy.Decision:
    reg = default_registry()
    return policy.decide(tool, args, PharmState(), roster={tool: reg.get(tool)}, history=[],
                         budget=policy.Budget())


# ── wrong shapes are refused, naming the shape each argument needs ────────────

def test_a_bare_string_models_value_is_rejected_even_when_it_is_a_valid_key():
    d = _decide("fit_pk_model", {"compare": True, "models": VALID})
    assert d.allow is False and d.mode == policy.MODE_REJECTED and d.stop is False
    assert "models" in d.reason and "list" in d.reason
    assert f"models='{VALID}'" in d.reason


def test_a_list_model_key_is_rejected_even_when_it_holds_a_valid_key():
    d = _decide("fit_pk_model", {"model_key": [VALID]})
    assert d.allow is False and d.mode == policy.MODE_REJECTED
    assert "model_key" in d.reason and "string" in d.reason


@pytest.mark.parametrize("tool,args", [
    ("fit_pk_model", {"model_key": 7}),
    ("fit_pk_model", {"model_key": {"name": VALID}}),
    ("fit_pk_model", {"models": 5}),
    ("fit_pk_model", {"models": {VALID: 1}}),
    ("run_nlme", {"model_key": [VALID]}),
    ("run_scm", {"model_key": [VALID]}),
    ("run_dose_sweep", {"model_key": (VALID,)}),
    ("simulate_pk_profile", {"model_key": [VALID]}),
])
def test_every_model_tool_refuses_the_wrong_shape(tool, args):
    d = _decide(tool, args)
    assert d.allow is False and d.mode == policy.MODE_REJECTED
    assert "must be" in d.reason


def test_a_wrong_shape_and_an_unknown_name_are_both_reported_in_one_reason():
    d = _decide("fit_pk_model", {"model_key": [VALID], "models": ["1cmt_abs"]})
    assert d.allow is False
    assert "model_key" in d.reason and "'1cmt_abs'" in d.reason
    assert "valid keys:" in d.reason and VALID in d.reason


# ── the right shapes are untouched ────────────────────────────────────────────

@pytest.mark.parametrize("args", [
    {"model_key": VALID},
    {"models": [VALID]},
    {"models": (VALID, "oral_2cmt")},
    {"model_key": VALID, "models": [VALID]},
])
def test_correct_shapes_still_reach_the_gate(args):
    d = _decide("fit_pk_model", args)
    assert d.allow is True and d.mode == policy.MODE_GATED


@pytest.mark.parametrize("args", [{"models": []}, {"models": None}, {"model_key": None}, {"model_key": ""}])
def test_unspecified_values_are_still_not_checked(args):
    # The tools treat a falsy value as "use the fitted model / the default set".
    d = _decide("fit_pk_model", args)
    assert d.allow is True


def test_fit_compartmental_models_stay_outside_the_check():
    # Its names ("1-compartment") are a different namespace, normalised by the tool.
    d = _decide("fit_compartmental", {"models": "1-compartment"})
    assert d.allow is True


# ── through the goal loop ─────────────────────────────────────────────────────

class _Scripted:
    def __init__(self, plans):
        self.plans = list(plans)
        self.calls = 0

    def plan(self, goal, state_summary, observations, history, tools):
        self.calls += 1
        return self.plans.pop(0) if len(self.plans) > 1 else self.plans[0]


def test_a_bare_string_models_proposal_never_reaches_a_gate_and_the_planner_can_correct_it(monkeypatch):
    monkeypatch.setitem(WORKFLOWS, "goal_shapes", {
        "name": "goal_shapes", "kind": "goal", "goal": "shape test",
        "allowed_tools": ["load_dataset", "fit_pk_model"]})
    llm = _Scripted([
        {"tool": "load_dataset", "args": {}, "rationale": "r"},
        {"tool": "fit_pk_model", "args": {"compare": True, "models": VALID}, "rationale": "string"},
        {"tool": "fit_pk_model", "args": {"compare": True, "models": [VALID]}, "rationale": "list"},
    ])
    orch = Orchestrator(llm=llm, clock=lambda c=itertools.count(): f"t{next(c)}",
                        store=SessionStore(":memory:"))
    sid = orch.create_session().id
    out = orch.start_workflow(sid, "goal_shapes", {"path": SAMPLE})

    assert out["status"] == "awaiting_review"
    assert out["state"]["pending_tool"]["args"] == {"compare": True, "models": [VALID]}
    assert out["state"]["pk_model_results"] is None            # nothing was compared
    hist = out["state"]["planner_history"]
    assert [h["allow"] for h in hist] == [True, False, True]
    assert "must be" in hist[1]["reason"] and "list" in hist[1]["reason"]
    assert out["audit_ok"]
