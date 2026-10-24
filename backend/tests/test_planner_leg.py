"""Phase A of the agentic planner: a goal-driven workflow leg.

An LLM proposes the next tool from the diagnostics in state (``LLM.plan``), the
deterministic policy (``app.core.policy``) admits, gates or rejects it, and an
admitted step still runs through ``registry.execute`` with audit. Keyless
throughout (MockLLM walks the poppk_full rails), and NO test here runs a real
population fit: the gate on ``run_nlme`` is the whole point of the leg.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest

from app.core import llm as llm_mod
from app.core import policy
from app.core.llm import MockLLM, OpenAICompatLLM
from app.core.orchestrator import Orchestrator
from app.core.pharmstate import PharmState
from app.core.store import SessionStore
from app.tools.base import Tool, ToolRegistry, ToolResult
from app.tools.builtins import default_registry
from app.workflows import WORKFLOWS, get_workflow

SAMPLE = str(Path(__file__).parent.parent / "sample_data" / "oral_pk.csv")
POPPK_FULL_ORDER = [s["tool"] for s in get_workflow("poppk_full")["steps"]]


def _orch(llm=None) -> Orchestrator:
    return Orchestrator(llm=llm or MockLLM(), clock=lambda c=itertools.count(): f"t{next(c)}",
                        store=SessionStore(":memory:"))


def _planner_actions(orch: Orchestrator, sid: str) -> list[tuple[str, str]]:
    return [(e.action, e.tool) for e in orch.get_session(sid).audit.entries if e.agent == "planner"]


class _Scripted:
    """A planner that returns the given proposals in order, then the last one forever.
    classify/select_tool are never reached by a goal leg."""

    def __init__(self, plans):
        self.plans = list(plans)
        self.calls = 0

    def plan(self, goal, state_summary, observations, history, tools):
        self.calls += 1
        if len(self.plans) > 1:
            return self.plans.pop(0)
        return self.plans[0]


def _goal_template(name: str, allowed: list[str], **extra) -> dict:
    return {"name": name, "kind": "goal", "goal": f"test goal {name}",
            "allowed_tools": allowed, **extra}


# ── template integrity ────────────────────────────────────────────────────────

def test_poppk_goal_is_a_goal_template_over_the_poppk_full_tool_set():
    wf = get_workflow("poppk_goal")
    reg = default_registry()
    assert wf["kind"] == "goal" and wf["name"] == "poppk_goal"
    assert wf["goal"].strip()
    assert "steps" not in wf
    for name in wf["allowed_tools"]:
        reg.get(name)                      # KeyError if a tool is not registered
    assert set(wf["allowed_tools"]) == set(POPPK_FULL_ORDER)
    budget = policy.budget_from(wf)
    assert budget.max_steps <= policy.DEFAULT_MAX_STEPS
    assert budget.max_gated <= policy.DEFAULT_MAX_GATED


def test_get_workflow_is_unchanged_for_step_templates():
    for name in ("nca_full", "poppk_full", "poppk_modeling"):
        wf = get_workflow(name)
        assert wf.get("kind") is None and wf["steps"]


def test_a_template_cannot_raise_the_budget_caps():
    wf = _goal_template("g", [], budget={"max_steps": 999, "max_gated": 99})
    assert policy.budget_from(wf) == policy.Budget(policy.DEFAULT_MAX_STEPS, policy.DEFAULT_MAX_GATED)
    assert policy.budget_from({"name": "g"}) == policy.Budget()


# ── policy: pure decisions ────────────────────────────────────────────────────

def _roster(*names: str) -> dict:
    reg = default_registry()
    return {n: reg.get(n) for n in names}


def test_policy_rejects_out_of_order_tools():
    roster = _roster(*POPPK_FULL_ORDER)
    empty = PharmState()
    d = policy.decide("run_nlme", {}, empty, roster=roster, history=[], budget=policy.Budget())
    assert not d.allow and d.mode == "rejected" and "pk_model_results" in d.reason and not d.stop
    for tool in ("run_scm", "run_diagnostics", "run_covariate_forest", "run_vpc"):
        d = policy.decide(tool, {}, PharmState(pk_model_results={"status": "ok"}),
                          roster=roster, history=[], budget=policy.Budget())
        assert not d.allow and "nlme_results" in d.reason, tool
    # A failed fit does not satisfy the prerequisite either.
    d = policy.decide("run_nlme", {}, PharmState(pk_model_results={"status": "error"}),
                      roster=roster, history=[], budget=policy.Budget())
    assert not d.allow
    # Once the prerequisite is in state the same call is admitted (gated: expensive).
    d = policy.decide("run_nlme", {}, PharmState(pk_model_results={"status": "ok"}),
                      roster=roster, history=[], budget=policy.Budget())
    assert d.allow and d.mode == "gated"


def test_policy_always_allows_review_and_report_prerequisite_wise():
    roster = _roster("adversarial_review", "generate_report")
    d = policy.decide("adversarial_review", {}, PharmState(), roster=roster, history=[], budget=policy.Budget())
    assert d.allow and d.mode == "autonomous"
    d = policy.decide("generate_report", {}, PharmState(), roster=roster, history=[], budget=policy.Budget())
    assert d.allow and d.mode == "gated"


def test_policy_rejects_an_identical_repeat_and_stops_the_loop():
    roster = _roster("validate_cdisc")
    history = [{"step": 0, "tool": "validate_cdisc", "args": {}, "allow": True, "mode": "autonomous"}]
    d = policy.decide("validate_cdisc", {}, PharmState(), roster=roster, history=history, budget=policy.Budget())
    assert not d.allow and d.stop and "identical" in d.reason
    # Re-proposing a call the policy just REJECTED is the stuck-planner case: it stops
    # too, before the roster / prerequisite refusal can be repeated 20 times.
    rejected = [{"step": 0, "tool": "run_nlme", "args": {}, "allow": False, "mode": "rejected"}]
    d = policy.decide("run_nlme", {}, PharmState(), roster=_roster("run_nlme"), history=rejected,
                      budget=policy.Budget())
    assert not d.allow and d.stop and "identical" in d.reason
    # Different arguments are not a repeat — the same rule Agent.run_turn applies.
    assert policy.canonical_call("validate_cdisc", {}) != policy.canonical_call("validate_cdisc", {"strict": True})
    d = policy.decide("validate_cdisc", {"strict": True}, PharmState(), roster=roster, history=history,
                      budget=policy.Budget())
    assert d.allow
    # Only the LAST proposal counts, exactly like run_turn's last_call.
    two_back = history + [{"step": 1, "tool": "load_dataset", "args": {}, "allow": True, "mode": "autonomous"}]
    d = policy.decide("validate_cdisc", {}, PharmState(), roster=roster, history=two_back, budget=policy.Budget())
    assert d.allow


def test_policy_rejects_tools_off_the_allow_list():
    d = policy.decide("compute_nca", {}, PharmState(), roster=_roster("load_dataset"), history=[],
                      budget=policy.Budget())
    assert not d.allow and "allowed_tools" in d.reason and not d.stop


def test_policy_classification_table():
    reg = default_registry()
    cases = [
        ("profile_pk_dataset", {}, "autonomous"),
        ("generate_spaghetti_plot", {}, "autonomous"),
        ("run_diagnostics", {"cwres_interaction": False}, "autonomous"),
        ("run_covariate_forest", {"source": "scm"}, "autonomous"),
        ("run_vpc", {}, "autonomous"),
        ("adversarial_review", {}, "autonomous"),
        ("fit_pk_model", {"compare": True}, "autonomous"),
        ("fit_pk_model", {"model_key": "onecmt_oral"}, "gated"),
        ("fit_pk_model", {"models": ["onecmt_oral"]}, "gated"),
        ("run_nlme", {}, "gated"),                      # expensive, always
        ("run_scm", {}, "gated"),                       # expensive, always
        ("run_engine_comparison", {}, "gated"),         # expensive, always
        ("generate_report", {}, "gated"),               # the deliverable
        ("compute_nca", {}, "gated"),                   # not on the autonomous list: fail closed
    ]
    for name, args, expected in cases:
        assert policy.classify(name, args, reg.get(name)) == expected, (name, args)
    assert policy.classify("no_such_tool", {}, None) == "gated"


def test_policy_enforces_the_gate_budget():
    roster = _roster("generate_report", "validate_cdisc")
    gates = [{"step": i, "tool": "run_nlme", "args": {"model_key": f"m{i}"}, "allow": True, "mode": "gated"}
             for i in range(2)]
    tight = policy.Budget(max_steps=20, max_gated=2)
    d = policy.decide("generate_report", {}, PharmState(), roster=roster, history=gates, budget=tight)
    assert not d.allow and d.stop and "max_gated" in d.reason
    # An autonomous tool is still admitted once the gates are spent.
    d = policy.decide("validate_cdisc", {}, PharmState(), roster=roster, history=gates, budget=tight)
    assert d.allow and d.mode == "autonomous"


def test_policy_enforces_the_step_budget():
    history = [{"step": i, "tool": f"t{i}", "args": {}, "allow": False, "mode": "rejected"} for i in range(3)]
    tight = policy.Budget(max_steps=3, max_gated=6)
    assert policy.budget_exhausted(history, tight) and "max_steps" in policy.budget_exhausted(history, tight)
    assert policy.budget_exhausted(history[:2], tight) is None
    d = policy.decide("validate_cdisc", {}, PharmState(), roster=_roster("validate_cdisc"), history=history,
                      budget=tight)
    assert not d.allow and d.stop


# ── MockLLM.plan degrades to the poppk_full rails ────────────────────────────

def test_mock_plan_reproduces_poppk_full_step_order():
    mock = MockLLM()
    tools = list(_roster(*POPPK_FULL_ORDER).values())
    history: list[dict] = []
    proposed: list[str] = []
    for _ in range(len(POPPK_FULL_ORDER) + 1):
        plan = mock.plan("goal", {}, [], history, tools)
        if plan["tool"] is None:
            break
        proposed.append(plan["tool"])
        history.append({"step": len(history), "tool": plan["tool"], "args": plan["args"],
                        "allow": True, "mode": "autonomous"})
    assert proposed == POPPK_FULL_ORDER
    # The template's own arguments travel with the proposal (no method is pinned).
    fit = next(h for h in history if h["tool"] == "fit_pk_model")
    assert fit["args"] == {"compare": True}
    nlme = next(h for h in history if h["tool"] == "run_nlme")
    assert "method" not in nlme["args"]
    # Exhausted: it says so instead of silently returning nothing.
    assert plan["tool"] is None and plan["stop_reason"] and plan["rationale"]


def test_mock_plan_skips_steps_already_marked_in_state_and_restricts_to_the_roster():
    mock = MockLLM()
    tools = list(_roster("load_dataset", "profile_pk_dataset", "generate_report").values())
    plan = mock.plan("g", {"dataset_id": "ds_1", "data_quality": []}, [], [], tools)
    assert plan["tool"] == "generate_report"       # load/profile done; validate etc. not offered
    plan = mock.plan("g", {}, [], [], tools)
    assert plan["tool"] == "load_dataset"


# ── the goal leg through the orchestrator ────────────────────────────────────

def test_poppk_goal_reaches_the_structural_gate_without_running_nlme():
    orch = _orch()
    sid = orch.create_session().id
    out = orch.start_workflow(sid, "poppk_goal", {"path": SAMPLE})

    assert out["status"] == "awaiting_review"
    assert [e["tool"] for e in out["executed"]] == [
        "load_dataset", "profile_pk_dataset", "validate_cdisc", "generate_spaghetti_plot", "fit_pk_model"]
    # Same record shape a step template produces (the UI renders one message per entry).
    for i, e in enumerate(out["executed"]):
        assert set(e) == {"step", "label", "tool", "agent", "summary"}
        assert e["step"] == i and e["agent"] == default_registry().get(e["tool"]).agent
        assert isinstance(e["summary"], str) and e["summary"]
    # The gated proposal sits exactly where a chat-proposed expensive tool goes.
    pending = out["state"]["pending_tool"]
    assert pending["tool"] == "run_nlme" and pending["agent"] == "modeler"
    assert pending["proposed_by"] == "planner"
    assert pending["rationale"] and pending["workflow"] == "poppk_goal"
    assert out["proposal"] == pending
    assert out["review"]["after_step"] == 4 and "run_nlme" in out["review"]["label"]
    # The expensive leg is untouched.
    assert out["state"]["nlme_results"] is None and out["state"]["scm_results"] is None
    assert out["audit_ok"]
    # Every proposal and decision is on the chain, as the planner.
    actions = _planner_actions(orch, sid)
    assert actions[0] == ("goal_start", "plan")
    assert actions.count(("plan", "fit_pk_model")) == 1
    assert ("policy_autonomous", "fit_pk_model") in actions
    assert actions[-2:] == [("policy_gated", "run_nlme"), ("propose_tool", "run_nlme")]
    # And the readable history the planner reasons over.
    hist = out["state"]["planner_history"]
    assert [h["tool"] for h in hist] == [*POPPK_FULL_ORDER[:6]]
    assert hist[-1]["mode"] == "gated" and hist[-1]["allow"] is True


def test_inline_approval_of_an_expensive_proposal_defers_to_the_job_queue():
    """allow_expensive is NOT weakened: a human approval is a scientific decision,
    not queue admission. Without the JobManager flag nothing is consumed, so the
    same approval is re-issued as a background job."""
    orch = _orch()
    sid = orch.create_session().id
    orch.start_workflow(sid, "poppk_goal", {"path": SAMPLE})
    n_audit = len(orch.get_session(sid).audit.entries)

    out = orch.resume_workflow(sid, approve=True, actor="alice", reason="model confirmed")
    assert out["status"] == "awaiting_job" and out["next_tool"] == "run_nlme"
    assert out["executed"] == []
    assert out["state"]["pending_tool"]["tool"] == "run_nlme"         # proposal kept
    assert orch.get_session(sid).pending_review is not None            # review kept
    assert out["state"]["nlme_results"] is None
    # Nothing was signed or spent: the chain did not grow.
    assert len(orch.get_session(sid).audit.entries) == n_audit


def test_rejecting_a_planner_proposal_signs_the_decision_and_clears_it():
    orch = _orch()
    sid = orch.create_session().id
    orch.start_workflow(sid, "poppk_goal", {"path": SAMPLE})

    out = orch.resume_workflow(sid, approve=False, actor="alice", reason="wrong structural model")
    assert out["status"] == "rejected"
    assert out["state"]["pending_tool"] is None
    assert out["state"]["nlme_results"] is None
    assert orch.get_session(sid).pending_review is None
    tail = [(e.agent, e.tool, e.action, e.actor, e.reason) for e in orch.get_session(sid).audit.entries[-2:]]
    # The existing signed human_review entry (format unchanged) then the proposal's reject_tool.
    assert tail[0] == ("qc", "human_review", "rejected after step 4", "alice", "wrong structural model")
    assert tail[1] == ("supervisor", "run_nlme", "reject_tool", "alice", "wrong structural model")
    assert out["audit_ok"]
    with pytest.raises(ValueError):
        orch.resume_workflow(sid, approve=True)


def test_a_proposal_decided_through_the_chat_path_still_closes_the_goal_gate_cleanly():
    """/chat/pending_tool approves or rejects the same state.pending_tool. Once it has
    been decided there, a workflow rejection just closes the review, and a workflow
    approval resumes planning from the state as it is — nothing is taken twice, and
    the leg still terminates (here: the mock re-proposes run_scm, whose prerequisite
    is unmet, and the repeat guard stops it) without running a fit."""
    orch = _orch()
    sid = orch.create_session().id
    orch.start_workflow(sid, "poppk_goal", {"path": SAMPLE})
    orch.reject_pending_tool(sid, actor="alice", reason="decided in chat")
    assert orch.get_session(sid).pending_review is not None
    out = orch.resume_workflow(sid, approve=True, actor="alice", reason="carry on")
    assert out["status"] == "stopped" and "identical" in out["stop_reason"]
    assert out["executed"] == [] and out["state"]["nlme_results"] is None
    assert out["state"]["scm_results"] is None and orch.get_session(sid).pending_review is None
    kinds = [(e.agent, e.tool, e.action) for e in orch.get_session(sid).audit.entries]
    assert ("qc", "human_review", "approved after step 4") in kinds
    assert kinds.count(("supervisor", "run_nlme", "approve_tool")) == 0     # nothing to take
    # Rejection on a stale review: closes it, signs it, computes nothing.
    orch.start_workflow(sid, "poppk_goal", {"path": SAMPLE})
    orch.reject_pending_tool(sid, actor="alice", reason="decided in chat")
    out = orch.resume_workflow(sid, approve=False, actor="alice", reason="close the gate")
    assert out["status"] == "rejected" and orch.get_session(sid).pending_review is None
    assert out["state"]["nlme_results"] is None and out["audit_ok"]


def test_a_chat_proposal_is_never_mistaken_for_the_planner_s(monkeypatch):
    """A chat-proposed expensive tool (proposed_by = actor id, no workflow) parked in
    state.pending_tool must not be taken by the goal resume, even for an actor whose
    id happens to be "planner"."""
    monkeypatch.setitem(WORKFLOWS, "goal_one", _goal_template("goal_one", ["load_dataset"]))
    llm = _Scripted([{"tool": "load_dataset", "args": {}, "rationale": "r"},
                     {"tool": None, "args": {}, "rationale": "done", "stop_reason": "goal met"}])
    orch = _orch(llm)
    sid = orch.create_session().id
    sess = orch.get_session(sid)
    from app.core.pharmstate import apply_writes
    chat_proposal = {"tool": "run_simest", "agent": "simulator", "args": {}, "proposed_at": "t",
                     "proposed_by": "planner", "message": "chat"}      # no "workflow" key
    sess.state = apply_writes(sess.state, "supervisor", {"pending_tool": chat_proposal})
    out = orch.start_workflow(sid, "goal_one", {"path": SAMPLE})
    assert out["status"] == "complete"
    assert out["state"]["pending_tool"] == chat_proposal                 # untouched
    assert orch._planner_pending(orch.get_session(sid)) is None


def test_approval_resumes_the_goal_loop_through_take_pending_tool(monkeypatch):
    """Exercised on a goal template whose only gate is the (cheap) report, so the
    branch under test is the approve → execute → keep planning handoff, not NLME."""
    monkeypatch.setitem(WORKFLOWS, "goal_report",
                        _goal_template("goal_report", ["load_dataset", "profile_pk_dataset", "generate_report"]))
    orch = _orch()
    sid = orch.create_session().id
    out = orch.start_workflow(sid, "goal_report", {"path": SAMPLE})
    assert out["status"] == "awaiting_review"
    assert [e["tool"] for e in out["executed"]] == ["load_dataset", "profile_pk_dataset"]
    assert out["proposal"]["tool"] == "generate_report" and out["proposal"]["proposed_by"] == "planner"

    res = orch.resume_workflow(sid, approve=True, actor="alice", reason="report wanted")
    assert res["status"] == "complete"
    assert res["stop_reason"]                                   # the planner said why it stopped
    assert [e["tool"] for e in res["executed"]] == ["generate_report"]
    assert res["executed"][0]["step"] == 2 and res["executed"][0]["agent"] == "report"
    assert res["state"]["report_path"]
    assert res["state"]["pending_tool"] is None
    assert orch.get_session(sid).pending_review is None
    assert res["audit_ok"]
    # The approval went through take_pending_tool: its confirm=True is the human's yes.
    entries = orch.get_session(sid).audit.entries
    kinds = [(e.agent, e.tool, e.action) for e in entries]
    i_review = kinds.index(("qc", "human_review", "approved after step 1"))
    i_approve = kinds.index(("supervisor", "generate_report", "approve_tool"))
    i_ran = next(i for i, k in enumerate(kinds) if k[:2] == ("report", "generate_report"))
    assert i_review < i_approve < i_ran
    assert entries[i_approve].actor == "alice" and entries[i_approve].reason == "report wanted"
    assert res["state"]["planner_history"][-1]["mode"] == "gated"


def test_goal_loop_stops_at_the_step_budget_with_a_stop_reason(monkeypatch):
    monkeypatch.setitem(WORKFLOWS, "goal_tight", _goal_template(
        "goal_tight", ["load_dataset", "validate_cdisc", "generate_spaghetti_plot"],
        budget={"max_steps": 4, "max_gated": 6}))
    # Alternating cheap tools: never an identical repeat, so only the budget can stop it.
    llm = _Scripted([{"tool": "load_dataset", "args": {}, "rationale": "r"},
                     {"tool": "validate_cdisc", "args": {}, "rationale": "r"},
                     {"tool": "generate_spaghetti_plot", "args": {}, "rationale": "r"},
                     {"tool": "validate_cdisc", "args": {}, "rationale": "r"},
                     {"tool": "generate_spaghetti_plot", "args": {}, "rationale": "r"}])
    orch = _orch(llm)
    sid = orch.create_session().id
    out = orch.start_workflow(sid, "goal_tight", {"path": SAMPLE})
    assert out["status"] == "stopped"
    assert "max_steps 4" in out["stop_reason"]
    assert len(out["executed"]) == 4 and llm.calls == 4
    assert ("stop", "plan") in _planner_actions(orch, sid)
    assert out["audit_ok"]


def test_goal_loop_stops_when_the_planner_repeats_itself(monkeypatch):
    monkeypatch.setitem(WORKFLOWS, "goal_repeat", _goal_template("goal_repeat", ["load_dataset", "validate_cdisc"]))
    llm = _Scripted([{"tool": "load_dataset", "args": {}, "rationale": "r"},
                     {"tool": "validate_cdisc", "args": {}, "rationale": "r"}])  # then validate_cdisc forever
    orch = _orch(llm)
    sid = orch.create_session().id
    out = orch.start_workflow(sid, "goal_repeat", {"path": SAMPLE})
    assert out["status"] == "stopped" and "identical" in out["stop_reason"]
    assert [e["tool"] for e in out["executed"]] == ["load_dataset", "validate_cdisc"]
    assert ("policy_rejected", "validate_cdisc") in _planner_actions(orch, sid)


def test_goal_loop_rejects_out_of_order_proposals_and_lets_the_planner_recover(monkeypatch):
    monkeypatch.setitem(WORKFLOWS, "goal_order", _goal_template("goal_order", ["load_dataset", "run_vpc"]))
    llm = _Scripted([{"tool": "run_vpc", "args": {}, "rationale": "too early"},
                     {"tool": "load_dataset", "args": {}, "rationale": "r"},
                     {"tool": None, "args": {}, "rationale": "done", "stop_reason": "goal met"}])
    orch = _orch(llm)
    sid = orch.create_session().id
    out = orch.start_workflow(sid, "goal_order", {"path": SAMPLE})
    assert out["status"] == "complete" and out["stop_reason"] == "goal met"
    assert [e["tool"] for e in out["executed"]] == ["load_dataset"]
    hist = out["state"]["planner_history"]
    assert hist[0]["tool"] == "run_vpc" and hist[0]["allow"] is False and "nlme_results" in hist[0]["reason"]
    assert out["state"]["vpc_results"] is None


def test_goal_loop_stops_on_a_tool_error_instead_of_retrying(monkeypatch):
    monkeypatch.setitem(WORKFLOWS, "goal_err", _goal_template("goal_err", ["profile_pk_dataset"]))
    llm = _Scripted([{"tool": "profile_pk_dataset", "args": {}, "rationale": "r"}])
    orch = _orch(llm)
    sid = orch.create_session().id
    out = orch.start_workflow(sid, "goal_err", {})          # nothing loaded: the tool refuses
    assert out["status"] == "stopped" and out["stop_reason"].startswith("profile_pk_dataset failed")
    assert out["executed"] == [] and llm.calls == 1
    assert ("tool_error", "profile_pk_dataset") in _planner_actions(orch, sid)
    assert out["audit_ok"]


def test_planner_never_runs_a_tool_outside_the_allow_list(monkeypatch):
    monkeypatch.setitem(WORKFLOWS, "goal_list", _goal_template("goal_list", ["load_dataset"]))
    llm = _Scripted([{"tool": "compute_nca", "args": {}, "rationale": "r"},
                     {"tool": None, "args": {}, "rationale": "no admissible step", "stop_reason": "nothing admissible"}])
    orch = _orch(llm)
    sid = orch.create_session().id
    out = orch.start_workflow(sid, "goal_list", {"path": SAMPLE})
    assert out["status"] == "complete" and out["executed"] == []
    assert out["state"]["nca_parameters"] is None
    assert out["state"]["planner_history"][0]["allow"] is False


def test_workflow_needs_job_for_a_goal_template():
    orch = _orch()
    wf = get_workflow("poppk_goal")
    assert orch.workflow_needs_job(wf, 0) is False       # start leg: the policy gates every fit
    assert orch.workflow_needs_job(wf, 5) is True        # resume may execute the approved fit
    cheap = _goal_template("cheap", ["load_dataset", "generate_report"])
    assert orch.workflow_needs_job(cheap, 3) is False


# ── hosted planners parse tool-use into the same proposal shape ──────────────

class _Resp:
    def __init__(self, payload: dict) -> None:
        self._payload, self.status_code = payload, 200

    def json(self) -> dict:
        return self._payload


def _chat(content, tool_calls=None) -> dict:
    msg: dict = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {"choices": [{"message": msg}]}


def test_openai_compat_plan_returns_tool_args_and_rationale(monkeypatch):
    sent: list[dict] = []
    replies = [_chat("Shrinkage is high; check the VPC.",
                     [{"id": "c1", "type": "function",
                       "function": {"name": "run_vpc", "arguments": json.dumps({})}}]),
               _chat("Goal met: review passed and report delivered.")]

    def post(url, json=None, headers=None, timeout=None):  # noqa: A002 - mirrors requests.post
        sent.append(json)
        return _Resp(replies.pop(0))

    monkeypatch.setattr(llm_mod.requests, "post", post)
    llm = OpenAICompatLLM(base_url="http://127.0.0.1:11434/v1", model="qwen3", api_key=None)
    tools = list(_roster("run_vpc", "generate_report").values())
    plan = llm.plan("goal", {"nlme_results": "present"}, [], [], tools)
    assert plan == {"tool": "run_vpc", "args": {}, "rationale": "Shrinkage is high; check the VPC.",
                    "stop_reason": None}
    assert [t["function"]["name"] for t in sent[0]["tools"]] == ["run_vpc", "generate_report"]
    assert "goal" in sent[0]["messages"][0]["content"]
    stop = llm.plan("goal", {}, [], [], tools)
    assert stop["tool"] is None and stop["stop_reason"] == "Goal met: review passed and report delivered."


# ── the JobManager path (allow_expensive=True) with a FAKE expensive tool ────

def _registry_with_fake_fit(calls: list[dict]) -> ToolRegistry:
    """load/profile/validate from the real registry plus a cheap stand-in for a
    long-running fit: declared expensive=True, writes nlme_results like run_nlme,
    records every execution. No population fit is ever run."""
    reg = ToolRegistry()
    base = default_registry()
    for name in ("load_dataset", "profile_pk_dataset", "validate_cdisc"):
        reg.register(base.get(name))

    def run(state, ctx, args):
        calls.append(dict(args))
        return ToolResult(summary="fake fit done", writes={"nlme_results": {"status": "ok"}},
                          result={"ok": True}, action="fake_fit")

    reg.register(Tool("fake_fit", "stand-in for a long-running fit", "modeler",
                      {"type": "object", "properties": {}, "required": []}, run, expensive=True))
    return reg


def _fake_orch(llm, calls: list[dict]) -> Orchestrator:
    return Orchestrator(llm=llm, registry=_registry_with_fake_fit(calls),
                        clock=lambda c=itertools.count(): f"t{next(c)}", store=SessionStore(":memory:"))


def test_job_path_still_gates_an_expensive_proposal_then_runs_it_exactly_once(monkeypatch):
    monkeypatch.setitem(WORKFLOWS, "goal_fake", _goal_template("goal_fake", ["load_dataset", "fake_fit", "validate_cdisc"]))
    calls: list[dict] = []
    llm = _Scripted([{"tool": "load_dataset", "args": {}, "rationale": "r"},
                     {"tool": "fake_fit", "args": {}, "rationale": "fit it"},
                     {"tool": "validate_cdisc", "args": {}, "rationale": "r"},
                     {"tool": None, "args": {}, "rationale": "goal met", "stop_reason": "goal met"}])
    orch = _fake_orch(llm, calls)
    sid = orch.create_session().id
    # (a) allow_expensive=True on the START leg (as main.py would pass on a job) does NOT
    #     let the planner run the fit: the policy gates it regardless of the flag.
    out = orch.start_workflow(sid, "goal_fake", {"path": SAMPLE}, allow_expensive=True)
    assert out["status"] == "awaiting_review" and out["proposal"]["tool"] == "fake_fit"
    assert calls == [] and [e["tool"] for e in out["executed"]] == ["load_dataset"]
    # (b) the job-path approval: signed, taken, run exactly once, planning continues.
    res = orch.resume_workflow(sid, approve=True, actor="alice", reason="go", allow_expensive=True)
    assert res["status"] == "complete" and res["stop_reason"] == "goal met"
    assert calls == [{"confirm": True}]                 # take_pending_tool's confirm IS the yes
    assert [e["tool"] for e in res["executed"]] == ["fake_fit", "validate_cdisc"]
    assert res["executed"][0]["label"] == "planner (approved): fake_fit"
    assert res["state"]["nlme_results"] == {"status": "ok"} and res["state"]["pending_tool"] is None
    kinds = [(e.agent, e.tool, e.action) for e in orch.get_session(sid).audit.entries]
    i_review = kinds.index(("qc", "human_review", "approved after step 0"))
    i_approve = kinds.index(("supervisor", "fake_fit", "approve_tool"))
    i_ran = kinds.index(("modeler", "fake_fit", "fake_fit"))
    assert i_review < i_approve < i_ran and kinds.count(("modeler", "fake_fit", "fake_fit")) == 1
    assert res["audit_ok"]


class _Raising(_Scripted):
    """Returns its scripted plans, then raises like a hosted planner losing its network."""

    def plan(self, goal, state_summary, observations, history, tools):
        if not self.plans:
            raise RuntimeError("LLM server error HTTP 503")
        self.calls += 1
        return self.plans.pop(0)


def test_a_planner_failure_after_an_approved_fit_keeps_the_fit(monkeypatch):
    monkeypatch.setitem(WORKFLOWS, "goal_fake2", _goal_template("goal_fake2", ["load_dataset", "fake_fit"]))
    calls: list[dict] = []
    llm = _Raising([{"tool": "load_dataset", "args": {}, "rationale": "r"},
                    {"tool": "fake_fit", "args": {}, "rationale": "fit it"}])
    orch = _fake_orch(llm, calls)
    sid = orch.create_session().id
    assert orch.start_workflow(sid, "goal_fake2", {"path": SAMPLE}, allow_expensive=True)["status"] == "awaiting_review"
    res = orch.resume_workflow(sid, approve=True, actor="alice", reason="go", allow_expensive=True)
    assert res["status"] == "stopped" and res["stop_reason"].startswith("planner failed:")
    assert calls == [{"confirm": True}]
    assert [e["tool"] for e in res["executed"]] == ["fake_fit"]
    # The fit's result and the approval survived in the store, and the failure is on the chain.
    row = orch.store.get(sid)
    assert row["state"]["nlme_results"] == {"status": "ok"} and row["state"]["current_step"] == 2
    assert ("plan_error", "plan") in _planner_actions(orch, sid)
    assert res["audit_ok"]


def test_a_stuck_planner_stops_after_one_repeat_not_the_whole_budget(monkeypatch):
    monkeypatch.setitem(WORKFLOWS, "goal_stuck", _goal_template("goal_stuck", ["load_dataset", "run_vpc"]))
    llm = _Scripted([{"tool": "load_dataset", "args": {}, "rationale": "r"},
                     {"tool": "run_vpc", "args": {}, "rationale": "too early"}])   # forever
    orch = _orch(llm)
    sid = orch.create_session().id
    out = orch.start_workflow(sid, "goal_stuck", {"path": SAMPLE})
    assert out["status"] == "stopped" and "identical" in out["stop_reason"]
    assert llm.calls == 3                       # load, run_vpc (prerequisite), run_vpc (repeat → stop)
    assert [h["allow"] for h in out["state"]["planner_history"]] == [True, False, False]


def test_a_silent_planner_stop_is_reported_as_stopped_not_complete(monkeypatch):
    monkeypatch.setitem(WORKFLOWS, "goal_silent", _goal_template("goal_silent", ["load_dataset"]))
    llm = _Scripted([{"tool": None, "args": {}, "rationale": "", "stop_reason": "planner selected no tool"}])
    orch = _orch(llm)
    sid = orch.create_session().id
    out = orch.start_workflow(sid, "goal_silent", {"path": SAMPLE})
    assert out["status"] == "stopped" and out["executed"] == []


def test_the_planner_cannot_repoint_the_analysis_dataset(monkeypatch):
    monkeypatch.setitem(WORKFLOWS, "goal_ds", _goal_template("goal_ds", ["load_dataset", "validate_cdisc"]))
    llm = _Scripted([{"tool": "load_dataset", "args": {"path": "/etc/passwd"}, "rationale": "r"},
                     {"tool": "validate_cdisc", "args": {"dataset_id": "someone_elses"}, "rationale": "r"},
                     {"tool": None, "args": {}, "rationale": "done", "stop_reason": "goal met"}])
    orch = _orch(llm)
    sid = orch.create_session().id
    out = orch.start_workflow(sid, "goal_ds", {"path": SAMPLE})
    assert out["status"] == "complete"
    assert out["state"]["dataset_path"] == SAMPLE                    # params win over the planner
    hist = out["state"]["planner_history"]
    assert hist[0]["args"] == {"path": SAMPLE} and "dataset_id" not in hist[1]["args"]


# ── the existing HTTP endpoints drive the goal leg unchanged ─────────────────

@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    import app.main as main
    main.orch = _orch()
    return TestClient(main.app)


def test_http_start_runs_the_goal_leg_inline_to_the_gate_and_reject_clears_it(client):
    sid = client.post("/api/sessions").json()["id"]
    out = client.post(f"/api/sessions/{sid}/workflow/start",
                      json={"workflow": "poppk_goal", "params": {"path": SAMPLE}}).json()
    assert out["status"] == "awaiting_review" and "job_id" not in out   # start leg never needs the queue
    assert out["proposal"]["tool"] == "run_nlme" and out["proposal"]["proposed_by"] == "planner"
    assert out["state"]["nlme_results"] is None
    # Rejection is inline and signed; the gate is cleared and no fit ran.
    rej = client.post(f"/api/sessions/{sid}/workflow/resume",
                      json={"approve": False, "reason": "wrong model"}).json()
    assert rej["status"] == "rejected" and rej["state"]["pending_tool"] is None
    assert rej["state"]["nlme_results"] is None
    assert client.get(f"/api/sessions/{sid}/audit").json()["integrity"]["chain_ok"] is True


def test_http_approval_of_an_expensive_proposal_is_always_backgrounded(client, monkeypatch):
    """The endpoint's own admission control routes the resume to the JobManager;
    the job is intercepted here so no real population fit runs."""
    import app.main as main
    submitted: list[dict] = []
    monkeypatch.setattr(main.jobs, "submit", lambda **kw: submitted.append(kw) or "job_fake")
    sid = client.post("/api/sessions").json()["id"]
    client.post(f"/api/sessions/{sid}/workflow/start",
                json={"workflow": "poppk_goal", "params": {"path": SAMPLE}})
    r = client.post(f"/api/sessions/{sid}/workflow/resume", json={"approve": True, "reason": "ok"}).json()
    assert r == {"job_id": "job_fake", "status": "running", "kind": "workflow_resume"}
    assert submitted and submitted[0]["kind"] == "workflow_resume"
    # Nothing ran on the request thread and the proposal still waits for the job.
    state = client.get(f"/api/sessions/{sid}/state").json()
    assert state["nlme_results"] is None and state["pending_tool"]["tool"] == "run_nlme"


def test_http_cheap_gate_resumes_inline_through_the_existing_endpoint(client, monkeypatch):
    monkeypatch.setitem(WORKFLOWS, "goal_report",
                        _goal_template("goal_report", ["load_dataset", "profile_pk_dataset", "generate_report"]))
    sid = client.post("/api/sessions").json()["id"]
    out = client.post(f"/api/sessions/{sid}/workflow/start",
                      json={"workflow": "goal_report", "params": {"path": SAMPLE}}).json()
    assert out["status"] == "awaiting_review" and out["proposal"]["tool"] == "generate_report"
    res = client.post(f"/api/sessions/{sid}/workflow/resume", json={"approve": True, "reason": "ok"}).json()
    assert res["status"] == "complete" and "job_id" not in res
    assert [e["tool"] for e in res["executed"]] == ["generate_report"]
    assert res["state"]["report_path"] and res["state"]["pending_tool"] is None
