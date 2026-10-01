"""The exposure-response bootstrap through the chat -> human approval -> job path.

Same guardrail as run_simest / run_bootstrap (tests/test_simest_chat.py): a chat
turn can only PROPOSE the expensive tool; nothing runs until the human approves,
and the approval is the ``confirm`` the tool requires. The plain fit and the dose
selection are cheap and run directly. No test here runs a real NLME fit.
"""
from __future__ import annotations

import itertools
import json

import pytest

from app.agents.base import Agent
from app.core.audit import AuditChain
from app.core.llm import MockLLM
from app.core.orchestrator import Orchestrator
from app.core.store import SessionStore
from app.tools.builtins import default_registry
from tests.test_er_tools import LOGIT_ARGS, TOX_ARGS, _frame, _state

# Two E-R phrases: 'bootstrap' against a single E-R phrase is a tie for the classifier.
MESSAGE = "bootstrap the exposure-response logistic model for the efficacy endpoint"


class _PickTool:
    def __init__(self, name: str, args: dict | None = None) -> None:
        self.name, self.args, self._served = name, args or {}, False

    def classify(self, message, options, descriptions):
        return options[0] if options else "data_manager"

    def select_tool(self, agent, message, tools, state_summary):
        if self._served:
            return None
        self._served = True
        return {"name": self.name, "input": dict(self.args)}


def _orch(llm=None) -> Orchestrator:
    return Orchestrator(llm=llm or MockLLM(), clock=lambda c=itertools.count(): f"t{next(c)}",
                        store=SessionStore(":memory:"))


def _session_with_fits(orch: Orchestrator) -> str:
    sid = orch.create_session().id
    sess = orch.get_session(sid)
    sess.ctx.dataset_store["d1"] = _frame()
    sess.state = sess.state.model_copy(update={"dataset_id": "d1"})
    for args in (LOGIT_ARGS, TOX_ARGS):
        orch.run_tool(sid, "fit_exposure_response", "er_dose", args, actor="alice")
    return sid


def test_agent_loop_turns_the_llms_bootstrap_call_into_a_proposal_without_its_confirm():
    state, ctx = _state()
    res = Agent(name="er_dose", system_prompt="").run_turn(
        state=state, message="bootstrap it", llm=_PickTool("bootstrap_exposure_response",
                                                           {"label": "efficacy", "confirm": True}),
        registry=default_registry(), ctx=ctx, audit=AuditChain(), clock=lambda: "t0", actor="t")
    assert res.proposal == {"tool": "bootstrap_exposure_response", "agent": "er_dose",
                            "args": {"label": "efficacy"}}               # the LLM's confirm is dropped
    assert res.state.er_results is None and res.state.pending_tool is None


def test_chat_proposes_then_the_human_approves_and_the_job_path_runs_it():
    orch = _orch(llm=_PickTool("bootstrap_exposure_response", {"label": "efficacy", "n_boot": 40, "seed": 9}))
    sid = _session_with_fits(orch)
    out = orch.chat(sid, MESSAGE, actor="alice")

    assert out["agent"] == "er_dose"
    pending = out["pending_tool"]
    assert pending["tool"] == "bootstrap_exposure_response" and pending["agent"] == "er_dose"
    assert "bootstrap" not in out["state"]["er_results"]["fits"]["efficacy"]   # nothing computed yet
    assert any("approve" in m.lower() for m in out["messages"])

    call = orch.take_pending_tool(sid, actor="alice", reason="ok")
    assert call["args"]["confirm"] is True and call["agent"] == "er_dose"
    done = orch.run_tool(sid, call["tool"], call["agent"], call["args"], actor="alice")
    boot = done["state"]["er_results"]["fits"]["efficacy"]["bootstrap"]
    assert boot["status"] == "ok" and boot["seed"] == 9 and done["audit_ok"] is True
    entries = orch.get_session(sid).audit.to_list()
    assert [e["action"] for e in entries if e["tool"] == "bootstrap_exposure_response"][:3] == [
        "propose_tool", "approve_tool", "bootstrap_exposure_response(efficacy)"]


def test_a_rejected_proposal_computes_nothing():
    orch = _orch(llm=_PickTool("bootstrap_exposure_response", {"label": "efficacy"}))
    sid = _session_with_fits(orch)
    orch.chat(sid, MESSAGE, actor="alice")
    out = orch.reject_pending_tool(sid, actor="alice", reason="not now")
    assert out["status"] == "rejected" and out["audit_ok"] is True
    assert "bootstrap" not in out["state"]["er_results"]["fits"]["efficacy"]
    with pytest.raises(KeyError):
        orch.take_pending_tool(sid)                                          # nothing left to approve twice


def test_run_tool_persists_audits_and_keeps_the_state_serialisable():
    orch = _orch()
    sid = _session_with_fits(orch)
    out = orch.run_tool(sid, "select_optimal_dose", "er_dose",
                        {"doses": [25, 50, 100, 200], "utility_weight": 1.0, "n_draws": 200, "seed": 3,
                         "exposure_mapping": {"type": "linear", "reference_dose": 100, "reference_exposure": 200}},
                        actor="alice")
    assert out["state"]["dose_selection_results"]["status"] == "ok" and out["audit_ok"] is True
    json.dumps(out["state"])
    assert out["state"]["er_history"][-1]["kind"] == "dose_selection"
