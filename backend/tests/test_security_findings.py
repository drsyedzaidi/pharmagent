"""Regression tests for the branch's security/integrity review findings.

Companion to test_identity_audit.py (which covers the versioned audit hash).

  F2  a long-running fit (NLME/SCM/engine comparison) must NOT execute
      synchronously from a chat turn — it is routed to its bounded job endpoint.
"""
from __future__ import annotations

import itertools
from pathlib import Path

SAMPLE = str(Path(__file__).parent.parent / "sample_data" / "oral_pk.csv")


# ── F2 — chat must not run expensive fits synchronously ───────────────────────


class _PickTool:
    """A stub LLM that selects `tool_name` once, then stops."""

    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name
        self._served = False

    def classify(self, message, options, descriptions):   # pragma: no cover - unused
        return options[0] if options else "modeler"

    def select_tool(self, agent, message, tools, state_summary):
        if self._served:
            return None
        self._served = True
        return {"name": self.tool_name, "input": {}}


def _modeler_turn(tool_name: str):
    from app.agents.base import Agent
    from app.core.audit import AuditChain
    from app.core.pharmstate import PharmState
    from app.tools.base import ToolContext
    from app.tools.builtins import default_registry

    agent = Agent(name="modeler", system_prompt="")
    return agent.run_turn(
        state=PharmState(), message="fit it", llm=_PickTool(tool_name),
        registry=default_registry(), ctx=ToolContext(), audit=AuditChain(),
        clock=lambda: "t0", actor="tester")


def test_expensive_tools_are_flagged():
    """The three long-running fits carry expensive=True; a cheap tool does not."""
    from app.tools.builtins import default_registry
    reg = default_registry()
    expensive = {n for n in reg.names() if reg.get(n).expensive}
    assert {"run_nlme", "run_scm", "run_engine_comparison"} <= expensive
    assert not reg.get("fit_pk_model").expensive


def test_registry_is_the_enforcement_point():
    """Enforcement lives at registry.execute (the single choke point), so it cannot
    be bypassed by naming a tool the calling agent does not own. Admission-
    controlled callers opt in with allow_expensive=True."""
    import pytest

    from app.core.audit import AuditChain
    from app.core.pharmstate import PharmState
    from app.tools.base import ExpensiveToolError, ToolContext
    from app.tools.builtins import default_registry
    reg = default_registry()
    kw = dict(state=PharmState(), ctx=ToolContext(), args={},
              audit=AuditChain(), timestamp="t0")
    with pytest.raises(ExpensiveToolError):
        reg.execute("run_nlme", **kw)                       # default: refused
    # the opt-in path is reachable (it fails later, on the missing dataset, not here)
    try:
        reg.execute("run_nlme", **kw, allow_expensive=True)
    except ExpensiveToolError:                              # pragma: no cover
        raise AssertionError("allow_expensive=True must not be refused") from None
    except Exception:
        pass                                                # any other failure is fine


def test_guard_holds_for_a_tool_outside_the_agents_own_list():
    """The pre-fix guard keyed off the agent's tool list, so an agent naming a tool
    it does not own (here: nca naming run_nlme) skipped admission control."""
    from app.agents.base import Agent
    from app.core.audit import AuditChain
    from app.core.pharmstate import PharmState
    from app.tools.base import ToolContext
    from app.tools.builtins import default_registry
    reg = default_registry()
    assert "run_nlme" not in {t.name for t in reg.for_agent("nca")}
    res = Agent(name="nca", system_prompt="").run_turn(
        state=PharmState(), message="go", llm=_PickTool("run_nlme"), registry=reg,
        ctx=ToolContext(), audit=AuditChain(), clock=lambda: "t0", actor="x")
    assert any(c.get("skipped") == "expensive" for c in res.tool_calls)


def test_workflow_leg_with_a_fit_is_not_run_inline():
    """Human approval is a scientific decision, not queue admission. A leg that
    reaches a long fit must stop rather than run it on the caller's thread."""
    from app.core.llm import MockLLM
    from app.core.orchestrator import Orchestrator
    from app.core.store import SessionStore
    from app.workflows import get_workflow
    orch = Orchestrator(llm=MockLLM(),
                        clock=lambda c=itertools.count(): f"t{next(c)}",
                        store=SessionStore(":memory:"))
    # poppk_modeling reaches run_engine_comparison BEFORE its gate
    assert orch.workflow_needs_job(get_workflow("poppk_modeling"), 0) is True
    # nca_full has no expensive step at all
    assert orch.workflow_needs_job(get_workflow("nca_full"), 0) is False
    # poppk_full's fits sit AFTER the gate, so reaching that gate needs no job
    assert orch.workflow_needs_job(get_workflow("poppk_full"), 0) is False
    out = orch.start_workflow(orch.create_session(owner="a").id,
                              "poppk_modeling", {"path": SAMPLE})
    assert out["status"] == "awaiting_job"
    assert out["next_tool"] == "run_engine_comparison"
    assert out["state"]["engine_comparison_results"] is None


def test_workflow_endpoint_hands_an_expensive_leg_to_the_job_queue():
    import itertools as _it

    from fastapi.testclient import TestClient

    import app.main as main
    from app.core.llm import MockLLM
    from app.core.orchestrator import Orchestrator
    from app.core.store import SessionStore
    main.orch = Orchestrator(llm=MockLLM(),
                             clock=lambda c=_it.count(): f"t{next(c)}",
                             store=SessionStore(":memory:"))
    client = TestClient(main.app)
    sid = client.post("/api/sessions").json()["id"]
    r = client.post(f"/api/sessions/{sid}/workflow",
                    json={"name": "poppk_modeling", "params": {"path": SAMPLE}}).json()
    assert r.get("kind") == "workflow_start" and "job_id" in r
    # a workflow with no expensive leg still runs inline
    sid2 = client.post("/api/sessions").json()["id"]
    r2 = client.post(f"/api/sessions/{sid2}/workflow",
                     json={"name": "nca_full", "params": {"path": SAMPLE}}).json()
    assert r2["status"] == "awaiting_review" and "job_id" not in r2


def test_skill_replay_refuses_expensive_steps():
    """Replaying a captured skill must not re-run NLME/SCM inline on the request
    thread: the step is recorded as an error, and no fit result is written."""
    from app.core.llm import MockLLM
    from app.core.orchestrator import Orchestrator
    from app.core.skills import Skill
    from app.core.store import SessionStore
    orch = Orchestrator(llm=MockLLM(),
                        clock=lambda c=itertools.count(): f"t{next(c)}",
                        store=SessionStore(":memory:"))
    orch.skills.save(Skill(name="replay-probe", description="probe", goal="probe",
                           steps=[{"agent": "modeler", "tool": "run_nlme", "args": {}}],
                           source_session=None, owner="alice", created_at="t0",
                           version=1))
    out = orch.run_skill("replay-probe", dataset_path=SAMPLE, owner="alice")
    step = [e for e in out["executed"] if e["tool"] == "run_nlme"]
    assert step and step[0]["status"] == "error"
    assert "long-running fit" in step[0]["error"]
    assert out["state"]["nlme_results"] is None


def test_chat_refuses_expensive_nlme():
    res = _modeler_turn("run_nlme")
    assert any(c.get("skipped") == "expensive" for c in res.tool_calls)
    assert res.state.nlme_results is None            # nothing was computed
    assert any("dedicated control" in m for m in res.messages)


def test_chat_refuses_expensive_scm_and_engine():
    for name, field in (("run_scm", "scm_results"),
                        ("run_engine_comparison", "engine_comparison_results")):
        res = _modeler_turn(name)
        assert any(c.get("skipped") == "expensive" for c in res.tool_calls), name
        assert getattr(res.state, field) is None, name
