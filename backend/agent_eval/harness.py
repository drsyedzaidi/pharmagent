"""Drive one scenario through the REAL Orchestrator with a given policy and record the trajectory."""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any

from app.core.orchestrator import Orchestrator
from app.core.store import SessionStore

from .recording import Call, RecordingRegistry
from .scenario import Scenario


@dataclass
class TurnRecord:
    message: str
    agent: str = ""
    routed_by: str = ""
    calls: list[Call] = field(default_factory=list)          # every registry.execute() this turn (ok or refused)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)   # the platform's own per-turn summary
    pending: dict[str, Any] | None = None
    messages: list[str] = field(default_factory=list)


@dataclass
class Trajectory:
    scenario_id: str
    policy: str
    turns: list[TurnRecord] = field(default_factory=list)
    audit_ok: bool = True
    crash: str = ""
    # tool -> numeric leaves of the reference run
    reference: dict[str, dict[str, float]] = field(default_factory=dict)


def numeric_leaves(obj: Any, prefix: str = "") -> dict[str, float]:
    """Flatten a tool result to {path: float}. Strings (ids, paths) and booleans are ignored on purpose."""
    out: dict[str, float] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(numeric_leaves(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out.update(numeric_leaves(v, f"{prefix}[{i}]"))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool) and math.isfinite(float(obj)):
        out[prefix] = float(obj)
    return out


def _orchestrator(policy) -> tuple[Orchestrator, RecordingRegistry]:
    orch = Orchestrator(llm=policy, store=SessionStore(":memory:"), clock=lambda c=itertools.count(): f"t{next(c):06d}")
    rec = RecordingRegistry(orch.registry)
    orch.registry = rec
    return orch, rec


def reference_results(scenario: Scenario) -> dict[str, dict[str, float]]:
    """Ground truth = the validated tools run directly (no model involved) on the same dataset, default arguments
    unless the scenario pins them. This is the bench's own philosophy: the tool IS the oracle."""
    from .policies import mock_policy
    out: dict[str, dict[str, float]] = {}
    if not scenario.reference_tools:
        return out
    orch, _ = _orchestrator(mock_policy())
    sid = orch.create_session().id
    if scenario.dataset:
        orch.run_tool(sid, "load_dataset", "data_manager", {"path": scenario.dataset})
    for tool in scenario.reference_tools:
        agent = orch.registry.get(tool).agent
        res = orch.run_tool(sid, tool, agent, dict(scenario.reference_args.get(tool, {})))
        out[tool] = numeric_leaves(res["result"])
    return out


def run_scenario(scenario: Scenario, policy, name: str = "policy") -> Trajectory:
    traj = Trajectory(scenario.scenario_id, name)
    traj.reference = reference_results(scenario)
    orch, rec = _orchestrator(policy)
    sid = orch.create_session().id
    if scenario.dataset:                                     # a precondition, not graded, run without the policy
        orch.run_tool(sid, "load_dataset", "data_manager", {"path": scenario.dataset})
    for i, spec in enumerate(scenario.turns):
        if hasattr(policy, "begin_turn"):
            policy.begin_turn(i)
        mark = rec.mark()
        turn = TurnRecord(spec.message)
        try:
            out = orch.chat(sid, spec.message)
        except Exception as exc:                             # a crash is a finding, not a harness failure
            traj.crash = f"{type(exc).__name__}: {exc}"
            turn.calls = rec.since(mark)
            traj.turns.append(turn)
            break
        turn.agent, turn.routed_by = out["agent"], out["routed_by"]
        turn.calls, turn.tool_calls = rec.since(mark), out["tool_calls"]
        turn.pending, turn.messages = out["pending_tool"], list(out["messages"])
        traj.turns.append(turn)
    try:
        traj.audit_ok = bool(orch.get_session(sid).audit.verify())
    except Exception as exc:
        traj.audit_ok = False
        traj.crash = traj.crash or f"audit verify raised {type(exc).__name__}"
    return traj
