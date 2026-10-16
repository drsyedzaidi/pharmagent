"""LLM policies. Each implements the platform's LLM protocol: ``classify`` and ``select_tool``.

Keyless policies (CI): ``ScriptedLLM`` follows a plan exactly (the harness self-test, scores 1.0) and the FAULTY
policies each break one thing on purpose, so the eval is shown to discriminate: wrong tool, no tool, overreach
(expensive tool with an injected ``confirm``), cross-agent call, unknown tool, garbage arguments, endless repeat.
``mock`` is the platform's own deterministic stub. A live model is opt-in (``live_policy``) and never used by tests.
"""
from __future__ import annotations

from typing import Any

from app.core.llm import MockLLM
from app.tools.base import Tool


class ScriptedLLM:
    """Plays a fixed plan: per turn, a list of {name, input} calls issued in order, then stops."""

    def __init__(self, plan: list[list[dict[str, Any]]], agents: list[str | None] | None = None) -> None:
        self.plan = plan
        self.agents = agents or []
        self.turn = 0
        self._i = 0

    def begin_turn(self, turn: int) -> None:
        self.turn, self._i = turn, 0

    def classify(self, message: str, options: list[str], descriptions: dict[str, str]) -> str:
        want = self.agents[self.turn] if self.turn < len(self.agents) else None
        return want if want in options else options[0]

    def select_tool(self, agent: str, message: str, tools: list[Tool], state_summary: dict[str, Any]):
        calls = self.plan[self.turn] if self.turn < len(self.plan) else []
        if self._i >= len(calls):
            return None
        call = calls[self._i]
        self._i += 1
        return {"name": call["name"], "input": dict(call.get("input") or {})}


class _Wrapping:
    """Base for faulty policies: delegate to a scripted reference and corrupt one aspect."""

    def __init__(self, reference: ScriptedLLM) -> None:
        self.ref = reference

    def begin_turn(self, turn: int) -> None:
        self.ref.begin_turn(turn)

    def classify(self, message, options, descriptions):
        return self.ref.classify(message, options, descriptions)


class WrongToolLLM(_Wrapping):
    """Picks a different, cheap tool than the plan asks for: a sibling of the same agent when it has one, else a
    harmless decoy (``list_pk_models``)."""

    DECOY = "list_pk_models"

    def select_tool(self, agent, message, tools, state_summary):
        choice = self.ref.select_tool(agent, message, tools, state_summary)
        if not choice:
            return None
        siblings = sorted(t.name for t in tools if t.name != choice["name"] and not t.expensive)
        return {"name": siblings[0] if siblings else self.DECOY, "input": {}}


class NoToolLLM(_Wrapping):
    """Never selects a tool (answers nothing)."""

    def select_tool(self, agent, message, tools, state_summary):
        return None


class OverreachLLM(_Wrapping):
    """Asks for an expensive fit and tries to approve it itself with ``confirm``."""

    def __init__(self, reference: ScriptedLLM, tool: str = "run_bootstrap") -> None:
        super().__init__(reference)
        self.tool, self.fired = tool, False

    def begin_turn(self, turn: int) -> None:
        super().begin_turn(turn)
        self.fired = False

    def select_tool(self, agent, message, tools, state_summary):
        if self.fired:
            return None
        self.fired = True
        return {"name": self.tool, "input": {"confirm": True}}


class CrossAgentLLM(_Wrapping):
    """Names a tool that belongs to another agent (the registry is the only guard)."""

    def __init__(self, reference: ScriptedLLM, tool: str = "generate_report") -> None:
        super().__init__(reference)
        self.tool, self.fired = tool, False

    def begin_turn(self, turn: int) -> None:
        super().begin_turn(turn)
        self.fired = False

    def select_tool(self, agent, message, tools, state_summary):
        if self.fired:
            return None
        self.fired = True
        return {"name": self.tool, "input": {}}


class UnknownToolLLM(_Wrapping):
    """Names a tool that does not exist."""

    def __init__(self, reference: ScriptedLLM) -> None:
        super().__init__(reference)
        self.fired = False

    def begin_turn(self, turn: int) -> None:
        super().begin_turn(turn)
        self.fired = False

    def select_tool(self, agent, message, tools, state_summary):
        if self.fired:
            return None
        self.fired = True
        return {"name": "definitely_not_a_tool", "input": {}}


class BadArgsLLM(_Wrapping):
    """The right tool with garbage arguments."""

    def select_tool(self, agent, message, tools, state_summary):
        choice = self.ref.select_tool(agent, message, tools, state_summary)
        if not choice:
            return None
        return {"name": choice["name"], "input": {"dataset_id": "ds_invented", "path": "../../etc/passwd", "n": -1}}


class RepeatLLM(_Wrapping):
    """Re-issues the first planned call forever (the platform must notice and stop)."""

    def select_tool(self, agent, message, tools, state_summary):
        calls = self.ref.plan[self.ref.turn] if self.ref.turn < len(self.ref.plan) else []
        return {"name": calls[0]["name"], "input": dict(calls[0].get("input") or {})} if calls else None


FAULTY = {
    "wrong_tool": WrongToolLLM, "no_tool": NoToolLLM, "overreach": OverreachLLM,
    "cross_agent": CrossAgentLLM, "unknown_tool": UnknownToolLLM, "bad_args": BadArgsLLM, "repeat": RepeatLLM,
}


def plan_from_scenario(scenario) -> list[list[dict[str, Any]]]:
    """The ideal call sequence implied by the scenario's own expectations (the reference policy)."""
    plan = []
    for turn in scenario.turns:
        calls = []
        for tool in turn.expect_tools:
            calls.append({"name": tool, "input": dict(turn.args_contain.get(tool, {}))})
        if turn.expect_pending:
            calls.append({"name": turn.expect_pending, "input": {}})
        for tool, args in turn.attempt:
            calls.append({"name": tool, "input": dict(args)})
        plan.append(calls)
    return plan


def reference_policy(scenario) -> ScriptedLLM:
    """An ideal model: routes to the expected agent where keywords cannot, then calls the expected tools."""
    return ScriptedLLM(plan_from_scenario(scenario), [t.expect_agent for t in scenario.turns])


def faulty_policy(kind: str, scenario):
    return FAULTY[kind](reference_policy(scenario))


def mock_policy() -> MockLLM:
    return MockLLM()


def live_policy():
    """The configured real model (Anthropic or an OpenAI-compatible server). Only the CLI with --allow-live calls
    this; it costs money and needs a key. Tests never do."""
    from app.core.llm import get_llm
    return get_llm()
