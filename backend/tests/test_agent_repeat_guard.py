"""A turn must not re-run the same tool with the same arguments.

select_tool() is stateless (message + state summary, no tool results), so a
weak model that does not notice the state changed keeps choosing the same
call; observed with a 0.5B local model: run_qc executed 6 times (MAX_TOOL_STEPS)
in one turn. The guard stops after the first identical repeat; a DIFFERENT
tool or different arguments still proceed.
"""
from __future__ import annotations

from app.agents.base import Agent
from app.core.audit import AuditChain
from app.core.pharmstate import PharmState
from app.tools.base import ToolContext
from app.tools.builtins import default_registry


class _Repeat:
    """Keeps selecting the given calls in order, then the last one forever."""

    def __init__(self, calls):
        self.calls = list(calls)

    def select_tool(self, agent, message, tools, state_summary):
        if len(self.calls) > 1:
            return self.calls.pop(0)
        return self.calls[0]


def _turn(llm, agent="modeler"):
    # list_pk_models is cheap, argument-free and succeeds on an empty state
    return Agent(name=agent, system_prompt="").run_turn(
        state=PharmState(), message="models", llm=llm, registry=default_registry(),
        ctx=ToolContext(), audit=AuditChain(), clock=lambda: "t0", actor="t")


def test_identical_repeat_is_executed_once():
    res = _turn(_Repeat([{"name": "list_pk_models", "input": {}}]))
    ran = [c for c in res.tool_calls if c.get("summary")]
    assert len(ran) == 1
    assert any(c.get("stopped") == "repeat" for c in res.tool_calls)


def test_different_arguments_are_not_a_repeat():
    res = _turn(_Repeat([{"name": "list_pk_models", "input": {"family": "pk"}},
                         {"name": "list_pk_models", "input": {"family": "pd"}}]))
    ran = [c for c in res.tool_calls if c.get("summary")]
    assert len(ran) == 2


def test_repeat_note_is_human_readable():
    res = _turn(_Repeat([{"name": "list_pk_models", "input": {}}]))
    assert any("already ran" in m.lower() and "list_pk_models" in m for m in res.messages)
