"""Scenario specification. Pure data: no tool names are hard-coded outside the scenario list."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

CATEGORIES = ("routing", "tool_value", "governance", "robustness", "multi_turn")


@dataclass(frozen=True)
class TurnSpec:
    """One user message and what the platform+model should do with it."""
    message: str
    expect_agent: str | None = None                 # the routed agent
    expect_tools: tuple[str, ...] = ()              # tools that must have EXECUTED this turn, in order
    forbid_tools: tuple[str, ...] = ()              # tools that must NOT have executed
    expect_no_tool: bool = False                    # nothing may execute (and no state may be written)
    expect_pending: str | None = None               # a proposal for this tool must be recorded, nothing computed
    args_contain: dict[str, dict[str, Any]] = field(default_factory=dict)   # tool -> subset its args must contain
    attempt: tuple[tuple[str, dict[str, Any]], ...] = ()    # calls the REFERENCE policy tries on purpose (guard probes)
    expect_refused: tuple[str, ...] = ()                    # attempted tools that must be refused (nothing ran)


@dataclass(frozen=True)
class AnalyticCheck:
    """An independent closed-form truth for a number in a tool result (no PharmAgent code involved)."""
    tool: str
    path: tuple[str, ...]                           # keys into the tool's result dict
    value: float
    rel: float = 1e-3


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    category: str
    description: str
    turns: tuple[TurnSpec, ...]
    dataset: str | None = None                      # path relative to backend/, loaded as a NON-graded precondition
    reference_tools: tuple[str, ...] = ()           # tools whose numeric result must equal a direct reference run
    reference_args: dict[str, dict[str, Any]] = field(default_factory=dict)
    analytic: tuple[AnalyticCheck, ...] = ()
    strict_agents: bool = False                     # a tool outside the routed agent's own list is a hard fail
    routing_stage: str = "keyword"                  # which stage SHOULD decide: keyword | llm (informational)

    def __post_init__(self) -> None:
        if self.category not in CATEGORIES:
            raise ValueError(f"unknown category {self.category!r}")
        if not self.turns:
            raise ValueError("a scenario needs at least one turn")
