"""Tool framework.

A Tool is deterministic Python. Each declares a JSON input schema (so the LLM
can call it via tool-use), an owning agent (for write-access), and a `run`
function returning a ToolResult. The registry executes a tool with full audit
wrapping and write-access enforcement — this is the single choke point through
which every computation flows.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from app.core.audit import AuditChain
from app.core.pharmstate import PharmState, apply_writes


@dataclass
class ToolContext:
    """Server-side resources tools may use. NEVER serialized to the LLM."""
    dataset_store: dict[str, pd.DataFrame] = field(default_factory=dict)
    data_dir: str = "data"


@dataclass
class ToolResult:
    summary: str                      # human/LLM-facing one-liner
    writes: dict[str, Any] = field(default_factory=dict)   # -> PharmState
    result: dict[str, Any] = field(default_factory=dict)   # payload for UI/LLM
    action: str = ""                  # audit action description


class ExpensiveToolError(RuntimeError):
    """Raised when a long-running fit is invoked from a path that has no admission
    control. Callers that ARE admission-controlled (the job-backed endpoints via
    ``Orchestrator.run_tool``, and workflow templates whose expensive legs sit
    behind a human gate) opt in with ``allow_expensive=True``."""


@dataclass
class Tool:
    name: str
    description: str
    agent: str                        # owning agent (write-access key)
    input_schema: dict[str, Any]
    run: Callable[[PharmState, ToolContext, dict[str, Any]], ToolResult]
    expensive: bool = False           # long-running fit: endpoint/job-only, never
    #                                   run synchronously from a chat turn

    def to_anthropic(self) -> dict[str, Any]:
        """Tool definition in Anthropic tool-use format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"unknown tool: {name}")
        return self._tools[name]

    def names(self) -> list[str]:
        return sorted(self._tools)

    def for_agent(self, agent: str) -> list[Tool]:
        return [t for t in self._tools.values() if t.agent == agent]

    def execute(
        self,
        name: str,
        *,
        state: PharmState,
        ctx: ToolContext,
        args: dict[str, Any],
        audit: AuditChain,
        timestamp: str,
        actor: str = "",
        allow_expensive: bool = False,
    ) -> tuple[PharmState, ToolResult]:
        """Run a tool: compute → audit → apply writes. The only execution path.

        ``actor`` is the authenticated identity, recorded (tamper-evidently) on
        the audit entry.

        Admission control lives HERE rather than in the agent loop: this is the
        single choke point every caller funnels through, so a tool named by an
        LLM that is not in the calling agent's own list cannot slip past it.
        ``allow_expensive`` must be set explicitly by callers that are themselves
        admission-controlled — see ExpensiveToolError.
        """
        tool = self.get(name)
        if tool.expensive and not allow_expensive:
            raise ExpensiveToolError(
                f"'{name}' is a long-running fit — launch it from its dedicated "
                "control (Population NLME / Covariate SCM / Compare engines), which "
                "runs it as a bounded background job. It cannot run on this path.")
        res = tool.run(state, ctx, args)
        audit.append(
            agent=tool.agent,
            tool=tool.name,
            action=res.action or res.summary,
            inputs=args,
            outputs=res.result,
            timestamp=timestamp,
            actor=actor,
        )
        new_state = apply_writes(state, tool.agent, res.writes) if res.writes else state
        return new_state, res
