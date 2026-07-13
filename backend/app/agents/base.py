"""Agent base.

An agent owns a system prompt, a set of bound tools (by ownership in the
registry), and a bounded tool-use loop. It mutates state ONLY by executing
tools through the registry (which enforces audit + write-access). Agents never
call each other.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.core.audit import AuditChain
from app.core.pharmstate import PharmState
from app.tools.base import ToolContext, ToolRegistry

MAX_TOOL_STEPS = 6


@dataclass
class AgentResult:
    state: PharmState
    messages: list[str] = field(default_factory=list)   # human-facing log
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Agent:
    name: str
    system_prompt: str

    def _state_summary(self, state: PharmState) -> dict[str, Any]:
        """Compact, privacy-safe view of state for LLM tool selection."""
        return {
            "dataset_id": state.dataset_id,
            "has_metadata": state.dataset_metadata is not None,
            "data_quality": (state.data_quality or {}).get("quality_flags"),
            "nca_parameters": "present" if state.nca_parameters else None,
            "be_results": "present" if state.be_results else None,
            "dose_prop_results": "present" if state.dose_prop_results else None,
            "compartmental_results": "present" if state.compartmental_results else None,
            "poppk_results": "present" if state.poppk_results else None,
            "pk_model_results": "present" if state.pk_model_results else None,
            "qc_verdict": state.qc_verdict,
            "report_path": state.report_path,
        }

    def run_turn(self, *, state: PharmState, message: str, llm, registry: ToolRegistry,
                 ctx: ToolContext, audit: AuditChain, clock, actor: str = "") -> AgentResult:
        tools = registry.for_agent(self.name)
        result = AgentResult(state=state)
        for _ in range(MAX_TOOL_STEPS):
            choice = llm.select_tool(self.name, message, tools, self._state_summary(result.state))
            if not choice:
                break
            try:
                new_state, tool_res = registry.execute(
                    choice["name"], state=result.state, ctx=ctx,
                    args=choice.get("input", {}), audit=audit, timestamp=clock(),
                    actor=actor,
                )
            except Exception as exc:
                # A tool failed (e.g. NCA requested before a dataset is loaded).
                # Surface it as a chat message rather than 500-ing the whole turn.
                result.messages.append(f"Could not run {choice['name']}: {exc}")
                result.tool_calls.append({"tool": choice["name"], "error": str(exc)})
                break
            result.state = new_state
            result.messages.append(tool_res.summary)
            result.tool_calls.append({"tool": choice["name"], "summary": tool_res.summary})
        if not result.messages:
            result.messages.append(f"[{self.name}] no action taken.")
        return result
