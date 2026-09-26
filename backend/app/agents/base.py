"""Agent base.

An agent owns a system prompt, a set of bound tools (by ownership in the
registry), and a bounded tool-use loop. It mutates state ONLY by executing
tools through the registry (which enforces audit + write-access). Agents never
call each other.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from app.core.audit import AuditChain
from app.core.pharmstate import PharmState
from app.tools.base import ExpensiveToolError, ToolContext, ToolRegistry

MAX_TOOL_STEPS = 6


def _n(items) -> int:
    return len(items or [])


def idle_hint(agent: str, state: PharmState) -> str:
    """One line for a turn that selected no tool: what IS loaded and what the
    user can ask for next. Never raw rows — counts and ids only.

    A bare "no action taken" read as "the app is broken" to a desktop user who
    typed "load dataset" with nothing uploaded; the hint has to name the
    prerequisite (upload / path) or the fact that the result already exists.
    """
    loaded = state.dataset_id is not None
    n_subj = (state.dataset_metadata or {}).get("n_subjects")
    subj = f", {n_subj} subjects" if n_subj is not None else ""
    n_nca = _n(state.nca_parameters)
    where = (f"dataset {state.dataset_id}{subj} is loaded" if loaded
             else "no dataset is loaded — use 'Click to upload' (or drag a CSV in), "
                  "or tell me the CSV path")
    if agent == "data_manager":
        if not loaded:
            return f"[data_manager] {where}."
        profiled = state.data_quality is not None
        return (f"[data_manager] {where}"
                f"{' and profiled' if profiled else ''}. Next: compute NCA, fit a PK model, "
                "run the QC review, or ask for the report.")
    if agent == "nca":
        if not loaded:
            return f"[nca] {where}; NCA needs a dataset first."
        if n_nca:
            return (f"[nca] NCA already computed for {n_nca} subjects on {state.dataset_id}. "
                    "Next: QC review, bioequivalence, dose proportionality, or the report.")
        return f"[nca] {where}; ask me to compute NCA."
    done = []
    if n_nca:
        done.append(f"NCA {n_nca} subjects")
    if state.pk_model_results:
        done.append("PK model fit")
    if (state.nlme_results or {}).get("status") == "ok":
        done.append("NLME fit")
    if state.qc_verdict:
        done.append(f"QC {state.qc_verdict}")
    if state.report_path:
        done.append("report")
    have = f" Done so far: {', '.join(done)}." if done else ""
    return f"[{agent}] nothing to run for that request — {where}.{have}"


@dataclass
class AgentResult:
    state: PharmState
    messages: list[str] = field(default_factory=list)   # human-facing log
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    # A proposable expensive tool the LLM selected: {tool, agent, args}. Not
    # executed here; the orchestrator records it as state.pending_tool.
    proposal: dict[str, Any] | None = None


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
            "nlme_results": "present" if (state.nlme_results or {}).get("status") == "ok" else None,
            "simest_results": "present" if state.simest_results else None,
            "stats_advice": "present" if state.stats_advice else None,
            "pending_tool": (state.pending_tool or {}).get("tool"),
            "qc_verdict": state.qc_verdict,
            "report_path": state.report_path,
        }

    def run_turn(self, *, state: PharmState, message: str, llm, registry: ToolRegistry,
                 ctx: ToolContext, audit: AuditChain, clock, actor: str = "") -> AgentResult:
        tools = registry.for_agent(self.name)
        result = AgentResult(state=state)
        last_call: tuple[str, str] | None = None
        for _ in range(MAX_TOOL_STEPS):
            choice = llm.select_tool(self.name, message, tools, self._state_summary(result.state))
            if not choice:
                break
            # select_tool is stateless (no tool results are fed back), so a model
            # that does not read the state summary re-picks the same call every
            # step. Every tool is deterministic, so an identical repeat cannot
            # produce anything new: stop instead of burning MAX_TOOL_STEPS.
            this_call = (choice["name"], json.dumps(choice.get("input") or {}, sort_keys=True,
                                                    default=str))
            if this_call == last_call:
                result.messages.append(
                    f"{choice['name']} already ran with these arguments this turn; "
                    "stopping (the result is above).")
                result.tool_calls.append({"tool": choice["name"], "stopped": "repeat"})
                break
            last_call = this_call
            try:
                # allow_expensive is NOT passed: a chat turn runs synchronously, so a
                # long-running fit here would block the worker AND bypass the job
                # queue. The registry refuses it (ExpensiveToolError) — enforcing
                # there, not on `tools`, also covers a tool the LLM names that is not
                # in this agent's own list.
                new_state, tool_res = registry.execute(
                    choice["name"], state=result.state, ctx=ctx,
                    args=choice.get("input", {}), audit=audit, timestamp=clock(),
                    actor=actor,
                )
            except ExpensiveToolError as exc:
                tool = registry.get(choice["name"])
                if tool.proposable:
                    # Reachable from the agent, never run by it: the refused call
                    # is handed back as a PROPOSAL. The human's approval is the
                    # tool's `confirm` (an LLM-supplied confirm is dropped), and
                    # the orchestrator submits the approved call as a bounded
                    # background job. Nothing is computed on this turn.
                    args = {k: v for k, v in (choice.get("input") or {}).items()
                            if k != "confirm"}
                    result.proposal = {"tool": tool.name, "agent": self.name, "args": args}
                    result.messages.append(
                        f"Proposed {tool.name} — a long-running fit (minutes to tens of "
                        "minutes). Nothing has been computed. Approve it to run as a "
                        "background job, or reject it.")
                    result.tool_calls.append({"tool": tool.name, "proposed": "expensive",
                                              "args": args})
                    break
                result.messages.append(f"{exc} Ask for it from that panel instead.")
                result.tool_calls.append({"tool": choice["name"], "skipped": "expensive"})
                break
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
            result.messages.append(idle_hint(self.name, result.state))
        return result
