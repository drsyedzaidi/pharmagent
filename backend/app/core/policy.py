"""Deterministic planner policy — pure functions, no I/O.

The goal-driven workflow leg lets an LLM PROPOSE the next tool from the
diagnostics in state; this module decides what happens to the proposal:

* ``autonomous`` — admitted; the orchestrator executes it inline (audited).
* ``gated``      — admitted as a PROPOSAL only; it is parked in
                   ``state.pending_tool`` for a human approve / reject exactly
                   like a chat-proposed expensive tool. Nothing is computed.
* rejected       — not admitted (out of the template's allow-list, scientific
                   prerequisites not in state, an identical repeat, or budget).

Every rule here is data the owner can read in one screen: the prerequisite
DAG, the autonomous allow-list, the argument keys that turn a cheap tool into
a scientific decision, and the step budgets. Nothing in this module touches
the registry, the audit chain, or the LLM — it only reads ``PharmState`` and
``Tool`` flags, so a decision is reproducible from its inputs.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from app.core.pharmstate import PharmState
from app.tools.base import Tool

MODE_AUTONOMOUS = "autonomous"
MODE_GATED = "gated"
MODE_REJECTED = "rejected"

DEFAULT_MAX_STEPS = 20      # planner proposals per goal run (rejections count)
DEFAULT_MAX_GATED = 6       # human gates per goal run

# Prerequisite DAG: a tool may run only once every listed state field reports
# status == "ok". Tools absent from this map have no state prerequisite here
# (they validate their own inputs and fail loudly).
PREREQUISITES: dict[str, tuple[str, ...]] = {
    "run_nlme": ("pk_model_results",),
    "run_scm": ("nlme_results",),
    "run_diagnostics": ("nlme_results",),
    "run_covariate_forest": ("nlme_results",),
    "run_vpc": ("nlme_results",),
}

# Tools the policy may admit WITHOUT a human gate when their arguments carry
# no scientific choice: data profiling, plots, diagnostics of an existing fit,
# and the adversarial reviewer. Anything not listed here is gated (fail closed).
AUTONOMOUS_TOOLS: frozenset[str] = frozenset({
    "load_dataset", "profile_pk_dataset", "validate_cdisc", "generate_spaghetti_plot",
    "fit_pk_model", "run_diagnostics", "run_covariate_forest", "run_vpc",
    "adversarial_review",
})

# Always a human decision regardless of arguments (the report is the deliverable).
GATED_TOOLS: frozenset[str] = frozenset({"generate_report"})

# Argument keys that encode a scientific model choice. Supplying one turns the
# call into a gated proposal even when the tool is otherwise autonomous.
# ``models`` is fit_pk_model's analogue of ``model_key`` (it restricts the
# candidate set the AIC comparison may pick from).
SCIENTIFIC_CHOICE_KEYS: dict[str, frozenset[str]] = {
    "fit_pk_model": frozenset({"model_key", "models"}),
    "run_nlme": frozenset({"method", "model_key", "error_model", "covariate_model"}),
    "run_scm": frozenset({"method", "model_key", "error_model", "covariate_model"}),
}


@dataclass(frozen=True)
class Budget:
    max_steps: int = DEFAULT_MAX_STEPS
    max_gated: int = DEFAULT_MAX_GATED


@dataclass(frozen=True)
class Decision:
    allow: bool
    mode: str                 # autonomous | gated | rejected
    reason: str
    stop: bool = False        # a rejection the loop must not plan past (repeat / budget)


def budget_from(template: Mapping[str, Any]) -> Budget:
    """The template's ``budget`` block over the defaults; a template cannot raise a cap."""
    raw = template.get("budget") or {}
    steps = int(raw.get("max_steps", DEFAULT_MAX_STEPS))
    gated = int(raw.get("max_gated", DEFAULT_MAX_GATED))
    return Budget(max_steps=min(max(steps, 1), DEFAULT_MAX_STEPS),
                  max_gated=min(max(gated, 0), DEFAULT_MAX_GATED))


def canonical_call(tool: str, args: Mapping[str, Any] | None) -> tuple[str, str]:
    """The identity of a call — the same form ``Agent.run_turn`` uses for its
    last_call repeat check, so both guards agree on what "identical" means."""
    return tool, json.dumps(dict(args or {}), sort_keys=True, default=str)


def unmet_prerequisite(tool: str, state: PharmState) -> str | None:
    """The first prerequisite field not at status == "ok", or None."""
    for field_name in PREREQUISITES.get(tool, ()):
        value = getattr(state, field_name, None)
        status = value.get("status") if isinstance(value, dict) else None
        if status != "ok":
            return f"{tool} requires {field_name}.status == 'ok' (found {status!r})"
    return None


def classify(tool_name: str, args: Mapping[str, Any] | None, tool: Tool | None) -> str:
    """autonomous | gated. Unknown tools and anything off the allow-list are gated."""
    if tool is not None and tool.expensive:
        return MODE_GATED
    if tool_name in GATED_TOOLS:
        return MODE_GATED
    chosen = SCIENTIFIC_CHOICE_KEYS.get(tool_name, frozenset()) & set(args or {})
    if chosen:
        return MODE_GATED
    return MODE_AUTONOMOUS if tool_name in AUTONOMOUS_TOOLS else MODE_GATED


def steps_taken(history: Sequence[Mapping[str, Any]]) -> int:
    return len(history)


def gates_taken(history: Sequence[Mapping[str, Any]]) -> int:
    return sum(1 for h in history if h.get("allow") and h.get("mode") == MODE_GATED)


def budget_exhausted(history: Sequence[Mapping[str, Any]], budget: Budget) -> str | None:
    """A stop reason when the step budget is spent, else None (checked before planning)."""
    n = steps_taken(history)
    if n >= budget.max_steps:
        return f"budget: max_steps {budget.max_steps} reached after {n} planner steps"
    return None


def decide(tool_name: str, args: Mapping[str, Any] | None, state: PharmState, *,
           roster: Mapping[str, Tool], history: Sequence[Mapping[str, Any]],
           budget: Budget) -> Decision:
    """Admit, gate or reject one planner proposal. Order matters: the cheapest,
    most categorical refusals come first so a reason names the real blocker."""
    exhausted = budget_exhausted(history, budget)
    if exhausted:
        return Decision(False, MODE_REJECTED, exhausted, stop=True)
    # The repeat check comes BEFORE roster / prerequisite refusals: a planner that
    # re-proposes a call just rejected for an unmet prerequisite is stuck, and must
    # stop after one repeat instead of burning the whole step budget.
    if history and canonical_call(tool_name, args) == canonical_call(
            history[-1].get("tool", ""), history[-1].get("args")):
        return Decision(False, MODE_REJECTED,
                        f"{tool_name} was already proposed with identical arguments; "
                        "every tool is deterministic, so a repeat cannot produce anything new",
                        stop=True)
    if tool_name not in roster:
        return Decision(False, MODE_REJECTED,
                        f"{tool_name} is not in this workflow's allowed_tools")
    unmet = unmet_prerequisite(tool_name, state)
    if unmet:
        return Decision(False, MODE_REJECTED, unmet)
    mode = classify(tool_name, args, roster[tool_name])
    if mode == MODE_GATED and gates_taken(history) >= budget.max_gated:
        return Decision(False, MODE_REJECTED,
                        f"budget: max_gated {budget.max_gated} human gates already used",
                        stop=True)
    reason = ("admitted: no scientific choice in arguments" if mode == MODE_AUTONOMOUS
              else _gate_reason(tool_name, args, roster[tool_name]))
    return Decision(True, mode, reason)


def _gate_reason(tool_name: str, args: Mapping[str, Any] | None, tool: Tool) -> str:
    if tool.expensive:
        return f"{tool_name} is a long-running fit: human approval required"
    if tool_name in GATED_TOOLS:
        return f"{tool_name} is always a human decision"
    chosen = sorted(SCIENTIFIC_CHOICE_KEYS.get(tool_name, frozenset()) & set(args or {}))
    if chosen:
        return f"{tool_name} sets a scientific choice ({', '.join(chosen)}): human approval required"
    return f"{tool_name} is not on the autonomous allow-list: human approval required"
