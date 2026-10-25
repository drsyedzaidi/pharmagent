"""Deterministic planner policy — pure functions, no I/O.

The goal-driven workflow leg lets an LLM PROPOSE the next tool from the
diagnostics in state; this module decides what happens to the proposal:

* ``autonomous`` — admitted; the orchestrator executes it inline (audited).
* ``gated``      — admitted as a PROPOSAL only; it is parked in
                   ``state.pending_tool`` for a human approve / reject exactly
                   like a chat-proposed expensive tool. Nothing is computed.
* rejected       — not admitted (out of the template's allow-list, a model name
                   outside the PK model library, scientific prerequisites not in
                   state, an identical repeat, or budget).

Every rule here is data the owner can read in one screen: the prerequisite
DAG, the autonomous allow-list, the argument keys that turn a cheap tool into
a scientific decision, and the step budgets. Nothing in this module touches
the registry, the audit chain, or the LLM — it only reads ``PharmState`` and
``Tool`` flags, so a decision is reproducible from its inputs.
"""
from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from app.compute.pk_models import REGISTRY
from app.core.pharmstate import PharmState
from app.tools.base import Tool

MODE_AUTONOMOUS = "autonomous"
MODE_GATED = "gated"
MODE_REJECTED = "rejected"

DEFAULT_MAX_STEPS = 20      # planner proposals per goal run (rejections count)
DEFAULT_MAX_GATED = 6       # human gates per goal run
MAX_REPAIRS_PER_STEP = 2    # bounded repair attempts after a rejected proposal

_RANDOM_DS_ID_RE = re.compile(r"\bds_[0-9a-f]{8}\b")
_DEFAULT_COMPARISON_MODELS = ["oral_1cmt", "oral_2cmt"]
_PRE_GATE_ORDER = (
    "load_dataset",
    "profile_pk_dataset",
    "generate_spaghetti_plot",
    "fit_pk_model",
    "run_nlme",
)

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


# Argument keys that name a model of the PK model library (the keys of
# ``app.compute.pk_models.REGISTRY``, e.g. ``oral_1cmt``). A name outside it is
# rejected before the gate: ``fit_pk_model`` silently drops unknown names from
# ``models``, so approving such a proposal would run a different comparison than
# the one the reviewer was shown. ``fit_compartmental`` is not listed: its own
# ``models`` argument is a different namespace ("1cmt" / "2cmt").
MODEL_KEY_ARGS: dict[str, tuple[str, ...]] = {
    "fit_pk_model": ("model_key", "models"),
    "run_nlme": ("model_key",),
    "run_scm": ("model_key",),
    "run_dose_sweep": ("model_key",),
    "simulate_pk_profile": ("model_key",),
}

# Arguments of MODEL_KEY_ARGS that carry a LIST of model names; every other one carries a
# single name (a string). The shape matters because the tools read them that way: a bare
# string in ``models`` is iterated character by character (fit_pk_model then compares 0
# models and still reports success), and a list ``model_key`` is looked up in the
# registry and raises TypeError (unhashable) after the human has approved it.
MODEL_LIST_ARGS: frozenset[str] = frozenset({"models"})


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


def invalid_model_keys(tool: str, args: Mapping[str, Any] | None) -> str | None:
    """A refusal reason when ``args`` names a model outside the PK model library or gives a
    model argument the wrong shape (see ``MODEL_LIST_ARGS``), or None.

    An absent or empty value is not checked: the tools treat it as "use the fitted
    model". Every offending name is reported with the valid keys, and every misshapen
    argument with the shape it needs, so the planner can correct the proposal from its
    own history.
    """
    bad: list[str] = []
    wrong_shape: list[str] = []
    for key in MODEL_KEY_ARGS.get(tool, ()):
        value = (args or {}).get(key)
        if not value:
            continue
        if key in MODEL_LIST_ARGS:
            names: Sequence[Any] | None = value if isinstance(value, (list, tuple)) else None
            needs = "a list of model keys"
        else:
            names = [value] if isinstance(value, str) else None
            needs = "one model key, as a string"
        if names is None:
            wrong_shape.append(f"{key}={value!r} (must be {needs})")
            continue
        bad.extend(repr(n) for n in names if not isinstance(n, str) or n not in REGISTRY)
    reasons: list[str] = []
    if wrong_shape:
        reasons.append(f"{tool} model arguments have the wrong shape: {', '.join(wrong_shape)}")
    if bad:
        reasons.append(f"{tool} names a model outside the PK model library: {', '.join(bad)}; "
                       f"valid keys: {', '.join(sorted(REGISTRY))}")
    return "; ".join(reasons) or None


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


def consecutive_rejections(history: Sequence[Mapping[str, Any]]) -> int:
    """Count trailing rejected proposals (since the last admitted step)."""
    count = 0
    for h in reversed(history):
        if h.get("allow"):
            break
        count += 1
    return count


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
    repair_limit_hit = consecutive_rejections(history) >= MAX_REPAIRS_PER_STEP
    if tool_name not in roster:
        return Decision(False, MODE_REJECTED,
                        f"{tool_name} is not in this workflow's allowed_tools",
                        stop=repair_limit_hit)
    invalid = invalid_model_keys(tool_name, args)
    if invalid:
        return Decision(False, MODE_REJECTED, invalid, stop=repair_limit_hit)
    unmet = unmet_prerequisite(tool_name, state)
    if unmet:
        return Decision(False, MODE_REJECTED, unmet, stop=repair_limit_hit)
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


def _summary_flags(state_summary: Mapping[str, Any],
                   history: Sequence[Mapping[str, Any]] = ()) -> dict[str, bool]:
    admitted = {h.get("tool") for h in history if h.get("allow")}
    loaded = bool(
        state_summary.get("dataset_loaded")
        or state_summary.get("dataset_id")
        or "load_dataset" in admitted
    )
    configured = bool(state_summary.get("dataset_configured") or loaded)
    profiled = bool(
        state_summary.get("dataset_profiled")
        or state_summary.get("data_quality") is not None
        or "profile_pk_dataset" in admitted
    )
    plotted = bool(
        state_summary.get("spaghetti_plotted")
        or state_summary.get("spaghetti_data") is not None
        or "generate_spaghetti_plot" in admitted
    )
    pk_fitted = bool(
        state_summary.get("pk_model_results") is not None
        or "fit_pk_model" in admitted
        or state_summary.get("pending_tool") == "fit_pk_model"
    )
    nlme_fitted = bool(
        state_summary.get("nlme_results") is not None
        or "run_nlme" in admitted
        or state_summary.get("pending_tool") == "run_nlme"
    )
    return {
        "dataset_configured": configured,
        "dataset_loaded": loaded,
        "dataset_profiled": profiled,
        "spaghetti_plotted": plotted,
        "pk_fitted": pk_fitted,
        "nlme_fitted": nlme_fitted,
    }


def admissible_actions(
    state_summary: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]] = (),
    tools: Sequence[Tool] | None = None,
) -> list[dict[str, Any]]:
    """Compute currently admissible actions for constrained planner selection.

    Walks the canonical pre-gate sequence (load_dataset -> profile_pk_dataset ->
    generate_spaghetti_plot -> fit_pk_model -> run_nlme) for whichever of those
    tools are offered in ``tools``, then falls back to any remaining unexecuted
    tools whose policy prerequisites are satisfied in ``state_summary``.
    """
    offered = [t.name for t in tools] if tools is not None else list(_PRE_GATE_ORDER)
    offered_set = set(offered)
    admitted = {h.get("tool") for h in history if h.get("allow")}
    flags = _summary_flags(state_summary, history)

    if not flags["dataset_loaded"]:
        if "load_dataset" in offered_set:
            if state_summary.get("dataset_configured") is False:
                return []
            return [{"tool": "load_dataset", "args": {}}]
        if state_summary.get("dataset_configured") is False:
            return []

    if flags["dataset_loaded"]:
        if not flags["dataset_profiled"] and "profile_pk_dataset" in offered_set:
            return [{"tool": "profile_pk_dataset", "args": {}}]
        if not flags["spaghetti_plotted"] and "generate_spaghetti_plot" in offered_set:
            return [{"tool": "generate_spaghetti_plot", "args": {}}]
        if not flags["pk_fitted"] and "fit_pk_model" in offered_set:
            return [{
                "tool": "fit_pk_model",
                "args": {"compare": True, "models": list(_DEFAULT_COMPARISON_MODELS)},
            }]

    if flags["pk_fitted"] and not flags["nlme_fitted"] and "run_nlme" in offered_set:
        return [{"tool": "run_nlme", "args": {}}]

    out: list[dict[str, Any]] = []
    for name in offered:
        if name in admitted or name in _PRE_GATE_ORDER:
            continue
        reqs = PREREQUISITES.get(name, ())
        if all(state_summary.get(r) is not None for r in reqs):
            out.append({"tool": name, "args": {}})
    return out


def goal_complete(
    state: PharmState | Mapping[str, Any],
    history: Sequence[Mapping[str, Any]] = (),
    tools: Sequence[Tool] | None = None,
) -> bool:
    """Explicit deterministic completion contract.

    Natural-language 'goal met' is never sufficient by itself when required
    workflow predicates are unsatisfied.
    """
    if isinstance(state, PharmState):
        pending_tool = (state.pending_tool or {}).get("tool")
        return bool(
            state.dataset_id is not None
            and state.data_quality is not None
            and state.spaghetti_data is not None
            and (
                state.pk_model_results is not None
                or pending_tool == "fit_pk_model"
            )
        )
    flags = _summary_flags(state, history)
    if not flags["dataset_configured"] and not flags["dataset_loaded"]:
        return True
    offered = {t.name for t in tools} if tools is not None else set(_PRE_GATE_ORDER[:4])
    if "load_dataset" in offered and not flags["dataset_loaded"]:
        return False
    if "profile_pk_dataset" in offered and not flags["dataset_profiled"]:
        return False
    if "generate_spaghetti_plot" in offered and not flags["spaghetti_plotted"]:
        return False
    if "fit_pk_model" in offered and not flags["pk_fitted"]:
        return False
    return len(admissible_actions(state, history, tools)) == 0


def planner_visible_state(state_summary: Mapping[str, Any],
                          history: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    """Stabilized state summary for the LLM prompt (strips incidental random IDs by default).

    Controlled by ``PHARMAGENT_PLANNER_DATASET_ID_MODE`` for causal ablation testing:
    * ``remove`` (default): omit ``dataset_id`` and expose deterministic booleans;
    * ``fixed``: replace any loaded ``dataset_id`` with ``"ds_fixed001"``;
    * ``random``: preserve the raw ``dataset_id`` value as-is.
    """
    mode = os.environ.get("PHARMAGENT_PLANNER_DATASET_ID_MODE", "remove").strip().lower()
    flags = _summary_flags(state_summary, history)
    clean: dict[str, Any] = {
        "dataset_configured": flags["dataset_configured"],
        "dataset_loaded": flags["dataset_loaded"],
        "dataset_profiled": flags["dataset_profiled"],
        "spaghetti_plotted": flags["spaghetti_plotted"],
    }
    for k, v in state_summary.items():
        if k in ("dataset_id", "turn_observations"):
            continue
        clean[k] = v
    if mode == "fixed":
        clean["dataset_id"] = "ds_fixed001" if flags["dataset_loaded"] else None
    elif mode == "random":
        clean["dataset_id"] = state_summary.get("dataset_id")
    return clean


def planner_visible_observations(
    observations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Stabilize observation summaries so random ``ds_<hex>`` IDs do not perturb decoding."""
    mode = os.environ.get("PHARMAGENT_PLANNER_DATASET_ID_MODE", "remove").strip().lower()
    replacement = "ds_fixed001" if mode == "fixed" else "dataset"
    out: list[dict[str, Any]] = []
    for obs in observations:
        item = dict(obs)
        if mode != "random" and isinstance(item.get("summary"), str):
            item["summary"] = _RANDOM_DS_ID_RE.sub(replacement, item["summary"])
        out.append(item)
    return out


def planner_visible_history(
    history: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Stabilize planner history by stripping orchestrator-injected filesystem paths."""
    out: list[dict[str, Any]] = []
    for entry in history:
        item = dict(entry)
        if isinstance(item.get("args"), Mapping):
            item["args"] = {k: v for k, v in item["args"].items() if k != "path"}
        out.append(item)
    return out

