"""A tool-USING LLM agent — the direct test of the benchmark's thesis.

The tool-free ``llm`` agents reason to a number in text. This agent instead gives
the model the *validated compute tools* by function-calling: the model must pick
the right tool and marshal the dataset into it; the tool then runs the same
validated function the oracle uses. If a real model, given the tools, reaches the
tool-grounded ceiling (~1.0), the benchmark's claim — that grounding, not scale,
closes the gap — is demonstrated rather than asserted. If it mis-routes or
mis-marshals, that is an honest, measured failure of tool orchestration.

The tools wrap ``agents.oracle_predict`` so they are exactly the validated path;
the model still has to choose the tool and supply the correct inputs.
"""
from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Any

from .llm import build_prompt, parse_answer
from .spec import Task

# Tool name → the oracle category it computes + the dataset keys the model must
# supply. The model reads these from the dataset shown in the prompt.
_TOOL_CATEGORY = {
    "nca": "nca",
    "bioequivalence": "be",
    "dose_proportionality": "dp",
    "fit_one_compartment": "compartmental",
    "steady_state_exposure": "exposure",
}

_num_array = {"type": "array", "items": {"type": "number"}}

TOOLS: list[dict[str, Any]] = [
    {"name": "nca",
     "description": "Non-compartmental analysis of a concentration-time profile. "
                    "Returns Cmax, AUC_inf (extrapolated), and terminal t_half.",
     "input_schema": {"type": "object", "properties": {
         "time": _num_array, "conc": _num_array, "dose": {"type": "number"}},
         "required": ["time", "conc", "dose"]}},
    {"name": "bioequivalence",
     "description": "Average bioequivalence for paired test/reference exposures. "
                    "Returns gmr_pct, 90% CI bounds, and the within-limits verdict.",
     "input_schema": {"type": "object", "properties": {
         "test": _num_array, "ref": _num_array}, "required": ["test", "ref"]}},
    {"name": "dose_proportionality",
     "description": "Power-model dose proportionality. Returns the slope and a "
                    "proportional (boolean) verdict.",
     "input_schema": {"type": "object", "properties": {
         "doses": _num_array, "values": _num_array}, "required": ["doses", "values"]}},
    {"name": "fit_one_compartment",
     "description": "Fit a one-compartment first-order oral model to a profile. "
                    "Returns CL, V, ka.",
     "input_schema": {"type": "object", "properties": {
         "time": _num_array, "conc": _num_array, "dose": {"type": "number"}},
         "required": ["time", "conc", "dose"]}},
    {"name": "steady_state_exposure",
     "description": "Forward-simulate steady-state exposure for a multiple-dose "
                    "regimen. Returns Cmax_ss and AUC_tau for the last interval.",
     "input_schema": {"type": "object", "properties": {
         "model": {"type": "string"},
         "params": {"type": "object", "description": "PK parameters, e.g. {CL, V, KA}"},
         "dose": {"type": "number"}, "tau": {"type": "number"},
         "n_doses": {"type": "integer"}},
         "required": ["model", "params", "dose", "tau", "n_doses"]}},
]


def execute_tool(name: str, inp: dict[str, Any]) -> dict[str, Any]:
    """Run the validated computation behind a tool call. Any bad/missing input
    surfaces as an error the model can see and retry, not a silent wrong answer."""
    from .agents import oracle_predict  # the validated path
    category = _TOOL_CATEGORY.get(name)
    if category is None:
        return {"error": f"unknown tool {name!r}"}
    try:
        return oracle_predict(category, inp)
    except Exception as exc:  # missing key, wrong type, non-convergence
        return {"error": f"{type(exc).__name__}: {exc}"}


def _tool_prompt(task: Task) -> str:
    return ("You are a pharmacometrics analyst with validated compute tools. Use "
            "the single appropriate tool to compute the answer from the dataset "
            "below — do NOT compute by hand — then report the tool's result as the "
            "JSON answer.\n\n" + build_prompt(task))


def make_tool_agent(model: str, max_rounds: int = 6) -> Callable[[Task], dict]:
    """A reference agent that gives a real model the validated tools via
    function-calling. Requires ANTHROPIC_API_KEY (no keyless fallback)."""
    box: list[Any] = []

    def agent(task: Task) -> dict:
        if not box:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError(
                    f"tool-using agent for {model} needs ANTHROPIC_API_KEY")
            import anthropic
            box.append(anthropic.Anthropic())
        client = box[0]
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": _tool_prompt(task)}]
        for _ in range(max_rounds):
            resp = client.messages.create(
                model=model, max_tokens=4096, tools=TOOLS, messages=messages)
            if resp.stop_reason != "tool_use":
                text = "".join(b.text for b in resp.content if b.type == "text")
                return parse_answer(text, task)
            messages.append({"role": "assistant", "content": resp.content})
            results = []
            for b in resp.content:
                if b.type == "tool_use":
                    out = execute_tool(b.name, b.input)
                    results.append({
                        "type": "tool_result", "tool_use_id": b.id,
                        "content": json.dumps(out),
                        "is_error": "error" in out})
            messages.append({"role": "user", "content": results})
        return {}  # ran out of rounds without a final answer (a parse-gap)

    return agent
