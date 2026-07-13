"""Reviewer Agent tool: adversarial, independent refutation of the current results.

Wraps the deterministic engine in app.compute.adversarial. Pulls the raw dataset
(when available) so the engine can recompute Cmax / AUC bands independently of the
analysis code path, then writes severity-ranked findings + a checkable goal verdict
to PharmState.review_results.
"""
from __future__ import annotations

from typing import Any

from app.compute import adversarial
from app.core.pharmstate import PharmState
from app.tools.base import Tool, ToolContext, ToolResult


def _roles(state: PharmState) -> dict[str, str]:
    if state.dataset_metadata and state.dataset_metadata.get("detected_roles"):
        return state.dataset_metadata["detected_roles"]
    return {}


def adversarial_review(state: PharmState, ctx: ToolContext,
                       args: dict[str, Any]) -> ToolResult:
    goal = (args.get("goal") or adversarial.DEFAULT_GOAL).strip() or adversarial.DEFAULT_GOAL
    df = None
    if state.dataset_id and state.dataset_id in ctx.dataset_store:
        df = ctx.dataset_store[state.dataset_id]

    result = adversarial.review(state.model_dump(), df, _roles(state), goal=goal)

    c = result["counts"]
    verdict = "GOAL MET" if result["goal_met"] else "FINDINGS BLOCK GOAL"
    summary = (
        f"Adversarial review: {verdict} — {result['n_findings']} finding(s) "
        f"[{c['CRITICAL']} critical, {c['HIGH']} high, {c['MEDIUM']} medium, "
        f"{c['LOW']} low]. Goal: {goal}."
    )
    return ToolResult(
        summary=summary,
        action=f"adversarial_review(goal={goal!r}) -> goal_met={result['goal_met']}",
        writes={"review_results": result},
        result=result,
    )


TOOLS = [
    Tool(
        "adversarial_review",
        "Adversarially review the current analysis: independently recompute key "
        "quantities from the raw data, challenge every reported value, and emit "
        "severity-ranked findings against a checkable goal (default: zero unresolved "
        "CRITICAL or HIGH). Args: goal (optional string).",
        "reviewer",
        {"type": "object",
         "properties": {"goal": {"type": "string"}}, "required": []},
        adversarial_review,
    ),
]
