"""Reviewer Agent tool: adversarial, independent refutation of the current results.

Wraps the deterministic engine in app.compute.adversarial. Pulls the raw dataset
(when available) so the engine can recompute Cmax / AUC bands independently of the
analysis code path, then writes severity-ranked findings + a checkable goal verdict
to PharmState.review_results.
"""
from __future__ import annotations

import logging
from typing import Any

from app.compute import adversarial
from app.core.pharmstate import PharmState
from app.tools.base import Tool, ToolContext, ToolResult

log = logging.getLogger("pharmagent")


def _roles(state: PharmState) -> dict[str, str]:
    if state.dataset_metadata and state.dataset_metadata.get("detected_roles"):
        return state.dataset_metadata["detected_roles"]
    return {}


def _wants_mikebrain(args: dict[str, Any]) -> bool:
    """Strict opt-in: only a boolean True (or the string "true") enables the advisory consult.

    The flag is honoured when a caller passes it (run_tool / the API) but deliberately
    NOT advertised in the LLM-facing schema below: it is synchronous and up to three
    local-LLM calls long, so a chat agent must not be able to switch it on by itself.
    Offering it to the planner/critic is an M3 decision (docs/mikebrain.md)."""
    flag = args.get("consult_mikebrain", False)
    if isinstance(flag, str):
        return flag.strip().lower() == "true"
    return flag is True


def adversarial_review(state: PharmState, ctx: ToolContext,
                       args: dict[str, Any]) -> ToolResult:
    goal = (args.get("goal") or adversarial.DEFAULT_GOAL).strip() or adversarial.DEFAULT_GOAL
    df = None
    if state.dataset_id and state.dataset_id in ctx.dataset_store:
        df = ctx.dataset_store[state.dataset_id]

    result = adversarial.review(state.model_dump(), df, _roles(state), goal=goal)
    advisory_note = ""
    if _wants_mikebrain(args):
        # Advisory only: literature advisories are severity-free, live under their own
        # key, are excluded from counts and can never change goal_met or status.
        from app.tools.knowledge_tools import review_advisories
        try:
            result = {**result, **review_advisories(result, state, ctx)}
        except Exception as exc:  # noqa: BLE001 - an advisory must never take the review down
            log.warning("mikebrain_advisory_failed", extra={"error": str(exc)[:200]})
            result = {**result, "advisories": [], "advisory_status": f"error: {str(exc)[:120]}"}
        advisory_note = (f" mikebrain: {len(result['advisories'])} literature advisory(ies) "
                         f"[{result['advisory_status']}] — advisory only, not counted.")

    c = result["counts"]
    # Use the engine's own verdict: it distinguishes UNVERIFIABLE (results present
    # but no raw data to recompute them from) from an actual pass.
    verdict = result.get("status") or ("GOAL MET" if result["goal_met"]
                                       else "FINDINGS BLOCK GOAL")
    summary = (
        f"Adversarial review: {verdict} — {result['n_findings']} finding(s) "
        f"[{c['CRITICAL']} critical, {c['HIGH']} high, {c['MEDIUM']} medium, "
        f"{c['LOW']} low]. Goal: {goal}."
    )
    if result.get("unverifiable"):
        summary += f" NOT independently verified: {result['unverifiable_reason']}."
    summary += advisory_note
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
