"""Isolated, read-only tool adapter for the PopPK Physiological Plausibility Checker.

Does not mutate PharmState (`writes={}`) and is intentionally not registered in
`builtins.py` in this slice so guarded runtime files remain untouched.
"""
from __future__ import annotations

from typing import Any

from app.compute.poppk_plausibility import evaluate_poppk_plausibility
from app.core.pharmstate import PharmState
from app.tools.base import Tool, ToolContext, ToolResult

_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "model_id": {"type": "string"},
        "context": {"type": "object"},
        "parameters": {"type": "array", "items": {"type": "object"}},
        "allometric_exponents": {"type": "object"},
        "iiv": {"type": "array", "items": {"type": "object"}},
        "omega_matrix": {"type": "array"},
        "sigma_matrix": {"type": "array"},
        "correlations": {"type": "object"},
        "residual_error": {"type": "array", "items": {"type": "object"}},
        "predictions": {"type": "array", "items": {"type": "object"}},
        "justifications": {"type": "array", "items": {"type": "object"}},
    },
    "required": ["parameters"],
}


def screen_poppk_plausibility(
    state: PharmState,
    ctx: ToolContext,
    args: dict[str, Any],
) -> ToolResult:
    """Run deterministic PopPK physiological plausibility screening without mutating PharmState."""
    del state, ctx  # strictly read-only evaluation of `args`
    report = evaluate_poppk_plausibility(args)
    payload = report.to_dict()
    flagged = report.summary_counts.get("FLAGGED", 0)
    unresolved = report.summary_counts.get("UNRESOLVED", 0)
    summary = (
        f"PopPK plausibility screening for '{report.model_id}': "
        f"status={report.screening_status} (flagged={flagged}, unresolved={unresolved}). "
        f"{report.disclaimer}"
    )
    return ToolResult(
        summary=summary,
        writes={},
        result=payload,
        action=f"screen_poppk_plausibility(model_id={report.model_id}, status={report.screening_status})",
    )


TOOLS = [
    Tool(
        name="screen_poppk_plausibility",
        description=(
            "Deterministic physiological plausibility screening for PopPK model parameters "
            "(20 rules across units, hepatic/renal flows, volumes, allometry, IIV/Omega PSD, "
            "residual error, and mAb bounds). Read-only screening only; not regulatory validation."
        ),
        agent="poppk",
        input_schema=_INPUT_SCHEMA,
        run=screen_poppk_plausibility,
    ),
]
