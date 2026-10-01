"""Statistician Agent tool: Bucher adjusted indirect comparison.

A model may PROPOSE the two direct comparisons (effect + SE or CI, copied from
the literature by the user); this tool COMPUTES the indirect contrast
deterministically (app.compute.indirect). Results accumulate, newest last and
bounded, in ``state.indirect_results["comparisons"]``; each comparison is exactly
the audited tool output, so the briefing memo can match it to its audit entry by
output hash.
"""
from __future__ import annotations

from typing import Any

from app.compute.indirect import (
    DEFAULT_CI_LEVEL,
    DIFFERENCE_SCALES,
    RATIO_SCALES,
    bucher_indirect,
)
from app.compute.levels import level_percent
from app.core.pharmstate import PharmState
from app.tools.base import Tool, ToolContext, ToolResult

MAX_COMPARISONS = 20

_DIRECT_SCHEMA = {
    "type": "object",
    "description": ("A direct comparison against the common comparator B: an effect "
                    "estimate with a standard error on the analysis scale (log for "
                    "HR/OR/RR) OR a confidence interval (ci_lower, ci_upper, ci_level)."),
    "properties": {
        "estimate": {"type": "number"},
        "se": {"type": "number"},
        "ci_lower": {"type": "number"},
        "ci_upper": {"type": "number"},
        "ci_level": {"type": "number",
                     "description": ("confidence level of THIS comparison's CI as published, e.g. "
                                     "0.90 for a 90% CI (default 0.95). Required when the output "
                                     "ci_level is not 0.95 and this leg is given as a CI.")},
        "label": {"type": "string"},
    },
    "required": ["estimate"],
}


def _check_leg_levels(args: dict[str, Any], level: float) -> None:
    """An output level other than the default must not leave an input CI's level implied:
    a model that copied "HR 0.62, 90% CI 0.45-0.88" and set only the top-level
    ci_level would otherwise have that CI read as 95% (SE too small)."""
    if isinstance(level, bool) or not isinstance(level, (int, float)) or level == DEFAULT_CI_LEVEL:
        return                      # an invalid level is rejected by bucher_indirect itself
    for leg in ("ab", "cb"):
        spec = args.get(leg)
        if not isinstance(spec, dict):
            continue
        has_ci = spec.get("ci_lower") is not None or spec.get("ci_upper") is not None
        if has_ci and spec.get("se") is None and spec.get("ci_level") is None:
            raise ValueError(
                f"{leg}.ci_level is required: {leg} is given as a CI and the output ci_level is "
                f"{level:g}. Set {leg}.ci_level to the level of that published CI (the top-level "
                "ci_level applies to the output interval only)")


def _input_levels(comparison: dict[str, Any]) -> str:
    """' Input CI levels: ...' for legs given as a CI; empty when both legs gave an SE."""
    t = comparison["treatments"]
    parts = []
    for leg, (x, y) in (("ab", ("A", "B")), ("cb", ("C", "B"))):
        d = comparison["inputs"][leg]
        if d.get("se_source") != "from_ci":
            continue
        assumed = " assumed" if d.get("ci_level_source") == "default" else ""
        parts.append(f"{t[x]} vs {t[y]} input CI{assumed} {level_percent(d['ci_level'])}%")
    return (" " + "; ".join(parts) + ".") if parts else ""


def indirect_comparison(state: PharmState, ctx: ToolContext,
                        args: dict[str, Any]) -> ToolResult:
    level = args.get("ci_level")
    level = DEFAULT_CI_LEVEL if level is None else level
    _check_leg_levels(args, level)
    comparison = bucher_indirect(
        args.get("ab"), args.get("cb"),
        scale=args.get("scale") or "",
        ci_level=level,
        treatments=(args.get("treatment_a") or "A", args.get("treatment_b") or "B",
                    args.get("treatment_c") or "C"))
    prior = (state.indirect_results or {}).get("comparisons") or []
    comparisons = [*prior, comparison][-MAX_COMPARISONS:]
    pct = level_percent(comparison["ci_level"])
    return ToolResult(
        summary=(f"Indirect comparison {comparison['contrast']}: {comparison['scale']} "
                 f"{comparison['estimate']:.3g} ({pct}% CI {comparison['ci_lower']:.3g} to "
                 f"{comparison['ci_upper']:.3g}), p = {comparison['p_value']:.3g}."
                 + _input_levels(comparison)),
        action=f"indirect_comparison({comparison['scale']}, {comparison['contrast']})",
        writes={"indirect_results": {"comparisons": comparisons}},
        result=comparison,
    )


TOOLS = [
    Tool(
        "indirect_comparison",
        "Bucher adjusted indirect comparison of A vs C through a common comparator "
        "B from two direct comparisons (A vs B and C vs B), each an effect estimate "
        "with a standard error or a confidence interval. Ratio scales (HR, OR, RR) "
        "are computed on the log scale; difference scales (MD, RD, SMD) as given. "
        "Returns the indirect estimate, SE, CI, z and p. Valid only under "
        "transitivity. All numbers are computed, never invented.",
        "statistician",
        {"type": "object",
         "properties": {
             "ab": _DIRECT_SCHEMA,
             "cb": _DIRECT_SCHEMA,
             "scale": {"type": "string", "enum": [*RATIO_SCALES, *DIFFERENCE_SCALES]},
             "ci_level": {"type": "number",
                          "description": ("level of the OUTPUT (indirect) interval only, default "
                                          "0.95; it does not apply to the input CIs, whose level "
                                          "is each leg's own ci_level")},
             "treatment_a": {"type": "string"},
             "treatment_b": {"type": "string", "description": "the common comparator"},
             "treatment_c": {"type": "string"},
         },
         "required": ["ab", "cb", "scale"]},
        indirect_comparison,
    ),
]
