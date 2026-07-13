"""Dose-Proportionality Agent tools: power-model assessment over NCA exposures."""
from __future__ import annotations

from typing import Any

from app.compute.dose_proportionality import assess_dose_proportionality
from app.core.pharmstate import PharmState
from app.tools.base import Tool, ToolContext, ToolResult

_DP_PARAMS = ("Cmax", "AUC_last", "AUC_inf")


def run_dose_proportionality(state: PharmState, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    params = state.nca_parameters
    if not params:
        status = {"status": "no_nca", "message": "Run NCA first; dose proportionality needs per-subject exposures."}
        return ToolResult(
            summary="Dose proportionality skipped: no NCA results yet.",
            action="run_dose_proportionality(no_nca)",
            writes={"dose_prop_results": status}, result=status)

    # align dose with each exposure, keeping only rows that have a dose
    rows = [r for r in params if r.get("dose") is not None]
    doses = [float(r["dose"]) for r in rows]
    values_by_param = {p: [r.get(p) for r in rows] for p in _DP_PARAMS}

    distinct = sorted(set(doses))
    if len(distinct) < 2:
        status = {"status": "single_dose",
                  "message": f"Need >=2 distinct dose levels; found {distinct}."}
        return ToolResult(
            summary=f"Dose proportionality skipped: only one dose level ({distinct}).",
            action="run_dose_proportionality(single_dose)",
            writes={"dose_prop_results": status}, result=status)

    res = assess_dose_proportionality(doses, values_by_param)
    res.update({"status": "ok", "dose_levels": distinct, "n_subjects": len(rows)})

    verdict = "dose-proportional" if res.get("proportional") else "NOT dose-proportional"
    return ToolResult(
        summary=(f"Dose proportionality (power model) across {len(distinct)} dose levels: "
                 f"{verdict} (slope CI vs critical region)."),
        action=f"run_dose_proportionality({len(distinct)} levels)",
        writes={"dose_prop_results": res}, result=res)


TOOLS = [
    Tool("run_dose_proportionality",
         "Assess dose proportionality with the power model (log exposure vs log dose); "
         "reports slope, its CI, and the Smith critical-region verdict for Cmax and AUC.",
         "dose_prop",
         {"type": "object",
          "properties": {"dataset_id": {"type": "string"}}, "required": []},
         run_dose_proportionality),
]
