"""PopPK Agent tools: two-stage (STS) population summary + covariate screen.

Deterministic two-stage approximation: typical value = geometric mean of the
individual estimates; IIV = between-subject geometric CV%. NOT a mixed-effects
(NLME) fit. Uses compartmental individual fits when available, else NCA CL/V.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from app.compute.poppk import covariate_effect, two_stage_summary
from app.core.pharmstate import PharmState
from app.core.schema_extractor import detect_roles
from app.tools.base import Tool, ToolContext, ToolResult

_COVARIATE_NAMES = {"wt", "weight", "bw", "bwt", "age", "crcl", "egfr", "bmi"}


def _roles(df: pd.DataFrame, state: PharmState) -> dict[str, str]:
    if state.dataset_metadata and state.dataset_metadata.get("detected_roles"):
        return state.dataset_metadata["detected_roles"]
    return detect_roles(list(df.columns))


def _from_compartmental(state: PharmState) -> list[dict[str, Any]] | None:
    comp = state.compartmental_results
    if not comp or not comp.get("fits"):
        return None
    rows: list[dict[str, Any]] = []
    for f in comp["fits"]:
        if not f.get("converged") or not f.get("params"):
            continue
        p = f["params"]
        rows.append({"subject": f.get("subject"), "CL_F": p.get("CL"),
                     "Vz_F": p.get("V", p.get("V1")), "ka": p.get("ka")})
    return rows or None


def _covariate_by_subject(state: PharmState, ctx: ToolContext) -> tuple[str, dict] | None:
    dsid = state.dataset_id
    if not dsid or dsid not in ctx.dataset_store:
        return None
    df = ctx.dataset_store[dsid]
    roles = _roles(df, state)
    id_col = next((c for c, r in roles.items() if r == "ID"), None)
    if not id_col:
        return None
    used = {c for c, r in roles.items() if r in {"ID", "TIME", "DV", "AMT", "EVID", "MDV", "CMT"}}
    cov_col = next((c for c in df.columns if c not in used and c.strip().lower() in _COVARIATE_NAMES), None)
    if not cov_col:
        return None
    vals = pd.to_numeric(df[cov_col], errors="coerce")
    mapping: dict[Any, float] = {}
    for sid, g in df.assign(_cov=vals).groupby(id_col):
        first = g["_cov"].dropna()
        if len(first):
            mapping[sid] = float(first.iloc[0])
    return (cov_col, mapping) if mapping else None


def run_poppk(state: PharmState, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    source = "compartmental"
    individuals = _from_compartmental(state)
    if individuals is None:
        individuals = state.nca_parameters
        source = "nca"
    if not individuals:
        status = {"status": "no_individuals",
                  "message": "Run NCA (or compartmental fit) first; pop summary needs individual estimates."}
        return ToolResult(
            summary="PopPK skipped: no individual estimates available.",
            action="run_poppk(no_individuals)",
            writes={"poppk_results": status}, result=status)

    summary = two_stage_summary(individuals, keys=("CL_F", "Vz_F", "ka"))
    summary.update({"status": "ok", "source": source})

    cov_writes: dict[str, Any] = {}
    cov = _covariate_by_subject(state, ctx)
    if cov is not None:
        cov_col, mapping = cov
        eff = covariate_effect(individuals, mapping, param_key="CL_F")
        eff["covariate"] = cov_col
        summary["covariate_screen"] = eff
        cov_writes["covariate_results"] = {"covariate": cov_col, "on": "CL_F", **eff}

    n = summary.get("n_subjects", len(individuals))
    keys = ", ".join(summary.get("parameters", {}).keys()) or "none"
    return ToolResult(
        summary=(f"PopPK two-stage summary from {source} estimates ({n} subjects): "
                 f"typical values + IIV for {keys}."),
        action=f"run_poppk(source={source})",
        writes={"poppk_results": summary, **cov_writes}, result=summary)


TOOLS = [
    Tool("run_poppk",
         "Two-stage population PK summary: typical values (geometric mean) and "
         "between-subject variability (IIV, geometric CV%) for CL/F, V/F, ka, plus "
         "a covariate screen. Not a mixed-effects fit.",
         "poppk",
         {"type": "object",
          "properties": {"dataset_id": {"type": "string"}}, "required": []},
         run_poppk),
]
