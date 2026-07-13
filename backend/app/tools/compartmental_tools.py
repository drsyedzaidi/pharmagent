"""Compartmental Agent tools: fit 1- and 2-compartment oral models per subject."""
from __future__ import annotations

from typing import Any

import pandas as pd

from app.compute.compartmental import fit_compartmental, fit_compartmental_ss
from app.compute.dosing import extract_ss_intervals, is_multiple_dose
from app.core.pharmstate import PharmState
from app.core.schema_extractor import detect_roles
from app.tools.base import Tool, ToolContext, ToolResult


def _roles(df: pd.DataFrame, state: PharmState) -> dict[str, str]:
    if state.dataset_metadata and state.dataset_metadata.get("detected_roles"):
        return state.dataset_metadata["detected_roles"]
    return detect_roles(list(df.columns))


def _compact(out: dict[str, Any], *, steady_state: bool) -> dict[str, Any]:
    fits = out["individual_fits"]
    sel = out["model_selection"]
    counts: dict[str, int] = {}
    for m in sel.values():
        if m:
            counts[m] = counts.get(m, 0) + 1
    return {
        "steady_state": steady_state,
        "n_subjects": out["n_subjects"],
        "n_converged": sum(1 for f in fits if f.get("converged")),
        "model_selection_counts": counts,
        "fits": [{"subject": f.get("subject"), "model": f.get("model"),
                  "converged": f.get("converged"), "params": f.get("params"),
                  "aic": f.get("aic"), "r_squared": f.get("r_squared")}
                 for f in fits],
    }


def fit_compartmental_models(state: PharmState, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    dsid = args.get("dataset_id") or state.dataset_id
    df = ctx.dataset_store[dsid].copy()
    roles = _roles(df, state)
    id_col = next((c for c, r in roles.items() if r == "ID"), None)
    time_col = next((c for c, r in roles.items() if r == "TIME"), None)
    dv_col = next((c for c, r in roles.items() if r == "DV"), None)
    amt_col = next((c for c, r in roles.items() if r == "AMT"), None)
    evid_col = next((c for c, r in roles.items() if r == "EVID"), None)
    ii_col = next((c for c, r in roles.items() if r == "II"), None)
    addl_col = next((c for c, r in roles.items() if r == "ADDL"), None)
    if not (id_col and time_col and dv_col):
        raise ValueError("dataset missing required ID/TIME/DV roles for compartmental fit")

    # Steady-state branch: fit SS models over the last dosing interval.
    records_all = df.to_dict("records")
    if amt_col and ii_col and is_multiple_dose(
            records_all, time_col=time_col, amt_col=amt_col, ii_col=ii_col,
            addl_col=addl_col, id_col=id_col):
        profiles = extract_ss_intervals(
            records_all, id_col=id_col, time_col=time_col, dv_col=dv_col,
            amt_col=amt_col, ii_col=ii_col, addl_col=addl_col)
        if profiles:
            out = fit_compartmental_ss(profiles)
            compact = _compact(out, steady_state=True)
            c = compact["model_selection_counts"]
            return ToolResult(
                summary=(f"Steady-state compartmental fit: {compact['n_converged']}/"
                         f"{compact['n_subjects']} converged "
                         f"({c.get('1cmt_ss',0)}×1-cmt, {c.get('2cmt_ss',0)}×2-cmt, by AIC)."),
                action=f"fit_compartmental_ss({dsid})",
                writes={"compartmental_results": compact}, result=compact)

    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df[dv_col] = pd.to_numeric(df[dv_col], errors="coerce")
    if amt_col:
        df[amt_col] = pd.to_numeric(df[amt_col], errors="coerce")

    if evid_col:
        ev = pd.to_numeric(df[evid_col], errors="coerce").fillna(0)
        dose_rows, obs = df[ev == 1], df[ev == 0]
    elif amt_col:
        dose_rows = df[df[amt_col].fillna(0) > 0]
        obs = df[df[amt_col].fillna(0) == 0]
    else:
        dose_rows, obs = df.iloc[0:0], df

    dose_by_subject: dict[Any, float] = {}
    if amt_col and len(dose_rows):
        for sid, g in dose_rows.groupby(id_col):
            amts = g[amt_col].dropna()
            if len(amts):
                dose_by_subject[sid] = float(amts.iloc[0])

    obs = obs.dropna(subset=[time_col, dv_col])
    records = obs[[id_col, time_col, dv_col]].to_dict("records")
    models = tuple(args.get("models", ("1cmt", "2cmt")))
    out = fit_compartmental(records, id_col=id_col, time_col=time_col, dv_col=dv_col,
                            dose_by_subject=dose_by_subject, models=models)

    compact = _compact(out, steady_state=False)
    c = compact["model_selection_counts"]
    return ToolResult(
        summary=(f"Compartmental fit: {compact['n_converged']}/{compact['n_subjects']} "
                 f"subjects converged ({c.get('1cmt',0)}×1-cmt, {c.get('2cmt',0)}×2-cmt, by AIC)."),
        action=f"fit_compartmental({dsid})",
        writes={"compartmental_results": compact}, result=compact)


TOOLS = [
    Tool("fit_compartmental",
         "Fit 1- and 2-compartment oral PK models per subject by least squares on "
         "log-concentration, select the best by AIC, and report ka/CL/V parameters.",
         "compartmental",
         {"type": "object",
          "properties": {"dataset_id": {"type": "string"},
                         "models": {"type": "array", "items": {"type": "string"}}},
          "required": []},
         fit_compartmental_models),
]
