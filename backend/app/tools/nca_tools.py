"""NCA Agent tools: compute non-compartmental parameters."""
from __future__ import annotations

import math
from typing import Any

import pandas as pd

from app.compute.dosing import extract_ss_intervals, is_multiple_dose
from app.compute.nca import run_nca
from app.compute.nca_ss import run_nca_ss
from app.core.pharmstate import PharmState
from app.core.schema_extractor import detect_roles
from app.tools.base import Tool, ToolContext, ToolResult


def _roles(df: pd.DataFrame, state: PharmState) -> dict[str, str]:
    if state.dataset_metadata and state.dataset_metadata.get("detected_roles"):
        return state.dataset_metadata["detected_roles"]
    return detect_roles(list(df.columns))


def compute_nca(state: PharmState, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    dsid = args.get("dataset_id") or state.dataset_id
    if not dsid or dsid not in ctx.dataset_store:
        raise ValueError("no dataset loaded — upload a CSV (or run the NCA workflow) first")
    df = ctx.dataset_store[dsid].copy()
    roles = _roles(df, state)
    id_col = next((c for c, r in roles.items() if r == "ID"), None)
    time_col = next((c for c, r in roles.items() if r == "TIME"), None)
    dv_col = next((c for c, r in roles.items() if r == "DV"), None)
    amt_col = next((c for c, r in roles.items() if r == "AMT"), None)
    evid_col = next((c for c, r in roles.items() if r == "EVID"), None)
    ii_col = next((c for c, r in roles.items() if r == "II"), None)
    addl_col = next((c for c, r in roles.items() if r == "ADDL"), None)
    cens_col = next((c for c, r in roles.items() if r == "CENS"), None)
    route_col = next((c for c, r in roles.items() if r == "ROUTE"), None)
    if not (id_col and time_col and dv_col):
        raise ValueError("dataset missing required ID/TIME/DV roles for NCA")

    # route: IV enables MRT/Vss; default extravascular. Explicit arg wins, else
    # infer from a ROUTE column (values like IV / bolus / infusion).
    if "is_iv" in args:
        is_iv = bool(args["is_iv"])
    elif route_col is not None:
        vals = df[route_col].astype(str).str.strip().str.lower()
        is_iv = bool(vals.isin({"iv", "bolus", "infusion", "i.v."}).any())
    else:
        is_iv = False

    # Steady-state branch: repeated dosing (ADDL / interdose interval) -> analyze
    # the last dosing interval with interval (tau) exposures.
    records_all = df.to_dict("records")
    if amt_col and ii_col and is_multiple_dose(
            records_all, time_col=time_col, amt_col=amt_col, ii_col=ii_col,
            addl_col=addl_col, id_col=id_col):
        profiles = extract_ss_intervals(
            records_all, id_col=id_col, time_col=time_col, dv_col=dv_col,
            amt_col=amt_col, ii_col=ii_col, addl_col=addl_col)
        if profiles:
            out = run_nca_ss(profiles)
            n = len(out["nca_parameters"])
            taus = {round(p.tau, 3) for p in profiles.values()}
            dv_num = pd.to_numeric(df[dv_col], errors="coerce")
            if cens_col is not None:
                n_blq = int((pd.to_numeric(df[cens_col], errors="coerce").fillna(0) == 1).sum())
            else:
                n_blq = int((dv_num <= 0).sum())
            out["nca_summary"]["route"] = "IV" if is_iv else "extravascular"
            out["nca_summary"]["blq"] = {"n_below_loq": n_blq,
                                         "rule": "M1: BLQ excluded; steady-state interval analysis."}
            return ToolResult(
                summary=(f"Steady-state NCA: {n} subjects over a {sorted(taus)} h dosing "
                         f"interval (AUC_tau, Cmax/Cmin, Cavg, CL/F=Dose/AUC_tau, "
                         f"fluctuation, accumulation)."),
                action=f"compute_nca_ss({dsid})",
                writes={"nca_parameters": out["nca_parameters"], "nca_summary": out["nca_summary"]},
                result={"n_subjects": n, "steady_state": True, "nca_summary": out["nca_summary"]},
            )

    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df[dv_col] = pd.to_numeric(df[dv_col], errors="coerce")
    if amt_col:
        df[amt_col] = pd.to_numeric(df[amt_col], errors="coerce")

    # split dosing vs observation rows
    if evid_col:
        ev = pd.to_numeric(df[evid_col], errors="coerce").fillna(0)
        dose_rows = df[ev == 1]
        obs = df[ev == 0]
    elif amt_col:
        dose_rows = df[df[amt_col].fillna(0) > 0]
        obs = df[(df[amt_col].fillna(0) == 0)]
    else:
        dose_rows = df.iloc[0:0]
        obs = df

    # dose per subject (single-dose: first dosing amount)
    dose_by_subject: dict[Any, float] = {}
    if amt_col and len(dose_rows):
        for sid, g in dose_rows.groupby(id_col):
            amts = g[amt_col].dropna()
            if len(amts):
                dose_by_subject[sid] = float(amts.iloc[0])

    # BLQ accounting (transparency): count below-LOQ observation records, by an
    # explicit CENS flag if present, else non-positive concentrations.
    if cens_col is not None:
        n_blq = int((pd.to_numeric(obs[cens_col], errors="coerce").fillna(0) == 1).sum())
    else:
        n_blq = int((obs[dv_col] <= 0).sum())

    obs = obs.dropna(subset=[time_col, dv_col])
    records = obs[[id_col, time_col, dv_col]].to_dict("records")
    out = run_nca(records, id_col=id_col, time_col=time_col, dv_col=dv_col,
                  dose_by_subject=dose_by_subject, is_iv=is_iv)
    out["nca_summary"]["route"] = "IV" if is_iv else "extravascular"
    out["nca_summary"]["blq"] = {
        "n_below_loq": n_blq,
        "rule": ("M1: BLQ records excluded from lambda_z and AUC (leading BLQ count "
                 "toward 0 baseline). Vss/MRT" + (" reported (IV)." if is_iv
                 else " not reported for extravascular data.")),
    }

    n = len(out["nca_parameters"])

    # Build per-subject plot data: all obs + lambda_z regression points + fit line
    lz_subjects = []
    for param in out["nca_parameters"]:
        sid = param["subject"]
        sub = obs[obs[id_col] == sid].copy()
        sub[time_col] = pd.to_numeric(sub[time_col], errors="coerce")
        sub[dv_col] = pd.to_numeric(sub[dv_col], errors="coerce")
        sub = sub.sort_values(time_col)
        pos = sub[sub[dv_col] > 0]
        plot_t = [round(float(v), 4) for v in pos[time_col].tolist()]
        plot_c = [round(float(v), 6) for v in pos[dv_col].tolist()]

        lz_t: list = param.get("lambda_z_pts_t") or []
        lz_c: list = param.get("lambda_z_pts_c") or []
        lz = param.get("lambda_z")
        intercept = param.get("lambda_z_intercept")

        fit_x: list = []
        fit_y: list = []
        if lz and intercept is not None and lz_t and plot_t:
            t0, t1 = lz_t[0], max(plot_t)
            for i in range(30):
                t = t0 + (t1 - t0) * i / 29
                fit_x.append(round(t, 3))
                fit_y.append(round(math.exp(intercept - lz * t), 6))

        lz_subjects.append({
            "id": str(sid),
            "x": plot_t, "y": plot_c,
            "lz_x": [round(v, 4) for v in lz_t],
            "lz_y": [round(v, 6) for v in lz_c],
            "fit_x": fit_x, "fit_y": fit_y,
            "lambda_z": param.get("lambda_z"),
            "t_half": param.get("t_half"),
            "r2_adj": param.get("lambda_z_r2_adj"),
            "n_pts": param.get("lambda_z_n_points"),
            "Tmax": param.get("Tmax"),
        })

    nca_plot_data = {"subjects": lz_subjects}

    return ToolResult(
        summary=f"NCA complete ({'IV' if is_iv else 'extravascular'}): {n} subjects, "
                f"{len(out['nca_summary']['by_dose'])} dose group(s), {n_blq} BLQ "
                f"(linear-up/log-down, best-fit lambda_z).",
        action=f"compute_nca({dsid})",
        writes={"nca_parameters": out["nca_parameters"], "nca_summary": out["nca_summary"],
                "nca_plot_data": nca_plot_data},
        result={"n_subjects": n, "nca_summary": out["nca_summary"]},  # derived stats only
    )


TOOLS = [
    Tool("compute_nca",
         "Compute non-compartmental PK parameters (Cmax, Tmax, AUClast, AUCinf, "
         "t1/2, CL/F, Vz/F, MRT, Vss) per subject using linear-up/log-down and "
         "best-fit terminal slope.",
         "nca",
         {"type": "object",
          "properties": {"dataset_id": {"type": "string"}}, "required": []},
         compute_nca),
]
