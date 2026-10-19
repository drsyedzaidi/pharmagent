"""NCA Agent tools: compute non-compartmental parameters."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from app.compute.dosing import extract_ss_intervals_checked, is_multiple_dose
from app.compute.nca import run_nca
from app.compute.nca_ss import run_nca_ss
from app.core.pharmstate import PharmState
from app.core.schema_extractor import detect_roles
from app.tools.base import Tool, ToolContext, ToolResult, require_dataset


def _roles(df: pd.DataFrame, state: PharmState) -> dict[str, str]:
    if state.dataset_metadata and state.dataset_metadata.get("detected_roles"):
        return state.dataset_metadata["detected_roles"]
    return detect_roles(list(df.columns))


def _apply_m1(obs: pd.DataFrame, id_col: str, time_col: str, dv_col: str, cens_col: str) -> pd.DataFrame:
    """Rule M1 for observations flagged below-LOQ (CENS == 1), per subject in time order: BLQ rows BEFORE the first
    quantifiable value are a 0 baseline (their value is set to 0, also when it was blank); BLQ rows after it, in the
    middle or trailing, are excluded. Quantifiable means unflagged WITH a finite numeric TIME and DV: a row without
    either carries no usable measurement and does not end the baseline. A subject with nothing quantifiable keeps
    all its rows as zeros (as before)."""
    flag = pd.to_numeric(obs[cens_col], errors="coerce").fillna(0) == 1
    work = obs.assign(_blq=flag.to_numpy(), _t=pd.to_numeric(obs[time_col], errors="coerce"))
    work = work.sort_values([id_col, "_t"], kind="stable")
    parts = []
    for _, g in work.groupby(id_col, sort=False):
        blq = g["_blq"].to_numpy()
        measured = (np.isfinite(pd.to_numeric(g[dv_col], errors="coerce").to_numpy(dtype=float))
                    & np.isfinite(g["_t"].to_numpy(dtype=float)))
        quantified = np.flatnonzero(~blq & measured)
        first_q = int(quantified[0]) if quantified.size else len(g)
        leading = blq & (np.arange(len(g)) < first_q)
        g = g.copy()
        g.loc[g.index[leading], dv_col] = 0.0
        parts.append(g[(~blq) | leading])
    return pd.concat(parts).drop(columns=["_blq", "_t"]) if parts else obs


def compute_nca(state: PharmState, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    dsid, df = require_dataset(ctx, state, args)
    df = df.copy()
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
        profiles, skipped = extract_ss_intervals_checked(
            records_all, id_col=id_col, time_col=time_col, dv_col=dv_col,
            amt_col=amt_col, ii_col=ii_col, addl_col=addl_col, blq_col=cens_col)
        # fail closed: a repeated-dose dataset is never re-analysed as single-dose NCA, and no subject is dropped
        if skipped or not profiles:
            detail = "; ".join(f"{sid!r}: {why}" for sid, why in skipped.items()) or "no subject profiles"
            raise ValueError("steady-state NCA stopped: these subject profiles could not form a complete dosing "
                             f"interval: {detail}")
        if profiles:
            out = run_nca_ss(profiles)
            for p in out["nca_parameters"]:      # endpoint provenance: was C(0) or C(tau) assumed?
                prof = profiles[p["subject"]]
                p["c0_source"], p["ctau_source"] = prof.c0_source, prof.ctau_source
            assumed = sorted((str(s) for s, prof in profiles.items() if prof.ctau_source == "c0_assumed"))
            n = len(out["nca_parameters"])
            taus = {round(p.tau, 3) for p in profiles.values()}
            if evid_col:
                is_obs = pd.to_numeric(df[evid_col], errors="coerce").fillna(0) == 0
            else:
                is_obs = pd.to_numeric(df[amt_col], errors="coerce").fillna(0) <= 0
            obs_ss = df[is_obs]                      # BLQ records are counted on observation rows only
            if cens_col is not None:
                n_blq = int((pd.to_numeric(obs_ss[cens_col], errors="coerce").fillna(0) == 1).sum())
            else:
                n_blq = int((pd.to_numeric(obs_ss[dv_col], errors="coerce") <= 0).sum())
            out["nca_summary"]["route"] = "IV" if is_iv else "extravascular"
            out["nca_summary"]["blq"] = {"n_below_loq": n_blq, "ctau_assumed_c0": assumed, "rule": (
                ("M1 applied within the final steady-state dosing interval: BLQ-flagged records before that interval's "
                 "first quantifiable value count as a 0 baseline; later BLQ records (middle or trailing) are excluded "
                 "from AUC_tau. Where the tau sample was excluded the interval is closed with C(tau) = C(0) "
                 "(listed in ctau_assumed_c0).") if cens_col is not None else
                ("No BLQ flag in the data: concentrations are used as given in the steady-state interval analysis "
                 "(a 0 is a measured zero)."))}
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

    if cens_col is not None:
        obs = _apply_m1(obs, id_col, time_col, dv_col, cens_col)
    obs = obs.dropna(subset=[time_col, dv_col])
    records = obs[[id_col, time_col, dv_col]].to_dict("records")
    out = run_nca(records, id_col=id_col, time_col=time_col, dv_col=dv_col,
                  dose_by_subject=dose_by_subject, is_iv=is_iv)
    out["nca_summary"]["route"] = "IV" if is_iv else "extravascular"
    vss_note = " reported (IV)." if is_iv else " not reported for extravascular data."
    out["nca_summary"]["blq"] = {
        "n_below_loq": n_blq,
        "rule": (("M1: BLQ-flagged records before the first quantifiable value count as a 0 baseline; later BLQ "
                  "records (middle or trailing) are excluded from AUC and lambda_z. AUC_last ends at Tlast. Vss/MRT"
                  + vss_note) if cens_col is not None else
                 ("No BLQ flag in the data: concentrations are used as given (a 0 is a measured zero), except that "
                  "AUC_last ends at Tlast, the last positive concentration. Vss/MRT" + vss_note)),
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
