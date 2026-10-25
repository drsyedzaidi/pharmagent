"""NCA Agent tools: compute non-compartmental parameters."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from app.compute.dosing import extract_ss_intervals_checked, is_multiple_dose, parse_blq_flag
from app.compute.nca import _round_sig, run_nca
from app.compute.nca_ss import run_nca_ss
from app.conventions.runtime import applied_conventions
from app.core.pharmstate import PharmState
from app.core.schema_extractor import detect_roles
from app.tools.base import Tool, ToolContext, ToolResult, require_dataset

# Catalog conventions each NCA path applies (app/conventions/catalog.json); named in results, no number depends on them.
_CONV_SD = ["nca.linear_up_log_down", "nca.auc_last_ends_at_tlast", "nca.pre_dose_records", "nca.lambda_z_window"]
_CONV_SS = ["nca.linear_up_log_down", "nca.lambda_z_window", "nca.ss_final_interval_scope", "nca.ss_endpoint_closure",
            "nca.ss_incomplete_interval_fail_closed"]
_CONV_BLQ = ["nca.blq_leading_middle_trailing", "nca.measured_zero", "nca.blq_flag_codes"]


def _roles(df: pd.DataFrame, state: PharmState) -> dict[str, str]:
    if state.dataset_metadata and state.dataset_metadata.get("detected_roles"):
        return state.dataset_metadata["detected_roles"]
    return detect_roles(list(df.columns))


def _blq_flags(values: pd.Series, col: str) -> pd.Series:
    """The below-LOQ flag column as booleans (``parse_blq_flag``). Unrecognised values stop the analysis: an unknown
    code must not quietly mean "not BLQ" while the result says M1 was applied."""
    bad: set[str] = set()

    def parse(v: Any) -> bool:
        try:
            return parse_blq_flag(v)
        except ValueError:
            bad.add(str(v))
            return False

    flags = values.map(parse).astype(bool)
    if bad:
        raise ValueError(f"below-LOQ flag column {col!r} has unrecognised value(s) "
                         f"{', '.join(repr(b) for b in sorted(bad))}: use 1/0, Y/N, YES/NO or TRUE/FALSE "
                         "(blank = not flagged)")
    return flags


def _from_dose_time(obs: pd.DataFrame, id_col: str, time_col: str, dv_col: str, cens_col: str | None,
                    dose_times: dict[Any, float]) -> pd.DataFrame:
    """Records before the dose add no area. Per subject (dose time = its first dose record, else TIME 0), rows before
    the dose are removed and remaining times are expressed as time since that dose (t - t_dose). When no usable record
    (BLQ-flagged, or a finite DV) is AT the dose time, the pre-dose record(s) closest to it supply C = 0 there if they
    are BLQ-flagged or a measured 0 (no drug before the dose); a positive pre-dose value is not carried to the dose
    time."""
    flagged = (obs[cens_col].astype(bool) if cens_col is not None
               else pd.Series(False, index=obs.index))
    usable = flagged | np.isfinite(obs[dv_col].to_numpy(dtype=float))
    parts = []
    for sid, g in obs.groupby(id_col, sort=False):
        td = dose_times.get(sid, 0.0)
        before = g[time_col] < td
        post = g[~before].copy()
        pre = g[before & usable.loc[g.index]]
        if len(pre) and not (usable.loc[post.index] & (post[time_col] == td)).any():
            latest = pre[pre[time_col] == pre[time_col].max()]
            if (flagged.loc[latest.index] | (latest[dv_col] == 0)).all():
                post = pd.concat([latest.iloc[[0]].assign(**{time_col: td, dv_col: 0.0}), post])
        if td != 0.0:
            post[time_col] = post[time_col] - td
        parts.append(post)
    return pd.concat(parts) if parts else obs


def _apply_m1(obs: pd.DataFrame, id_col: str, time_col: str, dv_col: str, cens_col: str) -> pd.DataFrame:
    """Rule M1 for observations flagged below-LOQ (``parse_blq_flag``), per subject in time order: BLQ rows BEFORE the
    first quantifiable value are a 0 baseline (their value is set to 0, also when it was blank); BLQ rows after it, in
    the middle or trailing, are excluded. Quantifiable means unflagged with a finite numeric TIME and a finite,
    POSITIVE DV: a row without a time or a DV carries no usable measurement, and an unflagged 0 is a measured value
    that is kept as an observed zero but is not a quantifiable concentration, so neither ends the baseline. A subject
    with nothing quantifiable keeps all its rows as zeros (as before)."""
    flag = obs[cens_col].map(parse_blq_flag).astype(bool)
    work = obs.assign(_blq=flag.to_numpy(), _t=pd.to_numeric(obs[time_col], errors="coerce"))
    work = work.sort_values([id_col, "_t"], kind="stable")
    parts = []
    for _, g in work.groupby(id_col, sort=False):
        blq = g["_blq"].to_numpy()
        dv = pd.to_numeric(g[dv_col], errors="coerce").to_numpy(dtype=float)
        measured = np.isfinite(dv) & np.isfinite(g["_t"].to_numpy(dtype=float))
        quantified = np.flatnonzero(~blq & measured & (dv > 0))
        first_q = int(quantified[0]) if quantified.size else len(g)
        leading = blq & (np.arange(len(g)) < first_q)
        g = g.copy()
        g.loc[g.index[leading], dv_col] = 0.0
        parts.append(g[((~blq) & measured) | leading])
    return pd.concat(parts).drop(columns=["_blq", "_t"]) if parts else obs


PRE_DOSE_RULE = ("Records before the dose (the subject's first dose record, else TIME 0) add no area: with no sample "
                 "at the dose time, a BLQ-flagged or zero pre-dose record closest to it sets C = 0 there; other "
                 "pre-dose records are not used. Times are reported relative to that dose time.")


def _ss_endpoint_rule(ctau_assumed: list[str], c0_assumed: list[str]) -> str:
    """How the steady-state interval ends were formed, naming every subject whose end was assumed."""
    text = (" Each interval end is the sample closest to it (C(0): from 2% of tau before the dose to 0.001 h after "
            "it; C(tau): within 2% of tau; ties go to the earlier sample); nothing before the dose or after tau is "
            "integrated. Where only one end has a sample, steady state is assumed for the other end alone and the "
            "assumed segment is integrated by linear-up/log-down.")
    if ctau_assumed:
        text += f" C(tau) = C(0) was assumed to close the interval for: {', '.join(ctau_assumed)}."
    if c0_assumed:
        text += f" C(0) = C(tau) was assumed to open the interval for: {', '.join(c0_assumed)}."
    if not (ctau_assumed or c0_assumed):
        text += " Both ends of every interval are samples."
    return text


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
    if cens_col is not None:
        df = df.assign(**{cens_col: _blq_flags(df[cens_col], cens_col)})

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
    if amt_col and is_multiple_dose(
            records_all, time_col=time_col, amt_col=amt_col, ii_col=ii_col,
            addl_col=addl_col, id_col=id_col):
        profiles, skipped = extract_ss_intervals_checked(
            records_all, id_col=id_col, time_col=time_col, dv_col=dv_col,
            amt_col=amt_col, ii_col=ii_col, addl_col=addl_col, blq_col=cens_col, require_complete_interval=True)
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
            opened = sorted((str(s) for s, prof in profiles.items() if prof.c0_source == "ctau_assumed"))
            n = len(out["nca_parameters"])
            taus = {round(p.tau, 3) for p in profiles.values()}
            if evid_col:
                is_obs = pd.to_numeric(df[evid_col], errors="coerce").fillna(0) == 0
            else:
                is_obs = pd.to_numeric(df[amt_col], errors="coerce").fillna(0) <= 0
            obs_ss = df[is_obs]                      # BLQ records are counted on observation rows only
            if cens_col is not None:
                n_blq = int(obs_ss[cens_col].sum())
            else:
                n_blq = int((pd.to_numeric(obs_ss[dv_col], errors="coerce") <= 0).sum())
            out["nca_summary"]["route"] = "IV" if is_iv else "extravascular"
            out["nca_summary"]["conventions"] = applied_conventions(_CONV_SS + (_CONV_BLQ if cens_col else []))
            out["nca_summary"]["blq"] = {
                "n_below_loq": n_blq, "ctau_assumed_c0": assumed, "c0_assumed_ctau": opened, "rule": (
                    ("M1 applied within the final steady-state dosing interval: quantifiable = unflagged with a "
                     "finite, positive DV. BLQ-flagged records before that interval's first quantifiable value count "
                     "as a 0 baseline (an unflagged 0 is kept as a measured zero but does not end that phase); later "
                     "BLQ records (middle or trailing) are excluded from AUC_tau.") if cens_col is not None else
                    ("No BLQ flag in the data: concentrations are used as given in the steady-state interval "
                     "analysis (a 0 is a measured zero).")) + _ss_endpoint_rule(assumed, opened)}
            return ToolResult(
                summary=(f"Steady-state NCA: {n} subjects over a {sorted(taus)} h dosing "
                         f"interval (AUC_tau, Cmax/Cmin, Cavg, CL/F=Dose/AUC_tau, "
                         f"fluctuation, accumulation)."),
                action=f"compute_nca_ss({dsid})",
                writes={"nca_parameters": out["nca_parameters"], "nca_summary": out["nca_summary"]},
                result={"n_subjects": n, "steady_state": True, "nca_summary": out["nca_summary"]},
            )

    df[time_col] = pd.to_numeric(df[time_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    df[dv_col] = pd.to_numeric(df[dv_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if amt_col:
        df[amt_col] = pd.to_numeric(df[amt_col], errors="coerce").replace([np.inf, -np.inf], np.nan)

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

    # dose per subject (single-dose: first dosing amount) and its time (the first dose record's TIME)
    dose_by_subject: dict[Any, float] = {}
    if amt_col and len(dose_rows):
        for sid, g in dose_rows.groupby(id_col):
            amts = g[amt_col].dropna()
            if len(amts):
                dose_by_subject[sid] = float(amts.iloc[0])
    timed = dose_rows[np.isfinite(dose_rows[time_col].to_numpy(dtype=float))]
    dose_times = {sid: float(t) for sid, t in timed.groupby(id_col)[time_col].min().items()}

    # BLQ accounting (transparency): count below-LOQ observation records, by an
    # explicit CENS flag if present, else non-positive concentrations.
    if cens_col is not None:
        n_blq = int(obs[cens_col].sum())
    else:
        n_blq = int((obs[dv_col] <= 0).sum())

    obs = _from_dose_time(obs, id_col, time_col, dv_col, cens_col, dose_times)
    if cens_col is not None:
        obs = _apply_m1(obs, id_col, time_col, dv_col, cens_col)
    obs = obs.dropna(subset=[time_col, dv_col])
    records = obs[[id_col, time_col, dv_col]].to_dict("records")
    out = run_nca(records, id_col=id_col, time_col=time_col, dv_col=dv_col,
                  dose_by_subject=dose_by_subject, is_iv=is_iv)
    out["nca_summary"]["route"] = "IV" if is_iv else "extravascular"
    out["nca_summary"]["conventions"] = applied_conventions(_CONV_SD + (_CONV_BLQ if cens_col else []))
    vss_note = " reported (IV)." if is_iv else " not reported for extravascular data."
    out["nca_summary"]["blq"] = {
        "n_below_loq": n_blq,
        "rule": (("M1: quantifiable = unflagged with a finite, positive DV. BLQ-flagged records before the first "
                  "quantifiable value count as a 0 baseline; later BLQ records (middle or trailing) are excluded from "
                  "AUC and lambda_z. An unflagged 0 is kept as a measured zero but does not end the leading-BLQ "
                  f"phase. {PRE_DOSE_RULE} AUC_last ends at Tlast. Vss/MRT" + vss_note) if cens_col is not None else
                 ("No BLQ flag in the data: concentrations are used as given (a 0 is a measured zero), except that "
                  f"AUC_last ends at Tlast, the last positive concentration. {PRE_DOSE_RULE} Vss/MRT" + vss_note)),
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
        plot_c = [_round_sig(float(v), 6) for v in pos[dv_col].tolist()]

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
                fit_y.append(_round_sig(math.exp(intercept - lz * t), 6))

        lz_subjects.append({
            "id": str(sid),
            "x": plot_t, "y": plot_c,
            "lz_x": [round(v, 4) for v in lz_t],
            "lz_y": [_round_sig(v, 6) for v in lz_c],
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
          "properties": {"dataset_id": {"type": "string"}, "is_iv": {"type": "boolean"}},
          "required": []},
         compute_nca),
]
