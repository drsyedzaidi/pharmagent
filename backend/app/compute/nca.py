"""Non-compartmental analysis — deterministic compute.

Pure functions, no agent/LLM dependencies, fully unit-testable. Implements the
FDA-recommended linear-up/log-down trapezoidal rule and best-fit terminal
slope (lambda_z) by maximizing adjusted R-squared.

Per-subject parameters returned:
    Cmax, Tmax, AUC_last, AUC_inf, lambda_z, t_half, CL_F, Vz_F, MRT, Vss,
    pct_AUC_extrap, lambda_z_n_points, lambda_z_r2_adj, lambda_z_intervals
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Profile:
    subject: Any
    time: np.ndarray
    conc: np.ndarray
    dose: float


def _auc_intervals(time: np.ndarray, conc: np.ndarray) -> tuple[float, float]:
    """Return (AUC, AUMC) over observed points via linear-up/log-down."""
    auc = 0.0
    aumc = 0.0
    for i in range(len(time) - 1):
        t1, t2 = time[i], time[i + 1]
        c1, c2 = conc[i], conc[i + 1]
        dt = t2 - t1
        if dt <= 0:
            continue
        if c2 < c1 and c1 > 0 and c2 > 0:  # log-down (declining phase)
            ln = math.log(c1 / c2)
            auc += dt * (c1 - c2) / ln
            aumc += dt * (t1 * c1 - t2 * c2) / ln + dt * dt * (c1 - c2) / (ln * ln)
        else:  # linear-up (or flat / zero)
            auc += dt * (c1 + c2) / 2.0
            aumc += dt * (t1 * c1 + t2 * c2) / 2.0
    return auc, aumc


def _best_lambda_z(time: np.ndarray, conc: np.ndarray, tmax: float) -> dict[str, Any]:
    """Best-fit terminal slope over points after Tmax, maximizing adj R^2.

    Tries every contiguous window of >=3 of the last points; requires a positive
    decline. Returns {} if no acceptable fit.
    """
    mask = (time > tmax) & (conc > 0)
    t = time[mask]
    c = conc[mask]
    if t.size < 3:
        return {}
    lnc = np.log(c)
    n = t.size
    best: dict[str, Any] = {}
    # window start index from 0..n-3, always extend to the last point
    for start in range(0, n - 2):
        tt = t[start:]
        ll = lnc[start:]
        k = tt.size
        if k < 3:
            continue
        slope, intercept = np.polyfit(tt, ll, 1)
        if slope >= 0:
            continue  # not declining
        pred = slope * tt + intercept
        ss_res = float(np.sum((ll - pred) ** 2))
        ss_tot = float(np.sum((ll - ll.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        adj = 1.0 - (1.0 - r2) * (k - 1) / (k - 2) if k > 2 else r2
        if not best or adj > best["lambda_z_r2_adj"]:
            best = {
                "lambda_z": float(-slope),
                "lambda_z_r2": round(r2, 6),
                "lambda_z_r2_adj": round(adj, 6),
                "lambda_z_n_points": int(k),
                "lambda_z_intervals": [float(tt[0]), float(tt[-1])],
                "lambda_z_intercept": float(intercept),
                "lambda_z_pts_t": tt.tolist(),
                "lambda_z_pts_c": np.exp(ll).tolist(),
            }
    return best


def refit_lambda_z_manual(times: list[float], concs: list[float]) -> dict[str, Any]:
    """Fit terminal slope on an exact caller-specified set of points.

    Raises ValueError with a user-readable message on bad inputs so the
    API endpoint can return it as a 400 without a traceback.
    """
    if len(times) < 3:
        raise ValueError("Select ≥ 3 points for λz estimation")
    t = np.array(times, dtype=float)
    c = np.array(concs, dtype=float)
    if np.any(c <= 0):
        raise ValueError("All selected concentrations must be > 0")
    lnc = np.log(c)
    slope, intercept = np.polyfit(t, lnc, 1)
    if slope >= 0:
        raise ValueError("Selected points do not show a declining trend (slope ≥ 0)")
    k = int(len(t))
    pred = slope * t + intercept
    ss_res = float(np.sum((lnc - pred) ** 2))
    ss_tot = float(np.sum((lnc - lnc.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    adj = 1.0 - (1.0 - r2) * (k - 1) / (k - 2) if k > 2 else r2
    lz = float(-slope)
    t_half = math.log(2) / lz
    t0, t1 = float(t.min()), float(t.max())
    fit_x = [round(t0 + (t1 - t0) * i / 29, 3) for i in range(30)]
    fit_y = [round(math.exp(float(intercept) + slope * tx), 6) for tx in fit_x]
    return {
        "lambda_z": round(lz, 6),
        "lambda_z_intercept": round(float(intercept), 6),
        "t_half": round(t_half, 4),
        "r2_adj": round(adj, 6),
        "n_pts": k,
        "lz_x": [round(float(v), 4) for v in times],
        "lz_y": [round(float(v), 6) for v in concs],
        "fit_x": fit_x,
        "fit_y": fit_y,
    }


def nca_subject(p: Profile, *, is_iv: bool = False) -> dict[str, Any]:
    """Compute NCA parameters for a single subject's profile.

    ``is_iv`` controls route-dependent parameters: MRT and Vss are only
    physiologically meaningful for intravenous data. For extravascular (oral)
    data the AUMC-derived MRT includes the mean absorption time, so Vss (= CL·MRT)
    would be biased high — it is therefore suppressed and MRT is flagged.
    """
    order = np.argsort(p.time)
    time = np.asarray(p.time, dtype=float)[order]
    conc = np.asarray(p.conc, dtype=float)[order]

    cmax = float(np.max(conc))
    tmax = float(time[int(np.argmax(conc))])
    auc_last, aumc_last = _auc_intervals(time, conc)

    # last measurable (positive) concentration
    pos = np.where(conc > 0)[0]
    t_last = float(time[pos[-1]]) if pos.size else float(time[-1])
    c_last = float(conc[pos[-1]]) if pos.size else 0.0

    out: dict[str, Any] = {
        "subject": p.subject,
        "dose": p.dose,
        "Cmax": round(cmax, 6),
        "Tmax": round(tmax, 6),
        "AUC_last": round(auc_last, 6),
        "Clast": round(c_last, 6),
        "Tlast": round(t_last, 6),
    }

    lz = _best_lambda_z(time, conc, tmax)
    if lz:
        lam = lz["lambda_z"]
        auc_inf = auc_last + c_last / lam
        aumc_inf = aumc_last + t_last * c_last / lam + c_last / (lam * lam)
        mrt = aumc_inf / auc_inf if auc_inf > 0 else float("nan")
        cl_f = p.dose / auc_inf if auc_inf > 0 else float("nan")
        vz_f = cl_f / lam if not math.isnan(cl_f) else float("nan")
        # Vss = CL·MRT is only valid for IV; MRT_oral includes absorption time.
        vss = cl_f * mrt if (is_iv and not math.isnan(cl_f)) else None
        out.update(
            {
                "lambda_z": round(lam, 6),
                "t_half": round(math.log(2) / lam, 6),
                "AUC_inf": round(auc_inf, 6),
                "pct_AUC_extrap": round(100.0 * (auc_inf - auc_last) / auc_inf, 4),
                "MRT": round(mrt, 6),
                "MRT_note": None if is_iv else "extravascular: MRT includes mean absorption time",
                "CL_F": round(cl_f, 6),
                "Vz_F": round(vz_f, 6),
                "Vss": round(vss, 6) if vss is not None else None,
                "route": "IV" if is_iv else "extravascular",
                "lambda_z_r2_adj": lz["lambda_z_r2_adj"],
                "lambda_z_n_points": lz["lambda_z_n_points"],
                "lambda_z_intervals": lz["lambda_z_intervals"],
                "lambda_z_intercept": lz["lambda_z_intercept"],
                "lambda_z_pts_t": lz["lambda_z_pts_t"],
                "lambda_z_pts_c": lz["lambda_z_pts_c"],
            }
        )
    else:
        out.update({k: None for k in (
            "lambda_z", "t_half", "AUC_inf", "pct_AUC_extrap", "MRT", "MRT_note",
            "CL_F", "Vz_F", "Vss", "route", "lambda_z_r2_adj", "lambda_z_n_points",
            "lambda_z_intervals", "lambda_z_intercept", "lambda_z_pts_t", "lambda_z_pts_c")})
    return out


def _geomean(vals: list[float]) -> float | None:
    vals = [v for v in vals if v is not None and v > 0]
    if not vals:
        return None
    return float(math.exp(sum(math.log(v) for v in vals) / len(vals)))


def _geocv(vals: list[float]) -> float | None:
    vals = [v for v in vals if v is not None and v > 0]
    if len(vals) < 2:
        return None
    logs = [math.log(v) for v in vals]
    sd = float(np.std(logs, ddof=1))
    return float(100.0 * math.sqrt(math.exp(sd * sd) - 1.0))


def summarize_by_dose(params: list[dict[str, Any]]) -> dict[str, Any]:
    """Dose-group geometric mean and geometric CV% for key parameters."""
    keys = ["Cmax", "AUC_last", "AUC_inf", "CL_F", "Vz_F"]
    groups: dict[float, list[dict]] = {}
    for row in params:
        groups.setdefault(row.get("dose"), []).append(row)

    summary = {"by_dose": [], "n_subjects": len(params)}
    for dose, rows in sorted(groups.items(), key=lambda kv: (kv[0] is None, kv[0])):
        entry: dict[str, Any] = {"dose": dose, "n": len(rows)}
        for k in keys:
            vals = [r.get(k) for r in rows]
            entry[f"{k}_geomean"] = _round(_geomean(vals))
            entry[f"{k}_geocv_pct"] = _round(_geocv(vals))
        thalf = [r.get("t_half") for r in rows if r.get("t_half") is not None]
        entry["t_half_median"] = _round(float(np.median(thalf))) if thalf else None
        summary["by_dose"].append(entry)
    return summary


def _round(x):
    return round(x, 4) if isinstance(x, (int, float)) else x


def run_nca(records: list[dict[str, Any]], *, id_col: str, time_col: str,
            dv_col: str, dose_by_subject: dict[Any, float],
            is_iv: bool = False) -> dict[str, Any]:
    """Top-level entry: build per-subject profiles and compute NCA + summary.

    `records` are plain dicts (already loaded). Observation rows only
    (EVID==0 / dosing rows excluded) should be passed in for `dv_col`.
    ``is_iv`` enables IV-only parameters (MRT/Vss); default extravascular.
    """
    by_subj: dict[Any, list[tuple[float, float]]] = {}
    for r in records:
        sid = r[id_col]
        t = r.get(time_col)
        dv = r.get(dv_col)
        if t is None or dv is None:
            continue
        by_subj.setdefault(sid, []).append((float(t), float(dv)))

    params = []
    for sid, pts in by_subj.items():
        pts.sort(key=lambda x: x[0])
        time = np.array([p[0] for p in pts])
        conc = np.array([p[1] for p in pts])
        dose = float(dose_by_subject.get(sid, float("nan")))
        params.append(nca_subject(Profile(subject=sid, time=time, conc=conc, dose=dose),
                                  is_iv=is_iv))

    params.sort(key=lambda r: str(r["subject"]))
    return {"nca_parameters": params, "nca_summary": summarize_by_dose(params)}
