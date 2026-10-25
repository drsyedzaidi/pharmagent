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
    t_inf: float = 0.0     # zero-order infusion duration (dose / rate); 0 = bolus. Only used for IV MRT/Vss


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


# Acceptance criteria for a terminal fit feeding AUC_inf-derived summaries (F-16). The picker itself only needs
# adj R^2 > 0, so a fit can be accepted by it and still be too poor to pool.
LZ_MIN_ADJ_R2 = 0.80
LZ_MIN_POINTS = 3
LZ_MIN_SPAN_RATIO = 2.0       # (t_last - t_first of the fit window) / t_half
LZ_MAX_PCT_EXTRAP = 20.0
INF_DERIVED = ("AUC_inf", "CL_F", "Vz_F", "Vss", "MRT")   # values that depend on the extrapolated tail

_MIN_LAMBDA_Z = 1e-9  # reject floating-point noise on flat terminal phases


def _round_sig(x: Any, ndigits: int = 6):
    """Round to ``ndigits`` decimal places when |x| >= 1, or ``ndigits`` significant
    figures when 0 < |x| < 1 so small concentration units never lose precision or
    round positive values to zero."""
    if not isinstance(x, (int, float)) or isinstance(x, bool):
        return x
    fx = float(x)
    if not math.isfinite(fx) or fx == 0.0:
        return 0.0 if fx == 0.0 else fx
    decimals = max(ndigits, ndigits - 1 - int(math.floor(math.log10(abs(fx)))))
    return round(fx, decimals)


def _best_lambda_z(time: np.ndarray, conc: np.ndarray, tmax: float) -> dict[str, Any]:
    """Best-fit terminal slope over points after Tmax, maximizing adj R^2.

    Tries every contiguous window of >=3 of the last points; requires a materially
    negative slope and positive adjusted R^2. Returns {} if no acceptable fit.
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
        if -slope <= _MIN_LAMBDA_Z:
            continue  # not materially declining
        pred = slope * tt + intercept
        ss_res = float(np.sum((ll - pred) ** 2))
        ss_tot = float(np.sum((ll - ll.mean()) ** 2))
        if ss_tot <= 0:
            continue
        r2 = 1.0 - ss_res / ss_tot
        adj = 1.0 - (1.0 - r2) * (k - 1) / (k - 2) if k > 2 else r2
        if adj <= 0:
            continue  # meaningless terminal fit
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
    if -slope <= _MIN_LAMBDA_Z:
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
    fit_y = [_round_sig(math.exp(float(intercept) + slope * tx), 6) for tx in fit_x]
    return {
        "lambda_z": _round_sig(lz, 6),
        "lambda_z_intercept": _round_sig(float(intercept), 6),
        "t_half": _round_sig(t_half, 4),
        "r2_adj": round(adj, 6),
        "n_pts": k,
        "lz_x": [round(float(v), 4) for v in times],
        "lz_y": [_round_sig(float(v), 6) for v in concs],
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
    raw_t = np.asarray(p.time, dtype=float)
    raw_c = np.asarray(p.conc, dtype=float)
    finite = np.isfinite(raw_t) & np.isfinite(raw_c)
    raw_t, raw_c = raw_t[finite], raw_c[finite]
    order = np.argsort(raw_t)
    time = raw_t[order]
    conc = raw_c[order]

    cmax = float(np.max(conc))
    tmax = float(time[int(np.argmax(conc))])

    # last measurable (positive) concentration. AUC_last is the area from the first sample to Tlast and NOT beyond it:
    # trailing zeros (below-LOQ values coded as 0) must not add the triangle down to the next zero row, which would
    # also make AUC_inf = AUC_last + Clast/lambda_z count the tail twice. A profile with nothing measurable keeps its
    # old reporting (area over the rows given, which is 0).
    pos = np.where(conc > 0)[0]
    t_last = float(time[pos[-1]]) if pos.size else float(time[-1])
    c_last = float(conc[pos[-1]]) if pos.size else 0.0
    end = int(pos[-1]) + 1 if pos.size else len(time)
    auc_last, aumc_last = _auc_intervals(time[:end], conc[:end])

    out: dict[str, Any] = {
        "subject": p.subject,
        "dose": p.dose,
        "Cmax": _round_sig(cmax, 6),
        "Tmax": _round_sig(tmax, 6),
        "AUC_last": _round_sig(auc_last, 6),
        "Clast": _round_sig(c_last, 6),
        "Tlast": _round_sig(t_last, 6),
    }

    lz = _best_lambda_z(time, conc, tmax)
    if lz:
        lam = lz["lambda_z"]
        auc_inf = auc_last + c_last / lam
        aumc_inf = aumc_last + t_last * c_last / lam + c_last / (lam * lam)
        mrt = aumc_inf / auc_inf if auc_inf > 0 else float("nan")
        if is_iv and p.t_inf > 0 and not math.isnan(mrt):
            mrt -= p.t_inf / 2.0       # MRT_iv = AUMC/AUC - T_inf/2 (infusion adds T_inf/2 to the apparent MRT)
        cl_f = p.dose / auc_inf if auc_inf > 0 else float("nan")
        vz_f = cl_f / lam if not math.isnan(cl_f) else float("nan")
        # Vss = CL·MRT is only valid for IV; MRT_oral includes absorption time.
        vss = cl_f * mrt if (is_iv and not math.isnan(cl_f)) else None
        out.update(
            {
                "lambda_z": _round_sig(lam, 6),
                "t_half": _round_sig(math.log(2) / lam, 6),
                "AUC_inf": _round_sig(auc_inf, 6),
                "pct_AUC_extrap": round(100.0 * (auc_inf - auc_last) / auc_inf, 4),
                "MRT": _round_sig(mrt, 6),
                "MRT_note": None if is_iv else "extravascular: MRT includes mean absorption time",
                "CL_F": _round_sig(cl_f, 6),
                "Vz_F": _round_sig(vz_f, 6),
                "Vss": _round_sig(vss, 6) if vss is not None else None,
                "route": "IV" if is_iv else "extravascular",
                "lambda_z_r2_adj": lz["lambda_z_r2_adj"],
                "lambda_z_n_points": lz["lambda_z_n_points"],
                "lambda_z_intervals": lz["lambda_z_intervals"],
                "lambda_z_intercept": lz["lambda_z_intercept"],
                "lambda_z_pts_t": lz["lambda_z_pts_t"],
                "lambda_z_pts_c": lz["lambda_z_pts_c"],
            }
        )
        out["lambda_z_acceptable"] = lambda_z_acceptable(out)
    else:
        out.update({k: None for k in (
            "lambda_z", "t_half", "AUC_inf", "pct_AUC_extrap", "MRT", "MRT_note",
            "CL_F", "Vz_F", "Vss", "route", "lambda_z_r2_adj", "lambda_z_n_points",
            "lambda_z_intervals", "lambda_z_intercept", "lambda_z_pts_t", "lambda_z_pts_c")})
        out["lambda_z_acceptable"] = None
    return out


def lambda_z_acceptable(out: dict[str, Any]) -> bool | None:
    """Is the terminal fit good enough for AUC_inf-derived summaries? None = no fit to judge.

    adj R^2 >= LZ_MIN_ADJ_R2, >= LZ_MIN_POINTS points, span ratio >= LZ_MIN_SPAN_RATIO (taken from
    ``lambda_z_span_ratio`` when the row carries it, else computed from the fit window) and
    %AUC_extrap <= LZ_MAX_PCT_EXTRAP."""
    lam, r2 = out.get("lambda_z"), out.get("lambda_z_r2_adj")
    if lam is None or r2 is None:
        return None
    span = out.get("lambda_z_span_ratio")
    window = out.get("lambda_z_intervals")
    if span is None and window and lam > 0:
        span = (window[1] - window[0]) * lam / math.log(2)
    extrap = out.get("pct_AUC_extrap")
    return bool(r2 >= LZ_MIN_ADJ_R2
                and (out.get("lambda_z_n_points") or 0) >= LZ_MIN_POINTS
                and (span is None or span >= LZ_MIN_SPAN_RATIO)
                and (extrap is None or extrap <= LZ_MAX_PCT_EXTRAP))


def apply_lambda_z_quality(rows: list[dict[str, Any]], include_unacceptable: bool = False
                           ) -> tuple[list[dict[str, Any]], list[Any]]:
    """Copies of ``rows`` with AUC_inf-derived values blanked where ``lambda_z_acceptable`` is False, plus the ids
    of those subjects. Rows without the flag (steady-state NCA, older results) are left as they are."""
    if include_unacceptable:
        return rows, []
    out, excluded = [], []
    for r in rows:
        if r.get("lambda_z_acceptable") is False:
            excluded.append(r.get("subject"))
            r = {**r, **{k: None for k in INF_DERIVED}}
        out.append(r)
    return out, excluded


def lambda_z_exclusion_note(excluded: list[Any], include_unacceptable: bool) -> dict[str, Any]:
    return {"n_subjects": len(excluded), "subjects": excluded, "include_unacceptable": include_unacceptable,
            "rule": (f"AUC_inf-derived values ({', '.join(INF_DERIVED)}) are excluded for subjects whose terminal "
                     f"fit has adj R^2 < {LZ_MIN_ADJ_R2}, < {LZ_MIN_POINTS} points, span ratio < "
                     f"{LZ_MIN_SPAN_RATIO} or %AUC_extrap > {LZ_MAX_PCT_EXTRAP}; pass "
                     "include_unacceptable_lambda_z=true to pool them anyway.")}


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


# Canonical order for the descriptive-statistics table (single dose, then
# steady-state, then derived). Only parameters present in the data are shown.
DESCRIPTIVE_PARAMS: tuple[str, ...] = (
    "Cmax", "Tmax", "Cmin", "Ctrough", "Cavg", "Clast", "Tlast",
    "AUC_last", "AUC_inf", "AUC_tau", "pct_AUC_extrap",
    "lambda_z", "t_half", "MRT", "CL_F", "Vz_F", "Vss",
    "accumulation_ratio", "fluctuation_pct", "swing_pct",
)


def _describe(vals: list[float]) -> dict[str, Any]:
    """N, mean, SD (n-1), CV%, median, min, max, geometric mean and geo-CV%
    (log-scale SD -> sqrt(exp(s^2)-1)); geometric stats use positive values only."""
    v = np.asarray([float(x) for x in vals], dtype=float)
    n = int(v.size)
    if n == 0:
        return {"n": 0, "mean": None, "sd": None, "cv_pct": None, "median": None,
                "min": None, "max": None, "geomean": None, "geocv_pct": None}
    mean = float(v.mean())
    sd = float(v.std(ddof=1)) if n > 1 else None
    cv = 100.0 * sd / mean if (sd is not None and mean != 0) else None
    pos = v[v > 0]
    geomean = float(np.exp(np.log(pos).mean())) if pos.size else None
    geocv = (100.0 * math.sqrt(math.exp(float(np.log(pos).var(ddof=1))) - 1.0)
             if pos.size > 1 else None)
    return {"n": n, "mean": _round(mean), "sd": _round(sd), "cv_pct": _round(cv),
            "median": _round(float(np.median(v))), "min": _round(float(v.min())),
            "max": _round(float(v.max())), "geomean": _round(geomean), "geocv_pct": _round(geocv)}


def descriptive_stats(params: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Descriptive statistics per parameter: an ``"all"`` group always, plus one
    group per dose level when there is more than one. Missing/non-numeric
    values are excluded from that parameter's N."""
    def group_block(group: Any, label: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
        out: list[dict[str, Any]] = []
        for k in DESCRIPTIVE_PARAMS:
            vals = [r[k] for r in rows
                    if isinstance(r.get(k), (int, float)) and not isinstance(r.get(k), bool)
                    and math.isfinite(float(r[k]))]
            if not vals:
                continue
            out.append({"parameter": k, **_describe(vals)})
        return {"group": group, "label": label, "n": len(rows), "parameters": out}

    groups: list[dict[str, Any]] = [group_block("all", "All subjects", params)]
    by_dose: dict[Any, list[dict[str, Any]]] = {}
    for r in params:
        by_dose.setdefault(r.get("dose"), []).append(r)
    if len(by_dose) > 1:
        for dose, rows in sorted(by_dose.items(), key=lambda kv: (kv[0] is None, kv[0])):
            groups.append(group_block(dose, f"Dose {dose}" if dose is not None else "Dose n/a", rows))
    return groups


def summarize_by_dose(params: list[dict[str, Any]], *,
                      include_unacceptable_lambda_z: bool = False) -> dict[str, Any]:
    """Dose-group geometric mean and geometric CV% for key parameters, plus the
    full descriptive-statistics table (``descriptive``). AUC_inf-derived values of subjects with an unacceptable
    terminal fit are left out unless ``include_unacceptable_lambda_z`` (the exclusion is reported)."""
    params, excluded = apply_lambda_z_quality(params, include_unacceptable_lambda_z)
    keys = ["Cmax", "AUC_last", "AUC_inf", "CL_F", "Vz_F"]
    groups: dict[float, list[dict]] = {}
    for row in params:
        groups.setdefault(row.get("dose"), []).append(row)

    summary = {"by_dose": [], "n_subjects": len(params)}
    for dose, rows in sorted(groups.items(), key=lambda kv: (kv[0] is None, kv[0])):
        entry: dict[str, Any] = {"dose": dose, "n": len(rows),
                                 "n_lambda_z_excluded": sum(1 for r in rows if r.get("subject") in excluded)}
        for k in keys:
            vals = [r.get(k) for r in rows]
            entry[f"{k}_geomean"] = _round(_geomean(vals))
            entry[f"{k}_geocv_pct"] = _round(_geocv(vals))
        thalf = [r.get("t_half") for r in rows if r.get("t_half") is not None]
        entry["t_half_median"] = _round(float(np.median(thalf))) if thalf else None
        summary["by_dose"].append(entry)
    summary["descriptive"] = descriptive_stats(params)
    summary["lambda_z_excluded"] = lambda_z_exclusion_note(excluded, include_unacceptable_lambda_z)
    return summary


def _round(x):
    return _round_sig(x, 4)


def run_nca(records: list[dict[str, Any]], *, id_col: str, time_col: str,
            dv_col: str, dose_by_subject: dict[Any, float],
            dose_time_by_subject: dict[Any, float] | None = None,
            t_inf_by_subject: dict[Any, float] | None = None,
            is_iv: bool = False, include_unacceptable_lambda_z: bool = False) -> dict[str, Any]:
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
        ft, fdv = float(t), float(dv)
        if not (math.isfinite(ft) and math.isfinite(fdv)):
            continue
        t_dose = float((dose_time_by_subject or {}).get(sid, 0.0))
        by_subj.setdefault(sid, []).append((ft - t_dose, fdv))

    params = []
    for sid, pts in by_subj.items():
        pts.sort(key=lambda x: x[0])
        time = np.array([p[0] for p in pts])
        conc = np.array([p[1] for p in pts])
        dose = float(dose_by_subject.get(sid, float("nan")))
        params.append(nca_subject(Profile(subject=sid, time=time, conc=conc, dose=dose,
                                          t_inf=float((t_inf_by_subject or {}).get(sid, 0.0))),
                                  is_iv=is_iv))

    params.sort(key=lambda r: str(r["subject"]))
    return {"nca_parameters": params, "nca_summary": summarize_by_dose(
        params, include_unacceptable_lambda_z=include_unacceptable_lambda_z)}
