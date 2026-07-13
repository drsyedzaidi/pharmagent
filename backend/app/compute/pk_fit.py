"""Fit structural PK models to data and summarize a population (two-stage).

For each subject the chosen model is fit to the concentration-time data by
least squares on the log scale (proportional error), simulating the actual
dosing schedule (single- or multiple-dose). Individual estimates are then
summarized into typical values (geometric mean) and between-subject
variability (IIV, geometric CV%) — a two-stage approximation, NOT NLME.

``compare_models`` fits several models and ranks them by total AIC.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from app.compute.pk_models import PKModel, get_model
from app.compute.pk_simulate import simulate
from app.compute.poppk import two_stage_summary

_EPS = 1e-12


def rse_from_jacobian(sol, ssr: float, n_obs: int, names: tuple[str, ...]) -> dict | None:
    """Asymptotic relative standard errors (%) from a least_squares solution.

    Parameters are fit on the log scale, so cov = sigma^2 (JᵀJ)^-1 gives the
    variance of log-params; by the delta method RSE%(theta) = 100·SE(log theta).
    Returns None if the information matrix is singular / ill-conditioned.
    """
    k = len(names)
    dof = n_obs - k
    if dof < 1:
        return None
    try:
        J = np.asarray(sol.jac, dtype=float)
        sigma2 = ssr / dof
        cov = sigma2 * np.linalg.pinv(J.T @ J)
        se_log = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        if not np.all(np.isfinite(se_log)):
            return None
        return {n: round(100.0 * float(se_log[i]), 2) for i, n in enumerate(names)}
    except Exception:
        return None


def _terminal_ke(t: np.ndarray, c: np.ndarray) -> float:
    pos = c > 0
    tt, cc = t[pos], c[pos]
    if tt.size >= 2:
        n = min(3, tt.size)
        slope = np.polyfit(tt[-n:], np.log(cc[-n:]), 1)[0]
        if slope < 0 and math.isfinite(slope):
            return float(-slope)
    return 0.1


def _init_guess(model: PKModel, dose: float, t: np.ndarray, c: np.ndarray) -> dict[str, float]:
    """Data-informed starting values; model defaults for the rest."""
    p = dict(model.defaults)
    cmax = float(np.max(c)) if c.size else 1.0
    ke = _terminal_ke(t, c)
    vkey = "V" if "V" in p else ("VC" if "VC" in p else None)
    if vkey and cmax > 0:
        p[vkey] = max(dose / cmax, 1e-3)
    if "CL" in p and vkey:
        p["CL"] = max(ke * p[vkey], 1e-3)
    return p


def fit_subject_model(model_key: str, doses: list[dict], obs_t, obs_c,
                      *, wt: float = 70.0) -> dict[str, Any]:
    """Fit one subject's PK profile to a structural model."""
    model = get_model(model_key)
    names = model.params
    t = np.asarray(obs_t, dtype=float)
    c = np.asarray(obs_c, dtype=float)
    order = np.argsort(t)
    t, c = t[order], c[order]
    mask = c > 0
    t, c = t[mask], c[mask]
    n_obs = int(t.size)

    failed = {"subject": None, "model": model_key, "converged": False,
              "params": {}, "aic": None, "r_squared": None, "n_obs": n_obs}
    if model.has_pd:
        failed["note"] = "PK/PD model needs a PD endpoint; not fit on PK data alone."
        return failed
    if n_obs < len(names) + 1 or not doses:
        return failed

    dose_amt = float(doses[-1]["amt"])
    guess = _init_guess(model, dose_amt, t, c)
    theta0 = np.array([max(guess[n], _EPS) for n in names], dtype=float)
    log_obs = np.log(c)

    def residuals(log_theta: np.ndarray) -> np.ndarray:
        p = {n: math.exp(v) for n, v in zip(names, log_theta)}
        try:
            pred = simulate(model, p, doses, t, wt=wt)["cp"]
        except Exception:
            return np.full(n_obs, 1e3)
        pred = np.maximum(pred, _EPS)
        return np.log(pred) - log_obs

    try:
        sol = least_squares(residuals, np.log(theta0), method="lm", max_nfev=4000)
        theta = np.exp(sol.x)
        if not np.all(np.isfinite(theta)):
            return failed
        res = residuals(sol.x)
        ssr = float(np.sum(res ** 2))
        if not math.isfinite(ssr):
            return failed
        k = len(names)
        aic = n_obs * math.log(max(ssr, _EPS) / n_obs) + 2 * k
        ss_tot = float(np.sum((log_obs - log_obs.mean()) ** 2))
        r2 = 1.0 - ssr / ss_tot if ss_tot > 0 else 0.0
        return {"subject": None, "model": model_key, "converged": True,
                "params": {n: round(float(v), 6) for n, v in zip(names, theta)},
                "rse_pct": rse_from_jacobian(sol, ssr, n_obs, names),
                "aic": round(aic, 4), "r_squared": round(r2, 6), "n_obs": n_obs}
    except Exception:
        return failed


# Parameters bounded to (0,1) are fit via logit; the rest via log (positivity).
_LOGIT_PARAMS = {"IMAX"}


def _pack(names: tuple[str, ...], p: dict[str, float]) -> np.ndarray:
    out = []
    for n in names:
        v = p[n]
        if n in _LOGIT_PARAMS:
            v = min(max(v, 1e-6), 1 - 1e-6)
            out.append(math.log(v / (1 - v)))
        else:
            out.append(math.log(max(v, _EPS)))
    return np.array(out, dtype=float)


def _unpack(names: tuple[str, ...], theta: np.ndarray) -> dict[str, float]:
    p: dict[str, float] = {}
    for n, x in zip(names, theta):
        p[n] = (1.0 / (1.0 + math.exp(-x))) if n in _LOGIT_PARAMS else math.exp(x)
    return p


def _pkpd_init(model, dose: float, pk_t, pk_c, pd_t, pd_e) -> dict[str, float]:
    p = dict(model.defaults)
    cmax = float(np.max(pk_c)) if len(pk_c) else 1.0
    ke = _terminal_ke(np.asarray(pk_t, float), np.asarray(pk_c, float))
    if "V" in p and cmax > 0:
        p["V"] = max(dose / cmax, 1e-3)
    if "CL" in p and "V" in p:
        p["CL"] = max(ke * p["V"], 1e-3)
    base = float(np.median(pd_e)) if len(pd_e) else p.get("E0", 1.0)
    span = float(np.max(pd_e) - np.min(pd_e)) if len(pd_e) else 1.0
    if "E0" in p:
        p["E0"] = max(abs(pd_e[int(np.argmin(np.asarray(pd_t)))]) if len(pd_e) else base, 1e-3)
    if "EMAX" in p:
        p["EMAX"] = max(span, 1e-3)
    if "SLOPE" in p and cmax > 0:
        p["SLOPE"] = max(span / cmax, 1e-3)
    if "KIN" in p and "KOUT" in p:           # resp(0)=KIN/KOUT ~ baseline
        p["KOUT"] = 1.0
        p["KIN"] = max(base, 1e-3)
    return p


def fit_subject_pkpd(model_key: str, doses: list[dict], pk_t, pk_c, pd_t, pd_e,
                     *, wt: float = 70.0) -> dict[str, Any]:
    """Dual-endpoint fit: PK params to concentrations (log scale) + PD params to
    the effect observations (additive scale), jointly by least squares."""
    model = get_model(model_key)
    names = model.params
    pk_t = np.asarray(pk_t, float); pk_c = np.asarray(pk_c, float)
    pd_t = np.asarray(pd_t, float); pd_e = np.asarray(pd_e, float)
    # PK obs must be positive (log); PD obs may be any finite value
    mpk = pk_c > 0
    pk_t, pk_c = pk_t[mpk], pk_c[mpk]
    mpd = np.isfinite(pd_e)
    pd_t, pd_e = pd_t[mpd], pd_e[mpd]
    n_obs = int(pk_t.size + pd_t.size)

    failed = {"subject": None, "model": model_key, "converged": False,
              "params": {}, "aic": None, "r_squared": None, "n_obs": n_obs}
    if pk_t.size < 2 or pd_t.size < 2 or n_obs < len(names) + 1 or not doses:
        return failed

    # balance the two endpoints by their observation spread
    w_pk = 1.0 / (np.std(np.log(pk_c)) or 1.0)
    w_pd = 1.0 / (np.std(pd_e) or 1.0)
    log_pk = np.log(pk_c)
    all_t = sorted(set(np.round(np.concatenate([pk_t, pd_t]), 9).tolist()))
    idx = {t: i for i, t in enumerate(all_t)}
    pk_i = [idx[round(float(t), 9)] for t in pk_t]
    pd_i = [idx[round(float(t), 9)] for t in pd_t]

    dose_amt = float(doses[-1]["amt"])
    theta0 = _pack(names, _pkpd_init(model, dose_amt, pk_t, pk_c, pd_t, pd_e))

    def residuals(theta):
        p = _unpack(names, theta)
        try:
            sim = simulate(model, p, doses, all_t, wt=wt)
        except Exception:
            return np.full(n_obs, 1e3)
        cp = np.maximum(sim["cp"][pk_i], _EPS)
        eff = sim["eff"][pd_i]
        return np.concatenate([(np.log(cp) - log_pk) * w_pk, (eff - pd_e) * w_pd])

    try:
        sol = least_squares(residuals, theta0, method="lm", max_nfev=4000)
        p = _unpack(names, sol.x)
        res = residuals(sol.x)
        ssr = float(np.sum(res ** 2))
        if not math.isfinite(ssr) or not all(math.isfinite(v) for v in p.values()):
            return failed
        k = len(names)
        aic = n_obs * math.log(max(ssr, _EPS) / n_obs) + 2 * k
        return {"subject": None, "model": model_key, "converged": True,
                "params": {n: round(float(v), 6) for n, v in p.items()},
                "aic": round(aic, 4), "n_pk_obs": int(pk_t.size),
                "n_pd_obs": int(pd_t.size), "n_obs": n_obs, "r_squared": None}
    except Exception:
        return failed


def fit_pk_dataset(subjects: list[dict], *, model_key: str) -> dict[str, Any]:
    """Fit one model across all subjects; two-stage population summary.

    ``subjects``: [{"subject", "doses":[{time,amt}], "obs_t", "obs_c", "wt"}].
    """
    model = get_model(model_key)
    fits: list[dict[str, Any]] = []
    for s in subjects:
        if model.has_pd:
            pd_t = s.get("pd_t")
            if pd_t is None or len(pd_t) == 0:
                f = {"subject": s["subject"], "model": model_key, "converged": False,
                     "params": {}, "aic": None, "r_squared": None, "n_obs": 0,
                     "note": "no PD endpoint for this subject"}
            else:
                f = fit_subject_pkpd(model_key, s["doses"], s["obs_t"], s["obs_c"],
                                     s["pd_t"], s["pd_e"], wt=float(s.get("wt", 70.0)))
        else:
            f = fit_subject_model(model_key, s["doses"], s["obs_t"], s["obs_c"],
                                  wt=float(s.get("wt", 70.0)))
        f["subject"] = s["subject"]
        fits.append(f)
    fits.sort(key=lambda r: str(r["subject"]))

    converged = [f for f in fits if f["converged"] and f.get("params")]
    indiv = [{"subject": f["subject"], **f["params"]} for f in converged]
    population = two_stage_summary(indiv, keys=model.params) if indiv else {
        "method": "two-stage (STS)", "n_subjects": 0, "parameters": {}}
    aics = [f["aic"] for f in converged if f["aic"] is not None]
    return {
        "model_key": model_key, "label": model.label, "group": model.group,
        "n_subjects": len(subjects), "n_converged": len(converged),
        "individual_fits": fits, "population": population,
        "mean_aic": round(float(np.mean(aics)), 4) if aics else None,
        "total_aic": round(float(np.sum(aics)), 4) if aics else None,
    }


def compare_models(subjects: list[dict], model_keys: list[str]) -> dict[str, Any]:
    """Fit several models and rank by total AIC (lower is better)."""
    results = [fit_pk_dataset(subjects, model_key=k) for k in model_keys]
    ranked = sorted(
        results,
        key=lambda r: (r["total_aic"] is None, r["total_aic"] if r["total_aic"] is not None else 0))
    summary = [{"model_key": r["model_key"], "label": r["label"], "group": r["group"],
                "n_converged": r["n_converged"], "n_subjects": r["n_subjects"],
                "total_aic": r["total_aic"], "mean_aic": r["mean_aic"]} for r in ranked]
    best = next((r for r in ranked if r["total_aic"] is not None), None)
    return {"ranking": summary,
            "best_model": best["model_key"] if best else None,
            "best": best}
