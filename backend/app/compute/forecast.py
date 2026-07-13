"""MAP/empirical-Bayes forecasting and TDM dose individualization.

Given a fitted population model (the stored NLME result) and a new patient's
sparse measured levels, estimate the patient's individual PK by MAP/empirical
Bayes, forecast their steady-state exposure, and (optionally) recommend a dose
to hit a target trough / average / peak / AUC.

Measured-level times are absolute hours from the first dose of the current
regimen (dose every tau). The patient's individual parameters come from
``nlme.map_estimate`` (the conditional-mode posterior used inside FOCE-I).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from app.compute.nlme import cv_pct_to_omega2, map_estimate
from app.compute.pk_models import PKModel, get_model
from app.compute.pk_simulate import simulate, simulate_timecourse

_SS_DOSES = 24                       # doses simulated to reach steady state
_METRICS = ("cmin", "cmax", "cavg", "auc_tau")


def _ss_metrics(model: PKModel, params: dict[str, float], dose: float,
                tau: float, wt: float, n_ss: int = _SS_DOSES) -> dict[str, float]:
    """Steady-state exposure metrics over the last dosing interval."""
    doses = [{"time": k * tau, "amt": dose} for k in range(n_ss)]
    t0 = (n_ss - 1) * tau
    grid = np.linspace(t0, t0 + tau, 64)
    cp = np.asarray(simulate(model, params, doses, grid, wt=wt)["cp"], dtype=float)
    auc = float(np.trapezoid(cp, grid))
    return {"cmax": float(np.max(cp)), "cmin": float(cp[-1]),
            "cavg": auc / tau, "auc_tau": auc}


def _optimize_dose(model: PKModel, params: dict[str, float], tau: float,
                   wt: float, metric: str, target: float,
                   n_ss: int = _SS_DOSES) -> float | None:
    """Find the dose whose steady-state ``metric`` equals ``target`` by bisection
    (the metric is monotone increasing in dose). Returns None if unreachable."""
    def m(d: float) -> float:
        return _ss_metrics(model, params, d, tau, wt, n_ss)[metric]

    lo, hi = 1e-3, 1.0
    for _ in range(50):                 # expand upper bound until it brackets
        if m(hi) >= target:
            break
        hi *= 2.0
    else:
        return None
    if m(lo) >= target:                 # target below the minimum testable dose
        return None
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if m(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def forecast(nlme: dict[str, Any], *, dose: float, tau: float,
             measured: list[dict], target: float | None = None,
             target_metric: str = "cmin", wt: float = 70.0,
             cov: dict | None = None, n_hist_doses: int | None = None,
             tmax: float | None = None) -> dict[str, Any]:
    """MAP-individualize a new patient and forecast steady-state exposure.

    Args:
        nlme: a fitted NLME result dict (theta, omega_cv_pct, sigma, iiv_params,
            error_model, covariate_effects, model_key).
        dose, tau: the patient's current regimen (amount, interval h).
        measured: [{"time", "conc"}] absolute hours from first dose.
        target / target_metric: optional dose-optimization goal.
        wt, cov: patient covariates (weight + named covariates for the model).
    """
    model_key = nlme.get("model_key")
    if not model_key:
        return {"status": "no_model", "message": "Fit a population (NLME) model first."}
    model = get_model(model_key)
    theta = dict(nlme.get("theta") or {})
    omega2 = {p: cv_pct_to_omega2(v) for p, v in (nlme.get("omega_cv_pct") or {}).items()}
    sig = nlme.get("sigma") or {}
    sigma_prop = float(sig.get("prop") or 0.0)
    sigma_add = float(sig.get("add") or 0.0)
    iiv = nlme.get("iiv_params") or ["CL", "V"]
    error_model = nlme.get("error_model") or "proportional"

    obs_t = [float(m["time"]) for m in (measured or [])]
    obs_c = [float(m["conc"]) for m in (measured or [])]
    tmax_obs = max(obs_t) if obs_t else tau
    n_doses = int(n_hist_doses or (int(tmax_obs // tau) + 1))
    hist_doses = [{"time": i * tau, "amt": dose} for i in range(max(n_doses, 1))]

    m = map_estimate(
        model_key, theta=theta, omega2=omega2, sigma_prop=sigma_prop,
        sigma_add=sigma_add, iiv_params=iiv, obs_t=obs_t, obs_c=obs_c,
        doses=hist_doses, wt=wt, cov=cov,
        covariate_effects=nlme.get("covariate_effects"), error_model=error_model)
    ind_params, typ_params = m["individual_params"], m["typical_params"]

    show_tmax = float(tmax or tau * 3)
    n_show = int(np.ceil(show_tmax / tau)) + 1
    ind_curve = simulate_timecourse(model, ind_params, dose=dose, tau=tau,
                                    n_doses=n_show, tmax=show_tmax, wt=wt)
    pop_curve = simulate_timecourse(model, typ_params, dose=dose, tau=tau,
                                    n_doses=n_show, tmax=show_tmax, wt=wt)

    out: dict[str, Any] = {
        "status": "ok", "model_key": model_key, "label": model.label,
        "eta": m["eta"], "n_obs": m["n_obs"],
        "individual_params": ind_params, "typical_params": typ_params,
        "dose": dose, "tau": tau, "wt": wt,
        "measured": [{"time": t, "conc": c} for t, c in zip(obs_t, obs_c)],
        "ss_individual": {k: round(v, 4) for k, v in
                          _ss_metrics(model, ind_params, dose, tau, wt).items()},
        "ss_population": {k: round(v, 4) for k, v in
                         _ss_metrics(model, typ_params, dose, tau, wt).items()},
        "forecast": {"times": ind_curve["times"], "individual": ind_curve["cp"],
                     "population": pop_curve["cp"]},
    }

    if target is not None and target_metric in _METRICS:
        rec = _optimize_dose(model, ind_params, tau, wt, target_metric, float(target))
        if rec is not None:
            out["recommendation"] = {
                "target_metric": target_metric, "target": float(target),
                "recommended_dose": round(rec, 3),
                "predicted": {k: round(v, 4) for k, v in
                              _ss_metrics(model, ind_params, rec, tau, wt).items()}}
        else:
            out["recommendation"] = {
                "target_metric": target_metric, "target": float(target),
                "recommended_dose": None,
                "note": "target not reachable within the searched dose range"}
    return out
