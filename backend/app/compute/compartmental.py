"""Compartmental oral PK model fitting — deterministic compute.

Pure functions, no agent/LLM dependencies, fully unit-testable. Implements
closed-form 1- and 2-compartment ORAL (first-order absorption) concentration
models and per-subject nonlinear least-squares fitting on the log
(proportional-error) scale via :func:`scipy.optimize.least_squares`.

Model parameterizations (F=1, single bolus dose D into the depot at t=0):

    1-cmt oral (ka, CL, V), ke = CL/V:
        C(t) = (D*ka)/(V*(ka-ke)) * (exp(-ke*t) - exp(-ka*t))

    2-cmt oral (ka, CL, V1, Q, V2):
        k = CL/V1; k12 = Q/V1; k21 = Q/V2
        beta  = 0.5*((k+k12+k21) - sqrt((k+k12+k21)**2 - 4*k21*k))
        alpha = k21*k/beta
        A = (D*ka/V1) * (k21 - alpha) / ((ka - alpha)*(beta - alpha))
        B = (D*ka/V1) * (k21 - beta)  / ((ka - beta)*(alpha - beta))
        C(t) = A*exp(-alpha*t) + B*exp(-beta*t) - (A+B)*exp(-ka*t)

Per-subject fit returns:
    subject, model, converged, params, aic, r_squared, n_obs, all_models
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import least_squares

# Small guard against division-by-zero when two rate constants coincide.
_EPS = 1e-9

# Parameter names per model, in optimization order.
_PARAM_NAMES: dict[str, tuple[str, ...]] = {
    "1cmt": ("ka", "CL", "V"),
    "2cmt": ("ka", "CL", "V1", "Q", "V2"),
    "1cmt_ss": ("ka", "CL", "V"),
    "2cmt_ss": ("ka", "CL", "V1", "Q", "V2"),
}


@dataclass
class _FitData:
    time: np.ndarray
    conc: np.ndarray
    dose: float


def conc_1cmt_oral(t: Any, dose: float, ka: float, CL: float, V: float) -> np.ndarray:
    """One-compartment oral (first-order absorption) concentration over time.

    Vectorized over ``t``. Guards the flip-flop degenerate case (ka == ke) by
    nudging the denominator off zero with a tiny epsilon.
    """
    t = np.asarray(t, dtype=float)
    ke = CL / V
    denom = ka - ke
    if abs(denom) < _EPS:
        denom = _EPS if denom >= 0 else -_EPS
    return (dose * ka) / (V * denom) * (np.exp(-ke * t) - np.exp(-ka * t))


def conc_2cmt_oral(t: Any, dose: float, ka: float, CL: float, V1: float,
                   Q: float, V2: float) -> np.ndarray:
    """Two-compartment oral (first-order absorption) concentration over time.

    Vectorized over ``t``. Guards the degenerate cases where any two of the
    rate constants (alpha, beta, ka) coincide by nudging denominators off zero.
    """
    t = np.asarray(t, dtype=float)
    k = CL / V1
    k12 = Q / V1
    k21 = Q / V2

    s = k + k12 + k21
    disc = s * s - 4.0 * k21 * k
    disc = max(disc, 0.0)
    beta = 0.5 * (s - math.sqrt(disc))
    if abs(beta) < _EPS:
        beta = _EPS
    alpha = k21 * k / beta

    def _guard(x: float) -> float:
        if abs(x) < _EPS:
            return _EPS if x >= 0 else -_EPS
        return x

    a_denom = _guard(ka - alpha) * _guard(beta - alpha)
    b_denom = _guard(ka - beta) * _guard(alpha - beta)
    A = (dose * ka / V1) * (k21 - alpha) / a_denom
    B = (dose * ka / V1) * (k21 - beta) / b_denom
    return A * np.exp(-alpha * t) + B * np.exp(-beta * t) - (A + B) * np.exp(-ka * t)


def _accum(rate: float, tau: float) -> float:
    """Steady-state accumulation factor 1/(1 - exp(-rate*tau)) for one exponential.
    As tau -> inf this -> 1, recovering the single-dose curve."""
    denom = 1.0 - math.exp(-rate * tau)
    if abs(denom) < _EPS:
        denom = _EPS
    return 1.0 / denom


def conc_1cmt_oral_ss(t: Any, dose: float, ka: float, CL: float, V: float,
                      tau: float) -> np.ndarray:
    """One-compartment oral concentration at STEADY STATE over a dosing interval.

    Each exponential carries its 1/(1-exp(-rate*tau)) accumulation factor; as
    tau -> inf this reduces to the single-dose ``conc_1cmt_oral``.
    """
    t = np.asarray(t, dtype=float)
    ke = CL / V
    denom = ka - ke
    if abs(denom) < _EPS:
        denom = _EPS if denom >= 0 else -_EPS
    return (dose * ka) / (V * denom) * (
        np.exp(-ke * t) * _accum(ke, tau) - np.exp(-ka * t) * _accum(ka, tau)
    )


def conc_2cmt_oral_ss(t: Any, dose: float, ka: float, CL: float, V1: float,
                      Q: float, V2: float, tau: float) -> np.ndarray:
    """Two-compartment oral concentration at STEADY STATE over a dosing interval.

    Same macro-constants (A, B, alpha, beta) as the single-dose form, each
    exponential scaled by its accumulation factor; tau -> inf recovers
    ``conc_2cmt_oral``.
    """
    t = np.asarray(t, dtype=float)
    k = CL / V1
    k12 = Q / V1
    k21 = Q / V2

    s = k + k12 + k21
    disc = max(s * s - 4.0 * k21 * k, 0.0)
    beta = 0.5 * (s - math.sqrt(disc))
    if abs(beta) < _EPS:
        beta = _EPS
    alpha = k21 * k / beta

    def _guard(x: float) -> float:
        if abs(x) < _EPS:
            return _EPS if x >= 0 else -_EPS
        return x

    a_denom = _guard(ka - alpha) * _guard(beta - alpha)
    b_denom = _guard(ka - beta) * _guard(alpha - beta)
    A = (dose * ka / V1) * (k21 - alpha) / a_denom
    B = (dose * ka / V1) * (k21 - beta) / b_denom
    return (A * np.exp(-alpha * t) * _accum(alpha, tau)
            + B * np.exp(-beta * t) * _accum(beta, tau)
            - (A + B) * np.exp(-ka * t) * _accum(ka, tau))


def _predict(model: str, theta: np.ndarray, t: np.ndarray, dose: float,
             tau: float | None = None) -> np.ndarray:
    """Evaluate the model at parameter vector ``theta`` (natural scale)."""
    if model == "1cmt":
        ka, CL, V = theta
        return conc_1cmt_oral(t, dose, ka, CL, V)
    if model == "2cmt":
        ka, CL, V1, Q, V2 = theta
        return conc_2cmt_oral(t, dose, ka, CL, V1, Q, V2)
    if model == "1cmt_ss":
        ka, CL, V = theta
        return conc_1cmt_oral_ss(t, dose, ka, CL, V, tau)
    if model == "2cmt_ss":
        ka, CL, V1, Q, V2 = theta
        return conc_2cmt_oral_ss(t, dose, ka, CL, V1, Q, V2, tau)
    raise ValueError(f"unknown model: {model}")


def _initial_estimates(model: str, time: np.ndarray, conc: np.ndarray,
                       dose: float) -> np.ndarray:
    """Crude starting values from the observed profile."""
    ka0 = 1.0

    # Terminal slope from the last (up to) 3 positive points.
    pos = conc > 0
    tt = time[pos]
    cc = conc[pos]
    ke0 = 0.1
    if tt.size >= 2:
        n = min(3, tt.size)
        slope = np.polyfit(tt[-n:], np.log(cc[-n:]), 1)[0]
        if slope < 0:
            ke0 = float(-slope)
    if not math.isfinite(ke0) or ke0 <= 0:
        ke0 = 0.1

    cmax = float(np.max(conc)) if conc.size else 1.0
    v0 = dose / cmax if cmax > 0 else dose
    if not math.isfinite(v0) or v0 <= 0:
        v0 = 1.0
    cl0 = ke0 * v0

    if model in ("1cmt", "1cmt_ss"):
        return np.array([ka0, cl0, v0], dtype=float)
    # 2cmt: seed Q0=CL0, V20=V0 per spec.
    return np.array([ka0, cl0, v0, cl0, v0], dtype=float)


def fit_one_subject(time: Any, conc: Any, dose: float, *, model: str,
                    tau: float | None = None) -> dict[str, Any]:
    """Fit one subject's oral PK profile on the log (proportional-error) scale.

    Optimizes the logs of the (positive) parameters so positivity is automatic.
    For steady-state models ("1cmt_ss"/"2cmt_ss") pass the dosing interval
    ``tau``. Returns a result dict with EXACT keys: subject (None here), model,
    converged, params, aic, r_squared, n_obs. On too-few-points, failure, or
    a non-finite result, returns ``converged=False`` with ``params={}``.
    """
    failed: dict[str, Any] = {
        "subject": None,
        "model": model,
        "converged": False,
        "params": {},
        "aic": None,
        "r_squared": None,
        "n_obs": 0,
    }
    if model not in _PARAM_NAMES:        # unknown model -> safe failure, not a crash
        return failed
    names = _PARAM_NAMES[model]
    n_params = len(names)

    time = np.asarray(time, dtype=float)
    conc = np.asarray(conc, dtype=float)
    order = np.argsort(time)
    time = time[order]
    conc = conc[order]

    # Single-dose oral models give C(0)=0 exactly, so a measurable t=0 sample is
    # structurally unfittable and would dominate the log residual -> exclude t<=0.
    # Steady-state models have C(0)=trough>0, so the t=0 point IS meaningful.
    is_ss = model.endswith("_ss")
    mask = (conc > 0) if is_ss else ((conc > 0) & (time > 0))
    t_fit = time[mask]
    c_fit = conc[mask]
    n_obs = int(t_fit.size)
    failed["n_obs"] = n_obs

    if n_obs < n_params + 1:
        return failed

    log_obs = np.log(c_fit)

    def residuals(log_theta: np.ndarray) -> np.ndarray:
        theta = np.exp(log_theta)
        pred = _predict(model, theta, t_fit, dose, tau)
        return np.log(np.maximum(pred, _EPS)) - log_obs

    try:
        theta0 = _initial_estimates(model, time, conc, dose)
        theta0 = np.maximum(theta0, _EPS)
        sol = least_squares(residuals, np.log(theta0), method="lm",
                            max_nfev=5000)
        theta_hat = np.exp(sol.x)

        if not np.all(np.isfinite(theta_hat)):
            return failed

        res = residuals(sol.x)
        ssr = float(np.sum(res ** 2))
        if not math.isfinite(ssr) or ssr < 0:
            return failed

        # A perfect fit (ssr==0, only with noiseless synthetic data) would give
        # AIC=-inf; floor SSR to a tiny value so the AIC stays finite and the
        # model remains eligible for AIC-based selection.
        ssr_aic = max(ssr, 1e-12)
        aic = n_obs * math.log(ssr_aic / n_obs) + 2 * n_params

        ss_tot = float(np.sum((log_obs - log_obs.mean()) ** 2))
        r2 = 1.0 - ssr / ss_tot if ss_tot > 0 else 0.0

        params = {name: round(float(val), 6) for name, val in zip(names, theta_hat)}

        # Relative standard errors (%) via the delta method on log-params:
        # cov = sigma^2 (JᵀJ)^-1, RSE%(theta) = 100·SE(log theta).
        rse = None
        dof = n_obs - n_params
        if dof >= 1:
            try:
                sigma2 = ssr / dof
                cov = sigma2 * np.linalg.pinv(sol.jac.T @ sol.jac)
                se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
                if np.all(np.isfinite(se)):
                    rse = {n: round(100.0 * float(se[i]), 2) for i, n in enumerate(names)}
            except Exception:
                rse = None

        return {
            "subject": None,
            "model": model,
            "converged": True,
            "params": params,
            "rse_pct": rse,
            "aic": round(aic, 4) if math.isfinite(aic) else None,
            "r_squared": round(r2, 6),
            "n_obs": n_obs,
        }
    except Exception:
        return failed


def fit_compartmental(records: list[dict[str, Any]], *, id_col: str, time_col: str,
                      dv_col: str, dose_by_subject: dict[Any, float],
                      models: tuple[str, ...] = ("1cmt", "2cmt")) -> dict[str, Any]:
    """Top-level entry: group by subject, fit every model, select lowest AIC.

    `records` are plain dicts (already loaded). Observation rows only
    (EVID==0 / dosing rows excluded) should be passed in for `dv_col`. For each
    subject, every model in `models` is fit and the converged fit with the
    LOWEST AIC is chosen as the best.
    """
    by_subj: dict[Any, list[tuple[float, float]]] = {}
    for r in records:
        sid = r[id_col]
        t = r.get(time_col)
        dv = r.get(dv_col)
        if t is None or dv is None:
            continue
        by_subj.setdefault(sid, []).append((float(t), float(dv)))

    individual_fits: list[dict[str, Any]] = []
    model_selection: dict[Any, str | None] = {}

    for sid, pts in by_subj.items():
        pts.sort(key=lambda x: x[0])
        time = np.array([p[0] for p in pts])
        conc = np.array([p[1] for p in pts])
        dose = float(dose_by_subject.get(sid, float("nan")))

        attempts: list[dict[str, Any]] = []
        for model in models:
            fit = fit_one_subject(time, conc, dose, model=model)
            fit["subject"] = sid
            attempts.append(fit)

        converged = [f for f in attempts if f["converged"] and f["aic"] is not None]
        if converged:
            best = min(converged, key=lambda f: f["aic"])
        else:
            # Fall back to first attempt so the subject is still represented.
            best = attempts[0]

        best = dict(best)
        best["all_models"] = attempts
        individual_fits.append(best)
        model_selection[sid] = best["model"] if best["converged"] else None

    individual_fits.sort(key=lambda r: str(r["subject"]))
    return {
        "individual_fits": individual_fits,
        "model_selection": model_selection,
        "n_subjects": len(by_subj),
    }


def fit_compartmental_ss(profiles: dict[Any, Any], *,
                         models: tuple[str, ...] = ("1cmt_ss", "2cmt_ss")) -> dict[str, Any]:
    """Fit steady-state models to per-subject dosing-interval profiles.

    ``profiles`` maps subject -> an object with ``.tad``, ``.conc``, ``.dose``
    and ``.tau`` (an ``app.compute.dosing.SSProfile``). For each subject every
    SS model is fit (passing tau) and the lowest-AIC converged fit is chosen.
    """
    individual_fits: list[dict[str, Any]] = []
    model_selection: dict[Any, str | None] = {}

    for sid, p in profiles.items():
        time = np.asarray(p.tad, dtype=float)
        conc = np.asarray(p.conc, dtype=float)
        dose = float(p.dose)
        tau = float(p.tau)

        attempts: list[dict[str, Any]] = []
        for model in models:
            fit = fit_one_subject(time, conc, dose, model=model, tau=tau)
            fit["subject"] = sid
            attempts.append(fit)

        converged = [f for f in attempts if f["converged"] and f["aic"] is not None]
        best = min(converged, key=lambda f: f["aic"]) if converged else attempts[0]
        best = dict(best)
        best["all_models"] = attempts
        individual_fits.append(best)
        model_selection[sid] = best["model"] if best["converged"] else None

    individual_fits.sort(key=lambda r: str(r["subject"]))
    return {
        "individual_fits": individual_fits,
        "model_selection": model_selection,
        "n_subjects": len(profiles),
    }
