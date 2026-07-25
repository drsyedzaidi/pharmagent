"""Bayesian-borrowing diagnostics for informative-prior (MAP) fits (IU PopPK
Week-15: `prior-distributions.R`, `bayes-diagnostics.R`).

Two pure, deterministic-given-seed functions:

* :func:`prior_predictive_check` — draw structural parameters from the prior,
  simulate concentration profiles, band them (2.5/50/97.5%), and report the
  fraction of observed data inside the band. Answers "is the prior consistent
  with the (e.g. pediatric) data before we fit?" — the prior-predictive check.
* :func:`prior_posterior_diagnostic` — from a MAP fit, compare each parameter's
  prior vs posterior (mean, SD) and report the SHRINKAGE the data provided
  (1 - posterior_sd/prior_sd on the log scale) plus a 95% credible interval.
  This is the informativeness read-out the Bayesian workflow produces without a
  full MCMC sampler.

No file I/O, no network, no global state; builds on the shared simulator.
"""
from __future__ import annotations

import numpy as np

from app.compute.pk_models import get_model
from app.compute.pk_simulate import simulate_timecourse

_PCTL = (2.5, 50.0, 97.5)


def _r(x: float | None, dp: int = 6) -> float | None:
    return None if x is None or not np.isfinite(x) else round(float(x), dp)


def _draw_theta_prior(theta_prior: dict, n: int, rng: np.random.Generator
                      ) -> tuple[list[str], np.ndarray]:
    """Draw n structural-parameter vectors from the prior LogNormal(mean_log,
    cov_log) — normal on the log/estimation scale, exponentiated to natural."""
    names = list(theta_prior["names"])
    mu = np.asarray(theta_prior["mean_log"], dtype=float)
    cov = theta_prior.get("cov_log")
    if cov is not None:
        c = np.asarray(cov, dtype=float)
        c = 0.5 * (c + c.T) + 1e-12 * np.eye(len(names))
        draws = rng.multivariate_normal(mu, c, size=n)
    else:
        sd = np.asarray(theta_prior.get("sd_log") or [0.3] * len(names), dtype=float)
        draws = mu + rng.normal(size=(n, len(names))) * sd
    return names, np.exp(draws)


def prior_predictive_check(
    model_key: str, *, theta_prior: dict, base_theta: dict, dose: float, tau: float,
    n_doses: int, obs_times: list[float] | None = None, obs_conc: list[float] | None = None,
    n_draws: int = 500, seed: int = 20250614, n_points: int = 80, wt: float = 70.0,
) -> dict:
    """Prior-predictive concentration band + observed-data coverage.

    Parameters not carried by the prior are held at ``base_theta`` (the fitted or
    default structural values). Returns ``{status, model_key, n_draws, band:
    [{time, lo, med, hi}], coverage_pct, observed:[{time, dv}]}``.
    """
    if not theta_prior or not theta_prior.get("names"):
        return {"status": "no_prior", "message": "no prior supplied for the predictive check."}
    n_draws = int(max(1, n_draws))
    model = get_model(model_key)
    names, draws = _draw_theta_prior(theta_prior, n_draws, np.random.default_rng(seed))
    tau = float(tau)
    n_doses = int(max(1, n_doses))
    tmax = tau * n_doses
    base = {**model.defaults, **{k: float(v) for k, v in base_theta.items()}}

    curves = []
    times = None
    for row in draws:
        p = {**base, **{nm: float(v) for nm, v in zip(names, row)}}
        sim = simulate_timecourse(model, p, dose=float(dose), tau=tau, n_doses=n_doses,
                                  tmax=tmax, n_points=n_points, wt=float(wt))
        cp = np.asarray(sim["cp"], dtype=float)
        if np.all(np.isfinite(cp)):
            curves.append(cp)
            times = np.asarray(sim["times"], dtype=float)
    if not curves or times is None:
        return {"status": "empty", "message": "no finite prior-predictive simulations."}

    c = np.vstack(curves)
    lo, med, hi = np.percentile(c, _PCTL, axis=0)
    band = [{"time": _r(times[i]), "lo": _r(lo[i]), "med": _r(med[i]), "hi": _r(hi[i])}
            for i in range(times.size)]

    coverage = None
    observed = []
    if obs_times and obs_conc:
        ot = np.asarray(obs_times, dtype=float)
        oc = np.asarray(obs_conc, dtype=float)
        ok = np.isfinite(ot) & np.isfinite(oc)
        ot, oc = ot[ok], oc[ok]
        if ot.size:
            lo_i = np.interp(ot, times, lo)
            hi_i = np.interp(ot, times, hi)
            coverage = _r(100.0 * float(np.mean((oc >= lo_i) & (oc <= hi_i))), 2)
            observed = [{"time": _r(t), "dv": _r(v)} for t, v in zip(ot, oc)]
    return {"status": "ok", "model_key": model_key, "n_draws": len(curves),
            "band": band, "coverage_pct": coverage, "observed": observed}


def prior_posterior_diagnostic(fit: dict) -> dict:
    """Per-parameter prior-vs-posterior summary + shrinkage for a MAP fit.

    Shrinkage = ``1 - posterior_sd/prior_sd`` on the log scale (0 = the data added
    nothing beyond the prior; ->1 = the data dominates). Requires a fit produced
    with ``theta_prior`` (carries ``theta_prior`` + ``theta_rse_pct``).
    """
    tp = fit.get("theta_prior")
    if not tp or not tp.get("names"):
        return {"status": "no_prior", "message": "fit was not a MAP fit (no theta_prior)."}
    names = list(tp["names"])
    mean_log = np.asarray(tp["mean_log"], dtype=float)
    cov = tp.get("cov_log")
    prior_sd = (np.sqrt(np.clip(np.diag(np.asarray(cov, dtype=float)), 0.0, None))
                if cov is not None
                else np.asarray(tp.get("sd_log") or [np.nan] * len(names), dtype=float))
    theta = fit.get("theta", {})
    rse = fit.get("theta_rse_pct", {}) or {}

    rows = []
    for i, nm in enumerate(names):
        pm = float(np.exp(mean_log[i]))                       # prior mean (natural)
        psd = float(prior_sd[i]) if i < prior_sd.size else None   # log-scale prior SD
        post = theta.get(nm)
        post_sd = (float(rse[nm]) / 100.0 if rse.get(nm) is not None else None)  # log-scale
        # Shrinkage in [0,1]: the posterior SD cannot exceed the prior SD (posterior
        # precision = Fisher + prior precision), so clamp finite-difference noise.
        shrink = (_r(min(1.0, max(0.0, 1.0 - post_sd / psd)), 4)
                  if (psd and post_sd is not None and psd > 0) else None)
        ci_lo = ci_hi = None
        if post is not None and post_sd is not None:
            ci_lo = _r(post * float(np.exp(-1.96 * post_sd)))
            ci_hi = _r(post * float(np.exp(1.96 * post_sd)))
        rows.append({"param": nm, "prior_mean": _r(pm),
                     "prior_sd_log": _r(psd) if psd else None, "post_mean": _r(post),
                     "post_sd_log": _r(post_sd) if post_sd else None,
                     "shrinkage": shrink, "ci95": [ci_lo, ci_hi]})
    shrinks = [r["shrinkage"] for r in rows if r["shrinkage"] is not None]
    return {"status": "ok", "params": rows,
            "mean_shrinkage": _r(float(np.mean(shrinks)), 4) if shrinks else None}
