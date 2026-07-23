"""Clinical trial simulation and probability of target attainment (PTA).

Week-12 "Simulation" of the IU PopPK course turns a fitted population model into
a dosing recommendation: sample a virtual population (covariates resampled from
the analysis dataset, between-subject variability drawn from Omega), simulate a
dosing regimen for every subject across a grid of dose levels, and report the
fraction of subjects whose exposure meets a clinical target — the probability of
target attainment. The recommended dose is the smallest dose reaching the target
fraction for an efficacy criterion (metric ABOVE a threshold), or the largest
dose still meeting it for a safety criterion (metric BELOW a threshold).

Pure, deterministic-given-seed compute on top of the shared simulator
(:func:`app.compute.pk_simulate.simulate_timecourse`) and the exposure-metric
extractor (:func:`app.compute.dose_sweep._interval_metrics`). No file I/O, no
network, no global mutable state.

Design notes matching the course lab (Exercise 2, ``zero_re(sigma)``):

  * Target attainment is judged on each subject's INDIVIDUAL exposure (IPRED) —
    residual/assay error is NOT added, since PTA concerns the patient's true
    exposure, not a noisy measurement.
  * Covariates are resampled as whole rows from the fitted subjects, preserving
    the real covariate correlation structure, rather than sampling marginals.
  * Per-subject typical values apply the fitted covariate model first, then IIV
    (``p_i = theta * cov_factor * exp(eta)``); WT allometry is applied by the
    simulator, so a fitted WT covariate is intentionally never added (it would
    double-count the built-in allometric scaling).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from app.compute.dose_sweep import _interval_metrics
from app.compute.pk_models import get_model
from app.compute.pk_simulate import simulate_timecourse

_METRICS = ("cmax", "auc_tau", "cavg", "ctrough")
_REPORT_PCTL = (5.0, 25.0, 50.0, 75.0, 95.0)


def _cv_pct_to_omega2(cv_pct: float | None) -> float:
    """Lognormal %CV -> variance omega2 = ln(1 + (cv/100)^2)."""
    if cv_pct is None:
        return 0.0
    cv = float(cv_pct)
    return float(np.log1p((cv / 100.0) ** 2)) if cv > 0 else 0.0


def _covariate_applier(covariate_effects: list[dict] | None):
    """Return f(theta, cov) -> covariate-adjusted typical values.

    Reuses the fitted covariate model (``nlme._CovEffect``) so the multiplicative
    factors are identical to what the estimator applied — no reimplementation to
    drift from. Returns the identity when there is no covariate model.
    """
    if not covariate_effects:
        return lambda theta, cov: theta
    from app.compute.nlme import _cov_effects_from_records  # lazy: heavy (scipy)

    effects, coefs = _cov_effects_from_records(covariate_effects)

    def apply(theta: dict[str, float], cov: dict) -> dict[str, float]:
        p = dict(theta)
        i = 0
        for eff in effects:
            c = coefs[i:i + eff.n_coef]
            if eff.param in p:
                p[eff.param] = p[eff.param] * eff.factor(c, (cov or {}).get(eff.covariate))
            i += eff.n_coef
        return p

    return apply


def _sample_population(cov_rows: list[dict] | None, wt_rows: list[float] | None,
                       n_subjects: int, rng: np.random.Generator,
                       wt_default: float) -> list[dict]:
    """Resample ``n_subjects`` virtual patients from the observed covariate/WT
    rows (whole-row bootstrap, preserving covariate correlations). Without a
    source, every subject gets the default weight and no covariates."""
    rows = cov_rows or []
    weights = wt_rows or []
    if not rows and not weights:
        return [{"wt": wt_default, "cov": {}} for _ in range(n_subjects)]
    n_src = max(len(rows), len(weights))
    idx = rng.integers(0, n_src, size=n_subjects)
    pop = []
    for j in idx:
        cov = dict(rows[j]) if j < len(rows) else {}
        wt = float(weights[j]) if j < len(weights) else float(cov.get("WT", wt_default))
        pop.append({"wt": wt, "cov": cov})
    return pop


def clinical_trial_simulation(
    model_key: str, *, theta: dict, omega_cv_pct: dict, iiv_params: list[str],
    doses: list[float], tau: float, n_doses: int,
    metric: str = "ctrough", threshold: float | None = None,
    direction: str = "above", target_fraction: float = 0.9,
    cov_rows: list[dict] | None = None, wt_rows: list[float] | None = None,
    covariate_effects: list[dict] | None = None, n_subjects: int = 500,
    seed: int = 20250614, wt_default: float = 70.0, n_points: int = 160,
    max_subjects: int = 2000, max_doses: int = 24,
) -> dict:
    """Probability of target attainment (PTA) across a dose grid.

    Simulate ``n_subjects`` virtual patients (covariates resampled from
    ``cov_rows``/``wt_rows``, IIV drawn from ``omega_cv_pct``) at every dose in
    ``doses`` for an ``n_doses``-dose regimen at interval ``tau``. For each dose,
    the PTA is the fraction of subjects whose ``metric`` (``cmax``/``auc_tau``/
    ``cavg``/``ctrough`` over the last interval) is on the target side of
    ``threshold`` (``direction`` = ``"above"`` for efficacy, ``"below"`` for
    safety). The recommended dose is the smallest (above) / largest (below) dose
    reaching ``target_fraction``.

    Returns ``{status, model_key, label, metric, threshold, direction,
    target_fraction, tau, n_doses, n_subjects, with_covariates, with_iiv,
    doses: [{dose, pta, n, metric_p05..p95}], recommended_dose,
    recommendation_note}``. ``threshold=None`` skips PTA and returns the exposure
    distribution per dose only.
    """
    if metric not in _METRICS:
        raise ValueError(f"metric must be one of {_METRICS}; got {metric!r}")
    if direction not in ("above", "below"):
        raise ValueError(f"direction must be 'above'|'below'; got {direction!r}")
    dose_grid = sorted({float(d) for d in doses if float(d) > 0})
    if not dose_grid:
        return {"status": "no_doses", "message": "no positive dose levels supplied."}
    if len(dose_grid) > max_doses:
        return {"status": "too_many_doses",
                "message": f"dose grid capped at {max_doses}; got {len(dose_grid)}."}
    n_subjects = int(max(1, min(n_subjects, max_subjects)))
    tau = float(tau)
    n_doses = int(max(1, n_doses))
    tmax = tau * n_doses

    model = get_model(model_key)
    typical = {**model.defaults, **{k: float(v) for k, v in theta.items()}}
    apply_cov = _covariate_applier(covariate_effects)
    sds = {p: float(np.sqrt(_cv_pct_to_omega2(omega_cv_pct.get(p))))
           for p in iiv_params}
    with_iiv = any(v > 0 for v in sds.values())

    rng = np.random.default_rng(seed)
    pop = _sample_population(cov_rows, wt_rows, n_subjects, rng, wt_default)
    # Pre-draw IIV etas (subjects x params) so results are seed-deterministic and
    # independent of the dose-grid iteration order.
    etas = {p: rng.normal(0.0, sds[p], size=n_subjects) if sds[p] > 0
            else np.zeros(n_subjects) for p in iiv_params}
    with_covariates = bool(covariate_effects) and bool(cov_rows)

    dose_rows: list[dict] = []
    for d in dose_grid:
        vals = np.empty(n_subjects)
        for si, subj in enumerate(pop):
            theta_i = apply_cov(typical, subj["cov"])
            params_i = {k: float(theta_i[k]) for k in theta_i}
            for p in iiv_params:
                if p in params_i:
                    params_i[p] = params_i[p] * float(np.exp(etas[p][si]))
            sim = simulate_timecourse(model, params_i, dose=d, tau=tau,
                                      n_doses=n_doses, tmax=tmax, n_points=n_points,
                                      wt=subj["wt"])
            m = _interval_metrics(sim["times"], sim["cp"],
                                  t_last=(n_doses - 1) * tau, tau=tau, tmax=tmax)
            vals[si] = m[metric]

        finite = vals[np.isfinite(vals)]
        pcts = (np.percentile(finite, _REPORT_PCTL) if finite.size
                else [float("nan")] * len(_REPORT_PCTL))
        row: dict[str, Any] = {
            "dose": round(d, 6), "n": int(finite.size),
            "metric_p05": _r(pcts[0]), "metric_p25": _r(pcts[1]),
            "metric_median": _r(pcts[2]), "metric_p75": _r(pcts[3]),
            "metric_p95": _r(pcts[4]),
        }
        if threshold is not None and finite.size:
            hit = finite > float(threshold) if direction == "above" else finite < float(threshold)
            row["pta"] = round(float(np.mean(hit)), 6)
        else:
            row["pta"] = None
        dose_rows.append(row)

    rec_dose, rec_note = _recommend(dose_rows, threshold, direction, target_fraction)
    return {
        "status": "ok", "model_key": model_key, "label": model.label,
        "metric": metric, "threshold": (None if threshold is None else round(float(threshold), 6)),
        "direction": direction, "target_fraction": round(float(target_fraction), 6),
        "tau": tau, "n_doses": n_doses, "n_subjects": n_subjects,
        "with_covariates": with_covariates, "with_iiv": with_iiv,
        "doses": dose_rows, "recommended_dose": rec_dose, "recommendation_note": rec_note,
    }


def _recommend(dose_rows: list[dict], threshold: float | None, direction: str,
               target_fraction: float) -> tuple[float | None, str]:
    """Smallest dose (efficacy/above) or largest dose (safety/below) whose PTA
    reaches ``target_fraction``."""
    if threshold is None:
        return None, "no threshold supplied — exposure distribution only, no PTA."
    meeting = [r for r in dose_rows if r.get("pta") is not None and r["pta"] >= target_fraction]
    if not meeting:
        return None, (f"no dose reaches {target_fraction:.0%} target attainment "
                      f"({direction} {threshold:g}).")
    if direction == "above":                 # efficacy: lowest sufficient dose
        best = min(meeting, key=lambda r: r["dose"])
    else:                                    # safety: highest still-safe dose
        best = max(meeting, key=lambda r: r["dose"])
    return best["dose"], (f"dose {best['dose']:g} attains {best['pta']:.1%} "
                          f"({direction} {threshold:g}), meeting the {target_fraction:.0%} target.")


def _r(x: float) -> float | None:
    return None if not np.isfinite(x) else round(float(x), 6)


def _coef_ses(covariate_effects: list[dict] | None) -> np.ndarray:
    """Per-coefficient standard errors from the stored ``rse_pct``
    (SE = rse_pct/100 * |coef|), flattened in the same order as
    ``nlme._cov_effects_from_records``. A missing/None RSE gives SE 0 (the
    coefficient is treated as known — its exposure ratio carries no width)."""
    ses: list[float] = []
    for r in covariate_effects or []:
        if r.get("kind", "power") == "categorical":
            levels = r.get("levels") or []
            cf = r.get("coefficient") or {}
            rse = r.get("rse_pct") or {}
            for lv in levels:
                c = float(cf.get(lv, 0.0))
                rp = rse.get(lv) if isinstance(rse, dict) else rse
                ses.append(abs(c) * float(rp) / 100.0 if rp else 0.0)
        else:
            c = float(r.get("coefficient") or 0.0)
            rp = r.get("rse_pct")
            ses.append(abs(c) * float(rp) / 100.0 if rp else 0.0)
    return np.asarray(ses, dtype=float)


def exposure_covariate_forest(
    model_key: str, *, theta: dict, covariate_effects: list[dict],
    scenarios: list[dict], reference_cov: dict, dose: float, tau: float,
    n_doses: int, ref_wt: float = 70.0, n_draws: int = 500, seed: int = 20250614,
    n_points: int = 160, band: tuple[float, float] = (0.8, 1.25),
) -> dict:
    """Simulated exposure covariate forest (Week-12 ``forest-plots.R``).

    For a reference patient and each covariate scenario level, forward-simulate a
    steady-state regimen (NO between-subject variability — ``zero_re``) and report
    the RELATIVE exposure (AUC over the last interval, and Cmax) versus the
    reference, with a 95% interval from covariate-coefficient uncertainty (drawn
    from the fitted coefficient ± its RSE). The shaded ``band`` (default
    0.8-1.25) marks the range usually judged not clinically meaningful.

    ``scenarios``: ``[{covariate, is_weight, levels: [{label, value}]}]``. A
    ``is_weight`` scenario varies the simulator's weight (allometric scaling); any
    other varies that covariate in the fitted covariate model. ``reference_cov``
    holds the reference covariate values (weight scenarios use ``ref_wt``).

    Returns ``{status, model_key, label, dose, tau, n_doses, n_draws, band,
    reference: {auc, cmax}, rows: [{covariate, label, value, is_weight,
    rel_auc: {median, lo, hi}, rel_cmax: {median, lo, hi}}]}``.
    """
    if not covariate_effects and not any(s.get("is_weight") for s in scenarios):
        return {"status": "no_covariate_model",
                "message": "no fitted covariate effects to build an exposure forest."}
    model = get_model(model_key)
    typical = {**model.defaults, **{k: float(v) for k, v in theta.items()}}

    from app.compute.nlme import _cov_effects_from_records  # lazy: heavy (scipy)
    effects, coefs0 = _cov_effects_from_records(covariate_effects)
    ses = _coef_ses(covariate_effects)
    if ses.size != coefs0.size:                      # defensive: shape mismatch
        ses = np.zeros_like(coefs0)

    tmax = tau * n_doses
    t_last = (n_doses - 1) * tau

    def _apply(coefs: np.ndarray, cov: dict) -> dict:
        p = dict(typical)
        i = 0
        for eff in effects:
            c = coefs[i:i + eff.n_coef]
            if eff.param in p:
                p[eff.param] = p[eff.param] * eff.factor(c, (cov or {}).get(eff.covariate))
            i += eff.n_coef
        return p

    def _exposure(coefs: np.ndarray, cov: dict, wt: float) -> tuple[float, float]:
        params = _apply(coefs, cov) if effects else dict(typical)
        sim = simulate_timecourse(model, params, dose=dose, tau=tau, n_doses=n_doses,
                                  tmax=tmax, n_points=n_points, wt=wt)
        m = _interval_metrics(sim["times"], sim["cp"], t_last=t_last, tau=tau, tmax=tmax)
        return m["auc_tau"], m["cmax"]

    rng = np.random.default_rng(seed)
    draws = (coefs0 + rng.standard_normal((int(n_draws), coefs0.size)) * ses
             if coefs0.size else np.zeros((int(n_draws), 0)))

    # Reference exposure per draw (shared across scenarios).
    ref_auc = np.empty(int(n_draws))
    ref_cmax = np.empty(int(n_draws))
    for k in range(int(n_draws)):
        ref_auc[k], ref_cmax[k] = _exposure(draws[k], reference_cov, ref_wt)

    def _summ(rel: np.ndarray) -> dict:
        f = rel[np.isfinite(rel)]
        if f.size == 0:
            return {"median": None, "lo": None, "hi": None}
        lo, hi = np.percentile(f, [2.5, 97.5])
        return {"median": _r(float(np.median(f))), "lo": _r(float(lo)), "hi": _r(float(hi))}

    rows = []
    for sc in scenarios:
        is_wt = bool(sc.get("is_weight"))
        for lv in sc.get("levels", []):
            ra = np.empty(int(n_draws))
            rc = np.empty(int(n_draws))
            for k in range(int(n_draws)):
                if is_wt:
                    a, cm = _exposure(draws[k], reference_cov, float(lv["value"]))
                else:
                    cov = {**reference_cov, sc["covariate"]: lv["value"]}
                    a, cm = _exposure(draws[k], cov, ref_wt)
                ra[k] = a / ref_auc[k] if ref_auc[k] else np.nan
                rc[k] = cm / ref_cmax[k] if ref_cmax[k] else np.nan
            rows.append({
                "covariate": sc["covariate"], "label": lv.get("label", str(lv["value"])),
                "value": lv["value"], "is_weight": is_wt,
                "rel_auc": _summ(ra), "rel_cmax": _summ(rc),
            })
    # Point-estimate reference exposure (coefs0 = fitted coefficients).
    ref0_auc, ref0_cmax = _exposure(coefs0, reference_cov, ref_wt)
    return {
        "status": "ok", "model_key": model_key, "label": model.label,
        "dose": round(float(dose), 6), "tau": float(tau), "n_doses": int(n_doses),
        "n_draws": int(n_draws), "band": [float(band[0]), float(band[1])],
        "reference": {"auc": _r(ref0_auc), "cmax": _r(ref0_cmax), "wt": float(ref_wt)},
        "rows": rows,
    }
