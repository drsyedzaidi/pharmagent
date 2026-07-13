"""Engine-agnostic goodness scoring — the common footing for cross-engine ranking.

Every engine's final estimates are pushed through the *same* app compute
primitives (``obs_vs_pred``, ``pcvpc``) on the *same* dataset, so two engines are
compared on identical predictions rather than on their (incomparable) native
likelihoods. Heavy imports stay lazy, matching the app's ``run_nlme`` convention.
"""
from __future__ import annotations

import logging
import math
from typing import Any

_LOG = logging.getLogger(__name__)


def _log_bias(observed: list[float], ipred: list[float]) -> float | None:
    """Mean log-scale prediction bias, mean(log ipred - log obs) over positive
    pairs. Self-contained (OQ-2 resolved without a new metric function)."""
    diffs = [math.log(p) - math.log(o)
             for o, p in zip(observed, ipred)
             if o > 0 and p > 0 and math.isfinite(o) and math.isfinite(p)]
    return sum(diffs) / len(diffs) if diffs else None


def vpc_coverage(pc: dict) -> float | None:
    """Fraction of pcVPC bins where the observed median falls inside the 90% CI
    of the simulated median. In [0, 1]; None when the pcVPC could not be built.

    A status=='ok' pcVPC can still contain bins whose ``obs_p50`` / ``sim_med_lo``
    / ``sim_med_hi`` are None (a degenerate/empty bin, rounded via ``_r`` to
    None). Such bins are un-assessable and excluded from BOTH numerator and
    denominator; if none remain assessable the coverage is None.
    """
    if pc.get("status") != "ok":
        return None
    assessable = [b for b in (pc.get("bins") or [])
                  if b.get("sim_med_lo") is not None
                  and b.get("obs_p50") is not None
                  and b.get("sim_med_hi") is not None]
    if not assessable:
        return None
    hit = sum(1 for b in assessable
              if b["sim_med_lo"] <= b["obs_p50"] <= b["sim_med_hi"])
    return round(hit / len(assessable), 6)


def score_from_population(model_key: str, subjects: list[dict], *,
                          theta: dict, omega_cv_pct: dict,
                          sigma_prop: float | None, sigma_add: float | None,
                          iiv_params: list[str],
                          error_model: str = "proportional",
                          covariate_effects: list[dict] | None = None) -> dict[str, Any]:
    """Score an engine given only its POPULATION model (θ, Ω%CV, σ).

    Individual predictions are derived on our side via the app's own
    ``map_estimate`` (empirical-Bayes) — identically for every engine — so an
    external engine that exposes only fixed effects is scored on exactly the same
    footing as the native fit. This is what makes the cross-engine comparison
    fair: only the population parameters differ between engines, never the
    prediction machinery.
    """
    from app.compute.nlme import cv_pct_to_omega2, map_estimate  # lazy: heavy

    omega2 = {p: cv_pct_to_omega2(v) for p, v in omega_cv_pct.items()}
    ind: dict[Any, dict] = {}
    n_fallback = 0
    for s in subjects:
        try:
            m = map_estimate(
                model_key, theta=theta, omega2=omega2,
                sigma_prop=sigma_prop or 0.0, sigma_add=sigma_add or 0.0,
                iiv_params=iiv_params, obs_t=s["obs_t"], obs_c=s["obs_c"],
                doses=s["doses"], wt=s.get("wt", 70.0), cov=s.get("cov"),
                covariate_effects=covariate_effects, error_model=error_model,
            )
            ind[s["subject"]] = m["individual_params"]
        except Exception as exc:
            n_fallback += 1
            _LOG.warning("map_estimate fell back to typical for subject %r "
                         "(model=%s): %s", s.get("subject"), model_key, exc)
            ind[s["subject"]] = theta  # fall back to typical prediction
    out = score_predictions(model_key, subjects, ind, theta,
                            omega_cv_pct, sigma_prop, sigma_add)
    # Surface the fallback count so a mis-fed score is distinguishable from a
    # genuinely poor fit (a full fallback collapses IPRED to PRED for all subjects).
    out["n_map_fallback"] = n_fallback
    return out


def score_predictions(model_key: str, subjects: list[dict],
                      individual_params_by_subject: dict, typical_params: dict,
                      iiv_cv_by_param: dict,
                      sigma_prop: float | None, sigma_add: float | None) -> dict[str, Any]:
    """Compute the engine-agnostic goodness fields from supplied predictions."""
    from app.compute.vpc import obs_vs_pred, pcvpc  # lazy: heavy

    ovp = obs_vs_pred(model_key, subjects, individual_params_by_subject, typical_params)
    gof = ovp.get("gof", {})
    try:
        pc = pcvpc(model_key, subjects, typical_params, iiv_cv_by_param,
                   sigma_prop=sigma_prop or 0.0, sigma_add=sigma_add or 0.0)
        cov90 = vpc_coverage(pc)
    except Exception:
        cov90 = None  # a pcVPC failure must not sink the whole score

    return {
        "pred_rmse": gof.get("rmse_log_ipred"),
        "pred_r2": gof.get("r2_log_ipred"),
        "pred_bias": _log_bias(ovp.get("observed", []), ovp.get("ipred", [])),
        "vpc_coverage90": cov90,
    }
