"""Covariate forest plot: parameter-level GMR + CI per fitted covariate effect.

Pure, deterministic compute on top of an ALREADY-FITTED NLME result (never
re-fits, never touches the dataset). One row per (parameter, covariate,
evaluation point): the geometric mean ratio (GMR) of the structural parameter
at that covariate value versus the model's own reference (the fitted
``center`` for a continuous covariate, or the reference level for a
categorical one), with a large-sample Wald confidence interval on the log
scale.

Scope: parameter-level GMR only (``theta_p(covariate) / theta_p(reference)``),
NOT a simulated exposure-metric (AUC/Cmax) ratio. Exposure translation
requires picking a regimen and correctly composing the fitted covariate effect
with the model's *separate* allometric weight scaling (``pk_simulate.scale_
params``) -- both are real sources of a silently wrong number, and are left as
explicit future work (flagged via ``allometric_note`` below) rather than
shipped half-verified. Parameter-level GMR is exactly what a stepwise
covariate model estimates and is unambiguous.

CI derivation: the fitted result exposes no covariate covariance MATRIX (only
each coefficient's own RSE%, ``nlme.py:_parameter_uncertainty``), so
``SE(beta) = rse_pct/100 * |beta|`` -- the diagonal only. The interval is a
first-order (delta-method) Wald interval on the log scale, NOT profile-
likelihood, NOT bootstrap, and does NOT account for the covariate-selection
step when the effect came from stepwise search (SCM) -- callers should surface
the fit's own ``selection_caveat`` alongside these rows.

All returned floats are exact JSON-safe Python (rounded to 6 dp; never NaN or
Infinity). A row that cannot be evaluated (degenerate extrapolation, missing
RSE) reports ``gmr=None`` with a ``ci_source`` explaining why, rather than a
fabricated point estimate.
"""
from __future__ import annotations

import math
from typing import Any

from scipy.stats import norm

from app.compute.pk_models import get_model

_ROUND_DP = 6
_EPS = 1e-12
# Column-name heuristic for "this covariate is a body-weight measure", mirrors
# app.tools.pkmodel_tools._WT_NAMES -- kept as a local literal (not imported)
# since this module must stay free of any tools/agent dependency.
_WT_NAMES = {"wt", "weight", "bw", "bwt", "bodyweight"}


def _z(ci_level: float) -> float:
    return float(norm.ppf(0.5 + ci_level / 2.0))


def _safe_round(x: float | None) -> float | None:
    if x is None or not math.isfinite(x):
        return None
    return round(float(x), _ROUND_DP)


def _row_for_continuous(eff: dict[str, Any], value: float, label: str, z: float,
                        allometric_note: bool, omega_cv: float | None) -> dict[str, Any]:
    kind = eff["kind"]
    center = float(eff.get("center") or 0.0)
    beta = float(eff["coefficient"])
    rse_pct = eff.get("rse_pct")
    se_beta = (abs(beta) * float(rse_pct) / 100.0) if rse_pct is not None else None

    d = value - center
    if kind == "power":
        v = max(value, _EPS)
        c = max(center, _EPS)
        ln_gmr = beta * math.log(v / c)
        d_ln_d_beta = math.log(v / c)
        ci_source = "wald_loglinear"
    elif kind == "exponential":
        ln_gmr = beta * d
        d_ln_d_beta = d
        ci_source = "wald_loglinear"
    elif kind == "linear":
        g = 1.0 + beta * d
        if g <= 0.0:
            return {
                "param": eff["param"], "covariate": eff["covariate"], "kind": kind,
                "eval_label": label, "eval_value": _safe_round(value),
                "gmr": None, "ci_lo": None, "ci_hi": None,
                "ci_source": "undefined_extrapolation",
                "omega_cv_pct": omega_cv, "allometric_note": allometric_note,
            }
        ln_gmr = math.log(g)
        d_ln_d_beta = d / g  # d ln(1+beta*d)/dbeta, first-order (delta method)
        ci_source = "delta_nonlinear"
    else:
        return {
            "param": eff["param"], "covariate": eff["covariate"], "kind": kind,
            "eval_label": label, "eval_value": _safe_round(value),
            "gmr": None, "ci_lo": None, "ci_hi": None, "ci_source": "unavailable",
            "omega_cv_pct": omega_cv, "allometric_note": allometric_note,
        }

    gmr = math.exp(ln_gmr)
    if se_beta is None:
        ci_lo = ci_hi = None
        ci_source = "unavailable"
    else:
        se_ln = abs(d_ln_d_beta) * se_beta
        ci_lo = math.exp(ln_gmr - z * se_ln)
        ci_hi = math.exp(ln_gmr + z * se_ln)

    return {
        "param": eff["param"], "covariate": eff["covariate"], "kind": kind,
        "eval_label": label, "eval_value": _safe_round(value),
        "gmr": _safe_round(gmr), "ci_lo": _safe_round(ci_lo), "ci_hi": _safe_round(ci_hi),
        "ci_source": ci_source, "omega_cv_pct": omega_cv, "allometric_note": allometric_note,
    }


def _row_for_categorical(eff: dict[str, Any], level: str, z: float,
                         ref_level: str | None, omega_cv: float | None) -> dict[str, Any]:
    coef_map: dict[str, float] = eff["coefficient"] or {}
    rse_map: dict[str, float | None] = eff.get("rse_pct") or {}
    ref_label = f" vs {ref_level}" if ref_level else " vs reference (unlabeled)"

    if level not in coef_map:
        # Level IS the reference (or unseen at fit time) -> GMR = 1 exactly.
        return {
            "param": eff["param"], "covariate": eff["covariate"], "kind": "categorical",
            "eval_label": f"{eff['covariate']}={level} (reference)",
            "eval_value": level, "gmr": 1.0, "ci_lo": 1.0, "ci_hi": 1.0,
            "ci_source": "reference", "omega_cv_pct": omega_cv, "allometric_note": False,
        }

    beta = float(coef_map[level])
    rse_pct = rse_map.get(level)
    gmr = math.exp(beta)
    if rse_pct is None:
        ci_lo = ci_hi = None
        ci_source = "unavailable"
    else:
        se_beta = abs(beta) * float(rse_pct) / 100.0
        ci_lo = math.exp(beta - z * se_beta)
        ci_hi = math.exp(beta + z * se_beta)
        ci_source = "wald_loglinear"

    return {
        "param": eff["param"], "covariate": eff["covariate"], "kind": "categorical",
        "eval_label": f"{eff['covariate']}={level}{ref_label}",
        "eval_value": level, "gmr": _safe_round(gmr),
        "ci_lo": _safe_round(ci_lo), "ci_hi": _safe_round(ci_hi),
        "ci_source": ci_source, "omega_cv_pct": omega_cv, "allometric_note": False,
    }


def covariate_forest(nlme: dict[str, Any], *,
                     cov_values: dict[str, list] | None = None,
                     ref_levels: dict[str, str] | None = None,
                     ci_level: float = 0.90,
                     bounds: tuple[float, float] | None = None) -> dict[str, Any]:
    """Forest-plot rows for every fitted covariate effect in ``nlme``.

    Parameters
    ----------
    nlme:
        A converged fit result carrying ``model_key``, ``omega_cv_pct``, and
        ``covariate_effects`` (the public per-effect records from
        ``app.compute.nlme._covariate_records`` / ``scm``'s ``final``).
    cov_values:
        ``{covariate: [values...]}`` -- the points to evaluate. For a
        continuous covariate these are typically the 5th/95th percentile of
        that covariate IN THE FITTED DATASET (supplied by the caller, since
        this function has no dataset access and must not silently mix a
        percentile computed from a different population than ``center``).
        For a categorical covariate, every level to display (including the
        reference, to render a GMR=1 anchor row). A covariate absent from
        ``cov_values`` is evaluated at ``center`` only (GMR = 1, informational).
    ref_levels:
        ``{covariate: reference_level_name}``. The fitted result does not
        itself record which level was the reference (`nlme.py`'s
        `_build_cov_effects` derives it per-fit from the modal level in the
        training data and never stores the name) -- supply it when known so
        categorical row labels are meaningful ("GENOTYPE=PM vs EM") rather
        than "vs reference (unlabeled)".
    ci_level:
        Confidence level for the Wald interval, e.g. 0.90 for a 90% CI
        (the FDA/EMA convention for covariate forest plots). Must be in
        (0, 1).
    bounds:
        Optional user-supplied reference band (lo, hi) on the GMR scale, e.g.
        a no-dose-adjustment boundary justified from exposure-response. NOT a
        default: an unjustified band (such as the 0.8-1.25 bioequivalence
        acceptance interval, which answers a different question) would read
        as a clinical-significance conclusion the analysis does not support.
        When supplied, must satisfy ``0 < lo < hi``.

    Returns
    -------
    ``{"rows": [...], "x_range": [lo, hi] | None, "bounds": [lo, hi] | None,
    "ci_level": float, "notes": [str, ...], "summary": {"n_rows", "n_effects"}}``.
    Every row's ``ci_lo``/``ci_hi``/``gmr`` may be ``None`` -- callers must
    check for that before comparing to ``bounds`` (never compare ``None`` to
    a bound directly).
    """
    if not (0.0 < ci_level < 1.0):
        raise ValueError(f"ci_level must be in (0, 1), got {ci_level!r}")
    if bounds is not None:
        lo_b, hi_b = float(bounds[0]), float(bounds[1])
        if not (0.0 < lo_b < hi_b):
            raise ValueError(f"bounds must satisfy 0 < lo < hi, got {bounds!r}")
        bounds = (lo_b, hi_b)

    effects: list[dict[str, Any]] = list(nlme.get("covariate_effects") or [])
    model_key = nlme.get("model_key")
    model = get_model(model_key) if model_key else None
    omega_cv_by_param = nlme.get("omega_cv_pct") or {}
    z = _z(ci_level)
    cov_values = cov_values or {}
    ref_levels = ref_levels or {}

    rows: list[dict[str, Any]] = []
    notes: list[str] = []
    seen_allometric = False

    for eff in effects:
        param = eff["param"]
        covariate = eff["covariate"]
        omega_cv = omega_cv_by_param.get(param)
        allometric_note = bool(
            model and covariate.strip().lower() in _WT_NAMES
            and param in model.allometric)
        if allometric_note:
            seen_allometric = True

        if eff["kind"] == "categorical":
            levels = cov_values.get(covariate)
            if not levels:
                levels = list(eff.get("levels") or [])
            ref = ref_levels.get(covariate)
            for lv in levels:
                rows.append(_row_for_categorical(eff, str(lv), z, ref, omega_cv))
            continue

        values = cov_values.get(covariate)
        if not values:
            center = float(eff.get("center") or 0.0)
            rows.append(_row_for_continuous(eff, center, f"{covariate}={center:g} (center)",
                                            z, allometric_note, omega_cv))
            continue
        for v in values:
            v = float(v)
            label = f"{covariate}={v:g}"
            rows.append(_row_for_continuous(eff, v, label, z, allometric_note, omega_cv))

    if seen_allometric:
        notes.append("One or more covariate effects act on a body-weight-scaled parameter "
                     "(model.allometric): the fitted covariate coefficient's GMR is the "
                     "ESTIMATED effect only and does NOT include the model's separate "
                     "allometric weight scaling — see the `allometric_note` flag per row.")
    if bounds is not None:
        notes.append("`bounds` is a user-supplied reference band and is NOT a bioequivalence "
                     "criterion; a no-effect boundary must be independently justified from "
                     "exposure-response, not assumed.")
    if any(r["ci_source"] == "undefined_extrapolation" for r in rows):
        notes.append("One or more `linear`-kind covariate effects extrapolate past the point "
                     "where 1 + beta*(value-center) <= 0 (the fitted model predicts a "
                     "non-positive parameter) — those rows report gmr=None.")

    finite_bounds = [r["gmr"] for r in rows if r["gmr"] is not None]
    finite_bounds += [r["ci_lo"] for r in rows if r["ci_lo"] is not None]
    finite_bounds += [r["ci_hi"] for r in rows if r["ci_hi"] is not None]
    x_range = ([round(0.9 * min(finite_bounds), _ROUND_DP), round(1.1 * max(finite_bounds), _ROUND_DP)]
              if finite_bounds else None)

    for r in rows:
        r["outside_reference_band"] = (
            bounds is not None and r["ci_lo"] is not None and r["ci_hi"] is not None
            and (r["ci_lo"] > bounds[1] or r["ci_hi"] < bounds[0]))

    return {
        "rows": rows,
        "x_range": x_range,
        "bounds": (list(bounds) if bounds is not None else None),
        "ci_level": round(float(ci_level), 4),
        "notes": notes,
        "summary": {"n_rows": len(rows), "n_effects": len(effects)},
    }
