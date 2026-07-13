"""Dose proportionality — power-model assessment (Smith et al. 2000).

Pure functions, no agent/LLM dependencies, fully unit-testable. Implements the
power model ln(value) = intercept + slope*ln(dose) fitted by OLS, with a
(1 - alpha) confidence interval on the slope and the Smith et al. (2000)
critical-region acceptance rule based on the observed dose ratio.

Smith DA, Beaumont K, Maurer TS, Di L. "Confidence interval criteria for
assessment of dose proportionality." Pharm Res. 2000;17(10):1278-1283.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats


def _clean_pairs(
    doses: list[float], values: list[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Pair doses with values by index, dropping non-positive / None entries."""
    xs: list[float] = []
    ys: list[float] = []
    for d, v in zip(doses, values):
        if d is None or v is None:
            continue
        if d <= 0 or v <= 0:
            continue
        xs.append(float(d))
        ys.append(float(v))
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def power_model(
    doses: list[float],
    values: list[float],
    *,
    alpha: float = 0.10,
    theta_lower: float = 0.8,
    theta_upper: float = 1.25,
) -> dict[str, Any]:
    """Fit the power model and assess dose proportionality (Smith criterion).

    Fits ln(value) = intercept + slope*ln(dose) by OLS over positive
    (dose, value) pairs, computes a (1 - alpha) CI on the slope, and tests it
    against the Smith critical region derived from the observed dose ratio.
    """
    dose_arr, value_arr = _clean_pairs(doses, values)
    n = int(dose_arr.size)
    n_doses = int(np.unique(dose_arr).size)

    if n < 3 or n_doses < 2:
        return {
            "n": n,
            "slope": None,
            "slope_ci_lower": None,
            "slope_ci_upper": None,
            "intercept": None,
            "r_squared": None,
            "dose_ratio": None,
            "critical_region": None,
            "proportional": None,
            "alpha": alpha,
            "note": "need n>=3 pairs and at least 2 distinct doses",
        }

    x = np.log(dose_arr)
    y = np.log(value_arr)
    xbar = float(np.mean(x))
    ybar = float(np.mean(y))

    sxx = float(np.sum((x - xbar) ** 2))
    sxy = float(np.sum((x - xbar) * (y - ybar)))
    slope = sxy / sxx
    intercept = ybar - slope * xbar

    yhat = intercept + slope * x
    ssr = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - ybar) ** 2))
    r_squared = 1.0 - ssr / ss_tot if ss_tot > 0 else 1.0

    se_slope = math.sqrt((ssr / (n - 2)) / sxx)
    tcrit = float(stats.t.ppf(1 - alpha / 2.0, n - 2))
    ci_lower = slope - tcrit * se_slope
    ci_upper = slope + tcrit * se_slope

    dose_ratio = float(np.max(dose_arr) / np.min(dose_arr))
    ln_r = math.log(dose_ratio)
    crit_lower = 1.0 + math.log(theta_lower) / ln_r
    crit_upper = 1.0 + math.log(theta_upper) / ln_r

    proportional = bool(ci_lower >= crit_lower and ci_upper <= crit_upper)

    return {
        "n": n,
        "slope": round(slope, 6),
        "slope_ci_lower": round(ci_lower, 6),
        "slope_ci_upper": round(ci_upper, 6),
        "intercept": round(intercept, 6),
        "r_squared": round(r_squared, 6),
        "dose_ratio": round(dose_ratio, 6),
        "critical_region": [round(crit_lower, 6), round(crit_upper, 6)],
        "proportional": proportional,
        "alpha": alpha,
    }


def assess_dose_proportionality(
    doses: list[float],
    values_by_param: dict[str, list[float]],
    *,
    alpha: float = 0.10,
    theta_lower: float = 0.8,
    theta_upper: float = 1.25,
) -> dict[str, Any]:
    """Run the power model for each parameter and aggregate proportionality.

    `doses` aligns by index with each value list in `values_by_param`. Overall
    `proportional` is True only when every assessed parameter is proportional.
    """
    parameters: dict[str, Any] = {}
    for param, values in values_by_param.items():
        parameters[param] = power_model(
            doses,
            values,
            alpha=alpha,
            theta_lower=theta_lower,
            theta_upper=theta_upper,
        )

    assessed = [p["proportional"] for p in parameters.values()]
    overall = bool(assessed) and all(p is True for p in assessed)

    return {
        "alpha": alpha,
        "theta": [theta_lower, theta_upper],
        "parameters": parameters,
        "proportional": overall,
    }
