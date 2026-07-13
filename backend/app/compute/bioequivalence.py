"""Average bioequivalence (ABE) — deterministic compute.

Pure functions, no agent/LLM dependencies, fully unit-testable. Implements the
standard two one-sided tests (TOST) confidence-interval approach on the log
scale for a single PK parameter (e.g. Cmax, AUC), for both crossover (paired)
and parallel two-group designs.

The geometric mean ratio (GMR) and its (1 - alpha) confidence interval are
computed on the natural-log scale and back-transformed via exp(). A parameter
is bioequivalent when the entire CI lies within the regulatory limits
(default 80-125%).

Per-parameter keys returned:
    n_test, n_ref, gmr_pct, ci_lower_pct, ci_upper_pct, within_limits,
    cv_intra_pct, alpha, limits
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats


def _clean_logs(vals: list[float]) -> np.ndarray:
    """Return ln() of strictly-positive, non-None values as a float array."""
    clean = [float(v) for v in vals if v is not None and v > 0]
    return np.log(np.asarray(clean, dtype=float)) if clean else np.empty(0)


def _null_result(n_test: int, n_ref: int, alpha: float,
                 lower: float, upper: float) -> dict[str, Any]:
    """Result shape when there are too few observations to assess BE."""
    return {
        "n_test": n_test,
        "n_ref": n_ref,
        "gmr_pct": None,
        "ci_lower_pct": None,
        "ci_upper_pct": None,
        "within_limits": False,
        "cv_intra_pct": None,
        "alpha": alpha,
        "limits": [lower, upper],
    }


def _ci_from_log(mean_diff: float, se: float, df: int, alpha: float
                 ) -> tuple[float, float, float]:
    """Back-transform a log-scale point estimate + SE to GMR% and CI%."""
    tcrit = float(stats.t.ppf(1.0 - alpha / 2.0, df))
    gmr_pct = 100.0 * math.exp(mean_diff)
    ci_lower_pct = 100.0 * math.exp(mean_diff - tcrit * se)
    ci_upper_pct = 100.0 * math.exp(mean_diff + tcrit * se)
    return gmr_pct, ci_lower_pct, ci_upper_pct


def be_one_parameter(test_vals: list[float], ref_vals: list[float], *,
                     paired: bool, alpha: float = 0.10,
                     lower: float = 80.0, upper: float = 125.0) -> dict[str, Any]:
    """Average BE assessment for a single PK parameter via TOST/CI on log scale.

    paired=True: crossover — test_vals[i] and ref_vals[i] are the same subject.
    paired=False: parallel — the two groups are independent.

    Returns a dict with point estimate, CI, and a within-limits verdict. If
    there are too few observations the estimate fields are None and
    within_limits is False.
    """
    if paired:
        return _be_paired(test_vals, ref_vals, alpha=alpha,
                          lower=lower, upper=upper)
    return _be_parallel(test_vals, ref_vals, alpha=alpha,
                        lower=lower, upper=upper)


def _be_paired(test_vals: list[float], ref_vals: list[float], *,
               alpha: float, lower: float, upper: float) -> dict[str, Any]:
    """Crossover point estimate from per-subject log differences."""
    pairs = [
        (float(t), float(r))
        for t, r in zip(test_vals, ref_vals)
        if t is not None and r is not None and t > 0 and r > 0
    ]
    n = len(pairs)
    if n < 2:
        return _null_result(n, n, alpha, lower, upper)

    diffs = np.array([math.log(t) - math.log(r) for t, r in pairs], dtype=float)
    mean_d = float(np.mean(diffs))
    sd_d = float(np.std(diffs, ddof=1))
    se = sd_d / math.sqrt(n)

    gmr_pct, ci_lower_pct, ci_upper_pct = _ci_from_log(mean_d, se, n - 1, alpha)
    cv_intra_pct = 100.0 * math.sqrt(math.exp(sd_d * sd_d) - 1.0)

    return {
        "n_test": n,
        "n_ref": n,
        "gmr_pct": round(gmr_pct, 4),
        "ci_lower_pct": round(ci_lower_pct, 4),
        "ci_upper_pct": round(ci_upper_pct, 4),
        "within_limits": (ci_lower_pct >= lower) and (ci_upper_pct <= upper),
        "cv_intra_pct": round(cv_intra_pct, 4),
        "alpha": alpha,
        "limits": [lower, upper],
    }


def _be_parallel(test_vals: list[float], ref_vals: list[float], *,
                 alpha: float, lower: float, upper: float) -> dict[str, Any]:
    """Parallel two-group estimate from pooled-variance log means."""
    log_t = _clean_logs(test_vals)
    log_r = _clean_logs(ref_vals)
    n1, n2 = log_t.size, log_r.size
    if n1 < 2 or n2 < 2:
        return _null_result(n1, n2, alpha, lower, upper)

    var1 = float(np.var(log_t, ddof=1))
    var2 = float(np.var(log_r, ddof=1))
    df = n1 + n2 - 2
    sp = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / df)
    se = sp * math.sqrt(1.0 / n1 + 1.0 / n2)
    diff = float(np.mean(log_t)) - float(np.mean(log_r))

    gmr_pct, ci_lower_pct, ci_upper_pct = _ci_from_log(diff, se, df, alpha)

    return {
        "n_test": n1,
        "n_ref": n2,
        "gmr_pct": round(gmr_pct, 4),
        "ci_lower_pct": round(ci_lower_pct, 4),
        "ci_upper_pct": round(ci_upper_pct, 4),
        "within_limits": (ci_lower_pct >= lower) and (ci_upper_pct <= upper),
        "cv_intra_pct": None,
        "alpha": alpha,
        "limits": [lower, upper],
    }


def assess_bioequivalence(test_by_param: dict[str, list[float]],
                          ref_by_param: dict[str, list[float]], *,
                          paired: bool, alpha: float = 0.10,
                          lower: float = 80.0, upper: float = 125.0
                          ) -> dict[str, Any]:
    """Run average-BE on every parameter present in both input dicts.

    The overall study is bioequivalent only when every assessed parameter's
    confidence interval lies within the limits.
    """
    params: dict[str, Any] = {}
    for name in test_by_param:
        if name not in ref_by_param:
            continue
        params[name] = be_one_parameter(
            test_by_param[name], ref_by_param[name],
            paired=paired, alpha=alpha, lower=lower, upper=upper,
        )

    bioequivalent = bool(params) and all(
        res["within_limits"] for res in params.values()
    )

    return {
        "design": "crossover (paired)" if paired else "parallel",
        "alpha": alpha,
        "limits": [lower, upper],
        "parameters": params,
        "bioequivalent": bioequivalent,
    }
