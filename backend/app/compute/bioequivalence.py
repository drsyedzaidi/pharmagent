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
    cv_intra_pct, alpha, limits, and the unrounded gmr_raw, ci_lower_raw, ci_upper_raw
    (the *_pct values are rounded to 4 decimals for display only)
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
        "gmr_raw": None,
        "ci_lower_raw": None,
        "ci_upper_raw": None,
    }


def _ci_from_log(mean_diff: float, se: float, df: int, alpha: float
                 ) -> tuple[float, float, float]:
    """Back-transform a log-scale point estimate + SE to GMR% and CI%."""
    tcrit = float(stats.t.ppf(1.0 - alpha / 2.0, df))
    gmr_pct = 100.0 * math.exp(mean_diff)
    ci_lower_pct = 100.0 * math.exp(mean_diff - tcrit * se)
    ci_upper_pct = 100.0 * math.exp(mean_diff + tcrit * se)
    return gmr_pct, ci_lower_pct, ci_upper_pct


PAIRED_LABEL = "crossover (paired, no period/sequence adjustment)"
MODEL_2X2 = "2x2 crossover: sequence, subject(sequence), period, treatment"


def be_one_parameter(test_vals: list[float], ref_vals: list[float], *,
                     paired: bool, alpha: float = 0.10,
                     lower: float = 80.0, upper: float = 125.0,
                     sequences: list[Any] | None = None,
                     test_periods: list[Any] | None = None,
                     ref_periods: list[Any] | None = None) -> dict[str, Any]:
    """Average BE assessment for a single PK parameter via TOST/CI on log scale.

    paired=True: crossover — test_vals[i] and ref_vals[i] are the same subject.
    paired=False: parallel — the two groups are independent.

    Returns a dict with point estimate, CI, and a within-limits verdict. If
    there are too few observations the estimate fields are None and
    within_limits is False.

    With ``sequences`` and ``test_periods`` (aligned to the value lists; ``ref_periods`` defaults to the other of two
    period levels) the crossover is fitted as a 2x2 fixed-effects model, see ``_be_crossover_2x2``.
    """
    if paired and sequences is not None and test_periods is not None:
        return _be_crossover_2x2(test_vals, ref_vals, sequences, test_periods, ref_periods,
                                 alpha=alpha, lower=lower, upper=upper)
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
        "df": n - 1,
        "gmr_pct": round(gmr_pct, 4),
        "ci_lower_pct": round(ci_lower_pct, 4),
        "ci_upper_pct": round(ci_upper_pct, 4),
        "within_limits": (ci_lower_pct >= lower) and (ci_upper_pct <= upper),
        "cv_intra_pct": round(cv_intra_pct, 4),
        "alpha": alpha,
        "limits": [lower, upper],
        "gmr_raw": gmr_pct,
        "ci_lower_raw": ci_lower_pct,
        "ci_upper_raw": ci_upper_pct,
    }


def _be_crossover_2x2(test_vals: list[float], ref_vals: list[float], sequences: list[Any],
                      test_periods: list[Any], ref_periods: list[Any] | None, *,
                      alpha: float, lower: float, upper: float) -> dict[str, Any]:
    """Standard 2x2 crossover ANOVA on log(parameter), completers only.

    ln(y) ~ subject(sequence) + period + treatment, fitted by least squares (fixed effects; sequence is absorbed by
    the nested subject term, so it needs no column of its own). The treatment contrast T - R, its standard error from
    the residual mean square and df = N_obs - rank (= n1 + n2 - 2 for a complete 2x2) give the 90% CI; CV_intra comes
    from the MSE. Unlike the plain paired t on log differences this removes a period effect and stays unbiased when
    the sequences are unbalanced. Subject i supplies one test and one reference value; ``test_periods[i]`` is the
    period the test was given in."""
    levels = sorted({p for p in test_periods if p is not None}, key=str)
    if ref_periods is None:
        if len(levels) != 2:
            raise ValueError("ref_periods are required unless the test periods take exactly two values")
        other = {levels[0]: levels[1], levels[1]: levels[0]}
        ref_periods = [other.get(p) for p in test_periods]
    keep = [i for i, (t, r, q, tp, rp) in enumerate(zip(test_vals, ref_vals, sequences, test_periods, ref_periods,
                                                      strict=True))
            if t is not None and r is not None and t > 0 and r > 0 and q is not None
            and tp is not None and rp is not None]
    n = len(keep)
    seqs = sorted({str(sequences[i]) for i in keep})
    per_levels = sorted({test_periods[i] for i in keep} | {ref_periods[i] for i in keep}, key=str)
    if n < 2 or len(seqs) < 2 or len(per_levels) < 2:
        return _null_result(n, n, alpha, lower, upper)

    rows, y = [], []
    for col, i in enumerate(keep):
        for trt, val, per in ((1.0, test_vals[i], test_periods[i]), (0.0, ref_vals[i], ref_periods[i])):
            x = np.zeros(n + (len(per_levels) - 1) + 1)
            x[col] = 1.0                                              # subject (nested in its sequence)
            k = per_levels.index(per)
            if k > 0:
                x[n + k - 1] = 1.0                                    # period
            x[-1] = trt                                               # treatment
            rows.append(x)
            y.append(math.log(float(val)))
    X, yv = np.asarray(rows), np.asarray(y)
    rank = int(np.linalg.matrix_rank(X))
    if rank != int(np.linalg.matrix_rank(X[:, :-1])) + 1:             # treatment not separable (e.g. one sequence)
        return _null_result(n, n, alpha, lower, upper)
    df = len(yv) - rank
    if df < 1:
        return _null_result(n, n, alpha, lower, upper)
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    resid = yv - X @ beta
    mse = float(resid @ resid) / df
    se = math.sqrt(max(mse, 0.0) * float(np.linalg.pinv(X.T @ X)[-1, -1]))
    gmr_pct, ci_lower_pct, ci_upper_pct = _ci_from_log(float(beta[-1]), se, df, alpha)
    cv_intra_pct = 100.0 * math.sqrt(math.exp(mse) - 1.0)
    return {
        "n_test": n,
        "n_ref": n,
        "df": df,
        "model": MODEL_2X2,
        "n_per_sequence": {q: sum(1 for i in keep if str(sequences[i]) == q) for q in seqs},
        "gmr_pct": round(gmr_pct, 4),
        "ci_lower_pct": round(ci_lower_pct, 4),
        "ci_upper_pct": round(ci_upper_pct, 4),
        "within_limits": (ci_lower_pct >= lower) and (ci_upper_pct <= upper),
        "cv_intra_pct": round(cv_intra_pct, 4),
        "alpha": alpha,
        "limits": [lower, upper],
        # unrounded values: a decision rule must round these once, never the 4-dp display copies above
        "gmr_raw": gmr_pct,
        "ci_lower_raw": ci_lower_pct,
        "ci_upper_raw": ci_upper_pct,
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
        "df": df,
        "gmr_pct": round(gmr_pct, 4),
        "ci_lower_pct": round(ci_lower_pct, 4),
        "ci_upper_pct": round(ci_upper_pct, 4),
        "within_limits": (ci_lower_pct >= lower) and (ci_upper_pct <= upper),
        "cv_intra_pct": None,
        "alpha": alpha,
        "limits": [lower, upper],
        # unrounded values: a decision rule must round these once, never the 4-dp display copies above
        "gmr_raw": gmr_pct,
        "ci_lower_raw": ci_lower_pct,
        "ci_upper_raw": ci_upper_pct,
    }


def assess_bioequivalence(test_by_param: dict[str, list[float]],
                          ref_by_param: dict[str, list[float]], *,
                          paired: bool, alpha: float = 0.10,
                          lower: float = 80.0, upper: float = 125.0,
                          crossover: dict[str, list[Any]] | None = None
                          ) -> dict[str, Any]:
    """Run average-BE on every parameter present in both input dicts.

    ``crossover`` = {"sequences", "test_periods"[, "ref_periods"]}, aligned to the value lists, selects the 2x2
    crossover model; without it a paired design is the plain paired t on log differences and says so in ``design``.

    The overall study is bioequivalent only when every assessed parameter's
    confidence interval lies within the limits.
    """
    params: dict[str, Any] = {}
    for name in test_by_param:
        if name not in ref_by_param:
            continue
        params[name] = be_one_parameter(
            test_by_param[name], ref_by_param[name],
            paired=paired, alpha=alpha, lower=lower, upper=upper, **(crossover or {}),
        )

    bioequivalent = bool(params) and all(
        res["within_limits"] for res in params.values()
    )

    adjusted = bool(paired and crossover)
    out_design = ("2x2 crossover (sequence, subject(sequence), period, treatment)" if adjusted
                  else PAIRED_LABEL if paired else "parallel")
    return {
        "design": out_design,
        "alpha": alpha,
        "limits": [lower, upper],
        "parameters": params,
        "bioequivalent": bioequivalent,
    }
