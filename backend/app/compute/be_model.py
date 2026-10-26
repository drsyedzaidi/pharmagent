"""Statistical estimators for average bioequivalence: the 2x2 crossover model and the parallel two-group comparison.

Crossover (two formulations, two periods, the two sequences TR and RT): fixed-effects linear model
on log(y) with sequence, subject within sequence, period and formulation. This is the model EMA CPMP/EWP/QWP/1401/98
Rev. 1 (sec. 4.1.8: "sequence, subject within sequence, period and formulation ... Fixed effects, rather than random
effects, should be used for all terms") names. ICH M13A (sec. 2.2.3.2) asks for "an appropriate parametric method,
e.g., general linear model (GLM) or mixed model" with tests of the same four effects; FDA (May 2026) describes a
mixed-effects or two-stage linear model with period and sequence terms. For COMPLETE 2x2 data (every analysed subject
has both periods, which the safety contract enforces) the subject effect cancels within subject, so the formulation
estimate and its standard error are expected not to depend on whether subject is fixed or random. That equivalence is
checked on the six crossover datasets in tests/fixtures/be_r_anova against R 4.5.1 (nlme::lme, REML): the estimate
agrees to 1e-10 or better and the standard error to the optimizer's tolerance (at most 1.5e-6 relative), with an
interior between-subject variance in every case. It is not unconditional: when the REML estimate of the
between-subject variance is zero, a mixed model collapses to ordinary least squares without a subject term and its
standard error differs; the fixed-effects model has no such case.

The formulation difference (T - R on the log scale) is the formulation coefficient of the full model; its standard
error uses the residual mean square on n - 2 degrees of freedom (n = number of subjects). The paired t-test used
before is NOT this: it has n - 1 df, leaves the period effect in the residual, and is biased when the sequences are
unbalanced and a period effect exists.

Parallel: independent samples on the log scale, pooled variance (n1 + n2 - 2 df) or Welch (Satterthwaite df); the
guidance texts read for this work do not choose one, so the choice is an explicit argument.

All functions are pure; every result carries the unrounded bounds (``*_raw``) the decision rule must use.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import stats

VARIANCES = ("pooled", "welch")
SEQUENCES = ("TR", "RT")          # test given first / reference given first
ANOVA_SOURCES = ("sequence", "subject(sequence)", "period", "formulation", "residual")


def _has_bool(values: Any) -> bool:
    items = values.ravel() if isinstance(values, np.ndarray) else list(values)
    return (isinstance(values, np.ndarray) and values.dtype == bool) or any(
        isinstance(v, (bool, np.bool_)) for v in items)


def _positive(name: str, values: Sequence[float]) -> np.ndarray:
    if _has_bool(values):
        raise ValueError(f"{name} must be positive numbers, not booleans")
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional list of numbers")
    if arr.size and (not np.all(np.isfinite(arr)) or np.any(arr <= 0)):
        raise ValueError(f"{name} must contain only finite positive values")
    return arr


def _flags(name: str, values: Sequence[bool]) -> np.ndarray:
    arr = np.asarray(values)
    if arr.dtype == bool:
        return arr
    if arr.dtype == object and arr.size and all(isinstance(v, (bool, np.bool_)) for v in arr.ravel()):
        return arr.astype(bool)
    if arr.dtype.kind in "iu" and np.all((arr == 0) | (arr == 1)):
        return arr.astype(bool)
    raise ValueError(f"{name} must be booleans or 0/1 integers (one per subject)")


def _rss(X: np.ndarray, y: np.ndarray) -> float:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return float(resid @ resid)


def _cv_intra_pct(mse: float) -> float | None:
    try:
        return round(100.0 * math.sqrt(math.expm1(mse)), 4)
    except OverflowError:
        return None


def _interval(est: float, se: float, df: float, alpha: float) -> tuple[float, float, float]:
    t = float(stats.t.ppf(1.0 - alpha / 2.0, df))
    try:
        return 100.0 * math.exp(est), 100.0 * math.exp(est - t * se), 100.0 * math.exp(est + t * se)
    except OverflowError:
        raise ValueError("the estimate or its interval is out of range for back-transformation "
                         "(the values differ by hundreds of orders of magnitude)") from None


def not_estimable(**extra: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "estimable": False, "estimate": None, "se": None, "mse": None, "gmr_raw": None, "gmr_pct": None,
        "ci_lower_raw": None, "ci_upper_raw": None, "ci_lower_pct": None, "ci_upper_pct": None,
        "cv_intra_pct": None, "anova": None}
    out.update(extra)
    return out


def _filled(est: float, se: float, df: float, mse: float | None, alpha: float, **extra: Any) -> dict[str, Any]:
    gmr, lo, hi = _interval(est, se, df, alpha)
    out: dict[str, Any] = {
        "estimable": True, "estimate": est, "se": se, "df": df, "mse": mse, "alpha": alpha,
        "gmr_raw": gmr, "ci_lower_raw": lo, "ci_upper_raw": hi,
        "gmr_pct": round(gmr, 4), "ci_lower_pct": round(lo, 4), "ci_upper_pct": round(hi, 4)}
    out.update(extra)
    return out


def _means(test: np.ndarray, ref: np.ndarray) -> dict[str, float | None]:
    def geo(a: np.ndarray) -> float | None:
        return float(math.exp(np.log(a).mean())) if a.size else None

    def arith(a: np.ndarray) -> float | None:
        return float(a.mean()) if a.size else None
    return {"geo_mean_test": geo(test), "geo_mean_ref": geo(ref),
            "arith_mean_test": arith(test), "arith_mean_ref": arith(ref)}


def crossover_2x2(test: Sequence[float], ref: Sequence[float], test_first: Sequence[bool], *,
                  alpha: float = 0.10) -> dict[str, Any]:
    """Average-BE estimate from a complete two-period crossover.

    ``test[i]`` and ``ref[i]`` are subject i's two positive values (raw scale); ``test_first[i]`` is True when
    subject i received the test product in the first period (sequence TR), False for RT. Both sequences must occur."""
    t, r = _positive("test", test), _positive("ref", ref)
    tf = _flags("test_first", test_first)
    if not (t.size == r.size == tf.size):
        raise ValueError("test, ref and test_first must have the same length (one entry per subject)")
    n = int(t.size)
    if n == 0:
        raise ValueError("no subjects")
    n_tr = int(tf.sum())
    n_rt = n - n_tr
    if n_tr == 0 or n_rt == 0:
        raise ValueError("a crossover needs subjects in both sequences (TR and RT): with one sequence, period and "
                         "formulation cannot be separated")
    extra = {"n": n, "n_per_sequence": {"TR": n_tr, "RT": n_rt}, **_means(t, r)}
    df = n - 2
    if df < 1:
        return not_estimable(df=max(df, 0), alpha=alpha, **extra)

    lt, lr = np.log(t), np.log(r)
    y = np.concatenate([np.where(tf, lt, lr), np.where(tf, lr, lt)])                 # first period, then second
    eye = np.eye(n)
    S = np.vstack([eye, eye])                                                        # subject dummies
    p_code = np.concatenate([np.full(n, -0.5), np.full(n, 0.5)])
    f_code = np.concatenate([np.where(tf, 0.5, -0.5), np.where(tf, -0.5, 0.5)])      # +0.5 where the test is given
    s_code = np.tile(np.where(tf, 0.5, -0.5), 2)
    ones = np.ones(2 * n)

    full = np.column_stack([S, p_code, f_code])
    Q, R = np.linalg.qr(full)
    beta = np.linalg.solve(R, Q.T @ y)
    resid = y - full @ beta
    rss_full = float(resid @ resid)
    mse = rss_full / df
    r_inv = np.linalg.inv(R)
    est = float(beta[-1])
    se = float(math.sqrt(mse * float(r_inv[-1] @ r_inv[-1])))

    ss = {
        "sequence": _rss(ones[:, None], y) - _rss(np.column_stack([ones, s_code]), y),
        "subject(sequence)": _rss(np.column_stack([ones, s_code]), y) - _rss(S, y),
        "period": _rss(np.column_stack([S, f_code]), y) - rss_full,
        "formulation": _rss(np.column_stack([S, p_code]), y) - rss_full,
        "residual": rss_full,
    }
    dfs = {"sequence": 1, "subject(sequence)": n - 2, "period": 1, "formulation": 1, "residual": df}
    total_ss = float(((y - y.mean()) ** 2).sum())
    # Data that are constant to within rounding have a centred total that is noise, so judge "no variation" against
    # the magnitude of the data, not against the noisy total itself.
    varies = total_ss > (1e-10 * float(np.abs(y).max())) ** 2 * y.size
    ss = {k: max(v, 0.0) for k, v in ss.items()}                  # rounding can leave a hair below zero
    ms = {k: ss[k] / dfs[k] for k in ss}
    errors = {"sequence": "subject(sequence)", "subject(sequence)": "residual", "period": "residual",
              "formulation": "residual", "residual": None}
    table = []
    for source in ANOVA_SOURCES:
        err = errors[source]
        f_val = p_val = None
        # no F-test when the error term is numerically zero relative to the data (constant or exactly fitted data)
        informative = err is not None and varies and ms[err] > 1e-12 * total_ss / (2 * n - 1)
        if informative:
            f_val = ms[source] / ms[err]
            p_val = float(stats.f.sf(f_val, dfs[source], dfs[err]))
        table.append({"source": source, "df": dfs[source], "ss": ss[source], "ms": ms[source], "f": f_val,
                      "p": p_val, "error_term": err})
    return _filled(est, se, df, mse, alpha, cv_intra_pct=_cv_intra_pct(mse), anova=table, **extra)


def parallel_two_group(test: Sequence[float], ref: Sequence[float], *, alpha: float = 0.10,
                       variance: str = "pooled") -> dict[str, Any]:
    """Average-BE estimate from two independent groups (raw positive values per subject)."""
    if variance not in VARIANCES:
        raise ValueError(f"variance must be one of {VARIANCES}; got {variance!r}")
    t, r = _positive("test", test), _positive("ref", ref)
    n1, n2 = int(t.size), int(r.size)
    extra = {"n_test": n1, "n_reference": n2, "variance": variance, **_means(t, r)}
    if n1 < 2 or n2 < 2:
        return not_estimable(df=0, alpha=alpha, **extra)
    lt, lr = np.log(t), np.log(r)
    v1, v2 = float(np.var(lt, ddof=1)), float(np.var(lr, ddof=1))
    est = float(lt.mean() - lr.mean())
    if variance == "pooled":
        df = n1 + n2 - 2
        pooled = ((n1 - 1) * v1 + (n2 - 1) * v2) / df
        se, mse = math.sqrt(pooled * (1.0 / n1 + 1.0 / n2)), pooled
    else:
        a, b = v1 / n1, v2 / n2
        se = math.sqrt(a + b)
        denom = a * a / (n1 - 1) + b * b / (n2 - 1)
        df, mse = ((a + b) ** 2 / denom if denom > 0 else float(n1 + n2 - 2)), None
    return _filled(est, se, df, mse, alpha, cv_intra_pct=None, anova=None, **extra)
