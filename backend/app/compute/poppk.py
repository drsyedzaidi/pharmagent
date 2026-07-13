"""Two-stage (STS) population PK summary — deterministic compute.

Pure functions, no agent/LLM dependencies, fully unit-testable. This module
implements a *two-stage approximation* to population PK summarization:

    1. Stage one (assumed already done by the caller): individual PK parameters
       are estimated per subject (e.g. via NCA in `nca.py` or per-subject curve
       fitting).
    2. Stage two (this module): those individual estimates are pooled into
       population-level descriptors.

The "typical value" of a parameter is reported as the *geometric mean* of the
individual estimates, and between-subject variability (IIV) is reported as the
*geometric coefficient of variation* (geometric CV%). This is a descriptive
summary only — it is NOT a mixed-effects / NLME estimation. In particular it is
NOT FOCE, NOT SAEM, and does NOT shrink individual estimates toward a
population mean. Standard errors, eta-shrinkage, and residual error are not
estimated here. For a true population model use a dedicated NLME engine.

Functions:
    _geomean, _geocv_pct, two_stage_summary, covariate_effect
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats


def _geomean(vals: list[float]) -> float | None:
    """Geometric mean of strictly positive values, or None if none usable.

    Non-positive, None, or non-finite values are dropped.
    """
    clean = [float(v) for v in vals
             if v is not None and np.isfinite(v) and v > 0]
    if not clean:
        return None
    return float(math.exp(sum(math.log(v) for v in clean) / len(clean)))


def _geocv_pct(vals: list[float]) -> float | None:
    """Geometric CV% = 100 * sqrt(exp(var(ln(x))) - 1), ddof=1.

    Requires at least two strictly positive values; otherwise returns None.
    """
    clean = [float(v) for v in vals
             if v is not None and np.isfinite(v) and v > 0]
    if len(clean) < 2:
        return None
    logs = np.log(np.asarray(clean, dtype=float))
    var = float(np.var(logs, ddof=1))
    return float(100.0 * math.sqrt(math.exp(var) - 1.0))


def _round(x: float | None, ndigits: int = 6) -> float | None:
    """Round a finite number; pass through None / non-numbers unchanged."""
    if isinstance(x, (int, float)) and np.isfinite(x):
        return round(float(x), ndigits)
    return x


def two_stage_summary(individual_params: list[dict], *,
                      keys: tuple = ("CL_F", "Vz_F", "ka")) -> dict[str, Any]:
    """Pool per-subject estimates into a two-stage (STS) population summary.

    For each requested key, the strictly positive individual values are gathered
    across subjects (missing / None / non-positive entries are skipped). Keys
    with no usable values are omitted from the returned ``parameters`` map.

    Args:
        individual_params: list of per-subject dicts of point estimates.
        keys: parameter names to summarize.

    Returns:
        dict with EXACTLY the keys ``method``, ``n_subjects``, ``parameters``.
        Each entry in ``parameters`` has EXACTLY ``typical_value``,
        ``iiv_cv_pct``, ``median``, ``n``.
    """
    parameters: dict[str, Any] = {}
    for key in keys:
        vals = [row.get(key) for row in individual_params]
        clean = [float(v) for v in vals
                 if v is not None and np.isfinite(v) and v > 0]
        if not clean:
            continue
        parameters[key] = {
            "typical_value": _round(_geomean(clean)),
            "iiv_cv_pct": _round(_geocv_pct(clean)),
            "median": _round(float(np.median(clean))),
            "n": len(clean),
        }

    return {
        "method": "two-stage (STS)",
        "n_subjects": len(individual_params),
        "parameters": parameters,
    }


def covariate_effect(individual_params: list[dict], covariate_by_subject: dict,
                     *, param_key: str, subject_key: str = "subject"
                     ) -> dict[str, Any]:
    """Regress ln(param) on a continuous covariate across subjects.

    Pairs each subject's strictly positive parameter value with its covariate
    (matched by ``subject_key``), then fits an ordinary least-squares line of
    ln(param) versus covariate. Requires at least three paired, finite points;
    otherwise ``slope``, ``pearson_r`` and ``r_squared`` are None.

    The slope is the change in ln(param) per unit covariate (i.e. an
    approximate fractional change per unit for small effects).

    Returns:
        dict with EXACTLY ``param``, ``n``, ``slope``, ``pearson_r``,
        ``r_squared``.
    """
    cov_list: list[float] = []
    log_param: list[float] = []
    for row in individual_params:
        sid = row.get(subject_key)
        if sid is None or sid not in covariate_by_subject:
            continue
        pval = row.get(param_key)
        cov = covariate_by_subject[sid]
        if pval is None or cov is None:
            continue
        if not (np.isfinite(pval) and pval > 0 and np.isfinite(cov)):
            continue
        cov_list.append(float(cov))
        log_param.append(math.log(float(pval)))

    n = len(cov_list)
    if n < 3:
        return {
            "param": param_key,
            "n": n,
            "slope": None,
            "pearson_r": None,
            "r_squared": None,
        }

    x = np.asarray(cov_list, dtype=float)
    y = np.asarray(log_param, dtype=float)
    slope, _intercept = np.polyfit(x, y, 1)
    r, _p = stats.pearsonr(x, y)
    r = float(r)
    return {
        "param": param_key,
        "n": n,
        "slope": _round(float(slope)),
        "pearson_r": _round(r),
        "r_squared": _round(r * r),
    }
