"""Seeded non-parametric bootstrap for the exposure-response fits.

Same contract and conventions as ``app.compute.bootstrap`` (the NLME bootstrap):

* whole SUBJECTS are resampled with replacement (``_resample_indices``,
  optionally stratified so every dose group keeps its size);
* the RNG seed is an INPUT and is echoed in the result, so a bootstrap can be
  re-run from its audit record;
* failed replicates (a resample that is separated, single-class, or has too few
  events) are COUNTED with a reason, never silently dropped -- they fail
  preferentially on awkward resamples, so dropping them narrows the interval
  exactly where the data are weakest;
* percentile intervals are reported over successful replicates, beside the
  success rate, and a low rate is itself flagged.

Unlike the NLME bootstrap this one is cheap (a logistic or Cox refit is
milliseconds), but a few thousand replicates still holds the session lock, so
the TOOL wrapper (er_tools.bootstrap_exposure_response) keeps the same
confirm / expensive / proposable contract as ``run_bootstrap``.

The logistic result also carries the replicate coefficient draws (``draws``):
dose selection (``optimus``) reuses them to propagate E-R uncertainty into the
probability each dose is optimal.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.special import expit

from app.compute.bootstrap import _percentile_ci, _resample_indices
from app.compute.er_common import (
    DEFAULT_SEED,
    MAX_N_BOOT,
    ErRefusal,
    sd_of,
    sig,
    sig_list,
    to_float_array,
)
from app.compute.er_models import check_separation, fit_logistic_er, logistic_mle, probability_curve
from app.compute.er_survival import cox_mle, cox_ph

_MIN_OK_FOR_CI = 20
_MIN_SUCCESS_RATE = 0.80
_MAX_SECONDS = 600.0
_DRAW_DP = 12


def _validate(n: int, strata: list[Any] | None, ci_level: float) -> None:
    if strata is not None and len(strata) != n:
        raise ValueError(f"strata has {len(strata)} labels for {n} subjects; they must align")
    if not 0.0 < ci_level < 1.0:
        raise ValueError(f"ci_level must be in (0, 1); got {ci_level}")


def _replicate_loop(n: int, fit_one: Callable[[np.ndarray], np.ndarray], *, n_boot: int, seed: int,
                    strata: list[Any] | None, max_seconds: float
                    ) -> tuple[list[np.ndarray], int, dict[str, int], bool]:
    """Run the resampling loop. ``fit_one(idx)`` returns a coefficient vector or
    raises ``ErRefusal`` / ``FloatingPointError`` for an unusable replicate."""
    rng = np.random.default_rng(seed)
    rows: list[np.ndarray] = []
    reasons: dict[str, int] = {}
    n_failed, stopped = 0, False
    t0 = time.time()
    for _ in range(n_boot):
        if time.time() - t0 > max_seconds:
            stopped = True
            break
        idx = np.array(_resample_indices(n, rng, strata))
        try:
            rows.append(fit_one(idx))
        except ErRefusal as exc:
            n_failed += 1
            reasons[exc.status] = reasons.get(exc.status, 0) + 1
        except (FloatingPointError, np.linalg.LinAlgError):
            n_failed += 1
            reasons["numerical_failure"] = reasons.get("numerical_failure", 0) + 1
    return rows, n_failed, reasons, stopped


def _accounting(rows: list[np.ndarray], n_failed: int, reasons: dict[str, int], stopped: bool,
                n_boot: int, strata: list[Any] | None, max_seconds: float) -> dict[str, Any]:
    done = len(rows) + n_failed
    rate = len(rows) / done if done else 0.0
    notes: list[str] = []
    if n_failed:
        notes.append(f"{n_failed} of {done} replicates failed to produce a usable fit "
                     f"({', '.join(f'{k}: {v}' for k, v in sorted(reasons.items()))}); they are counted, "
                     "not dropped silently.")
    if rate < _MIN_SUCCESS_RATE:
        notes.append(f"only {100 * rate:.0f}% of replicates converged; fits fail preferentially on awkward "
                     "resamples, so these intervals are likely optimistic (too narrow).")
    if stopped:
        notes.append(f"stopped at the {max_seconds:.0f}s budget after {done} replicates.")
    if strata is None:
        notes.append("unstratified resampling: with small dose groups a replicate can under-represent one; "
                     "pass `strata` to resample within groups.")
    if len(rows) < 500:
        notes.append(f"{len(rows)} successful replicates; 500-1000 is the usual recommendation for stable "
                     "95% percentile limits.")
    return {"n_boot_requested": n_boot, "n_completed": done, "n_ok": len(rows), "n_failed": n_failed,
            "success_rate": sig(rate), "failure_reasons": reasons, "stratified": bool(strata),
            "stopped_early": stopped, "notes": notes}


def _interval(vals: np.ndarray, ci_level: float, estimate: float | None) -> dict[str, float | None]:
    med, lo, hi = _percentile_ci(vals, ci_level)
    return {"estimate": sig(estimate), "boot_median": sig(med), "lo": sig(lo), "hi": sig(hi),
            "se": sig(float(np.std(vals, ddof=1))) if vals.size > 1 else None}


def _too_few(acct: dict[str, Any], seed: int, ci_level: float) -> dict[str, Any]:
    return {"status": "too_few_successful_fits", "seed": seed, "ci_level": ci_level,
            "min_required": _MIN_OK_FOR_CI, **acct,
            "message": (f"only {acct['n_ok']} of {acct['n_completed']} replicates produced a usable fit; "
                        f"a percentile CI needs at least {_MIN_OK_FOR_CI}.")}


def bootstrap_logistic(exposure: Any, response: Any, *, n_boot: int = 500, seed: int = DEFAULT_SEED,
                       ci_level: float = 0.95, strata: list[Any] | None = None,
                       grid: list[float] | None = None, max_seconds: float = _MAX_SECONDS) -> dict[str, Any]:
    """Subject-level bootstrap of the logistic E-R fit.

    Reports percentile intervals for the slope, the odds ratio per unit and per
    SD (of the ORIGINAL sample, so the replicates are comparable), a percentile
    band for P(response) on ``grid`` (default: 25 points over the observed
    exposure range), and the replicate coefficient draws.
    """
    base = fit_logistic_er(exposure, response, ci_level=ci_level)
    if base["status"] != "ok":
        return base                                           # a refused fit is never bootstrapped
    x = to_float_array(exposure, "exposure")
    y = to_float_array(response, "response")
    keep = np.isfinite(x) & np.isfinite(y)
    _validate(keep.size, strata, ci_level)
    x, y = x[keep], y[keep]
    strata_kept = [s for s, k in zip(strata, keep) if k] if strata is not None else None
    n_boot = max(1, min(int(n_boot), MAX_N_BOOT))
    budget = min(float(max_seconds), _MAX_SECONDS)

    def fit_one(idx: np.ndarray) -> np.ndarray:
        xr, yr = x[idx], y[idx]
        if yr.min() == yr.max():
            raise ErRefusal("single_class", "resample contains one response class only")
        check_separation(xr, yr)
        return logistic_mle(xr, yr).beta

    rows, n_failed, reasons, stopped = _replicate_loop(
        x.size, fit_one, n_boot=n_boot, seed=seed, strata=strata_kept, max_seconds=budget)
    acct = _accounting(rows, n_failed, reasons, stopped, n_boot, strata, budget)
    if len(rows) < _MIN_OK_FOR_CI:
        return _too_few(acct, seed, ci_level)

    draws = np.array(rows)
    sd = sd_of(x)
    b1 = draws[:, 1]
    slope_hat = base["coef"]["slope"]["estimate"]
    grid_pts = [float(g) for g in grid] if grid else [float(v) for v in np.linspace(
        base["exposure_range"][0], base["exposure_range"][1], 25)]
    p_draws = expit(draws[:, [0]] + draws[:, [1]] * np.array(grid_pts)[None, :])
    q = [(1.0 - ci_level) / 2.0, 0.5, 1.0 - (1.0 - ci_level) / 2.0]
    lo_p, med_p, hi_p = (np.quantile(p_draws, v, axis=0) for v in q)
    s_int = _interval(b1, ci_level, slope_hat)
    return {
        "status": "ok", "method": "non-parametric bootstrap (subject-level resampling), logistic E-R",
        "seed": seed, "ci_level": ci_level, "n_subjects": int(x.size), **acct,
        "slope": s_int,
        "or_per_unit": {"estimate": base["or_per_unit"]["estimate"], "boot_median": sig(float(np.exp(np.median(b1)))),
                        "lo": sig(float(np.exp(s_int["lo"]))), "hi": sig(float(np.exp(s_int["hi"])))},
        "or_per_sd": {"sd": sig(sd), "estimate": base["or_per_sd"]["estimate"],
                      "boot_median": sig(float(np.exp(np.median(b1) * sd))),
                      "lo": sig(float(np.exp(s_int["lo"] * sd))), "hi": sig(float(np.exp(s_int["hi"] * sd)))},
        "curve": {"exposure": sig_list(grid_pts), "prob_median": sig_list(med_p), "lo": sig_list(lo_p),
                  "hi": sig_list(hi_p),
                  "point": probability_curve(base, grid=grid_pts)["prob"]},
        "draws": [[round(float(a), _DRAW_DP), round(float(b), _DRAW_DP)] for a, b in draws],
    }


def bootstrap_cox(time_: Any, event: Any, covariates: dict[str, Any], *, ties: str = "efron",
                  n_boot: int = 500, seed: int = DEFAULT_SEED, ci_level: float = 0.95,
                  strata: list[Any] | None = None, max_seconds: float = _MAX_SECONDS) -> dict[str, Any]:
    """Subject-level bootstrap of the Cox fit; the first covariate is the exposure."""
    base = cox_ph(time_, event, covariates, ties=ties, ci_level=ci_level)
    if base["status"] != "ok":
        return base
    names = list(covariates)
    t = to_float_array(time_, "time")
    e = to_float_array(event, "event")
    cols = np.column_stack([to_float_array(covariates[nm], nm) for nm in names])
    keep = np.isfinite(t) & np.isfinite(e) & np.all(np.isfinite(cols), axis=1)
    _validate(keep.size, strata, ci_level)
    t, e, cols = t[keep], e[keep], cols[keep]
    strata_kept = [s for s, k in zip(strata, keep) if k] if strata is not None else None
    n_boot = max(1, min(int(n_boot), MAX_N_BOOT))
    budget = min(float(max_seconds), _MAX_SECONDS)

    def fit_one(idx: np.ndarray) -> np.ndarray:
        return cox_mle(t[idx], e[idx], cols[idx], ties).beta

    rows, n_failed, reasons, stopped = _replicate_loop(
        t.size, fit_one, n_boot=n_boot, seed=seed, strata=strata_kept, max_seconds=budget)
    acct = _accounting(rows, n_failed, reasons, stopped, n_boot, strata, budget)
    if len(rows) < _MIN_OK_FOR_CI:
        return _too_few(acct, seed, ci_level)

    beta = np.array(rows)[:, 0]
    sd = sd_of(cols[:, 0])
    est = base["covariates"][0]["coef"]
    c_int = _interval(beta, ci_level, est)
    return {
        "status": "ok", "method": f"non-parametric bootstrap (subject-level resampling), Cox ({ties} ties)",
        "seed": seed, "ci_level": ci_level, "n_subjects": int(t.size), "exposure": names[0], **acct,
        "coef": c_int, "sd_exposure": sig(sd),
        "hr_per_unit": {"estimate": sig(float(np.exp(est))), "boot_median": sig(float(np.exp(np.median(beta)))),
                        "lo": sig(float(np.exp(c_int["lo"]))), "hi": sig(float(np.exp(c_int["hi"])))},
        "hr_per_sd": {"estimate": sig(float(np.exp(est * sd))),
                      "boot_median": sig(float(np.exp(np.median(beta) * sd))),
                      "lo": sig(float(np.exp(c_int["lo"] * sd))), "hi": sig(float(np.exp(c_int["hi"] * sd)))},
    }
