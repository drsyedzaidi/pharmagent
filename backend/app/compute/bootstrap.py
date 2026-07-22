"""Non-parametric bootstrap for population-PK parameter uncertainty.

The FDA's 2022 Population Pharmacokinetics guidance names four ways to estimate
parameter precision -- "bootstrap procedures, sampling importance resampling,
log-likelihood profiling, or using the asymptotic standard errors" -- and is
explicit that "no single model validation method is generally sufficient".
PharmAgent has had only the last of those (``nlme._parameter_uncertainty``).
This module adds the first.

Why it matters beyond box-ticking: the asymptotic standard error assumes the
uncertainty is multivariate normal on the estimation scale. That assumption is
untestable from the standard errors themselves. A bootstrap makes no such
assumption -- it reads the sampling distribution off the data -- so comparing
the two intervals is a direct check on whether the cheap method was adequate.
That comparison is the headline output here (``comparison``), not an extra.

Procedure (Efron's non-parametric bootstrap, at the SUBJECT level):

1. Draw ``n`` subjects with replacement from the ``n`` analysis subjects.
   Sampling whole subjects, not observations, preserves the within-subject
   correlation structure that the random effects describe -- resampling
   observations would destroy exactly what a population model estimates.
2. Re-fit the model on each replicate dataset.
3. The 95% CI of each parameter is the 2.5th / 97.5th percentile of its
   bootstrap distribution.

This module owns no estimator. ``fit_fn`` is injected, so the expensive loop
belongs to the caller (a background job) and every test here runs against a
fake fit.

Two things the literature warns about, both handled explicitly rather than
assumed away:

* **Representativeness.** With small strata, an unstratified resample can drop
  a study arm or dose level entirely, so the replicate is not a sample of the
  same design. ``strata`` resamples WITHIN each stratum, preserving its size.
* **Enough replicates.** Percentile CIs converge slowly in the tails. The
  returned ``stability`` trace re-computes each CI using only the first k
  replicates so the caller can see whether the interval has settled, rather
  than trusting a fixed count.

Failed replicates are counted, never quietly dropped: a fit fails preferentially
on awkward resamples, so excluding them narrows the interval exactly where the
data are weakest. Percentiles are reported over successful fits, but the
success rate is reported beside them and a low rate is itself the finding.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any

import numpy as np

_ROUND_DP = 6
_DEFAULT_N_BOOT = 200
_MAX_N_BOOT = 1000
_MAX_SECONDS = 60.0 * 60.0        # hard ceiling regardless of the caller's value
_MIN_OK_FOR_CI = 20               # below this a percentile CI is not meaningful
_MIN_SUCCESS_RATE = 0.80          # below this the run is flagged, not silently used
_Z95 = 1.959963984540054


def _round(x: float | None, dp: int = _ROUND_DP) -> float | None:
    if x is None or not isinstance(x, (int, float)) or not math.isfinite(float(x)):
        return None
    return round(float(x), dp)


def _resample_indices(n: int, rng: np.random.Generator,
                      strata: list[Any] | None) -> list[int]:
    """Subject indices for one replicate, drawn with replacement.

    Without ``strata`` this is a plain draw of ``n`` from ``n``. With strata,
    each stratum is resampled to its own original size, so every study arm /
    dose level keeps its representation -- an unstratified draw can omit a
    small arm entirely, which silently changes the design being bootstrapped.
    """
    if not strata:
        return [int(i) for i in rng.integers(0, n, size=n)]
    groups: dict[Any, list[int]] = {}
    for i, s in enumerate(strata):
        groups.setdefault(s, []).append(i)
    out: list[int] = []
    for key in sorted(groups, key=str):          # sorted -> reproducible order
        idx = groups[key]
        out.extend(int(idx[j]) for j in rng.integers(0, len(idx), size=len(idx)))
    return out


def _collect(fit: dict[str, Any], params: tuple[str, ...]) -> dict[str, float] | None:
    """Pull the parameters of interest out of one replicate fit, or None if the
    replicate is unusable."""
    if not isinstance(fit, dict) or fit.get("status") not in (None, "ok"):
        return None
    if not fit.get("converged"):
        return None
    theta = fit.get("theta") or {}
    omega = fit.get("omega_cv_pct") or {}
    sigma = fit.get("sigma") or {}
    row: dict[str, float] = {}
    for p in params:
        if p in theta:
            v = theta[p]
        elif p.startswith("omega_") and p[6:] in omega:
            v = omega[p[6:]]
        elif p.startswith("sigma_") and p[6:] in sigma:
            v = sigma[p[6:]]
        else:
            return None
        if v is None or not math.isfinite(float(v)):
            return None
        row[p] = float(v)
    return row


def _percentile_ci(vals: np.ndarray, level: float) -> tuple[float, float, float]:
    """(median, lower, upper) percentile interval."""
    a = (1.0 - level) / 2.0
    return (float(np.median(vals)),
            float(np.quantile(vals, a)),
            float(np.quantile(vals, 1.0 - a)))


def _default_params(nlme_result: dict[str, Any]) -> tuple[str, ...]:
    """Structural thetas plus each IIV %CV -- what a report tabulates."""
    theta = list((nlme_result.get("theta") or {}).keys())
    iiv = [f"omega_{p}" for p in (nlme_result.get("iiv_params") or [])]
    return tuple(theta + iiv)


def _asymptotic_ci(nlme_result: dict[str, Any], param: str
                   ) -> tuple[float | None, float | None, float | None]:
    """(estimate, lo, hi) from the asymptotic SE, for comparison.

    RSE% in this codebase is 100*SE on the LOG scale for log-linked parameters
    (see nlme._parameter_uncertainty), so the interval is built multiplicatively
    -- ``est * exp(+/- 1.96 * rse/100)`` -- not as ``est +/- 1.96*SE``. Using
    the additive form here would produce a different, wrong interval and make
    the bootstrap comparison meaningless.
    """
    theta = nlme_result.get("theta") or {}
    rse_t = nlme_result.get("theta_rse_pct") or {}
    omega = nlme_result.get("omega_cv_pct") or {}
    rse_o = nlme_result.get("omega_rse_pct") or {}
    if param in theta:
        est, rse = theta.get(param), rse_t.get(param)
    elif param.startswith("omega_"):
        est, rse = omega.get(param[6:]), rse_o.get(param[6:])
    else:
        return None, None, None
    if est is None or rse is None or not math.isfinite(float(rse)):
        return (_round(est) if est is not None else None), None, None
    s = float(rse) / 100.0
    return (float(est), float(est) * math.exp(-_Z95 * s),
            float(est) * math.exp(_Z95 * s))


def run_bootstrap(model_key: str, subjects: list[dict], nlme_result: dict[str, Any],
                  *, fit_fn: Callable[[list[dict], int], dict],
                  n_boot: int = _DEFAULT_N_BOOT,
                  params: tuple[str, ...] | None = None,
                  strata: list[Any] | None = None,
                  ci_level: float = 0.95,
                  seed: int = 20250614,
                  max_seconds: float = _MAX_SECONDS,
                  progress: Callable[[dict], None] | None = None
                  ) -> dict[str, Any]:
    """Non-parametric bootstrap of a fitted population model.

    Parameters
    ----------
    model_key:
        Structural model key; echoed into the result for provenance only.
    subjects:
        The ANALYSIS subjects, in the form ``fit_fn`` expects. Resampled whole.
    nlme_result:
        The converged fit being bootstrapped. Supplies the point estimates and
        the asymptotic intervals the bootstrap is compared against.
    fit_fn:
        ``(replicate_subjects, seed) -> fit result`` with the same shape as
        ``nlme.population_fit``. This module never fits anything itself.
    n_boot:
        Replicates, capped at ``_MAX_N_BOOT``. 200 is a reasonable default for
        a 95% CI; the literature's 500-1000 is better and much slower.
    params:
        Names to summarize: a theta key, ``omega_<P>`` for an IIV %CV, or
        ``sigma_prop`` / ``sigma_add``. Defaults to all thetas + all IIV terms.
    strata:
        Optional per-subject stratum label (study, dose group, ...), same length
        and order as ``subjects``. Resampling happens within strata.
    ci_level:
        Interval mass, default 0.95 -> 2.5th/97.5th percentiles.
    max_seconds:
        Wall-clock budget. The loop stops early and reports what it has, rather
        than being killed mid-run.

    Returns
    -------
    A JSON-safe dict with ``status``, ``parameters`` (per-parameter bootstrap
    median + CI + the asymptotic CI beside it), ``comparison`` (bootstrap vs
    asymptotic interval width), ``stability`` (CI recomputed at increasing
    replicate counts), and the failure accounting.
    """
    n = len(subjects)
    if n < 2:
        return {"status": "insufficient_subjects", "n_subjects": n,
                "message": "bootstrap needs at least 2 subjects to resample."}
    if (nlme_result or {}).get("status") not in (None, "ok") or not nlme_result.get("theta"):
        return {"status": "no_fit",
                "message": "bootstrap needs a converged NLME fit to resample around."}
    if strata is not None and len(strata) != n:
        raise ValueError(
            f"strata has {len(strata)} labels for {n} subjects; they must align")
    if not 0.0 < ci_level < 1.0:
        raise ValueError(f"ci_level must be in (0, 1); got {ci_level}")

    n_boot = max(1, min(int(n_boot), _MAX_N_BOOT))
    budget = min(float(max_seconds), _MAX_SECONDS)
    names = tuple(params) if params else _default_params(nlme_result)
    if not names:
        return {"status": "no_parameters",
                "message": "no parameters to bootstrap in the supplied fit."}

    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []
    n_failed = 0
    reasons: dict[str, int] = {}
    t0 = time.time()
    stopped_early = False

    for k in range(n_boot):
        if time.time() - t0 > budget:
            stopped_early = True
            break
        idx = _resample_indices(n, rng, strata)
        rep = [subjects[i] for i in idx]
        try:
            fit = fit_fn(rep, seed + k + 1)
        except Exception as exc:                       # a replicate may fail
            n_failed += 1
            reasons[type(exc).__name__] = reasons.get(type(exc).__name__, 0) + 1
            continue
        row = _collect(fit, names)
        if row is None:
            n_failed += 1
            key = "not_converged" if isinstance(fit, dict) else "bad_result"
            reasons[key] = reasons.get(key, 0) + 1
            continue
        rows.append(row)
        if progress is not None:
            progress({"completed": k + 1, "requested": n_boot,
                      "ok": len(rows), "failed": n_failed})

    n_done = len(rows) + n_failed
    n_ok = len(rows)
    if n_ok < _MIN_OK_FOR_CI:
        return {"status": "too_few_successful_fits", "model_key": model_key,
                "n_boot_requested": n_boot, "n_completed": n_done,
                "n_ok": n_ok, "n_failed": n_failed, "failure_reasons": reasons,
                "min_required": _MIN_OK_FOR_CI, "stopped_early": stopped_early,
                "message": (f"only {n_ok} of {n_done} replicates produced a usable "
                            f"fit; a percentile CI needs at least {_MIN_OK_FOR_CI}.")}

    out_params: list[dict[str, Any]] = []
    comparison: list[dict[str, Any]] = []
    for p in names:
        vals = np.array([r[p] for r in rows], dtype=float)
        med, lo, hi = _percentile_ci(vals, ci_level)
        est, alo, ahi = _asymptotic_ci(nlme_result, p)
        rec = {"parameter": p, "estimate": _round(est),
               "boot_median": _round(med), "boot_lo": _round(lo), "boot_hi": _round(hi),
               "boot_se": _round(float(np.std(vals, ddof=1))) if vals.size > 1 else None,
               "asymptotic_lo": _round(alo), "asymptotic_hi": _round(ahi),
               "boot_bias_pct": (_round(100.0 * (med - est) / est)
                                 if est not in (None, 0) else None)}
        out_params.append(rec)
        if alo is not None and ahi is not None and hi > lo:
            wb, wa = hi - lo, ahi - alo
            comparison.append({
                "parameter": p,
                "boot_width": _round(wb), "asymptotic_width": _round(wa),
                # >1 => the asymptotic interval is OPTIMISTIC (too narrow)
                "width_ratio_boot_over_asymptotic": _round(wb / wa) if wa > 0 else None,
            })

    # CI stability: recompute using only the first k replicates. A CI that is
    # still moving at k = n_ok has not converged, whatever the nominal count.
    marks = sorted({m for m in (25, 50, 100, 200, 500, n_ok) if _MIN_OK_FOR_CI <= m <= n_ok})
    stability = [
        {"n_replicates": m,
         "parameters": [
             {"parameter": p,
              "lo": _round(_percentile_ci(
                  np.array([r[p] for r in rows[:m]], dtype=float), ci_level)[1]),
              "hi": _round(_percentile_ci(
                  np.array([r[p] for r in rows[:m]], dtype=float), ci_level)[2])}
             for p in names]}
        for m in marks
    ]

    rate = n_ok / n_done if n_done else 0.0
    notes: list[str] = []
    if rate < _MIN_SUCCESS_RATE:
        notes.append(
            f"only {100 * rate:.0f}% of replicates converged; fits fail "
            "preferentially on awkward resamples, so these intervals are "
            "likely optimistic (too narrow).")
    if stopped_early:
        notes.append(f"stopped at the {budget:.0f}s budget after {n_done} replicates.")
    if strata is None:
        notes.append(
            "unstratified resampling: with small study arms or dose groups a "
            "replicate can under-represent one, widening its intervals; pass "
            "`strata` to resample within groups.")
    if n_ok < 500:
        notes.append(
            f"{n_ok} successful replicates; 500-1000 is the usual "
            "recommendation for stable 95% percentile limits — check `stability`.")

    return {
        "status": "ok",
        "model_key": model_key,
        "method": "non-parametric bootstrap (subject-level resampling)",
        "n_subjects": n,
        "n_boot_requested": n_boot,
        "n_completed": n_done,
        "n_ok": n_ok,
        "n_failed": n_failed,
        "success_rate": _round(rate),
        "failure_reasons": reasons,
        "stratified": bool(strata),
        "n_strata": len(set(map(str, strata))) if strata else 0,
        "ci_level": ci_level,
        "stopped_early": stopped_early,
        "seconds": _round(time.time() - t0, 1),
        "parameters": out_params,
        "comparison": comparison,
        "stability": stability,
        "notes": notes,
    }
