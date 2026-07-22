"""Sampling Importance Resampling (SIR) for parameter uncertainty.

Dosne, Bergstrand, Harling & Karlsson, "Improving the estimation of parameter
uncertainty distributions in nonlinear mixed effects models using sampling
importance resampling", J Pharmacokinet Pharmacodyn 2016;43(6):583-596.
Named by the FDA's 2022 PopPK guidance as one of the accepted ways to estimate
parameter precision.

SIR sits between the two methods PharmAgent already has, and its appeal is
that it needs NO re-estimation at all:

    asymptotic SE   cheap; assumes uncertainty is multivariate normal
    SIR             ~M objective evaluations, NO refits; makes no distributional
                    assumption about the RESULT, and carries a diagnostic that
                    says whether it worked
    bootstrap       hundreds of full refits; no distributional assumption

Three steps (Dosne Eq. 1):

1. **Sample.** Draw M parameter vectors from a proposal distribution, normally
   the covariance matrix of the estimates, optionally inflated.
2. **Weight.** Give each vector an importance ratio

       IR = exp(-dOFV / 2) / relPDF

   where ``dOFV`` is its objective minus the objective at the final estimates,
   and ``relPDF`` is its proposal density relative to the density at those
   estimates. For a multivariate-normal proposal the relative density is
   ``exp(-d^2 / 2)`` with ``d^2`` the squared Mahalanobis distance, so

       log IR = -(dOFV - d^2) / 2

   which is what this module computes. A vector the DATA like better than the
   PROPOSAL expected is up-weighted; one the proposal over-produced is
   down-weighted. That is the whole mechanism: it corrects a wrong proposal.
3. **Resample.** Draw m vectors WITHOUT replacement with probability
   proportional to IR. Without replacement is what makes the M/m ratio matter
   (Dosne recommends >= 5): with replacement a single dominant vector could
   fill the resample and the ratio would be largely irrelevant.

Diagnostics, because an importance sampler can fail silently:

* **dOFV vs chi-square.** If the resampled vectors were the true uncertainty,
  their dOFV would follow a chi-square whose df is at most the number of
  estimated parameters -- lower in practice, since random effects and bounded
  parameters do not carry a full degree of freedom. df is estimated here by
  matching the mean (``E[chi2_df] = df``). Far ABOVE the reference means the
  proposal is too narrow; far below, too wide.
* **Effective sample size**, ``ESS = 1 / sum(w^2)`` on normalised weights. The
  standard importance-sampling degeneracy measure: if a handful of vectors
  carry the weight, ESS collapses and the interval is being read off those few
  draws no matter how large M was. Dosne's temporal-trends plot targets the
  same failure; ESS states it as one number.

This module runs no estimator and no fit -- ``ofv_fn`` is injected -- so it is
testable against objectives whose answer is known in closed form.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any

import numpy as np

_ROUND_DP = 6
_DEFAULT_M_OVER_m = 5             # Dosne's recommended M/m ratio
_DEFAULT_N_RESAMPLE = 1000        # m: enough for a 95% CI
_MAX_SAMPLES = 20000              # hard cap on M (each costs one OFV evaluation)
_MIN_ESS = 50.0                   # below this the resample is degenerate
_MAX_SECONDS = 60.0 * 60.0


def _round(x: float | None, dp: int = _ROUND_DP) -> float | None:
    if x is None or not isinstance(x, (int, float, np.floating)):
        return None
    xf = float(x)
    return round(xf, dp) if math.isfinite(xf) else None


def _psd_chol(cov: np.ndarray) -> np.ndarray:
    """Cholesky of a covariance, repaired if sampling noise left it indefinite.

    The proposal only has to be a usable sampling distribution, so flooring the
    eigenvalues is legitimate here in a way it would not be for an estimate.
    """
    sym = 0.5 * (np.asarray(cov, dtype=float) + np.asarray(cov, dtype=float).T)
    try:
        return np.linalg.cholesky(sym)
    except np.linalg.LinAlgError:
        w, v = np.linalg.eigh(sym)
        floor = max(float(np.max(w)) * 1e-10, 1e-300)
        return np.linalg.cholesky(v @ np.diag(np.maximum(w, floor)) @ v.T)


def run_sir(*, ofv_fn: Callable[[np.ndarray], float],
            x_hat: np.ndarray, cov: np.ndarray, ofv_hat: float,
            decode_fn: Callable[[np.ndarray], dict[str, float]],
            n_resample: int = _DEFAULT_N_RESAMPLE,
            n_samples: int | None = None,
            inflation: float = 1.0,
            n_estimated_params: int | None = None,
            ci_level: float = 0.95,
            seed: int = 20250614,
            max_seconds: float = _MAX_SECONDS,
            proposal_note: str | None = None,
            progress: Callable[[dict], None] | None = None) -> dict[str, Any]:
    """Sampling importance resampling on a fitted model's parameter uncertainty.

    Parameters
    ----------
    ofv_fn:
        ``packed parameter vector -> objective value``. Called M times; this is
        the entire cost. No fit is ever run.
    x_hat, cov, ofv_hat:
        Final estimates, their covariance (the proposal), and the objective
        there -- all on the same packed/estimation scale ``ofv_fn`` expects.
    decode_fn:
        ``packed vector -> {name: natural-scale value}``, used only to report.
        Keeping the sampling on the estimation scale and decoding afterwards is
        what lets SIR produce ASYMMETRIC intervals on the natural scale.
    n_resample:
        m. 1000 is Dosne's recommendation for a 95% CI.
    n_samples:
        M. Defaults to ``_DEFAULT_M_OVER_m * m``; capped at ``_MAX_SAMPLES``.
    inflation:
        SD-scale multiplier on the proposal (``cov * inflation^2``). Dosne finds
        a somewhat inflated proposal gives coverage as good as or better than
        the covariance matrix, whereas a deflated one is actively harmful --
        importance weighting can correct a proposal that is too WIDE, but
        cannot invent samples in a region the proposal never visited.
    n_estimated_params:
        Reference df for the chi-square diagnostic. Defaults to ``len(x_hat)``.
    proposal_note:
        Surfaced first in ``notes``. Used to carry forward a caveat about how
        the proposal was built -- e.g. that the information matrix was
        near-singular and had to be regularized, which makes the proposal look
        better conditioned than the fit that produced it.

    Returns
    -------
    JSON-safe dict: ``parameters`` (SIR median + CI per parameter, beside the
    asymptotic interval implied by the proposal), ``diagnostics`` (ESS, the
    estimated df against its reference, M/m), and ``notes``.
    """
    x_hat = np.asarray(x_hat, dtype=float).ravel()
    cov = np.asarray(cov, dtype=float)
    n_par = x_hat.size
    if n_par == 0:
        return {"status": "no_parameters", "message": "empty parameter vector."}
    if cov.shape != (n_par, n_par):
        raise ValueError(f"cov must be {n_par}x{n_par} to match x_hat; got {cov.shape}")
    if not math.isfinite(float(ofv_hat)):
        return {"status": "no_fit", "message": "objective at the estimates is not finite."}
    if not 0.0 < ci_level < 1.0:
        raise ValueError(f"ci_level must be in (0, 1); got {ci_level}")
    if inflation <= 0.0:
        raise ValueError(f"inflation must be > 0; got {inflation}")

    m = max(1, int(n_resample))
    M = int(n_samples) if n_samples else _DEFAULT_M_OVER_m * m
    M = max(m, min(M, _MAX_SAMPLES))

    rng = np.random.default_rng(seed)
    prop_cov = cov * float(inflation) ** 2
    chol = _psd_chol(prop_cov)
    try:
        prec = np.linalg.inv(prop_cov)
    except np.linalg.LinAlgError:
        prec = np.linalg.pinv(prop_cov)

    t0 = time.time()
    draws = np.empty((M, n_par), dtype=float)
    dofv = np.full(M, np.nan, dtype=float)
    maha = np.empty(M, dtype=float)
    n_bad = 0
    stopped_early = False
    for i in range(M):
        if time.time() - t0 > min(float(max_seconds), _MAX_SECONDS):
            stopped_early = True
            draws, dofv, maha = draws[:i], dofv[:i], maha[:i]
            M = i
            break
        z = rng.normal(0.0, 1.0, n_par)
        xi = x_hat + chol @ z
        draws[i] = xi
        d = xi - x_hat
        maha[i] = float(d @ prec @ d)
        try:
            val = float(ofv_fn(xi))
        except Exception:
            val = math.nan
        dofv[i] = val - float(ofv_hat) if math.isfinite(val) else math.nan
        if not math.isfinite(dofv[i]):
            n_bad += 1
        if progress is not None and (i + 1) % 250 == 0:
            progress({"completed": i + 1, "requested": M, "bad": n_bad})

    ok = np.isfinite(dofv)
    n_ok = int(ok.sum())
    if n_ok < m:
        return {"status": "too_few_usable_samples", "n_samples": int(M),
                "n_usable": n_ok, "n_resample_requested": m,
                "n_failed_objective": n_bad, "stopped_early": stopped_early,
                "message": (f"only {n_ok} of {M} sampled vectors gave a finite "
                            f"objective; cannot resample {m} without replacement.")}

    draws, dofv, maha = draws[ok], dofv[ok], maha[ok]

    # log IR = -(dOFV - mahalanobis^2)/2  (Dosne Eq. 1 for an MVN proposal).
    # Shift by the max before exponentiating: dOFV routinely spans hundreds of
    # units, so exp() of the raw values underflows to all-zero weights.
    log_ir = -0.5 * (dofv - maha)
    w = np.exp(log_ir - float(np.max(log_ir)))
    total = float(w.sum())
    if not math.isfinite(total) or total <= 0.0:
        return {"status": "degenerate_weights", "n_samples": int(M),
                "message": "importance weights underflowed to zero; the proposal "
                           "is far from the true uncertainty."}
    w = w / total
    ess = float(1.0 / np.sum(w ** 2))

    n_positive = int(np.count_nonzero(w > 0.0))
    if n_positive < m:
        return {"status": "degenerate_weights", "n_samples": int(M),
                "n_positive_weights": n_positive, "n_resample_requested": m,
                "ess": _round(ess),
                "message": (f"only {n_positive} vectors carry non-zero weight; "
                            f"cannot resample {m} without replacement. The "
                            "proposal is far from the true uncertainty.")}

    # WITHOUT replacement -- this is what makes the M/m ratio meaningful.
    pick = rng.choice(len(w), size=m, replace=False, p=w)
    res_draws, res_dofv = draws[pick], dofv[pick]

    a = (1.0 - ci_level) / 2.0
    decoded = [decode_fn(x) for x in res_draws]
    names = list(decoded[0].keys()) if decoded else []
    point = decode_fn(x_hat)
    # Asymptotic interval implied by the proposal itself, decoded the same way,
    # so the two are compared on the natural scale rather than across scales.
    z95 = 1.959963984540054
    sd = np.sqrt(np.maximum(np.diag(cov), 0.0))
    lo_dec = decode_fn(x_hat - z95 * sd)
    hi_dec = decode_fn(x_hat + z95 * sd)

    params: list[dict[str, Any]] = []
    for nm in names:
        vals = np.array([d[nm] for d in decoded], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        med = float(np.median(vals))
        lo, hi = float(np.quantile(vals, a)), float(np.quantile(vals, 1.0 - a))
        est = float(point.get(nm, math.nan))
        alo, ahi = float(lo_dec.get(nm, math.nan)), float(hi_dec.get(nm, math.nan))
        # Asymmetry is the reason to run SIR at all: a symmetric normal
        # approximation cannot express it.
        asym = None
        if math.isfinite(est) and hi > lo:
            upper, lower = hi - est, est - lo
            if lower > 0:
                asym = _round(upper / lower)
        params.append({
            "parameter": nm, "estimate": _round(est),
            "sir_median": _round(med), "sir_lo": _round(lo), "sir_hi": _round(hi),
            "asymptotic_lo": _round(alo) if math.isfinite(alo) else None,
            "asymptotic_hi": _round(ahi) if math.isfinite(ahi) else None,
            "asymmetry_ratio": asym,
        })

    df_ref = int(n_estimated_params) if n_estimated_params else n_par
    # E[chi2_df] = df, so the mean of the resampled dOFV estimates df.
    df_hat = float(np.mean(res_dofv))
    notes: list[str] = []
    if ess < _MIN_ESS:
        notes.append(
            f"effective sample size {ess:.0f} is very low: the resample is "
            "dominated by a few vectors, so these intervals rest on far less "
            "information than M suggests. Increase inflation or M.")
    # Direction matters and is easy to invert: samples drawn from a TOO-NARROW
    # proposal cluster near the optimum, so their dOFV is LOW -- Dosne describes
    # a too-narrow proposal as sitting "below the Chi square distribution".
    # High dOFV is the opposite: the proposal is scattering vectors into regions
    # the data dislike.
    if df_hat < df_ref * 0.5:
        notes.append(
            f"resampled dOFV averages {df_hat:.1f}, well BELOW the reference df "
            f"of {df_ref}: the proposal is too NARROW, so the true uncertainty "
            "is probably wider than this reports. Re-run with inflation > 1 — "
            "importance weighting can shrink a proposal that is too wide, but "
            "cannot recover a region the proposal never sampled.")
    elif df_hat > df_ref * 1.5:
        notes.append(
            f"resampled dOFV averages {df_hat:.1f}, ABOVE the reference df of "
            f"{df_ref}: the proposal is wider than the true uncertainty. SIR "
            "corrects for this, but at the cost of effective sample size.")
    if M < _DEFAULT_M_OVER_m * m:
        notes.append(
            f"M/m = {M / m:.1f}; Dosne recommends at least "
            f"{_DEFAULT_M_OVER_m} so the resample is not dominated by a few "
            "high-weight vectors.")
    if stopped_early:
        notes.append(f"stopped at the time budget after {M} of the requested samples.")
    if proposal_note:
        notes.insert(0, proposal_note)
    notes.append(
        "df is estimated by matching the mean of the resampled dOFV; it is "
        f"expected to be at or BELOW the {df_ref} estimated parameters, since "
        "random effects and bounded parameters do not carry a full degree of "
        "freedom (Dosne et al. 2016).")

    return {
        "status": "ok",
        "method": "sampling importance resampling (Dosne et al. 2016)",
        "n_samples": int(M),
        "n_usable": n_ok,
        "n_resample": m,
        "m_over_m_ratio": _round(M / m, 2),
        "inflation": _round(float(inflation)),
        "n_failed_objective": n_bad,
        "stopped_early": stopped_early,
        "ci_level": ci_level,
        "seconds": _round(time.time() - t0, 1),
        "parameters": params,
        "diagnostics": {
            "effective_sample_size": _round(ess),
            "ess_fraction_of_m": _round(ess / m),
            "dofv_mean_resampled": _round(df_hat),
            "df_reference": df_ref,
            "dofv_mean_proposal": _round(float(np.mean(dofv))),
            "max_weight": _round(float(np.max(w))),
        },
        "notes": notes,
    }
