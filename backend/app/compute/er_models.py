"""Binary exposure-response: logistic regression of response on one exposure metric.

    logit P(response = 1) = b0 + b1 * exposure

Used for efficacy responders or adverse-event incidence against a per-subject
exposure (AUC, Cavg, Cmax). The estimator is maximum likelihood by Newton's
method with step halving on the analytic score and information (exactly the
IRLS iteration of R's ``glm``), run on the standardised exposure for numerical
stability and mapped back to the original units. Wald standard errors come from
the inverse observed information; the probability band is the delta method on
the logit scale. Validated against R ``glm`` (tests/test_er_models.py).

Time-to-event models (Kaplan-Meier, Cox) live in ``er_survival`` and are
re-exported here so ``app.compute.er_models`` is the one import for the family.

Refusals (structured, never a number) -- see ``er_common``:

* ``separation``: exposure perfectly (or quasi-completely) separates responders
  from non-responders, so the likelihood has no finite maximum. Reporting the
  iterate where a solver happened to stop would be garbage; Firth/penalised
  fitting is a different estimator and is deliberately not substituted silently.
* ``insufficient_events``: fewer than ``MIN_EVENTS_HARD`` in either class.
* ``zero_variance_exposure``, ``invalid_input``, ``non_convergence``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import expit
from scipy.stats import chi2, rankdata

from app.compute.er_common import (
    EVENTS_PER_VARIABLE,
    MIN_EVENTS_HARD,
    ErRefusal,
    is_constant,
    refusal,
    sd_of,
    sig,
    sig_list,
    sig_matrix,
    to_float_array,
    wald_row,
    z_crit,
)
from app.compute.er_survival import cox_ph, kaplan_meier, logrank_test  # noqa: F401  (re-export)

_MAX_ITER = 100
_STEP_TOL = 1e-12
_NLL_TOL = 1e-12                # relative slack for rounding noise in the monotonicity check


@dataclass(frozen=True)
class LogisticMLE:
    """Original-scale MLE: ``beta = (intercept, slope)`` and its covariance."""
    beta: np.ndarray
    cov: np.ndarray
    nll: float
    n_iter: int
    converged: bool


def check_separation(x: np.ndarray, y: np.ndarray) -> None:
    """Exact one-predictor test for complete or quasi-complete separation: the
    MLE exists iff the two classes' exposure ranges overlap strictly."""
    x0, x1 = x[y == 0], x[y == 1]
    if x0.max() <= x1.min() or x1.max() <= x0.min():
        raise ErRefusal(
            "separation",
            "Exposure completely (or quasi-completely) separates responders from "
            "non-responders, so the maximum-likelihood estimate does not exist "
            "(the slope diverges). No coefficients are reported; this usually means "
            "too few subjects or an exposure range that is too narrow.")


def _nll(eta: np.ndarray, y: np.ndarray) -> float:
    return float(np.sum(np.logaddexp(0.0, eta) - y * eta))


def logistic_mle(x: np.ndarray, y: np.ndarray) -> LogisticMLE:
    """MLE on validated, finite, binary data. Raises ``ErRefusal``."""
    n_events = int(y.sum())
    if min(n_events, y.size - n_events) < MIN_EVENTS_HARD:
        raise ErRefusal("insufficient_events",
                        f"Only {min(n_events, y.size - n_events)} subjects in the smaller response class "
                        f"(minimum {MIN_EVENTS_HARD}); a logistic exposure-response model is not estimable "
                        "from so few.")
    if is_constant(x):
        raise ErRefusal("zero_variance_exposure", "Exposure is constant across subjects; its effect "
                                                  "cannot be estimated.")
    check_separation(x, y)

    mean, sd = float(x.mean()), sd_of(x)
    z = (x - mean) / sd
    design = np.column_stack([np.ones_like(z), z])
    ybar = y.mean()
    beta = np.array([np.log(ybar / (1.0 - ybar)), 0.0])
    nll = _nll(design @ beta, y)
    converged, n_iter = False, 0
    try:
        for n_iter in range(1, _MAX_ITER + 1):
            p = expit(design @ beta)
            grad = design.T @ (y - p)
            info = design.T @ (design * (p * (1.0 - p))[:, None])
            step = np.linalg.solve(info, grad)
            scale = 1.0
            for _ in range(40):                              # step halving: nll must not increase
                cand = beta + scale * step
                cand_nll = _nll(design @ cand, y)
                if cand_nll <= nll + _NLL_TOL * (1.0 + abs(nll)):
                    break
                scale *= 0.5
            else:
                break
            beta, nll = cand, cand_nll
            if np.max(np.abs(scale * step)) < _STEP_TOL:
                converged = True
                break
        p = expit(design @ beta)
        cov_z = np.linalg.inv(design.T @ (design * (p * (1.0 - p))[:, None]))
    except np.linalg.LinAlgError as exc:
        raise ErRefusal("non_convergence", "The information matrix is singular (fitted probabilities "
                                           "saturate); the likelihood is effectively flat or unbounded.") from exc
    if not converged:
        raise ErRefusal("non_convergence", f"Newton iterations did not converge in {_MAX_ITER} steps.")
    # map (b0, b1) on z = (x - mean) / sd back to the original exposure units
    t = np.array([[1.0, -mean / sd], [0.0, 1.0 / sd]])
    return LogisticMLE(beta=t @ beta, cov=t @ cov_z @ t.T, nll=nll, n_iter=n_iter, converged=True)


def _clean_pair(exposure: Any, response: Any) -> tuple[np.ndarray, np.ndarray, int]:
    x = to_float_array(exposure, "exposure")
    y = to_float_array(response, "response")
    if x.size != y.size:
        raise ErRefusal("invalid_input", f"exposure has {x.size} values but response has {y.size}")
    keep = np.isfinite(x) & np.isfinite(y)
    dropped = int((~keep).sum())
    x, y = x[keep], y[keep]
    if x.size == 0:
        raise ErRefusal("invalid_input", "no complete (exposure, response) pairs")
    if not np.all((y == 0.0) | (y == 1.0)):
        raise ErRefusal("invalid_input", "response must be binary (0/1 or boolean)")
    return x, y, dropped


def _roc_auc(score: np.ndarray, y: np.ndarray) -> float:
    ranks = rankdata(score)
    n1 = float(y.sum())
    n0 = float(y.size) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1.0) / 2.0) / (n1 * n0))


def fit_logistic_er(exposure: Any, response: Any, *, ci_level: float = 0.95,
                    exposure_label: str = "exposure") -> dict[str, Any]:
    """Fit the logistic E-R model. Always returns a JSON-safe dict with ``status``."""
    try:
        zc = z_crit(ci_level)
        x, y, dropped = _clean_pair(exposure, response)
        mle = logistic_mle(x, y)
    except ErRefusal as exc:
        return refusal(exc.status, exc.message)

    n, n_events = int(x.size), int(y.sum())
    b0, b1 = float(mle.beta[0]), float(mle.beta[1])
    se = np.sqrt(np.diag(mle.cov))
    sd = sd_of(x)
    ybar = n_events / n
    null_nll = -(n_events * np.log(ybar) + (n - n_events) * np.log(1.0 - ybar))
    lr = 2.0 * (null_nll - mle.nll)
    slope = wald_row(b1, float(se[1]), zc)
    warnings: list[str] = []
    min_class = min(n_events, n - n_events)
    if min_class < EVENTS_PER_VARIABLE:
        warnings.append(
            f"Only {min_class} events per variable (smaller response class, 1 predictor; guideline >= "
            f"{EVENTS_PER_VARIABLE}): the coefficient is imprecise and Wald intervals may be unreliable.")
    return {
        "status": "ok", "model": "logistic", "link": "logit",
        "exposure_label": exposure_label, "ci_level": ci_level,
        "n": n, "n_events": n_events, "n_nonevents": n - n_events, "event_rate": sig(ybar),
        "n_dropped_non_finite": dropped,
        "exposure_range": [sig(float(x.min())), sig(float(x.max()))],
        "exposure_summary": {"mean": sig(float(x.mean())), "sd": sig(sd), "median": sig(float(np.median(x)))},
        "coef": {"intercept": wald_row(b0, float(se[0]), zc), "slope": slope},
        "cov": sig_matrix(mle.cov),
        "or_per_unit": {"estimate": sig(np.exp(b1)), "lo": sig(np.exp(slope["lo"])),
                        "hi": sig(np.exp(slope["hi"]))},
        "or_per_sd": {"sd": sig(sd), "estimate": sig(np.exp(b1 * sd)),
                      "lo": sig(np.exp(slope["lo"] * sd)), "hi": sig(np.exp(slope["hi"] * sd))},
        "loglik": sig(-mle.nll), "null_loglik": sig(-null_nll),
        "deviance": sig(2.0 * mle.nll), "null_deviance": sig(2.0 * null_nll),
        "aic": sig(2.0 * mle.nll + 4.0),
        "lr_chisq": sig(lr), "lr_p": sig(chi2.sf(lr, 1)),
        "mcfadden_r2": sig(1.0 - mle.nll / null_nll),
        "auc_roc": sig(_roc_auc(b0 + b1 * x, y)),
        "converged": mle.converged, "n_iter": mle.n_iter,
        "warnings": warnings,
    }


def predict_logistic(fit: dict[str, Any], exposure: Any, *,
                     ci_level: float | None = None) -> dict[str, Any]:
    """Probability of response (and delta-method band) at the given exposures.

    Uses the coefficients and covariance stored on ``fit``, so a stored result
    can be evaluated later (e.g. by dose selection) without refitting.
    ``extrapolated[i]`` is True when the exposure lies outside the observed range.
    """
    if fit.get("status") != "ok" or fit.get("model") != "logistic":
        raise ValueError("predict_logistic needs a successful logistic fit")
    zc = z_crit(ci_level if ci_level is not None else float(fit["ci_level"]))
    xs = to_float_array(exposure, "exposure")
    beta = np.array([fit["coef"]["intercept"]["estimate"], fit["coef"]["slope"]["estimate"]])
    cov = np.array(fit["cov"], dtype=float)
    design = np.column_stack([np.ones_like(xs), xs])
    link = design @ beta
    link_se = np.sqrt(np.einsum("ij,jk,ik->i", design, cov, design))
    lo_x, hi_x = fit["exposure_range"]
    return {
        "exposure": sig_list(xs), "link": sig_list(link), "link_se": sig_list(link_se),
        "prob": sig_list(expit(link)),
        "lo": sig_list(expit(link - zc * link_se)), "hi": sig_list(expit(link + zc * link_se)),
        "extrapolated": [bool(v < lo_x or v > hi_x) for v in xs],
    }


def probability_curve(fit: dict[str, Any], *, n_points: int = 40,
                      grid: list[float] | None = None) -> dict[str, Any]:
    """The fitted P(response) curve with its CI band on an exposure grid
    (default: ``n_points`` evenly spaced over the observed exposure range)."""
    lo_x, hi_x = fit["exposure_range"]
    pts = list(grid) if grid else [float(v) for v in np.linspace(lo_x, hi_x, max(2, int(n_points)))]
    return predict_logistic(fit, pts)
