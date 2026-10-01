"""Logistic exposure-response: golden values from R's ``glm`` plus edge handling.

VALIDATION METHOD. The expected numbers below were produced by R 4.5.1
(``arch -arm64 Rscript``) on the SAME fixed data embedded here, then hard-coded.
Agreement is asserted at 1e-6 relative for every coefficient-level quantity.

R's ``glm`` stops its IRLS loop at epsilon = 1e-8 on the deviance and ``summary``
takes the weights of the PENULTIMATE iterate, so its standard errors are only
accurate to ~1e-6 relative. The golden run therefore tightens the stopping rule
(``control = glm.control(epsilon = 1e-14, maxit = 100)``); the stock-default
R standard errors are asserted separately at 5e-6 (``test_stock_r_default_se``).

R script (verbatim; output follows):

    show <- function(tag, v) cat(sprintf("%s: %s\n", tag, paste(sprintf("%.12g", v), collapse = ", ")))
    x1 <- c(95.2, 139.5, 75.6, 106.5, 175.2, 82.8, 113.7, 237.4, 221.9, 62.0, 88.2, 141.9, 81.7, 115.4, 120.1, 99.3, 253.6, 115.5, 197.9, 162.6, 221.7, 77.1, 113.2, 105.0, 179.8, 66.6, 51.1, 117.8, 102.6, 176.3)
    y1 <- c(0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1)
    f1 <- glm(y1 ~ x1, family = binomial, control = glm.control(epsilon = 1e-14, maxit = 100))
    cf <- summary(f1)$coefficients
    show("L1 coef", cf[, 1]); show("L1 se", cf[, 2]); show("L1 z", cf[, 3]); show("L1 p", cf[, 4])
    show("L1 deviance", deviance(f1)); show("L1 null.deviance", f1$null.deviance)
    show("L1 aic", AIC(f1)); show("L1 loglik", as.numeric(logLik(f1))); show("L1 sd_x", sd(x1))
    show("L1 ci_slope_wald", confint.default(f1)[2, ])
    pr <- predict(f1, newdata = data.frame(x1 = c(60, 100, 150, 240)), se.fit = TRUE, type = "link")
    show("L1 pred_link", pr$fit); show("L1 pred_link_se", pr$se.fit); show("L1 pred_prob", plogis(pr$fit))
    show("L1 pred_lo", plogis(pr$fit - qnorm(0.975) * pr$se.fit)); show("L1 pred_hi", plogis(pr$fit + qnorm(0.975) * pr$se.fit))
    lr <- f1$null.deviance - deviance(f1); show("L1 lr_chisq", lr); show("L1 lr_p", pchisq(lr, 1, lower.tail = FALSE))
    (L2: same calls on x2 = Cmax-like values ~1300-5500, y2; n = 40)

R output (tight convergence):

    L1 coef: -3.93844538763, 0.0317702713872
    L1 se: 1.52206946345, 0.0124561694536
    L1 z: -2.58755955769, 2.55056512402
    L1 p: 0.00966584843831, 0.0107548428902
    L1 deviance: 29.7633764069      L1 null.deviance: 41.5888308336
    L1 aic: 33.7633764069           L1 loglik: -14.8816882034      L1 sd_x: 55.2336724717
    L1 ci_slope_wald: 0.00735662787287, 0.0561839149015
    L1 pred_link: -2.0322291044, -0.761418248909, 0.82709532045, 3.68641974529
    L1 pred_link_se: 0.837785600191, 0.495952466052, 0.611781601557, 1.60006202846
    L1 pred_prob: 0.115860384247, 0.318338428252, 0.695740400544, 0.975551157919
    L1 pred_lo: 0.0247405797433, 0.150144621335, 0.408063978521, 0.634218134576
    L1 pred_hi: 0.40366927576, 0.552463859342, 0.883517003323, 0.998912171873
    L1 lr_chisq: 11.8254544267      L1 lr_p: 0.000584264533081
    L2 coef: -7.93623177485, 0.00336817702119
    L2 se: 2.47587051658, 0.00104451490164
    L2 z: -3.20543086632, 3.22463280889
    L2 p: 0.00134860438951, 0.00126134336165
    L2 deviance: 31.0842348312      L2 null.deviance: 54.5483686985      L2 sd_x: 850.511396312

Stock-default glm (epsilon 1e-8) L1 se: 1.52206796669, 0.0124561539183 (rel. diff ~1e-6, see above).
"""
import json
import math

import numpy as np
import pytest

from app.compute.er_models import fit_logistic_er, predict_logistic

REL = 1e-6   # stated tolerance vs R glm

X1 = [95.2, 139.5, 75.6, 106.5, 175.2, 82.8, 113.7, 237.4, 221.9, 62.0, 88.2, 141.9, 81.7, 115.4, 120.1, 99.3,
      253.6, 115.5, 197.9, 162.6, 221.7, 77.1, 113.2, 105.0, 179.8, 66.6, 51.1, 117.8, 102.6, 176.3]
Y1 = [0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1]
X2 = [3971, 3979, 3351, 2959, 2969, 3246, 2156, 2789, 3341, 1991, 1915, 3891, 1574, 2273, 2410, 2016, 2139, 2461,
      2670, 2040, 3525, 3272, 2870, 2175, 5535, 2025, 1964, 2884, 3606, 2083, 3248, 1276, 1928, 2707, 1543, 1697,
      1955, 2684, 2597, 1822]
Y2 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0,
      0, 1, 1, 0]


def _close(got, want, rel=REL):
    assert got == pytest.approx(want, rel=rel), (got, want)


# ── golden: R glm ────────────────────────────────────────────────────────────

def test_l1_coefficients_and_standard_errors_match_r_glm():
    r = fit_logistic_er(X1, Y1)
    assert r["status"] == "ok" and r["converged"]
    c = r["coef"]
    _close(c["intercept"]["estimate"], -3.93844538763)
    _close(c["slope"]["estimate"], 0.0317702713872)
    _close(c["intercept"]["se"], 1.52206946345)
    _close(c["slope"]["se"], 0.0124561694536)
    _close(c["intercept"]["z"], -2.58755955769)
    _close(c["slope"]["z"], 2.55056512402)
    _close(c["intercept"]["p"], 0.00966584843831)
    _close(c["slope"]["p"], 0.0107548428902)


def test_l1_fit_statistics_match_r():
    r = fit_logistic_er(X1, Y1)
    _close(r["deviance"], 29.7633764069)
    _close(r["null_deviance"], 41.5888308336)
    _close(r["aic"], 33.7633764069)
    _close(r["loglik"], -14.8816882034)
    _close(r["lr_chisq"], 11.8254544267)
    _close(r["lr_p"], 0.000584264533081)
    assert r["n"] == 30 and r["n_events"] == 15 and r["n_nonevents"] == 15


def test_l1_odds_ratios_per_unit_and_per_sd():
    r = fit_logistic_er(X1, Y1)
    sd = 55.2336724717
    _close(r["or_per_sd"]["sd"], sd)
    b, lo, hi = 0.0317702713872, 0.00735662787287, 0.0561839149015   # R coef, confint.default
    _close(r["or_per_unit"]["estimate"], math.exp(b))
    _close(r["or_per_unit"]["lo"], math.exp(lo))
    _close(r["or_per_unit"]["hi"], math.exp(hi))
    _close(r["or_per_sd"]["estimate"], math.exp(b * sd))
    _close(r["or_per_sd"]["lo"], math.exp(lo * sd))
    _close(r["or_per_sd"]["hi"], math.exp(hi * sd))


def test_l2_large_unit_scale_matches_r():
    """Exposure ~1e3: the fit must be scale-safe (standardised internally)."""
    r = fit_logistic_er(X2, Y2)
    assert r["status"] == "ok"
    _close(r["coef"]["intercept"]["estimate"], -7.93623177485)
    _close(r["coef"]["slope"]["estimate"], 0.00336817702119)
    _close(r["coef"]["intercept"]["se"], 2.47587051658)
    _close(r["coef"]["slope"]["se"], 0.00104451490164)
    _close(r["coef"]["slope"]["p"], 0.00126134336165)
    _close(r["deviance"], 31.0842348312)
    _close(r["null_deviance"], 54.5483686985)
    _close(r["or_per_sd"]["sd"], 850.511396312)


def test_predicted_probability_and_delta_method_band_match_r():
    r = fit_logistic_er(X1, Y1)
    p = predict_logistic(r, [60, 100, 150, 240])
    for got, want in zip(p["link"], [-2.0322291044, -0.761418248909, 0.82709532045, 3.68641974529]):
        _close(got, want)
    for got, want in zip(p["link_se"], [0.837785600191, 0.495952466052, 0.611781601557, 1.60006202846]):
        _close(got, want)
    for got, want in zip(p["prob"], [0.115860384247, 0.318338428252, 0.695740400544, 0.975551157919]):
        _close(got, want)
    for got, want in zip(p["lo"], [0.0247405797433, 0.150144621335, 0.408063978521, 0.634218134576]):
        _close(got, want)
    for got, want in zip(p["hi"], [0.40366927576, 0.552463859342, 0.883517003323, 0.998912171873]):
        _close(got, want)


def test_prediction_flags_extrapolation_outside_observed_exposure_range():
    r = fit_logistic_er(X1, Y1)
    p = predict_logistic(r, [10.0, 100.0, 1000.0])
    assert p["extrapolated"] == [True, False, True]


# ── synthetic recovery, fixed seed ───────────────────────────────────────────

def test_recovers_known_coefficients_within_sampling_error():
    rng = np.random.default_rng(7)
    n, a_true, b_true = 3000, -3.0, 0.03
    x = np.exp(rng.normal(np.log(100.0), 0.4, n))
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-(a_true + b_true * x)))).astype(int)
    r = fit_logistic_er(x.tolist(), y.tolist())
    for key, truth in (("intercept", a_true), ("slope", b_true)):
        est, se = r["coef"][key]["estimate"], r["coef"][key]["se"]
        assert abs(est - truth) < 3.0 * se, (key, est, truth, se)


def test_odds_ratio_per_sd_is_invariant_to_exposure_units():
    a = fit_logistic_er(X1, Y1)
    b = fit_logistic_er([v * 1000.0 for v in X1], Y1)         # mg -> ug style rescale
    _close(b["coef"]["slope"]["estimate"], a["coef"]["slope"]["estimate"] / 1000.0)
    _close(b["or_per_sd"]["estimate"], a["or_per_sd"]["estimate"])
    _close(b["or_per_sd"]["lo"], a["or_per_sd"]["lo"])


# ── refusals: structured, never garbage ──────────────────────────────────────

def test_complete_separation_is_refused_not_fitted():
    x = list(range(1, 13))
    y = [0] * 6 + [1] * 6
    r = fit_logistic_er(x, y)
    assert r["status"] == "separation"
    assert "coef" not in r and "does not exist" in r["message"].lower()


def test_quasi_complete_separation_at_a_tie_is_refused():
    x = [1, 2, 3, 4, 5, 5, 5, 6, 7, 8, 9, 10]
    y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]       # overlap only at the tie x = 5
    assert fit_logistic_er(x, y)["status"] == "separation"


def test_zero_variance_exposure_is_refused():
    r = fit_logistic_er([5.0] * 20, [0, 1] * 10)
    assert r["status"] == "zero_variance_exposure"


def test_too_few_events_is_refused_and_borderline_is_warned():
    x = [float(i) for i in range(1, 41)]
    few = [0] * 37 + [1, 0, 1]
    # three events only -> below the hard minimum
    assert fit_logistic_er(x, few)["status"] == "insufficient_events"
    rng = np.random.default_rng(3)
    y = np.zeros(40, dtype=int)
    y[rng.choice(40, 7, replace=False)] = 1
    r = fit_logistic_er(list(rng.normal(10, 2, 40)), y.tolist())
    assert r["status"] == "ok" and any("events per" in w.lower() for w in r["warnings"])


def test_invalid_inputs_are_structured_errors():
    assert fit_logistic_er([1, 2, 3], [0, 1])["status"] == "invalid_input"           # length mismatch
    assert fit_logistic_er([1, 2, 3, 4], [0, 1, 2, 1])["status"] == "invalid_input"  # non-binary
    assert fit_logistic_er([], [])["status"] == "invalid_input"


def test_non_finite_rows_are_dropped_and_counted():
    x = list(X1) + [float("nan"), 120.0]
    y = list(Y1) + [1, float("nan")]
    r = fit_logistic_er(x, y)
    assert r["status"] == "ok" and r["n"] == 30 and r["n_dropped_non_finite"] == 2
    _close(r["coef"]["slope"]["estimate"], 0.0317702713872)


def test_result_is_json_safe_and_carries_covariance_for_prediction():
    r = fit_logistic_er(X1, Y1)
    s = json.dumps(r)
    assert "NaN" not in s and "Infinity" not in s
    assert len(r["cov"]) == 2 and len(r["cov"][0]) == 2
    assert r["exposure_range"] == [min(X1), max(X1)]


def test_booleans_are_accepted_as_binary_response():
    r = fit_logistic_er(X1, [bool(v) for v in Y1])
    _close(r["coef"]["slope"]["estimate"], 0.0317702713872)


def test_stock_r_default_se():
    """R's default-epsilon glm SEs (1.52206796669, 0.0124561539183) are accurate to ~1e-6 only."""
    c = fit_logistic_er(X1, Y1)["coef"]
    _close(c["intercept"]["se"], 1.52206796669, rel=5e-6)
    _close(c["slope"]["se"], 0.0124561539183, rel=5e-6)


def test_large_sample_converges():
    """Regression guard: at n = 40000 the log-likelihood is ~1e4 and its rounding noise exceeds any
    absolute tolerance, which is why the step-halving monotonicity check is relative."""
    rng = np.random.default_rng(13)
    n = 40000
    x = np.exp(rng.normal(np.log(100.0), 0.4, n))
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-(-3.0 + 0.03 * x)))).astype(int)
    r = fit_logistic_er(x.tolist(), y.tolist())
    assert r["status"] == "ok" and r["converged"]
    assert abs(r["coef"]["slope"]["estimate"] - 0.03) < 3.0 * r["coef"]["slope"]["se"]


def test_newton_mle_agrees_with_an_independent_scipy_optimiser():
    """Independent of R: BFGS on the analytic gradient reaches the same maximum (L1 and L2)."""
    from scipy.optimize import minimize
    from scipy.special import expit

    for xs, ys in ((X1, Y1), (X2, Y2)):
        x = (np.array(xs) - np.mean(xs)) / np.std(xs, ddof=1)
        y = np.array(ys, dtype=float)

        def nll(b):
            eta = b[0] + b[1] * x
            return float(np.sum(np.logaddexp(0.0, eta) - y * eta))

        def grad(b):
            r = expit(b[0] + b[1] * x) - y
            return np.array([r.sum(), (r * x).sum()])

        opt = minimize(nll, [0.0, 0.0], jac=grad, method="BFGS", options={"gtol": 1e-10})
        fit = fit_logistic_er(xs, ys)
        sd, mean = np.std(xs, ddof=1), np.mean(xs)
        slope = fit["coef"]["slope"]["estimate"] * sd              # slope on the standardised scale
        icpt = fit["coef"]["intercept"]["estimate"] + fit["coef"]["slope"]["estimate"] * mean
        _close(slope, opt.x[1], rel=1e-5)
        _close(icpt, opt.x[0], rel=1e-5)
        _close(fit["loglik"], -opt.fun, rel=1e-9)
