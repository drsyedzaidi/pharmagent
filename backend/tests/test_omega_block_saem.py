"""P1: SAEM estimation of a correlated (block) Omega.

Tolerances here are ANALYTIC, from the sampling distribution of a correlation
coefficient -- SE(r) ~ (1 - r^2)/sqrt(n) -- rather than tuned to observed runs.
A band fitted to a handful of runs encodes whatever those runs happened to do;
a band derived from the estimator's own sampling error states what it is.
"""
import math

import numpy as np
import pytest

from app.compute.nlme import (
    _VAR_FLOOR,
    _ind_obj,
    _laplace_subject,
    _make_predictor_cache,
    _numeric_hessian,
    _omega_prior,
    _PopSpec,
    _prepare_subjects,
    _project_block,
    _residual_variance,
    _shrink_to_pd,
    population_fit,
)
from app.compute.pk_models import get_model
from app.compute.pk_simulate import simulate

MODEL_KEY = "oral_1cmt"
MODEL = get_model(MODEL_KEY)
TRUE = {"CL": 4.0, "V": 40.0, "KA": 1.2}
OBS_T = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0]
DOSES = [{"time": 0.0, "amt": 100.0}]
ETA_SD = np.array([0.30, 0.25])
N_SUBJ = 60


def _cohort(r_true: float, seed: int) -> list[dict]:
    """Cohort whose CL/V random effects have a KNOWN correlation."""
    om = np.array([[ETA_SD[0] ** 2, r_true * ETA_SD[0] * ETA_SD[1]],
                   [r_true * ETA_SD[0] * ETA_SD[1], ETA_SD[1] ** 2]])
    chol = np.linalg.cholesky(om)
    rng = np.random.default_rng(seed)
    subs = []
    for i in range(N_SUBJ):
        e = chol @ rng.normal(0.0, 1.0, 2)
        p = {"CL": TRUE["CL"] * math.exp(e[0]),
             "V": TRUE["V"] * math.exp(e[1]), "KA": TRUE["KA"]}
        cp = np.asarray(simulate(MODEL, p, DOSES, OBS_T, wt=70.0)["cp"], dtype=float)
        subs.append({"subject": i, "doses": list(DOSES), "obs_t": list(OBS_T),
                     "obs_c": (cp * (1 + rng.normal(0.0, 0.08, cp.size))).tolist(),
                     "wt": 70.0})
    return subs


def _se_r(r: float, n: int = N_SUBJ) -> float:
    """Large-sample SE of a correlation estimate."""
    return (1.0 - r ** 2) / math.sqrt(n)


def _fit(subs, block=("CL", "V"), seed=7):
    return population_fit(MODEL_KEY, subs, method="saem", iiv_params=["CL", "V"],
                          error_model="proportional", max_iter=250, seed=seed,
                          compute_uncertainty=False,
                          omega_block=list(block) if block else None)


R_TRUE = 0.60


@pytest.fixture(scope="module")
def recovered():
    return _fit(_cohort(R_TRUE, seed=20250614))


@pytest.fixture(scope="module")
def null_fit():
    return _fit(_cohort(0.0, seed=999))


# ── the objective carries the block prior ───────────────────────────────────

def test_laplace_m2ll_matches_an_independent_assembly():
    """THE guard against the highest-risk silent failure: `prior` must reach
    BOTH _ind_obj consumers inside _laplace_subject -- _conditional_mode AND
    the half_obj closure used for the Hessian. If the closure were left on the
    diagonal penalty the mode would come from Omega^-1 while log_det_H came
    from diag(1/omega2); the correlation would still look right, so a recovery
    test alone would NOT catch it, while every OFV/AIC/LRT number silently broke.

    Re-derived here from Omega^-1 explicitly, sharing no code with the module.
    """
    spec = _PopSpec(MODEL, ["CL", "V"], "proportional", omega_block=["CL", "V"])
    subj = _prepare_subjects([{
        "subject": 1, "doses": list(DOSES), "obs_t": [0.5, 1.0, 2.0, 4.0, 8.0],
        "obs_c": [3.1, 5.2, 6.0, 4.4, 2.1], "wt": 70.0}])[0]
    om = np.array([[0.093, 0.042], [0.042, 0.065]])
    o2v = np.array([om[0, 0], om[1, 1]])
    prior = _omega_prior(om)

    m2ll, eta_hat = _laplace_subject(spec, subj, TRUE, o2v, 0.15, 0.0, None, prior)

    pred = _make_predictor_cache(spec, subj, TRUE)
    o_inv = np.linalg.inv(om)

    def ind(eta):
        f = pred(np.asarray(eta, dtype=float))
        var = _residual_variance(spec, f, 0.15, 0.0)
        resid = subj.c - f
        return float(np.sum(resid ** 2 / var + np.log(2 * np.pi * var))
                     + eta @ o_inv @ eta)

    hess = _numeric_hessian(lambda e: 0.5 * ind(e), eta_hat)
    ev = np.clip(np.linalg.eigvalsh(0.5 * (hess + hess.T)), _VAR_FLOOR, None)
    expect = ind(eta_hat) + float(np.log(np.linalg.det(om))) + float(np.sum(np.log(ev)))
    assert m2ll == pytest.approx(expect, abs=1e-6)


def test_ind_obj_penalty_uses_the_full_precision():
    spec = _PopSpec(MODEL, ["CL", "V"], "proportional", omega_block=["CL", "V"])
    subj = _prepare_subjects([{
        "subject": 1, "doses": list(DOSES), "obs_t": [1.0, 4.0],
        "obs_c": [5.0, 4.0], "wt": 70.0}])[0]
    pred = _make_predictor_cache(spec, subj, TRUE)
    om = np.array([[0.093, 0.042], [0.042, 0.065]])
    o2v = np.array([om[0, 0], om[1, 1]])
    eta = np.array([0.2, -0.15])
    diag_val = _ind_obj(eta, pred, subj, spec, o2v, 0.15, 0.0, None)
    blk_val = _ind_obj(eta, pred, subj, spec, o2v, 0.15, 0.0, _omega_prior(om))
    # the two penalties differ by exactly the change in the quadratic form
    delta = float(eta @ np.linalg.inv(om) @ eta) - float(np.sum(eta ** 2 / o2v))
    assert blk_val - diag_val == pytest.approx(delta, abs=1e-9)


# ── recovery and the null control ───────────────────────────────────────────

def test_saem_recovers_a_known_correlation(recovered):
    r_hat = recovered["omega_block_corr"]["CL~V"]
    bound = 2.6 * _se_r(R_TRUE)          # ~99% of the sampling distribution
    assert abs(r_hat - R_TRUE) < bound, f"r_hat={r_hat} vs {R_TRUE} +/- {bound}"


def test_saem_recovers_the_marginal_variances(recovered):
    om = np.asarray(recovered["omega_matrix"], dtype=float)
    assert om[0, 0] == pytest.approx(ETA_SD[0] ** 2, rel=0.55)
    assert om[1, 1] == pytest.approx(ETA_SD[1] ** 2, rel=0.55)


def test_saem_recovers_theta_with_a_block(recovered):
    theta = recovered["theta"]
    assert TRUE["CL"] * 0.8 < theta["CL"] < TRUE["CL"] * 1.2
    assert TRUE["V"] * 0.8 < theta["V"] < TRUE["V"] * 1.2


def test_block_does_not_manufacture_correlation(null_fit):
    """Uncorrelated truth must not come back correlated."""
    r_hat = null_fit["omega_block_corr"]["CL~V"]
    bound = 2.6 * _se_r(0.0)
    assert abs(r_hat) < bound, f"|r_hat|={abs(r_hat)} exceeds {bound} at r=0"


def test_fitted_omega_is_symmetric_and_positive_definite(recovered):
    om = np.asarray(recovered["omega_matrix"], dtype=float)
    np.testing.assert_allclose(om, om.T, rtol=0, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(om) > 0.0)


def test_block_is_deterministic_for_a_fixed_seed():
    subs = _cohort(R_TRUE, seed=4242)
    a, b = _fit(subs, seed=11), _fit(subs, seed=11)
    assert a["omega_matrix"] == b["omega_matrix"]
    assert a["theta"] == b["theta"]


# ── result contract ─────────────────────────────────────────────────────────

def test_block_keys_present_only_for_a_block_fit(recovered):
    for key in ("omega_block", "omega_matrix", "omega_corr", "omega_block_corr"):
        assert key in recovered
    assert recovered["omega_block"] == ["CL", "V"]
    diag = _fit(_cohort(0.0, seed=5), block=None)
    for key in ("omega_block", "omega_matrix", "omega_corr", "omega_block_corr"):
        assert key not in diag, f"{key} leaked into a diagonal payload"


def test_reported_correlation_matches_the_reported_matrix(recovered):
    om = np.asarray(recovered["omega_matrix"], dtype=float)
    expect = om[0, 1] / math.sqrt(om[0, 0] * om[1, 1])
    assert recovered["omega_block_corr"]["CL~V"] == pytest.approx(expect, abs=1e-5)


def test_omega_cv_pct_uses_the_marginal_variances(recovered):
    om = np.asarray(recovered["omega_matrix"], dtype=float)
    for k, p in enumerate(["CL", "V"]):
        expect = 100.0 * math.sqrt(math.exp(om[k, k]) - 1.0)
        assert recovered["omega_cv_pct"][p] == pytest.approx(expect, rel=1e-3)


@pytest.mark.parametrize("method", ["saem", "focei", "focei_saem", "auto"])
def test_every_method_honours_omega_block(method):
    """A method that accepted omega_block but returned a DIAGONAL fit would let
    a caller report 'no correlation' as a finding. Each path must actually
    carry the block through to the result."""
    subs = _cohort(0.0, seed=3)
    r = population_fit(MODEL_KEY, subs, method=method, iiv_params=["CL", "V"],
                       error_model="proportional", max_iter=25, seed=3,
                       compute_uncertainty=False, omega_block=["CL", "V"])
    assert r["omega_block"] == ["CL", "V"]
    om = np.asarray(r["omega_matrix"], dtype=float)
    assert om.shape == (2, 2)
    np.testing.assert_allclose(om, om.T, rtol=0, atol=1e-12)


# ── structural helpers ──────────────────────────────────────────────────────

def test_project_block_zeroes_off_block_covariances():
    spec = _PopSpec(MODEL, ["CL", "V", "KA"], "proportional", omega_block=["CL", "KA"])
    dense = np.array([[0.09, 0.02, 0.03], [0.02, 0.05, 0.04], [0.03, 0.04, 0.06]])
    out = _project_block(spec, dense)
    assert out[0, 2] == 0.03 and out[2, 0] == 0.03      # inside the block: kept
    assert out[0, 1] == 0.0 and out[1, 2] == 0.0        # outside: zeroed
    np.testing.assert_allclose(np.diag(out), np.diag(dense))


def test_shrink_to_pd_repairs_an_indefinite_matrix():
    bad = np.array([[0.09, 0.30], [0.30, 0.05]])        # |r| > 1, not PD
    assert np.min(np.linalg.eigvalsh(bad)) < 0.0
    fixed = _shrink_to_pd(bad)
    assert np.all(np.linalg.eigvalsh(fixed) > 0.0)
    np.testing.assert_allclose(np.diag(fixed), np.diag(bad), rtol=0, atol=1e-12)


def test_shrink_to_pd_leaves_a_valid_matrix_untouched():
    ok = np.array([[0.093, 0.042], [0.042, 0.065]])
    np.testing.assert_allclose(_shrink_to_pd(ok), ok, rtol=0, atol=1e-15)


# ── delta-method standard errors for the block ──────────────────────────────
# The packed block parameters are Cholesky entries -- log-scale on the diagonal,
# raw-scale below it -- so the "RSE% = 100*SE" shortcut that holds for every
# other population parameter does not apply, and each reported quantity needs
# its own gradient.

@pytest.fixture(scope="module")
def recovered_se():
    """Same cohort as `recovered`, fitted WITH the uncertainty pass."""
    return population_fit(MODEL_KEY, _cohort(R_TRUE, seed=20250614), method="saem",
                          iiv_params=["CL", "V"], error_model="proportional",
                          max_iter=250, seed=7, compute_uncertainty=True,
                          omega_block=["CL", "V"])


def _delta_se(spec, om, cov_seg):
    from app.compute.nlme import _omega_delta_se, _seg_from_omega_full
    return _omega_delta_se(spec, _seg_from_omega_full(spec, om), cov_seg)


def test_delta_se_matches_monte_carlo_through_the_same_decode():
    """The delta method is a FIRST-ORDER approximation, so this pins how good
    it actually is rather than asserting it is exact: sample the Cholesky
    parameters from their own covariance, push each draw through the same
    decode the fit uses, and compare the empirical spread.
    """
    from app.compute.nlme import _block_corr, _omega_full_from_seg, _seg_from_omega_full
    spec = _PopSpec(MODEL, ["CL", "V"], "proportional", omega_block=["CL", "V"])
    om = np.array([[0.093, 0.042], [0.042, 0.065]])
    seg = _seg_from_omega_full(spec, om)
    cov_seg = np.diag([0.09 ** 2, 0.09 ** 2, 0.09 ** 2 * 0.35])
    _var_se, corr_se = _delta_se(spec, om, cov_seg)

    rng = np.random.default_rng(4)
    chol = np.linalg.cholesky(cov_seg)
    draws = seg[None, :] + (chol @ rng.normal(0.0, 1.0, (seg.size, 40000))).T
    r = np.array([_block_corr(_omega_full_from_seg(spec, s))[0, 1] for s in draws])
    ratio = corr_se["CL~V"] / float(r.std(ddof=1))
    # First-order truncation makes the analytic SE slightly SMALL; it must not
    # be wild in either direction.
    assert 0.90 < ratio < 1.05, f"delta/MC = {ratio}"


def test_delta_se_converges_to_monte_carlo_as_uncertainty_tightens():
    """Sharper check on the same approximation: shrinking the covariance must
    drive the ratio toward 1, because the linearization becomes exact."""
    from app.compute.nlme import _block_corr, _omega_full_from_seg, _seg_from_omega_full
    spec = _PopSpec(MODEL, ["CL", "V"], "proportional", omega_block=["CL", "V"])
    om = np.array([[0.093, 0.042], [0.042, 0.065]])
    seg = _seg_from_omega_full(spec, om)
    rng = np.random.default_rng(5)
    ratios = []
    for scale in (0.09, 0.03):
        cov_seg = np.diag([scale ** 2, scale ** 2, scale ** 2 * 0.35])
        _v, corr_se = _delta_se(spec, om, cov_seg)
        chol = np.linalg.cholesky(cov_seg)
        draws = seg[None, :] + (chol @ rng.normal(0.0, 1.0, (seg.size, 40000))).T
        r = np.array([_block_corr(_omega_full_from_seg(spec, s))[0, 1] for s in draws])
        ratios.append(corr_se["CL~V"] / float(r.std(ddof=1)))
    assert abs(ratios[1] - 1.0) < abs(ratios[0] - 1.0), ratios
    assert ratios[1] > 0.97


def test_block_members_get_a_real_rse_not_none(recovered_se):
    """Before the delta method these were None because a block member's
    marginal variance has no single slot in the packed vector."""
    for p in ("CL", "V"):
        rse = recovered_se["omega_rse_pct"][p]
        assert rse is not None, f"{p} RSE still missing"
        assert 0.0 < rse < 100.0, f"{p} RSE implausible: {rse}"


def test_correlation_has_a_standard_error(recovered_se):
    se = recovered_se["omega_corr_se"]["CL~V"]
    assert se is not None and math.isfinite(se) and se > 0.0


def test_correlation_confidence_interval_covers_the_truth(recovered_se):
    """The point of an SE: a Wald interval that behaves."""
    r_hat = recovered_se["omega_block_corr"]["CL~V"]
    se = recovered_se["omega_corr_se"]["CL~V"]
    lo, hi = r_hat - 1.96 * se, r_hat + 1.96 * se
    assert lo < R_TRUE < hi, f"95% CI [{lo}, {hi}] misses r={R_TRUE}"
    assert -1.0 <= lo and hi <= 1.5          # sane magnitude, not a runaway


def test_correlation_interval_excludes_zero_when_correlation_is_real(recovered_se):
    r_hat = recovered_se["omega_block_corr"]["CL~V"]
    se = recovered_se["omega_corr_se"]["CL~V"]
    assert abs(r_hat) > 1.96 * se, "a true r=0.6 should be distinguishable from 0"


def test_diagonal_fit_reports_no_correlation_se():
    """The key must not leak into a diagonal payload."""
    diag = population_fit(MODEL_KEY, _cohort(0.0, seed=8), method="saem",
                          iiv_params=["CL", "V"], error_model="proportional",
                          max_iter=60, seed=7, compute_uncertainty=True)
    assert "omega_corr_se" not in diag
    for p in ("CL", "V"):
        assert p in diag["omega_rse_pct"]


def test_delta_se_is_zero_when_the_parameters_are_certain():
    """Degenerate but load-bearing: a zero covariance must give zero SE, not
    a NaN out of the square root."""
    spec = _PopSpec(MODEL, ["CL", "V"], "proportional", omega_block=["CL", "V"])
    om = np.array([[0.093, 0.042], [0.042, 0.065]])
    var_se, corr_se = _delta_se(spec, om, np.zeros((3, 3)))
    assert corr_se["CL~V"] == 0.0
    assert all(v == 0.0 for v in var_se.values())


# ── FOCE-I block (the Cholesky entries move in the outer Powell vector) ──────

@pytest.fixture(scope="module")
def focei_block():
    """n=40 keeps the outer optimization affordable in CI."""
    om = np.array([[ETA_SD[0] ** 2, R_TRUE * ETA_SD[0] * ETA_SD[1]],
                   [R_TRUE * ETA_SD[0] * ETA_SD[1], ETA_SD[1] ** 2]])
    chol = np.linalg.cholesky(om)
    rng = np.random.default_rng(20250614)
    subs = []
    for i in range(40):
        e = chol @ rng.normal(0.0, 1.0, 2)
        p = {"CL": TRUE["CL"] * math.exp(e[0]),
             "V": TRUE["V"] * math.exp(e[1]), "KA": TRUE["KA"]}
        cp = np.asarray(simulate(MODEL, p, DOSES, OBS_T, wt=70.0)["cp"], dtype=float)
        subs.append({"subject": i, "doses": list(DOSES), "obs_t": list(OBS_T),
                     "obs_c": (cp * (1 + rng.normal(0.0, 0.08, cp.size))).tolist(),
                     "wt": 70.0})
    return subs


def test_focei_recovers_a_known_correlation(focei_block):
    r = population_fit(MODEL_KEY, focei_block, method="focei",
                       iiv_params=["CL", "V"], error_model="proportional",
                       max_iter=200, compute_uncertainty=False,
                       omega_block=["CL", "V"])
    r_hat = r["omega_block_corr"]["CL~V"]
    assert abs(r_hat - R_TRUE) < 2.6 * _se_r(R_TRUE, 40)


def test_focei_block_beats_its_own_diagonal_on_ofv(focei_block):
    """The block nests the diagonal, so on data that really is correlated the
    extra parameter must buy a materially better objective -- this is the
    likelihood-ratio evidence that justifies fitting a block at all."""
    kw = dict(iiv_params=["CL", "V"], error_model="proportional",
              max_iter=200, compute_uncertainty=False)
    diag = population_fit(MODEL_KEY, focei_block, method="focei", **kw)
    blk = population_fit(MODEL_KEY, focei_block, method="focei",
                         omega_block=["CL", "V"], **kw)
    # chi-square(1) at p=0.05 is 3.84; require a clearly significant drop.
    assert blk["ofv"] < diag["ofv"] - 3.84


def test_focei_block_is_deterministic(focei_block):
    kw = dict(method="focei", iiv_params=["CL", "V"], error_model="proportional",
              max_iter=120, compute_uncertainty=False, omega_block=["CL", "V"])
    a = population_fit(MODEL_KEY, focei_block, **kw)
    b = population_fit(MODEL_KEY, focei_block, **kw)
    assert a["omega_matrix"] == b["omega_matrix"]
    assert a["ofv"] == b["ofv"]


def test_warm_start_carries_the_off_diagonals():
    """_warm_init builds omega2 from omega_cv_pct, which holds only MARGINAL
    variances; without the matrix a seeded block fit would silently restart
    from an uncorrelated Omega and throw away what the seed found."""
    from app.compute.nlme import _warm_init
    res = {"theta": {"CL": 4.0, "V": 40.0}, "omega_cv_pct": {"CL": 30.0, "V": 25.0},
           "sigma": {"prop": 0.1, "add": None}, "covariate_effects": [],
           "omega_matrix": [[0.093, 0.042], [0.042, 0.065]]}
    init = _warm_init(res)
    assert init["omega_matrix"] == [[0.093, 0.042], [0.042, 0.065]]


def test_focei_block_reports_a_correlation_standard_error(focei_block):
    r = population_fit(MODEL_KEY, focei_block, method="focei",
                       iiv_params=["CL", "V"], error_model="proportional",
                       max_iter=200, compute_uncertainty=True,
                       omega_block=["CL", "V"])
    se = r["omega_corr_se"]["CL~V"]
    assert se is not None and math.isfinite(se) and se > 0.0
    for p in ("CL", "V"):
        assert r["omega_rse_pct"][p] is not None
