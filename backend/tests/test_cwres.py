"""Tests for CWRES (app.compute.nlme.posthoc_residuals / _cwres_subject /
_jacobian / _whiten) -- conditional weighted residuals, Hooker, Staatz &
Karlsson 2007.

The Jacobian (G = df/deta at eta_hat) and the whitening root are each
validated against a closed form / an implementation-independent property
before any pooled statistic is trusted: a wrong G or a wrong root can still
produce mean~0/sd~1 pooled CWRES (see test_whitening_root_choice_matters),
so those pooled checks alone would not catch either defect.
"""
from __future__ import annotations

import json
import math

import numpy as np

from app.compute.nlme import (
    _cwres_subject,
    _jacobian,
    _PopSpec,
    _prepare_subjects,
    _sid,
    _whiten,
    posthoc_residuals,
)
from app.compute.pk_models import get_model
from app.compute.pk_simulate import simulate

IV_KEY = "iv_1cmt"
ORAL_KEY = "oral_1cmt"


def _spec(model_key: str, iiv_params: list[str], error_model: str = "proportional") -> _PopSpec:
    return _PopSpec(get_model(model_key), iiv_params, error_model)


# ── 1. Jacobian vs closed form (mandatory, against the shipped code path) ────

def test_jacobian_matches_closed_form_iv_bolus():
    # iv_1cmt, single bolus, wt=70 (allometric factor exactly 1 for both CL
    # and V, so it cannot mask a wrong Jacobian): C(t) = (D/V)*exp(-(CL/V)t).
    # With eta on [CL, V]:
    #   dC/deta_CL = C * (-CL*t/V)
    #   dC/deta_V  = C * (CL*t/V - 1)
    spec = _spec(IV_KEY, ["CL", "V"])
    subjects_raw = [{"subject": "S1", "doses": [{"time": 0.0, "amt": 100.0}],
                     "obs_t": [0.5, 1.0, 2.0, 4.0, 8.0, 12.0],
                     "obs_c": [1.0] * 6, "wt": 70.0}]
    subj = _prepare_subjects(subjects_raw)[0]
    theta = {"CL": 5.0, "V": 50.0}
    eta_hat = np.array([0.15, -0.10])

    g = _jacobian(spec, subj, theta, eta_hat)
    assert g.shape == (subj.t.size, 2)

    cl_i = theta["CL"] * math.exp(eta_hat[0])
    v_i = theta["V"] * math.exp(eta_hat[1])
    t = subj.t
    c = (100.0 / v_i) * np.exp(-(cl_i / v_i) * t)
    dc_deta_cl = c * (-cl_i * t / v_i)
    dc_deta_v = c * (cl_i * t / v_i - 1.0)

    np.testing.assert_allclose(g[:, 0], dc_deta_cl, rtol=1e-4)
    np.testing.assert_allclose(g[:, 1], dc_deta_v, rtol=1e-4)


def test_jacobian_stable_on_nonlinear_path():
    # iv_1cmt_mm (Michaelis-Menten elimination) has no matrix-exponential fast
    # path (an ODE is solved), covering the FD Jacobian on the LSODA path
    # nothing else here touches. Two FD steps should agree to ~1e-2 relative.
    from app.compute.nlme import _CWRES_FD_STEP
    spec = _spec("iv_1cmt_mm", ["VMAX"])
    subjects_raw = [{"subject": "S1", "doses": [{"time": 0.0, "amt": 100.0}],
                     "obs_t": [0.5, 2.0, 6.0], "obs_c": [1.0, 1.0, 1.0], "wt": 70.0}]
    subj = _prepare_subjects(subjects_raw)[0]
    model = get_model("iv_1cmt_mm")
    theta = dict(model.defaults)
    eta_hat = np.array([0.1])

    g1 = _jacobian(spec, subj, theta, eta_hat, step=_CWRES_FD_STEP)
    g2 = _jacobian(spec, subj, theta, eta_hat, step=_CWRES_FD_STEP * 3.0)
    assert np.all(np.isfinite(g1)) and np.all(np.isfinite(g2))
    assert np.any(np.abs(g1) > 0)
    np.testing.assert_allclose(g1, g2, rtol=2e-2, atol=1e-6)


# ── 2. CWRES is a real linearization, not disguised IWRES ───────────────────

def test_g_is_nonzero_and_cwres_differs_from_iwres():
    # At realistic (nonzero) omega, a correct Jacobian has every column
    # nonzero, and CWRES != IWRES in general (they coincide only in the
    # omega->0 limit, which is a separate, weaker check -- see below).
    spec = _spec(IV_KEY, ["CL", "V"])
    subjects_raw = [{"subject": "S1", "doses": [{"time": 0.0, "amt": 100.0}],
                     "obs_t": [0.5, 1.0, 2.0, 4.0, 8.0], "obs_c": [2.0, 1.5, 1.0, 0.5, 0.2],
                     "wt": 70.0}]
    subj = _prepare_subjects(subjects_raw)[0]
    theta = {"CL": 5.0, "V": 50.0}
    eta_hat = np.array([0.2, -0.15])
    omega2_vec = np.array([0.09, 0.04])

    out = _cwres_subject(spec, subj, theta, eta_hat, omega2_vec, 0.1, 0.5, interaction=True)
    assert out is not None
    assert np.all(np.linalg.norm(out["g"], axis=0) > 0)
    assert not np.allclose(out["cwres"], out["iwres"])


def test_cwres_reduces_to_iwres_when_omega_vanishes():
    # LIMIT check only (NOT a substitute for the nonzero-Jacobian guard above:
    # omega->0 makes Cov ~ diag(R) regardless of G, so this configuration
    # cannot distinguish a correct G from an all-zero one).
    spec = _spec(IV_KEY, ["CL", "V"])
    subjects_raw = [{"subject": "S1", "doses": [{"time": 0.0, "amt": 100.0}],
                     "obs_t": [0.5, 1.0, 2.0, 4.0], "obs_c": [2.0, 1.5, 1.0, 0.5],
                     "wt": 70.0}]
    subj = _prepare_subjects(subjects_raw)[0]
    theta = {"CL": 5.0, "V": 50.0}
    eta_hat = np.zeros(2)
    omega2_vec = np.array([1e-9, 1e-9])

    out = _cwres_subject(spec, subj, theta, eta_hat, omega2_vec, 0.1, 0.5, interaction=True)
    assert out is not None
    np.testing.assert_allclose(out["cwres"], out["iwres"], rtol=1e-3)
    np.testing.assert_allclose(out["cpred"], out["ipred"], rtol=1e-6)


def test_noiseless_self_consistent_residual_is_g_dot_eta():
    # When the "observation" is exactly f(eta_hat) (no noise), the FOCE
    # residual y - E_FOCE(y) = f_hat - (f_hat - G@eta_hat) = G@eta_hat exactly
    # -- an algebraic identity independent of the whitening root. Combined
    # error model so both sigma_prop and sigma_add passed below are active in
    # `_residual_variance` (a plain "proportional" spec would silently ignore
    # sigma_add, which is exactly what this test's own recomputed expectation
    # must mirror).
    spec = _spec(IV_KEY, ["CL", "V"], error_model="combined")
    theta = {"CL": 5.0, "V": 50.0}
    eta_hat = np.array([0.25, -0.2])
    doses = [{"time": 0.0, "amt": 100.0}]
    obs_t = [0.5, 1.0, 2.0, 4.0, 8.0]
    from app.compute.nlme import _individual_params
    p_ind = _individual_params(spec, theta, eta_hat)
    f_hat = simulate(get_model(IV_KEY), p_ind, doses, obs_t, wt=70.0)["cp"]

    subjects_raw = [{"subject": "S1", "doses": doses, "obs_t": obs_t,
                     "obs_c": [float(v) for v in f_hat], "wt": 70.0}]
    subj = _prepare_subjects(subjects_raw)[0]
    omega2_vec = np.array([0.09, 0.04])

    out = _cwres_subject(spec, subj, theta, eta_hat, omega2_vec, 0.1, 0.5, interaction=True)
    assert out is not None
    resid = out["obs"] - out["cpred"]
    g = out["g"]
    np.testing.assert_allclose(resid, g @ eta_hat, rtol=1e-9, atol=1e-9)
    expected_cwres, _ = _whiten(resid, g, omega2_vec,
                                np.maximum((0.1 * out["ipred"]) ** 2 + 0.5 ** 2, 1e-10))
    np.testing.assert_allclose(out["cwres"], expected_cwres, rtol=1e-9)


# ── 3. Whitening: implementation-independent property + root choice ─────────

def test_whitening_produces_an_actual_whitening_transform():
    # Property every valid Cov^{-1/2} must satisfy, independent of which root
    # convention is used: (Cov^{-1/2}) @ Cov @ (Cov^{-1/2}).T == I.
    rng = np.random.default_rng(7)
    n_obs, n_eta = 5, 2
    g = rng.normal(size=(n_obs, n_eta))
    omega2_vec = np.array([0.09, 0.04])
    r_var = rng.uniform(0.5, 2.0, size=n_obs)
    resid = rng.normal(size=n_obs)

    _cwres, fallback = _whiten(resid, g, omega2_vec, r_var)
    assert not fallback

    cov = g @ np.diag(omega2_vec) @ g.T + np.diag(r_var)
    w, v = np.linalg.eigh(cov)
    inv_sqrt = v @ np.diag(w ** -0.5) @ v.T
    identity_check = inv_sqrt @ cov @ inv_sqrt.T
    np.testing.assert_allclose(identity_check, np.eye(n_obs), atol=1e-8)
    # And _whiten's own output must equal this direct reconstruction.
    np.testing.assert_allclose(_cwres, inv_sqrt @ resid, atol=1e-8)


def test_whitening_root_choice_matters_for_correlated_observations():
    # Demonstrates WHY the root choice is load-bearing (motivates using the
    # symmetric/eigendecomposition root, verified against real NONMEM output
    # in the module docstring): for a subject with >1 correlated observation,
    # a Cholesky root and the symmetric root give DIFFERENT whitened vectors,
    # even though both are valid square roots of the same covariance (a
    # naive mean~0/sd~1 pooled check cannot tell them apart -- see below).
    rng = np.random.default_rng(11)
    n_obs, n_eta = 6, 1
    g = rng.normal(size=(n_obs, n_eta)) + 0.5  # correlated (shared eta -> shared G column)
    omega2_vec = np.array([0.16])
    r_var = np.full(n_obs, 0.5)
    resid = rng.normal(size=n_obs)

    cwres_sym, _ = _whiten(resid, g, omega2_vec, r_var)

    cov = g @ np.diag(omega2_vec) @ g.T + np.diag(r_var)
    chol_l = np.linalg.cholesky(cov)
    cwres_chol = np.linalg.solve(chol_l, resid)

    assert not np.allclose(cwres_sym, cwres_chol)
    # Both are nonetheless legitimate whitening transforms of the SAME resid.
    for whitened, root_cov in ((cwres_sym, cov), (cwres_chol, cov)):
        np.testing.assert_allclose(float(np.dot(whitened, whitened)),
                                   float(resid @ np.linalg.solve(root_cov, resid)), rtol=1e-6)


def test_whitening_eigenvalue_floor_flags_near_singular_covariance():
    # Two proportional Jacobian columns -> Cov is near rank-deficient in the
    # eta-driven direction; the relative eigenvalue floor must engage.
    g = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])  # column 2 = 2 * column 1
    omega2_vec = np.array([1e-4, 1e-4])
    r_var = np.full(3, 1e-12)
    resid = np.array([0.1, -0.2, 0.05])
    cwres, fallback = _whiten(resid, g, omega2_vec, r_var)
    assert fallback
    assert np.all(np.isfinite(cwres))


# ── 4. Row-level guards (BLQ, numerical floor) ───────────────────────────────

def test_blq_rows_are_dropped_before_forming_g_and_cov():
    spec = _spec(IV_KEY, ["CL", "V"])
    subjects_raw = [{"subject": "S1", "doses": [{"time": 0.0, "amt": 100.0}],
                     "obs_t": [0.5, 1.0, 2.0, 4.0], "obs_c": [2.0, 1.5, 0.05, 0.02],
                     "wt": 70.0, "obs_blq": [False, False, True, True], "lloq": 0.1}]
    subj = _prepare_subjects(subjects_raw)[0]
    assert subj.blq.sum() == 2
    theta = {"CL": 5.0, "V": 50.0}
    eta_hat = np.array([0.1, -0.05])
    omega2_vec = np.array([0.09, 0.04])

    out = _cwres_subject(spec, subj, theta, eta_hat, omega2_vec, 0.1, 0.5, interaction=True)
    assert out is not None
    assert out["n_blq_dropped"] == 2
    assert out["time"].size == 2


def test_floored_prediction_rows_are_dropped():
    # A deep-trough observation time where the structural prediction floors at
    # _EPS: the row must be excluded, not produce an unbounded CWRES.
    spec = _spec(IV_KEY, ["CL", "V"])
    subjects_raw = [{"subject": "S1", "doses": [{"time": 0.0, "amt": 100.0}],
                     "obs_t": [0.5, 500.0], "obs_c": [2.0, 1e-13], "wt": 70.0}]
    subj = _prepare_subjects(subjects_raw)[0]
    theta = {"CL": 5.0, "V": 50.0}
    eta_hat = np.zeros(2)
    omega2_vec = np.array([0.09, 0.04])

    out = _cwres_subject(spec, subj, theta, eta_hat, omega2_vec, 0.1, 0.5, interaction=True)
    assert out is not None
    assert out["n_floored_dropped"] >= 1
    assert np.all(np.abs(out["cwres"]) < 50.0)


# ── 5. Interaction flag ──────────────────────────────────────────────────────

def test_interaction_flag_changes_only_the_variance_term():
    spec = _spec(IV_KEY, ["CL", "V"])
    subjects_raw = [{"subject": "S1", "doses": [{"time": 0.0, "amt": 100.0}],
                     "obs_t": [0.5, 1.0, 2.0, 4.0, 8.0], "obs_c": [2.0, 1.5, 1.0, 0.5, 0.2],
                     "wt": 70.0}]
    subj = _prepare_subjects(subjects_raw)[0]
    theta = {"CL": 5.0, "V": 50.0}
    eta_hat = np.array([0.3, -0.2])  # nonzero, so f(eta_hat) != f(eta=0)
    omega2_vec = np.array([0.09, 0.04])

    out_i = _cwres_subject(spec, subj, theta, eta_hat, omega2_vec, 0.2, 0.1, interaction=True)
    out_ni = _cwres_subject(spec, subj, theta, eta_hat, omega2_vec, 0.2, 0.1, interaction=False)
    assert out_i is not None and out_ni is not None
    np.testing.assert_allclose(out_i["g"], out_ni["g"])
    np.testing.assert_allclose(out_i["cpred"], out_ni["cpred"])
    np.testing.assert_allclose(out_i["ipred"], out_ni["ipred"])
    assert not np.allclose(out_i["cwres"], out_ni["cwres"])


# ── 6. posthoc_residuals: the public entry point ─────────────────────────────

def _fitted_subject(model, theta, eta, obs_t, wt=70.0, sid="S1"):
    doses = [{"time": 0.0, "amt": 100.0}]
    from app.compute.nlme import _individual_params
    spec = _PopSpec(model, ["CL", "V"], "proportional")
    p_ind = _individual_params(spec, theta, eta)
    cp = simulate(model, p_ind, doses, obs_t, wt=wt)["cp"]
    return {"subject": sid, "doses": doses, "obs_t": list(obs_t), "obs_c": [float(v) for v in cp], "wt": wt}


def test_posthoc_residuals_reuses_stored_etas_without_reoptimizing():
    model = get_model(IV_KEY)
    theta = {"CL": 5.0, "V": 50.0}
    obs_t = [0.5, 1.0, 2.0, 4.0, 8.0]
    eta_true = {"CL": 0.1, "V": -0.05}
    subj = _fitted_subject(model, theta, np.array([eta_true["CL"], eta_true["V"]]), obs_t, sid="S1")

    out = posthoc_residuals(IV_KEY, [subj], theta=theta,
                            omega2={"CL": 0.09, "V": 0.04}, sigma_prop=0.1, sigma_add=0.5,
                            iiv_params=["CL", "V"], etas={"S1": eta_true})
    assert out["summary"]["n_etas_reused"] == 1
    assert out["summary"]["n_etas_resolved"] == 0
    assert out["summary"]["n"] == len(obs_t)
    assert out["summary"]["cwres_mean"] is not None


def test_posthoc_residuals_resolves_missing_etas():
    model = get_model(IV_KEY)
    theta = {"CL": 5.0, "V": 50.0}
    obs_t = [0.5, 1.0, 2.0, 4.0, 8.0]
    subj = _fitted_subject(model, theta, np.array([0.05, -0.02]), obs_t, sid="S1")

    out = posthoc_residuals(IV_KEY, [subj], theta=theta,
                            omega2={"CL": 0.09, "V": 0.04}, sigma_prop=0.1, sigma_add=0.5,
                            iiv_params=["CL", "V"], etas=None)
    assert out["summary"]["n_etas_reused"] == 0
    assert out["summary"]["n_etas_resolved"] == 1
    assert out["summary"]["n"] == len(obs_t)


def test_posthoc_residuals_eta_join_survives_json_roundtrip():
    # Subject ids from `_build_subjects` (pandas groupby) are numpy scalars; a
    # persisted-and-reloaded NLME result has been through
    # json.dumps(..., default=str). Both must join correctly via `_sid`.
    model = get_model(IV_KEY)
    theta = {"CL": 5.0, "V": 50.0}
    obs_t = [0.5, 1.0, 2.0, 4.0, 8.0]
    eta_true = {"CL": 0.1, "V": -0.05}
    subj = _fitted_subject(model, theta, np.array([eta_true["CL"], eta_true["V"]]), obs_t,
                           sid=np.int64(7))

    etas_raw = {7: eta_true}  # as if freshly fitted (python int / np.int64 key)
    etas_roundtripped = json.loads(json.dumps(etas_raw, default=str))
    assert list(etas_roundtripped.keys()) == ["7"]  # confirms the round-trip actually stringifies

    out = posthoc_residuals(IV_KEY, [subj], theta=theta,
                            omega2={"CL": 0.09, "V": 0.04}, sigma_prop=0.1, sigma_add=0.5,
                            iiv_params=["CL", "V"], etas=etas_roundtripped)
    assert out["summary"]["n_etas_reused"] == 1
    assert out["summary"]["n_etas_resolved"] == 0


def test_posthoc_residuals_is_json_safe_and_deterministic():
    model = get_model(IV_KEY)
    theta = {"CL": 5.0, "V": 50.0}
    obs_t = [0.5, 1.0, 2.0, 4.0, 8.0]
    subjects = [_fitted_subject(model, theta, np.array([0.1, -0.05]), obs_t, sid="A"),
               _fitted_subject(model, theta, np.array([-0.2, 0.15]), obs_t, sid="B")]
    kwargs = dict(theta=theta, omega2={"CL": 0.09, "V": 0.04}, sigma_prop=0.1,
                 sigma_add=0.5, iiv_params=["CL", "V"])
    a = posthoc_residuals(IV_KEY, subjects, **kwargs)
    b = posthoc_residuals(IV_KEY, subjects, **kwargs)
    assert a == b
    json.dumps(a)


def test_posthoc_residuals_empty_subjects_returns_none_summary():
    out = posthoc_residuals(IV_KEY, [], theta={"CL": 5.0, "V": 50.0},
                            omega2={"CL": 0.09, "V": 0.04}, sigma_prop=0.1, sigma_add=0.5,
                            iiv_params=["CL", "V"])
    assert out["summary"]["n"] == 0
    assert out["summary"]["cwres_mean"] is None
    assert out["cwres"] == []


def test_sid_coerces_numpy_and_python_types_consistently():
    assert _sid(np.int64(5)) == _sid(5) == _sid("5") == "5"
