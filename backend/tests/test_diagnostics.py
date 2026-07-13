"""Tests for app.compute.diagnostics — IWRES and simulation-based PDE/NPDE.

Validated against known analytic / distributional properties:
  * Self-consistency: when individual == typical == data-generating params,
    IPRED reproduces observed exactly -> IWRES ~ 0 (max |iwres| < 1e-6).
  * NPDE calibration: observations drawn from the SAME typical+IIV population
    that the predictive distribution is built from are ~N(0,1).
  * Determinism: a fixed seed reproduces the npde arrays bit-for-bit, and the
    outlier percentage stays within [0, 100].
"""
import numpy as np
import pytest

from app.compute.diagnostics import _cv_pct_to_sd, fit_residuals, npde
from app.compute.pk_models import get_model
from app.compute.pk_simulate import simulate

MODEL_KEY = "oral_1cmt"
TRUE_S1 = {"CL": 4.0, "V": 40.0, "KA": 1.2}
TRUE_S2 = {"CL": 7.0, "V": 55.0, "KA": 0.8}
OBS_T = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0]
DOSES = [{"time": 0.0, "amt": 100.0}]


def _make_subject(sid: str, params: dict, wt: float, obs_t=OBS_T) -> dict:
    """Subject whose observed concentrations are the model's own predictions."""
    model = get_model(MODEL_KEY)
    cp = simulate(model, params, DOSES, obs_t, wt=wt)["cp"]
    return {
        "subject": sid,
        "doses": list(DOSES),
        "obs_t": list(obs_t),
        "obs_c": [float(v) for v in cp],
        "wt": wt,
    }


# --- fit_residuals: self-consistency ---------------------------------------

def test_fit_residuals_identity_gives_zero_iwres():
    subjects = [
        _make_subject("S1", TRUE_S1, wt=70.0),
        _make_subject("S2", TRUE_S2, wt=82.0),
    ]
    indiv = {"S1": TRUE_S1, "S2": TRUE_S2}
    out = fit_residuals(MODEL_KEY, subjects, indiv, TRUE_S1)

    assert out["summary"]["n"] == 2 * len(OBS_T)
    for key in ("time", "obs", "ipred", "pred", "iwres", "iwres_std"):
        assert len(out[key]) == out["summary"]["n"]

    iwres = np.asarray(out["iwres"], dtype=float)
    assert np.max(np.abs(iwres)) < 1e-6
    # Mean ~ 0 and (near) zero spread -> standardized IWRES guarded to ~0.
    assert out["summary"]["iwres_mean"] == pytest.approx(0.0, abs=1e-6)


def test_fit_residuals_pred_uses_typical_not_individual():
    # S2 individual params differ from typical (S1): PRED column diverges from
    # observed even though IPRED reproduces it.
    subjects = [_make_subject("S2", TRUE_S2, wt=70.0)]
    out = fit_residuals(MODEL_KEY, subjects, {"S2": TRUE_S2}, TRUE_S1)
    assert out["summary"]["n"] == len(OBS_T)
    np.testing.assert_allclose(out["ipred"], out["obs"], atol=1e-6)
    diffs = np.abs(np.asarray(out["pred"]) - np.asarray(out["obs"]))
    assert diffs.max() > 1e-3


def test_fit_residuals_skips_subjects_without_individual_params():
    subjects = [
        _make_subject("S1", TRUE_S1, wt=70.0),
        _make_subject("S2", TRUE_S2, wt=70.0),
    ]
    out = fit_residuals(MODEL_KEY, subjects, {"S1": TRUE_S1}, TRUE_S1)
    assert out["summary"]["n"] == len(OBS_T)


def test_fit_residuals_empty_returns_none_summary():
    out = fit_residuals(MODEL_KEY, [], {}, TRUE_S1)
    assert out["summary"] == {"n": 0, "iwres_mean": None, "iwres_sd": None}
    assert out["iwres"] == []


def test_fit_residuals_standardized_has_zero_mean_unit_sd():
    # Perturb individual params so IWRES has real spread; standardized IWRES
    # must then have (population) mean 0 and sd 1.
    subjects = [_make_subject("S1", TRUE_S1, wt=70.0)]
    wrong = {"CL": 5.0, "V": 45.0, "KA": 1.0}
    out = fit_residuals(MODEL_KEY, subjects, {"S1": wrong}, TRUE_S1)
    std = np.asarray(out["iwres_std"], dtype=float)
    assert np.mean(std) == pytest.approx(0.0, abs=1e-6)
    assert np.std(std) == pytest.approx(1.0, abs=1e-6)


# --- npde: calibration -----------------------------------------------------

def test_npde_is_well_calibrated_against_its_own_population():
    # One design; generate each pseudo-subject's observation as ONE draw from
    # the SAME typical + IIV population the predictive distribution is built on.
    # The resulting PDEs should be ~N(0,1).
    typical = TRUE_S1
    iiv = {"CL": 30.0, "V": 25.0, "KA": 20.0}
    obs_t = [1.0, 4.0, 8.0]
    model = get_model(MODEL_KEY)
    sds = {p: _cv_pct_to_sd(iiv.get(p)) for p in model.params}

    gen = np.random.default_rng(12345)
    n_subjects = 200
    subjects = []
    for k in range(n_subjects):
        params_k = {}
        for p in model.params:
            eta = gen.normal(0.0, sds[p]) if sds[p] > 0 else 0.0
            params_k[p] = float(typical[p]) * float(np.exp(eta))
        cp = simulate(model, params_k, DOSES, obs_t, wt=70.0)["cp"]
        subjects.append({
            "subject": f"P{k}",
            "doses": list(DOSES),
            "obs_t": list(obs_t),
            "obs_c": [float(v) for v in cp],
            "wt": 70.0,
        })

    out = npde(MODEL_KEY, subjects, typical, iiv, n_sim=500, seed=777)
    npde_arr = np.asarray(out["npde"], dtype=float)

    assert out["summary"]["n"] == n_subjects * len(obs_t)
    assert np.all(np.isfinite(npde_arr))
    assert abs(out["summary"]["mean"]) < 0.15
    assert 0.8 < out["summary"]["sd"] < 1.25


def test_npde_values_within_clip_bounds():
    # PDEs are bounded by norm.ppf of the clip interval [1/(2n), 1-1/(2n)].
    n_sim = 500
    subjects = [_make_subject("S1", TRUE_S1, wt=70.0)]
    out = npde(MODEL_KEY, subjects, TRUE_S1, {"CL": 30.0, "V": 25.0},
               n_sim=n_sim, seed=42)
    from scipy.stats import norm
    bound = norm.ppf(1.0 - 1.0 / (2.0 * n_sim))
    npde_arr = np.asarray(out["npde"], dtype=float)
    assert np.all(np.isfinite(npde_arr))
    assert np.all(np.abs(npde_arr) <= bound + 1e-9)


def test_npde_is_deterministic_for_fixed_seed():
    subjects = [
        _make_subject("S1", TRUE_S1, wt=70.0),
        _make_subject("S2", TRUE_S2, wt=80.0),
    ]
    iiv = {"CL": 30.0, "V": 25.0, "KA": 15.0}
    a = npde(MODEL_KEY, subjects, TRUE_S1, iiv, n_sim=500, seed=2025)
    b = npde(MODEL_KEY, subjects, TRUE_S1, iiv, n_sim=500, seed=2025)
    assert a == b
    assert 0.0 <= a["summary"]["pct_outside_1_96"] <= 100.0


def test_npde_different_seed_differs():
    subjects = [_make_subject("S1", TRUE_S1, wt=70.0)]
    iiv = {"CL": 40.0, "V": 30.0, "KA": 25.0}
    a = npde(MODEL_KEY, subjects, TRUE_S1, iiv, n_sim=500, seed=1)
    b = npde(MODEL_KEY, subjects, TRUE_S1, iiv, n_sim=500, seed=2)
    assert a["npde"] != b["npde"]


def test_npde_empty_returns_none_summary():
    out = npde(MODEL_KEY, [], TRUE_S1, {"CL": 30.0})
    assert out["summary"] == {"n": 0, "mean": None, "sd": None,
                              "pct_outside_1_96": None}
    assert out["npde"] == []
