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
    assert out["summary"] == {"n": 0, "iwres_mean": None, "iwres_sd": None, "n_tad_null": 0}
    assert out["iwres"] == []
    assert out["tad"] == []


def test_fit_residuals_standardized_has_zero_mean_unit_sd():
    # Perturb individual params so IWRES has real spread; standardized IWRES
    # must then have (population) mean 0 and sd 1.
    subjects = [_make_subject("S1", TRUE_S1, wt=70.0)]
    wrong = {"CL": 5.0, "V": 45.0, "KA": 1.0}
    out = fit_residuals(MODEL_KEY, subjects, {"S1": wrong}, TRUE_S1)
    std = np.asarray(out["iwres_std"], dtype=float)
    assert np.mean(std) == pytest.approx(0.0, abs=1e-6)
    assert np.std(std) == pytest.approx(1.0, abs=1e-6)


def test_fit_residuals_reports_tad_per_observation():
    # A single dose at t=0 (DOSES fixture): TAD == TIME for every observation.
    subjects = [_make_subject("S1", TRUE_S1, wt=70.0)]
    out = fit_residuals(MODEL_KEY, subjects, {"S1": TRUE_S1}, TRUE_S1)
    assert out["tad"] == out["time"]
    assert out["summary"]["n_tad_null"] == 0


def test_fit_residuals_drops_blq_flagged_observations():
    # A subject carrying an obs_blq flag: the flagged rows are censored and must
    # never enter the residuals (a below-quantitation value would otherwise be
    # scored as a quantified observation). Two of six rows flagged -> n == 4.
    subj = _make_subject("S1", TRUE_S1, wt=70.0)
    subj["obs_blq"] = [True, True, False, False, False, False]
    out = fit_residuals(MODEL_KEY, [subj], {"S1": TRUE_S1}, TRUE_S1)
    assert out["summary"]["n"] == len(OBS_T) - 2
    # The retained times are exactly the non-flagged ones.
    assert out["time"] == [float(t) for t in OBS_T[2:]]


# --- npde: sigma is required -------------------------------------------

def test_npde_requires_sigma():
    # sigma_prop/sigma_add are keyword-only with NO default: a caller that
    # forgets them gets a loud TypeError, not a silently-narrow predictive
    # cloud (the exact defect this function used to have).
    subjects = [_make_subject("S1", TRUE_S1, wt=70.0)]
    with pytest.raises(TypeError):
        npde(MODEL_KEY, subjects, TRUE_S1, {"CL": 30.0}, n_sim=50, seed=1)


# --- npde: calibration -------------------------------------------------

def _predictive_draw(model, typical, sds, sigma_prop, sigma_add, obs_t, gen):
    """One observation drawn from the FULL predictive distribution
    y = f(theta_k) + eps, theta_k a between-subject draw and eps residual
    error — i.e. exactly the distribution npde() is supposed to reproduce.
    Used only to GENERATE test data; must not share code with npde() itself.
    """
    params_k = {}
    for p in model.params:
        eta = gen.normal(0.0, sds[p]) if sds[p] > 0 else 0.0
        params_k[p] = float(typical[p]) * float(np.exp(eta))
    cp = np.asarray(simulate(model, params_k, DOSES, obs_t, wt=70.0)["cp"], dtype=float)
    sd_eps = np.sqrt(sigma_add ** 2 + (sigma_prop * cp) ** 2)
    return cp + sd_eps * gen.normal(0.0, 1.0, cp.size)


def test_npde_is_well_calibrated_against_full_predictive_distribution():
    # Generate each pseudo-subject's observation as ONE draw from the SAME
    # FULL predictive distribution npde() builds: a between-subject parameter
    # draw PLUS a residual-error draw. The generator must include residual
    # error — otherwise the "observed" cloud is narrower than the predictive
    # distribution under test and the test would pass even with a missing
    # residual-error term (unable to catch the defect it exists to catch).
    typical = TRUE_S1
    iiv = {"CL": 30.0, "V": 25.0, "KA": 20.0}
    sigma_prop, sigma_add = 0.15, 0.02
    obs_t = [1.0, 4.0, 8.0]
    model = get_model(MODEL_KEY)
    sds = {p: _cv_pct_to_sd(iiv.get(p)) for p in model.params}

    gen = np.random.default_rng(12345)
    n_subjects = 200
    subjects = []
    for k in range(n_subjects):
        obs = _predictive_draw(model, typical, sds, sigma_prop, sigma_add, obs_t, gen)
        subjects.append({
            "subject": f"P{k}", "doses": list(DOSES), "obs_t": list(obs_t),
            "obs_c": [float(v) for v in obs], "wt": 70.0,
        })

    out = npde(MODEL_KEY, subjects, typical, iiv,
               sigma_prop=sigma_prop, sigma_add=sigma_add, n_sim=500, seed=777)
    npde_arr = np.asarray(out["npde"], dtype=float)

    assert out["metric"] == "npd"
    assert out["summary"]["n"] == n_subjects * len(obs_t)
    assert np.all(np.isfinite(npde_arr))
    # Loose by design (xWRES-family statistics have no exact known marginal
    # distribution under a correctly specified nonlinear model) — this is a
    # sanity check, never tighten into a normality gate.
    assert abs(out["summary"]["mean"]) < 0.15
    assert 0.8 < out["summary"]["sd"] < 1.25
    assert abs(out["summary"]["pct_outside_1_96"] - 5.0) < 4.0


def test_npde_omitting_residual_error_is_relatively_overdispersed():
    # The "teeth" test: reproduces the historical defect by calling npde()
    # with sigma_prop=sigma_add=0.0 on data generated WITH residual error, and
    # checks the corrected call (real sigma) is calibrated while the
    # zero-sigma call is not — self-calibrating (relative to the corrected
    # run), not a hardcoded constant that can drift with n_sim/n_subjects.
    typical = TRUE_S1
    iiv = {"CL": 30.0, "V": 25.0, "KA": 20.0}
    sigma_prop, sigma_add = 0.15, 0.02
    obs_t = [1.0, 4.0, 8.0]
    model = get_model(MODEL_KEY)
    sds = {p: _cv_pct_to_sd(iiv.get(p)) for p in model.params}

    gen = np.random.default_rng(999)
    subjects = []
    for k in range(200):
        obs = _predictive_draw(model, typical, sds, sigma_prop, sigma_add, obs_t, gen)
        subjects.append({
            "subject": f"P{k}", "doses": list(DOSES), "obs_t": list(obs_t),
            "obs_c": [float(v) for v in obs], "wt": 70.0,
        })

    corrected = npde(MODEL_KEY, subjects, typical, iiv,
                     sigma_prop=sigma_prop, sigma_add=sigma_add, n_sim=500, seed=1)
    zero_sigma = npde(MODEL_KEY, subjects, typical, iiv,
                      sigma_prop=0.0, sigma_add=0.0, n_sim=500, seed=1)

    sd_corrected = corrected["summary"]["sd"]
    sd_zero = zero_sigma["summary"]["sd"]
    assert sd_zero > sd_corrected
    assert abs(sd_zero - 1.0) > 3.0 * abs(sd_corrected - 1.0)


def test_npde_combined_error_calibrates_against_independent_generator():
    # Combined additive+proportional error, generated from a construction that
    # shares NO code with npde()'s own formula: y = f + sigma_add*eps1 +
    # sigma_prop*f*eps2 (sum of two independent normals). This is equal in
    # distribution to N(f, sigma_add^2 + (sigma_prop*f)^2) but written
    # differently, so a transposed or double-squared term in the
    # implementation would NOT cancel against a copy of itself in the test.
    typical = {"CL": 4.0, "V": 40.0, "KA": 1.2}
    sigma_prop, sigma_add = 0.12, 0.05
    obs_t = [1.0, 12.0]  # two strata: keeps n_i == 1 per stratum for KS validity
    model = get_model(MODEL_KEY)

    gen = np.random.default_rng(4242)
    subjects = []
    for k in range(300):
        cp = np.asarray(simulate(model, typical, DOSES, obs_t, wt=70.0)["cp"], dtype=float)
        obs = cp + sigma_add * gen.normal(0.0, 1.0, cp.size) + sigma_prop * cp * gen.normal(0.0, 1.0, cp.size)
        subjects.append({
            "subject": f"P{k}", "doses": list(DOSES), "obs_t": list(obs_t),
            "obs_c": [float(v) for v in obs], "wt": 70.0,
        })

    out = npde(MODEL_KEY, subjects, typical, {},
               sigma_prop=sigma_prop, sigma_add=sigma_add, n_sim=2000, seed=13)
    from scipy.stats import kstest
    npde_arr = np.asarray(out["npde"], dtype=float)
    for j, t in enumerate(obs_t):
        stratum = npde_arr[j::len(obs_t)]
        stat = kstest(stratum, "norm")
        assert stat.pvalue > 0.001, f"stratum t={t}: KS p={stat.pvalue}"


def test_npde_clip_fraction_decreases_with_n_sim():
    # The fraction of npd values sitting exactly at the clip bound
    # norm.ppf(1 - 1/(2*n_sim)) should shrink as n_sim grows, for a correctly
    # specified (well-calibrated) model — a direct probe on the clipping
    # mechanism rather than an indirect moment check.
    from scipy.stats import norm as _norm
    typical = TRUE_S1
    iiv = {"CL": 30.0, "V": 25.0, "KA": 20.0}
    obs_t = [4.0]
    model = get_model(MODEL_KEY)
    sds = {p: _cv_pct_to_sd(iiv.get(p)) for p in model.params}
    gen = np.random.default_rng(77)
    subjects = []
    for k in range(300):
        obs = _predictive_draw(model, typical, sds, 0.15, 0.02, obs_t, gen)
        subjects.append({
            "subject": f"P{k}", "doses": list(DOSES), "obs_t": list(obs_t),
            "obs_c": [float(v) for v in obs], "wt": 70.0,
        })

    fracs = []
    for n_sim in (200, 2000):
        out = npde(MODEL_KEY, subjects, typical, iiv,
                   sigma_prop=0.15, sigma_add=0.02, n_sim=n_sim, seed=5)
        bound = _norm.ppf(1.0 - 1.0 / (2.0 * n_sim))
        arr = np.asarray(out["npde"], dtype=float)
        fracs.append(np.mean(np.abs(arr) >= bound - 1e-9))
    assert fracs[1] <= fracs[0]


def test_npde_values_within_clip_bounds():
    # npd values are bounded by norm.ppf of the clip interval [1/(2n), 1-1/(2n)].
    n_sim = 500
    subjects = [_make_subject("S1", TRUE_S1, wt=70.0)]
    out = npde(MODEL_KEY, subjects, TRUE_S1, {"CL": 30.0, "V": 25.0},
               sigma_prop=0.1, sigma_add=0.5, n_sim=n_sim, seed=42)
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
    a = npde(MODEL_KEY, subjects, TRUE_S1, iiv,
             sigma_prop=0.1, sigma_add=0.5, n_sim=500, seed=2025)
    b = npde(MODEL_KEY, subjects, TRUE_S1, iiv,
             sigma_prop=0.1, sigma_add=0.5, n_sim=500, seed=2025)
    assert a == b
    assert 0.0 <= a["summary"]["pct_outside_1_96"] <= 100.0


def test_npde_different_seed_differs():
    subjects = [_make_subject("S1", TRUE_S1, wt=70.0)]
    iiv = {"CL": 40.0, "V": 30.0, "KA": 25.0}
    a = npde(MODEL_KEY, subjects, TRUE_S1, iiv,
             sigma_prop=0.1, sigma_add=0.5, n_sim=500, seed=1)
    b = npde(MODEL_KEY, subjects, TRUE_S1, iiv,
             sigma_prop=0.1, sigma_add=0.5, n_sim=500, seed=2)
    assert a["npde"] != b["npde"]


def test_npde_empty_returns_none_summary():
    out = npde(MODEL_KEY, [], TRUE_S1, {"CL": 30.0}, sigma_prop=0.1, sigma_add=0.5)
    assert out["metric"] == "npd"
    assert out["summary"]["n"] == 0
    assert out["summary"]["mean"] is None
    assert out["summary"]["sd"] is None
    assert out["summary"]["pct_outside_1_96"] is None
    assert out["summary"]["sigma_prop"] == 0.1
    assert out["summary"]["sigma_add"] == 0.5
    assert out["npde"] == []


def test_npde_reports_tad_per_observation():
    subjects = [_make_subject("S1", TRUE_S1, wt=70.0)]
    out = npde(MODEL_KEY, subjects, TRUE_S1, {"CL": 30.0, "V": 25.0},
               sigma_prop=0.1, sigma_add=0.5, n_sim=100, seed=1)
    assert out["tad"] == out["time"]  # single dose at t=0 -> TAD == TIME
    assert out["summary"]["n_tad_null"] == 0


def test_npde_drops_blq_flagged_observations():
    subj = _make_subject("S1", TRUE_S1, wt=70.0)
    subj["obs_blq"] = [True, False, False, False, False, False]
    out = npde(MODEL_KEY, [subj], TRUE_S1, {"CL": 30.0, "V": 25.0},
               sigma_prop=0.1, sigma_add=0.5, n_sim=200, seed=3)
    assert out["summary"]["n"] == len(OBS_T) - 1
