"""Tests for app.compute.nlme — true NLME estimation (FOCE-I and SAEM).

The test strategy is parameter recovery on a *seeded* simulated population: a
cohort is generated from the ``oral_1cmt`` model with known typical values
(theta), lognormal between-subject variability (Omega), and proportional
residual error (sigma). Each estimator is then asked to recover those truths
from data only, and the recovered values are checked against the truth within
pharmacometric tolerances.

Determinism is enforced both for data generation (a fixed RNG seed) and for
SAEM (same ``seed`` -> identical theta). FOCE-I is deterministic by construction.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from app.compute.nlme import (
    _build_cov_effects,
    _combined_sigma_mle,
    _PopSpec,
    _prepare_subjects,
    _saem_update_theta,
    focei_fit,
    population_fit,
    saem_fit,
)

# ── truth used to generate the synthetic population ──────────────────────────
MODEL_KEY = "oral_1cmt"
TRUE_THETA = {"CL": 5.0, "V": 50.0, "KA": 1.0}
TRUE_CV = {"CL": 0.30, "V": 0.20}          # %CV/100 for CL and V (lognormal IIV)
TRUE_SIGMA_PROP = 0.10                       # proportional residual error
DOSE_AMT = 100.0
OBS_TIMES = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 24.0])
N_SUBJECTS = 30
DATA_SEED = 12345


def _cv_to_omega2(cv: float) -> float:
    """Convert a lognormal %CV/100 to the variance omega2 = ln(1 + cv^2)."""
    return math.log(1.0 + cv ** 2)


def _make_population(seed: int = DATA_SEED, n: int = N_SUBJECTS) -> list[dict[str, Any]]:
    """Build a seeded ``oral_1cmt`` population with lognormal IIV + prop. error.

    For each subject: draw eta_CL, eta_V ~ N(0, omega2), realize individual
    CL_i, V_i, simulate the concentration profile, then add proportional
    residual noise y = f * (1 + sigma_prop * N(0,1)). KA is shared (no IIV).
    """
    from app.compute.pk_models import get_model
    from app.compute.pk_simulate import simulate

    rng = np.random.default_rng(seed)
    model = get_model(MODEL_KEY)
    omega2 = {p: _cv_to_omega2(cv) for p, cv in TRUE_CV.items()}
    sd = {p: math.sqrt(v) for p, v in omega2.items()}

    subjects: list[dict[str, Any]] = []
    for i in range(n):
        eta_cl = rng.normal(0.0, sd["CL"])
        eta_v = rng.normal(0.0, sd["V"])
        params = {
            "CL": TRUE_THETA["CL"] * math.exp(eta_cl),
            "V": TRUE_THETA["V"] * math.exp(eta_v),
            "KA": TRUE_THETA["KA"],
        }
        doses = [{"time": 0.0, "amt": DOSE_AMT}]
        f = simulate(model, params, doses, OBS_TIMES, wt=70.0)["cp"]
        noise = rng.normal(0.0, 1.0, size=f.size)
        obs_c = f * (1.0 + TRUE_SIGMA_PROP * noise)
        obs_c = np.maximum(obs_c, 1e-6)  # keep concentrations positive
        subjects.append({
            "subject": f"S{i + 1:02d}",
            "doses": doses,
            "obs_t": OBS_TIMES.copy(),
            "obs_c": obs_c,
            "wt": 70.0,
        })
    return subjects


# ── shared, expensive fixtures (one fit each, reused across assertions) ───────

@pytest.fixture(scope="module")
def population() -> list[dict[str, Any]]:
    return _make_population()


@pytest.fixture(scope="module")
def focei_result(population: list[dict[str, Any]]) -> dict[str, Any]:
    # Cap iterations to keep the suite within budget; warm-started inner solves
    # converge well within this. Uncertainty is exercised separately
    # (focei_unc fixture) so the recovery fits stay cheap.
    return population_fit(MODEL_KEY, population, method="focei", max_iter=30,
                          compute_uncertainty=False)


@pytest.fixture(scope="module")
def saem_result(population: list[dict[str, Any]]) -> dict[str, Any]:
    return population_fit(MODEL_KEY, population, method="saem", max_iter=80,
                          seed=20250614, compute_uncertainty=False)


@pytest.fixture(scope="module")
def unc_pop() -> list[dict[str, Any]]:
    """Smaller cohort for the (expensive) covariance pass — the full OFV Hessian
    is ~2*n_par^2 population passes, so a 12-subject cohort keeps it affordable
    while still rich enough for plausible RSE%s."""
    return _make_population(seed=DATA_SEED, n=12)


@pytest.fixture(scope="module")
def focei_unc(unc_pop: list[dict[str, Any]]) -> dict[str, Any]:
    """One FOCE-I fit WITH the asymptotic covariance pass, reused by all
    uncertainty assertions."""
    return population_fit(MODEL_KEY, unc_pop, method="focei", max_iter=25,
                          compute_uncertainty=True)


# ── result-shape contract ─────────────────────────────────────────────────────

_REQUIRED_KEYS = {
    "method", "model_key", "label", "iiv_params", "error_model", "theta",
    "theta_rse_pct", "omega_cv_pct", "omega_rse_pct", "sigma", "sigma_rse_pct",
    "covariate_effects", "ofv", "condition_number", "cov_note", "shrinkage_pct",
    "n_subjects", "n_obs", "n_blq", "converged", "individual", "iterations",
}


def test_result_shape_focei(focei_result: dict[str, Any]) -> None:
    r = focei_result
    assert _REQUIRED_KEYS.issubset(r.keys())
    assert r["method"] == "FOCE-I"
    assert r["model_key"] == MODEL_KEY
    assert r["label"] == "1-cmt oral (linear)"
    assert r["iiv_params"] == ["CL", "V"]
    assert r["error_model"] == "proportional"
    # theta is reported for every structural param (CL, V, KA).
    assert set(r["theta"]) == {"CL", "V", "KA"}
    assert set(r["omega_cv_pct"]) == {"CL", "V"}
    assert r["sigma"]["prop"] is not None and r["sigma"]["add"] is None
    assert r["n_subjects"] == N_SUBJECTS
    assert r["n_obs"] == N_SUBJECTS * OBS_TIMES.size


def test_individual_ebes_length_matches_subjects(focei_result: dict[str, Any]) -> None:
    indiv = focei_result["individual"]
    assert len(indiv) == focei_result["n_subjects"] == N_SUBJECTS
    rec = indiv[0]
    assert set(rec.keys()) == {"subject", "eta", "params"}
    assert set(rec["eta"]) == {"CL", "V"}
    assert set(rec["params"]) == {"CL", "V", "KA"}


# ── FOCE-I parameter recovery ─────────────────────────────────────────────────

def test_focei_recovers_theta(focei_result: dict[str, Any]) -> None:
    th = focei_result["theta"]
    assert th["CL"] == pytest.approx(TRUE_THETA["CL"], rel=0.20)
    assert th["V"] == pytest.approx(TRUE_THETA["V"], rel=0.20)
    assert th["KA"] == pytest.approx(TRUE_THETA["KA"], rel=0.30)


def test_focei_recovers_omega(focei_result: dict[str, Any]) -> None:
    cv = focei_result["omega_cv_pct"]
    # Omega %CV within 12 ABSOLUTE percentage points of truth (30% and 20%).
    assert cv["CL"] == pytest.approx(100.0 * TRUE_CV["CL"], abs=12.0)
    assert cv["V"] == pytest.approx(100.0 * TRUE_CV["V"], abs=12.0)


def test_focei_recovers_sigma(focei_result: dict[str, Any]) -> None:
    assert focei_result["sigma"]["prop"] == pytest.approx(TRUE_SIGMA_PROP, abs=0.06)


def test_focei_ofv_finite_and_converged(focei_result: dict[str, Any]) -> None:
    assert math.isfinite(focei_result["ofv"])
    assert focei_result["converged"] is True


def test_focei_shrinkage_reported(focei_result: dict[str, Any]) -> None:
    shr = focei_result["shrinkage_pct"]
    assert set(shr) == {"CL", "V"}
    # Rich sampling -> low shrinkage; loose bound just guards sane reporting.
    for p in ("CL", "V"):
        assert shr[p] < 60.0


# ── SAEM parameter recovery (slightly looser tolerances) ──────────────────────

def test_saem_recovers_theta(saem_result: dict[str, Any]) -> None:
    th = saem_result["theta"]
    assert th["CL"] == pytest.approx(TRUE_THETA["CL"], rel=0.25)
    assert th["V"] == pytest.approx(TRUE_THETA["V"], rel=0.25)
    assert th["KA"] == pytest.approx(TRUE_THETA["KA"], rel=0.35)


def test_saem_recovers_omega(saem_result: dict[str, Any]) -> None:
    cv = saem_result["omega_cv_pct"]
    assert cv["CL"] == pytest.approx(100.0 * TRUE_CV["CL"], abs=15.0)
    assert cv["V"] == pytest.approx(100.0 * TRUE_CV["V"], abs=15.0)


def test_saem_recovers_sigma(saem_result: dict[str, Any]) -> None:
    assert saem_result["sigma"]["prop"] == pytest.approx(TRUE_SIGMA_PROP, abs=0.08)


def test_saem_ofv_finite(saem_result: dict[str, Any]) -> None:
    assert math.isfinite(saem_result["ofv"])
    assert saem_result["method"] == "SAEM"


def test_saem_individual_ebes_length(saem_result: dict[str, Any]) -> None:
    assert len(saem_result["individual"]) == N_SUBJECTS


# ── determinism: same seed -> identical theta ─────────────────────────────────

def test_saem_deterministic_same_seed(population: list[dict[str, Any]]) -> None:
    # Determinism is independent of convergence, so a short run suffices and
    # keeps the suite fast.
    a = population_fit(MODEL_KEY, population, method="saem", max_iter=12, seed=777)
    b = population_fit(MODEL_KEY, population, method="saem", max_iter=12, seed=777)
    for p in ("CL", "V", "KA"):
        assert a["theta"][p] == b["theta"][p]


# ── cross-method agreement ────────────────────────────────────────────────────

def test_focei_and_saem_agree_on_cl(focei_result: dict[str, Any],
                                    saem_result: dict[str, Any]) -> None:
    cl_focei = focei_result["theta"]["CL"]
    cl_saem = saem_result["theta"]["CL"]
    # SAEM CL within 20% of FOCE-I CL.
    assert cl_saem == pytest.approx(cl_focei, rel=0.20)


# ── edge cases: sparse subject / tiny cohort handled gracefully ───────────────

def test_sparse_subject_is_handled() -> None:
    """A subject with too few points must not crash the fit (skipped or graceful).

    Spec: subjects below the usable-observation threshold (no doses or no usable
    concentrations) are dropped; the fit proceeds on the remainder. We build a
    small cohort, append a no-dose subject and an empty-observation subject, and
    confirm the run completes with only the well-formed subjects contributing.
    """
    base = _make_population(seed=42, n=4)
    augmented = base + [
        {  # no doses -> not usable, must be skipped
            "subject": "NODOSE",
            "doses": [],
            "obs_t": np.array([1.0, 2.0, 4.0]),
            "obs_c": np.array([1.2, 1.0, 0.6]),
            "wt": 70.0,
        },
        {  # no usable concentrations -> skipped
            "subject": "EMPTY",
            "doses": [{"time": 0.0, "amt": DOSE_AMT}],
            "obs_t": np.array([]),
            "obs_c": np.array([]),
            "wt": 70.0,
        },
    ]
    r = focei_fit(MODEL_KEY, augmented, iiv_params=["CL", "V"],
                  error_model="proportional", max_iter=2)
    assert math.isfinite(r["ofv"])
    ids = {rec["subject"] for rec in r["individual"]}
    # Only the 4 well-formed subjects contribute; degenerate ones are dropped.
    assert r["n_subjects"] == 4 == len(ids)
    assert "NODOSE" not in ids and "EMPTY" not in ids


def test_tiny_cohort_does_not_crash() -> None:
    """Fitting a 2-subject cohort completes without error."""
    pop = _make_population(seed=99, n=2)
    r = focei_fit(MODEL_KEY, pop, iiv_params=["CL", "V"],
                  error_model="proportional", max_iter=2)
    assert r["n_subjects"] == 2
    assert math.isfinite(r["ofv"])
    assert len(r["individual"]) == 2


def test_empty_cohort_returns_unconverged() -> None:
    """No usable subjects -> graceful, non-converged result (no exception)."""
    bad = [{"subject": "X", "doses": [], "obs_t": np.array([]),
            "obs_c": np.array([]), "wt": 70.0}]
    r = saem_fit(MODEL_KEY, bad, iiv_params=["CL", "V"],
                 error_model="proportional", max_iter=5)
    assert r["n_subjects"] == 0
    assert r["converged"] is False
    assert r["individual"] == []


# ── parameter uncertainty (RSE%, condition number) ───────────────────────────

def test_uncertainty_disabled_yields_empty_rse(focei_result: dict[str, Any]) -> None:
    """With compute_uncertainty=False the keys exist but RSEs are empty/None."""
    assert focei_result["theta_rse_pct"] == {}
    assert focei_result["omega_rse_pct"] == {}
    assert focei_result["sigma_rse_pct"] == {"prop": None, "add": None}
    assert focei_result["condition_number"] is None


def test_theta_rse_present_for_every_structural_param(focei_unc: dict[str, Any]) -> None:
    """A converged fit reports an RSE% for CL, V and KA."""
    rse = focei_unc["theta_rse_pct"]
    assert set(rse) == {"CL", "V", "KA"}
    for p, v in rse.items():
        assert isinstance(v, float) and math.isfinite(v) and v > 0.0, p


def test_theta_rse_magnitudes_are_plausible(focei_unc: dict[str, Any]) -> None:
    """For a rich 30-subject design the well-identified structural parameters
    (CL, V) should have small RSE% — sane precision, not noise."""
    rse = focei_unc["theta_rse_pct"]
    assert 0.0 < rse["CL"] < 30.0, rse["CL"]
    assert 0.0 < rse["V"] < 30.0, rse["V"]
    # KA is less informed but must still be a finite, sub-100% RSE here.
    assert 0.0 < rse["KA"] < 100.0, rse["KA"]


def test_omega_and_sigma_rse_reported(focei_unc: dict[str, Any]) -> None:
    """Variance (Omega) and residual-error (sigma) RSEs are emitted and finite."""
    orse = focei_unc["omega_rse_pct"]
    assert set(orse) == {"CL", "V"}
    for p, v in orse.items():
        assert isinstance(v, float) and math.isfinite(v) and v > 0.0, p
    srse = focei_unc["sigma_rse_pct"]
    assert srse["prop"] is not None and math.isfinite(srse["prop"]) and srse["prop"] > 0.0
    assert srse["add"] is None  # proportional-only error model


def test_condition_number_is_well_behaved(focei_unc: dict[str, Any]) -> None:
    """A well-identified 1-cmt model on rich data has a finite, modest condition
    number (not the >1000 over-parameterization red flag)."""
    cond = focei_unc["condition_number"]
    assert isinstance(cond, float) and math.isfinite(cond)
    assert cond >= 1.0
    assert cond < 1.0e3, cond


def test_uncertainty_is_deterministic() -> None:
    """FOCE-I covariance is reproducible for identical inputs (no RNG). Uses a
    tiny cohort — reproducibility is size-independent and this keeps it cheap."""
    pop = _make_population(seed=7, n=6)
    a = population_fit(MODEL_KEY, pop, method="focei", max_iter=10,
                       compute_uncertainty=True)["theta_rse_pct"]
    b = population_fit(MODEL_KEY, pop, method="focei", max_iter=10,
                       compute_uncertainty=True)["theta_rse_pct"]
    assert a == b and a != {}


# ── covariate model + stepwise covariate modeling (SCM) ──────────────────────

def _make_cov_population(seed: int, n: int, beta_wt: float = 0.75
                         ) -> list[dict[str, Any]]:
    """Population with a true WT-on-CL power effect (CL=5*(WT/70)^beta) plus an
    independent noise covariate AGE that influences nothing."""
    from app.compute.pk_models import get_model
    from app.compute.pk_simulate import simulate
    rng = np.random.default_rng(seed)
    model = get_model(MODEL_KEY)
    sd = {p: math.sqrt(_cv_to_omega2(c)) for p, c in {"CL": 0.25, "V": 0.20}.items()}
    subjects: list[dict[str, Any]] = []
    for i in range(n):
        wt = float(rng.uniform(45.0, 115.0))
        age = float(rng.uniform(20.0, 70.0))            # pure noise covariate
        params = {"CL": 5.0 * (wt / 70.0) ** beta_wt * math.exp(rng.normal(0, sd["CL"])),
                  "V": 50.0 * math.exp(rng.normal(0, sd["V"])), "KA": 1.0}
        doses = [{"time": 0.0, "amt": DOSE_AMT}]
        f = simulate(model, params, doses, OBS_TIMES, wt=70.0)["cp"]
        obs_c = np.maximum(f * (1.0 + 0.08 * rng.normal(0, 1, f.size)), 1e-6)
        subjects.append({"subject": f"C{i + 1:02d}", "doses": doses,
                         "obs_t": OBS_TIMES.copy(), "obs_c": obs_c, "wt": 70.0,
                         "cov": {"WT": wt, "AGE": age}})
    return subjects


def test_no_covariate_model_yields_empty_effects(focei_result: dict[str, Any]) -> None:
    """A fit without a covariate model reports an empty covariate_effects list."""
    assert focei_result["covariate_effects"] == []


@pytest.fixture(scope="module")
def cov_fit() -> dict[str, Any]:
    # Well-powered design (n=40, wide WT range) so the WT-on-CL effect is
    # genuinely identifiable and its SE is cleanly estimable.
    pop = _make_cov_population(seed=2024, n=40, beta_wt=0.75)
    return population_fit(
        MODEL_KEY, pop, method="focei", max_iter=30, compute_uncertainty=True,
        covariate_model=[{"param": "CL", "covariate": "WT",
                          "kind": "power", "center": 70.0}])


def test_covariate_recovers_known_exponent(cov_fit: dict[str, Any]) -> None:
    """The estimated WT-on-CL power exponent recovers the true 0.75."""
    effects = cov_fit["covariate_effects"]
    assert len(effects) == 1
    eff = effects[0]
    assert eff["param"] == "CL" and eff["covariate"] == "WT" and eff["kind"] == "power"
    assert isinstance(eff["coefficient"], float)
    assert abs(eff["coefficient"] - 0.75) < 0.4, eff["coefficient"]
    # structural CL recovers ~5 once the covariate effect is accounted for.
    assert abs(cov_fit["theta"]["CL"] - 5.0) < 1.5, cov_fit["theta"]["CL"]


def test_covariate_coefficient_rse_reported(cov_fit: dict[str, Any]) -> None:
    """The covariate coefficient gets a finite, positive RSE%."""
    rse = cov_fit["covariate_effects"][0]["rse_pct"]
    assert isinstance(rse, float) and math.isfinite(rse) and rse > 0.0, rse


_SCM_CANDIDATES = [
    {"param": "CL", "covariate": "WT", "kind": "power"},
    {"param": "CL", "covariate": "AGE", "kind": "power"},
]


def test_scm_selects_true_covariate_and_rejects_noise() -> None:
    """SCM adds the real WT-on-CL effect and never adds the noise AGE effect.
    Serial (parallel=False) keeps the test deterministic and process-free."""
    from app.compute.nlme import scm
    pop = _make_cov_population(seed=2024, n=20, beta_wt=0.75)
    res = scm(MODEL_KEY, pop, candidates=_SCM_CANDIDATES, iiv_params=["CL", "V"],
              error_model="proportional", max_iter=15, parallel=False)
    assert res["status"] == "ok"
    keys = {f"{s['param']}~{s['covariate']}" for s in res["selected"]}
    assert "CL~WT" in keys
    assert "CL~AGE" not in keys
    # selecting WT lowers the OFV relative to the base model.
    assert res["final_ofv"] < res["base_ofv"]
    added = [s for s in res["steps"] if s["decision"] == "added"]
    assert any(s["effect"] == "CL~WT" for s in added)


def test_scm_parallel_matches_serial_selection() -> None:
    """The ProcessPool path is deterministic (FOCE-I has no RNG) and must select
    the same covariate as the serial path. Small cohort keeps it cheap."""
    from app.compute.nlme import scm
    pop = _make_cov_population(seed=2024, n=14, beta_wt=0.75)
    kw = dict(candidates=_SCM_CANDIDATES, iiv_params=["CL", "V"],
              error_model="proportional", max_iter=12)
    serial = scm(MODEL_KEY, pop, parallel=False, **kw)
    par = scm(MODEL_KEY, pop, parallel=True, **kw)
    sel = lambda r: {f"{s['param']}~{s['covariate']}" for s in r["selected"]}
    assert sel(serial) == sel(par)
    # OFVs from the two paths agree (deterministic estimator).
    assert abs((serial["final_ofv"] or 0) - (par["final_ofv"] or 0)) < 1e-3


# ── BLQ / M3 (below-quantification-limit handling) ───────────────────────────

def test_default_path_reports_no_blq(focei_result):
    """A fit without an LLOQ reports n_blq == 0 (default = drop, byte-identical)."""
    assert focei_result["n_blq"] == 0


def _censored_population(seed: int, n: int, lloq: float):
    """Population with terminal samples driven below an LLOQ; BLQ rows flagged
    (obs_blq) and carry the LLOQ as DV, per the NONMEM M3 data convention."""
    from app.compute.pk_models import get_model
    from app.compute.pk_simulate import simulate
    rng = np.random.default_rng(seed)
    model = get_model(MODEL_KEY)
    sd = {p: math.sqrt(_cv_to_omega2(c)) for p, c in TRUE_CV.items()}
    times = np.array([0.25, 0.5, 1, 2, 4, 6, 8, 12, 18, 24, 36, 48])
    subs = []
    for i in range(n):
        cl = TRUE_THETA["CL"] * math.exp(rng.normal(0, sd["CL"]))
        v = TRUE_THETA["V"] * math.exp(rng.normal(0, sd["V"]))
        f = simulate(model, {"CL": cl, "V": v, "KA": 1.0},
                     [{"time": 0.0, "amt": DOSE_AMT}], times, wt=70.0)["cp"]
        c = f * (1.0 + TRUE_SIGMA_PROP * rng.normal(0, 1, f.size))
        blq = c < lloq
        c = np.where(blq, lloq, np.maximum(c, 1e-6))   # BLQ rows carry the LLOQ
        subs.append({"subject": f"B{i}", "doses": [{"time": 0.0, "amt": DOSE_AMT}],
                     "obs_t": times.copy(), "obs_c": c, "lloq": lloq,
                     "obs_blq": blq.tolist()})
    return subs


@pytest.fixture(scope="module")
def m3_fit():
    return population_fit(MODEL_KEY, _censored_population(7, 24, lloq=0.3),
                          method="focei", max_iter=30, compute_uncertainty=False)


def test_m3_counts_blq_records(m3_fit):
    assert m3_fit["n_blq"] > 0          # BLQ records kept + counted, not dropped


def test_m3_recovers_parameters_with_censoring(m3_fit):
    """With ~25% of samples below the LLOQ, the M3 censored likelihood still
    recovers the structural parameters within pharmacometric tolerance."""
    th = m3_fit["theta"]
    assert abs(th["CL"] - 5.0) / 5.0 < 0.20, th["CL"]
    assert abs(th["V"] - 50.0) / 50.0 < 0.25, th["V"]
    assert m3_fit["converged"] is True


# ── combined residual error: the two variance components must be separated ────
#
# Regression guard. The SAEM M-step used to estimate each component as if it
# alone explained the whole residual:
#
#     sigma_prop^2 <- mean((y/f - 1)^2)        sigma_add^2 <- mean((y - f)^2)
#
# Under the combined model Var = sigma_add^2 + (sigma_prop*f)^2 those two
# formulas are each the MLE only when that component is the *sole* error source;
# applied together they double-count the residual. On data spanning a wide
# concentration range, mean((y-f)^2) is dominated by the large *absolute*
# residuals of the high-concentration samples — whose scatter is really
# proportional — so sigma_add inflates past most of the observed concentrations.
# On a real 120-subject/1943-obs oral 2-cmt dataset this drove SAEM to
# sigma_add = 247 (FOCE-I on the same data: 4.64; NONMEM 7.5.0: 3.71) and cost
# ~3100 OFV units. The components must be estimated *jointly*.

TRUE_COMB_ADD = 0.5
TRUE_COMB_PROP = 0.15
COMB_DOSE = 5000.0
COMB_TIMES = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0,
                       18.0, 24.0, 36.0, 48.0, 72.0])


def _combined_residuals(seed: int = 99, n: int = 4000
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Residuals from a known combined error model over a 3000-fold prediction
    range (0.05 -> 150) — the regime that forces the components apart."""
    rng = np.random.default_rng(seed)
    f = np.exp(np.linspace(math.log(0.05), math.log(150.0), n))
    sd = np.sqrt(TRUE_COMB_ADD ** 2 + (TRUE_COMB_PROP * f) ** 2)
    return rng.normal(0.0, 1.0, f.size) * sd, f


def test_combined_sigma_mle_recovers_both_components() -> None:
    """The joint solve separates additive from proportional scatter."""
    resid, f = _combined_residuals()
    var_prop, var_add = _combined_sigma_mle(
        resid, f, TRUE_COMB_PROP ** 2, TRUE_COMB_ADD ** 2)
    assert math.sqrt(var_add) == pytest.approx(TRUE_COMB_ADD, abs=0.10)
    assert math.sqrt(var_prop) == pytest.approx(TRUE_COMB_PROP, abs=0.03)


def test_independent_component_estimates_are_biased() -> None:
    """Pins the defect itself: estimating each component alone (the pre-fix
    M-step) inflates *both* many-fold on the same residuals the joint solve
    handles correctly."""
    resid, f = _combined_residuals()
    naive_add = math.sqrt(float(np.mean(resid ** 2)))
    naive_prop = math.sqrt(float(np.mean((resid / f) ** 2)))
    assert naive_add > 5.0 * TRUE_COMB_ADD
    assert naive_prop > 5.0 * TRUE_COMB_PROP

    var_prop, var_add = _combined_sigma_mle(
        resid, f, TRUE_COMB_PROP ** 2, TRUE_COMB_ADD ** 2)
    assert math.sqrt(var_add) < 0.25 * naive_add
    assert math.sqrt(var_prop) < 0.25 * naive_prop


def test_combined_sigma_mle_is_warm_start_invariant() -> None:
    """The M-step warm-starts from the running estimates, so the optimum must
    not depend on where it starts — otherwise Robbins-Monro would drift with
    its own history instead of converging on the likelihood."""
    resid, f = _combined_residuals()
    starts = [(1e-8, 1e-8), (10.0, 100.0), (1.0, 0.0025)]   # (var_prop0, var_add0)
    out = [_combined_sigma_mle(resid, f, vp, va) for vp, va in starts]
    for var_prop, var_add in out[1:]:
        assert var_prop == pytest.approx(out[0][0], rel=1e-3)
        assert var_add == pytest.approx(out[0][1], rel=1e-3)


def _make_combined_population(seed: int = 2024, n: int = 24) -> list[dict[str, Any]]:
    """Seeded cohort with *combined* residual error and a wide concentration
    range (peak ~100, 72 h tail well under 1)."""
    from app.compute.pk_models import get_model
    from app.compute.pk_simulate import simulate

    rng = np.random.default_rng(seed)
    model = get_model(MODEL_KEY)
    sd = {p: math.sqrt(_cv_to_omega2(cv)) for p, cv in TRUE_CV.items()}
    subjects: list[dict[str, Any]] = []
    for i in range(n):
        params = {
            "CL": TRUE_THETA["CL"] * math.exp(rng.normal(0.0, sd["CL"])),
            "V": TRUE_THETA["V"] * math.exp(rng.normal(0.0, sd["V"])),
            "KA": TRUE_THETA["KA"],
        }
        doses = [{"time": 0.0, "amt": COMB_DOSE}]
        f = simulate(model, params, doses, COMB_TIMES, wt=70.0)["cp"]
        noise_sd = np.sqrt(TRUE_COMB_ADD ** 2 + (TRUE_COMB_PROP * f) ** 2)
        obs_c = f + rng.normal(0.0, 1.0, size=f.size) * noise_sd
        subjects.append({
            "subject": f"C{i + 1:02d}",
            "doses": doses,
            "obs_t": COMB_TIMES.copy(),
            "obs_c": np.maximum(obs_c, 1e-4),
            "wt": 70.0,
        })
    return subjects


@pytest.fixture(scope="module")
def combined_pop() -> list[dict[str, Any]]:
    return _make_combined_population()


@pytest.fixture(scope="module")
def saem_combined(combined_pop: list[dict[str, Any]]) -> dict[str, Any]:
    return population_fit(MODEL_KEY, combined_pop, method="saem",
                          iiv_params=["CL", "V"], error_model="combined",
                          max_iter=100, seed=20250614, compute_uncertainty=False)


def test_saem_combined_recovers_both_sigmas(saem_combined: dict[str, Any]) -> None:
    sigma = saem_combined["sigma"]
    assert sigma["prop"] == pytest.approx(TRUE_COMB_PROP, abs=0.07)
    assert sigma["add"] == pytest.approx(TRUE_COMB_ADD, abs=0.40)


def test_saem_combined_additive_sigma_is_not_degenerate(
        saem_combined: dict[str, Any],
        combined_pop: list[dict[str, Any]]) -> None:
    """The headline symptom of the defect: an additive SD on the order of — or
    above — the observed concentrations."""
    sigma_add = saem_combined["sigma"]["add"]
    max_obs = max(float(np.max(s["obs_c"])) for s in combined_pop)
    assert 0.05 < sigma_add < 3.0, sigma_add
    assert sigma_add < 0.05 * max_obs, (sigma_add, max_obs)


def test_saem_combined_keeps_structural_params_and_shrinkage_sane(
        saem_combined: dict[str, Any]) -> None:
    """A degenerate sigma_add flattens the individual objective (near-constant,
    huge residual variance), which over-shrinks the etas and drags the typical
    values with it — so these travel with the sigma fix."""
    th = saem_combined["theta"]
    assert th["CL"] == pytest.approx(TRUE_THETA["CL"], rel=0.25)
    assert th["V"] == pytest.approx(TRUE_THETA["V"], rel=0.25)
    shr = saem_combined["shrinkage_pct"]
    for p in ("CL", "V"):
        assert abs(shr[p]) < 40.0, (p, shr[p])


# ── combined error: the theta M-step must weight on the LIKELIHOOD ────────────
#
# Second regression guard, same defect class as the sigma one. The structural
# M-step (`_saem_update_theta`) weights each residual by 1/Var. Under the
# combined model Var = sigma_add^2 + (sigma_prop*f)^2 depends on the prediction f
# and hence on the parameters being fit. Recomputing the weight from every trial
# f — as the code used to — silently drops the log|Var| term of the likelihood
# and pays the optimizer to inflate predictions (bigger f -> bigger Var ->
# smaller weighted residual), biasing the typical values (CL low). The fix
# freezes the weights at the entry prediction (one-step IRLS/GLS), whose fixed
# point solves the unbiased score sum g*(y-f)/Var = 0. Verified independently:
# at sigma_prop=0.30 the old trial-weight step recovers CL ~= 4.7 (-6%), the
# frozen-weight step ~= 5.1 (unbiased).

THETA_MSTEP_PROP = 0.30       # large enough that the log|Var| omission bites


def _combined_at_truth(seed: int, n: int = 60) -> list[dict[str, Any]]:
    """Cohort simulated at the TRUE typical values (eta = 0) with combined error
    over a wide concentration range — isolates the structural M-step's bias."""
    from app.compute.pk_models import get_model
    from app.compute.pk_simulate import simulate

    rng = np.random.default_rng(seed)
    model = get_model(MODEL_KEY)
    doses = [{"time": 0.0, "amt": COMB_DOSE}]
    subjects: list[dict[str, Any]] = []
    for i in range(n):
        f = simulate(model, TRUE_THETA, doses, COMB_TIMES, wt=70.0)["cp"]
        noise_sd = np.sqrt(TRUE_COMB_ADD ** 2 + (THETA_MSTEP_PROP * f) ** 2)
        obs_c = f + rng.normal(0.0, 1.0, size=f.size) * noise_sd
        subjects.append({
            "subject": f"T{i + 1:02d}", "doses": doses,
            "obs_t": COMB_TIMES.copy(),
            "obs_c": np.maximum(obs_c, 1e-4), "wt": 70.0,
        })
    return subjects


def test_combined_theta_mstep_is_unbiased_from_truth() -> None:
    """One `_saem_update_theta` step started AT the truth must not systematically
    pull CL below it. The old (trial-weight) M-step drove CL to ~-6%; the
    frozen-weight step keeps it centered. Averaged over seeds to beat noise."""
    from app.compute.pk_models import get_model

    cls: list[float] = []
    for seed in (1, 7, 42):
        prepared = _prepare_subjects(_combined_at_truth(seed))
        spec = _PopSpec(get_model(MODEL_KEY), ["CL", "V"], "combined",
                        _build_cov_effects(None, prepared))
        etas = [np.zeros(spec.n_omega) for _ in prepared]
        new_theta, _ = _saem_update_theta(
            spec, prepared, dict(TRUE_THETA), np.zeros(spec.n_cov),
            etas, THETA_MSTEP_PROP, TRUE_COMB_ADD)
        cls.append(new_theta["CL"])
        # Per-seed: never biased low the way the trial-weight step was (~4.7).
        assert new_theta["CL"] > 4.85, (seed, new_theta["CL"])
    mean_cl = float(np.mean(cls))
    assert TRUE_THETA["CL"] * 0.99 < mean_cl < TRUE_THETA["CL"] * 1.06, (mean_cl, cls)


# ── SAEM-seeded FOCE-I ───────────────────────────────────────────────────────
# Plain FOCE-I minimizes its outer problem with Powell, a LOCAL derivative-free
# method: on a rough/multimodal surface it converges to whichever basin the cold
# data-derived start falls in, and raising max_iter only searches that same
# basin harder. Seeding it with a short SAEM run (whose stochastic E-step
# explores rather than descends) supplies a basin that FOCE-I then sharpens.
# These tests pin the mechanism; the basin-rescue itself is validated offline
# against the IU PopPK Week-8 benchmark (too expensive for the suite).

@pytest.fixture(scope="module")
def focei_saem_result(population: list[dict[str, Any]]) -> dict[str, Any]:
    return population_fit(MODEL_KEY, population, method="focei_saem", max_iter=30,
                          seed=20250614, compute_uncertainty=False)


def test_focei_saem_reports_seeded_provenance(focei_saem_result: dict[str, Any]) -> None:
    r = focei_saem_result
    assert _REQUIRED_KEYS.issubset(r.keys())
    # Relabelled so a reader can never mistake a seeded fit for a cold one.
    assert r["method"] == "FOCE-I (SAEM-seeded)"
    seeded = r["seeded_by"]
    assert seeded["method"] == "SAEM"
    # Seed length is min(max_iter, _SAEM_SEED_ITER); max_iter=30 binds here.
    assert seeded["iterations"] == 30
    assert isinstance(seeded["ofv"], float)


def test_focei_saem_recovers_truth(focei_saem_result: dict[str, Any]) -> None:
    """Seeding must not degrade the easy case: same recovery as plain FOCE-I."""
    theta = focei_saem_result["theta"]
    assert TRUE_THETA["CL"] * 0.80 < theta["CL"] < TRUE_THETA["CL"] * 1.20
    assert TRUE_THETA["V"] * 0.80 < theta["V"] < TRUE_THETA["V"] * 1.20


def test_focei_saem_is_deterministic(population: list[dict[str, Any]]) -> None:
    """The SAEM stage is the only stochastic part, so a fixed seed must give
    bit-identical estimates -- otherwise the audit trail is not reproducible."""
    kw = dict(method="focei_saem", max_iter=20, seed=7, compute_uncertainty=False)
    a = population_fit(MODEL_KEY, population, **kw)
    b = population_fit(MODEL_KEY, population, **kw)
    assert a["theta"] == b["theta"]
    assert a["ofv"] == b["ofv"]


def test_focei_saem_seed_length_is_capped(population: list[dict[str, Any]]) -> None:
    """A large max_iter must not make the throwaway seed run unboundedly long:
    it is capped at _SAEM_SEED_ITER because the seed only has to find the basin."""
    from app.compute.nlme import _SAEM_SEED_ITER

    r = population_fit(MODEL_KEY, population, method="focei_saem",
                       max_iter=_SAEM_SEED_ITER + 500, seed=3,
                       compute_uncertainty=False)
    assert r["seeded_by"]["iterations"] == _SAEM_SEED_ITER


def test_focei_saem_degrades_gracefully_without_a_usable_seed() -> None:
    """No usable data -> no warm start. The fit must still return the standard
    contract and must NOT claim a seed that never happened."""
    r = population_fit(MODEL_KEY, [], method="focei_saem", max_iter=5,
                       compute_uncertainty=False)
    assert _REQUIRED_KEYS.issubset(r.keys())
    assert r["seeded_by"] is None
    assert r["method"] == "FOCE-I"      # honest: this was a cold start


def test_plain_focei_is_unchanged_by_the_seeding_feature(
        focei_result: dict[str, Any]) -> None:
    """Guard: method='focei' must stay a cold start with no seed provenance."""
    assert focei_result["method"] == "FOCE-I"
    assert focei_result.get("seeded_by") is None


# ── method="auto": escalating FOCE-I with OFV arbitration ────────────────────
# Neither cold FOCE-I nor a single SAEM seed is reliable on a multimodal
# surface, and neither failure announces itself (both report converged=True).
# What IS reliable is the objective: every candidate is a converged FOCE-I fit
# of the same model on the same data, so OFVs are comparable and the minimum
# wins. `auto` probes with two independent starts and only pays for a
# multi-start search when they disagree.

@pytest.fixture(scope="module")
def auto_result(population: list[dict[str, Any]]) -> dict[str, Any]:
    return population_fit(MODEL_KEY, population, method="auto", max_iter=200,
                          seed=1, compute_uncertainty=False)


def test_auto_skips_escalation_when_starts_agree() -> None:
    """The fast path: when two independent starts land on the same optimum,
    `auto` stops at 2 candidates instead of paying for a multi-start search.

    Uses a deliberately small cohort. Measured cold-vs-seeded OFV gap by cohort
    size (max_iter=200): n=8 -> 0.09, n=10 -> 0.34, n=12 -> 2.0, n=16 -> 11.5
    against a tolerance of 1.0. Agreement is only reachable on small cohorts
    because this FOCE-I really is start-dependent at realistic sizes -- which is
    the very reason `auto` exists. See
    test_auto_escalates_on_a_realistic_cohort for the other side of that."""
    small = _make_population(n=8)
    r = population_fit(MODEL_KEY, small, method="auto", max_iter=200, seed=1,
                       compute_uncertainty=False)
    a = r["auto"]
    assert a["escalated"] is False
    assert a["n_candidates"] == 2
    assert a["reason"] == "two independent starts converged and agreed"
    assert _REQUIRED_KEYS.issubset(r.keys())


def test_auto_escalates_on_a_realistic_cohort(auto_result: dict[str, Any]) -> None:
    """On the 30-subject cohort the two starts disagree, so `auto` escalates.

    Both starts report converged=True after 0-1 outer iterations yet land ~4 OFV
    apart: Powell's convergence test fires early enough that the stopping point
    depends on where it began. The escalation is not wasted work -- it buys a
    genuinely lower OFV (asserted in test_auto_never_worse_than_plain_focei)."""
    a = auto_result["auto"]
    assert a["escalated"] is True
    assert a["reason"] == "starts disagreed on OFV"
    assert a["n_candidates"] > 2


def test_auto_selects_the_minimum_ofv_candidate(auto_result: dict[str, Any]) -> None:
    """The returned fit must BE the best candidate, not merely near it."""
    a = auto_result["auto"]
    ofvs = {k: v for k, v in a["candidate_ofv"].items() if v is not None}
    assert ofvs, "no finite candidate"
    best_name = min(ofvs, key=lambda k: ofvs[k])
    assert a["winner"] == best_name
    assert auto_result["ofv"] == pytest.approx(ofvs[best_name])
    assert auto_result["method"] == f"FOCE-I (auto: {best_name})"


def test_auto_never_worse_than_plain_focei(
        auto_result: dict[str, Any], population: list[dict[str, Any]]) -> None:
    """The guarantee `auto` actually makes: it evaluates cold FOCE-I as one of
    its candidates and returns the OFV minimum, so it can never come back worse
    than plain FOCE-I on the same data."""
    cold_ofv = auto_result["auto"]["candidate_ofv"]["cold"]
    assert cold_ofv is not None
    assert auto_result["ofv"] <= cold_ofv + 1e-9


def test_auto_recovers_truth(auto_result: dict[str, Any]) -> None:
    theta = auto_result["theta"]
    assert TRUE_THETA["CL"] * 0.80 < theta["CL"] < TRUE_THETA["CL"] * 1.20
    assert TRUE_THETA["V"] * 0.80 < theta["V"] < TRUE_THETA["V"] * 1.20


def test_auto_is_deterministic(population: list[dict[str, Any]]) -> None:
    """Cold FOCE-I is deterministic and every seeded start derives its seed from
    `seed`, so the whole escalation is reproducible -- required for the audit
    trail."""
    kw = dict(method="auto", max_iter=200, seed=5, compute_uncertainty=False)
    a = population_fit(MODEL_KEY, population, **kw)
    b = population_fit(MODEL_KEY, population, **kw)
    assert a["theta"] == b["theta"]
    assert a["ofv"] == b["ofv"]
    assert a["auto"]["candidate_ofv"] == b["auto"]["candidate_ofv"]


def test_auto_escalates_when_a_start_fails_to_converge(
        population: list[dict[str, Any]]) -> None:
    """A start stopped by the iteration cap has not identified any optimum. Its
    OFV gap would otherwise read as multimodality, so `auto` treats
    non-convergence as its own escalation trigger."""
    r = population_fit(MODEL_KEY, population, method="auto", max_iter=2, seed=2,
                       compute_uncertainty=False)
    a = r["auto"]
    assert a["escalated"] is True
    assert a["n_candidates"] > 2


def test_auto_handles_degenerate_input() -> None:
    r = population_fit(MODEL_KEY, [], method="auto", max_iter=5,
                       compute_uncertainty=False)
    assert _REQUIRED_KEYS.issubset(r.keys())
    assert r["auto"]["n_candidates"] >= 2
