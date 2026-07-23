"""Tests for clinical trial simulation / PTA and the simulated exposure forest
(IU PopPK Week 12 "Simulation": Exercises 1-2 and forest-plots.R).

Validated against analytic properties of the linear 1-compartment model:
  * PTA (efficacy trough above a threshold) is monotone non-decreasing in dose.
  * Exposure is dose-proportional, so the recommended safe/efficacious dose and
    the metric percentiles scale with dose.
  * A renal (EGFR power) covariate produces relative AUC = (EGFR/center)^-beta at
    steady state, matched to a tight tolerance.
  * Every degenerate path returns a status, never raises; payloads are JSON-safe.
"""
import json
import math

import numpy as np
import pandas as pd
import pytest

from app.compute.clinsim import (
    clinical_trial_simulation,
    exposure_covariate_forest,
)
from app.compute.pk_models import get_model
from app.compute.pk_simulate import simulate
from app.core.pharmstate import PharmState
from app.tools.base import ToolContext
from app.tools.builtins import default_registry
from app.tools.pkmodel_tools import run_clinsim, run_exposure_forest

MODEL_KEY = "oral_1cmt"
THETA = {"CL": 5.0, "V": 50.0, "KA": 1.0}
OMEGA = {"CL": 30.0, "V": 20.0}
IIV = ["CL", "V"]


# --- CTS / PTA -------------------------------------------------------------

def test_pta_is_monotone_in_dose():
    res = clinical_trial_simulation(
        MODEL_KEY, theta=THETA, omega_cv_pct=OMEGA, iiv_params=IIV,
        doses=[50, 100, 200, 400, 800], tau=24, n_doses=7,
        metric="ctrough", threshold=2.0, direction="above",
        n_subjects=300)
    assert res["status"] == "ok" and res["with_iiv"]
    ptas = [d["pta"] for d in res["doses"]]
    assert all(b >= a - 1e-9 for a, b in zip(ptas, ptas[1:]))   # non-decreasing


def test_exposure_is_dose_proportional():
    res = clinical_trial_simulation(
        MODEL_KEY, theta=THETA, omega_cv_pct=OMEGA, iiv_params=IIV,
        doses=[100, 200, 400], tau=24, n_doses=7, metric="ctrough",
        n_subjects=200)
    med = {d["dose"]: d["metric_median"] for d in res["doses"]}
    assert med[200] / med[100] == pytest.approx(2.0, rel=0.05)
    assert med[400] / med[100] == pytest.approx(4.0, rel=0.05)


def test_recommendation_efficacy_lowest_safety_highest():
    # Efficacy: lowest dose reaching the target. Threshold low enough that all pass.
    eff = clinical_trial_simulation(
        MODEL_KEY, theta=THETA, omega_cv_pct=OMEGA, iiv_params=IIV,
        doses=[100, 200, 400], tau=24, n_doses=7, metric="ctrough",
        threshold=0.05, direction="above", target_fraction=0.8, n_subjects=200)
    assert eff["recommended_dose"] == 100
    # Safety: highest dose still below the Cmax cap.
    saf = clinical_trial_simulation(
        MODEL_KEY, theta=THETA, omega_cv_pct=OMEGA, iiv_params=IIV,
        doses=[100, 200, 400], tau=24, n_doses=7, metric="cmax",
        threshold=1e6, direction="below", target_fraction=0.9, n_subjects=200)
    assert saf["recommended_dose"] == 400


def test_pta_deterministic_and_no_threshold_path():
    kw = dict(theta=THETA, omega_cv_pct=OMEGA, iiv_params=IIV,
              doses=[100, 200], tau=24, n_doses=7, metric="ctrough", n_subjects=150)
    a = clinical_trial_simulation(MODEL_KEY, threshold=2.0, **kw)
    b = clinical_trial_simulation(MODEL_KEY, threshold=2.0, **kw)
    assert [d["pta"] for d in a["doses"]] == [d["pta"] for d in b["doses"]]
    none = clinical_trial_simulation(MODEL_KEY, threshold=None, **kw)
    assert none["status"] == "ok"
    assert all(d["pta"] is None for d in none["doses"])          # exposure only
    assert none["recommended_dose"] is None


def test_pta_no_iiv_is_deterministic_cohort():
    res = clinical_trial_simulation(
        MODEL_KEY, theta=THETA, omega_cv_pct={"CL": 0.0, "V": 0.0}, iiv_params=[],
        doses=[100, 200], tau=24, n_doses=7, metric="ctrough", threshold=2.0,
        n_subjects=50)
    assert res["status"] == "ok" and res["with_iiv"] is False
    # No IIV, no covariates -> every subject identical -> PTA is 0 or 1.
    assert all(d["pta"] in (0.0, 1.0) for d in res["doses"])


def test_cts_degenerate_paths():
    assert clinical_trial_simulation(MODEL_KEY, theta=THETA, omega_cv_pct=OMEGA,
                                     iiv_params=IIV, doses=[0, -5], tau=24,
                                     n_doses=1)["status"] == "no_doses"
    with pytest.raises(ValueError):
        clinical_trial_simulation(MODEL_KEY, theta=THETA, omega_cv_pct=OMEGA,
                                  iiv_params=IIV, doses=[100], tau=24, n_doses=1,
                                  metric="bogus")
    with pytest.raises(ValueError):
        clinical_trial_simulation(MODEL_KEY, theta=THETA, omega_cv_pct=OMEGA,
                                  iiv_params=IIV, doses=[100], tau=24, n_doses=1,
                                  direction="sideways")


# --- exposure covariate forest --------------------------------------------

_EGFR_EFF = [{"param": "CL", "covariate": "EGFR", "kind": "power",
             "center": 90.0, "coefficient": 0.7, "rse_pct": 20.0}]


def test_exposure_forest_recovers_renal_ratio():
    # AUC ∝ 1/CL, CL ∝ EGFR^0.7 -> relAUC(EGFR) = (EGFR/90)^-0.7.
    scen = [{"covariate": "EGFR", "is_weight": False,
             "levels": [{"label": "Severe", "value": 22.5},
                        {"label": "Mild", "value": 75.0}]}]
    res = exposure_covariate_forest(
        MODEL_KEY, theta=THETA, covariate_effects=_EGFR_EFF, scenarios=scen,
        reference_cov={"EGFR": 90.0}, dose=1e5, tau=24, n_doses=7, n_draws=300)
    assert res["status"] == "ok"
    by = {r["label"]: r["rel_auc"]["median"] for r in res["rows"]}
    assert by["Severe"] == pytest.approx((22.5 / 90.0) ** -0.7, rel=0.05)
    assert by["Mild"] == pytest.approx((75.0 / 90.0) ** -0.7, rel=0.05)
    # Both EGFR levels are below the reference (90), so both raise AUC, monotonically.
    assert by["Severe"] > by["Mild"] > 1.0


def test_exposure_forest_ci_widens_with_uncertainty():
    scen = [{"covariate": "EGFR", "is_weight": False,
             "levels": [{"label": "Severe", "value": 22.5}]}]
    tight = exposure_covariate_forest(
        MODEL_KEY, theta=THETA,
        covariate_effects=[{**_EGFR_EFF[0], "rse_pct": 5.0}], scenarios=scen,
        reference_cov={"EGFR": 90.0}, dose=1e5, tau=24, n_doses=7, n_draws=400)
    wide = exposure_covariate_forest(
        MODEL_KEY, theta=THETA,
        covariate_effects=[{**_EGFR_EFF[0], "rse_pct": 40.0}], scenarios=scen,
        reference_cov={"EGFR": 90.0}, dose=1e5, tau=24, n_doses=7, n_draws=400)
    tw = tight["rows"][0]["rel_auc"]
    ww = wide["rows"][0]["rel_auc"]
    assert (ww["hi"] - ww["lo"]) > (tw["hi"] - tw["lo"])          # more RSE -> wider CI


def test_exposure_forest_zero_rse_is_a_point():
    scen = [{"covariate": "EGFR", "is_weight": False,
             "levels": [{"label": "Severe", "value": 22.5}]}]
    res = exposure_covariate_forest(
        MODEL_KEY, theta=THETA,
        covariate_effects=[{**_EGFR_EFF[0], "rse_pct": 0.0}], scenarios=scen,
        reference_cov={"EGFR": 90.0}, dose=1e5, tau=24, n_doses=7, n_draws=50)
    row = res["rows"][0]["rel_auc"]
    assert row["lo"] == pytest.approx(row["hi"], rel=1e-6)         # no width


def test_exposure_forest_no_covariate_model():
    res = exposure_covariate_forest(
        MODEL_KEY, theta=THETA, covariate_effects=[], scenarios=[],
        reference_cov={}, dose=100, tau=24, n_doses=1)
    assert res["status"] == "no_covariate_model"


def test_exposure_forest_weight_scenario_direction():
    # Allometric CL ∝ WT^0.75 -> lighter patient has lower CL -> higher AUC.
    scen = [{"covariate": "WT", "is_weight": True,
             "levels": [{"label": "50 kg", "value": 50.0},
                        {"label": "90 kg", "value": 90.0}]}]
    res = exposure_covariate_forest(
        MODEL_KEY, theta=THETA, covariate_effects=_EGFR_EFF, scenarios=scen,
        reference_cov={"EGFR": 90.0}, dose=1e5, tau=24, n_doses=7, ref_wt=70.0, n_draws=50)
    by = {r["label"]: r["rel_auc"]["median"] for r in res["rows"]}
    assert by["50 kg"] > 1.0 > by["90 kg"]


# --- tool layer ------------------------------------------------------------

def _dataset() -> pd.DataFrame:
    rows = []
    for sid in range(1, 13):
        egfr = 30.0 + sid * 6.0
        wt = 60.0 + sid
        rows.append({"ID": sid, "TIME": 0.0, "DV": np.nan, "AMT": 100.0, "EGFR": egfr, "WT": wt})
        cp = simulate(get_model(MODEL_KEY), {"CL": 5.0, "V": 50.0, "KA": 1.0},
                      [{"time": 0.0, "amt": 100.0}], [0.5, 1, 2, 4, 8, 12], wt=wt)["cp"]
        for t, c in zip([0.5, 1, 2, 4, 8, 12], cp):
            rows.append({"ID": sid, "TIME": float(t), "DV": float(c), "AMT": np.nan,
                         "EGFR": egfr, "WT": wt})
    return pd.DataFrame(rows)


def _pk_model_results() -> dict:
    return {"status": "ok", "mode": "fit", "model_key": MODEL_KEY,
            "individual_fits": [{"subject": s, "converged": True,
                                 "params": {"CL": 5.0, "V": 50.0, "KA": 1.0}} for s in range(1, 13)],
            "population": {"parameters": {
                "CL": {"typical_value": 5.0, "iiv_cv_pct": 30.0},
                "V": {"typical_value": 50.0, "iiv_cv_pct": 20.0},
                "KA": {"typical_value": 1.0, "iiv_cv_pct": 0.0}}}}


def _nlme_with_cov() -> dict:
    return {"status": "ok", "model_key": MODEL_KEY, "theta": {"CL": 5.0, "V": 50.0, "KA": 1.0},
            "omega_cv_pct": {"CL": 30.0, "V": 20.0}, "iiv_params": ["CL", "V"],
            "sigma": {"prop": 0.1, "add": 0.0}, "covariate_effects": _EGFR_EFF}


@pytest.fixture
def loaded():
    ctx = ToolContext(dataset_store={"d1": _dataset()})
    state = PharmState(dataset_id="d1", pk_model_results=_pk_model_results(),
                       nlme_results=_nlme_with_cov(),
                       dataset_metadata={"detected_roles":
                                         {"ID": "ID", "TIME": "TIME", "DV": "DV", "AMT": "AMT"}})
    return state, ctx


def test_tools_registered():
    assert default_registry().get("run_clinsim").agent == "simulator"
    assert default_registry().get("run_exposure_forest").agent == "simulator"


def test_run_clinsim_no_fit_is_graceful():
    res = run_clinsim(PharmState(), ToolContext(), {})
    assert res.writes["clinsim_results"]["status"] == "no_fit"


def test_run_clinsim_uses_covariates_and_is_json_safe(loaded):
    state, ctx = loaded
    res = run_clinsim(state, ctx, {"doses": [50, 100, 200], "tau": 24, "n_doses": 7,
                                   "metric": "ctrough", "threshold": 1.0, "n_subjects": 150})
    cs = res.writes["clinsim_results"]
    assert cs["status"] == "ok" and cs["with_covariates"] and cs["with_iiv"]
    json.dumps(cs)


def test_run_exposure_forest_builds_scenarios(loaded):
    state, ctx = loaded
    res = run_exposure_forest(state, ctx, {"dose": 100, "tau": 24, "n_doses": 7, "n_draws": 100})
    ef = res.writes["exposure_forest_results"]
    assert ef["status"] == "ok"
    covs = {r["covariate"] for r in ef["rows"]}
    assert "EGFR" in covs and "WT" in covs           # covariate + allometric WT scenario
    json.dumps(ef)


def test_run_exposure_forest_no_covariate_model(loaded):
    state, ctx = loaded
    state.nlme_results = {"status": "ok", "model_key": MODEL_KEY, "theta": THETA,
                          "covariate_effects": []}
    res = run_exposure_forest(state, ctx, {})
    assert res.writes["exposure_forest_results"]["status"] == "no_covariate_model"


def test_run_exposure_forest_sources_scm_final(loaded):
    # An SCM covariate model (no plain NLME) must feed the exposure forest, same
    # single-provenance sourcing as run_covariate_forest.
    state, ctx = loaded
    state.nlme_results = None
    state.scm_results = {"status": "ok", "final": {
        "model_key": MODEL_KEY, "theta": {"CL": 5.0, "V": 50.0, "KA": 1.0},
        "covariate_effects": _EGFR_EFF}}
    res = run_exposure_forest(state, ctx, {"dose": 100, "tau": 24, "n_doses": 7, "n_draws": 60})
    ef = res.writes["exposure_forest_results"]
    assert ef["status"] == "ok"
    assert any(r["covariate"] == "EGFR" for r in ef["rows"])


def test_forest_ratio_matches_closed_form_math():
    # Guard the analytic identity the recovery test relies on.
    assert (22.5 / 90.0) ** -0.7 == pytest.approx(math.exp(-0.7 * math.log(22.5 / 90.0)))


def test_clinsim_request_rejects_nonpositive_target_fraction():
    # Adversarial-review finding: target_fraction=0 would trivially green-light
    # every dose (PTA >= 0). The request model must reject it (defense in depth
    # behind the frontend fallback).
    from pydantic import ValidationError

    from app.main import ClinsimRequest
    assert ClinsimRequest(target_fraction=0.9).target_fraction == 0.9
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValidationError):
            ClinsimRequest(target_fraction=bad)
