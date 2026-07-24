"""Tests for the Week-13 Biologics simulations: special-population (renal)
exposure vs a reference band, the representative reference population, and
per-subject steady-state exposures from EBEs.

Validated against analytic properties of the linear model with an EGFR-on-CL
covariate: lower eGFR -> lower CL -> higher AUCss, so renal-impaired strata sit
above the normal reference band, and dose reduction brings them back inside.
"""
import json

import numpy as np
import pandas as pd
import pytest

from app.compute.clinsim import (
    _renal_label,
    individual_exposures,
    reference_population,
    special_population_simulation,
)
from app.compute.pk_models import get_model
from app.compute.pk_simulate import simulate
from app.core.pharmstate import PharmState
from app.tools.base import ToolContext
from app.tools.builtins import default_registry
from app.tools.pkmodel_tools import run_individual_exposures, run_special_population

MODEL_KEY = "oral_1cmt"
THETA = {"CL": 5.0, "V": 50.0, "KA": 1.0}
OMEGA = {"CL": 30.0, "V": 20.0}
IIV = ["CL", "V"]
EGFR_EFF = [{"param": "CL", "covariate": "EGFR", "kind": "power",
             "center": 90.0, "coefficient": 0.7, "rse_pct": 20.0}]


def _renal_source(seed: int = 1):
    """Covariate rows spanning eGFR 20-149 (all renal categories present)."""
    cov = [{"EGFR": float(e), "WT": 70.0} for e in range(20, 150, 3)]
    return cov, [70.0] * len(cov)


# --- renal labels ----------------------------------------------------------

def test_renal_label_boundaries():
    assert _renal_label(20.0) == "Severe"      # <30
    assert _renal_label(45.0) == "Moderate"    # 30-60
    assert _renal_label(75.0) == "Mild"        # 60-90
    assert _renal_label(120.0) == "Normal"     # >=90
    assert _renal_label(30.0) == "Moderate"    # edge -> upper bin


# --- special-population simulation -----------------------------------------

def test_renal_strata_exposure_ordering():
    cov, wt = _renal_source()
    res = special_population_simulation(
        MODEL_KEY, theta=THETA, omega_cv_pct=OMEGA, iiv_params=IIV, cov_rows=cov,
        wt_rows=wt, stratify_by="EGFR", doses=[100, 200], tau=12, n_doses=14,
        covariate_effects=EGFR_EFF, reference_stratum="Normal", reference_dose=100,
        n_per_stratum=200)
    assert res["status"] == "ok" and res["kind"] == "renal"
    labels = [s["label"] for s in res["strata"]]
    assert labels == ["Severe", "Moderate", "Mild", "Normal"]   # ordered severe->normal
    # AUCss median rises as renal function falls (lower eGFR -> lower CL -> higher AUC).
    med = {s["label"]: [d for d in s["doses"] if d["dose"] == 100][0]["auc_tau"]["p50"]
           for s in res["strata"]}
    assert med["Severe"] > med["Moderate"] > med["Mild"] > med["Normal"]


def test_impaired_above_reference_band_and_dose_reduction():
    cov, wt = _renal_source()
    res = special_population_simulation(
        MODEL_KEY, theta=THETA, omega_cv_pct=OMEGA, iiv_params=IIV, cov_rows=cov,
        wt_rows=wt, stratify_by="EGFR", doses=[25, 50, 100, 200], tau=12, n_doses=14,
        covariate_effects=EGFR_EFF, reference_stratum="Normal", reference_dose=100,
        n_per_stratum=250)
    by = {s["label"]: s for s in res["strata"]}
    band = res["reference_band"]["auc_tau"]
    # Normal at the reference dose is within its own band by construction.
    normal_100 = [d for d in by["Normal"]["doses"] if d["dose"] == 100][0]["auc_tau"]
    assert normal_100["within_ref"]
    # Severe at the reference dose overshoots the band; a lower dose is recommended.
    severe_100 = [d for d in by["Severe"]["doses"] if d["dose"] == 100][0]["auc_tau"]
    assert severe_100["p50"] > band["hi"] and not severe_100["within_ref"]
    assert by["Severe"]["recommended_dose"] is None or by["Severe"]["recommended_dose"] < 100


def test_special_pop_deterministic():
    cov, wt = _renal_source()
    kw = dict(theta=THETA, omega_cv_pct=OMEGA, iiv_params=IIV, cov_rows=cov, wt_rows=wt,
              stratify_by="EGFR", doses=[100], tau=12, n_doses=14, covariate_effects=EGFR_EFF,
              n_per_stratum=120)
    a = special_population_simulation(MODEL_KEY, **kw)
    b = special_population_simulation(MODEL_KEY, **kw)
    ma = [s["doses"][0]["auc_tau"]["p50"] for s in a["strata"]]
    mb = [s["doses"][0]["auc_tau"]["p50"] for s in b["strata"]]
    assert ma == mb


def test_special_pop_degenerate():
    cov, wt = _renal_source()
    assert special_population_simulation(MODEL_KEY, theta=THETA, omega_cv_pct=OMEGA,
        iiv_params=IIV, cov_rows=cov, wt_rows=wt, stratify_by="EGFR", doses=[0],
        tau=12, n_doses=14)["status"] == "no_doses"
    assert special_population_simulation(MODEL_KEY, theta=THETA, omega_cv_pct=OMEGA,
        iiv_params=IIV, cov_rows=[], wt_rows=[], stratify_by="EGFR", doses=[100],
        tau=12, n_doses=14)["status"] == "no_covariates"
    assert special_population_simulation(MODEL_KEY, theta=THETA, omega_cv_pct=OMEGA,
        iiv_params=IIV, cov_rows=[{"WT": 70}], wt_rows=[70], stratify_by="NOPE",
        doses=[100], tau=12, n_doses=14)["status"] == "missing_covariate"
    with pytest.raises(ValueError):
        special_population_simulation(MODEL_KEY, theta=THETA, omega_cv_pct=OMEGA,
            iiv_params=IIV, cov_rows=cov, wt_rows=wt, stratify_by="EGFR", doses=[100],
            tau=12, n_doses=14, metrics=("bogus",))


# --- representative reference population ------------------------------------

def test_reference_population_spans_renal_categories():
    cov, wt = reference_population(n=3000, seed=1)
    assert len(cov) == 3000 and len(wt) == 3000
    cats = {_renal_label(r["EGFR"]) for r in cov}
    assert cats == {"Severe", "Moderate", "Mild", "Normal"}    # all four present
    for r in cov:
        assert 5.0 <= r["EGFR"] <= 150.0 and 40.0 <= r["WT"] <= 160.0
        assert r["SEX"] in (0, 1)
    # deterministic: same n + same seed -> identical draw (per-variable arrays
    # interleave RNG state by n, so cross-n prefixes are not comparable).
    cov2, _ = reference_population(n=3000, seed=1)
    assert [r["EGFR"] for r in cov] == [r["EGFR"] for r in cov2]
    cov3, _ = reference_population(n=3000, seed=2)
    assert [r["EGFR"] for r in cov] != [r["EGFR"] for r in cov3]   # seed matters


def test_reference_sex_is_float_so_categorical_effect_applies():
    # Regression: a fitted categorical covariate stores levels as str(float value)
    # ("1.0"), because the dataset pipeline casts covariates to float. If
    # reference_population emitted SEX as int, str(1)="1" would miss the "1.0"
    # level and silently drop the effect. Prove the SEX effect is actually applied.
    from app.compute.clinsim import _covariate_applier
    cov, _ = reference_population(n=50, seed=1)
    assert all(isinstance(r["SEX"], float) for r in cov)          # float, not int
    sex_eff = [{"param": "CL", "covariate": "SEX", "kind": "categorical",
                "levels": ["1.0"], "coefficient": {"1.0": -0.22}, "rse_pct": {"1.0": 15.0}}]
    apply = _covariate_applier(sex_eff)
    female = next(r for r in cov if r["SEX"] == 1.0)
    male = next(r for r in cov if r["SEX"] == 0.0)
    cl_f = apply({"CL": 5.0}, female)["CL"]
    cl_m = apply({"CL": 5.0}, male)["CL"]
    assert cl_m == pytest.approx(5.0)                             # reference level: no change
    assert cl_f == pytest.approx(5.0 * np.exp(-0.22))            # effect applied, NOT dropped
    assert cl_f < cl_m


# --- individual exposures --------------------------------------------------

def test_individual_exposures_and_renal_grouping():
    subs = [{"subject": str(i), "cov": {"EGFR": float(20 + i * 8)}, "wt": 70.0} for i in range(16)]
    etas = {str(i): {"CL": 0.05 * (i - 8)} for i in range(16)}
    res = individual_exposures(MODEL_KEY, theta=THETA, subjects=subs, etas=etas,
                               covariate_effects=EGFR_EFF, iiv_params=["CL"],
                               dose=100, tau=12, n_doses=14, group_key="EGFR")
    assert res["status"] == "ok" and len(res["subjects"]) == 16
    assert all(s["auc_ss"] is not None and s["cmax_ss"] is not None for s in res["subjects"])
    # renal covariate is binned into KDIGO categories (not one group per subject)
    groups = {g["group"] for g in res["groups"]}
    assert groups <= {"Severe", "Moderate", "Mild", "Normal"} and len(groups) >= 2
    assert individual_exposures(MODEL_KEY, theta=THETA, subjects=[], etas={},
                                dose=100, tau=12, n_doses=14)["status"] == "empty"


# --- tool layer ------------------------------------------------------------

def _dataset() -> pd.DataFrame:
    rows = []
    for sid in range(1, 17):
        egfr = 25.0 + sid * 7.0
        rows.append({"ID": sid, "TIME": 0.0, "DV": np.nan, "AMT": 100.0, "EGFR": egfr, "WT": 60.0 + sid})
        cp = simulate(get_model(MODEL_KEY), {"CL": 5.0, "V": 50.0, "KA": 1.0},
                      [{"time": 0.0, "amt": 100.0}], [0.5, 1, 2, 4, 8, 12], wt=60.0 + sid)["cp"]
        for t, c in zip([0.5, 1, 2, 4, 8, 12], cp):
            rows.append({"ID": sid, "TIME": float(t), "DV": float(c), "AMT": np.nan,
                         "EGFR": egfr, "WT": 60.0 + sid})
    return pd.DataFrame(rows)


def _nlme() -> dict:
    return {"status": "ok", "model_key": MODEL_KEY, "theta": {"CL": 5.0, "V": 50.0, "KA": 1.0},
            "omega_cv_pct": {"CL": 30.0, "V": 20.0}, "iiv_params": ["CL", "V"],
            "sigma": {"prop": 0.1, "add": 0.0}, "covariate_effects": EGFR_EFF,
            "individual": [{"subject": sid, "eta": {"CL": 0.05 * (sid - 8), "V": 0.0}}
                           for sid in range(1, 17)]}


def _pk_model_results() -> dict:
    return {"status": "ok", "mode": "fit", "model_key": MODEL_KEY,
            "individual_fits": [{"subject": s, "converged": True,
                                 "params": {"CL": 5.0, "V": 50.0, "KA": 1.0}} for s in range(1, 17)],
            "population": {"parameters": {
                "CL": {"typical_value": 5.0, "iiv_cv_pct": 30.0},
                "V": {"typical_value": 50.0, "iiv_cv_pct": 20.0},
                "KA": {"typical_value": 1.0, "iiv_cv_pct": 0.0}}}}


@pytest.fixture
def loaded():
    ctx = ToolContext(dataset_store={"d1": _dataset()})
    state = PharmState(dataset_id="d1", pk_model_results=_pk_model_results(), nlme_results=_nlme(),
                       dataset_metadata={"detected_roles":
                                         {"ID": "ID", "TIME": "TIME", "DV": "DV", "AMT": "AMT"}})
    return state, ctx


def test_tools_registered():
    assert default_registry().get("run_special_population").agent == "simulator"
    assert default_registry().get("run_individual_exposures").agent == "simulator"


def test_run_special_population_auto_detects_renal(loaded):
    state, ctx = loaded
    payload = run_special_population(state, ctx, {"doses": [50, 100, 200], "tau": 12,
                                                  "n_doses": 14, "reference_dose": 100,
                                                  "n_per_stratum": 120}).writes["special_pop_results"]
    assert payload["status"] == "ok" and payload["stratify_by"] == "EGFR"
    assert payload["covariate_in_model"] is True and payload["population_source"] == "dataset"
    json.dumps(payload)


def test_run_special_population_reference_source_gets_all_categories(loaded):
    state, ctx = loaded
    payload = run_special_population(state, ctx, {"source": "reference", "n_reference": 1200,
                                                  "doses": [50, 100], "tau": 12, "n_doses": 14,
                                                  "n_per_stratum": 120}).writes["special_pop_results"]
    assert payload["status"] == "ok" and payload["population_source"] == "reference"
    assert {s["label"] for s in payload["strata"]} == {"Severe", "Moderate", "Mild", "Normal"}
    json.dumps(payload)


def test_run_individual_exposures_needs_nlme():
    res = run_individual_exposures(PharmState(), ToolContext(), {})
    assert res.writes["individual_exposures"]["status"] == "needs_nlme"


def test_run_individual_exposures_ok(loaded):
    state, ctx = loaded
    payload = run_individual_exposures(state, ctx, {"dose": 100, "tau": 12,
                                                    "n_doses": 14}).writes["individual_exposures"]
    assert payload["status"] == "ok" and len(payload["subjects"]) == 16
    assert payload.get("groups")                          # renal-grouped summary present
    json.dumps(payload)
