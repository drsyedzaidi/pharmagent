"""MAP/empirical-Bayes forecasting + TDM dose optimization."""
from __future__ import annotations

import numpy as np

from app.compute.forecast import forecast
from app.compute.pk_models import get_model
from app.compute.pk_simulate import simulate

# fitted population: typical CL=5, V=50, KA=1; IIV 30/20%; proportional error 10%
POP = {
    "model_key": "oral_1cmt", "label": "1-cmt oral (linear)",
    "theta": {"CL": 5.0, "V": 50.0, "KA": 1.0},
    "omega_cv_pct": {"CL": 30.0, "V": 20.0},
    "sigma": {"prop": 0.1, "add": None},
    "iiv_params": ["CL", "V"], "error_model": "proportional",
    "covariate_effects": [],
}


def _levels_for(cl: float, v: float, dose: float = 100.0, tau: float = 24.0):
    """Clean measured levels in the 3rd dosing interval for a patient (CL, V)."""
    model = get_model("oral_1cmt")
    doses = [{"time": i * tau, "amt": dose} for i in range(3)]
    t = np.array([48.5, 60.0, 72.0])
    cp = simulate(model, {"CL": cl, "V": v, "KA": 1.0}, doses, t, wt=70.0)["cp"]
    return [{"time": float(tt), "conc": float(cc)} for tt, cc in zip(t, cp)]


def test_map_pulls_individual_toward_truth():
    """A fast-clearing patient's MAP CL lands between the population typical and
    the true value (Bayesian shrinkage), well above the typical."""
    r = forecast(POP, dose=100.0, tau=24.0, measured=_levels_for(9.0, 55.0), wt=70.0)
    assert r["status"] == "ok"
    cl_map = r["individual_params"]["CL"]
    assert 5.0 < cl_map < 9.5, cl_map           # moved up from typical toward 9
    assert abs(cl_map - 9.0) < abs(5.0 - 9.0)   # closer to truth than typical is


def test_no_levels_returns_population_typical():
    """With no measured levels the individual equals the population typical."""
    r = forecast(POP, dose=100.0, tau=24.0, measured=[])
    assert r["n_obs"] == 0
    assert r["individual_params"]["CL"] == r["typical_params"]["CL"]


def test_dose_optimization_hits_target():
    """The recommended dose reproduces the requested steady-state trough."""
    r = forecast(POP, dose=100.0, tau=24.0, measured=_levels_for(5.0, 50.0),
                 target=0.5, target_metric="cmin")
    rec = r["recommendation"]
    assert rec["recommended_dose"] is not None and rec["recommended_dose"] > 0
    assert abs(rec["predicted"]["cmin"] - 0.5) < 0.05    # bisection converged to target


def test_ss_metrics_present_and_ordered():
    r = forecast(POP, dose=100.0, tau=24.0, measured=_levels_for(5.0, 50.0))
    si = r["ss_individual"]
    assert {"cmin", "cmax", "cavg", "auc_tau"} <= set(si)
    assert si["cmax"] >= si["cavg"] >= si["cmin"]        # peak >= average >= trough
    assert len(r["forecast"]["times"]) == len(r["forecast"]["individual"]) > 0
