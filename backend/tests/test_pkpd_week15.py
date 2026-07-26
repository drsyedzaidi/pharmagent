"""Tests for the Week-15 PK/PD suite:

* the transit / cell-lifespan (Friberg-style transduction) PD model,
* informative theta priors + MAP estimation (FOCE-I + Gaussian prior penalty),
* Bayesian-borrowing diagnostics (prior-predictive check + prior-vs-posterior
  shrinkage).

Validated against analytic properties: a MAP fit with an informative prior
pulls a sparse-data estimate toward the prior mean and can never widen a
parameter beyond its prior (shrinkage in [0,1]); the default (no prior) FOCE-I
path is byte-for-byte unchanged.
"""
import json
import math

import numpy as np
import pandas as pd
import pytest

from app.compute.bayes import prior_posterior_diagnostic, prior_predictive_check
from app.compute.nlme import _build_theta_prior, population_fit
from app.compute.pk_fit import fit_subject_pkpd
from app.compute.pk_models import PKPD_KEYS, get_model
from app.compute.pk_simulate import simulate, simulate_timecourse
from app.core.pharmstate import PharmState
from app.tools.base import ToolContext
from app.tools.builtins import default_registry
from app.tools.pkmodel_tools import run_nlme, run_prior_check

MK = "oral_1cmt"


def _sparse_peds(n=14, cl_true=7.0, v_true=55.0, seed=4):
    """A sparse pediatric-like cohort (CL above the adult prior of 5)."""
    m = get_model(MK)
    rng = np.random.default_rng(seed)
    subs = []
    for i in range(n):
        cl = cl_true * math.exp(rng.normal(0, 0.25))
        v = v_true * math.exp(rng.normal(0, 0.25))
        sim = simulate_timecourse(m, {"CL": cl, "V": v, "KA": 1.0}, dose=100, tau=24,
                                  n_doses=1, tmax=24, n_points=200)
        ts = np.array(sim["times"])
        tp = [0.5, 1, 2, 4, 8]
        c = [float(np.interp(x, ts, sim["cp"])) * (1 + rng.normal(0, 0.08)) for x in tp]
        subs.append({"subject": str(i), "doses": [{"time": 0.0, "amt": 100.0}],
                     "obs_t": tp, "obs_c": c, "wt": 70.0, "cov": {}})
    return subs


_ADULT_PRIOR = {"names": ["CL", "V"], "mean_log": [math.log(5.0), math.log(50.0)],
                "cov_log": [[0.01, 0.0], [0.0, 0.02]]}


# --- transit / cell-lifespan PD model ---------------------------------------

def test_transit_model_registered_and_delayed():
    assert "pkpd_transit_lifespan" in PKPD_KEYS
    m = get_model("pkpd_transit_lifespan")
    assert m.has_pd and m.amat is None and m.n_cmt >= 4
    p = {"CL": 5.0, "V": 50.0, "KA": 1.0, "BASE": 100.0, "EMAX": 1.0, "EC50": 5.0, "MTT": 24.0}
    sim = simulate_timecourse(m, p, dose=100, tau=24, n_doses=6, tmax=144, n_points=200)
    eff = np.array(sim["eff"])
    cp = np.array(sim["cp"])
    t = np.array(sim["times"])
    assert abs(eff[0] - 100.0) < 1e-6                 # starts at baseline (BASE)
    assert eff.max() > 100.0                          # stimulation raises the count
    assert t[int(np.argmax(eff))] > t[int(np.argmax(cp))]   # effect peak lags cp peak


def test_transit_no_drug_stays_at_baseline():
    m = get_model("pkpd_transit_lifespan")
    p = {"CL": 5.0, "V": 50.0, "KA": 1.0, "BASE": 80.0, "EMAX": 1.0, "EC50": 5.0, "MTT": 24.0}
    sim = simulate_timecourse(m, p, dose=0.0, tau=24, n_doses=3, tmax=72, n_points=100)
    assert np.allclose(sim["eff"], 80.0, atol=1e-6)   # no drug -> count stays at BASE


def test_transit_pd_recovery_with_pk_fixed():
    # Joint 7-param single-subject fits are ill-conditioned (hence the lab fits
    # sequentially); with PK fixed the PD params recover.
    m = get_model("pkpd_transit_lifespan")
    from scipy.optimize import least_squares
    truth = {"CL": 5.0, "V": 50.0, "KA": 1.0, "BASE": 100.0, "EMAX": 1.0, "EC50": 1.0, "MTT": 24.0}
    tp = [1, 2, 4, 8, 12, 24, 36, 48, 72, 96, 120, 144]
    sim = simulate_timecourse(m, truth, dose=400, tau=24, n_doses=6, tmax=144, n_points=500)
    ts = np.array(sim["times"])
    obs = np.array([float(np.interp(x, ts, sim["eff"])) for x in tp])
    pd_names = ["BASE", "EMAX", "EC50", "MTT"]

    def resid(lp):
        p = dict(truth)
        for n, v in zip(pd_names, lp):
            p[n] = float(np.exp(v))
        s = simulate_timecourse(m, p, dose=400, tau=24, n_doses=6, tmax=144, n_points=500)
        pred = np.array([float(np.interp(x, np.array(s["times"]), s["eff"])) for x in tp])
        return (pred - obs) / np.maximum(obs, 1e-6)

    res = least_squares(resid, np.log([80.0, 0.5, 2.0, 12.0]), method="lm", max_nfev=400)
    rec = {n: float(np.exp(v)) for n, v in zip(pd_names, res.x)}
    assert res.success
    assert rec["BASE"] == pytest.approx(100.0, rel=0.1)
    assert rec["MTT"] == pytest.approx(24.0, rel=0.2)
    assert rec["EMAX"] == pytest.approx(1.0, rel=0.3)


def test_transit_fits_via_pkpd_dataset():
    # The joint PK/PD fitter accepts the model (identifiability aside).
    m = get_model("pkpd_transit_lifespan")
    truth = {"CL": 5.0, "V": 50.0, "KA": 1.0, "BASE": 100.0, "EMAX": 1.0, "EC50": 1.0, "MTT": 24.0}
    tp = [0.5, 1, 2, 4, 8, 24, 48, 96, 144]
    sim = simulate_timecourse(m, truth, dose=400, tau=24, n_doses=6, tmax=144, n_points=400)
    ts = np.array(sim["times"])
    pk = [float(np.interp(x, ts, sim["cp"])) for x in tp]
    pd = [float(np.interp(x, ts, sim["eff"])) for x in tp]
    fit = fit_subject_pkpd("pkpd_transit_lifespan", [{"time": 0.0, "amt": 400.0}], tp, pk, tp, pd)
    assert "params" in fit and set(("BASE", "EMAX", "EC50", "MTT")) <= set(fit["params"])


# --- theta priors + MAP -----------------------------------------------------

def test_build_theta_prior_filters_and_shapes():
    m = get_model(MK)
    from app.compute.nlme import _PopSpec, _resolve_iiv
    spec = _PopSpec(m, _resolve_iiv(m, ["CL", "V"]), "proportional", [])
    tp = _build_theta_prior(spec, {"names": ["CL", "V", "NOPE"],
                                   "mean_log": [1.6, 3.9, 0.0], "cov_log": [[0.01, 0, 0], [0, 0.02, 0], [0, 0, 1]]})
    assert tp is not None and set(tp.names) == {"CL", "V"} and tp.prec.shape == (2, 2)
    assert _build_theta_prior(spec, None) is None
    assert _build_theta_prior(spec, {"names": ["ZZZ"], "mean_log": [1.0]}) is None
    # malformed dimensions fail soft to None (not an uncaught IndexError)
    assert _build_theta_prior(spec, {"names": ["CL", "V"], "mean_log": [1.6, 3.9],
                                     "cov_log": [[0.01]]}) is None
    assert _build_theta_prior(spec, {"names": ["CL", "V"], "mean_log": [1.6, 3.9],
                                     "sd_log": [0.1]}) is None


def test_map_default_byte_identical():
    subs = _sparse_peds()
    a = population_fit(MK, subs, method="focei", iiv_params=["CL", "V"], compute_uncertainty=False)
    b = population_fit(MK, subs, method="focei", iiv_params=["CL", "V"], compute_uncertainty=False,
                       theta_prior=None)
    assert a["theta"] == b["theta"] and a["ofv"] == b["ofv"]
    assert "map" not in a and "theta_prior" not in a


def test_map_shrinks_toward_prior_and_reports():
    subs = _sparse_peds()
    no_prior = population_fit(MK, subs, method="focei", iiv_params=["CL", "V"], compute_uncertainty=False)
    mapfit = population_fit(MK, subs, method="focei", iiv_params=["CL", "V"], theta_prior=_ADULT_PRIOR)
    # informative prior at CL=5 pulls the sparse-data estimate toward 5
    assert abs(mapfit["theta"]["CL"] - 5.0) < abs(no_prior["theta"]["CL"] - 5.0)
    assert mapfit.get("map") is True
    assert "ofv_likelihood" in mapfit and mapfit["ofv"] >= mapfit["ofv_likelihood"] - 1e-6
    assert set(mapfit["theta_prior"]["names"]) == {"CL", "V"}
    assert len(mapfit["theta_prior"]["sd_log"]) == 2


def test_weak_prior_follows_data_more_than_tight():
    subs = _sparse_peds()
    tight = population_fit(MK, subs, method="focei", iiv_params=["CL", "V"],
                           compute_uncertainty=False, theta_prior=_ADULT_PRIOR)
    weak = population_fit(MK, subs, method="focei", iiv_params=["CL", "V"], compute_uncertainty=False,
                          theta_prior={"names": ["CL", "V"], "mean_log": [math.log(5.0), math.log(50.0)],
                                       "cov_log": [[1.0, 0.0], [0.0, 1.0]]})
    # data CL ~ 7 > prior 5, so a weaker prior lands the estimate higher (closer to data)
    assert weak["theta"]["CL"] > tight["theta"]["CL"]


def test_prior_only_on_focei():
    subs = _sparse_peds(n=6)
    with pytest.raises(ValueError):
        population_fit(MK, subs, method="saem", theta_prior=_ADULT_PRIOR)


# --- Bayesian diagnostics ---------------------------------------------------

def test_prior_predictive_band_and_coverage():
    subs = _sparse_peds()
    ot = [t for s in subs for t in s["obs_t"]]
    oc = [c for s in subs for c in s["obs_c"]]
    ppc = prior_predictive_check(MK, theta_prior=_ADULT_PRIOR, base_theta={"CL": 5, "V": 50, "KA": 1},
                                 dose=100, tau=24, n_doses=1, obs_times=ot, obs_conc=oc, n_draws=300)
    assert ppc["status"] == "ok" and len(ppc["band"]) > 1
    assert ppc["coverage_pct"] is not None and 0.0 <= ppc["coverage_pct"] <= 100.0
    assert prior_predictive_check(MK, theta_prior={}, base_theta={"CL": 5},
                                  dose=100, tau=24, n_doses=1)["status"] == "no_prior"


def test_prior_predictive_includes_residual_error():
    # The band must reflect the observation model (residual), not just the
    # structural prediction — otherwise coverage of scattered DV is biased low.
    m = get_model(MK)
    sim = simulate_timecourse(m, {"CL": 7.0, "V": 50.0, "KA": 1.0}, dose=100, tau=8,
                              n_doses=1, tmax=8, n_points=200)
    ts = np.array(sim["times"])
    ot = [0.5, 1, 2, 4, 8]
    oc = [float(np.interp(x, ts, sim["cp"])) for x in ot]
    prior = {"names": ["CL", "V"], "mean_log": [math.log(5.0), math.log(50.0)],
             "cov_log": [[0.005, 0], [0, 0.01]]}
    no_sig = prior_predictive_check(MK, theta_prior=prior, base_theta={"CL": 5, "V": 50, "KA": 1},
                                    dose=100, tau=8, n_doses=1, obs_times=ot, obs_conc=oc, n_draws=300)
    with_sig = prior_predictive_check(MK, theta_prior=prior, base_theta={"CL": 5, "V": 50, "KA": 1},
                                      dose=100, tau=8, n_doses=1, obs_times=ot, obs_conc=oc,
                                      sigma={"prop": 0.2, "add": 0.0}, n_draws=300)
    assert with_sig["coverage_pct"] >= no_sig["coverage_pct"]     # residual widens the band


def test_map_estimate_block_omega_uses_correlated_prior():
    # A correlated (block) Omega must enter the individual MAP as the full
    # precision — an observed CL deviation should then inform V. Diagonal path
    # (omega_matrix=None) is unchanged.
    from app.compute.nlme import map_estimate
    m = get_model(MK)
    th = {"CL": 5.0, "V": 50.0, "KA": 1.0}
    om2 = {"CL": 0.09, "V": 0.09}
    sd = np.sqrt([0.09, 0.09])
    om = np.outer(sd, sd) * np.array([[1.0, 0.9], [0.9, 1.0]])
    sim = simulate_timecourse(m, {"CL": 7.0, "V": 50.0, "KA": 1.0}, dose=100, tau=24,
                              n_doses=1, tmax=24, n_points=200)
    ts = np.array(sim["times"])
    ot, oc = [1.0, 4.0], [float(np.interp(x, ts, sim["cp"])) for x in [1.0, 4.0]]
    kw = dict(theta=th, omega2=om2, sigma_prop=0.1, sigma_add=0.0, iiv_params=["CL", "V"],
              obs_t=ot, obs_c=oc, doses=[{"time": 0, "amt": 100}])
    diag = map_estimate(MK, **kw)
    block = map_estimate(MK, omega_matrix=om.tolist(), **kw)
    assert abs(diag["eta"]["V"] - block["eta"]["V"]) > 1e-3
    # a diagonal matrix passed as omega_matrix must reproduce the diagonal MAP
    dm = map_estimate(MK, omega_matrix=np.diag([0.09, 0.09]).tolist(), **kw)
    assert dm["eta"]["V"] == pytest.approx(diag["eta"]["V"], abs=1e-6)


def test_prior_posterior_shrinkage_in_range():
    subs = _sparse_peds()
    fit = population_fit(MK, subs, method="focei", iiv_params=["CL", "V"], theta_prior=_ADULT_PRIOR)
    diag = prior_posterior_diagnostic(fit)
    assert diag["status"] == "ok"
    for row in diag["params"]:
        assert row["prior_mean"] is not None
        if row["shrinkage"] is not None:
            assert 0.0 <= row["shrinkage"] <= 1.0
    # a non-MAP fit has no prior to compare against
    plain = population_fit(MK, subs, method="focei", iiv_params=["CL", "V"], compute_uncertainty=False)
    assert prior_posterior_diagnostic(plain)["status"] == "no_prior"


# --- tool layer -------------------------------------------------------------

def _peds_df(seed=4):
    m = get_model(MK)
    rng = np.random.default_rng(seed)
    rows = []
    for sid in range(1, 15):
        cl = 7.0 * math.exp(rng.normal(0, 0.25))
        v = 55.0 * math.exp(rng.normal(0, 0.25))
        rows.append({"ID": sid, "TIME": 0.0, "DV": np.nan, "AMT": 100.0})
        cp = simulate(m, {"CL": cl, "V": v, "KA": 1.0}, [{"time": 0.0, "amt": 100.0}],
                      [0.5, 1, 2, 4, 8], wt=70.0)["cp"]
        for t, c in zip([0.5, 1, 2, 4, 8], cp):
            rows.append({"ID": sid, "TIME": float(t), "DV": round(float(c * (1 + rng.normal(0, 0.08))), 4),
                         "AMT": np.nan})
    return pd.DataFrame(rows)


_ADULT_FIT = {"status": "ok", "model_key": MK, "label": "1-cmt oral (linear)",
              "theta": {"CL": 5.0, "V": 50.0, "KA": 1.0}, "theta_rse_pct": {"CL": 8.0, "V": 9.0, "KA": 12.0},
              "omega_cv_pct": {"CL": 30.0, "V": 20.0}, "iiv_params": ["CL", "V"], "sigma": {"prop": 0.1, "add": 0.0}}


def test_tools_registered():
    assert default_registry().get("run_prior_check").agent == "modeler"


def test_run_nlme_prior_from_builds_map():
    ctx = ToolContext(dataset_store={"peds": _peds_df()})
    state = PharmState(dataset_id="peds", nlme_results=_ADULT_FIT,
                       dataset_metadata={"detected_roles": {"ID": "ID", "TIME": "TIME", "DV": "DV", "AMT": "AMT"}})
    payload = run_nlme(state, ctx, {"model_key": MK, "iiv_params": ["CL", "V"],
                                    "prior_from": "nlme"}).writes["nlme_results"]
    assert payload["status"] == "ok" and payload.get("map") is True
    assert set(payload["theta_prior"]["names"]) == {"CL", "V", "KA"}
    # MAP CL sits between the adult prior (5) and the pediatric data (~7)
    assert 5.0 < payload["theta"]["CL"] < 7.5
    json.dumps(payload)


def test_run_prior_check_needs_map_then_ok():
    assert run_prior_check(PharmState(), ToolContext(), {}).writes["prior_check_results"]["status"] == "needs_map"
    ctx = ToolContext(dataset_store={"peds": _peds_df()})
    state = PharmState(dataset_id="peds", nlme_results=_ADULT_FIT,
                       dataset_metadata={"detected_roles": {"ID": "ID", "TIME": "TIME", "DV": "DV", "AMT": "AMT"}})
    fit = run_nlme(state, ctx, {"model_key": MK, "iiv_params": ["CL", "V"],
                                "prior_from": "nlme"}).writes["nlme_results"]
    state.nlme_results = fit
    pc = run_prior_check(state, ctx, {"n_draws": 200}).writes["prior_check_results"]
    assert pc["status"] == "ok" and pc["diagnostic"]["status"] == "ok"
    assert pc["prior_predictive"]["status"] == "ok"
    json.dumps(pc)
