"""Tests for the Week-14 pediatric dose-finding simulation: age x weight
stratification, the external adult reference band, the %-within-range
dose-selection metric, the weight-for-age reference population, and the
model-estimated allometric exponent (opt-in) vs the built-in fixed 0.75/1.0.

Validated against the analytic property of the 2-cmt model with an estimated
weight-CL exponent: for wt < 70 kg a SMALLER exponent (0.663 < 0.75) gives a
LARGER (wt/70)^b factor -> higher CL -> lower AUCss, so switching from fixed to
estimated allometry lowers young-child exposure and shifts the matched dose up.
"""
import json

import numpy as np
import pandas as pd
import pytest

from app.compute.clinsim import (
    _pediatric_strata,
    _scale_wt,
    individual_exposures,
    pediatric_reference_population,
    pediatric_simulation,
)
from app.compute.pk_models import get_model
from app.compute.pk_simulate import simulate
from app.core.pharmstate import PharmState
from app.tools.base import ToolContext
from app.tools.builtins import default_registry
from app.tools.pkmodel_tools import run_pediatric_simulation

MODEL = "oral_2cmt"
THETA = {"CL": 5.0, "VC": 50.0, "Q": 10.0, "VP": 100.0, "KA": 1.0}
OMEGA = {"CL": 30.0, "VC": 20.0}
IIV = ["CL", "VC"]
EGFR_EFF = [{"param": "CL", "covariate": "EGFR", "kind": "power",
             "center": 90.0, "coefficient": 0.519, "rse_pct": 15.0}]
DOSES = [5, 10, 15, 20, 25]


def _adult_reference(dose: float = 25.0) -> dict:
    """Adult AUCss/Cmax,ss distribution at the label dose (the comparison band)."""
    subs = [{"subject": str(i), "cov": {"EGFR": 90.0 + (i % 40)}, "wt": 60.0 + (i % 30)}
            for i in range(200)]
    rng = np.random.default_rng(7)
    etas = {str(i): {"CL": float(rng.normal(0, 0.3)), "VC": float(rng.normal(0, 0.2))}
            for i in range(200)}
    ie = individual_exposures(MODEL, theta=THETA, subjects=subs, etas=etas,
                              covariate_effects=EGFR_EFF, iiv_params=IIV, dose=dose,
                              tau=12, n_doses=14)
    return {"auc_tau": [s["auc_ss"] for s in ie["subjects"]],
            "cmax": [s["cmax_ss"] for s in ie["subjects"]]}


# --- weight-scaling helper (gap 1) -----------------------------------------

def test_scale_wt_reference_and_custom():
    p = {"CL": 5.0, "VC": 50.0, "KA": 1.0}
    # no exponents -> unchanged, sim at the subject weight (built-in allometry)
    q, wt = _scale_wt(p, 40.0, None)
    assert q == p and wt == 40.0
    # at 70 kg the custom factor is exactly 1
    q, wt = _scale_wt(p, 70.0, {"CL": 0.663, "VC": 1.087})
    assert q["CL"] == pytest.approx(5.0) and wt == 70.0
    # below 70 kg, applied explicitly; sim weight neutralised to 70
    q, wt = _scale_wt(p, 35.0, {"CL": 0.663})
    assert q["CL"] == pytest.approx(5.0 * (35.0 / 70.0) ** 0.663) and wt == 70.0
    assert q["VC"] == 50.0                      # untouched param


# --- 2-D age x weight stratification ---------------------------------------

def test_pediatric_strata_cross_and_ranges():
    rows = [{"AGE": 4.0, "WT": 15.0}, {"AGE": 4.0, "WT": 20.0},   # 2-<6: two wt bins
            {"AGE": 9.0, "WT": 30.0}, {"AGE": 15.0, "WT": 80.0},  # 6-<12, 12-<18
            {"AGE": 1.0, "WT": 10.0},                             # <2 y dropped
            {"AGE": 4.0, "WT": 40.0}]                             # wt out of 2-<6 range
    parts, meta, order = _pediatric_strata(rows, "AGE", "WT")
    assert "2 to <6 y · 12 to <18 kg" in parts and "2 to <6 y · 18 to <25 kg" in parts
    assert meta["6 to <12 y · 20 to <40 kg"] == ("6 to <12 y", "20 to <40 kg")
    assert sum(len(v) for v in parts.values()) == 4      # 2 dropped (age<2, wt out of range)
    assert order == sorted(order, key=lambda x: (["2", "6", "12"].index(x.split(" ")[0]), x))


# --- weight-for-age reference population (gap 5) ----------------------------

def test_pediatric_reference_population_weight_for_age():
    cov, wt = pediatric_reference_population(n=4000, seed=1)
    assert len(cov) == 4000
    for r in cov:
        assert 2.0 <= r["AGE"] < 18.0 and 8.0 <= r["WT"] <= 120.0 and r["EGFR"] == 90.0
    # weight correlates with age (older children are heavier)
    ages = np.array([r["AGE"] for r in cov]); wts = np.array([r["WT"] for r in cov])
    assert np.corrcoef(ages, wts)[0, 1] > 0.6
    young = wts[ages < 6].mean(); teen = wts[ages >= 12].mean()
    assert teen > 2.0 * young
    # most age x weight cells populate
    _p, _m, order = _pediatric_strata(cov, "AGE", "WT")
    assert len(order) >= 5
    # deterministic
    cov2, _ = pediatric_reference_population(n=4000, seed=1)
    assert [r["WT"] for r in cov] == [r["WT"] for r in cov2]


# --- pediatric simulation (gaps 2-4) ---------------------------------------

def test_pediatric_simulation_bands_and_recommendation():
    cov, wt = pediatric_reference_population(n=6000, seed=2)
    ref = _adult_reference(25.0)
    r = pediatric_simulation(MODEL, theta=THETA, omega_cv_pct=OMEGA, iiv_params=IIV,
                             cov_rows=cov, wt_rows=wt, reference_exposures=ref, doses=DOSES,
                             tau=12, n_doses=14, covariate_effects=EGFR_EFF, n_per_stratum=300)
    assert r["status"] == "ok" and r["kind"] == "pediatric" and r["allometry"] == "fixed"
    band = r["reference_band"]["auc_tau"]
    assert band["lo"] is not None and band["hi"] > band["lo"] and band["n"] == 200
    # every dose row carries a pct_within_ref, and the recommended dose maximizes it
    for s in r["strata"]:
        pcts = {d["dose"]: d["auc_tau"]["pct_within_ref"] for d in s["doses"]}
        assert all(0.0 <= p <= 100.0 for p in pcts.values())
        if s["recommended_dose"] is not None:            # None only when no overlap anywhere
            assert pcts[s["recommended_dose"]] == max(pcts.values()) and max(pcts.values()) > 0.0
    # younger/lighter children need a lower dose than older/heavier ones
    by = {s["label"]: s["recommended_dose"] for s in r["strata"]}
    youngest = by["2 to <6 y · 12 to <18 kg"]
    oldest = by["12 to <18 y · 70 to <100 kg"]
    assert youngest is not None and oldest is not None and youngest < oldest


def test_estimated_exponent_shifts_dose_vs_fixed():
    cov, wt = pediatric_reference_population(n=6000, seed=3)
    ref = _adult_reference(25.0)
    kw = dict(theta=THETA, omega_cv_pct=OMEGA, iiv_params=IIV, cov_rows=cov, wt_rows=wt,
              reference_exposures=ref, doses=DOSES, tau=12, n_doses=14,
              covariate_effects=EGFR_EFF, n_per_stratum=300)
    fixed = pediatric_simulation(MODEL, **kw)
    est = pediatric_simulation(MODEL, wt_exponents={"CL": 0.663, "Q": 0.663,
                                                    "VC": 1.087, "VP": 1.087}, **kw)
    assert est["allometry"] == "estimated"
    # a smaller CL exponent raises young-child CL -> lower AUCss than fixed 0.75
    lb = "2 to <6 y · 12 to <18 kg"
    f50 = [d for d in next(s for s in fixed["strata"] if s["label"] == lb)["doses"]
           if d["dose"] == 25][0]["auc_tau"]["p50"]
    e50 = [d for d in next(s for s in est["strata"] if s["label"] == lb)["doses"]
           if d["dose"] == 25][0]["auc_tau"]["p50"]
    assert e50 < f50
    # the two allometry choices do not give identical recommendations everywhere
    fr = {s["label"]: s["recommended_dose"] for s in fixed["strata"]}
    er = {s["label"]: s["recommended_dose"] for s in est["strata"]}
    assert fr != er


def test_no_overlap_recommends_none():
    # An adult band from a very LOW dose -> tiny AUCss; pediatric at high doses sits
    # entirely above it, so no dose matches -> recommended_dose is None (not dose 0%).
    cov, wt = pediatric_reference_population(n=3000, seed=8)
    low_ref = _adult_reference(1.0)                     # adults at 1 mg -> tiny exposure
    r = pediatric_simulation(MODEL, theta=THETA, omega_cv_pct=OMEGA, iiv_params=IIV,
                             cov_rows=cov, wt_rows=wt, reference_exposures=low_ref,
                             doses=[50, 100, 200], tau=12, n_doses=14,
                             covariate_effects=EGFR_EFF, n_per_stratum=200)
    assert r["status"] == "ok"
    # at least one stratum is entirely above the tiny band -> None + directional note
    nones = [s for s in r["strata"] if s["recommended_dose"] is None]
    assert nones and all("above the adult range" in s["note"] for s in nones)
    for s in nones:
        assert all(d["auc_tau"]["pct_within_ref"] == 0.0 for d in s["doses"])


def test_weight_for_age_populates_young_bands():
    # After the CDG-fit quadratic, median weight at age ~12 is near the 40 kg floor
    # so the 12-<18 y 40-70 kg band is not starved of its youngest members.
    cov, _ = pediatric_reference_population(n=8000, seed=1)
    ages = np.array([r["AGE"] for r in cov]); wts = np.array([r["WT"] for r in cov])
    med12 = np.median(wts[(ages >= 11.5) & (ages < 12.5)])
    assert med12 >= 38.0                                # ~40 kg, was ~36 with the log-linear curve
    parts, _m, order = _pediatric_strata(cov, "AGE", "WT")
    assert len(order) == 6                              # all six age×weight cells populated
    assert all(len(parts[lb]) >= 30 for lb in order)    # none starved


def test_pediatric_deterministic():
    cov, wt = pediatric_reference_population(n=2000, seed=5)
    ref = _adult_reference(25.0)
    kw = dict(theta=THETA, omega_cv_pct=OMEGA, iiv_params=IIV, cov_rows=cov, wt_rows=wt,
              reference_exposures=ref, doses=[10, 25], tau=12, n_doses=14,
              covariate_effects=EGFR_EFF, n_per_stratum=120)
    a = pediatric_simulation(MODEL, **kw)
    b = pediatric_simulation(MODEL, **kw)
    assert [s["recommended_dose"] for s in a["strata"]] == [s["recommended_dose"] for s in b["strata"]]


def test_individual_exposures_wt_exponent_default_identical():
    subs = [{"subject": str(i), "cov": {"EGFR": 90.0}, "wt": 40.0 + i} for i in range(6)]
    base = individual_exposures(MODEL, theta=THETA, subjects=subs, etas={}, iiv_params=[],
                                dose=25, tau=12, n_doses=14)
    same = individual_exposures(MODEL, theta=THETA, subjects=subs, etas={}, iiv_params=[],
                                dose=25, tau=12, n_doses=14, wt_exponents=None)
    assert [s["auc_ss"] for s in base["subjects"]] == [s["auc_ss"] for s in same["subjects"]]


def test_pediatric_degenerate():
    cov, wt = pediatric_reference_population(n=500, seed=6)
    ref = _adult_reference(25.0)
    assert pediatric_simulation(MODEL, theta=THETA, omega_cv_pct=OMEGA, iiv_params=IIV,
        cov_rows=cov, wt_rows=wt, reference_exposures=ref, doses=[0], tau=12,
        n_doses=14)["status"] == "no_doses"
    assert pediatric_simulation(MODEL, theta=THETA, omega_cv_pct=OMEGA, iiv_params=IIV,
        cov_rows=[], wt_rows=[], reference_exposures=ref, doses=DOSES, tau=12,
        n_doses=14)["status"] == "no_covariates"
    assert pediatric_simulation(MODEL, theta=THETA, omega_cv_pct=OMEGA, iiv_params=IIV,
        cov_rows=cov, wt_rows=wt, reference_exposures={}, doses=DOSES, tau=12,
        n_doses=14)["status"] == "no_reference"
    assert pediatric_simulation(MODEL, theta=THETA, omega_cv_pct=OMEGA, iiv_params=IIV,
        cov_rows=[{"AGE": 40.0, "WT": 70.0}], wt_rows=[70.0], reference_exposures=ref,
        doses=DOSES, tau=12, n_doses=14)["status"] == "no_strata"      # adults only
    with pytest.raises(ValueError):
        pediatric_simulation(MODEL, theta=THETA, omega_cv_pct=OMEGA, iiv_params=IIV,
            cov_rows=cov, wt_rows=wt, reference_exposures=ref, doses=DOSES, tau=12,
            n_doses=14, metrics=("bogus",))


# --- tool layer ------------------------------------------------------------

def _dataset() -> pd.DataFrame:
    rows = []
    for sid in range(1, 25):
        egfr = 88.0 + sid; w = 60.0 + sid * 0.7
        rows.append({"ID": sid, "TIME": 0.0, "DV": np.nan, "AMT": 25.0, "EGFR": egfr, "WT": w})
        cp = simulate(get_model(MODEL), THETA, [{"time": 0.0, "amt": 25.0}],
                      [0.5, 1, 2, 4, 8, 12], wt=w)["cp"]
        for t, c in zip([0.5, 1, 2, 4, 8, 12], cp):
            rows.append({"ID": sid, "TIME": float(t), "DV": float(c), "AMT": np.nan,
                         "EGFR": egfr, "WT": w})
    return pd.DataFrame(rows)


def _pk_results() -> dict:
    return {"status": "ok", "mode": "fit", "model_key": MODEL,
            "individual_fits": [{"subject": s, "converged": True, "params": THETA} for s in range(1, 25)],
            "population": {"parameters": {
                "CL": {"typical_value": 5.0, "iiv_cv_pct": 30.0},
                "VC": {"typical_value": 50.0, "iiv_cv_pct": 20.0},
                "Q": {"typical_value": 10.0, "iiv_cv_pct": 0.0},
                "VP": {"typical_value": 100.0, "iiv_cv_pct": 0.0},
                "KA": {"typical_value": 1.0, "iiv_cv_pct": 0.0}}}}


def _nlme() -> dict:
    return {"status": "ok", "model_key": MODEL, "theta": THETA,
            "omega_cv_pct": OMEGA, "iiv_params": IIV, "sigma": {"prop": 0.1, "add": 0.0},
            "covariate_effects": EGFR_EFF}


@pytest.fixture
def loaded():
    ctx = ToolContext(dataset_store={"d1": _dataset()})
    state = PharmState(dataset_id="d1", pk_model_results=_pk_results(), nlme_results=_nlme(),
                       dataset_metadata={"detected_roles":
                                         {"ID": "ID", "TIME": "TIME", "DV": "DV", "AMT": "AMT"}})
    return state, ctx


def test_tool_registered_as_simulator():
    assert default_registry().get("run_pediatric_simulation").agent == "simulator"


def test_run_pediatric_needs_fit():
    res = run_pediatric_simulation(PharmState(), ToolContext(), {})
    assert res.writes["pediatric_results"]["status"] == "no_fit"


def test_run_pediatric_reference_source(loaded):
    state, ctx = loaded
    payload = run_pediatric_simulation(state, ctx, {"doses": DOSES, "n_per_stratum": 150,
                                                    "n_pediatric": 4000}).writes["pediatric_results"]
    assert payload["status"] == "ok" and payload["population_source"] == "reference"
    assert payload["reference_source"] == "dataset adults" and payload["allometry"] == "fixed"
    assert len(payload["strata"]) >= 5
    json.dumps(payload)


def test_run_pediatric_estimated_exponent(loaded):
    state, ctx = loaded
    payload = run_pediatric_simulation(state, ctx, {"doses": DOSES, "n_per_stratum": 150,
                                                    "wt_exponent_cl": 0.663,
                                                    "wt_exponent_v": 1.087}).writes["pediatric_results"]
    assert payload["status"] == "ok" and payload["allometry"] == "estimated"
    json.dumps(payload)


def test_run_pediatric_strips_wt_covariate_no_double_count(loaded):
    # A WT covariate in the fitted model must be stripped (weight already enters via
    # the simulator's built-in allometry) — so results are IDENTICAL with/without it,
    # on the DEFAULT fixed-allometry path (the double-count the guard prevents).
    state, ctx = loaded
    args = {"doses": DOSES, "n_per_stratum": 150}
    without = run_pediatric_simulation(state, ctx, args).writes["pediatric_results"]
    nl_wt = _nlme()
    nl_wt["covariate_effects"] = EGFR_EFF + [
        {"param": "CL", "covariate": "WT", "kind": "power", "center": 70.0,
         "coefficient": 0.75, "rse_pct": 20.0}]
    state_wt = PharmState(dataset_id="d1", pk_model_results=_pk_results(), nlme_results=nl_wt,
                          dataset_metadata=state.dataset_metadata)
    withwt = run_pediatric_simulation(state_wt, ctx, args).writes["pediatric_results"]
    med = lambda p: [d["auc_tau"]["p50"] for s in p["strata"] for d in s["doses"]]
    assert med(without) == med(withwt)          # WT stripped -> no effect -> identical
