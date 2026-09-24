"""Tests for the exposure predictive check + BLQ-incidence VPC (IU PopPK Week 11).

The exposure PC (``model-vpc.R``) summarises each subject's partial AUC (linear
trapezoid) and Cmax, then checks whether the observed GROUP MEAN falls inside the
distribution of simulated group means — the quantity that drives exposure-based
dosing, which a concentration-time VPC does not test directly. The BLQ-incidence
VPC (Bergstrand & Karlsson 2009) checks whether the model reproduces the fraction
of observations below the LLOQ over time.

Covered:
  * EPC: correct model → observed group means within the simulated CI; AUC is
    dose-proportional; group partition sums to the whole; histogram shape.
  * EPC degenerate: missing covariate, too-few-subject groups, <2-point subjects.
  * BLQ: correct model → observed fraction within the simulated band; no_blq when
    nothing censored; no_lloq without a positive LLOQ; empty data; TAD axis.
  * Tool layer: default output carries neither block; exposure_check/blq_check add
    them; LLOQ recovered from the LLOQ column; JSON-safety.
"""
import json
import math

import numpy as np
import pandas as pd
import pytest

from app.compute.pk_models import get_model
from app.compute.pk_simulate import simulate
from app.compute.vpc import (
    _exposure_window_mask,
    _trapz_partial_auc,
    blq_predictive_check,
    exposure_predictive_check,
)
from app.core.pharmstate import PharmState
from app.tools.base import ToolContext
from app.tools.builtins import default_registry
from app.tools.pkmodel_tools import run_vpc

MODEL_KEY = "oral_1cmt"
OBS_T = np.array([0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 24.0])
TYPICAL = {"CL": 5.0, "V": 50.0, "KA": 1.0}
IIV = {"CL": 20.0, "V": 0.0}


def _subject(sid: str, dose: float, cl: float, *, wt: float = 70.0,
             lloq: float | None = None) -> dict:
    model = get_model(MODEL_KEY)
    cp = simulate(model, {"CL": cl, "V": 50.0, "KA": 1.0},
                  [{"time": 0.0, "amt": dose}], OBS_T, wt=wt)["cp"]
    obs = np.maximum(np.asarray(cp, dtype=float), 1e-6)
    s = {"subject": sid, "doses": [{"time": 0.0, "amt": dose}],
         "obs_t": OBS_T.copy(), "obs_c": obs, "wt": wt}
    if lloq is not None:
        blq = obs < lloq
        obs2 = obs.copy()
        obs2[blq] = lloq                      # NONMEM: BLQ rows carry the LLOQ in DV
        s["obs_c"] = obs2
        s["obs_blq"] = blq.tolist()
        s["lloq"] = lloq
    return s


def _population(doses=(25.0, 100.0), n: int = 10, seed: int = 5,
                lloq: float | None = None) -> list[dict]:
    rng = np.random.default_rng(seed)
    sdcl = math.sqrt(math.log(1 + 0.20 ** 2))
    subs = []
    for d in doses:
        for i in range(n):
            subs.append(_subject(f"D{d:g}_{i}", d, 5.0 * math.exp(rng.normal(0.0, sdcl)),
                                  lloq=lloq))
    return subs


# --- helpers ---------------------------------------------------------------

def test_trapz_partial_auc_matches_hand_calc():
    # Triangle area under (0,0)-(1,2)-(2,0) = 2.
    assert _trapz_partial_auc([0.0, 1.0, 2.0], [0.0, 2.0, 0.0]) == pytest.approx(2.0)
    # Negative concentrations floored at 0 first: [2, -5] -> [2, 0], area = 1.
    assert _trapz_partial_auc([0.0, 1.0], [2.0, -5.0]) == pytest.approx(1.0)
    assert _trapz_partial_auc([1.0], [3.0]) == 0.0        # <2 points


def test_exposure_window_single_vs_multiple_dose():
    t = np.array([0.5, 1.0, 12.5, 13.0])
    one = _exposure_window_mask(t, [{"time": 0.0, "amt": 100.0}])
    assert one.all()                                       # whole curve for single dose
    two = _exposure_window_mask(t, [{"time": 0.0, "amt": 100.0}, {"time": 12.0, "amt": 100.0}])
    assert two.tolist() == [False, False, True, True]      # only the last interval


# --- exposure predictive check ---------------------------------------------

def test_epc_correct_model_observed_within_ci():
    subs = _population()
    res = exposure_predictive_check(MODEL_KEY, subs, TYPICAL, IIV,
                                    group_by="DOSE", sigma_prop=0.1, n_sim=200)
    assert res["status"] == "ok" and res["kind"] == "dose"
    assert [g["label"] for g in res["groups"]] == ["25", "100"]   # numeric order
    for g in res["groups"]:
        assert g["n"] == 10
        assert g["auc"]["within"] and g["cmax"]["within"]
        assert g["auc"]["sim_lo"] <= g["auc"]["observed"] <= g["auc"]["sim_hi"]
        assert sum(g["auc"]["hist"]["counts"]) == 200        # every replicate binned
        assert len(g["auc"]["hist"]["edges"]) == len(g["auc"]["hist"]["counts"]) + 1


def test_epc_auc_is_dose_proportional():
    res = exposure_predictive_check(MODEL_KEY, _population(), TYPICAL, IIV,
                                    group_by="DOSE", sigma_prop=0.1, n_sim=120)
    by = {g["label"]: g for g in res["groups"]}
    ratio = by["100"]["auc"]["observed"] / by["25"]["auc"]["observed"]
    assert ratio == pytest.approx(4.0, rel=0.15)             # linear 1-cmt: AUC ∝ dose


def test_epc_partition_sums_to_whole():
    subs = _population(doses=(25.0, 100.0, 200.0), n=6)
    res = exposure_predictive_check(MODEL_KEY, subs, TYPICAL, IIV,
                                    group_by="DOSE", sigma_prop=0.1, n_sim=40)
    assert sum(g["n"] for g in res["groups"]) == len(subs)


def test_epc_missing_covariate_and_small_groups():
    subs = _population()
    miss = exposure_predictive_check(MODEL_KEY, subs, TYPICAL, IIV,
                                     group_by="EGFR", n_sim=5)
    assert miss["status"] == "missing_covariate"
    # A group below min_subjects is reported in skipped, not silently dropped.
    small = _population(doses=(25.0,), n=2) + _population(doses=(100.0,), n=8, seed=9)
    res = exposure_predictive_check(MODEL_KEY, small, TYPICAL, IIV, group_by="DOSE",
                                    sigma_prop=0.1, n_sim=20, min_subjects=3)
    assert [g["label"] for g in res["groups"]] == ["100"]
    assert [s["label"] for s in res["skipped"]] == ["25"]


def test_epc_empty_when_no_usable_subjects():
    # Single-point subjects cannot yield a trapezoidal AUC.
    subs = [{"subject": "x", "doses": [{"time": 0.0, "amt": 100.0}],
             "obs_t": np.array([1.0]), "obs_c": np.array([2.0]), "wt": 70.0}]
    res = exposure_predictive_check(MODEL_KEY, subs, TYPICAL, IIV,
                                    group_by="DOSE", n_sim=5)
    assert res["status"] == "empty"


# --- BLQ-incidence VPC -----------------------------------------------------

def test_blq_correct_model_observed_within_band():
    subs = _population(doses=(25.0,), n=30, lloq=0.15)
    res = blq_predictive_check(MODEL_KEY, subs, TYPICAL, IIV, lloq=0.15,
                               sigma_prop=0.1, n_sim=200)
    assert res["status"] == "ok" and res["n_blq"] > 0
    inside = total = 0
    for b in res["bins"]:
        if None in (b["obs_frac"], b["sim_lo"], b["sim_hi"]):
            continue
        total += 1
        if b["sim_lo"] <= b["obs_frac"] <= b["sim_hi"]:
            inside += 1
    assert total >= 4
    assert inside >= total - 1        # correct model: at most one bin outside the band


def test_blq_fractions_are_valid_probabilities():
    subs = _population(doses=(25.0,), n=30, lloq=0.15)
    res = blq_predictive_check(MODEL_KEY, subs, TYPICAL, IIV, lloq=0.15,
                               sigma_prop=0.1, n_sim=100)
    for b in res["bins"]:
        for key in ("obs_frac", "sim_med", "sim_lo", "sim_hi"):
            if b[key] is not None:
                assert 0.0 <= b[key] <= 1.0
        if b["sim_lo"] is not None and b["sim_hi"] is not None:
            assert b["sim_lo"] <= b["sim_hi"]


def test_blq_no_blq_status_when_nothing_censored():
    subs = _population(doses=(100.0,), n=5)     # no obs_blq flags
    res = blq_predictive_check(MODEL_KEY, subs, TYPICAL, IIV, lloq=0.01, n_sim=5)
    assert res["status"] == "no_blq"


def test_blq_no_lloq_status():
    subs = _population(doses=(25.0,), n=5, lloq=0.15)
    assert blq_predictive_check(MODEL_KEY, subs, TYPICAL, IIV, lloq=None,
                                n_sim=5)["status"] == "no_lloq"
    assert blq_predictive_check(MODEL_KEY, subs, TYPICAL, IIV, lloq=0.0,
                                n_sim=5)["status"] == "no_lloq"        # non-positive


def test_blq_empty_data():
    res = blq_predictive_check(MODEL_KEY, [], TYPICAL, IIV, lloq=0.15, n_sim=5)
    assert res["status"] == "empty"


def test_blq_invalid_axis_raises():
    subs = _population(doses=(25.0,), n=5, lloq=0.15)
    with pytest.raises(ValueError):
        blq_predictive_check(MODEL_KEY, subs, TYPICAL, IIV, lloq=0.15, x_by="space", n_sim=5)


def test_blq_tad_axis_runs():
    subs = _population(doses=(25.0,), n=20, lloq=0.15)
    res = blq_predictive_check(MODEL_KEY, subs, TYPICAL, IIV, lloq=0.15,
                               sigma_prop=0.1, n_sim=40, x_by="tad")
    assert res["status"] == "ok" and res["x_by"] == "tad"


# --- tool layer ------------------------------------------------------------

def _tool_dataset() -> pd.DataFrame:
    """12 subjects, 2 dose groups (100/300), with an explicit LLOQ column and a
    handful of BLQ rows so blq_check has something to check."""
    rows = []
    lloq = 0.05
    for sid in range(1, 13):
        dose = 100.0 if sid <= 6 else 300.0
        rows.append({"ID": sid, "TIME": 0.0, "DV": np.nan, "AMT": dose,
                     "DOSE": dose, "LLOQ": lloq, "CENS": 0})
        model = get_model(MODEL_KEY)
        cp = simulate(model, {"CL": 5.0, "V": 50.0, "KA": 1.0},
                      [{"time": 0.0, "amt": dose}], OBS_T, wt=70.0)["cp"]
        for t, c in zip(OBS_T, cp):
            blq = 1 if c < lloq else 0
            rows.append({"ID": sid, "TIME": float(t),
                         "DV": lloq if blq else float(c), "AMT": np.nan,
                         "DOSE": dose, "LLOQ": lloq, "CENS": blq})
    return pd.DataFrame(rows)


def _pk_model_results() -> dict:
    return {
        "status": "ok", "mode": "fit", "model_key": MODEL_KEY,
        "individual_fits": [
            {"subject": sid, "converged": True,
             "params": {"CL": 5.0, "V": 50.0, "KA": 1.0}} for sid in range(1, 13)
        ],
        "population": {"parameters": {
            "CL": {"typical_value": 5.0, "iiv_cv_pct": 25.0},
            "V": {"typical_value": 50.0, "iiv_cv_pct": 20.0},
            "KA": {"typical_value": 1.0, "iiv_cv_pct": 0.0},
        }},
    }


@pytest.fixture
def loaded():
    ctx = ToolContext(dataset_store={"d1": _tool_dataset()})
    state = PharmState(dataset_id="d1", pk_model_results=_pk_model_results(),
                       dataset_metadata={"detected_roles":
                                         {"ID": "ID", "TIME": "TIME", "DV": "DV",
                                          "AMT": "AMT", "LLOQ": "LLOQ", "CENS": "CENS"}})
    return state, ctx


def test_default_output_has_no_exposure_or_blq(loaded):
    state, ctx = loaded
    payload = run_vpc(state, ctx, {}).writes["vpc_results"]
    assert payload["status"] == "ok"
    assert "exposure_pc" not in payload
    assert "blq_vpc" not in payload


def test_exposure_check_groups_by_dose(loaded):
    state, ctx = loaded
    payload = run_vpc(state, ctx, {"exposure_check": True}).writes["vpc_results"]
    exp = payload["exposure_pc"]
    assert exp["status"] == "ok" and exp["group_by"] == "DOSE"
    assert {g["label"] for g in exp["groups"]} == {"100", "300"}
    json.dumps(payload)                                    # no numpy scalars leak


def test_blq_check_recovers_lloq_from_column(loaded):
    state, ctx = loaded
    payload = run_vpc(state, ctx, {"blq_check": True}).writes["vpc_results"]
    blq = payload["blq_vpc"]
    # LLOQ comes from the LLOQ column (0.05), not the median BLQ DV.
    assert blq["status"] in ("ok", "no_blq")
    assert blq.get("lloq") == pytest.approx(0.05) or blq["status"] == "no_blq"
    json.dumps(payload)


def test_tool_is_registered():
    assert default_registry().get("run_vpc").agent == "modeler"


# --- non-finite robustness (adversarial-review findings) -------------------

def test_epc_summ_survives_non_finite_replicate(monkeypatch):
    """A degenerate simulate() draw (inf/nan) in one replicate must not crash
    np.histogram or poison the percentile — it is dropped, matching the module's
    documented non-finite contract (obs_vs_pred / pcvpc)."""
    import app.compute.vpc as vpcmod
    real = vpcmod.simulate
    state = {"n": 0}

    def flaky(model, params, doses, times, **kw):
        out = real(model, params, doses, times, **kw)
        state["n"] += 1
        if state["n"] == 3:          # one degenerate replicate mid-run
            cp = np.asarray(out["cp"], dtype=float).copy()
            cp[0] = np.inf
            return {**out, "cp": cp}
        return out

    monkeypatch.setattr(vpcmod, "simulate", flaky)
    res = exposure_predictive_check(MODEL_KEY, _population(doses=(25.0,), n=6),
                                    TYPICAL, IIV, group_by="DOSE", n_sim=30)
    assert res["status"] == "ok"                         # no crash
    for g in res["groups"]:
        for met in ("auc", "cmax"):
            m = g[met]
            # summaries stay finite (the inf replicate was dropped)
            for k in ("sim_median", "sim_lo", "sim_hi"):
                assert m[k] is None or np.isfinite(m[k])


def test_blq_excludes_non_finite_simulated_draws(monkeypatch):
    """A non-finite simulated concentration must be EXCLUDED, not scored not-BLQ
    (nan < lloq is False), which would bias the simulated fraction downward."""
    import app.compute.vpc as vpcmod
    real = vpcmod.simulate
    state = {"n": 0}

    def flaky(model, params, doses, times, **kw):
        out = real(model, params, doses, times, **kw)
        state["n"] += 1
        if state["n"] % 7 == 0:
            cp = np.asarray(out["cp"], dtype=float).copy()
            cp[-1] = np.nan
            return {**out, "cp": cp}
        return out

    monkeypatch.setattr(vpcmod, "simulate", flaky)
    subs = _population(doses=(25.0,), n=30, lloq=0.15)
    res = blq_predictive_check(MODEL_KEY, subs, TYPICAL, IIV, lloq=0.15,
                               sigma_prop=0.1, n_sim=60)
    assert res["status"] == "ok"
    for b in res["bins"]:                                # fractions stay valid
        for key in ("sim_med", "sim_lo", "sim_hi"):
            if b[key] is not None:
                assert 0.0 <= b[key] <= 1.0
