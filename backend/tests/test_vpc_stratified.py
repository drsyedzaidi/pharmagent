"""Tests for stratified / dose-normalized VPCs (Week-11 evaluation).

The IU PopPK Week-11 lecture is explicit that a VPC pooled across dose groups is
misleading — the lowest dose informs the lower percentile and the highest dose
the upper — so pooled data must be *dose-normalized* or *stratified by dose*.
These tests validate the two remedies against that stated failure mode, and pin
the inertness guarantee: ``correction="pred", x_by="time"`` is byte-for-byte the
original prediction-corrected VPC.

Covered:
  * The pooling artifact and its two fixes (the headline test).
  * ``_partition_by_stratum``: sums to the whole; DOSE / categorical / quartile.
  * ``correction="pred" + x_by="time"`` == the pre-existing pcvpc output.
  * Degenerate inputs return a status, never raise.
  * TAD binning == TIME for single-dose; folds intervals for multiple doses.
  * Tool layer: default payload unchanged; ``bad_stratum`` lists covariates;
    stratified block is JSON-safe.
"""
import json
import math

import numpy as np
import pandas as pd
import pytest

from app.compute.pk_models import get_model
from app.compute.pk_simulate import simulate
from app.compute.vpc import _partition_by_stratum, pcvpc, stratified_vpc
from app.core.pharmstate import PharmState
from app.tools.base import ToolContext
from app.tools.builtins import default_registry
from app.tools.pkmodel_tools import run_vpc

MODEL_KEY = "oral_1cmt"
OBS_T = np.array([0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0])


# --- populations -----------------------------------------------------------

def _one_dose_pop(n: int = 24, dose: float = 100.0, seed: int = 5) -> list[dict]:
    """A single-dose population with 20% CV on CL, data == model predictions."""
    model = get_model(MODEL_KEY)
    rng = np.random.default_rng(seed)
    sdcl = math.sqrt(math.log(1 + 0.20 ** 2))
    subs = []
    for i in range(n):
        cl = 5.0 * math.exp(rng.normal(0.0, sdcl))
        cp = simulate(model, {"CL": cl, "V": 50.0, "KA": 1.0},
                      [{"time": 0.0, "amt": dose}], OBS_T, wt=70.0)["cp"]
        subs.append({"subject": f"S{i}", "doses": [{"time": 0.0, "amt": dose}],
                     "obs_t": OBS_T.copy(), "obs_c": np.maximum(cp, 1e-6), "wt": 70.0})
    return subs


def _two_dose_pop(n: int = 20, seed: int = 7) -> list[dict]:
    """Two dose groups (25 and 200 mg) with IDENTICAL underlying PK. For a
    linear 1-compartment model the concentrations scale with dose, so the raw
    pooled spread is inflated ~8x — exactly the lecture's artifact."""
    subs = []
    for grp, dose in enumerate((25.0, 200.0)):
        subs += _one_dose_pop(n=n, dose=dose, seed=seed + grp)
    return subs


def _cov_pop(cov_name: str, values: list, seed: int = 3) -> list[dict]:
    """One subject per covariate value, carrying it in ``subj['cov']``."""
    base = _one_dose_pop(n=len(values), seed=seed)
    for s, v in zip(base, values):
        s["cov"] = {cov_name: v}
    return base


IIV = {"CL": 20.0, "V": 0.0}
TYPICAL = {"CL": 5.0, "V": 50.0, "KA": 1.0}


def _max_obs_spread(res: dict) -> float:
    """Widest observed 95/5 ratio across populated bins — the pooling artifact
    magnitude. Independent of the simulated replicates."""
    ratios = [b["obs_p95"] / b["obs_p05"] for b in res["bins"]
              if b["obs_p05"] not in (None, 0)]
    return max(ratios) if ratios else float("nan")


# --- headline: the pooling artifact and its two fixes ----------------------

def test_dose_normalization_collapses_the_pooling_artifact():
    subs = _two_dose_pop()
    # obs percentiles do not depend on the simulated band -> n_sim=1 for speed.
    raw = pcvpc(MODEL_KEY, subs, TYPICAL, IIV, correction="none", n_sim=1)
    norm = pcvpc(MODEL_KEY, subs, TYPICAL, IIV, correction="dose", n_sim=1)

    raw_spread = _max_obs_spread(raw)
    norm_spread = _max_obs_spread(norm)
    # Raw pooling stacks a 25 mg and a 200 mg group -> a ~8x-inflated band;
    # dose-normalization scales both to a common dose and collapses it.
    assert raw_spread > 3.0 * norm_spread
    assert norm_spread < 3.0            # residual spread is IIV only (20% CV)


def test_stratify_by_dose_splits_the_two_groups():
    subs = _two_dose_pop()
    res = stratified_vpc(MODEL_KEY, subs, TYPICAL, IIV, stratify_by="DOSE", n_sim=1)
    assert res["status"] == "ok" and res["kind"] == "dose"
    labels = [st["label"] for st in res["strata"]]
    assert labels == ["25", "200"]                    # numeric-aware ordering
    assert all(st["n"] == 20 for st in res["strata"])
    assert res["skipped"] == []


# --- inertness: default path is byte-for-byte the original -----------------

def test_pred_time_is_byte_identical_to_the_original_pcvpc():
    subs = _one_dose_pop()
    default = pcvpc(MODEL_KEY, subs, TYPICAL, IIV, sigma_prop=0.1, n_sim=40)
    explicit = pcvpc(MODEL_KEY, subs, TYPICAL, IIV, sigma_prop=0.1, n_sim=40,
                     correction="pred", x_by="time")
    assert default == explicit               # same seed + same code path


def test_invalid_correction_and_axis_raise():
    subs = _one_dose_pop(n=4)
    with pytest.raises(ValueError):
        pcvpc(MODEL_KEY, subs, TYPICAL, IIV, correction="bogus", n_sim=1)
    with pytest.raises(ValueError):
        pcvpc(MODEL_KEY, subs, TYPICAL, IIV, x_by="space", n_sim=1)


# --- _partition_by_stratum -------------------------------------------------

def test_partition_by_dose_sums_to_the_whole():
    subs = _two_dose_pop()
    parts, kind = _partition_by_stratum(subs, "DOSE")
    assert kind == "dose"
    assert sum(len(v) for v in parts.values()) == len(subs)
    assert set(parts) == {"25", "200"}


def test_partition_categorical_covariate():
    subs = _cov_pop("SEX", [0, 1, 0, 1, 0, 1, 0, 1])
    parts, kind = _partition_by_stratum(subs, "SEX")
    assert kind == "categorical"          # 2 distinct numeric values < threshold
    assert set(parts) == {"0", "1"}
    assert sum(len(v) for v in parts.values()) == len(subs)


def test_partition_continuous_covariate_quartiles():
    subs = _cov_pop("WT", list(range(50, 106, 2)))   # 28 distinct values
    parts, kind = _partition_by_stratum(subs, "WT")
    assert kind == "quartile"
    assert set(parts) == {"Q1", "Q2", "Q3", "Q4"}
    assert sum(len(v) for v in parts.values()) == len(subs)
    # quartiles are balanced: no bucket holds more than half the subjects
    assert all(len(v) <= len(subs) // 2 + 1 for v in parts.values())


# --- degenerate inputs never raise -----------------------------------------

def test_stratified_missing_covariate_is_reported():
    subs = _one_dose_pop(n=6)             # no 'cov' key at all
    res = stratified_vpc(MODEL_KEY, subs, TYPICAL, IIV, stratify_by="EGFR", n_sim=1)
    assert res["status"] == "missing_covariate"
    assert res["stratify_by"] == "EGFR"


def test_stratified_skips_underpowered_strata():
    # 3 subjects at 25 mg (below min_subjects), 20 at 200 mg.
    subs = _one_dose_pop(n=3, dose=25.0, seed=1) + _one_dose_pop(n=20, dose=200.0, seed=2)
    res = stratified_vpc(MODEL_KEY, subs, TYPICAL, IIV, stratify_by="DOSE",
                         n_sim=1, min_subjects=5)
    assert res["status"] == "ok"
    assert [st["label"] for st in res["strata"]] == ["200"]
    assert [sk["label"] for sk in res["skipped"]] == ["25"]
    assert res["skipped"][0]["n"] == 3


def test_stratified_all_too_small_returns_no_strata():
    subs = _one_dose_pop(n=2, dose=25.0) + _one_dose_pop(n=2, dose=200.0, seed=9)
    res = stratified_vpc(MODEL_KEY, subs, TYPICAL, IIV, stratify_by="DOSE",
                         n_sim=1, min_subjects=5)
    assert res["status"] == "no_strata"
    assert res["strata"] == []


def test_pcvpc_empty_population_is_graceful():
    res = pcvpc(MODEL_KEY, [], TYPICAL, IIV, n_sim=1)
    assert res["status"] == "empty"
    assert res["bins"] == []


# --- TAD binning -----------------------------------------------------------

def test_tad_equals_time_for_single_dose():
    subs = _one_dose_pop()
    by_time = pcvpc(MODEL_KEY, subs, TYPICAL, IIV, sigma_prop=0.1, n_sim=20, x_by="time")
    by_tad = pcvpc(MODEL_KEY, subs, TYPICAL, IIV, sigma_prop=0.1, n_sim=20, x_by="tad")
    # dose at t=0 -> time-after-dose == absolute time -> identical binning.
    assert by_tad["bins"] == by_time["bins"]


def test_tad_folds_multiple_dose_intervals():
    model = get_model(MODEL_KEY)
    doses = [{"time": 0.0, "amt": 100.0}, {"time": 12.0, "amt": 100.0}]
    t = np.array([1.0, 3.0, 6.0, 11.0, 13.0, 15.0, 18.0, 23.0])   # spans both intervals
    subs = []
    for i in range(12):
        cp = simulate(model, {"CL": 5.0 + 0.1 * i, "V": 50.0, "KA": 1.0}, doses, t, wt=70.0)["cp"]
        subs.append({"subject": f"M{i}", "doses": doses, "obs_t": t.copy(),
                     "obs_c": np.maximum(cp, 1e-6), "wt": 70.0})
    by_tad = pcvpc(MODEL_KEY, subs, TYPICAL, IIV, sigma_prop=0.1, n_sim=1, x_by="tad")
    by_time = pcvpc(MODEL_KEY, subs, TYPICAL, IIV, sigma_prop=0.1, n_sim=1, x_by="time")
    assert by_tad["status"] == "ok"
    tad_max = max(b["t"] for b in by_tad["bins"] if b["t"] is not None)
    time_max = max(b["t"] for b in by_time["bins"] if b["t"] is not None)
    # TAD folds the second interval back to <= the dosing interval (12 h),
    # whereas absolute time runs out to the last sample (23 h).
    assert tad_max <= 12.0 < time_max


# --- tool layer ------------------------------------------------------------

def _tool_dataset() -> pd.DataFrame:
    """12 subjects across two dose groups (100 / 300 mg), 6 each, plus an RF
    covariate column so `available` has a real covariate to list."""
    rows = []
    for sid in range(1, 13):
        dose = 100.0 if sid <= 6 else 300.0
        rf = float(sid % 3)                       # 0/1/2 categorical covariate
        rows.append({"ID": sid, "TIME": 0.0, "DV": np.nan, "AMT": dose, "RF": rf})
        for t, c in zip([0.5, 1, 2, 4, 8, 12], [0.8, 1.4, 1.2, 0.9, 0.5, 0.2]):
            rows.append({"ID": sid, "TIME": t, "DV": c * dose / 100.0 * (1 + 0.02 * sid),
                         "AMT": np.nan, "RF": rf})
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
                                         {"ID": "ID", "TIME": "TIME", "DV": "DV", "AMT": "AMT"}})
    return state, ctx


def test_tool_is_registered():
    assert default_registry().get("run_vpc").agent == "modeler"


def test_default_output_has_no_stratified_block(loaded):
    state, ctx = loaded
    payload = run_vpc(state, ctx, {}).writes["vpc_results"]
    assert payload["status"] == "ok"
    assert "stratified" not in payload           # default path unchanged
    assert payload["pcvpc"]["correction"] == "pred"


def test_bad_stratum_lists_available_covariates(loaded):
    state, ctx = loaded
    payload = run_vpc(state, ctx, {"stratify_by": "NOPE"}).writes["vpc_results"]
    strat = payload["stratified"]
    assert strat["status"] == "bad_stratum"
    assert "DOSE" in strat["available"]
    assert "RF" in strat["available"]            # the real covariate is offered


def test_stratify_by_dose_is_json_safe(loaded):
    state, ctx = loaded
    res = run_vpc(state, ctx, {"stratify_by": "DOSE"})
    payload = res.writes["vpc_results"]
    strat = payload["stratified"]
    assert strat["status"] == "ok"
    assert {st["label"] for st in strat["strata"]} == {"100", "300"}
    json.dumps(payload)                          # no numpy scalars leak through


def test_dose_normalize_pools_into_one_stratum(loaded):
    state, ctx = loaded
    payload = run_vpc(state, ctx, {"dose_normalize": True}).writes["vpc_results"]
    strat = payload["stratified"]
    assert strat["status"] == "ok"
    assert strat["correction"] == "dose"
    assert [st["label"] for st in strat["strata"]] == ["All"]   # pooled, normalized
