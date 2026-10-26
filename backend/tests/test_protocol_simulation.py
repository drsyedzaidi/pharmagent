"""Tests for the governed ``protocol_simulation`` workflow.

Covers:
  - Unit tests for ``load_approved_fit``, ``propose_simulation_plan``,
    ``simulate_trial_pta``, ``simulate_special_populations``, and
    ``propagate_parameter_uncertainty`` (bootstrap, SIR, profile).
  - Triggering and non-triggering fixtures for all six protocol-simulation
    adversarial-review checks in ``app.compute.adversarial``.
  - Determinism test: two runs from the same approved plan produce identical
    PTA tables.
  - Chat routing for PTA, target attainment, dose selection, trial simulation,
    and regimen comparison (precondition explanation vs. approved-fit execution).
  - Full workflow integration test through all human gates to the rendered
    report section, verifying that every reported number matches ``PharmState``.
"""
from __future__ import annotations

import copy
import itertools
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from docx import Document

from app.agents.supervisor import Supervisor
from app.compute import adversarial
from app.compute.protocol_sim import build_simulation_plan
from app.core.llm import MockLLM
from app.core.orchestrator import Orchestrator
from app.core.pharmstate import PharmState, apply_writes
from app.core.store import SessionStore
from app.tools.base import ToolContext
from app.tools.protocol_sim_tools import (
    load_approved_fit,
    propagate_parameter_uncertainty,
    propose_simulation_plan,
    simulate_special_populations,
    simulate_trial_pta,
)
from app.tools.report_tools import generate_report
from app.workflows import get_workflow

# ── Helpers & Fixtures ───────────────────────────────────────────────────────

def _approved_nlme_state(with_maturation_range: bool = False) -> PharmState:
    nlme: dict[str, Any] = {
        "status": "ok",
        "converged": True,
        "fit_id": "fit_cov_1cmt_01",
        "model_key": "1cmt_oral",
        "label": "1-compartment oral",
        "method": "focei",
        "ofv": 112.4,
        "theta": {"ka": 1.15, "CL": 4.8, "V": 38.0},
        "theta_rse_pct": {"ka": 12.0, "CL": 8.5, "V": 9.2},
        "omega_cv_pct": {"CL": 24.0, "V": 18.0},
        "iiv_params": ["CL", "V"],
        "sigma": {"prop": 0.12, "add": 0.0},
        "covariate_effects": [],
        "covariate_ranges": {
            "wt_kg": [48.0, 105.0],
            "age_yr": [20.0, 75.0],
            "crcl_ml_min": [45.0, 125.0],
        },
    }
    if with_maturation_range:
        nlme["maturation_data_range"] = {
            "pma_weeks": [60.0, 900.0],
            "age_yr": [0.5, 18.0],
        }
    rev = {
        "goal": adversarial.DEFAULT_GOAL,
        "goal_met": True,
        "status": "GOAL MET",
        "findings": [],
        "counts": {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0},
        "n_findings": 0,
    }
    return PharmState(nlme_results=nlme, review_results=rev)


def _complete_clean_protocol_sim_state(tmp_path: Path) -> PharmState:
    ctx = ToolContext(data_dir=str(tmp_path))
    st = _approved_nlme_state()
    r1 = load_approved_fit(st, ctx, {})
    st = apply_writes(st, "modeler", r1.writes)
    r2 = propose_simulation_plan(st, ctx, {"n_subjects": 40, "replicates": 5, "seed": 123})
    st = apply_writes(st, "simulator", r2.writes)
    r3 = simulate_trial_pta(st, ctx, {})
    st = apply_writes(st, "simulator", r3.writes)
    r4 = simulate_special_populations(st, ctx, {})
    st = apply_writes(st, "simulator", r4.writes)
    r5 = propagate_parameter_uncertainty(st, ctx, {"n_draws": 10})
    st = apply_writes(st, "simulator", r5.writes)
    return st


# ── 1. Unit Tests for New Tools ──────────────────────────────────────────────

def test_load_approved_fit_requires_converged_nlme_and_approved_review(tmp_path: Path):
    ctx = ToolContext(data_dir=str(tmp_path))
    # Missing NLME fit
    with pytest.raises(ValueError, match="converged NLME fit"):
        load_approved_fit(PharmState(), ctx, {})

    # Non-converged NLME fit
    st_unconv = PharmState(
        nlme_results={"status": "ok", "converged": False},
        review_results={"goal_met": True},
    )
    with pytest.raises(ValueError, match="converged NLME fit"):
        load_approved_fit(st_unconv, ctx, {})

    # Unapproved adversarial review
    st_unrev = PharmState(
        nlme_results={"status": "ok", "converged": True, "theta": {"CL": 5.0, "V": 40.0, "ka": 1.0}},
        review_results={"goal_met": False},
    )
    with pytest.raises(ValueError, match="approved adversarial review"):
        load_approved_fit(st_unrev, ctx, {})

    # Valid state succeeds and writes source_fit + provenance
    st_ok = _approved_nlme_state()
    res = load_approved_fit(st_ok, ctx, {})
    st_ok = apply_writes(st_ok, "modeler", res.writes)
    assert st_ok.protocol_sim["source_fit"]["fit_id"] == "fit_cov_1cmt_01"
    assert st_ok.protocol_sim["provenance"]["source_fit_id"] == "fit_cov_1cmt_01"
    assert st_ok.protocol_sim["provenance"]["review_hash"]


def test_propose_simulation_plan_writes_structured_plan_and_hash(tmp_path: Path):
    ctx = ToolContext(data_dir=str(tmp_path))
    st = _approved_nlme_state()
    st = apply_writes(st, "modeler", load_approved_fit(st, ctx, {}).writes)

    res = propose_simulation_plan(
        st,
        ctx,
        {
            "n_subjects": 60,
            "replicates": 8,
            "seed": 2026,
            "uncertainty_method": "sir",
        },
    )
    st = apply_writes(st, "simulator", res.writes)
    plan = st.protocol_sim["plan"]
    assert plan["n_subjects"] == 60
    assert plan["replicates"] == 8
    assert plan["seed"] == 2026
    assert plan["uncertainty_method"] == "sir"
    assert isinstance(plan["bands"], list) and len(plan["bands"]) >= 1
    assert isinstance(plan["regimens"], list) and len(plan["regimens"]) == 3
    assert plan["target_window"]["provenance"]["source"] == "protocol_specification"
    assert st.protocol_sim["provenance"]["plan_hash"] == res.result["plan_hash"]


def test_simulate_trial_pta_computes_mc_se_and_recommendations(tmp_path: Path):
    ctx = ToolContext(data_dir=str(tmp_path))
    st = _approved_nlme_state()
    st = apply_writes(st, "modeler", load_approved_fit(st, ctx, {}).writes)
    st = apply_writes(
        st,
        "simulator",
        propose_simulation_plan(st, ctx, {"n_subjects": 50, "replicates": 6, "seed": 77}).writes,
    )
    res = simulate_trial_pta(st, ctx, {})
    st = apply_writes(st, "simulator", res.writes)
    pta = st.protocol_sim["pta"]
    assert pta["status"] == "ok"
    assert pta["seed"] == 77
    assert len(pta["rows"]) == 3
    for row in pta["rows"]:
        assert 0.0 <= row["pta"] <= 1.0
        assert row["mc_se"] is not None and row["mc_se"] >= 0.0
        assert row["replicates"] == 6
    assert len(pta["recommendations"]) == 1
    assert pta["recommendations"][0]["recommended_regimen"] is not None


def test_simulate_special_populations_runs_only_plan_bands_with_plan_seed(tmp_path: Path):
    ctx = ToolContext(data_dir=str(tmp_path))
    st = _approved_nlme_state()
    st = apply_writes(st, "modeler", load_approved_fit(st, ctx, {}).writes)
    custom_bands = [
        {"name": "mild_renal_adult", "wt_kg": [55.0, 90.0], "age_yr": [30.0, 70.0], "crcl_ml_min": [60.0, 89.0]},
        {"name": "normal_renal_adult", "wt_kg": [55.0, 90.0], "age_yr": [25.0, 65.0], "crcl_ml_min": [90.0, 120.0]},
    ]
    st = apply_writes(
        st,
        "simulator",
        propose_simulation_plan(st, ctx, {"bands": custom_bands, "n_subjects": 40, "replicates": 4, "seed": 99}).writes,
    )
    res = simulate_special_populations(st, ctx, {})
    st = apply_writes(st, "simulator", res.writes)
    sp = st.protocol_sim["special_populations"]
    assert sp["status"] == "ok"
    assert sp["seed"] == 99
    assert sp["bands_simulated"] == ["mild_renal_adult", "normal_renal_adult"]
    assert [s["label"] for s in sp["strata"]] == ["mild_renal_adult", "normal_renal_adult"]


def _real_engine_result(method: str, theta: dict[str, float]) -> dict[str, Any]:
    """A result shaped exactly as the real engine writes it to state
    (sir.py: sir_median/sir_lo/sir_hi; bootstrap.py: boot_median/boot_lo/boot_hi;
    profile.py: estimate/profile_lo/profile_hi)."""
    keys = {
        "sir": ("sir_median", "sir_lo", "sir_hi"),
        "bootstrap": ("boot_median", "boot_lo", "boot_hi"),
        "profile": ("estimate", "profile_lo", "profile_hi"),
    }[method]
    return {
        "status": "ok",
        "parameters": [
            {"parameter": name, "estimate": est,
             keys[0]: est, keys[1]: round(est * 0.8, 4), keys[2]: round(est * 1.3, 4)}
            for name, est in theta.items()
        ],
    }


def _planned_state(tmp_path: Path, plan_args: dict[str, Any]) -> tuple[PharmState, ToolContext]:
    ctx = ToolContext(data_dir=str(tmp_path))
    st = _approved_nlme_state()
    st = apply_writes(st, "modeler", load_approved_fit(st, ctx, {}).writes)
    st = apply_writes(st, "simulator", propose_simulation_plan(st, ctx, plan_args).writes)
    st = apply_writes(st, "simulator", simulate_trial_pta(st, ctx, {}).writes)
    return st, ctx


@pytest.mark.parametrize("method", ["bootstrap", "sir", "profile"])
def test_propagate_without_a_real_engine_result_is_labelled_asymptotic_not_the_requested_method(
    tmp_path: Path, method: str,
):
    st, ctx = _planned_state(
        tmp_path, {"n_subjects": 30, "replicates": 4, "seed": 55, "uncertainty_method": method})
    res = propagate_parameter_uncertainty(st, ctx, {"n_draws": 8})
    st = apply_writes(st, "simulator", res.writes)

    unc = st.protocol_sim["uncertainty"]
    assert unc["status"] == "ok"
    assert unc["method"] == "asymptotic_lognormal"
    assert unc["requested_method"] == method
    assert unc["approximation"] is True
    assert unc["n_draws"] == 8
    # The summary a human reads at the gate names both the approximation and what was asked for.
    assert "asymptotic" in res.summary and method in res.summary
    assert res.action == "propagate_parameter_uncertainty(asymptotic_lognormal)"
    for row in st.protocol_sim["pta"]["rows"]:
        assert row["uncertainty_method"] == "asymptotic_lognormal"
        assert 0.0 <= row["pta_ci_lo"] <= row["pta_ci_hi"] <= 1.0
    for rec in st.protocol_sim["recommendations"]:
        if rec.get("recommended_regimen"):
            assert rec["uncertainty_method"] == "asymptotic_lognormal"


@pytest.mark.parametrize("method", ["bootstrap", "sir", "profile"])
def test_propagate_reuses_the_real_engine_result_stored_in_state(tmp_path: Path, method: str):
    ctx = ToolContext(data_dir=str(tmp_path))
    st = _approved_nlme_state()
    st = apply_writes(st, "modeler", load_approved_fit(st, ctx, {}).writes)
    st = apply_writes(st, "simulator",
                      {f"{method}_results": _real_engine_result(method, st.nlme_results["theta"])})
    st = apply_writes(st, "simulator",
                      propose_simulation_plan(st, ctx, {"n_subjects": 30, "replicates": 4, "seed": 55}).writes)
    # The proposed plan defaults to the engine whose result is actually present.
    assert st.protocol_sim["plan"]["uncertainty_method"] == method
    assert st.protocol_sim["plan"]["uncertainty_methods_available"] == [method]
    st = apply_writes(st, "simulator", simulate_trial_pta(st, ctx, {}).writes)
    res = propagate_parameter_uncertainty(st, ctx, {"n_draws": 8})
    st = apply_writes(st, "simulator", res.writes)

    unc = st.protocol_sim["uncertainty"]
    assert unc["status"] == "ok"
    assert unc["method"] == method
    assert unc["requested_method"] == method
    assert unc["approximation"] is False
    assert unc["engine_summary"]["interval_source"] == f"{method}_results"
    assert sorted(unc["engine_summary"]["parameters_drawn"]) == sorted(st.nlme_results["theta"])
    assert res.action == f"propagate_parameter_uncertainty({method})"
    for row in st.protocol_sim["pta"]["rows"]:
        assert row["uncertainty_method"] == method


def test_propagate_fails_closed_when_the_fit_has_no_rse_and_no_engine_result(tmp_path: Path):
    ctx = ToolContext(data_dir=str(tmp_path))
    st = _approved_nlme_state()
    nl = dict(st.nlme_results)
    nl.pop("theta_rse_pct")
    st = apply_writes(st, "modeler", {"nlme_results": nl})
    st = apply_writes(st, "modeler", load_approved_fit(st, ctx, {}).writes)
    fit = st.protocol_sim["source_fit"]
    assert fit["theta_rse_pct"] == {}                       # no silent 15 % default
    assert set(nl["theta"]) <= set(fit["theta_rse_missing"])
    st = apply_writes(st, "simulator",
                      propose_simulation_plan(st, ctx, {"n_subjects": 30, "replicates": 4, "seed": 55}).writes)
    st = apply_writes(st, "simulator", simulate_trial_pta(st, ctx, {}).writes)
    res = propagate_parameter_uncertainty(st, ctx, {"n_draws": 8})
    st = apply_writes(st, "simulator", res.writes)

    unc = st.protocol_sim["uncertainty"]
    assert unc["status"] == "unverifiable"
    assert res.result["status"] == "unverifiable"
    assert res.action == "propagate_parameter_uncertainty(unverifiable)"
    assert "RSE" in unc["reason"]
    for row in st.protocol_sim["pta"]["rows"]:
        assert row.get("pta_ci_lo") is None and row.get("pta_ci_hi") is None
    rev = adversarial.review(st.model_dump(), None, None)
    assert rev["goal_met"] is False
    f = next(f for f in rev["findings"] if f["id"] == "sim-uncertainty-omitted")
    assert f["severity"] == "HIGH"
    assert "RSE" in f["evidence"]


def test_review_check7_flags_an_approximation_used_in_place_of_the_approved_method(tmp_path: Path):
    st, ctx = _planned_state(
        tmp_path, {"n_subjects": 30, "replicates": 4, "seed": 55, "uncertainty_method": "sir"})
    st = apply_writes(st, "simulator", propagate_parameter_uncertainty(st, ctx, {"n_draws": 8}).writes)
    rev = adversarial.review(st.model_dump(), None, None)
    assert rev["goal_met"] is False
    f = next(f for f in rev["findings"] if f["id"] == "sim-uncertainty-method-deviation")
    assert f["severity"] == "HIGH"
    assert f["target"] == "protocol_sim.uncertainty"
    assert "sir" in f["evidence"] and "asymptotic_lognormal" in f["evidence"]


def test_review_check7_passes_when_the_plan_itself_requested_the_approximation(tmp_path: Path):
    st = _complete_clean_protocol_sim_state(tmp_path)
    assert st.protocol_sim["plan"]["uncertainty_method"] == "asymptotic_lognormal"
    rev = adversarial.review(st.model_dump(), None, None)
    assert not any(f["id"] == "sim-uncertainty-method-deviation" for f in rev["findings"])
    assert rev["goal_met"] is True


def test_build_simulation_plan_defaults_to_an_engine_result_present_else_asymptotic():
    fit = {"theta": {"CL": 4.8, "V": 38.0}, "covariate_ranges": {}, "route": "oral"}
    assert build_simulation_plan(fit, {})["uncertainty_method"] == "asymptotic_lognormal"
    assert build_simulation_plan(fit, {}, available_methods=["profile"])["uncertainty_method"] == "profile"
    assert build_simulation_plan(fit, {}, available_methods=["profile", "sir"])["uncertainty_method"] == "sir"
    assert build_simulation_plan(fit, {"uncertainty_method": "asymptotic_lognormal"},
                                 available_methods=["sir"])["uncertainty_method"] == "asymptotic_lognormal"
    with pytest.raises(ValueError, match="uncertainty_method"):
        build_simulation_plan(fit, {"uncertainty_method": "delta"})


def test_report_discloses_the_uncertainty_approximation_and_what_was_requested(tmp_path: Path):
    st, ctx = _planned_state(
        tmp_path, {"n_subjects": 30, "replicates": 4, "seed": 55, "uncertainty_method": "sir"})
    st = apply_writes(st, "simulator", propagate_parameter_uncertainty(st, ctx, {"n_draws": 8}).writes)
    res = generate_report(st, ctx, {"study_title": "Approximation disclosure"})
    sec = res.writes["report_sections"]["protocol_simulation"]
    assert sec["uncertainty_method"] == "asymptotic_lognormal"
    assert sec["uncertainty_requested"] == "sir"
    assert sec["uncertainty_approximation"] is True
    md = sec["rendered_markdown"]
    assert "asymptotic log-normal approximation" in md
    assert "requested: sir" in md
    assert "| uncertainty_method | asymptotic_lognormal |" in md
    assert "| uncertainty_requested | sir |" in md


# ── 2. Determinism Test ──────────────────────────────────────────────────────

def test_determinism_identical_pta_tables_from_same_approved_plan(tmp_path: Path):
    ctx = ToolContext(data_dir=str(tmp_path))
    base = _approved_nlme_state()
    base = apply_writes(base, "modeler", load_approved_fit(base, ctx, {}).writes)
    base = apply_writes(
        base,
        "simulator",
        propose_simulation_plan(base, ctx, {"n_subjects": 50, "replicates": 6, "seed": 314159}).writes,
    )

    st_a = apply_writes(base, "simulator", simulate_trial_pta(base, ctx, {}).writes)
    st_a = apply_writes(st_a, "simulator", propagate_parameter_uncertainty(st_a, ctx, {"n_draws": 10}).writes)

    st_b = apply_writes(base, "simulator", simulate_trial_pta(base, ctx, {}).writes)
    st_b = apply_writes(st_b, "simulator", propagate_parameter_uncertainty(st_b, ctx, {"n_draws": 10}).writes)

    assert st_a.protocol_sim["pta"]["rows"] == st_b.protocol_sim["pta"]["rows"]
    assert st_a.protocol_sim["uncertainty"]["intervals"] == st_b.protocol_sim["uncertainty"]["intervals"]
    assert st_a.protocol_sim["provenance"]["plan_hash"] == st_b.protocol_sim["provenance"]["plan_hash"]
    assert st_a.protocol_sim["provenance"]["seed"] == 314159


# ── 3. Six Adversarial Review Checks (Triggering & Non-Triggering) ───────────

def test_review_check1_target_window_provenance_triggers_on_missing_or_circular(tmp_path: Path):
    st = _complete_clean_protocol_sim_state(tmp_path)
    dump = st.model_dump()
    # Circular provenance
    dump["protocol_sim"]["plan"]["target_window"]["provenance"] = {
        "source": "this_simulation",
        "citation": "Derived from current simulation PTA",
    }
    rev = adversarial.review(dump, None, None)
    assert rev["goal_met"] is False
    finding = next(f for f in rev["findings"] if f["id"] == "sim-target-window-provenance")
    assert finding["severity"] == "HIGH"
    assert finding["target"] == "protocol_sim.plan.target_window"


def test_review_check1_target_window_provenance_passes_on_external_citation(tmp_path: Path):
    st = _complete_clean_protocol_sim_state(tmp_path)
    rev = adversarial.review(st.model_dump(), None, None)
    assert not any(f["id"] == "sim-target-window-provenance" for f in rev["findings"])
    assert rev["goal_met"] is True


def test_review_check2_pta_mc_se_triggers_when_mc_se_missing_or_fewer_replicates(tmp_path: Path):
    st = _complete_clean_protocol_sim_state(tmp_path)
    dump = st.model_dump()
    dump["protocol_sim"]["pta"]["rows"][0]["mc_se"] = None
    dump["protocol_sim"]["pta"]["rows"][1]["replicates"] = 2  # plan has 5
    rev = adversarial.review(dump, None, None)
    assert rev["goal_met"] is False
    mc_findings = [f for f in rev["findings"] if f["id"].startswith("sim-pta-mc-se")]
    assert len(mc_findings) >= 2
    assert all(f["severity"] == "HIGH" and f["target"] == "protocol_sim.pta" for f in mc_findings)


def test_review_check2_pta_mc_se_passes_when_mc_se_and_replicates_match_plan(tmp_path: Path):
    st = _complete_clean_protocol_sim_state(tmp_path)
    rev = adversarial.review(st.model_dump(), None, None)
    assert not any(f["id"].startswith("sim-pta-mc-se") for f in rev["findings"])
    assert rev["goal_met"] is True


def test_review_check3_covariate_extrapolation_triggers_outside_fit_range(tmp_path: Path):
    st = _complete_clean_protocol_sim_state(tmp_path)
    dump = st.model_dump()
    # Fit range is wt_kg=[48.0, 105.0], age_yr=[20.0, 75.0]; extrapolate to [25.0, 130.0] kg
    dump["protocol_sim"]["plan"]["bands"][0]["wt_kg"] = [25.0, 130.0]
    rev = adversarial.review(dump, None, None)
    assert rev["goal_met"] is False
    finding = next(f for f in rev["findings"] if f["id"].startswith("sim-covariate-extrapolation"))
    assert finding["severity"] == "HIGH"
    assert finding["target"] == "protocol_sim.plan.bands"
    assert "48" in finding["evidence"] and "105" in finding["evidence"]


def test_review_check3_covariate_extrapolation_passes_within_fit_range(tmp_path: Path):
    st = _complete_clean_protocol_sim_state(tmp_path)
    rev = adversarial.review(st.model_dump(), None, None)
    assert not any(f["id"].startswith("sim-covariate-extrapolation") for f in rev["findings"])
    assert rev["goal_met"] is True


def test_review_check4_limit_violation_triggers_when_recommended_regimen_fails_cmax_or_ctrough(tmp_path: Path):
    st = _complete_clean_protocol_sim_state(tmp_path)
    dump = st.model_dump()
    rec = dump["protocol_sim"]["recommendations"][0]
    rec_reg = rec["recommended_regimen"]
    for row in dump["protocol_sim"]["pta"]["rows"]:
        if row["regimen"] == rec_reg:
            row["peak_limit_failed"] = True
            row["cmax"]["limit_failed"] = True
    rev = adversarial.review(dump, None, None)
    assert rev["goal_met"] is False
    finding = next(f for f in rev["findings"] if f["id"].startswith("sim-limit-violation"))
    assert finding["severity"] == "HIGH"
    assert finding["target"] == "protocol_sim.recommendations"


def test_review_check4_limit_violation_passes_when_recommended_regimen_within_limits(tmp_path: Path):
    st = _complete_clean_protocol_sim_state(tmp_path)
    rev = adversarial.review(st.model_dump(), None, None)
    assert not any(f["id"].startswith("sim-limit-violation") for f in rev["findings"])
    assert rev["goal_met"] is True


def test_review_check5_uncertainty_omitted_triggers_when_recommendation_lacks_parameter_uncertainty(tmp_path: Path):
    st = _complete_clean_protocol_sim_state(tmp_path)
    dump = st.model_dump()
    dump["protocol_sim"]["uncertainty"] = None
    for row in dump["protocol_sim"]["pta"]["rows"]:
        row["pta_ci_lo"] = None
        row["pta_ci_hi"] = None
    for rec in dump["protocol_sim"]["recommendations"]:
        rec["pta_ci_lo"] = None
        rec["pta_ci_hi"] = None
    rev = adversarial.review(dump, None, None)
    assert rev["goal_met"] is False
    finding = next(f for f in rev["findings"] if f["id"] == "sim-uncertainty-omitted")
    assert finding["severity"] == "HIGH"
    assert finding["target"] == "protocol_sim.uncertainty"


def test_review_check5_uncertainty_omitted_passes_when_uncertainty_propagated(tmp_path: Path):
    st = _complete_clean_protocol_sim_state(tmp_path)
    rev = adversarial.review(st.model_dump(), None, None)
    assert not any(f["id"] == "sim-uncertainty-omitted" for f in rev["findings"])
    assert rev["goal_met"] is True


def test_review_check6_pediatric_maturation_triggers_when_range_missing_or_below_data_bound(tmp_path: Path):
    st = _complete_clean_protocol_sim_state(tmp_path)
    dump = st.model_dump()
    # Case A: Pediatric band present, but fit carries no maturation_data_range
    dump["protocol_sim"]["plan"]["bands"].append({
        "name": "infant_band",
        "wt_kg": [5.0, 10.0],
        "age_yr": [0.2, 1.0],
        "pma_weeks": [48.0, 92.0],
        "is_pediatric": True,
    })
    dump["protocol_sim"]["source_fit"]["covariate_ranges"]["wt_kg"] = [3.0, 110.0]
    dump["protocol_sim"]["source_fit"]["covariate_ranges"]["age_yr"] = [0.1, 75.0]
    rev_missing = adversarial.review(dump, None, None)
    assert rev_missing["goal_met"] is False
    f_missing = next(f for f in rev_missing["findings"] if f["id"].startswith("sim-pediatric-maturation-missing"))
    assert f_missing["severity"] == "HIGH"
    assert f_missing["target"] == "protocol_sim.source_fit.maturation_data_range"
    assert "does not carry maturation_data_range" in f_missing["evidence"]

    # Case B: Fit carries maturation_data_range [60, 900] weeks PMA, band starts at 48 weeks
    dump_below = copy.deepcopy(dump)
    dump_below["protocol_sim"]["source_fit"]["maturation_data_range"] = {
        "pma_weeks": [60.0, 900.0],
        "age_yr": [0.5, 18.0],
    }
    rev_below = adversarial.review(dump_below, None, None)
    assert rev_below["goal_met"] is False
    f_below = next(f for f in rev_below["findings"] if f["id"].startswith("sim-pediatric-maturation-extrapolation"))
    assert f_below["severity"] == "HIGH"
    assert f_below["target"] == "protocol_sim.source_fit.maturation_data_range"
    assert "60" in f_below["evidence"]


def test_review_check6_pediatric_maturation_passes_when_within_maturation_data_range(tmp_path: Path):
    st = _complete_clean_protocol_sim_state(tmp_path)
    dump = st.model_dump()
    dump["protocol_sim"]["source_fit"]["covariate_ranges"]["wt_kg"] = [8.0, 110.0]
    dump["protocol_sim"]["source_fit"]["covariate_ranges"]["age_yr"] = [1.0, 75.0]
    dump["protocol_sim"]["source_fit"]["maturation_data_range"] = {
        "pma_weeks": [60.0, 900.0],
        "age_yr": [0.5, 18.0],
    }
    dump["protocol_sim"]["plan"]["bands"].append({
        "name": "child_band",
        "wt_kg": [10.0, 25.0],
        "age_yr": [1.5, 6.0],
        "pma_weeks": [118.0, 352.0],
        "is_pediatric": True,
    })
    rev = adversarial.review(dump, None, None)
    assert not any(f["id"].startswith("sim-pediatric-maturation") for f in rev["findings"])
    assert rev["goal_met"] is True


# ── 4. Chat Routing & Precondition Behavior ──────────────────────────────────

@pytest.mark.parametrize(
    "prompt",
    [
        "Run PTA across dosing regimens",
        "Compute probability of target attainment for the cohort",
        "Help with dose selection for the phase 3 protocol",
        "Run a clinical trial simulation for regimen comparison",
        "Compare regimens using protocol simulation",
    ],
)
def test_chat_routing_pta_target_attainment_dose_selection_trial_sim_regimen_comparison(
    tmp_path: Path, prompt: str
):
    sup = Supervisor(MockLLM())
    agent, method = sup.route(prompt)
    assert agent == "simulator"
    assert method == "keyword"

    # Without an approved PopPK fit, chat explains that an approved PopPK fit is required first
    orch = Orchestrator(
        llm=MockLLM(),
        clock=lambda c=itertools.count(): f"t{next(c)}",
        store=SessionStore(":memory:"),
    )
    sid = orch.create_session().id
    reply_no_fit = orch.chat(sid, prompt)
    assert reply_no_fit["agent"] == "simulator"
    assert any("requires an approved PopPK fit first" in m for m in reply_no_fit["messages"])

    # With an approved PopPK fit in state, chat proposes the simulation plan
    sess = orch.get_session(sid)
    approved = _approved_nlme_state()
    sess.state = apply_writes(sess.state, "modeler", {"nlme_results": approved.nlme_results})
    sess.state = apply_writes(sess.state, "reviewer", {"review_results": approved.review_results})
    orch._persist(sess)

    reply_with_fit = orch.chat(sid, prompt)
    assert reply_with_fit["agent"] == "simulator"
    assert any(tc.get("tool") == "propose_simulation_plan" for tc in reply_with_fit["tool_calls"])
    assert reply_with_fit["state"]["protocol_sim"]["plan"] is not None


@pytest.mark.parametrize(
    ("prompt", "expected_agent"),
    [
        ("load dataset /x/pharmagent-protocol-simulation/data.csv", "data_manager"),
        (r"load dataset C:\data\protocol-simulation\pk.csv", "data_manager"),
        ("load dataset dose-selection_study.csv", "data_manager"),
        ("compute PTA for 100 mg bid", "simulator"),
        ("compare regimens 50 vs 100 mg", "simulator"),
    ],
)
def test_router_strips_paths_and_filenames_before_sim_anchor(prompt: str, expected_agent: str):
    sup = Supervisor(MockLLM())
    agent, method = sup.route(prompt)
    assert agent == expected_agent
    assert method == "keyword"


# ── 5. Report Graceful Degradation & End-to-End Integration ──────────────────

def test_report_section_renders_gracefully_when_optional_steps_skipped(tmp_path: Path):
    ctx = ToolContext(data_dir=str(tmp_path))
    st = _approved_nlme_state()
    st = apply_writes(st, "modeler", load_approved_fit(st, ctx, {}).writes)
    st = apply_writes(st, "simulator", propose_simulation_plan(st, ctx, {"n_subjects": 25, "replicates": 3}).writes)
    # Skip run_dose_sweep, simulate_special_populations, and propagate_parameter_uncertainty
    st = apply_writes(st, "simulator", simulate_trial_pta(st, ctx, {}).writes)

    rep = generate_report(st, ctx, {"title": "Partial Protocol Simulation Report"})
    st = apply_writes(st, "report", rep.writes)
    ps_sec = st.report_sections["protocol_simulation"]
    assert "run_dose_sweep" in ps_sec["skipped_steps"]
    assert "simulate_special_populations" in ps_sec["skipped_steps"]
    assert "propagate_parameter_uncertainty" in ps_sec["skipped_steps"]
    assert "Skipped step: simulate_special_populations" in ps_sec["rendered_markdown"]
    assert "Skipped step: propagate_parameter_uncertainty" in ps_sec["rendered_markdown"]


def test_full_protocol_simulation_workflow_end_to_end_with_gates_and_report(tmp_path: Path):
    wf = get_workflow("protocol_simulation")
    assert wf["requires"] == ["nlme_results.status == 'ok'"]
    assert [s["tool"] for s in wf["steps"]] == [
        "load_approved_fit",
        "propose_simulation_plan",
        "run_dose_sweep",
        "simulate_trial_pta",
        "simulate_special_populations",
        "propagate_parameter_uncertainty",
        "adversarial_review",
        "generate_report",
    ]

    orch = Orchestrator(
        llm=MockLLM(),
        clock=lambda c=itertools.count(): f"2026-10-23T12:00:{next(c):02d}Z",
        store=SessionStore(":memory:"),
    )
    sess = orch.create_session(owner="pharmacometrician_test")
    sid = sess.id
    sess.ctx = ToolContext(data_dir=str(tmp_path))

    # Load cov_pk.csv and seed the approved PopPK fit precondition
    sample_csv = str(Path(__file__).parent.parent / "sample_data" / "cov_pk.csv")
    orch.run_tool(sid, "load_dataset", "data_manager", {"path": sample_csv}, actor="pharmacometrician_test")
    approved = _approved_nlme_state()
    sess = orch.get_session(sid)
    sess.state = apply_writes(sess.state, "modeler", {"nlme_results": approved.nlme_results})
    sess.state = apply_writes(sess.state, "reviewer", {"review_results": approved.review_results})
    orch._persist(sess)

    # Leg 1: start_workflow runs load_approved_fit -> propose_simulation_plan and pauses at Gate 1
    leg1 = orch.start_workflow(sid, "protocol_simulation", actor="pharmacometrician_test")
    assert leg1["status"] == "awaiting_review"
    assert leg1["review"]["after_step"] == 1
    assert [s["tool"] for s in leg1["executed"]] == ["load_approved_fit", "propose_simulation_plan"]

    # Human edits the plan at Gate 1 (adjusts seed and replicates) before approving
    sess = orch.get_session(sid)
    edited_ps = copy.deepcopy(dict(sess.state.protocol_sim))
    edited_ps["plan"]["seed"] = 20261023
    edited_ps["plan"]["n_subjects"] = 50
    edited_ps["plan"]["replicates"] = 6
    sess.state = apply_writes(sess.state, "simulator", {"protocol_sim": edited_ps})
    orch._persist(sess)

    # Leg 2: resume_workflow runs run_dose_sweep -> simulate_trial_pta -> simulate_special_populations (Gate 2)
    leg2 = orch.resume_workflow(
        sid,
        approve=True,
        actor="pharmacometrician_test",
        reason="Simulation plan reviewed and approved with seed=20261023",
    )
    assert leg2["status"] == "awaiting_review"
    assert leg2["review"]["after_step"] == 4
    assert [s["tool"] for s in leg2["executed"]] == [
        "run_dose_sweep",
        "simulate_trial_pta",
        "simulate_special_populations",
    ]

    # Verify next leg requires a background job (propagate_parameter_uncertainty is expensive=True)
    assert orch.workflow_needs_job(wf, from_step=5) is True

    # Leg 3: resume_workflow (as background job with allow_expensive=True) runs
    # propagate_parameter_uncertainty -> adversarial_review (Gate 3)
    leg3 = orch.resume_workflow(
        sid,
        approve=True,
        actor="pharmacometrician_test",
        reason="Approve special-population results and launch SIR uncertainty job",
        allow_expensive=True,
    )
    assert leg3["status"] == "awaiting_review"
    assert leg3["review"]["after_step"] == 6
    assert [s["tool"] for s in leg3["executed"]] == [
        "propagate_parameter_uncertainty",
        "adversarial_review",
    ]
    assert leg3["state"]["review_results"]["goal_met"] is True
    assert leg3["state"]["review_results"]["counts"]["HIGH"] == 0
    assert leg3["state"]["review_results"]["counts"]["CRITICAL"] == 0

    # Leg 4: resume_workflow approves adversarial review gate -> runs generate_report -> complete
    leg4 = orch.resume_workflow(
        sid,
        approve=True,
        actor="pharmacometrician_test",
        reason="Zero HIGH/CRITICAL simulation findings; approve for report",
    )
    assert leg4["status"] == "complete"
    assert [s["tool"] for s in leg4["executed"]] == ["generate_report"]
    assert leg4["audit_ok"] is True

    final_state = leg4["state"]
    ps = final_state["protocol_sim"]
    rep_sec = final_state["report_sections"]["protocol_simulation"]

    # Every number in the report section must match PharmState
    assert rep_sec["source_fit_id"] == ps["provenance"]["source_fit_id"]
    assert rep_sec["review_hash"] == ps["provenance"]["review_hash"]
    assert rep_sec["plan_hash"] == ps["provenance"]["plan_hash"]
    assert rep_sec["seed"] == 20261023
    assert rep_sec["n_pta_rows"] == len(ps["pta"]["rows"])
    for row in ps["pta"]["rows"]:
        assert f"{row['pta']:.4f}" in rep_sec["rendered_markdown"]
        assert f"{row['mc_se']:.4f}" in rep_sec["rendered_markdown"]
        assert f"{row['pta_ci_lo']:.4f}–{row['pta_ci_hi']:.4f}" in rep_sec["rendered_markdown"]

    # Verify DOCX file on disk contains the Protocol simulation heading and provenance
    doc = Document(final_state["report_path"])
    full_text = "\n".join(p.text for p in doc.paragraphs)
    assert "Protocol simulation" in full_text
    assert "seed = 20261023" in full_text


# ── Fix 2: covariate ranges are observed, never invented ─────────────────────

def _nlme_without_ranges() -> dict[str, Any]:
    nl = dict(_approved_nlme_state().nlme_results)
    nl.pop("covariate_ranges")
    return nl


def test_load_approved_fit_does_not_invent_covariate_ranges(tmp_path: Path):
    ctx = ToolContext(data_dir=str(tmp_path))
    st = PharmState(nlme_results=_nlme_without_ranges(),
                    review_results=_approved_nlme_state().review_results)
    st = apply_writes(st, "modeler", load_approved_fit(st, ctx, {}).writes)
    fit = st.protocol_sim["source_fit"]
    assert fit["covariate_ranges"] == {}
    assert fit["covariate_range_sources"] == {}


def test_load_approved_fit_records_where_each_covariate_range_came_from(tmp_path: Path):
    ctx = ToolContext(data_dir=str(tmp_path))
    ctx.dataset_store["ds1"] = pd.DataFrame({
        "ID": [1, 1, 2, 2], "TIME": [0, 2, 0, 2], "DV": [0, 1.0, 0, 1.2],
        "AGE": [31.0, 31.0, 64.0, 64.0], "CRCL": [88.0, 88.0, 52.0, 52.0],
    })
    base = _approved_nlme_state()
    # Dataset only: AGE and CRCL observed, no weight column -> no wt_kg range at all.
    st = PharmState(dataset_id="ds1", nlme_results=_nlme_without_ranges(),
                    review_results=base.review_results)
    st = apply_writes(st, "modeler", load_approved_fit(st, ctx, {}).writes)
    fit = st.protocol_sim["source_fit"]
    assert fit["covariate_ranges"] == {"age_yr": [31.0, 64.0], "crcl_ml_min": [52.0, 88.0]}
    assert fit["covariate_range_sources"] == {"age_yr": "dataset", "crcl_ml_min": "dataset"}
    # Ranges carried by the fit win over the dataset and are labelled as the fit's.
    st2 = PharmState(dataset_id="ds1", nlme_results=base.nlme_results,
                     review_results=base.review_results)
    st2 = apply_writes(st2, "modeler", load_approved_fit(st2, ctx, {}).writes)
    fit2 = st2.protocol_sim["source_fit"]
    assert fit2["covariate_ranges"]["age_yr"] == [20.0, 75.0]
    assert fit2["covariate_range_sources"] == {"wt_kg": "fit", "age_yr": "fit", "crcl_ml_min": "fit"}


def test_default_band_only_spans_covariates_the_fit_actually_observed(tmp_path: Path):
    ctx = ToolContext(data_dir=str(tmp_path))
    st = PharmState(nlme_results=_nlme_without_ranges(),
                    review_results=_approved_nlme_state().review_results)
    st = apply_writes(st, "modeler", load_approved_fit(st, ctx, {}).writes)
    st = apply_writes(st, "simulator", propose_simulation_plan(
        st, ctx, {"n_subjects": 20, "replicates": 3, "seed": 7}).writes)
    plan = st.protocol_sim["plan"]
    band = plan["bands"][0]
    assert band["name"] == "standard_adult"
    assert not any(k in band for k in ("wt_kg", "age_yr", "crcl_ml_min"))
    assert plan["covariates_without_observed_range"] == ["age_yr", "crcl_ml_min", "wt_kg"]
    assert plan["covariate_range_sources"] == {}
    # An honest default plan is not an extrapolation: the review raises no covariate finding.
    st = apply_writes(st, "simulator", simulate_trial_pta(st, ctx, {}).writes)
    st = apply_writes(st, "simulator", simulate_special_populations(st, ctx, {}).writes)
    st = apply_writes(st, "simulator", propagate_parameter_uncertainty(st, ctx, {"n_draws": 6}).writes)
    rev = adversarial.review(st.model_dump(), None, None)
    assert not any(f["id"].startswith("sim-covariate-") for f in rev["findings"]), rev["findings"]
    assert rev["goal_met"] is True


def test_review_check3_is_unverifiable_not_silent_when_the_fit_has_no_range_for_a_band_covariate(
    tmp_path: Path,
):
    ctx = ToolContext(data_dir=str(tmp_path))
    st = PharmState(nlme_results=_nlme_without_ranges(),
                    review_results=_approved_nlme_state().review_results)
    st = apply_writes(st, "modeler", load_approved_fit(st, ctx, {}).writes)
    st = apply_writes(st, "simulator", propose_simulation_plan(st, ctx, {
        "n_subjects": 20, "replicates": 3, "seed": 7,
        "bands": [{"name": "heavy", "wt_kg": [90.0, 130.0], "is_pediatric": False}],
    }).writes)
    st = apply_writes(st, "simulator", simulate_trial_pta(st, ctx, {}).writes)
    rev = adversarial.review(st.model_dump(), None, None)
    assert rev["goal_met"] is False
    f = next(f for f in rev["findings"] if f["id"] == "sim-covariate-range-unverifiable-heavy")
    assert f["severity"] == "HIGH"
    assert f["target"] == "protocol_sim.plan.bands"
    assert "wt_kg" in f["evidence"] and "no observed range" in f["evidence"]


def test_review_check3_also_covers_creatinine_clearance(tmp_path: Path):
    st = _complete_clean_protocol_sim_state(tmp_path)
    dump = st.model_dump()
    # Fit range is crcl_ml_min=[45.0, 125.0]; propose a band up to 200 mL/min.
    dump["protocol_sim"]["plan"]["bands"][0]["crcl_ml_min"] = [60.0, 200.0]
    rev = adversarial.review(dump, None, None)
    f = next(f for f in rev["findings"] if f["id"].startswith("sim-covariate-extrapolation"))
    assert f["severity"] == "HIGH"
    assert "crcl_ml_min" in f["evidence"] and "125" in f["evidence"]
