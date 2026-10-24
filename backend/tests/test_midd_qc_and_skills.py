"""Tests for MIDD-inspired enhancements and review blocker fixes:
1. NCA lambda_z_span_ratio computation, QC check (including missing/NaN span ratios),
   non-bypassable analyst accept/revise gate (review_lambda_z_fits), and report column.
2. PopPK (NLME) estimation-health QC (run_poppk_qc) — PASS / CONDITIONAL PASS / FAIL,
   fit-digest provenance binding (stale detection after a subsequent fit), blocking
   generate_report on absent/stale/FAIL PopPK QC, missing/NaN shrinkage/BSV/condition-number
   warnings, fit-configuration theta_bounds coverage, estimation-health scope framing,
   and strict threshold validation (including rejecting non-integral min_points=2.9).
3. Skill Integrity Linter (check_skills) on repository skills, file:///{skill_dir}/...
   single-pass resolution, and pre-read symlink/size containment enforcement.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from docx import Document

from app.agents.definitions import AGENTS
from app.compute.nca import Profile, _best_lambda_z, nca_subject, refit_lambda_z_manual
from app.compute.nca_ss import nca_ss_subject
from app.core.audit import AuditChain
from app.core.llm import MockLLM
from app.core.pharmstate import PharmState
from app.core.skill_linter import _MAX_SKILL_BYTES, DEFAULT_SKILLS_DIR, check_skills
from app.tools.base import ToolContext
from app.tools.builtins import default_registry
from app.tools.qc_tools import (
    POPPK_QC_SCOPE,
    evaluate_poppk_qc,
    is_poppk_qc_current,
    nlme_fit_digest,
    review_lambda_z_fits,
    run_poppk_qc,
    run_qc,
)
from app.tools.report_tools import generate_report

# ---------------------------------------------------------------------------
# 1. NCA lambda_z_span_ratio & non-bypassable analyst accept/revise gate
# ---------------------------------------------------------------------------


def test_lambda_z_span_ratio_monoexponential():
    """Exact monoexponential decay: t_half = 2 h, terminal window [2, 12] -> span = 10 h -> ratio = 5.0."""
    lam = math.log(2) / 2.0  # t_half = 2.0 h
    times = np.array([0.0, 1.0, 2.0, 4.0, 8.0, 12.0])
    concs = np.array([0.0, 100.0, 80.0, 40.0, 10.0, 2.5])
    lz = _best_lambda_z(times, concs, tmax=1.0)
    assert lz["lambda_z"] == pytest.approx(lam, rel=1e-5)
    assert lz["lambda_z_intervals"] == [2.0, 12.0]
    assert lz["lambda_z_span_ratio"] == pytest.approx((12.0 - 2.0) / 2.0, rel=1e-5)

    subj = nca_subject(Profile(subject="S1", time=times, conc=concs, dose=100.0))
    assert subj["lambda_z_span_ratio"] == pytest.approx(5.0, rel=1e-5)

    ss = nca_ss_subject("S1", times, concs, dose=100.0, tau=12.0)
    assert ss["lambda_z_span_ratio"] == pytest.approx(5.0, rel=1e-5)

    manual = refit_lambda_z_manual([2.0, 4.0, 8.0, 12.0], [80.0, 40.0, 10.0, 2.5])
    assert manual["span_ratio"] == pytest.approx(5.0, rel=1e-3)


def test_nca_subject_without_terminal_fit_sets_span_ratio_none():
    """When fewer than 3 post-Tmax points exist, lambda_z_span_ratio is None."""
    times = np.array([0.0, 1.0, 2.0])
    concs = np.array([0.0, 10.0, 8.0])
    subj = nca_subject(Profile(subject="S_short", time=times, conc=concs, dose=100.0))
    assert subj["lambda_z"] is None
    assert subj["lambda_z_span_ratio"] is None


def _good_nca_params() -> list[dict]:
    return [
        {
            "subject": i,
            "Cmax": 10.0,
            "Tmax": 1.0,
            "Tlast": 24.0,
            "CL_F": 5.0,
            "Vz_F": 50.0,
            "lambda_z_n_points": 4,
            "lambda_z_r2_adj": 0.99,
            "lambda_z_span_ratio": 2.5,
            "pct_AUC_extrap": 5.0,
        }
        for i in range(1, 7)
    ]


def test_run_qc_flags_short_missing_and_nan_lambda_z_span_ratio():
    """run_qc warns on < 2.0, None, or NaN lambda_z_span_ratio (never PASS)."""
    good_params = _good_nca_params()
    res_pass = run_qc(PharmState(nca_parameters=good_params), ToolContext(), {})
    assert res_pass.result["verdict"] == "PASS"
    span_check = next(
        c for c in res_pass.result["checks"] if c["check"] == "Lambda_z span ratio >= 2.0"
    )
    assert span_check["status"] == "PASS"
    assert res_pass.writes["nca_flagged_subjects"] == []
    assert res_pass.writes["lambda_z_review"]["status"] == "not_required"

    # Case A: short span (< 2.0)
    short_params = [dict(p) for p in good_params]
    short_params[0]["lambda_z_span_ratio"] = 1.4
    res_short = run_qc(PharmState(nca_parameters=short_params), ToolContext(), {})
    assert res_short.result["verdict"] == "CONDITIONAL PASS"
    span_check_s = next(
        c for c in res_short.result["checks"] if c["check"] == "Lambda_z span ratio >= 2.0"
    )
    assert span_check_s["status"] == "WARN"
    assert any("short lambda_z span ratio" in iss["issue"] for iss in res_short.result["issues"])

    # Case B: missing (None) span ratio must NOT count as PASS
    none_params = [dict(p) for p in good_params]
    none_params[1]["lambda_z_span_ratio"] = None
    res_none = run_qc(PharmState(nca_parameters=none_params), ToolContext(), {})
    assert res_none.result["verdict"] == "CONDITIONAL PASS"
    span_check_n = next(
        c for c in res_none.result["checks"] if c["check"] == "Lambda_z span ratio >= 2.0"
    )
    assert span_check_n["status"] == "WARN"
    assert "missing/NaN" in span_check_n["detail"]
    assert any("missing or NaN lambda_z span ratio" in iss["issue"] for iss in res_none.result["issues"])

    # Case C: NaN span ratio must NOT count as PASS
    nan_params = [dict(p) for p in good_params]
    nan_params[2]["lambda_z_span_ratio"] = float("nan")
    res_nan = run_qc(PharmState(nca_parameters=nan_params), ToolContext(), {})
    assert res_nan.result["verdict"] == "CONDITIONAL PASS"
    span_check_nan = next(
        c for c in res_nan.result["checks"] if c["check"] == "Lambda_z span ratio >= 2.0"
    )
    assert span_check_nan["status"] == "WARN"
    assert res_nan.writes["nca_flagged_subjects"][0]["id"] == 3


def test_flagged_lambda_z_analyst_gate_blocks_ordinary_generate_report(tmp_path):
    """Ordinary generate_report({}) calls are blocked while flagged lambda_z fits are unreviewed or revised."""
    reg = default_registry()
    audit = AuditChain()
    ctx = ToolContext(data_dir=str(tmp_path))

    params = _good_nca_params()
    params[0]["lambda_z_span_ratio"] = 1.35  # Subject 1 flagged
    state = PharmState(
        session_id="s-lz-gate",
        dataset_metadata={"dataset_id": "oral_pk", "n_records": 36, "n_subjects": 6, "dose_levels": [100]},
        nca_parameters=params,
    )

    # 1. Even before run_qc is called, ordinary generate_report({}) is blocked if nca_parameters has flagged subjects
    with pytest.raises(ValueError, match="awaiting analyst decision"):
        generate_report(state, ctx, {})

    # 2. After run_qc, status is awaiting_analyst_decision -> ordinary generate_report({}) is blocked
    state, qc_res = reg.execute(
        "run_qc", state=state, ctx=ctx, args={}, audit=audit, timestamp="t1", caller_agent="qc"
    )
    assert state.lambda_z_review["status"] == "awaiting_analyst_decision"
    assert "analyst decision required" in qc_res.summary
    with pytest.raises(ValueError, match="awaiting analyst decision"):
        generate_report(state, ctx, {})

    # 3. Analyst chooses REVISE -> ordinary generate_report({}) is blocked
    state, _ = reg.execute(
        "review_lambda_z_fits",
        state=state,
        ctx=ctx,
        args={"decision": "revise", "notes": "Exclude t=4h point for subject 1"},
        audit=audit,
        timestamp="t2",
        caller_agent="qc",
    )
    assert state.lambda_z_review["status"] == "revision_requested"
    with pytest.raises(ValueError, match="revision_requested"):
        generate_report(state, ctx, {})

    # 4. Analyst chooses ACCEPT -> ordinary generate_report({}) succeeds
    state, _ = reg.execute(
        "review_lambda_z_fits",
        state=state,
        ctx=ctx,
        args={"decision": "accept", "notes": "Retain subject 1 with span ratio caveat"},
        audit=audit,
        timestamp="t3",
        caller_agent="qc",
    )
    assert state.lambda_z_review["status"] == "accepted"
    state, rep_res = reg.execute(
        "generate_report",
        state=state,
        ctx=ctx,
        args={},
        audit=audit,
        timestamp="t4",
        caller_agent="report",
    )
    assert rep_res.result["sections"]["lambda_z_review_status"] == "accepted"

    # 5. If nca_parameters are modified after review_lambda_z_fits, the review is stale and blocks generate_report({})
    mutated_params = [dict(p) for p in params]
    mutated_params[0]["lambda_z_span_ratio"] = 1.10
    stale_state = state.model_copy(update={"nca_parameters": mutated_params})
    with pytest.raises(ValueError, match="NCA parameters changed after lambda_z review"):
        generate_report(stale_state, ctx, {})

    # 6. Invalid decision raises ValueError
    with pytest.raises(ValueError, match="must be 'accept' or 'revise'"):
        review_lambda_z_fits(state, ctx, {"decision": "ignore"})


# ---------------------------------------------------------------------------
# 2. PopPK (NLME) estimation-health QC, provenance binding & blocking gates
# ---------------------------------------------------------------------------


def _healthy_nlme() -> dict:
    return {
        "status": "ok",
        "method": "FOCE-I",
        "model_key": "oral_1cmt",
        "label": "1-cmt oral (linear)",
        "iiv_params": ["CL", "V"],
        "error_model": "proportional",
        "theta": {"CL": 5.2, "V": 48.0, "KA": 1.1},
        "theta_bounds": {
            "CL": [5.0e-7, 5.0e7],
            "V": [50.0e-7, 50.0e7],
            "KA": [1.0e-7, 1.0e7],
        },
        "theta_rse_pct": {"CL": 8.5, "V": 10.2, "KA": 18.0},
        "omega_cv_pct": {"CL": 28.0, "V": 22.0},
        "omega_rse_pct": {"CL": 19.0, "V": 24.0},
        "sigma": {"prop": 0.15, "add": None},
        "sigma_rse_pct": {"prop": 12.0, "add": None},
        "covariate_effects": [],
        "ofv": 312.45,
        "condition_number": 14.2,
        "cov_note": "",
        "shrinkage_pct": {"CL": 9.5, "V": 14.0},
        "n_subjects": 12,
        "n_obs": 96,
        "n_blq": 0,
        "converged": True,
    }


def test_run_poppk_qc_passes_healthy_fit_and_records_provenance_and_scope():
    """run_poppk_qc records poppk_qc_nlme_digest, estimation_health_only scope, and check provenance."""
    nl = _healthy_nlme()
    state = PharmState(nlme_results=nl)
    res = run_poppk_qc(state, ToolContext(), {})
    assert res.result["verdict"] == "PASS"
    assert res.result["scope"] == POPPK_QC_SCOPE
    assert "does not establish model or regulatory acceptance" in res.result["qualification_note"]
    assert res.writes["poppk_qc_verdict"] == "PASS"
    assert res.writes["poppk_qc_nlme_digest"] == nlme_fit_digest(nl)
    assert res.writes["poppk_qc_issues"] == []

    checks_by_name = {c["check"]: c for c in res.result["checks"]}
    assert {k: v["status"] for k, v in checks_by_name.items()} == {
        "ofv_finite": "PASS",
        "covariance_step": "PASS",
        "structural_se_present": "PASS",
        "theta_not_on_bound": "PASS",
        "structural_rse": "PASS",
        "eta_shrinkage": "PASS",
        "bsv_range": "PASS",
        "condition_number": "PASS",
        "residual_error_positive": "PASS",
    }
    assert checks_by_name["condition_number"]["source"] == "pharmagent_extension"
    for upstream_name in (
        "ofv_finite",
        "covariance_step",
        "structural_se_present",
        "theta_not_on_bound",
        "structural_rse",
        "eta_shrinkage",
        "bsv_range",
        "residual_error_positive",
    ):
        assert checks_by_name[upstream_name]["source"] == "ai_assisted_midd"


def test_poppk_qc_staleness_and_report_blocking(tmp_path):
    """PopPK QC absent, FAIL, or stale after a subsequent NLME fit blocks generate_report."""
    reg = default_registry()
    audit = AuditChain()
    ctx = ToolContext(data_dir=str(tmp_path))

    nl1 = _healthy_nlme()
    state = PharmState(
        session_id="s-poppk-stale",
        dataset_metadata={"dataset_id": "oral_pk", "n_records": 96, "n_subjects": 12, "dose_levels": [100]},
        nlme_results=nl1,
    )

    # 1. Absent PopPK QC blocks generate_report when nlme_results is present
    assert is_poppk_qc_current(state) is False
    with pytest.raises(ValueError, match="PopPK estimation QC .* has not been run"):
        generate_report(state, ctx, {})

    # 2. Run PopPK QC -> PASS -> generate_report succeeds
    state, _ = reg.execute(
        "run_poppk_qc", state=state, ctx=ctx, args={}, audit=audit, timestamp="t1", caller_agent="qc"
    )
    assert is_poppk_qc_current(state) is True
    state, _ = reg.execute(
        "generate_report", state=state, ctx=ctx, args={}, audit=audit, timestamp="t2", caller_agent="report"
    )

    # 3. Subsequent NLME fit (different model or updated OFV/theta) makes PopPK QC stale!
    nl2 = _healthy_nlme()
    nl2["model_key"] = "oral_2cmt"
    nl2["ofv"] = 298.10
    state_after_refit = state.model_copy(update={"nlme_results": nl2})
    assert is_poppk_qc_current(state_after_refit) is False
    summary_view = AGENTS["qc"]._state_summary(state_after_refit)
    assert summary_view["poppk_qc_verdict"] is None
    assert summary_view["poppk_qc_stale"] is True

    with pytest.raises(ValueError, match="PopPK estimation QC verdict is stale"):
        generate_report(state_after_refit, ctx, {})

    # 4. PopPK QC FAIL blocks generate_report
    nl_fail = _healthy_nlme()
    nl_fail["converged"] = False
    state_fail = state.model_copy(update={"nlme_results": nl_fail})
    state_fail, _ = reg.execute(
        "run_poppk_qc", state=state_fail, ctx=ctx, args={}, audit=audit, timestamp="t3", caller_agent="qc"
    )
    assert state_fail.poppk_qc_verdict == "FAIL"
    with pytest.raises(ValueError, match="PopPK estimation QC verdict is FAIL"):
        generate_report(state_fail, ctx, {})


def test_missing_or_nan_shrinkage_bsv_and_condition_number_warn_never_pass():
    """Missing or NaN shrinkage, BSV CV%, or condition_number always emit WARN rows (never omitted or PASS)."""
    # 1. Omitted dicts / None condition_number
    nl_empty_diag = _healthy_nlme()
    nl_empty_diag["shrinkage_pct"] = {}
    nl_empty_diag["omega_cv_pct"] = {}
    nl_empty_diag["condition_number"] = None
    out_empty = evaluate_poppk_qc(nl_empty_diag)
    assert out_empty["verdict"] == "CONDITIONAL PASS"
    by_name = {c["check"]: c for c in out_empty["checks"]}
    assert by_name["eta_shrinkage"]["status"] == "WARN"
    assert by_name["bsv_range"]["status"] == "WARN"
    assert by_name["condition_number"]["status"] == "WARN"

    # 2. NaN values in shrinkage, omega_cv_pct, and condition_number
    nl_nan_diag = _healthy_nlme()
    nl_nan_diag["shrinkage_pct"]["CL"] = float("nan")
    nl_nan_diag["omega_cv_pct"]["V"] = float("nan")
    nl_nan_diag["condition_number"] = float("nan")
    out_nan = evaluate_poppk_qc(nl_nan_diag)
    assert out_nan["verdict"] == "CONDITIONAL PASS"
    by_name_nan = {c["check"]: c for c in out_nan["checks"]}
    assert by_name_nan["eta_shrinkage"]["status"] == "WARN"
    assert "missing/NaN: CL" in by_name_nan["eta_shrinkage"]["detail"]
    assert by_name_nan["bsv_range"]["status"] == "WARN"
    assert "missing/NaN: V" in by_name_nan["bsv_range"]["detail"]
    assert by_name_nan["condition_number"]["status"] == "WARN"
    assert "missing or non-finite" in by_name_nan["condition_number"]["detail"]


def test_custom_qc_thresholds_and_non_integral_min_points_rejected():
    """Custom QC thresholds reject NaN, inf, booleans, negative values, and non-integral min_points=2.9."""
    nl = _healthy_nlme()
    state = PharmState(nlme_results=nl, nca_parameters=_good_nca_params())
    ctx = ToolContext()

    for bad_val in (float("nan"), float("inf"), float("-inf"), -10.0, 0.0, True, "not-a-num"):
        with pytest.raises(ValueError):
            evaluate_poppk_qc(nl, max_rse_pct=bad_val)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            run_poppk_qc(state, ctx, {"max_rse_pct": bad_val})
        with pytest.raises(ValueError):
            run_qc(state, ctx, {"min_span_ratio": bad_val})

    # Non-integral min_points (e.g. 2.9 or 3.0 float) must be rejected, never truncated
    for bad_pts in (2.9, 3.0, 1, 0, -3, True, "3", float("nan")):
        with pytest.raises(ValueError, match="min_points"):
            run_qc(state, ctx, {"min_points": bad_pts})

    with pytest.raises(ValueError, match="max_shrinkage_pct"):
        evaluate_poppk_qc(nl, max_shrinkage_pct=120.0)
    with pytest.raises(ValueError, match="min_bsv_cv_pct"):
        evaluate_poppk_qc(nl, min_bsv_cv_pct=50.0, max_bsv_cv_pct=20.0)
    with pytest.raises(ValueError, match="max_condition_number"):
        evaluate_poppk_qc(nl, max_condition_number=0.5)


def test_theta_not_on_bound_requires_complete_fit_configuration_bounds():
    """theta_bounds comes from the fit record and must cover all fitted parameters."""
    # 1. Complete fit-recorded theta_bounds with interior estimates -> PASS
    nl_ok = _healthy_nlme()
    out_ok = evaluate_poppk_qc(nl_ok)
    chk_ok = next(c for c in out_ok["checks"] if c["check"] == "theta_not_on_bound")
    assert chk_ok["status"] == "PASS"
    assert "all fitted parameters covered" in chk_ok["detail"]

    # 2. Partial theta_bounds in fit record (covers CL only, omits V and KA) -> FAIL
    nl_partial = _healthy_nlme()
    nl_partial["theta_bounds"] = {"CL": [0.1, 50.0]}
    out_partial = evaluate_poppk_qc(nl_partial)
    chk_partial = next(c for c in out_partial["checks"] if c["check"] == "theta_not_on_bound")
    assert chk_partial["status"] == "FAIL"
    assert "incomplete theta_bounds in fit configuration" in chk_partial["detail"]
    assert "V" in chk_partial["detail"] and "KA" in chk_partial["detail"]

    # 3. Complete theta_bounds where KA (1.1) is within tol = 1e-4 * max(1, |est|) of upper bound -> FAIL
    nl_at_hi = _healthy_nlme()
    nl_at_hi["theta_bounds"] = {
        "CL": [0.1, 50.0],
        "V": [1.0, 500.0],
        "KA": [0.05, 1.10005],
    }
    out_hi = evaluate_poppk_qc(nl_at_hi)
    chk_hi = next(c for c in out_hi["checks"] if c["check"] == "theta_not_on_bound")
    assert chk_hi["status"] == "FAIL"
    assert "on bound: KA" in chk_hi["detail"]

    # 4. Fit record without explicit theta_bounds derives bounds from REGISTRY[model_key].defaults
    nl_default_box = _healthy_nlme()
    nl_default_box.pop("theta_bounds")
    out_def = evaluate_poppk_qc(nl_default_box)
    chk_def = next(c for c in out_def["checks"] if c["check"] == "theta_not_on_bound")
    assert chk_def["status"] == "PASS"
    assert "model estimation box" in chk_def["detail"]

    # 5. evaluate_poppk_qc does NOT accept caller-supplied theta_bounds kwarg
    with pytest.raises(TypeError):
        evaluate_poppk_qc(nl_ok, theta_bounds={"CL": (0.1, 50.0)})  # type: ignore[call-arg]


def test_covariance_step_status_not_inferred_from_single_finite_rse():
    """covariance_step fails when cov_note indicates near-singular/failed Hessian even if some RSE is finite."""
    nl_near_sing = _healthy_nlme()
    nl_near_sing["theta_rse_pct"] = {"CL": 8.5, "V": 10.2, "KA": None}
    nl_near_sing["cov_note"] = (
        "information matrix near-singular — some standard errors are "
        "unavailable (model likely over-parameterized for this data)"
    )
    out_ns = evaluate_poppk_qc(nl_near_sing)
    assert out_ns["verdict"] == "FAIL"
    cov_chk = next(c for c in out_ns["checks"] if c["check"] == "covariance_step")
    se_chk = next(c for c in out_ns["checks"] if c["check"] == "structural_se_present")
    assert cov_chk["status"] == "FAIL"
    assert "near-singular" in cov_chk["detail"]
    assert se_chk["status"] == "FAIL"


def test_run_poppk_qc_registry_and_mock_llm_dispatch(tmp_path):
    """run_poppk_qc is registered on 'qc', writes to PharmState, and renders in generate_report."""
    reg = default_registry()
    audit = AuditChain()
    state = PharmState(
        session_id="s-poppk-qc",
        dataset_metadata={"dataset_id": "oral_pk", "n_records": 96, "n_subjects": 12, "dose_levels": [100]},
        nca_parameters=[
            {
                "subject": 1,
                "dose": 100,
                "Cmax": 12.3,
                "Tmax": 1.0,
                "AUC_last": 80.0,
                "AUC_inf": 85.0,
                "t_half": 4.2,
                "CL_F": 1.18,
                "Vz_F": 7.1,
                "lambda_z_r2_adj": 0.995,
                "lambda_z_n_points": 5,
                "lambda_z_span_ratio": 3.2,
                "pct_AUC_extrap": 5.9,
            }
        ],
        qc_verdict="PASS",
        qc_checklist=[{"check": "Lambda_z span ratio >= 2.0", "status": "PASS", "detail": "0 below"}],
        nlme_results=_healthy_nlme(),
    )

    llm = MockLLM()
    choice = llm.select_tool(
        "qc",
        "run poppk qc on the nlme fit",
        reg.for_agent("qc"),
        {"nlme_results": "present", "poppk_qc_verdict": None, "qc_verdict": "PASS"},
    )
    assert choice == {"name": "run_poppk_qc", "input": {}}

    ctx = ToolContext(data_dir=str(tmp_path))
    state, _ = reg.execute(
        "run_poppk_qc",
        state=state,
        ctx=ctx,
        args={},
        audit=audit,
        timestamp="t1",
        caller_agent="qc",
    )
    assert state.poppk_qc_verdict == "PASS"
    assert state.poppk_qc_nlme_digest == nlme_fit_digest(state.nlme_results)

    state, rep_res = reg.execute(
        "generate_report",
        state=state,
        ctx=ctx,
        args={},
        audit=audit,
        timestamp="t2",
        caller_agent="report",
    )
    assert rep_res.result["sections"]["poppk_qc_verdict"] == "PASS"
    doc = Document(state.report_path)
    all_text = "\n".join(p.text for p in doc.paragraphs)
    assert "PopPK estimation QC verdict: PASS" in all_text
    assert "numerical estimation & covariance health only" in all_text
    assert "[PASS] ofv_finite:" in all_text
    header_cells = [c.text for c in doc.tables[0].rows[0].cells]
    assert "Span ratio" in header_cells


# ---------------------------------------------------------------------------
# 3. Skill Integrity Linter (check_skills) & symlink/size containment
# ---------------------------------------------------------------------------


def test_repo_skills_pass_skill_linter():
    """Every authored skill in the repository's skills/ directory passes."""
    findings = check_skills(DEFAULT_SKILLS_DIR)
    assert findings, "expected at least one skill finding"
    bad = [f for f in findings if f.status != "pass"]
    assert bad == [], f"Unexpected skill linter findings: {bad}"


def test_skill_linter_rejects_symlinks_and_oversized_files_before_reading(tmp_path):
    """Symlinked SKILL.md, symlinked skill folders, symlinked refs, and oversized files fail containment without reading."""
    outside_secret = tmp_path / "secret.txt"
    outside_secret.write_text("---\nname: leak\ndescription: secret\n---\n", encoding="utf-8")

    skills_root = tmp_path / "skills"
    skills_root.mkdir()

    # 1. Symlinked SKILL.md pointing outside skills_root
    sym_md_dir = skills_root / "symlink-md"
    sym_md_dir.mkdir()
    (sym_md_dir / "SKILL.md").symlink_to(outside_secret)

    # 2. Symlinked skill folder pointing outside skills_root
    outside_dir = tmp_path / "outside-skill-dir"
    outside_dir.mkdir()
    (outside_dir / "SKILL.md").write_text("---\nname: sym-dir\ndescription: x\n---\n", encoding="utf-8")
    (skills_root / "sym-dir").symlink_to(outside_dir)

    # 3. Oversized SKILL.md (> _MAX_SKILL_BYTES)
    huge_dir = skills_root / "huge-skill"
    huge_dir.mkdir()
    (huge_dir / "SKILL.md").write_bytes(b"#" * (_MAX_SKILL_BYTES + 1024))

    findings = check_skills(skills_root)
    by_skill_check = {(f.skill, f.check): f for f in findings}

    assert by_skill_check[("symlink-md", "skill_containment")].status == "fail"
    assert "symlink" in by_skill_check[("symlink-md", "skill_containment")].detail
    # Frontmatter was NEVER parsed from the symlink target
    assert ("symlink-md", "frontmatter_present") not in by_skill_check

    assert by_skill_check[("sym-dir", "skill_containment")].status == "fail"
    assert "symlink" in by_skill_check[("sym-dir", "skill_containment")].detail

    assert by_skill_check[("huge-skill", "skill_containment")].status == "fail"
    assert "exceeds" in by_skill_check[("huge-skill", "skill_containment")].detail


def test_skill_linter_file_skill_dir_uri_not_double_parsed(tmp_path):
    """[label](file:///{skill_dir}/references/guide.md) and inline code pass cleanly."""
    skills_root = tmp_path / "skills"
    good_dir = skills_root / "uri-skill"
    refs_dir = good_dir / "references"
    refs_dir.mkdir(parents=True)
    (refs_dir / "guide.md").write_text("# Guide\n", encoding="utf-8")

    (good_dir / "SKILL.md").write_text(
        """---
name: uri-skill
description: Valid skill using file:///{skill_dir}/ links in markdown and inline code.
allowed_tools:
  - compute_nca
produced_artifacts:
  - nca_parameters
---

# URI Skill

Read [guide.md](file:///{skill_dir}/references/guide.md#section-1) or `file:///{skill_dir}/references/guide.md`.
""",
        encoding="utf-8",
    )

    findings = check_skills(skills_root)
    bad = [f for f in findings if f.status != "pass"]
    assert bad == [], f"Expected file:///{{skill_dir}}/... links to pass, got: {bad}"

    # External file:///etc/passwd, symlinked reference, or missing file:///{skill_dir}/references/missing.md must fail
    outside_ref = tmp_path / "outside_ref.md"
    outside_ref.write_text("# Outside\n", encoding="utf-8")
    (refs_dir / "sym.md").symlink_to(outside_ref)

    (good_dir / "SKILL.md").write_text(
        """---
name: uri-skill
description: Skill with external file URI, symlinked ref, and missing skill_dir URI.
---

# URI Skill

See [leak](file:///etc/passwd), [sym](references/sym.md), [missing](file:///{skill_dir}/references/missing.md), and [guide](file:///{skill_dir}/references/guide.md).
""",
        encoding="utf-8",
    )
    findings_bad = check_skills(skills_root)
    by_check = {(f.skill, f.check): f for f in findings_bad}
    assert by_check[("uri-skill", "own_paths_exist")].status == "fail"
    detail = by_check[("uri-skill", "own_paths_exist")].detail
    assert "file:///etc/passwd (escapes skill directory)" in detail
    assert "symlink not permitted" in detail
    assert "references/missing.md" in detail


def test_skill_linter_catches_invalid_frontmatter_tools_artifacts_and_links(tmp_path):
    """Synthetic invalid skills trigger deterministic fail/warn findings."""
    skills_root = tmp_path / "skills"
    bad_dir = skills_root / "bad-skill"
    refs_dir = bad_dir / "references"
    refs_dir.mkdir(parents=True)
    (refs_dir / "orphan.md").write_text("# Orphan reference\n", encoding="utf-8")

    (bad_dir / "SKILL.md").write_text(
        """---
name: wrong-slug
description: Bad description with <xml> tag
allowed_tools:
  - compute_nca
  - nonexistent_magic_tool
produced_artifacts:
  - nca_parameters
  - nonexistent_state_field
---

# Bad Skill

Broken link [missing](references/does-not-exist.md) and traversal [escape](../outside.md).
""",
        encoding="utf-8",
    )

    findings = check_skills(skills_root)
    by_check = {(f.skill, f.check): f for f in findings}

    assert by_check[("bad-skill", "name_matches_folder")].status == "fail"
    assert by_check[("bad-skill", "description_no_xml")].status == "fail"
    assert by_check[("bad-skill", "allowed_tools_registered")].status == "fail"
    assert "nonexistent_magic_tool" in by_check[("bad-skill", "allowed_tools_registered")].detail
    assert by_check[("bad-skill", "produced_artifacts_valid")].status == "fail"
    assert "nonexistent_state_field" in by_check[("bad-skill", "produced_artifacts_valid")].detail
    assert by_check[("bad-skill", "own_paths_exist")].status == "fail"
    assert "escapes skill directory" in by_check[("bad-skill", "own_paths_exist")].detail
    assert by_check[("bad-skill", "references_linked")].status == "warn"
    assert "orphan.md" in by_check[("bad-skill", "references_linked")].detail
