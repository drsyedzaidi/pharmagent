"""Tests for MIDD-inspired enhancements and review corrections:
1. NCA lambda_z_span_ratio computation, QC check (including missing/NaN span ratios),
   explicit analyst accept/revise gate (review_lambda_z_fits), and report column.
2. PopPK (NLME) acceptance check tool (run_poppk_qc) — PASS / CONDITIONAL PASS / FAIL,
   threshold validation (rejecting NaN/inf/negative), explicit theta_bounds proximity vs
   default-box degeneration, true covariance-step status, and condition_number provenance.
3. Skill Integrity Linter (check_skills) on repository skills, file:///{skill_dir}/...
   single-pass resolution, and synthetic edge cases.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from docx import Document

from app.compute.nca import Profile, _best_lambda_z, nca_subject, refit_lambda_z_manual
from app.compute.nca_ss import nca_ss_subject
from app.core.audit import AuditChain
from app.core.llm import MockLLM
from app.core.pharmstate import PharmState
from app.core.skill_linter import DEFAULT_SKILLS_DIR, check_skills
from app.tools.base import ToolContext
from app.tools.builtins import default_registry
from app.tools.qc_tools import (
    evaluate_poppk_qc,
    review_lambda_z_fits,
    run_poppk_qc,
    run_qc,
)
from app.tools.report_tools import generate_report

# ---------------------------------------------------------------------------
# 1. Item 2 & Corrections 1, 7: lambda_z_span_ratio & analyst accept/revise gate
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
    """Correction 1: run_qc warns on < 2.0, None, or NaN lambda_z_span_ratio (never PASS)."""
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


def test_flagged_lambda_z_analyst_accept_revise_gate(tmp_path):
    """Correction 7: Flagged lambda_z fits open an explicit analyst accept/revise gate."""
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

    state, qc_res = reg.execute(
        "run_qc", state=state, ctx=ctx, args={}, audit=audit, timestamp="t1", caller_agent="qc"
    )
    assert state.nca_flagged_subjects is not None
    assert len(state.nca_flagged_subjects) == 1
    assert state.nca_flagged_subjects[0]["id"] == 1
    assert state.lambda_z_review["status"] == "awaiting_analyst_decision"
    assert state.lambda_z_review["gate_open"] is True
    assert "analyst decision required" in qc_res.summary

    # If generate_report is called with require_lambda_z_review=True before review, it blocks
    with pytest.raises(ValueError, match="awaiting analyst decision"):
        generate_report(state, ctx, {"require_lambda_z_review": True})

    # Analyst chooses to REVISE -> gate blocks report generation
    state, rev_res = reg.execute(
        "review_lambda_z_fits",
        state=state,
        ctx=ctx,
        args={"decision": "revise", "notes": "Exclude t=4h point for subject 1"},
        audit=audit,
        timestamp="t2",
        caller_agent="qc",
    )
    assert state.lambda_z_review["status"] == "revision_requested"
    assert state.lambda_z_review["gate_open"] is True
    assert "REVISE" in rev_res.summary

    with pytest.raises(ValueError, match="revision_requested"):
        generate_report(state, ctx, {})

    # Analyst chooses to ACCEPT -> gate closes and report generation succeeds
    state, acc_res = reg.execute(
        "review_lambda_z_fits",
        state=state,
        ctx=ctx,
        args={"decision": "accept", "notes": "Retain subject 1 with span ratio caveat"},
        audit=audit,
        timestamp="t3",
        caller_agent="qc",
    )
    assert state.lambda_z_review["status"] == "accepted"
    assert state.lambda_z_review["gate_open"] is False
    assert "ACCEPT" in acc_res.summary

    state, rep_res = reg.execute(
        "generate_report",
        state=state,
        ctx=ctx,
        args={"require_lambda_z_review": True},
        audit=audit,
        timestamp="t4",
        caller_agent="report",
    )
    assert rep_res.result["sections"]["lambda_z_review_status"] == "accepted"

    # Invalid decision raises ValueError
    with pytest.raises(ValueError, match="must be 'accept' or 'revise'"):
        review_lambda_z_fits(state, ctx, {"decision": "ignore"})


# ---------------------------------------------------------------------------
# 2. Item 1 & Corrections 2, 4, 5, 6: run_poppk_qc & evaluate_poppk_qc
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


def test_run_poppk_qc_passes_healthy_fit_and_tags_provenance():
    """Correction 6: condition_number is tagged as pharmagent_extension; upstream 8 as ai_assisted_midd."""
    state = PharmState(nlme_results=_healthy_nlme())
    res = run_poppk_qc(state, ToolContext(), {})
    assert res.result["verdict"] == "PASS"
    assert res.writes["poppk_qc_verdict"] == "PASS"
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
    assert "PharmAgent extension" in checks_by_name["condition_number"]["detail"]
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


def test_custom_qc_thresholds_reject_nan_inf_and_negative_values():
    """Correction 2: Custom QC thresholds reject NaN, inf, booleans, and invalid ranges."""
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

    # Shrinkage > 100% or min_bsv >= max_bsv rejected
    with pytest.raises(ValueError, match="max_shrinkage_pct"):
        evaluate_poppk_qc(nl, max_shrinkage_pct=120.0)
    with pytest.raises(ValueError, match="min_bsv_cv_pct"):
        evaluate_poppk_qc(nl, min_bsv_cv_pct=50.0, max_bsv_cv_pct=20.0)
    with pytest.raises(ValueError, match="max_condition_number"):
        evaluate_poppk_qc(nl, max_condition_number=0.5)


def test_theta_not_on_bound_declared_bounds_vs_default_degeneration():
    """Correction 4: theta_not_on_bound checks declared theta_bounds proximity and honest default-box status."""
    # 1. Without declared theta_bounds: honest detail that no estimation bounds were declared
    nl_unconstrained = _healthy_nlme()
    out_unc = evaluate_poppk_qc(nl_unconstrained)
    chk_unc = next(c for c in out_unc["checks"] if c["check"] == "theta_not_on_bound")
    assert chk_unc["status"] == "PASS"
    assert "unconstrained" in chk_unc["detail"]

    # 2. With declared theta_bounds where estimate is well inside bounds -> PASS ("no THETA on declared bound")
    out_bounded_ok = evaluate_poppk_qc(
        nl_unconstrained,
        theta_bounds={"CL": (0.1, 50.0), "V": (1.0, 500.0), "KA": (0.05, 10.0)},
    )
    chk_ok = next(c for c in out_bounded_ok["checks"] if c["check"] == "theta_not_on_bound")
    assert chk_ok["status"] == "PASS"
    assert chk_ok["detail"] == "no THETA on declared bound"

    # 3. With declared theta_bounds where KA (1.1) is within tol = 1e-4 * max(1, |est|) of upper bound 1.10005 -> FAIL
    out_at_hi = evaluate_poppk_qc(
        nl_unconstrained,
        theta_bounds={"CL": (0.1, 50.0), "V": (1.0, 500.0), "KA": (0.05, 1.10005)},
    )
    chk_hi = next(c for c in out_at_hi["checks"] if c["check"] == "theta_not_on_bound")
    assert chk_hi["status"] == "FAIL"
    assert "on declared bound: KA" in chk_hi["detail"]

    # 4. Default-box degeneration (without declared bounds) reports "degenerate relative to model defaults"
    nl_deg = _healthy_nlme()
    nl_deg["degenerate_params"] = ["KA"]
    out_deg = evaluate_poppk_qc(nl_deg)
    chk_deg = next(c for c in out_deg["checks"] if c["check"] == "theta_not_on_bound")
    assert chk_deg["status"] == "FAIL"
    assert "degenerate relative to model defaults: KA" in chk_deg["detail"]


def test_covariance_step_status_not_inferred_from_single_finite_rse():
    """Correction 5: covariance_step fails when cov_note indicates near-singular/failed Hessian even if some RSE is finite."""
    # 1. Near-singular Hessian where CL and V have finite RSEs, KA is suppressed (None), and cov_note warns
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

    # 2. Even if all structural RSEs are numerically finite, an explicit covariance_step_ok=False or
    #    indefinite-Hessian cov_note fails covariance_step
    nl_indef = _healthy_nlme()
    nl_indef["cov_note"] = (
        "information matrix indefinite (fit not at a true minimum); standard errors unreliable"
    )
    out_indef = evaluate_poppk_qc(nl_indef)
    assert next(c for c in out_indef["checks"] if c["check"] == "covariance_step")["status"] == "FAIL"

    # 3. If covariance_step_ok=True is explicitly provided by an external engine while one SE is missing,
    #    covariance_step passes and structural_se_present fails
    nl_ext = _healthy_nlme()
    nl_ext["covariance_step_ok"] = True
    nl_ext["theta_rse_pct"]["KA"] = None
    out_ext = evaluate_poppk_qc(nl_ext)
    assert next(c for c in out_ext["checks"] if c["check"] == "covariance_step")["status"] == "PASS"
    assert next(c for c in out_ext["checks"] if c["check"] == "structural_se_present")["status"] == "FAIL"


def test_run_poppk_qc_warns_on_caveats_without_failing():
    nl = _healthy_nlme()
    nl["theta_rse_pct"]["KA"] = 62.5         # > 50%
    nl["shrinkage_pct"]["V"] = 38.0          # > 30%
    nl["omega_cv_pct"]["CL"] = 0.4           # < 1%
    nl["condition_number"] = 1450.0          # > 1000
    out = evaluate_poppk_qc(nl)
    assert out["verdict"] == "CONDITIONAL PASS"
    statuses = {c["check"]: c["status"] for c in out["checks"]}
    assert statuses["structural_rse"] == "WARN"
    assert statuses["eta_shrinkage"] == "WARN"
    assert statuses["bsv_range"] == "WARN"
    assert statuses["condition_number"] == "WARN"
    assert statuses["ofv_finite"] == "PASS"
    assert statuses["covariance_step"] == "PASS"
    assert statuses["structural_se_present"] == "PASS"
    assert statuses["theta_not_on_bound"] == "PASS"
    assert statuses["residual_error_positive"] == "PASS"


def test_run_poppk_qc_fails_on_blocking_conditions():
    # 1. Non-converged / non-finite OFV
    nl_bad_ofv = _healthy_nlme()
    nl_bad_ofv["converged"] = False
    out1 = evaluate_poppk_qc(nl_bad_ofv)
    assert out1["verdict"] == "FAIL"
    assert next(c for c in out1["checks"] if c["check"] == "ofv_finite")["status"] == "FAIL"

    # 2. Complete covariance failure
    nl_no_cov = _healthy_nlme()
    nl_no_cov["theta_rse_pct"] = {"CL": None, "V": None, "KA": None}
    out3 = evaluate_poppk_qc(nl_no_cov)
    assert out3["verdict"] == "FAIL"
    assert next(c for c in out3["checks"] if c["check"] == "covariance_step")["status"] == "FAIL"

    # 3. Non-positive residual error
    nl_sigma = _healthy_nlme()
    nl_sigma["sigma"]["prop"] = 0.0
    out5 = evaluate_poppk_qc(nl_sigma)
    assert out5["verdict"] == "FAIL"
    assert next(c for c in out5["checks"] if c["check"] == "residual_error_positive")["status"] == "FAIL"


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
    assert "PopPK QC verdict: PASS." in all_text
    assert "[PASS] ofv_finite:" in all_text
    header_cells = [c.text for c in doc.tables[0].rows[0].cells]
    assert "Span ratio" in header_cells


# ---------------------------------------------------------------------------
# 3. Item 3 & Correction 3: Skill Integrity Linter (check_skills)
# ---------------------------------------------------------------------------


def test_repo_skills_pass_skill_linter():
    """Every authored skill in the repository's skills/ directory (using file:///{skill_dir}/...) passes."""
    findings = check_skills(DEFAULT_SKILLS_DIR)
    assert findings, "expected at least one skill finding"
    bad = [f for f in findings if f.status != "pass"]
    assert bad == [], f"Unexpected skill linter findings: {bad}"


def test_skill_linter_file_skill_dir_uri_not_double_parsed(tmp_path):
    """Correction 3: [label](file:///{skill_dir}/references/guide.md) and inline code pass cleanly."""
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

    # External file:///etc/passwd or missing file:///{skill_dir}/references/missing.md must fail
    (good_dir / "SKILL.md").write_text(
        """---
name: uri-skill
description: Skill with external file URI and missing skill_dir URI.
---

# URI Skill

See [leak](file:///etc/passwd) and [missing](file:///{skill_dir}/references/missing.md) and [guide](file:///{skill_dir}/references/guide.md).
""",
        encoding="utf-8",
    )
    findings_bad = check_skills(skills_root)
    by_check = {(f.skill, f.check): f for f in findings_bad}
    assert by_check[("uri-skill", "own_paths_exist")].status == "fail"
    assert "file:///etc/passwd (escapes skill directory)" in by_check[("uri-skill", "own_paths_exist")].detail
    assert "references/missing.md" in by_check[("uri-skill", "own_paths_exist")].detail


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
