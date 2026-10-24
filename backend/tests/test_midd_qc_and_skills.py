"""Tests for MIDD-inspired enhancements:
1. NCA lambda_z_span_ratio computation, QC check, and report column.
2. PopPK (NLME) acceptance check tool (run_poppk_qc) — PASS / CONDITIONAL PASS / FAIL.
3. Skill Integrity Linter (check_skills) on repository skills and synthetic edge cases.
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
from app.tools.qc_tools import evaluate_poppk_qc, run_poppk_qc, run_qc

# ---------------------------------------------------------------------------
# 1. Item 2: lambda_z_span_ratio in NCA, Steady-State NCA, QC, and Report
# ---------------------------------------------------------------------------

def test_lambda_z_span_ratio_monoexponential():
    """Exact monoexponential decay: t_half = 2 h, terminal window [2, 12] -> span = 10 h -> ratio = 5.0."""
    lam = math.log(2) / 2.0  # t_half = 2.0 h
    times = np.array([0.0, 1.0, 2.0, 4.0, 8.0, 12.0])
    # Peak at t=1.0, then pure monoexponential decline from t=2..12
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


def test_run_qc_flags_short_lambda_z_span_ratio():
    """run_qc warns when any subject has lambda_z_span_ratio < 2.0 and passes when >= 2.0."""
    good_params = [
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
    res_pass = run_qc(PharmState(nca_parameters=good_params), ToolContext(), {})
    assert res_pass.result["verdict"] == "PASS"
    span_check = next(c for c in res_pass.result["checks"] if c["check"] == "Lambda_z span ratio >= 2.0")
    assert span_check["status"] == "PASS"

    short_params = [dict(p) for p in good_params]
    short_params[0]["lambda_z_span_ratio"] = 1.4
    res_warn = run_qc(PharmState(nca_parameters=short_params), ToolContext(), {})
    assert res_warn.result["verdict"] == "CONDITIONAL PASS"
    span_check_w = next(c for c in res_warn.result["checks"] if c["check"] == "Lambda_z span ratio >= 2.0")
    assert span_check_w["status"] == "WARN"
    assert any("short lambda_z span ratio" in iss["issue"] for iss in res_warn.result["issues"])


# ---------------------------------------------------------------------------
# 2. Item 1: run_poppk_qc (Population PK / NLME acceptance checks)
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


def test_run_poppk_qc_passes_healthy_fit():
    state = PharmState(nlme_results=_healthy_nlme())
    res = run_poppk_qc(state, ToolContext(), {})
    assert res.result["verdict"] == "PASS"
    assert res.writes["poppk_qc_verdict"] == "PASS"
    assert res.writes["poppk_qc_issues"] == []
    statuses = {c["check"]: c["status"] for c in res.result["checks"]}
    assert statuses == {
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
    # Blocking checks still pass
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

    # 2. Missing structural SE (partial covariance failure)
    nl_missing_se = _healthy_nlme()
    nl_missing_se["theta_rse_pct"]["KA"] = None
    out2 = evaluate_poppk_qc(nl_missing_se)
    assert out2["verdict"] == "FAIL"
    assert next(c for c in out2["checks"] if c["check"] == "covariance_step")["status"] == "PASS"
    assert next(c for c in out2["checks"] if c["check"] == "structural_se_present")["status"] == "FAIL"

    # 3. Complete covariance failure
    nl_no_cov = _healthy_nlme()
    nl_no_cov["theta_rse_pct"] = {"CL": None, "V": None, "KA": None}
    out3 = evaluate_poppk_qc(nl_no_cov)
    assert out3["verdict"] == "FAIL"
    assert next(c for c in out3["checks"] if c["check"] == "covariance_step")["status"] == "FAIL"

    # 4. Degenerate / on-bound theta
    nl_bound = _healthy_nlme()
    nl_bound["degenerate_params"] = ["KA"]
    out4 = evaluate_poppk_qc(nl_bound)
    assert out4["verdict"] == "FAIL"
    assert next(c for c in out4["checks"] if c["check"] == "theta_not_on_bound")["status"] == "FAIL"

    # 5. Non-positive residual error
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
# 3. Item 3: Skill Integrity Linter (check_skills)
# ---------------------------------------------------------------------------

def test_repo_skills_pass_skill_linter():
    """Every authored skill in the repository's skills/ directory passes with 0 failures or warnings."""
    findings = check_skills(DEFAULT_SKILLS_DIR)
    assert findings, "expected at least one skill finding"
    bad = [f for f in findings if f.status != "pass"]
    assert bad == [], f"Unexpected skill linter findings: {bad}"


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
