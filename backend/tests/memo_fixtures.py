"""Shared fixtures for the briefing-memo tests: a fully populated PharmState and the
audit trail that 'produced' it, with known entry indices.

Audit entry indices (the trace tags the memo must carry):
  0 session  1 load_dataset  2 profile_pk_dataset  3 compute_nca  4 run_qc
  5 fit_pk_model  6 run_nlme  7 run_scm  8 run_covariate_forest  9 run_vpc
  10 run_diagnostics  11 indirect_comparison (HR)  12 indirect_comparison (MD)
  13 fit_exposure_response (efficacy)  14 fit_exposure_response (toxicity)
  15 fit_exposure_response (pfs)  16 bootstrap_exposure_response (efficacy)
  17 select_optimal_dose
Each E-R entry's audited output is that run's own result (the fit without its later
bootstrap, the bootstrap without its draws), as the tools record them.
"""
from __future__ import annotations

from typing import Any

from app.compute.indirect import bucher_indirect
from app.compute.memo import AuditTrace
from app.core.audit import AuditChain
from app.core.pharmstate import PharmState, StudyInfo

IDX = {"load": 1, "profile": 2, "nca": 3, "qc": 4, "pk": 5, "nlme": 6, "scm": 7,
       "forest": 8, "vpc": 9, "diag": 10, "ind_hr": 11, "ind_md": 12,
       "er_eff": 13, "er_tox": 14, "er_pfs": 15, "er_boot": 16, "dose": 17}


def _desc(parameter: str, gm: float, gcv: float, med: float, lo: float, hi: float, n: int = 12):
    return {"parameter": parameter, "n": n, "mean": gm * 1.01, "sd": gm * 0.2, "cv_pct": 20.0,
            "median": med, "min": lo, "max": hi, "geomean": gm, "geocv_pct": gcv}


NCA_SUMMARY = {
    "n_subjects": 12, "route": "extravascular",
    "blq": {"n_below_loq": 9, "rule": "M1: BLQ records excluded"},
    "by_dose": [{"dose": 320.0, "n": 12, "Cmax_geomean": 8.6462, "CL_F_geomean": 2.7465}],
    "descriptive": [{
        "group": "all", "label": "All subjects", "n": 12,
        "parameters": [
            _desc("Cmax", 8.6462, 16.9778, 8.465, 6.44, 11.4),
            _desc("Tmax", 1.0, 40.0, 1.0, 0.5, 2.0),
            _desc("AUC_last", 98.1234, 25.5, 90.0, 71.697, 147.2347),
            _desc("AUC_inf", 118.7651, 28.2, 105.3, 81.7433, 214.9236),
            _desc("t_half", 8.1234, 30.0, 7.6, 6.6593, 14.3044),
            _desc("CL_F", 2.7123, 28.9, 2.9, 1.4889, 3.9147),
            _desc("Vz_F", 32.5, 18.0, 31.9, 25.4396, 42.7481),
            _desc("lambda_z_r2_adj", 0.99, 1.0, 0.99, 0.98, 1.0),
        ]}],
}

QC_CHECKLIST = [
    {"check": "Sample size adequacy", "status": "PASS", "detail": "12 subjects (>= 6 recommended)"},
    {"check": "AUC %extrap <= 20%", "status": "WARN", "detail": "1 subject(s) > 20.0%"},
    {"check": "Lambda_z adj R^2 >= 0.80", "status": "FAIL", "detail": "2 subject(s) below threshold"},
]

PK_COMPARE = {
    "status": "ok", "mode": "compare", "best_model": "oral_1cmt_transit",
    "ranking": [
        {"model_key": "oral_1cmt_transit", "label": "1-cmt oral transit abs.", "n_converged": 12,
         "n_subjects": 12, "total_aic": -511.4832, "mean_aic": -42.6236},
        {"model_key": "oral_1cmt", "label": "1-cmt oral (linear)", "n_converged": 11,
         "n_subjects": 12, "total_aic": -465.0591, "mean_aic": -38.7549},
    ],
    "best": {"model_key": "oral_1cmt_transit", "label": "1-cmt oral transit abs.",
             "n_subjects": 12, "n_converged": 12, "mean_aic": -42.6236, "total_aic": -511.4832,
             "population": {"parameters": {
                 "CL": {"typical_value": 2.705676, "iiv_cv_pct": 28.931006, "n": 12},
                 "V": {"typical_value": 33.945821, "iiv_cv_pct": 16.616887, "n": 12},
                 "MTT": {"typical_value": 0.613288, "iiv_cv_pct": None, "n": 12}}}},
}

NLME = {
    "status": "ok", "method": "focei", "label": "1-cmt oral (linear)", "iiv_params": ["CL", "V"],
    "error_model": "proportional", "ofv": -234.5, "converged": True, "condition_number": 38.2,
    "theta": {"CL": 2.71, "V": 32.4, "KA": 1.23},
    "omega_cv_pct": {"CL": 28.5, "V": 19.3},
    "theta_rse_pct": {"CL": 8.2, "V": 11.4, "KA": 15.6},
    "shrinkage_pct": {"CL": 12.4, "V": 18.7},
    "covariate_effects": [{"param": "CL", "covariate": "CRCL", "kind": "power",
                           "description": "CL increased with CRCL (exponent 0.75; 38% change low-high)",
                           "rse_pct": 18.3}],
}

SCM = {"status": "ok", "label": "1-cmt oral (linear)", "base_ofv": -234.5, "final_ofv": -248.7,
       "forward_p": 0.05, "backward_p": 0.01, "n_candidates": 4,
       "selected": [{"param": "CL", "covariate": "CRCL", "kind": "power", "delta_ofv": 14.2}]}

FOREST = {"status": "ok", "source": "nlme", "label": "1-cmt oral (linear)", "ci_level": 0.95,
          "rows": [
              {"param": "CL", "covariate": "CRCL", "eval_label": "5th percentile (45 mL/min)",
               "gmr": 0.84, "ci_lo": 0.71, "ci_hi": 0.99, "ci_source": "delta"},
              {"param": "CL", "covariate": "SEX", "eval_label": "F vs M",
               "gmr": 1.1, "ci_lo": None, "ci_hi": None, "ci_source": "none"}]}

VPC = {"status": "ok", "label": "1-cmt oral (linear)",
       "gof": {"r2_log_ipred": 0.967181, "rmse_log_ipred": 0.115075, "n": 120}}

DIAG = {"status": "ok", "label": "1-cmt oral (linear)",
        "residuals": {"summary": {"n": 120, "iwres_mean": -0.0, "iwres_sd": 0.115075}},
        "cwres": {"status": "ok", "summary": {"n": 120, "cwres_mean": 0.031, "cwres_sd": 1.02}},
        "npde": {"status": "ok", "summary": {"n": 120, "mean": -0.02, "sd": 0.98,
                                             "pct_outside_1_96": 5.8}}}


def _logistic(n_events: int, or_sd: tuple[float, float, float], p: float) -> dict[str, Any]:
    return {"status": "ok", "model": "logistic", "exposure_label": "AUC", "ci_level": 0.95, "n": 150,
            "n_events": n_events, "endpoint": "binary",
            "coef": {"slope": {"estimate": 0.0176, "se": 0.0029, "z": 6.08, "p": p, "lo": 0.0119, "hi": 0.0233}},
            "or_per_sd": {"sd": 160.7, "estimate": or_sd[0], "lo": or_sd[1], "hi": or_sd[2]},
            "data": {"exposure_label": "AUC"}}


ER_RESULTS = {"last_label": "pfs", "fits": {
    "efficacy": {**_logistic(72, (16.9012, 6.7922, 42.0557), 1.21e-09), "label": "efficacy",
                 "bootstrap": {"status": "ok", "n_ok": 198, "n_completed": 200, "seed": 11,
                               "or_per_sd": {"estimate": 16.9012, "boot_median": 17.2, "lo": 7.4411,
                                             "hi": 45.103}}},
    "toxicity": {**_logistic(31, (2.4567, 1.6012, 3.7698), 4.4e-05), "label": "toxicity"},
    "pfs": {"status": "ok", "model": "cox", "label": "pfs", "endpoint": "time_to_event", "ci_level": 0.95,
            "n": 150, "n_events": 94, "data": {"exposure_label": "AUC"},
            "cox": {"ties": "efron", "covariates": [{"name": "AUC", "p": 5.33e-12, "hr_per_sd": 2.1039,
                                                     "hr_per_sd_lo": 1.703, "hr_per_sd_hi": 2.5991}]}}}}

DOSE_SELECTION = {
    "status": "ok", "toxicity_cap": None,
    "utility": {"formula": "U(d) = P_eff(d) - w * P_tox(d)", "w": 1.0, "w_source": "declared"},
    "uncertainty": {"n_draws": 200, "seed": 3, "ci_level": 0.95},
    "doses": [
        {"dose": 50.0, "p_eff": 0.0812, "p_tox": 0.0201, "utility": 0.0611, "utility_lo": 0.0102,
         "utility_hi": 0.1401, "prob_optimal": 0.0, "feasible": True},
        {"dose": 200.0, "p_eff": 0.9539, "p_tox": 0.5292, "utility": 0.4247, "utility_lo": 0.2903,
         "utility_hi": 0.5468, "prob_optimal": 0.9, "feasible": True}],
    "selected": {"dose": 200.0, "utility": 0.4247, "p_eff": 0.9539, "p_tox": 0.5292, "prob_optimal": 0.9},
    "warnings": ["The selected dose 200 is at the edge of the dose grid."],
}


def indirect_comparisons() -> list[dict[str, Any]]:
    return [
        bucher_indirect({"estimate": 0.50, "se": 0.20}, {"estimate": 0.80, "se": 0.30},
                        scale="HR", treatments=("Drug A", "Placebo", "Drug C")),
        bucher_indirect({"estimate": 5.0, "se": 0.3}, {"estimate": 2.0, "se": 0.4}, scale="MD"),
    ]


def full_state() -> PharmState:
    return PharmState(
        session_id="s1",
        dataset_metadata={"dataset_id": "ds_978f9ca8", "n_records": 144, "n_subjects": 12,
                          "dose_levels": [267.84, 319.99, 320.65], "time_range": [0.0, 24.65]},
        data_quality={"n_observations": 132, "blq_pct": 6.82, "total_missing_pct": 12.5,
                      "n_sparse_subjects": 0, "quality_flags": ["BLQ 6.82%"]},
        nca_summary=NCA_SUMMARY,
        qc_verdict="CONDITIONAL PASS",
        qc_issues=[{"severity": "MEDIUM", "issue": "high %extrap: [1]"}],
        qc_checklist=QC_CHECKLIST,
        pk_model_results=PK_COMPARE, nlme_results=NLME, scm_results=SCM,
        forest_results=FOREST, vpc_results=VPC, diagnostics_results=DIAG,
        indirect_results={"comparisons": indirect_comparisons()},
        er_results=ER_RESULTS, dose_selection_results=DOSE_SELECTION,
        study_info=StudyInfo(drug_name="Bupropion", study_id="STUDY-101", sponsor="Acme Pharma"),
    )


def _er_steps() -> list[tuple[str, str, dict[str, Any]]]:
    fits = ER_RESULTS["fits"]
    core = {k: {f: v for f, v in fit.items() if f != "bootstrap"} for k, fit in fits.items()}
    boot = {**fits["efficacy"]["bootstrap"], "label": "efficacy"}
    return [*(("er_dose", "fit_exposure_response", core[k]) for k in ("efficacy", "toxicity", "pfs")),
            ("er_dose", "bootstrap_exposure_response", boot)]


def full_chain() -> AuditChain:
    chain = AuditChain()
    comps = indirect_comparisons()
    steps = [("system", "session", {}), ("data_manager", "load_dataset", {}),
             ("data_manager", "profile_pk_dataset", {}), ("nca", "compute_nca", {}),
             ("qc", "run_qc", {}), ("modeler", "fit_pk_model", {}), ("modeler", "run_nlme", {}),
             ("modeler", "run_scm", {}), ("modeler", "run_covariate_forest", {}),
             ("modeler", "run_vpc", {}), ("modeler", "run_diagnostics", {}),
             ("statistician", "indirect_comparison", comps[0]),
             ("statistician", "indirect_comparison", comps[1]),
             *_er_steps(), ("er_dose", "select_optimal_dose", DOSE_SELECTION)]
    for agent, tool, out in steps:
        chain.append(agent=agent, tool=tool, action=tool, inputs={}, outputs=out, timestamp="t")
    return chain


def full_trail() -> AuditTrace:
    return AuditTrace(full_chain().to_list())
