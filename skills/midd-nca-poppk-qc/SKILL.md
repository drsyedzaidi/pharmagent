---
name: midd-nca-poppk-qc
description: Runs deterministic NCA and Population PK (NLME) acceptance checks before regulatory reporting. Use when verifying terminal half-life span ratio, analyst accept or revise review of flagged lambda_z fits, or NLME convergence, covariance step, RSE%, shrinkage, and residual error bounds.
allowed_tools:
  - compute_nca
  - run_qc
  - review_lambda_z_fits
  - run_nlme
  - run_poppk_qc
  - generate_report
produced_artifacts:
  - nca_parameters
  - nca_summary
  - qc_verdict
  - qc_issues
  - qc_checklist
  - nca_flagged_subjects
  - lambda_z_review
  - nlme_results
  - poppk_qc_verdict
  - poppk_qc_issues
  - poppk_qc_checklist
  - report_path
---

# MIDD NCA & PopPK Quality Control (`midd-nca-poppk-qc`)

Deterministic acceptance checks for Non-Compartmental Analysis (NCA) and Nonlinear Mixed-Effects (NLME) Population PK fits, adapted from the `AI-assisted-MIDD` pharmacometrics skill standard.

## Workflow Steps

1. **Compute NCA (`compute_nca`)**
   - Derives per-subject exposures (`Cmax`, `Tmax`, `AUC_last`, `AUC_inf`, `t_half`, `CL_F`, `Vz_F`) and terminal slope diagnostics (`lambda_z_r2_adj`, `lambda_z_n_points`, `lambda_z_span_ratio`).
2. **Run NCA QC (`run_qc`) & Analyst $\lambda_z$ Gate (`review_lambda_z_fits`)**
   - Evaluates the 8-point NCA diagnostic checklist including `Lambda_z span ratio >= 2.0` (`(t_last_window - t_first_window) / t_half >= 2.0`). Missing or `NaN` span ratios are flagged as `WARN` (never `PASS`).
   - Populates `nca_flagged_subjects` and opens the `lambda_z_review` gate (`status="awaiting_analyst_decision"`) whenever any subject has `span_ratio < 2.0`, `r2_adj < 0.80`, `n_points < 3`, or a missing terminal fit.
   - Following the upstream `AI-assisted-MIDD` `pk-nca` workflow, list the flagged subjects and ask the analyst whether to **accept** the fits as caveats or **revise** the terminal window via `review_lambda_z_fits(decision="accept" | "revise")`.
3. **Run Population PK Fit (`run_nlme`)**
   - Fits FOCE-I or SAEM mixed-effects models and computes asymptotic standard errors (`theta_rse_pct`, `omega_rse_pct`, `sigma_rse_pct`, `condition_number`, `shrinkage_pct`).
4. **Run PopPK Acceptance Checks (`run_poppk_qc`)**
   - Evaluates the 8 upstream `AI-assisted-MIDD` criteria (`source="ai_assisted_midd"`) plus PharmAgent's `condition_number <= 1000` extension (`source="pharmagent_extension"`), separating blocking failures (`FAIL`) from diagnostic caveats (`WARN` -> `CONDITIONAL PASS`).
5. **Generate Regulatory Report (`generate_report`)**
   - Refuses to finalize if `lambda_z_review.status == "revision_requested"`, and emits the per-subject NCA table (including `Span ratio`), the $\lambda_z$ review status, and both NCA and PopPK QC checklists.

## Reference Criteria

See [acceptance-criteria.md](file:///{skill_dir}/references/acceptance-criteria.md) for the complete table of blocking (`FAIL`) vs caveat (`WARN`) thresholds and their provenance.
