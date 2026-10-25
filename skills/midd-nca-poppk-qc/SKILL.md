---
name: midd-nca-poppk-qc
description: Runs deterministic NCA and Population PK (NLME) estimation-health QC checks and enforces analyst review gates before reporting. Use when verifying terminal half-life span ratio, reviewing flagged lambda_z fits, or checking NLME convergence, covariance step, RSE%, shrinkage, and residual error bounds.
allowed_tools:
  - compute_nca
  - run_qc
  - run_nlme
  - run_poppk_qc
  - run_diagnostics
  - run_vpc
  - adversarial_review
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
  - poppk_qc_nlme_digest
  - diagnostics_results
  - vpc_results
  - review_results
  - report_path
---

# MIDD NCA & PopPK Estimation Quality Control (`midd-nca-poppk-qc`)

Deterministic quality control checks for Non-Compartmental Analysis (NCA) and Nonlinear Mixed-Effects (NLME) Population PK estimation health, adapted from the `AI-assisted-MIDD` pharmacometrics skill standard.

> **Scope & Qualification Caveat:** `run_poppk_qc` evaluates **numerical estimation and parameter-uncertainty health only** (`scope="estimation_health_only"`). Passing `run_poppk_qc` is a necessary prerequisite but does **not** establish scientific or regulatory model acceptance, which additionally requires goodness-of-fit (GOF), residual diagnostics (IWRES/CWRES/NPDE via `run_diagnostics`), visual predictive checks (`run_vpc`), evaluation for the intended context of use, and human pharmacometrician review (`adversarial_review` + human gate).

## Workflow Steps

1. **Compute NCA (`compute_nca`)**
   - Derives per-subject exposures (`Cmax`, `Tmax`, `AUC_last`, `AUC_inf`, `t_half`, `CL_F`, `Vz_F`) and terminal slope diagnostics (`lambda_z_r2_adj`, `lambda_z_n_points`, `lambda_z_span_ratio`).
2. **Run NCA QC (`run_qc`) & Non-Bypassable Analyst $\lambda_z$ Gate (`review_lambda_z_fits`)**
   - Evaluates the 8-point NCA diagnostic checklist including `Lambda_z span ratio >= 2.0` (`(t_last_window - t_first_window) / t_half >= 2.0`). Missing or `NaN` span ratios are flagged as `WARN` (never `PASS`).
   - Populates `nca_flagged_subjects` and opens the `lambda_z_review` gate (`status="awaiting_analyst_decision"`) bound to `nca_parameters_digest`. Thresholds are fixed policy; `run_qc` and `run_poppk_qc` reject any threshold or decision argument.
   - `generate_report` is blocked while `lambda_z_review.status` is `"awaiting_analyst_decision"` or `"revision_requested"`, while any flagged subject is not individually accepted, or if `nca_parameters` changed after review.
   - `review_lambda_z_fits` is a **human-only** tool (not in this skill's `allowed_tools`): the registry refuses it on every agent, workflow, goal-step and skill-replay path. The analyst submits it through `POST /api/sessions/{sid}/lambda-z-review` (`decision`, optional `subject_ids`, `notes`); the audit entry records the authenticated actor. Agents must stop and ask the analyst.
3. **Run Population PK Fit (`run_nlme`)**
   - Fits FOCE-I or SAEM mixed-effects models, records the fit's estimation bounds (`theta_bounds`) for all fitted structural parameters, and computes asymptotic standard errors (`theta_rse_pct`, `omega_rse_pct`, `sigma_rse_pct`, `condition_number`, `shrinkage_pct`).
4. **Run PopPK Estimation-Health QC (`run_poppk_qc`)**
   - Runs in `poppk_full` right after `run_nlme` and is admissible in `poppk_goal`.
   - Evaluates the 8 upstream `AI-assisted-MIDD` estimation checks (`source="ai_assisted_midd"`) plus PharmAgent's `condition_number <= 1000` extension (`source="pharmagent_extension"`), binding the verdict to `poppk_qc_nlme_digest` (`nlme_fit_digest(state.nlme_results)`).
   - Missing or `NaN` shrinkage, BSV CV%, or condition-number diagnostics always emit `WARN` rows (never omitted or silently passed).
5. **Run Qualification Diagnostics & Human Review (`run_diagnostics`, `run_vpc`, `adversarial_review`)**
   - Evaluate IWRES/CWRES/NPDE residual diagnostics, GOF/pcVPC predictive checks, and adversarial review prior to reporting.
6. **Generate Report (`generate_report`)**
   - Refuses to generate a report if NCA QC is `FAIL`, if the $\lambda_z$ analyst gate is unreviewed/open/stale, or—when an NLME fit is present—if `run_poppk_qc` has not been run, is stale (`poppk_qc_nlme_digest` mismatch), or returned `FAIL`.

## Reference Criteria

See [acceptance-criteria.md](file:///{skill_dir}/references/acceptance-criteria.md) for the complete table of blocking (`FAIL`) vs caveat (`WARN`) thresholds and their provenance.
