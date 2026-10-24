---
name: midd-nca-poppk-qc
description: Runs deterministic NCA and Population PK (NLME) acceptance checks before regulatory reporting. Use when verifying terminal half-life span ratio, AUC extrapolation, or NLME convergence, covariance step, RSE%, shrinkage, and residual error bounds.
allowed_tools:
  - compute_nca
  - run_qc
  - run_nlme
  - run_poppk_qc
  - generate_report
produced_artifacts:
  - nca_parameters
  - nca_summary
  - qc_verdict
  - qc_issues
  - qc_checklist
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
2. **Run NCA QC (`run_qc`)**
   - Evaluates the 8-point NCA diagnostic checklist including `Lambda_z span ratio >= 2.0` (`(t_last_window - t_first_window) / t_half >= 2.0`).
3. **Run Population PK Fit (`run_nlme`)**
   - Fits FOCE-I or SAEM mixed-effects models and computes asymptotic standard errors (`theta_rse_pct`, `omega_rse_pct`, `sigma_rse_pct`, `condition_number`, `shrinkage_pct`).
4. **Run PopPK Acceptance Checks (`run_poppk_qc`)**
   - Separates blocking failures (`FAIL`) from diagnostic caveats (`WARN` -> `CONDITIONAL PASS`).
5. **Generate Regulatory Report (`generate_report`)**
   - Emits the per-subject NCA table (including `Span ratio`) and both NCA and PopPK QC checklists.

## Reference Criteria

See [acceptance-criteria.md](references/acceptance-criteria.md) for the complete table of blocking (`FAIL`) vs caveat (`WARN`) thresholds.
