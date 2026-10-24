# NCA and PopPK Estimation-Health QC Criteria

## 1. Non-Compartmental Analysis (`run_qc` & `review_lambda_z_fits`)

| Check | Threshold | Status if Violated | Rationale |
|---|---|---|---|
| `Sample size adequacy` | `n >= 6` subjects | `WARN` | Minimum cohort size for reliable summary statistics |
| `Missing data` | `<= 20%` missing | `WARN` | Protects against informative dropout |
| `Lambda_z points >= 3` | `>= 3` terminal points (exact integer threshold) | `WARN` | Minimum degrees of freedom for log-linear regression |
| `Lambda_z adj R^2 >= 0.80` | `r2_adj >= 0.80` and finite | `WARN` | Goodness of log-linear terminal fit |
| `Lambda_z span ratio >= 2.0` | `(t_end - t_start) / t_half >= 2.0` and finite | `WARN` | Terminal sampling window must be estimable and span at least 2 half-lives; missing/`NaN` span ratios count as `WARN` |
| `AUC %extrap <= 20%` | `pct_AUC_extrap <= 20%` | `WARN` | FDA/EMA guideline for reliable `AUC_inf` |
| `Parameter plausibility` | `CL_F > 0` and `Vz_F > 0` (finite) | `FAIL` | Physiological positivity constraint |
| `Tmax before terminal phase` | `Tmax < Tlast` | `WARN` | Ensures declining phase exists after `Tmax` |

### Non-Bypassable Analyst Accept/Revise Gate for Flagged $\lambda_z$ Fits (`review_lambda_z_fits`)
- When `run_qc` detects any subject with `span_ratio < 2.0` (or missing/`NaN`), `r2_adj < 0.80`, or `n_points < 3`, it records them in `state.nca_flagged_subjects` and sets `state.lambda_z_review["status"] = "awaiting_analyst_decision"` bound to `nca_parameters_digest(state.nca_parameters)`.
- `generate_report` unconditionally blocks while `lambda_z_review["status"]` is `"awaiting_analyst_decision"` or `"revision_requested"`, or if `state.nca_parameters` has unreviewed flagged subjects or changed after review.
- The analyst must explicitly decide via `review_lambda_z_fits`:
  - `decision="accept"` -> `status="accepted"` (flagged fits retained with documented caveats).
  - `decision="revise"` -> `status="revision_requested"` (keeps gate open until terminal windows are revised and accepted).

## 2. Population PK Mixed-Effects Estimation-Health QC (`run_poppk_qc`)

> **Important Scope Limitation:** These checks verify **numerical estimation and covariance health only** (`scope="estimation_health_only"`). They do **not** establish model or regulatory acceptance, which additionally requires goodness-of-fit (GOF), CWRES/NPDE residual diagnostics (`run_diagnostics`), VPC/pcVPC predictive checks (`run_vpc`), evaluation for the intended context of use, and human pharmacometrician review (`adversarial_review`).

### 2.1 Upstream `AI-assisted-MIDD` Estimation Criteria (`source="ai_assisted_midd"`)
Modeled on `AI-assisted-MIDD` (`poppk-estimation/scripts/poppk_checks.R`):

| Check | Rule | Status if Violated | Meaning |
|---|---|---|---|
| `ofv_finite` | `converged == True` and `isfinite(ofv)` | `FAIL` | Optimizer must converge to a finite `-2LL` objective |
| `covariance_step` | Covariance step succeeded (positive-definite Hessian inverted without `near-singular`/`failed` `cov_note`, and `covariance_step_ok != False`) | `FAIL` | Actual covariance-step success required for parameter uncertainty |
| `structural_se_present` | Every structural `theta` has a finite `%RSE` | `FAIL` | Unidentified structural parameters block reporting |
| `theta_not_on_bound` | Fit configuration bounds (`nl["theta_bounds"]` or model estimation box) cover all fitted `theta` parameters, no estimate is within `1e-4 * max(1, |est|)` of a bound, and no default-box degeneration (`[1e-7x, 1e7x]`) | `FAIL` | Boundary proximity, incomplete bound configuration, or collapse indicates unidentifiability |
| `residual_error_positive` | Active `sigma` components (`prop`/`add`) `> 0` and finite | `FAIL` | Positive residual variance required |
| `structural_rse` | All structural `%RSE <= 50%` and finite | `WARN` | Flag poorly estimated fixed effects as caveats |
| `eta_shrinkage` | Present, finite for all IIV parameters, and `<= 30%` | `WARN` | Missing/`NaN` or `> 30%` shrinkage makes EBE-based plots unreliable |
| `bsv_range` | Present, finite for all IIV parameters, and within `1%` to `100%` | `WARN` | Missing/`NaN`, near-zero, or extreme IIV suggests over-parameterization |

### 2.2 PharmAgent Extensions (`source="pharmagent_extension"`)
Additional check native to PharmAgent (`app.compute.nlme._COND_RED_FLAG`), documented separately from upstream `AI-assisted-MIDD`:

| Check | Rule | Status if Violated | Meaning |
|---|---|---|---|
| `condition_number` | Present, finite `> 0`, and `<= 1000` | `WARN` | Eigenvalue ratio of the correlation matrix; missing/`NaN` or `> 1000` indicates ill-conditioning |
