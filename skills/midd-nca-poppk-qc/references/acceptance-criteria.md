# NCA and PopPK Acceptance Criteria

## 1. Non-Compartmental Analysis (`run_qc`)

| Check | Threshold | Status if Violated | Rationale |
|---|---|---|---|
| `Sample size adequacy` | `n >= 6` subjects | `WARN` | Minimum cohort size for reliable summary statistics |
| `Missing data` | `<= 20%` missing | `WARN` | Protects against informative dropout |
| `Lambda_z points >= 3` | `>= 3` terminal points | `WARN` | Minimum degrees of freedom for log-linear regression |
| `Lambda_z adj R^2 >= 0.80` | `r2_adj >= 0.80` | `WARN` | Goodness of log-linear terminal fit |
| `Lambda_z span ratio >= 2.0` | `(t_end - t_start) / t_half >= 2.0` | `WARN` | Terminal sampling window should span at least 2 half-lives |
| `AUC %extrap <= 20%` | `pct_AUC_extrap <= 20%` | `WARN` | FDA/EMA guideline for reliable `AUC_inf` |
| `Parameter plausibility` | `CL_F > 0` and `Vz_F > 0` | `FAIL` | Physiological positivity constraint |
| `Tmax before terminal phase` | `Tmax < Tlast` | `WARN` | Ensures declining phase exists after `Tmax` |

## 2. Population PK Mixed-Effects (`run_poppk_qc`)

Modeled on `AI-assisted-MIDD` (`poppk-estimation/scripts/poppk_checks.R`):

| Check | Rule | Status if Violated | Meaning |
|---|---|---|---|
| `ofv_finite` | `converged == True` and `isfinite(ofv)` | `FAIL` | Optimizer must converge to a finite `-2LL` objective |
| `covariance_step` | Inverted Fisher/Hessian covariance available | `FAIL` | Required for parameter uncertainty and CIs |
| `structural_se_present` | Every structural `theta` has a finite `%RSE` | `FAIL` | Unidentified structural parameters block reporting |
| `theta_not_on_bound` | No `degenerate_params` and `theta` within bounds | `FAIL` | Boundary collapse indicates structural unidentifiability |
| `residual_error_positive` | Active `sigma` components (`prop`/`add`) `> 0` | `FAIL` | Positive residual variance required |
| `structural_rse` | All structural `%RSE <= 50%` | `WARN` | Flag poorly estimated fixed effects as caveats |
| `eta_shrinkage` | All `shrinkage_pct <= 30%` | `WARN` | High shrinkage makes EBE-based plots unreliable |
| `bsv_range` | All `omega_cv_pct` within `1%` to `100%` | `WARN` | Near-zero or extreme IIV suggests over-parameterization |
| `condition_number` | `condition_number <= 1000` | `WARN` | High collinearity in the correlation matrix |
