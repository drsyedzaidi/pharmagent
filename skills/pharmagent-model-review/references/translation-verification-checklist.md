# Model Translation & Export Verification Checklist

Use this confined reference checklist during **Step 6** of `pharmagent.model_review` when reviewing NONMEM (`.ctl`), mrgsolve (`.cpp`), and cross-engine (`run_engine_comparison`) translations.

## 1. Closed-Form Export Mapping Table (`app.compute.nmexport`)

| PharmAgent `model_key` | NONMEM Subroutine | NONMEM Scaling & Symbol Map | mrgsolve `$CMT` & `$ODE` | Export Supported? |
|---|---|---|---|---|
| `iv_1cmt` | `ADVAN1 TRANS2` | `S1 = V`; `CL->CL, V->V` | `CENT` | Yes |
| `oral_1cmt` | `ADVAN2 TRANS2` | `S2 = V`; `CL->CL, V->V, KA->KA` | `DEPOT CENT` | Yes |
| `oral_1cmt_lag` | `ADVAN2 TRANS2` | `S2 = V`, `ALAG1`; `CL, V, KA, ALAG` | `DEPOT CENT` (`ALAG_DEPOT`) | Yes |
| `iv_2cmt` | `ADVAN3 TRANS4` | `S1 = V1`; `CL->CL, VC->V1, Q->Q, VP->V2` | `CENT PERIPH` | Yes |
| `oral_2cmt` | `ADVAN4 TRANS4` | `S2 = V2`; `CL->CL, VC->V2, Q->Q, VP->V3, KA->KA` | `DEPOT CENT PERIPH` | Yes |
| `iv_3cmt` | `ADVAN11 TRANS4` | `S1 = V1`; `CL->CL, VC->V1, Q2->Q2, VP2->V2, Q3->Q3, VP3->V3` | `CENT PERIPH2 PERIPH3` | Yes |
| `oral_1cmt_transit`, `*_mm`, `iv_1cmt_mixed`, `pkpd_*` | None (refused) | No closed-form ADVAN mapping | None (refused) | **No** (fitted in PharmAgent via `fit_pk_model` / `run_nlme` / `simulate_pk_profile`) |

## 2. Translation Audit Checks

- [ ] **No Runtime Execution Claim**: Stated clearly that NONMEM and mrgsolve control streams are exported text artifacts and are not executed inside PharmAgent.
- [ ] **Contiguous Block `OMEGA` Ordering**: When `omega_block` is present on non-adjacent parameters in `iiv_params`, verified that exported `ETA` indices in `$PK` / `$MAIN` match the contiguous `$OMEGA BLOCK(n)` / `@block` order.
- [ ] **Residual Error Variance vs SD**: Verified that `$SIGMA` writes variance ($\sigma_{\text{prop}}^2$, $\sigma_{\text{add}}^2$), whereas `run_nlme` reports `sigma.prop` and `sigma.add` on the standard-deviation scale.
- [ ] **Categorical Covariate Flag**: Confirmed that any categorical covariate effect flagged with `; NOTE:` (NONMEM) or `// NOTE:` (mrgsolve) is completed and verified by the human modeler before external execution.
- [ ] **Cross-Engine Metric Discipline**: Confirmed that `run_engine_comparison` rankings use engine-agnostic prediction RMSE and VPC coverage, never cross-engine `OFV`/`AIC`/`BIC`.
- [ ] **Reference Trajectory**: Recorded `simulate_pk_profile` reference outputs (`cmax`, `params`, `regimen`) so the human reviewer can compare external NONMEM/mrgsolve runs against PharmAgent's trajectory.
