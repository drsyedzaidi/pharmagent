---
name: pharmagent-model-review
description: Authored procedural playbook for structural PK/PK-PD model review, parameter and unit extraction, random-effects and residual-error diagnostics, covariate assessment, cross-engine comparison, and NONMEM/mrgsolve translation verification within PharmAgent's supported model library.
playbook_id: pharmagent.model_review
version: 1.0.0
domain: pharmacometrics_model_review
status: reviewed
approval_status: pending_owner_review
license: LicenseRef-PharmAgent-Internal
source: PharmAgent internal authored procedure; format inspired by agentskills.io and Nova In Silico ACoP 2026 M-019 (design inspiration only; no external skill content imported)
allowed_tools:
  - load_dataset
  - profile_pk_dataset
  - validate_cdisc
  - list_pk_models
  - fit_compartmental
  - run_poppk
  - fit_pk_model
  - run_nlme
  - run_scm
  - run_diagnostics
  - run_vpc
  - run_covariate_forest
  - run_prior_check
  - run_engine_comparison
  - simulate_pk_profile
  - run_bootstrap
  - run_sir
  - run_profile
  - calc_allometric
  - calc_half_life
  - adversarial_review
  - build_briefing_memo
required_inputs:
  - dataset_id_or_fitted_model_state
  - candidate_structural_model_keys
  - error_and_iiv_specification
  - target_review_or_translation_format
produced_artifacts:
  - pk_model_results
  - nlme_results
  - scm_results
  - diagnostics_results
  - vpc_results
  - forest_results
  - engine_comparison_results
  - review_results
limitations:
  - PharmAgent does not have a PBPK or QSP execution engine; model review and execution are restricted to the 19 structural PK and PK/PD models in the built-in library.
  - PharmAgent can export closed-form fitted NLME models (iv_1cmt, oral_1cmt, oral_1cmt_lag, iv_2cmt, oral_2cmt, iv_3cmt) to NONMEM (.ctl) and mrgsolve (.cpp) code, but cannot execute NONMEM or mrgsolve at runtime and cannot parse arbitrary external NONMEM or mrgsolve files.
  - Cross-engine execution via run_engine_comparison supports pharmagent_focei, pharmagent_saem, and (when locally installed) nlmixr2 or monolix; native OFV/AIC/BIC are never compared across engines.
  - Automated physiological plausibility screening (PR #21) is currently unmerged on main and is not available as a runtime tool.
requires_human_review: true
compatibility:
  product: PharmAgent
  min_version: 0.1.0
  engines:
    - pharmagent_focei
    - pharmagent_saem
    - nlmixr2
    - monolix
  notes: Scoped strictly to PharmAgent's built-in structural model registry and export mappings.
runtime_available: true
---

# Scientific Playbook: Model Review and Translation (`pharmagent.model_review`)

This playbook guides an agent through reviewing, diagnosing, comparing, and verifying structural PK and PK/PD models using PharmAgent's deterministic tools, and documents the exact boundaries of NONMEM/mrgsolve code translation.

Refer to [Translation Verification Checklist](references/translation-verification-checklist.md) when reviewing parameter mappings, block `OMEGA` ordering, and export fidelity.

## Scope and Capability Boundaries (Read First)

1. **Supported Model Library Only**: PharmAgent executes only the 19 structural models registered in `list_pk_models`:
   - **IV linear**: `iv_1cmt`, `iv_2cmt`, `iv_3cmt`.
   - **Oral linear / transit**: `oral_1cmt`, `oral_1cmt_lag`, `oral_2cmt`, `oral_1cmt_transit`.
   - **Nonlinear / Michaelis-Menten**: `iv_1cmt_mm`, `oral_1cmt_mm`, `iv_1cmt_mixed`.
   - **PK/PD (1-compartment oral PK base)**: `pkpd_direct_linear`, `pkpd_direct_emax`, `pkpd_direct_sigmoid`, `pkpd_effect_emax`, `pkpd_idr1_inhib_kin`, `pkpd_idr2_inhib_kout`, `pkpd_idr3_stim_kin`, `pkpd_idr4_stim_kout`, `pkpd_transit_lifespan`.
2. **No PBPK or QSP Engine**: PharmAgent does **not** contain a PBPK or QSP solver. Never claim that PharmAgent can execute organ-compartment PBPK or mechanistic QSP network models.
3. **NONMEM and mrgsolve Export vs Execution**:
   - PharmAgent can **export** seeded NONMEM control streams (`.ctl`) and mrgsolve C++ model files (`.cpp`) from a converged `run_nlme` fit for the 6 closed-form compartmental models (`iv_1cmt`, `oral_1cmt`, `oral_1cmt_lag`, `iv_2cmt`, `oral_2cmt`, `iv_3cmt`).
   - Models without a closed-form mapping (`oral_1cmt_transit`, Michaelis-Menten `*_mm` / `iv_1cmt_mixed`, and all `pkpd_*` models) do **not** emit NONMEM/mrgsolve control streams; PharmAgent refuses export rather than emitting an incorrect control stream.
   - PharmAgent **cannot execute NONMEM or mrgsolve at runtime** and does not parse arbitrary uploaded `.ctl`/`.mod`/`.cpp` scripts.
4. **No Invented Diagnostics or Parameters**: Every parameter estimate, RSE%, IIV CV%, shrinkage%, OFV, AIC, residual diagnostic, or VPC percentile must come from a deterministic PharmAgent tool.

---

## Step-by-Step Procedure

### Step 1: Model Structure Inventory and Dataset Alignment
1. Call `list_pk_models` to inventory supported structural models (`key`, `label`, `group`, `is_iv`, `has_pd`, `params`).
2. Confirm dataset readiness via `load_dataset`, `profile_pk_dataset`, and `validate_cdisc`:
   - Verify required structural columns (`ID`, `TIME`, `DV`, `AMT`) and optional dosing/endpoint columns (`II`, `ADDL`, `RATE`, `CMT`, `DVID`, `PD`, `CENS`).
   - Verify whether a PD endpoint is present before selecting any `pkpd_*` model (`has_pd=True`).
   - Check dosing compartment conventions: compartment `0` is the dosing compartment (`CENT` for IV, `DEPOT` or `TR1` for oral), and standard allometric exponents in the library are `0.75` on clearances/flows (`CL`, `Q`, `Q2`, `Q3`, `VMAX`) and `1.0` on volumes (`V`, `VC`, `VP`, `VP2`, `VP3`) centered at `70 kg`.

### Step 2: Parameter and Unit Extraction
1. For initial per-subject or two-stage screening, use `fit_compartmental`, `run_poppk`, or `fit_pk_model` (single model or `compare=True` across candidates ranked by AIC).
2. For population mixed-effects estimation, propose or inspect `run_nlme` (`method` in `["focei", "saem", "focei_saem", "auto"]`; note that `run_nlme` supports the 10 PK-only structural models where `has_pd=False`, is `expensive=True`, and runs via the job-backed endpoint; `pkpd_*` models where `has_pd=True` are fitted via `fit_pk_model` and forward-simulated via `simulate_pk_profile`).
3. Extract and tabulate:
   - Fixed-effect typical values (`theta`) and relative standard errors (`theta_rse_pct`),
   - Derived secondary parameters (`calc_half_life`, `calc_allometric`) with explicit units,
   - Informative-prior borrowing diagnostics via `run_prior_check` when `prior_from="nlme"` was used.
4. **Optional Future Integration Note (PR #21 — Unmerged)**:
   - Deterministic PopPK physiological plausibility screening (PR #21) is currently **unmerged** on `main` and is not registered as a runtime tool. Until PR #21 is merged and owner-approved, perform reference-scale checks only through `calc_allometric`, `calc_half_life`, and human pharmacometrician review; do not reference or invoke an unmerged plausibility tool.

### Step 3: Random-Effects (IIV / Omega) Review
From a converged `run_nlme` result, inspect:
1. **IIV parameter assignment (`iiv_params`)**: Which structural parameters carry lognormal between-subject variability ($\theta_i = \theta_{\text{pop}} \cdot \exp(\eta_i)$).
2. **Magnitude and precision (`omega_cv_pct`, `omega_matrix`)**: Report %CV ($\sqrt{\exp(\omega^2) - 1} \times 100$) and check whether a correlated block (`omega_block`) was estimated.
3. **Eta-shrinkage (`shrinkage_pct`)**: Flag high shrinkage (e.g., $>30\%$) where empirical Bayes estimates (EBEs) regress toward the population mean and diagnostic plots relying on EBEs become uninformative.
4. **Condition number and parameter uncertainty**: Inspect the covariance condition number; when parameter confidence intervals are needed, propose `run_bootstrap`, `run_sir`, or `run_profile` (all `expensive=True, proposable=True`, requiring human approval).

### Step 4: Residual-Error and Goodness-of-Fit Review
1. Inspect the residual error specification (`error_model` in `["proportional", "additive", "combined"]` and `sigma` estimates).
2. Run `run_diagnostics` to evaluate:
   - Two-stage unweighted log residuals (`IWRES`),
   - Single-provenance conditional weighted residuals (`CWRES`, Hooker 2007, with `cwres_interaction=True` by default for FOCE-I) and `CPRED` (requires a converged `run_nlme` fit of the same structural model; never mixes two-stage and NLME provenance),
   - Simulation-based normalized prediction discrepancy (`npd`).
3. Run `run_vpc` to evaluate observed-vs-predicted (`IPRED`, `PRED`) agreement and the 5th/50th/95th percentile visual predictive check band.

### Step 5: Covariate Model Review
1. If stepwise covariate modeling was performed via `run_scm` (`expensive=True`, forward selection default $p < 0.05$, backward elimination default $p < 0.01$), review the forward/backward step path, OFV drops, and selected functional forms (`power`, `linear`, `exponential`, or `categorical`).
2. Run `run_covariate_forest` (`source="auto" | "nlme" | "scm"`) to inspect geometric mean ratios (GMR) and confidence intervals across dataset percentiles (default 5th and 95th) or categorical levels. Never apply an unjustified default bioequivalence band unless explicitly supplied by the user.

### Step 6: Translation Checklist and Cross-Engine Verification
When translating or comparing a PharmAgent model across engines or into NONMEM/mrgsolve exports:
1. **Cross-engine execution (`run_engine_comparison`)**:
   - Note that `run_engine_comparison` is `expensive=True` (job-backed endpoint).
   - It compares candidate models across available adapters (`pharmagent_focei`, `pharmagent_saem`, and if locally installed, `nlmixr2` or `monolix`) using **engine-agnostic prediction accuracy** (prediction RMSE and VPC coverage).
   - **Never compare native `OFV`, `AIC`, or `BIC` across different engines** because likelihood constants and approximations differ between estimators.
2. **NONMEM (`.ctl`) and mrgsolve (`.cpp`) export review** (see [Translation Verification Checklist](references/translation-verification-checklist.md)):
   - Verify closed-form mapping eligibility (`iv_1cmt` -> `ADVAN1 TRANS2`, `oral_1cmt`/`oral_1cmt_lag` -> `ADVAN2 TRANS2`, `iv_2cmt` -> `ADVAN3 TRANS4`, `oral_2cmt` -> `ADVAN4 TRANS4`, `iv_3cmt` -> `ADVAN11 TRANS4`).
   - Verify parameter symbol mapping (`VC -> V1/V2`, `VP -> V2/V3` in NONMEM vs explicit `$ODE` compartments `DEPOT`, `CENT`, `PERIPH`, `PERIPH2`, `PERIPH3` in mrgsolve).
   - Verify `OMEGA` variance conversion ($\omega^2 = \ln(1 + (\text{CV\%}/100)^2)$) and contiguous `BLOCK(n)` / `@block` eta re-ordering when `omega_block` is present.
   - Verify absorption lag placement (`ALAG1` in NONMEM `$PK`; `ALAG_DEPOT` in mrgsolve `$MAIN`).
   - Verify categorical covariates: exported NONMEM/mrgsolve files intentionally emit a manual-coding comment (`; NOTE:` / `// NOTE:`) for categorical covariate effects rather than guessing `IF/ELSE` coding.
3. **Numerical profile check (`simulate_pk_profile`, `adversarial_review`)**:
   - Run `simulate_pk_profile` on the translated/fitted parameters to record reference time-course checkpoints (`cmax`, `cp` trajectory) for external verification in the user's NONMEM or mrgsolve environment.
   - Run `adversarial_review` and optionally `build_briefing_memo` to produce an audit-traced model review summary gated on human pharmacometrician approval.
