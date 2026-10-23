---
name: pharmagent-protocol-simulation
description: Authored procedural playbook for clinical trial protocol simulation, dose sweeps, probability of target attainment (PTA), special-population/pediatric exposure matching, and exposure-response evaluation using reviewed deterministic PharmAgent tools.
playbook_id: pharmagent.protocol_simulation
version: 1.0.0
domain: clinical_trial_simulation
status: reviewed
approval_status: pending_owner_review
license: LicenseRef-PharmAgent-Internal
source: PharmAgent internal authored procedure; format inspired by agentskills.io and Nova In Silico ACoP 2026 M-019 (design inspiration only; no external skill content imported)
allowed_tools:
  - load_dataset
  - profile_pk_dataset
  - validate_cdisc
  - generate_spaghetti_plot
  - compute_nca
  - run_dose_proportionality
  - list_pk_models
  - fit_pk_model
  - run_nlme
  - simulate_pk_profile
  - run_dose_sweep
  - run_clinsim
  - run_exposure_forest
  - run_special_population
  - run_individual_exposures
  - run_pediatric_simulation
  - run_simest
  - fit_exposure_response
  - bootstrap_exposure_response
  - select_optimal_dose
  - calc_dose_regimen
  - calc_accumulation
  - calc_half_life
  - calc_allometric
  - calc_renal_function
  - convert_concentration
  - quick_one_compartment
  - run_qc
  - adversarial_review
  - build_briefing_memo
  - generate_report
required_inputs:
  - scientific_question
  - target_population_and_covariates
  - route_and_formulation
  - candidate_dosing_regimens
  - primary_pk_or_er_endpoints
  - model_or_dataset_source
produced_artifacts:
  - simulation_results
  - dose_sweep_results
  - clinsim_results
  - special_pop_results
  - pediatric_results
  - er_results
  - dose_selection_results
  - memo_results
limitations:
  - PharmAgent does not contain a PBPK or QSP execution engine; simulations are restricted to the 19 compartmental PK and PK/PD models in the reviewed library.
  - All numerical outputs must come from deterministic PharmAgent tools; the agent must never calculate or interpolate exposure metrics in prose.
  - Expensive tools run only as admission-controlled jobs. run_nlme is not proposable and is started by the user through the job-backed endpoint; run_simest and bootstrap_exposure_response may be proposed and run only after human approval.
  - Approval status is pending_owner_review; this playbook is procedural guidance, not regulatory validation or endorsement.
requires_human_review: true
compatibility:
  product: PharmAgent
  min_version: 0.1.0
  engines:
    - pharmagent_focei
    - pharmagent_saem
  notes: Requires deterministic tool registry; no external network or script execution.
runtime_available: true
---

# Scientific Playbook: Protocol Simulation Review (`pharmagent.protocol_simulation`)

This playbook defines the standard operating procedure for evaluating candidate clinical trial dosing protocols, exposure targets, covariate strata, and exposure-response trade-offs in PharmAgent.

Refer to [Scenario Audit Template](references/scenario-audit-template.md) when assembling the final scenario matrix and audit summary.

## Hard Prohibitions (Mandatory Guardrails)

1. **No invented simulation outputs**: Every concentration, `Cmax`, `AUC_tau`, `Cavg`, `Ctrough`, probability of target attainment (PTA), odds ratio, hazard ratio, or utility score must come directly from a deterministic PharmAgent tool execution. If a tool has not been run or refuses an input, label the quantity **UNRESOLVED**.
2. **No silent unit conversion**: Never convert mass, molar, volume, or time units in free text. Use `convert_concentration` (requires explicit molar mass in `g/mol`) or report a unit mismatch as an explicit blocker.
3. **No silent imputation**: Never impute missing body weights, renal function values, dosing intervals, LLOQ thresholds, or baseline covariates. Missing inputs must be surfaced as explicit assumptions or unresolved prerequisites.
4. **No claim of regulatory endorsement**: Never state that a simulated regimen or threshold is "FDA-approved", "EMA-endorsed", or "ICH-validated". Cite only the mathematical convention or user-supplied threshold actually evaluated.
5. **No automatic dose recommendation without human review**: Tool outputs (such as `run_clinsim` or `select_optimal_dose`) are decision-support proposals. Final protocol and dose selection remains with the human pharmacometrician of record.
6. **Strict tool allowlist confinement**: Only tools declared in this playbook's `allowed_tools` frontmatter may be selected. Never request shell commands, external scripts, or unverified tools.

---

## Step-by-Step Procedure

### Step 1: Clarify Protocol Specification and Scientific Scope
Before running any simulation tool, verify and document the 10 protocol dimensions from the user's request or loaded session state:
1. **Scientific question**: e.g., Phase 2 dose selection, renal/hepatic dose adjustment, pediatric bridging, sampling schedule precision, or regimen switching (q12h vs q24h).
2. **Target population**: healthy volunteers, patient indication, adult vs pediatric (`2 to <18 y`), or renal impairment strata.
3. **Route of administration**: IV bolus, IV infusion (`rate > 0`), or extravascular/oral.
4. **Formulation characteristics**: immediate-release, lag-time (`ALAG`), or transit-compartment absorption (`MTT`).
5. **Dose and dosing interval (`tau`)**: candidate doses (`doses` grid or reference `dose`) and inter-dose interval `tau` in hours (must be finite and `> 0`).
6. **Treatment duration (`n_doses`, `tmax`)**: single-dose (`n_doses = 1`) vs multiple-dose to steady state (`1 <= n_doses <= 1000`).
7. **Covariates**: body weight (`wt`, default allometric reference `70 kg`), renal function (`eGFR` / `CrCL`), age, or categorical factors.
8. **Endpoints and thresholds**: PK exposure metrics (`cmax`, `auc_tau`, `cavg`, `ctrough`), PD biomarker response, or binary/time-to-event clinical endpoints (`fit_exposure_response`).
9. **Comparator or reference regimen**: adult reference dose, normal-renal-function stratum, or baseline regimen for GMR/band comparison.
10. **Uncertainty and sensitivity scenarios**: between-subject variability (IIV), parameter estimation uncertainty (`param_uncertainty=True`), utility weight `w` sensitivity, or alternative structural models.

If any required input is ambiguous or missing, state the exact missing parameter and mark dependent outputs **UNRESOLVED** until clarified.

### Step 2: Validate Input Dataset, Schema, and Units
When an observed PK or PK/PD dataset is part of the workflow:
1. Load the dataset with `load_dataset` and inspect the schema-only metadata (`detected_roles`, column names, subject count).
2. Run `profile_pk_dataset` and `validate_cdisc` to check time ordering, missingness, BLQ fraction, and dosing record integrity (`AMT`, `TIME`, `II`, `ADDL`, `RATE`, `CMT`).
3. Optionally call `generate_spaghetti_plot` to inspect concentration-time profiles across dose groups, and `compute_nca` / `run_dose_proportionality` to establish observed non-compartmental baseline exposures (`Cmax`, `AUC_last`, `AUC_inf`, `AUC_tau`) and power-model proportionality before parametric simulation.
4. For unit conversions between `mg/L` and `umol/L`, call `convert_concentration` with explicit `molar_mass`. For bedside/analytical sanity checks of half-life, accumulation factor, loading/maintenance dose, allometric scaling, or renal function (`Cockcroft-Gault` / `CKD-EPI 2021`), use `calc_half_life`, `calc_accumulation`, `calc_dose_regimen`, `calc_allometric`, `calc_renal_function`, or `quick_one_compartment`.

### Step 3: Select the Reviewed PharmAgent Simulation Tool Matching the Question
Inspect available library models with `list_pk_models`. Ensure a valid structural model is fitted (`fit_pk_model` or `run_nlme`) or explicitly specified (`model_key` + `params`), then map the scientific question to the appropriate reviewed tool:
- **Parameter provenance & session-state note**: When PopPK variability is required, run `fit_pk_model` prior to `run_nlme` so `pk_model_results` is populated for `run_clinsim`, `run_special_population`, and `run_pediatric_simulation`. When calling `simulate_pk_profile` or `run_dose_sweep` from an NLME fit (`nlme_results["theta"]`) without a matching `fit_pk_model`, pass `params=nlme_results["theta"]` explicitly so the tool does not fall back to library default parameters (`parameter_source="default_library"`).
- **Deterministic single-regimen time-course (`simulate_pk_profile`)**: Forward-simulate concentration (and PD effect if `has_pd=True`) for a specified `model_key`, `dose`, `tau`, `n_doses`, `tmax`, `wt`, and infusion `rate`. Note that `simulate_pk_profile` returns `cmax` and the time-concentration trajectory (`time`, `concentration`, and `effect` if PD), whereas `auc_tau`, `cavg`, and `ctrough` over the final dosing interval are returned by `run_dose_sweep`.
- **Multi-dose regimen comparison (`run_dose_sweep`)**: Compare concentration profiles and last-interval summary metrics (`cmax`, `auc_tau`, `cavg`, `ctrough`) across a vector of candidate `doses` for bolus/oral dosing (`rate=0.0`). When interpreting `auc_tau`, `cavg`, and `ctrough` as steady-state metrics (`AUC_tau,ss`, `Cavg,ss`, `Ctrough,ss`), ensure `n_doses * tau >= ~5 * t_half`; for IV infusion (`rate > 0`), use `simulate_pk_profile`.
- **Virtual population & Probability of Target Attainment (`run_clinsim`)**: Simulate a virtual population (`n_subjects`, resampled covariates + fitted IIV, optional `param_uncertainty=True`) across `doses` against a clinical `threshold` (`direction="above"` for efficacy or `"below"` for safety) and `target_fraction`.
- **Covariate exposure impact (`run_exposure_forest` / `run_individual_exposures`)**: Quantify steady-state relative `AUC` and `Cmax` across covariate extremes (e.g., 5th/95th percentiles) or summarize per-subject EBE exposures (`AUCss`, `Cmax,ss`).
- **Renal or special-population dose adjustment (`run_special_population`)**: Stratify virtual subjects by renal function (`Severe`, `Moderate`, `Mild`, `Normal`) or another covariate and compare steady-state exposures against the reference stratum 90% band.
- **Pediatric exposure matching (`run_pediatric_simulation`)**: Stratify pediatric virtual subjects (`2 to <18 y`) by age and weight using allometric scaling exponents (`wt_exponent_cl`, `wt_exponent_v`) to match adult reference exposure ranges.
- **Sampling schedule precision check (`run_simest`)**: Propose a simulation-estimation check for a candidate blood-sampling schedule (note: `run_simest` supports covariate-free `run_nlme` models only, is `expensive=True, proposable=True`, and executes only after human approval).

### Step 4: Build the Scenario Matrix Without Altering the Scientific Question
Construct an explicit scenario table assigning each scenario a stable `scenario_id` (e.g., `SCEN-01`, `SCEN-02`), mapping:
- `scenario_id`, purpose (base case, dose sensitivity, interval switch, special population, uncertainty),
- exact tool name from `allowed_tools`,
- exact input arguments (`model_key`, `dose`, `doses`, `tau`, `n_doses`, `wt`, `rate`, `threshold`, `direction`),
- source of model parameters (`fitted_typical`, `nlme_fit`, or `user_supplied_defaults`).

Do not silently substitute a simpler question (for example, do not replace a virtual-population PTA question with a single typical-subject profile without stating why).

### Step 5: Execute Simulations via Reviewed Tools
Invoke the selected tools in sequence. Respect execution governance:
- Synchronous tools (`simulate_pk_profile`, `run_dose_sweep`, `run_clinsim`, `run_special_population`, `run_pediatric_simulation`, `run_exposure_forest`, `run_individual_exposures`) execute directly through the audited `ToolRegistry`.
- Expensive tools never run synchronously from chat. `run_nlme` is expensive but not proposable: it runs only through the job-backed endpoint, so the agent must ask the user to start it there and must not attempt to propose it. `run_simest` and `bootstrap_exposure_response` are expensive and proposable: the agent may propose them, and they run as jobs only after human approval. Never claim results from any of these tools exist until the job has completed.

### Step 6: Compare Exposure Metrics Across Scenarios
Tabulate only tool-computed metrics (`cmax`, `auc_tau`, `cavg`, `ctrough`, PTA fraction, stratum GMR/band overlap) by `scenario_id`. Contrast each candidate regimen against the declared comparator or target window.

### Step 7: Run Exposure-Response and Utility Dose Selection Only When Applicable
When per-subject clinical efficacy or toxicity endpoint data exist in the loaded dataset:
1. Fit binary (`endpoint="binary"`, logistic regression) or time-to-event (`endpoint="time_to_event"`, Kaplan-Meier + log-rank + Cox PH) models using `fit_exposure_response` with explicit `label="efficacy"` and `label="toxicity"`.
2. If bootstrap intervals are required, propose `bootstrap_exposure_response` (`confirm=True` required upon human approval).
3. For Project Optimus utility-based dose selection (`U(d) = P_eff(d) - w * P_tox(d)`), run `select_optimal_dose` only when both `efficacy` and `toxicity` logistic fits are stored, an exposure mapping (`exposure_mapping` or `exposure_from="dose_proportionality"`) is valid, and the clinical utility weight `w` is explicitly provided by the user (or `utility="default_w=1"` is explicitly acknowledged).
4. If no clinical endpoint columns exist, skip Step 7 and state that E-R tools are not applicable to the dataset.

### Step 8: Report Assumptions and Unresolved Inputs
List every modeling assumption (structural form, allometric exponents, linear vs nonlinear elimination, steady-state assumption, resampled vs reference covariate distribution) and every unresolved input.

### Step 9: Require Human Pharmacometrician Review Gate
Explicitly mark any regimen comparison or dose-selection output as **Pending Human Pharmacometrician Review**. State the trade-offs, boundary conditions, and QC findings (`run_qc`, `adversarial_review`) that the reviewer must verify before adopting a protocol.

### Step 10: Produce the Audit-Friendly Summary
Conclude with a structured audit summary (and optionally generate traced artifacts via `build_briefing_memo` or `generate_report`) containing:
- **Playbook identity**: `playbook_id` (`pharmagent.protocol_simulation`), `version`, `canonical_digest`, and `approval_status`.
- **Tools called**: ordered list of tool names and audit actions.
- **Scenario matrix**: `scenario_id`, tool inputs, and tool-returned outputs.
- **Unresolved items & limitations**: explicit list of unverified assumptions and software scope limits.
- **Human review status**: confirmation that final protocol sign-off requires the pharmacometrician of record.
