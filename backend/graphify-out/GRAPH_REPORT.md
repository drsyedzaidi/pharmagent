# Graph Report - backend  (2026-06-20)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 1173 nodes · 2805 edges · 79 communities (71 shown, 8 thin omitted)
- Extraction: 89% EXTRACTED · 11% INFERRED · 0% AMBIGUOUS · INFERRED: 321 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Access Control & Session Management|Access Control & Session Management]]
- [[_COMMUNITY_Data Models & Pharmacokinetics|Data Models & Pharmacokinetics]]
- [[_COMMUNITY_Regulatory Report Generation|Regulatory Report Generation]]
- [[_COMMUNITY_Population Model Export|Population Model Export]]
- [[_COMMUNITY_Compartmental Model Parameters|Compartmental Model Parameters]]
- [[_COMMUNITY_Agent Routing & Supervision|Agent Routing & Supervision]]
- [[_COMMUNITY_Goodness-of-Fit Diagnostics|Goodness-of-Fit Diagnostics]]
- [[_COMMUNITY_Bioequivalence Assessment|Bioequivalence Assessment]]
- [[_COMMUNITY_Population PK Summary|Population PK Summary]]
- [[_COMMUNITY_Workflow Orchestration|Workflow Orchestration]]
- [[_COMMUNITY_PK Model Fitting|PK Model Fitting]]
- [[_COMMUNITY_Adversarial Review Engine|Adversarial Review Engine]]
- [[_COMMUNITY_NLME Estimation Tests|NLME Estimation Tests]]
- [[_COMMUNITY_Core Framework & Configuration|Core Framework & Configuration]]
- [[_COMMUNITY_Report Generation Tools|Report Generation Tools]]
- [[_COMMUNITY_Model Residual Diagnostics|Model Residual Diagnostics]]
- [[_COMMUNITY_Dose Proportionality Analysis|Dose Proportionality Analysis]]
- [[_COMMUNITY_Dose Forecasting & TDM|Dose Forecasting & TDM]]
- [[_COMMUNITY_Non-Compartmental Analysis|Non-Compartmental Analysis]]
- [[_COMMUNITY_CDISC Export|CDISC Export]]
- [[_COMMUNITY_Job Management & Downloads|Job Management & Downloads]]
- [[_COMMUNITY_Population Model Fitting|Population Model Fitting]]
- [[_COMMUNITY_Compartmental Model Fitting|Compartmental Model Fitting]]
- [[_COMMUNITY_Covariate Effect Summary|Covariate Effect Summary]]
- [[_COMMUNITY_PK Dataset Model Fitting|PK Dataset Model Fitting]]
- [[_COMMUNITY_Shared Analysis State|Shared Analysis State]]
- [[_COMMUNITY_Oral Compartmental Modeling|Oral Compartmental Modeling]]
- [[_COMMUNITY_Data Loading & Profiling|Data Loading & Profiling]]
- [[_COMMUNITY_True NLME Estimation|True NLME Estimation]]
- [[_COMMUNITY_Empirical Bayes Estimation|Empirical Bayes Estimation]]
- [[_COMMUNITY_Audit Trail & Hashing|Audit Trail & Hashing]]
- [[_COMMUNITY_Adversarial Refutation Logic|Adversarial Refutation Logic]]
- [[_COMMUNITY_End-to-End Orchestration Tests|End-to-End Orchestration Tests]]
- [[_COMMUNITY_State Management & Write Access|State Management & Write Access]]
- [[_COMMUNITY_Session Persistence|Session Persistence]]
- [[_COMMUNITY_Dose Sweep Analysis|Dose Sweep Analysis]]
- [[_COMMUNITY_NCA Subject Profiling|NCA Subject Profiling]]
- [[_COMMUNITY_Multiple Dose Extraction|Multiple Dose Extraction]]
- [[_COMMUNITY_Bioequivalence Agent Tools|Bioequivalence Agent Tools]]
- [[_COMMUNITY_Population PK Agent Tools|Population PK Agent Tools]]
- [[_COMMUNITY_Covariate Effect Modeling|Covariate Effect Modeling]]
- [[_COMMUNITY_Tool Registry & Review|Tool Registry & Review]]
- [[_COMMUNITY_Steady-State Computation|Steady-State Computation]]
- [[_COMMUNITY_Generic ODE Simulation|Generic ODE Simulation]]
- [[_COMMUNITY_Compartmental Agent Tools|Compartmental Agent Tools]]
- [[_COMMUNITY_Identity & Audit Provenance|Identity & Audit Provenance]]
- [[_COMMUNITY_HTTP API Tests|HTTP API Tests]]
- [[_COMMUNITY_Parameter Uncertainty|Parameter Uncertainty]]
- [[_COMMUNITY_LLM Client Integration|LLM Client Integration]]
- [[_COMMUNITY_Run Provenance Tracking|Run Provenance Tracking]]
- [[_COMMUNITY_Privacy-Safe Schema Extraction|Privacy-Safe Schema Extraction]]
- [[_COMMUNITY_Population Model Estimation|Population Model Estimation]]
- [[_COMMUNITY_Population Test Fixtures|Population Test Fixtures]]
- [[_COMMUNITY_QC Agent Tools|QC Agent Tools]]
- [[_COMMUNITY_Documentation|Documentation]]
- [[_COMMUNITY_Structured Logging|Structured Logging]]
- [[_COMMUNITY_Covariate Model Testing|Covariate Model Testing]]
- [[_COMMUNITY_Application Settings|Application Settings]]
- [[_COMMUNITY_HTTP Error Handling|HTTP Error Handling]]
- [[_COMMUNITY_Censored Data Fitting|Censored Data Fitting]]
- [[_COMMUNITY_Static File Serving|Static File Serving]]
- [[_COMMUNITY_Residual Error Computation|Residual Error Computation]]
- [[_COMMUNITY_Below-LLOQ Censoring Tests|Below-LLOQ Censoring Tests]]
- [[_COMMUNITY_Test Configuration|Test Configuration]]
- [[_COMMUNITY_Validation References|Validation References]]
- [[_COMMUNITY_Sample Dataset Generation|Sample Dataset Generation]]
- [[_COMMUNITY_Empty Cohort Handling|Empty Cohort Handling]]
- [[_COMMUNITY_Variance Parameter Tests|Variance Parameter Tests]]
- [[_COMMUNITY_Covariate Model Tests|Covariate Model Tests]]
- [[_COMMUNITY_Covariate Recovery Tests|Covariate Recovery Tests]]
- [[_COMMUNITY_Covariate RSE Tests|Covariate RSE Tests]]

## God Nodes (most connected - your core abstractions)
1. `PharmState` - 131 edges
2. `ToolContext` - 76 edges
3. `get_model()` - 58 edges
4. `Orchestrator` - 58 edges
5. `simulate()` - 46 edges
6. `AuditChain` - 38 edges
7. `ToolResult` - 37 edges
8. `AccessError` - 34 edges
9. `generate_272()` - 33 edges
10. `JobManager` - 31 edges

## Surprising Connections (you probably didn't know these)
- `DataFrame` --uses--> `PharmState`  [INFERRED]
  tests/test_adversarial.py → app/core/pharmstate.py
- `Path` --uses--> `ToolContext`  [INFERRED]
  tests/test_report_272.py → app/agents/base.py
- `PharmState` --uses--> `ToolContext`  [INFERRED]
  tests/test_report_272.py → app/agents/base.py
- `ToolContext` --uses--> `ToolContext`  [INFERRED]
  tests/test_report_272.py → app/agents/base.py
- `test_audit_chain_detects_tampering()` --calls--> `AuditChain`  [EXTRACTED]
  tests/test_audit_and_state.py → app/core/audit.py

## Import Cycles
- None detected.

## Communities (79 total, 8 thin omitted)

### Community 0 - "Access Control & Session Management"
Cohesion: 0.07
Nodes (54): actor_id(), adversarial_review(), chat(), ChatRequest, current_owner(), DoseSweepRequest, export_cdisc(), export_control() (+46 more)

### Community 1 - "Data Models & Pharmacokinetics"
Cohesion: 0.08
Nodes (51): Any, pk_models(), Any, DataFrame, PharmState, ToolContext, ToolResult, dose_events() (+43 more)

### Community 2 - "Regulatory Report Generation"
Cohesion: 0.12
Nodes (48): Any, Document, PharmState, ToolContext, ToolResult, Compound/study metadata for CTD regulatory reports.      Populated by the caller, StudyInfo, StudyInfo (+40 more)

### Community 3 - "Population Model Export"
Cohesion: 0.09
Nodes (43): Any, PharmState, build_mrgsolve(), build_nonmem(), _center_default(), _cov_omega_sigma(), _cv_to_omega2(), _g() (+35 more)

### Community 4 - "Compartmental Model Parameters"
Cohesion: 0.06
Nodes (11): _allo(), _effect_cmt_rhs(), _idr1_rhs(), _idr2_rhs(), _idr3_rhs(), _idr4_rhs(), _pd_direct_rhs(), _pkpd_pk() (+3 more)

### Community 5 - "Agent Routing & Supervision"
Cohesion: 0.14
Nodes (23): Agent, AgentResult, Compact, privacy-safe view of state for LLM tool selection., Domain agent definitions (Phase 1 roster)., Supervisor — Level 0 routing.  Two-stage: (1) weighted keyword scoring over doma, Return (agent_name, routing_method)., score(), Supervisor (+15 more)

### Community 6 - "Goodness-of-Fit Diagnostics"
Cohesion: 0.11
Nodes (31): _cv_pct_to_sd(), _gof_log(), obs_vs_pred(), pcvpc(), Goodness-of-fit (observed vs predicted) and visual predictive check (VPC).  Pure, Log-scale R^2 and RMSE of observed vs individual prediction.      Both inputs ar, Visual predictive check band of predicted concentration over time.      Simulate, Prediction-corrected VPC (Bergstrand et al. 2011).      Each observation is pred (+23 more)

### Community 7 - "Bioequivalence Assessment"
Cohesion: 0.11
Nodes (30): Any, ndarray, assess_bioequivalence(), be_one_parameter(), _be_paired(), _be_parallel(), _ci_from_log(), _clean_logs() (+22 more)

### Community 8 - "Population PK Summary"
Cohesion: 0.13
Nodes (25): Any, covariate_effect(), _geocv_pct(), _geomean(), Two-stage (STS) population PK summary — deterministic compute.  Pure functions,, Regress ln(param) on a continuous covariate across subjects.      Pairs each sub, Geometric mean of strictly positive values, or None if none usable.      Non-pos, Geometric CV% = 100 * sqrt(exp(var(ln(x))) - 1), ddof=1.      Requires at least (+17 more)

### Community 9 - "Workflow Orchestration"
Cohesion: 0.19
Nodes (10): Any, get_workflow(), Any, Workflow templates — ordered steps with agent/tool assignment and review gates., Orchestrator, Execute a single tool directly (UI-driven diagnostics) with audit + writes., Run the adversarial reviewer in a loop until the checkable goal is met         (, Apply user column-role overrides to the dataset metadata so every         downst (+2 more)

### Community 10 - "PK Model Fitting"
Cohesion: 0.18
Nodes (24): fit_subject_model(), Fit one subject's PK profile to a structural model., get_model(), Apply allometric WT scaling (centered at 70 kg) to flow/volume params., Simulate the model.      ``doses``: list of {"time", "amt", optional "cmt", opti, scale_params(), simulate(), PK model library: simulator vs analytic/closed-form, fit recovery, scaling. (+16 more)

### Community 11 - "Adversarial Review Engine"
Cohesion: 0.16
Nodes (23): _clean_nca(), _ids(), DataFrame, Adversarial reviewer: independent refutation engine + loop driver.  Covers:   -, _raw_df(), _seed_session_with_nca(), _sev(), test_auc_band_violation_is_critical_recompute() (+15 more)

### Community 12 - "NLME Estimation Tests"
Cohesion: 0.13
Nodes (23): Any, Tests for app.compute.nlme — true NLME estimation (FOCE-I and SAEM).  The test s, With compute_uncertainty=False the keys exist but RSEs are empty/None., A converged fit reports an RSE% for CL, V and KA., For a rich 30-subject design the well-identified structural parameters     (CL,, A well-identified 1-cmt model on rich data has a finite, modest condition     nu, test_condition_number_is_well_behaved(), test_focei_and_saem_agree_on_cl() (+15 more)

### Community 13 - "Core Framework & Configuration"
Cohesion: 0.11
Nodes (13): Agent base.  An agent owns a system prompt, a set of bound tools (by ownership i, Application configuration.  Loaded from environment variables (or a .env file)., Any, AuditChain, PharmState, LLM client.  Two responsibilities only — everything quantitative is a tool:   1., Tool framework.  A Tool is deterministic Python. Each declares a JSON input sche, Server-side resources tools may use. NEVER serialized to the LLM. (+5 more)

### Community 14 - "Report Generation Tools"
Cohesion: 0.19
Nodes (21): Any, Document, PharmState, ToolContext, ToolResult, _dose_grouped_meaningfully(), _extra_sections(), _fmt() (+13 more)

### Community 15 - "Model Residual Diagnostics"
Cohesion: 0.18
Nodes (20): _cv_pct_to_sd(), fit_residuals(), npde(), Model goodness-of-fit residual diagnostics for the PharmAgent PK platform.  Pure, Simulation-based prediction-distribution errors (PDE; see module docs).      Par, Convert a between-subject CV% to a log-normal SD.      Mirrors ``app.compute.vpc, Log-scale individual weighted residuals (IWRES).      Parameters     ----------, _make_subject() (+12 more)

### Community 16 - "Dose Proportionality Analysis"
Cohesion: 0.16
Nodes (19): Any, ndarray, assess_dose_proportionality(), _clean_pairs(), power_model(), Dose proportionality — power-model assessment (Smith et al. 2000).  Pure functio, Run the power model for each parameter and aggregate proportionality.      `dose, Pair doses with values by index, dropping non-positive / None entries. (+11 more)

### Community 17 - "Dose Forecasting & TDM"
Cohesion: 0.14
Nodes (19): Any, PKModel, forecast(), _optimize_dose(), MAP/empirical-Bayes forecasting and TDM dose individualization.  Given a fitted, Steady-state exposure metrics over the last dosing interval., Find the dose whose steady-state ``metric`` equals ``target`` by bisection     (, MAP-individualize a new patient and forecast steady-state exposure.      Args: (+11 more)

### Community 18 - "Non-Compartmental Analysis"
Cohesion: 0.16
Nodes (18): Any, ndarray, Any, _auc_intervals(), _best_lambda_z(), _geocv(), _geomean(), Non-compartmental analysis — deterministic compute.  Pure functions, no agent/LL (+10 more)

### Community 19 - "CDISC Export"
Cohesion: 0.17
Nodes (18): Any, DataFrame, PharmState, build_adpc(), build_adpp(), build_define_xml(), build_package(), _csv_bytes() (+10 more)

### Community 20 - "Job Management & Downloads"
Cohesion: 0.14
Nodes (14): Any, download_report(), download_report_272(), _frontend_dist(), Path, upload_for_session(), JobManager, Background job execution for long-running tools (NLME, SCM).  These tools run fo (+6 more)

### Community 21 - "Population Model Fitting"
Cohesion: 0.15
Nodes (20): focei_fit(), _initial_theta(), _pack(), _PopSpec, _post_fit_uncertainty(), Refit typical structural values AND covariate coefficients, etas fixed.      For, Fit a population PK model by SAEM (stochastic approximation EM).      Explorator, Flatten the current SAEM estimates into a vector for change tracking. (+12 more)

### Community 22 - "Compartmental Model Fitting"
Cohesion: 0.18
Nodes (18): Any, conc_1cmt_oral(), fit_compartmental(), fit_compartmental_ss(), fit_one_subject(), Fit one subject's oral PK profile on the log (proportional-error) scale.      Op, Top-level entry: group by subject, fit every model, select lowest AIC.      `rec, Fit steady-state models to per-subject dosing-interval profiles.      ``profiles (+10 more)

### Community 23 - "Covariate Effect Summary"
Cohesion: 0.15
Nodes (18): ndarray, _assemble(), _covariate_records(), _cv_pct(), _individual_params(), _individual_records(), _predict(), Human-readable effect summary given the fitted coefficient(s). (+10 more)

### Community 24 - "PK Dataset Model Fitting"
Cohesion: 0.18
Nodes (18): Any, ndarray, PKModel, compare_models(), fit_pk_dataset(), fit_subject_pkpd(), _init_guess(), _pack() (+10 more)

### Community 25 - "Shared Analysis State"
Cohesion: 0.18
Nodes (16): Any, PharmState, ToolContext, ToolResult, Any, DataFrame, PharmState, ToolContext (+8 more)

### Community 26 - "Oral Compartmental Modeling"
Cohesion: 0.18
Nodes (16): ndarray, _accum(), conc_2cmt_oral(), conc_2cmt_oral_ss(), _FitData, _initial_estimates(), _predict(), Compartmental oral PK model fitting — deterministic compute.  Pure functions, no (+8 more)

### Community 27 - "Data Loading & Profiling"
Cohesion: 0.25
Nodes (15): Any, DataFrame, Path, PharmState, ToolContext, ToolResult, generate_spaghetti_plot(), load_dataset() (+7 more)

### Community 28 - "True NLME Estimation"
Cohesion: 0.18
Nodes (16): _candidate_key(), _cov_effects_from_records(), cv_pct_to_omega2(), _fit_batch(), _prepare_subjects(), True nonlinear mixed-effects (NLME) estimation for the PK model library.  This m, Wrap raw subject dicts; skip those too sparse to contribute., Build a focei_fit ``init`` warm-start from an incumbent fit result.      Carries (+8 more)

### Community 29 - "Empirical Bayes Estimation"
Cohesion: 0.18
Nodes (16): _apply_cov(), _conditional_mode(), _laplace_subject(), _make_predictor_cache(), map_estimate(), _population_ofv(), One E-step: random-walk Metropolis update of every subject's eta.      The targe, Maximum-a-posteriori (empirical-Bayes) estimate of a NEW patient's random     ef (+8 more)

### Community 30 - "Audit Trail & Hashing"
Cohesion: 0.16
Nodes (9): AuditEntry, hash_payload(), Hash-chain audit trail.  Every tool invocation appends an entry whose hash incor, Recompute the chain; return True iff intact., Rebuild a chain from persisted entry dicts (e.g. loaded from the DB)., Stable SHA-256 of an arbitrary JSON-serializable payload., _sha256(), An entry persisted before actor/reason existed (no such keys) must still     reb (+1 more)

### Community 31 - "Adversarial Refutation Logic"
Cohesion: 0.30
Nodes (14): Any, DataFrame, _finding(), _nca_dose_monotonic(), _nca_internal(), _nca_recompute(), _nlme_refute(), _obs_by_subject() (+6 more)

### Community 32 - "End-to-End Orchestration Tests"
Cohesion: 0.20
Nodes (13): MockLLM, Deterministic, keyless. Drives the core NCA flow heuristically., client(), _orch(), End-to-end orchestration: routing, the NCA workflow, and the review gate.  Runs, dataset_metadata must not carry raw row values., A session (state + audit chain) reloads from the DB after a 'restart'., test_chat_routes_and_loads() (+5 more)

### Community 33 - "State Management & Write Access"
Cohesion: 0.19
Nodes (12): Any, apply_writes(), PharmStateError, PharmState — the typed communication bus.  Agents never call each other. They re, Apply ``writes`` to ``state`` if ``agent`` owns every targeted field.      Retur, Raised when an agent writes to a field it does not own., Exception, Audit hash-chain integrity and PharmState write-access enforcement. (+4 more)

### Community 34 - "Session Persistence"
Cohesion: 0.20
Nodes (8): Any, Path, SQLite persistence for sessions and their audit trails.  Sessions, PharmState, t, SessionStore, Row, test_session_ownership_enforced(), _orch(), ToolContext

### Community 35 - "Dose Sweep Analysis"
Cohesion: 0.23
Nodes (13): dose_sweep(), _interval_metrics(), Compute cmax / auc_tau / cavg / ctrough over the last dosing interval.      ``cm, Simulate ``model_key`` across several dose levels and report exposure.      For, Tests for app.compute.dose_sweep — analytic PK-property validation.  These tests, test_cavg_equals_auc_over_tau(), test_linear_pk_is_dose_proportional(), test_linear_pk_triple_dose_triples_exposure() (+5 more)

### Community 36 - "NCA Subject Profiling"
Cohesion: 0.31
Nodes (13): nca_subject(), Profile, Top-level entry: build per-subject profiles and compute NCA + summary.      `rec, Compute NCA parameters for a single subject's profile.      ``is_iv`` controls r, run_nca(), _mono_profile(), NCA compute validated against an analytic mono-exponential profile.  For C(t) =, test_aucinf_matches_analytic() (+5 more)

### Community 37 - "Multiple Dose Extraction"
Cohesion: 0.24
Nodes (12): Any, _dose_times(), extract_ss_intervals(), is_multiple_dose(), _num(), Multiple-dose / steady-state dataset extraction.  Turns a NONMEM-style record se, Coerce a cell to float; '.', '', NA -> None., Expand ADDL/II into the full list of dose times; return (times, dose, tau). (+4 more)

### Community 38 - "Bioequivalence Agent Tools"
Cohesion: 0.27
Nodes (12): Any, DataFrame, PharmState, ToolContext, ToolResult, _exposures_by_treatment(), _find_treatment_col(), _pick_levels() (+4 more)

### Community 39 - "Population PK Agent Tools"
Cohesion: 0.27
Nodes (12): Any, DataFrame, PharmState, ToolContext, ToolResult, detect_roles(), Map each column to a PK role. Exact names win; otherwise ordered     substring r, _covariate_by_subject() (+4 more)

### Community 40 - "Covariate Effect Modeling"
Cohesion: 0.17
Nodes (9): PKModel, _build_cov_effects(), _CovEffect, _error_components(), Return (has_prop, has_add) for the named residual-error model., One parameter-covariate relationship (continuous or categorical)., Resolve a covariate-model spec into _CovEffects, computing centers and     categ, Choose IIV parameters: requested ∩ model params, else CL/V, else first 2. (+1 more)

### Community 41 - "Tool Registry & Review"
Cohesion: 0.20
Nodes (10): ToolRegistry, Any, PharmState, ToolContext, ToolResult, default_registry(), Assemble the default tool registry from all tool modules., adversarial_review() (+2 more)

### Community 42 - "Steady-State Computation"
Cohesion: 0.24
Nodes (11): conc_1cmt_oral_ss(), One-compartment oral concentration at STEADY STATE over a dosing interval., Steady-state compute: closed-form limits, superposition, NCA, fitting, extractio, As tau -> inf there is no accumulation; SS curve == single-dose curve., SS conc must equal the sum of many prior single doses (superposition)., Fit the SS 1-cmt model to a simulated SS profile -> recover CL, V., test_run_nca_ss_groups_by_dose(), test_ss_1cmt_matches_superposition() (+3 more)

### Community 43 - "Generic ODE Simulation"
Cohesion: 0.25
Nodes (9): Any, ndarray, PKModel, Dose sweep — multi-level exposure metrics over a dosing regimen.  Pure, determin, PKModel, _initial(), Generic ODE simulator for the PK / PK-PD model library.  Integrates any ``PKMode, Forward-simulate a dosing regimen on a dense time grid (for plotting).      Retu (+1 more)

### Community 44 - "Compartmental Agent Tools"
Cohesion: 0.27
Nodes (10): Any, DataFrame, PharmState, ToolContext, ToolResult, ToolResult, _compact(), fit_compartmental_models() (+2 more)

### Community 45 - "Identity & Audit Provenance"
Cohesion: 0.25
Nodes (10): file_sha256(), SHA-256 of a file's bytes (for dataset integrity), or 'n/a' if unreadable., _orch(), Identity-aware audit + provenance: actor/reason are tamper-evident, the chain st, test_actor_and_reason_recorded_and_chain_verifies(), test_file_sha256_stable_and_missing_safe(), test_human_review_is_a_signed_audit_entry(), test_session_creation_stamps_provenance_genesis() (+2 more)

### Community 46 - "HTTP API Tests"
Cohesion: 0.22
Nodes (5): _poll(), HTTP-layer tests: endpoints, bearer-token auth, ownership, role overrides, and t, test_jobmanager_reports_done_and_error(), test_nlme_runs_as_background_job(), test_scm_submits_a_job()

### Community 47 - "Parameter Uncertainty"
Cohesion: 0.24
Nodes (9): Any, _empty_uncertainty(), _is_num(), _numeric_hessian(), _parameter_uncertainty(), Multiplicative factor on the typical parameter for one subject., Central-difference Hessian of a scalar function at ``x``.      Used both for the, Uncertainty payload when standard errors are unavailable. (+1 more)

### Community 48 - "LLM Client Integration"
Cohesion: 0.28
Nodes (4): Any, Anthropic Claude client., RealLLM, Tool

### Community 49 - "Run Provenance Tracking"
Cohesion: 0.28
Nodes (8): Path, collect_provenance(), _git_sha(), _pkg_version(), Run provenance: software versions, platform, and content hashes.  Captured into, Short git SHA of the working tree, or 'n/a' outside a repo., Software/platform fingerprint of this run (cached; constant per process)., test_provenance_has_versions_and_is_constant()

### Community 50 - "Privacy-Safe Schema Extraction"
Cohesion: 0.33
Nodes (8): Any, DataFrame, extract_schema(), _num_summary(), SchemaExtractor — the privacy boundary.  Before any dataset is described to the, Produce a metadata-only summary safe to send to the LLM., _safe(), Series

### Community 51 - "Population Model Estimation"
Cohesion: 0.22
Nodes (9): population_fit(), Estimate a population PK model by the requested NLME method.      Args:, focei_result(), focei_unc(), One FOCE-I fit WITH the asymptotic covariance pass, reused by all     uncertaint, FOCE-I covariance is reproducible for identical inputs (no RNG). Uses a     tiny, saem_result(), test_saem_deterministic_same_seed() (+1 more)

### Community 52 - "Population Test Fixtures"
Cohesion: 0.22
Nodes (9): _make_population(), population(), Smaller cohort for the (expensive) covariance pass — the full OFV Hessian     is, A subject with too few points must not crash the fit (skipped or graceful)., Fitting a 2-subject cohort completes without error., Build a seeded ``oral_1cmt`` population with lognormal IIV + prop. error.      F, test_sparse_subject_is_handled(), test_tiny_cohort_does_not_crash() (+1 more)

### Community 53 - "QC Agent Tools"
Cohesion: 0.32
Nodes (7): Any, PharmState, ToolContext, ToolResult, _check(), QC Agent tools: independent diagnostic review of an NCA analysis.  A subset of t, run_qc()

### Community 54 - "Documentation"
Cohesion: 0.25
Nodes (7): Adding a dataset, Adding your own tool reference (exact concordance), Current result (Theophylline), Reference validation, Running, What it checks, Why bands, not exact equality

### Community 55 - "Structured Logging"
Cohesion: 0.29
Nodes (5): configure_logging(), JsonFormatter, Structured JSON logging — one line per record, no external deps.  Production log, Render a LogRecord (plus any ``extra=`` fields) as a single JSON line., LogRecord

### Community 56 - "Covariate Model Testing"
Cohesion: 0.29
Nodes (7): cov_fit(), _make_cov_population(), Population with a true WT-on-CL power effect (CL=5*(WT/70)^beta) plus an     ind, SCM adds the real WT-on-CL effect and never adds the noise AGE effect.     Seria, The ProcessPool path is deterministic (FOCE-I has no RNG) and must select     th, test_scm_parallel_matches_serial_selection(), test_scm_selects_true_covariate_and_rejects_noise()

### Community 57 - "Application Settings"
Cohesion: 0.33
Nodes (4): Path, Roots a dataset path may be read from (anti path-traversal)., Settings, BaseSettings

### Community 58 - "HTTP Error Handling"
Cohesion: 0.47
Nodes (6): _error_body(), http_exception_handler(), Assign/propagate a request id, time the request, and emit a structured     acces, request_context(), JSONResponse, Request

### Community 59 - "Censored Data Fitting"
Cohesion: 0.40
Nodes (5): _censored_population(), _cv_to_omega2(), m3_fit(), Convert a lognormal %CV/100 to the variance omega2 = ln(1 + cv^2)., Population with terminal samples driven below an LLOQ; BLQ rows flagged     (obs

### Community 60 - "Static File Serving"
Cohesion: 0.50
Nodes (4): Serve a real static file if it exists, else index.html for client-side         r, _spa_fallback(), _spa_root(), _FR

### Community 61 - "Residual Error Computation"
Cohesion: 0.50
Nodes (4): _ind_obj(), Per-observation residual variance under the configured error model., Individual conditional objective (the quantity minimized for the EBE).      ind_, _residual_variance()

### Community 62 - "Below-LLOQ Censoring Tests"
Cohesion: 0.33
Nodes (4): A fit without an LLOQ reports n_blq == 0 (default = drop, byte-identical)., With ~25% of samples below the LLOQ, the M3 censored likelihood still     recove, test_default_path_reports_no_blq(), test_m3_recovers_parameters_with_censoring()

## Knowledge Gaps
- **18 isolated node(s):** `ndarray`, `_FitData`, `ndarray`, `Any`, `Path` (+13 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **8 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `PharmState` connect `Shared Analysis State` to `Access Control & Session Management`, `Data Models & Pharmacokinetics`, `Regulatory Report Generation`, `Population Model Export`, `State Management & Write Access`, `Agent Routing & Supervision`, `Bioequivalence Agent Tools`, `Population PK Agent Tools`, `Session Persistence`, `Workflow Orchestration`, `Tool Registry & Review`, `Adversarial Review Engine`, `Compartmental Agent Tools`, `Core Framework & Configuration`, `Report Generation Tools`, `CDISC Export`, `QC Agent Tools`, `Data Loading & Profiling`?**
  _High betweenness centrality (0.359) - this node is a cross-community bridge._
- **Why does `get_model()` connect `PK Model Fitting` to `Data Models & Pharmacokinetics`, `Dose Sweep Analysis`, `Compartmental Model Parameters`, `Goodness-of-Fit Diagnostics`, `Generic ODE Simulation`, `NLME Estimation Tests`, `Model Residual Diagnostics`, `Dose Forecasting & TDM`, `Population Model Estimation`, `Population Test Fixtures`, `Population Model Fitting`, `PK Dataset Model Fitting`, `Covariate Model Testing`, `Censored Data Fitting`, `True NLME Estimation`, `Empirical Bayes Estimation`?**
  _High betweenness centrality (0.168) - this node is a cross-community bridge._
- **Why does `ToolResult` connect `Compartmental Agent Tools` to `Data Models & Pharmacokinetics`, `Regulatory Report Generation`, `Agent Routing & Supervision`, `Bioequivalence Agent Tools`, `Population PK Agent Tools`, `Tool Registry & Review`, `Core Framework & Configuration`, `Report Generation Tools`, `QC Agent Tools`, `Shared Analysis State`, `Data Loading & Profiling`?**
  _High betweenness centrality (0.120) - this node is a cross-community bridge._
- **Are the 85 inferred relationships involving `PharmState` (e.g. with `Agent` and `AgentResult`) actually correct?**
  _`PharmState` has 85 INFERRED edges - model-reasoned connections that need verification._
- **Are the 74 inferred relationships involving `ToolContext` (e.g. with `Agent` and `AgentResult`) actually correct?**
  _`ToolContext` has 74 INFERRED edges - model-reasoned connections that need verification._
- **Are the 31 inferred relationships involving `Orchestrator` (e.g. with `ChatRequest` and `DoseSweepRequest`) actually correct?**
  _`Orchestrator` has 31 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Agent base.  An agent owns a system prompt, a set of bound tools (by ownership i`, `Compact, privacy-safe view of state for LLM tool selection.`, `Domain agent definitions (Phase 1 roster).` to the rest of the system?**
  _318 weakly-connected nodes found - possible documentation gaps or missing edges._