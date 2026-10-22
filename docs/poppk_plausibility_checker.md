# PopPK Physiological Plausibility Checker

> **Disclaimer:** Deterministic physiological plausibility screening only. This module does not constitute regulatory validation, clinical approval, or proof of model validity.

## 1. Intended Use & Non-Intended Use

### Intended Use
- **Pre-review & QA screening** of population pharmacokinetic (`PopPK`) parameter tables before simulation, reporting, or downstream regulatory authoring.
- **Deterministic detection** of mathematical impossibilities (non-positive structural parameters, negative variances, non-positive-semidefinite $\Omega/\Sigma$ matrices, correlations $|r_{ij}| > 1$, negative predicted concentrations, incompatible physical units) and physiologically implausible parameter estimates relative to context-scaled human reference bounds.
- **Traceable human justification governance** that records why an extreme or atypical physiological parameter (e.g., extrahepatic blood-esterase clearance exceeding hepatic blood flow, or flip-flop kinetics in an extended-release formulation) was accepted by a qualified reviewer without erasing the underlying finding.

### Non-Intended Use
- **Not regulatory validation:** Passing all 20 plausibility rules (`SCREENING_PASS`) never proves that a model is statistically well-specified, identifiable, or fit for regulatory submission.
- **Not an automatic parameter modifier:** The engine never rewrites, clips, drops, or imputes submitted model parameters.

---

## 2. Rule Taxonomy (`PL-001` through `PL-020`)

All rules are defined in `backend/app/compute/poppk_plausibility/catalog.json` (version `1.0.0`) with explicit source classifications (`mathematical_identity`, `physiological_reference`, `allometric_theory`, `modeling_heuristic`) and owner approval tracking (`pending_owner_review` by default).

| Rule ID | Slug | Category | Default Severity | Override Allowed | Summary |
|---|---|---|---|---|---|
| `PL-001` | `positive_finite_parameters` | `parameter_domain` | `CRITICAL` | `False` | Structural PK parameters ($CL, V, Q, k_a, t_{1/2}$) must be finite and $> 0$. |
| `PL-002` | `unit_compatibility_and_normalization` | `units` | `CRITICAL` | `False` | Units must be explicit, recognized, and dimensionally compatible with the parameter class. |
| `PL-003` | `bioavailability_bounds` | `absorption_bioavailability` | `HIGH` | `True` ($F>1$) / `False` ($F\le 0$) | Absolute bioavailability $F \in (0, 1]$ (or $(0, 100]\%$ when explicitly typed as `%`). |
| `PL-004` | `systemic_clearance_vs_hepatic_flow` | `physiological_clearance` | `WARN` | `True` | Systemic $CL$ (distinguished from $CL/F$) vs. scaled hepatic blood flow ($Q_H \approx 90\text{ L/h}$) and cardiac output ($\approx 300\text{ L/h}$). |
| `PL-005` | `renal_clearance_vs_gfr_and_renal_flow` | `physiological_clearance` | `WARN` | `True` | Renal clearance $CL_R$ vs. filtration ($f_u \cdot \text{GFR}$) and effective renal plasma flow ($Q_{R,\text{plasma}} \approx 36\text{ L/h}$). |
| `PL-006` | `central_volume_vs_physiological_space` | `physiological_volume` | `HIGH` | `True` | Central volume $V_c$ compared with plasma floor ($0.03\text{ L/kg}$) and tissue ceiling ($20\text{–}50\text{ L/kg}$). |
| `PL-007` | `steady_state_volume_consistency` | `physiological_volume` | `HIGH` | `True` | $V_{ss}$ physiological bounds ($[0.03, 100]\text{ L/kg}$) and equality with $V_c + \sum V_p$ ($\pm 5\%$). |
| `PL-008` | `half_life_clearance_volume_consistency` | `derived_kinetics` | `HIGH` | `True` | Reported $t_{1/2}$ consistency with $\ln(2)\cdot V_c/CL$ ($\pm 10\%$) and $[0.01, 5000]\text{ h}$ window. |
| `PL-009` | `absorption_ka_and_flip_flop` | `absorption_bioavailability` | `WARN` | `True` | Absorption $k_a \in [0.05, 10]\text{ h}^{-1}$ and flip-flop kinetics detection ($k_a \le k_e = CL/V_c$). |
| `PL-010` | `allometric_clearance_exponent` | `allometry` | `WARN` | `True` | Allometric weight exponent on clearance/flow near $0.75$ (pass $[0.60, 0.90]$, warn $[0.30, 1.20]$). |
| `PL-011` | `allometric_volume_exponent` | `allometry` | `WARN` | `True` | Allometric weight exponent on volume near $1.0$ (pass $[0.80, 1.20]$, warn $[0.50, 1.50]$). |
| `PL-012` | `iiv_magnitude_review` | `random_effects` | `WARN` | `True` | Log-normal IIV $\%CV \le 100\%$ (warn) / $\le 200\%$ (high) and $\omega^2 \leftrightarrow \%CV$ consistency. |
| `PL-013` | `variance_nonnegative_finite` | `random_effects` | `CRITICAL` | `False` | All variance terms ($\omega^2, \sigma^2$, matrix diagonals) must be finite and $\ge 0$. |
| `PL-014` | `covariance_matrix_psd` | `random_effects` | `CRITICAL` | `False` | $\Omega$ and $\Sigma$ matrices must be square, symmetric ($\le 10^{-8}$), and positive semidefinite ($\lambda_{\min} \ge -10^{-9}$). |
| `PL-015` | `correlation_bounds` | `random_effects` | `CRITICAL` | `False` ($|r|>1$) / `True` ($|r|\ge 0.95$) | Correlations $r_{ij} \in [-1, 1]$ (`CRITICAL` outside) and collinearity screen $|r_{ij}| < 0.95$ (`WARN`). |
| `PL-016` | `residual_error_plausibility` | `residual_error` | `WARN` | `True` | Residual error terms finite, $\ge 0$, not all zero, and proportional RUV $\le 50\%\text{ CV}$ (warn) / $\le 100\%\text{ CV}$ (high). |
| `PL-017` | `mab_central_volume` | `biologics_mab` | `HIGH` | `True` | Monoclonal antibody $V_c \in [0.03, 0.10]\text{ L/kg}$ (approximating vascular plasma + interstitial space). |
| `PL-018` | `mab_linear_clearance` | `biologics_mab` | `HIGH` | `True` | Monoclonal antibody linear $CL \in [0.003, 0.05]\text{ L/h}$ ($70\text{ kg}$ eq; $\approx 0.07\text{–}1.2\text{ L/day}$). |
| `PL-019` | `intercompartmental_micro_rate_consistency` | `structural_consistency` | `HIGH` | `False` (micro-rates) | Micro-rate consistency ($k_{12}=Q/V_c$, $k_{21}=Q/V_p$, $k_{10}=CL/V_c$ within $\pm 2\%$) and $Q \le$ cardiac output. |
| `PL-020` | `model_prediction_plausibility` | `predictions` | `CRITICAL` | `False` | Supplied `PRED`/`IPRED` points must be finite, nonnegative ($\ge 0$), and $\le 10^6\text{ mg/L}$. |

---

## 3. Unit Normalization & Context Requirements

### Deterministic Unit Normalization (`units.py`)
- **Clearance / Flow (`CL`, `CL_R`, `Q`):** `L/h`, `mL/min`, `mL/h`, `L/day`, `L/d`, `mL/day`, `L/min` $\to$ canonical `L/h`; `L/h/kg`, `mL/min/kg` $\to$ `L/h/kg`.
- **Volume (`Vc`, `Vp`, `Vss`):** `L`, `mL`, `dL` $\to$ canonical `L`; `L/kg`, `mL/kg` $\to$ `L/kg`.
- **Rate Constants (`ka`, `ke`, `k10`, `k12`, `k21`):** `1/h`, `h^-1`, `/h`, `1/min`, `1/d`, `1/day` $\to$ canonical `1/h`.
- **Half-Life (`t_half`):** `h`, `hr`, `min`, `d`, `day` $\to$ canonical `h`.
- **Bioavailability (`F`):** `fraction`, `unitless`, `1`, `-`, `%`, `percent` $\to$ canonical `fraction`.
- Missing units on dimensional parameters yield `UNRESOLVED` rather than guessing; unrecognized or dimensionally incompatible units (e.g., `mg` for `CL`) yield `CRITICAL` `FLAGGED`.

### Explicit Context Resolution (`context.py`)
- **Apparent vs. Systemic Disambiguation:** `CL/F` and `Vc/F` (or `scale="apparent"`) are never conflated with systemic `CL` and `Vc`. Rule `PL-004` marks apparent-only `CL/F` as `NOT_APPLICABLE` unless bioavailability $F$ is resolved.
- **Missing Context $\to$ `UNRESOLVED`:**
  - `population="pediatric"` or `"neonatal"` without `reference_weight_kg` returns `UNRESOLVED` on weight-scaled rules (`PL-004`, `PL-006`, `PL-007`) so adult $70\text{ kg}$ bounds are never silently applied to children.
  - `population="pregnancy"` without `gestational_age_weeks` returns `UNRESOLVED`.
  - `population="hepatic_impairment"` without `child_pugh` (`"A"`, `"B"`, or `"C"`) returns `UNRESOLVED`.
  - `population="renal_impairment"` without `egfr_ml_min_1_73m2` returns `UNRESOLVED` for `PL-005`.
  - `modality="unknown"` returns `UNRESOLVED` for mAb rules (`PL-017`, `PL-018`).

---

## 4. Human Justification & Override Governance

A `JustificationRecord` requires four non-blank provenance fields in addition to `rule_id` and `parameter`:
1. `rationale`
2. `scientific_basis_or_citation`
3. `reviewer`
4. `timestamp`

### Non-Destructive Rules
- **Original finding is never deleted:** Even when a valid justification is supplied, `finding.status` remains `"FLAGGED"` so the audit trail and report table always show what was flagged.
- **Overridable findings (`override_allowed=True`, `severity in {"WARN", "HIGH"}`):** Transition `override_status` from `"justification_required"` to `"accepted_with_justification"`. If all flagged findings have accepted justifications and no `CRITICAL` or `UNRESOLVED` findings exist, `report.screening_status` becomes `"REVIEW_ACCEPTED_WITH_JUSTIFICATION"`.
- **Non-overridable mathematical/physical impossibilities (`override_allowed=False` or `severity == "CRITICAL"`):** Transition `override_status` to `"rejected"`, retain `report.screening_status == "CRITICAL_IMPLAUSIBILITY"`, and attach an explicit rejection note.

---

## 5. Limitations

1. **Human linear compartmental focus:** Reference physiological flows and spaces in `catalog.json` apply to human systemic pharmacokinetics (`species="human"`). Preclinical species return `UNRESOLVED`.
2. **Saturable / TMDD kinetics:** Rules `PL-008` and `PL-019` evaluate linear clearance/volume identities. Models with Michaelis-Menten ($V_{\max}, K_m$) or quasi-steady-state TMDD parameters should set `has_tmdd_or_nonlinear_cl=True` and supply a human justification where applicable.
3. **Screening thresholds are heuristics, not clinical cutoffs:** Passing screening does not imply adequacy of goodness-of-fit, covariate selection, or predictive performance (VPC/NPDE).

---

## 6. Offline Benchmark Methodology

`backend/app/compute/poppk_plausibility/benchmark.py` executes 23 deterministic offline cases across three groups:
1. **`known_plausible_synthetic` (3 cases):** Adult IV 2-compartment small molecule, adult oral 1-compartment small molecule, and 20 kg pediatric IV small molecule. Target `false_flag_rate_on_clean_models == 0.0`.
2. **`corrupted_planted_violation` (18 cases):** Targeted single- and dual-rule corruptions spanning all 20 rules (`PL-001` through `PL-020`) plus incomplete pediatric context. Target `planted_violation_recall == 1.0` and `unresolved_rate_on_incomplete_context == 1.0`.
3. **`published_model_reference` (2 cases):** Each entry carries an explicit `source_status` (`"literature_reference"` for Dirks & Meibohm 2010 typical IgG1 mAb parameters; `"synthetic_placeholder"` for the justified esterase-cleared compound).

---

## 7. How to Add or Review a Rule

1. Add the rule metadata and thresholds to `backend/app/compute/poppk_plausibility/catalog.json` with `approval.status = "pending_owner_review"`.
2. Implement the pure evaluation function in `rules_structural.py`, `rules_stochastic.py`, or `rules.py` and register it in `evaluate_all_rules`.
3. Add RED/GREEN unit, boundary, and mutation-guard tests in `backend/tests/test_poppk_plausibility.py` and `backend/tests/test_poppk_plausibility_benchmark_and_mutation.py`.
4. Only set `approval.status = "approved"` (with `by` and ISO `date`) after explicit owner review.

---

## 8. MCP & Tool Adapter Integration Plan

- **Isolated Read-Only Tool Adapter (`backend/app/tools/poppk_plausibility_tools.py`):** Wraps `evaluate_poppk_plausibility` as a read-only `Tool` (`screen_poppk_plausibility`, owned by `"poppk"`, `writes={}`) that never mutates `PharmState` and requires no changes to guarded files (`builtins.py`, `main.py`, `orchestrator.py`, `pharmstate.py`).
- **Runtime Registration Gate:** Wiring `TOOLS` from `poppk_plausibility_tools.py` into `backend/app/tools/builtins.py` or exposing an MCP server endpoint is intentionally deferred to a separate owner-approved integration PR so guarded runtime files remain untouched in this feature slice.
