# PopPK Physiological Plausibility Checker

> **Disclaimer:** Deterministic physiological plausibility screening only. This module does not constitute regulatory validation, clinical approval, or proof of model validity.

## 1. Intended Use & Non-Intended Use

### Intended Use
- **Pre-review & QA screening** of population pharmacokinetic (`PopPK`) parameter tables before simulation, reporting, or downstream regulatory authoring.
- **Deterministic detection** of mathematical identities/impossibilities (non-positive structural parameters, negative variances, non-positive-semidefinite $\Omega/\Sigma$ matrices, correlations $|r_{ij}| > 1$, negative predicted concentrations, incompatible physical units), physiological reference violations, and modeling heuristics.
- **Traceable human justification governance** that records why an extreme or atypical physiological parameter (e.g., extrahepatic blood-esterase clearance exceeding hepatic blood flow, or flip-flop kinetics in an immediate-release formulation) was accepted by a qualified reviewer without erasing the underlying finding.

### Non-Intended Use
- **Not regulatory or scientific validation:** Passing all 20 plausibility rules (`SCREENING_PASS`) never proves that a model is statistically well-specified, identifiable, or fit for regulatory submission.
- **Not an automatic parameter modifier:** The engine never rewrites, clips, drops, or imputes submitted model parameters.

---

## 2. Rule Taxonomy & Provenance (`PL-001` through `PL-020`)

All rules are defined in `backend/app/compute/poppk_plausibility/catalog.json` (`catalog_version = "1.1.0"`). Every rule explicitly separates:
- `rule_type`:
  - `"mathematical_identity"`: Hard mathematical or dimensional identities (`PL-001`, `PL-002`, `PL-003`, `PL-007`, `PL-008`, `PL-013`, `PL-014`, `PL-015`, `PL-019`, `PL-020`).
  - `"physiological_reference"`: Comparisons against published anatomical/physiological reference flows or spaces (`PL-004`, `PL-005`, `PL-010`, `PL-011`, `PL-017`, `PL-018`).
  - `"modeling_heuristic"`: Empirical PopPK modeling review heuristics (`PL-006`, `PL-009`, `PL-012`, `PL-016`).
- `approval_status`: Every rule and every special-population scaling rule is set to `"pending_owner_review"` until formally reviewed and approved by the repository owner.
- `reference_basis`: Complete, source-verified bibliographic citations (author(s), full title, journal/book, year, volume/edition, and page numbers).

| Rule ID | Rule Type | Category | Default Severity | Override Allowed | Summary |
|---|---|---|---|---|---|
| `PL-001` | `mathematical_identity` | `structural_domain` | `CRITICAL` | `False` | Structural PK parameters ($CL, V, Q, k_a, t_{1/2}$) must be finite and $> 0$. |
| `PL-002` | `mathematical_identity` | `unit_compatibility` | `CRITICAL` | `False` | Units must be explicit, recognized, dimensionally compatible, and free of duplicate canonical aliases. |
| `PL-003` | `mathematical_identity` | `bioavailability` | `HIGH` | `True` ($F>1$) / `False` ($F\le 0$) | Bioavailability $F \in (0, 1]$ (or $(0, 100]\%$ when typed as `%`). |
| `PL-004` | `physiological_reference` | `clearance_physiology` | `WARN` | `True` | Systemic whole-blood $CL$ ($CL_{\text{blood}} = CL_{\text{plasma}} / R_b$) vs. scaled hepatic blood flow ($Q_H = 90\text{ L/h}$) and cardiac output ($300\text{ L/h}$) when `elimination_pathway in {"hepatic", "mixed"}` and `modality in {"small_molecule", "peptide"}`. |
| `PL-005` | `physiological_reference` | `clearance_physiology` | `WARN` | `True` | Renal clearance $CL_R \le CL$ (`override_allowed=False`), $CL_R \le Q_{R,\text{plasma}}$ ($36\text{ L/h}$), and $CL_R \le f_u \cdot \text{GFR}$ ($7.5\text{ L/h}$) without known active tubular secretion. |
| `PL-006` | `modeling_heuristic` | `volume_physiology` | `HIGH` | `True` | Unusual estimated central compartment volume $V_c$ compared with typical plasma volume ($0.03\text{ L/kg}$) and tissue distribution bounds ($20\text{–}50\text{ L/kg}$). |
| `PL-007` | `mathematical_identity` | `volume_physiology` | `HIGH` | `True` (bounds) / `False` (sum mismatch) | Mammillary central-elimination identity $V_{ss} = V_c + \sum V_p$ ($\pm 5\%$, requiring `compartmental_structure == "mammillary_central_elimination"`) and $[0.04, 50]\text{ L/kg}$ screening bounds. |
| `PL-008` | `mathematical_identity` | `derived_kinetics` | `HIGH` | `True` (range) / `False` (identity) | Strictly distinguishes terminal elimination half-life (`t_half`, $t_{1/2,\beta} = \ln(2)/\beta$) from effective half-life (`t_half_eff`, $t_{1/2,\text{eff}} = \ln(2)V_{ss}/CL$); never treats them as interchangeable. |
| `PL-009` | `modeling_heuristic` | `absorption_kinetics` | `WARN` | `True` | Route- and formulation-dependent (`formulation_release`) screening of $k_a$ and flip-flop kinetics ($k_a \le \beta$); flip-flop is expected for ER/depot/SC mAbs and is an identifiability advisory for immediate-release models. |
| `PL-010` | `physiological_reference` | `allometric_scaling` | `WARN` | `True` | Allometric weight exponent on clearance/flow near $0.75$ (pass $[0.60, 0.90]$, warn $[0.30, 1.20]$). |
| `PL-011` | `physiological_reference` | `allometric_scaling` | `WARN` | `True` | Allometric weight exponent on volume near $1.0$ (pass $[0.80, 1.20]$, warn $[0.50, 1.50]$). |
| `PL-012` | `modeling_heuristic` | `random_effects` | `WARN` | `True` | Log-normal IIV $\%CV \le 100\%$ (warn) / $\le 200\%$ (high) and $\omega^2 \leftrightarrow \%CV$ consistency. |
| `PL-013` | `mathematical_identity` | `random_effects` | `CRITICAL` | `False` | All variance terms ($\omega^2, \sigma^2$, matrix diagonals) must be finite and $\ge 0$. |
| `PL-014` | `mathematical_identity` | `random_effects` | `CRITICAL` | `False` | $\Omega$ and $\Sigma$ matrices must be square, symmetric ($\le 10^{-8}$), and positive semidefinite ($\lambda_{\min} \ge -10^{-9}$). |
| `PL-015` | `mathematical_identity` | `random_effects` | `CRITICAL` | `False` ($|r|>1$) / `True` ($|r|\ge 0.95$) | Correlations $r_{ij} \in [-1, 1]$ (`CRITICAL` outside) and collinearity screen $|r_{ij}| < 0.95$ (`WARN`). |
| `PL-016` | `modeling_heuristic` | `residual_error` | `WARN` | `True` | Residual error terms finite, $\ge 0$, not all zero, and proportional RUV $\le 50\%\text{ CV}$ (warn) / $\le 100\%\text{ CV}$ (high). |
| `PL-017` | `physiological_reference` | `biologics_mab` | `HIGH` | `True` | Monoclonal antibody $V_c \in [0.03, 0.10]\text{ L/kg}$ (vascular plasma + interstitial space). |
| `PL-018` | `physiological_reference` | `biologics_mab` | `HIGH` | `True` | Monoclonal antibody linear $CL \in [0.003, 0.05]\text{ L/h}$ ($70\text{ kg}$ eq; $\approx 0.07\text{–}1.2\text{ L/day}$). |
| `PL-019` | `mathematical_identity` | `micro_rate_consistency` | `HIGH` | `False` (micro-rates) | Micro-rate consistency ($k_{12}=Q/V_c$, $k_{21}=Q/V_p$, $k_{10}=CL/V_c$ within $\pm 2\%$) and $Q \le$ cardiac output. |
| `PL-020` | `mathematical_identity` | `prediction_plausibility` | `CRITICAL` | `False` | Supplied `PRED`/`IPRED` points must be finite, nonnegative ($\ge 0$), and $\le 10^6\text{ mg/L}$. |

---

## 3. Scientific Applicability & Context Requirements

### Clearance Semantics, Elimination Context & Blood-to-Plasma Ratio (`PL-004`)
- **Scale Disambiguation:** On extravascular or unknown routes (`oral`, `sc`, `im`, `inhalation`, `unknown`), `scale="auto"` on clearance/volume parameters without an explicit valid bioavailability `F` resolves to `resolved_scale="ambiguous"`, causing `PL-004`, `PL-005`, `PL-006`, `PL-007`, `PL-017`, `PL-018`, and `PL-019` to return `UNRESOLVED` rather than conflating apparent `CL/F` or `Vc/F` with systemic `CL` or `Vc`.
- **Blood-to-Plasma Ratio ($R_b = C_{\text{blood}} / C_{\text{plasma}}$):** Whole-blood physiological flows ($Q_H = 90\text{ L/h}$, $CO = 300\text{ L/h}$ per Rowland & Tozer 2011 / Davies & Morris 1993) are never multiplied or divided by $R_b$ in `context.py`. Instead, when `clearance_matrix == "plasma"`, `blood_to_plasma_ratio` ($R_b > 0$) is required and plasma clearance is converted to whole-blood clearance via $CL_{\text{blood}} = CL_{\text{plasma}} / R_b$ before comparison with $Q_H$ and $CO$.
- **Elimination Pathway & Modality:** `PL-004` returns `NOT_APPLICABLE` when `elimination_pathway in {"renal", "catabolic", "extrahepatic"}` or `modality in {"monoclonal_antibody", "biologic_other"}`, and returns `UNRESOLVED` when `elimination_pathway == "unknown"` or `modality == "unknown"`.

### Special Populations (`context.py`)
- Adjustments for `pediatric`, `neonate`, `pregnancy`, `elderly`, `renal_impairment`, and `hepatic_impairment` require **both**:
  1. The mandatory clinical covariate (`age_years` for `pediatric`/`neonate`/`elderly`, `gestational_age_weeks` for `pregnancy`, `egfr_ml_min_1_73m2` for `renal_impairment`, `child_pugh` for `hepatic_impairment`), **and**
  2. `reference_physiology.special_population_rules_status[population] == "approved"` in `catalog.json`.
- Because all special-population rules are currently `"pending_owner_review"`, special-population models deterministically return `UNRESOLVED` (`INCOMPLETE_CONTEXT`) rather than applying unapproved heuristic scaling factors.

---

## 4. Deep Immutability & Human Justification Governance

- **Deep Immutability (`schema.py`, `report.py`, `benchmark.py`):** Every model inherits from `_FrozenModel` (`ConfigDict(extra="forbid", frozen=True)`) and wraps all mapping fields in `types.MappingProxyType` (`FrozenMap`, `FrozenFloatMap`, `FrozenIntMap`, `FrozenStrMap`) and all sequence fields in `tuple`. Any attempt to mutate a nested mapping or sequence in `PopPKModelInput`, `RuleCatalog`, `PlausibilityFinding`, `PlausibilityReport`, or `BenchmarkSummary` raises `TypeError`.
- **Non-Destructive Overrides:** A `JustificationRecord` requires `rationale`, `scientific_basis_or_citation`, `reviewer`, and `timestamp`. Overridable findings transition `override_status` to `"accepted_with_justification"` while retaining `status="FLAGGED"`. Non-overridable (`override_allowed=False` or `CRITICAL`) findings transition `override_status` to `"rejected"`.

---

## 5. Offline Software Regression Harness (`benchmark.py`)

`backend/app/compute/poppk_plausibility/benchmark.py` executes 23 deterministic cases (`known_plausible_synthetic`, `corrupted_planted_violation`, `published_model_reference`) to verify software execution, planted-violation recall (`1.0`), and false-flag rate (`0.0`) on synthetic fixtures. **This harness is a software regression check only and is not scientific or clinical validation.**
