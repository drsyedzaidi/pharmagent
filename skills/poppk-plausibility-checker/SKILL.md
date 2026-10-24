---
name: poppk-plausibility-checker
description: >
  Deterministic physiological plausibility screening for population PK (PopPK)
  models across 20 domain, unit, physiological flow/volume, allometric,
  random-effect, residual-error, and mAb rules with non-destructive human
  justification governance.
user_invocable: true
---

# PopPK Physiological Plausibility Checker Skill

Use this skill to screen a population pharmacokinetic (`PopPK`) model parameter table for mathematical impossibilities, physiological reference violations, and modeling heuristics before simulation, reporting, or regulatory authoring.

## Mandatory Guardrails

1. **Screening Only (Never Validation):** Always refer to the output as `"physiological plausibility screening"`. Never claim a model is `"valid"` or `"validated"`, and never describe the synthetic benchmark harness as scientific validation.
2. **Never Silently Alter Parameters:** Report flagged parameters with their rule ID, `rule_type` (`mathematical_identity`, `physiological_reference`, or `modeling_heuristic`), `approval_status` (`pending_owner_review`), observed and normalized values, expected range, and verified bibliographic citations. Never rewrite, clip, or drop implausible values.
3. **Require Explicit Context & Applicability:**
   - `PL-004` applies only when clearance semantics (`systemic` scale or `CL/F` with valid `F`, plus `blood` matrix or `plasma` matrix with `blood_to_plasma_ratio` $R_b = C_{\text{blood}} / C_{\text{plasma}}$ where $CL_{\text{blood}} = CL_{\text{plasma}} / R_b$) and elimination context (`elimination_pathway in {"hepatic", "mixed"}` and `modality in {"small_molecule", "peptide"}`) support comparison with hepatic blood flow $Q_H$.
   - `PL-006` screens for unusual estimated central compartment volume ($V_c$) relative to typical vascular plasma volume ($0.03\text{ L/kg}$) and tissue distribution bounds as a `modeling_heuristic` (never framed as universally physically impossible).
   - `PL-007` enforces $V_{ss} = V_c + \sum V_p$ only when `compartmental_structure == "mammillary_central_elimination"`.
   - `PL-008` strictly distinguishes terminal elimination half-life (`t_half`, $t_{1/2,\beta} = \ln(2)/\beta$) from effective half-life (`t_half_eff`, $t_{1/2,\text{eff}} = \ln(2)V_{ss}/CL$); they are never interchangeable.
   - `PL-009` evaluates $k_a$ and flip-flop kinetics ($k_a \le \beta$) conditional on `route` and `formulation_release` (`immediate_release`, `extended_release`, `depot_or_long_acting`, `unknown`), explicitly noting that flip-flop kinetics are expected for modified-release/depot/SC mAb formulations and are not inherently erroneous.
   - Special populations (`pediatric`, `neonate`, `pregnancy`, `elderly`, `renal_impairment`, `hepatic_impairment`) remain `UNRESOLVED` unless both the required clinical covariate (`age_years`, `gestational_age_weeks`, `egfr_ml_min_1_73m2`, `child_pugh`) and an `"approved"` special-population rule in `catalog.json` are available.
4. **Preserve Overridden Findings:** Human justifications (`JustificationRecord`) require `rationale`, `scientific_basis_or_citation`, `reviewer`, and `timestamp`. Valid justifications transition overridable `WARN`/`HIGH` findings to `accepted_with_justification` while keeping `status="FLAGGED"`. Hard mathematical impossibilities (`CRITICAL` / `override_allowed=False`) cannot be overridden.

## Usage

```python
from app.compute.poppk_plausibility import evaluate_poppk_plausibility

report = evaluate_poppk_plausibility({
    "model_id": "candidate_poppk_v1",
    "context": {
        "species": "human",
        "population": "adult",
        "reference_weight_kg": 70.0,
        "route": "iv",
        "modality": "small_molecule",
        "elimination_pathway": "hepatic",
        "clearance_matrix": "blood",
        "compartmental_structure": "mammillary_central_elimination",
        "formulation_release": "immediate_release",
    },
    "parameters": [
        {"name": "CL", "value": 12.0, "unit": "L/h", "scale": "systemic"},
        {"name": "Vc", "value": 25.0, "unit": "L", "scale": "systemic"},
    ],
})
print(report.to_markdown())
```
