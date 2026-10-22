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

Use this skill to screen a population pharmacokinetic (`PopPK`) model parameter table for mathematical impossibilities and physiological implausibilities before simulation, reporting, or regulatory authoring.

## Mandatory Guardrails

1. **Screening Only (Never Validation):** Always refer to the output as `"physiological plausibility screening"`. Never claim a model is `"valid"` or `"validated"`.
2. **Never Silently Alter Parameters:** Report flagged parameters with their rule ID, observed and normalized values, expected range, and scientific rationale. Never rewrite, clip, or drop implausible values.
3. **Require Explicit Context:** Do not assume a 70 kg healthy adult when screening pediatric, neonatal, pregnancy, or organ-impaired models. Missing required context must remain `UNRESOLVED`.
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
    },
    "parameters": [
        {"name": "CL", "value": 12.0, "unit": "L/h", "scale": "systemic"},
        {"name": "Vc", "value": 25.0, "unit": "L", "scale": "systemic"},
    ],
})
print(report.to_markdown())
```
