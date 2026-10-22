# Protocol Simulation Scenario & Audit Summary Template

Use this confined reference template when completing **Step 4** and **Step 10** of `pharmagent.protocol_simulation`.

## 1. Playbook Provenance Header
- **Playbook ID**: `pharmagent.protocol_simulation`
- **Playbook Version**: `1.0.0`
- **Canonical Digest**: `<populate from loaded ScientificPlaybook.canonical_digest>`
- **Approval Status**: `pending_owner_review`
- **Human Review Required**: `true`

## 2. Scenario Matrix Template

| Scenario ID | Question / Stratum | Reviewed Tool | Key Inputs (`dose`, `tau`, `n_doses`, `wt`, `rate`) | Output Metrics (`cmax`, `auc_tau`, `cavg`, `ctrough`, `pta`) | Status |
|---|---|---|---|---|---|
| `SCEN-01` | Reference regimen | `simulate_pk_profile` | `<from tool args>` | `<from tool result only>` | `computed` / `UNRESOLVED` |
| `SCEN-02` | Candidate dose grid | `run_dose_sweep` | `<from tool args>` | `<from tool result only>` | `computed` / `UNRESOLVED` |
| `SCEN-03` | Virtual population PTA | `run_clinsim` | `<from tool args>` | `<from tool result only>` | `computed` / `UNRESOLVED` |

## 3. Mandatory Verification Checklist
- [ ] Dataset schema and units checked via `load_dataset`, `profile_pk_dataset`, and `validate_cdisc`.
- [ ] Every numerical value in the table traces to a deterministic tool (`compute_nca`, `simulate_pk_profile`, `run_dose_sweep`, `run_clinsim`, `run_special_population`, `run_pediatric_simulation`, `fit_exposure_response`, `select_optimal_dose`).
- [ ] No silent unit conversion or covariate imputation was performed.
- [ ] No PBPK/QSP simulation or regulatory endorsement was claimed.
- [ ] Independent quality and adversarial checks run via `run_qc` / `adversarial_review` where applicable.
- [ ] Final protocol decision gated on human pharmacometrician review.
