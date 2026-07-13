# PharmacometricsBench — design sketch

A benchmark for **agentic clinical pharmacology / pharmacometrics**. Fills the
gap left by Scale's DrugDiscoveryBench (which stops at early *discovery* — target
ID, hits, leads). This one starts where the drug is already a molecule with data:
PK/PD, popPK, PBPK, TDM, BE, and regulatory reasoning.

> Positioning: the first reproducible eval for whether an AI agent can *run* a
> pharmacometric analysis correctly, not just describe one. Thought-leadership +
> a paper + a quality gate for PharmAgent, in one artifact.

Borrows DrugDiscoveryBench's proven methodology: single verifiable answer per
task, rubric-graded, multiple reasoning-effort levels, N runs averaged, hard
timeout — plus a **guided-vs-unguided** contrast to quantify the planning
bottleneck for *this* domain.

---

## 1. Design principles

1. **Deterministic-first grading.** Wherever the answer is a number or a
   category, grade it against a computed oracle — no LLM judge. Reserve the
   LLM judge only for prose/interpretation.
2. **Ground truth from assets you already own.** Your validated tools, published
   PBPK models, and reference datasets *are* the answer key. Minimize new
   labeling.
3. **One verifiable answer per task.** Cmax to a tolerance, a pass/fail, a model
   choice, a cited guideline — never an open essay as the graded object.
4. **Measure the planning lift.** Run each task twice: unguided, then with an
   expert step sequence. The delta quantifies where autonomy fails (the
   DrugDiscoveryBench headline, reproduced for pharmacometrics).
5. **Contamination-resistant.** Seed randomized parameters/errors so answers
   can't be memorized from literature.

---

## 2. Task taxonomy

Twelve categories along the workflow. Each maps to an existing tool or a
published model that supplies ground truth.

| # | Category | Example task | Ground-truth source | Grade tier |
|---|----------|--------------|---------------------|------------|
| 1 | Data engineering & QC | Find the BLQ mishandling / unit error / duplicate dose in this dataset | Seeded-error datasets (`data_tools`, `qc_tools`) | B categorical |
| 2 | NCA | Compute Cmax, AUC₀–∞, t½, CL/F for subject X | `nca_tools` oracle + Theophylline reference | A numeric |
| 3 | Structural PK | Is this 1- or 2-compartment? Estimate CL, V | Simulated data, known params (`compartmental_tools`) | A + B |
| 4 | PopPK | Fit base model + screen weight covariate; report θ, IIV | Theophylline published popPK (`poppk_tools`, `nlme`) | A numeric |
| 5 | PK/PD & E–R | Fit Emax; report EC50, Emax | Simulated with known truth | A numeric |
| 6 | PBPK | Predict AUC/Cmax; is this partition coefficient plausible? | **Your Metformin (×2) + Bupropion published models** (GMFE bands, param values) | A + C |
| 7 | Bioequivalence | 90% CI on GMR; BE pass/fail | `be_tools` oracle | A + B |
| 8 | Dose-proportionality | Power-model slope + CI; proportional? | `dp_tools` oracle | A + B |
| 9 | TDM / Bayesian forecast | Given 2 levels, recommend a dose to hit target | mapbayr oracle (`forecast`) | A numeric |
| 10 | Regulatory reasoning | Which FDA/EMA/ICH guidance governs this question? | Curated Q→citation key | B categorical |
| 11 | Reporting / methods | Draft the methods paragraph from this audit trail | Expert rubric | C rubric |
| 12 | Identifiability & diagnostics | Which parameter is non-identifiable / mis-specified here? | **Your Metformin Paper-2 identifiability cases** | B + C |

Categories 6 and 12 are your **moat** — no one else has validated PBPK models
and a published identifiability analysis to grade against.

---

## 3. Grading design

Three tiers, deterministic wherever possible:

- **Tier A — numeric.** Score = pass if within tolerance of oracle. Use domain
  tolerances, not naive %: **GMFE / twofold bands** for PK parameters, **CI
  overlap** for BE, **relative error < X%** for NCA. Report the tolerance per
  task.
- **Tier B — categorical.** Exact match: model order, pass/fail, correct
  citation key, the seeded error found.
- **Tier C — prose/interpretation.** LLM judge against an expert-written rubric
  (0–4 per criterion). Only the answer's *reasoning quality*, never a number.

Every task also checks **process, not just answer**: did the agent use the
deterministic tool (correct) or free-hand the number in tokens (fail, even if
close)? PharmAgent's audit chain makes this directly checkable — a differentiator
DrugDiscoveryBench can't offer.

### Scoring

```
task_score ∈ [0,1]
category_score = mean(task_scores in category)
overall = mean(category_scores)          # equal weight, not task-count weight
```

Report, per model × reasoning effort:
- overall + per-category bars
- **planning lift** = guided_overall − unguided_overall
- **tool-fidelity rate** = % of numeric answers that came from a tool call
- N=3 runs averaged, 120-min timeout (match DrugDiscoveryBench for comparability)

---

## 4. What you have vs need to build

**Already have (ground truth for free):**
- Self-grading oracles: NCA, BE, dose-prop, compartmental, popPK, TDM — the tool
  computes the answer key, so tasks are *generatable*, not hand-labeled.
- Published answer keys: Metformin PBPK ×2, Bupropion stereoselective PBPK.
- Reference dataset: Theophylline (NCA + popPK).
- Audit chain → tool-fidelity grading, out of the box.

**Need to build:**
- A task spec format (JSON: prompt, dataset ref, oracle/rubric, tolerance, effort levels).
- A runner harness (feed task → agent → capture answer + audit → grade).
- Seeded-error dataset generator for QC tasks (randomize the injected fault).
- Expert step sequences (the "guided" arm) for the planning-lift measurement.
- Rubrics for Tier-C tasks (methods writing, identifiability interpretation).

---

## 5. v0 — IMPLEMENTED (as of 2026-07)

Shipped: **30 tasks across 5 deterministic categories**, all auto-generated from
existing tool oracles, zero new labeling. Reference agents self-test the harness.

- Categories: **NCA, bioequivalence, dose-proportionality, one-compartment
  structural PK, and steady-state exposure prediction** (forward simulation of
  Cmax_ss / AUC_tau, graded within **2-fold** — the human-PK-prediction field
  standard, cf. Käser et al., *Mol. Pharm.* 2026).
- Reference agents: oracle **1.000** (all 5 categories → harness well-formed),
  naïve **0.778**, keyless mock LLM **0.778** (pipeline-faithful). Pinned run:
  `papers/PAPER2_bench_run.txt`; suites: `tests/test_pharmacometricsbench.py` +
  `tests/test_pmbench_llm.py`.
- Guided-vs-unguided planning-lift and the tool-fidelity metric are **designed,
  not implemented** (the oracle-vs-naïve gap is the current proxy).

### Moat category — DESIGNED, needs curated data
A **first-in-human PK prediction** category (categories 6/12 above), modelled on
Käser et al. 2026: predict human AUC∞/Cmax from preclinical/in-vitro inputs,
grade within 2-fold against observed clinical values, compare methods (standard
PBPK / HT-PBPK / ML / **ensemble** — the ensemble result cross-references the
cross-engine paper). Requires a PBPK engine or a curated drug set with published
human PK as answer keys — **author-supplied, never fabricated**. This is the
headline differentiator once the data exists.

> Note: the implemented "exposure" category is forward simulation *from given PK
> parameters* — it exercises the 2-fold-graded prediction mechanism but is NOT
> the FIH-from-preclinical prediction of the moat category. Keep the distinction
> explicit in any writeup.

---

## 6. Open decisions

- **Weighting:** equal-per-category (recommended, avoids easy-category flooding)
  vs task-count.
- **Guided arm scope:** all tasks, or only the ones unguided agents fail?
- **Public vs private split:** hold out a private set to resist contamination if
  this becomes a public leaderboard.
- **Naming/venue:** internal gate first, then a methods paper (target: CPT:PSP /
  J Pharmacokinet Pharmacodyn) — your ARS pipeline can draft it.
