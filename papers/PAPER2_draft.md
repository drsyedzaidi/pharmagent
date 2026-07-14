# PharmacometricsBench: a reproducible, tool-grounded benchmark for agentic pharmacometric analysis

**Manuscript type.** Methods / resource article.
**Target venue.** *CPT: Pharmacometrics & Systems Pharmacology* (redirectable — *J Pharmacokinet Pharmacodyn*, or a benchmark track).
**Draft status.** Full prose draft v1, generated from the confirmed outline and this session's verified results. Both prior `[CITATION NEEDED]` items (DrugDiscoveryBench figures; Theophylline reference values) are now filled from verified sources. Remaining `[verify]` flags in the reference list mark bibliographic details (exact titles, author initials, page numbers) to confirm before submission. No data, number, or citation is fabricated.

---

## Abstract

Language-model agents are increasingly used to carry out quantitative pharmacology, where an incorrect number is a safety-relevant error rather than a stylistic one. Existing agent benchmarks score free-text answers against fixed keys and so cannot distinguish a number obtained by validated computation from one produced by plausible generation. We introduce PharmacometricsBench, a benchmark whose ground truth for each task is the output a validated deterministic tool produces on the provided data; the score therefore measures whether an agent reproduces correct, reproducible numbers rather than approximating them. Version 0 comprises 30 seed-deterministic tasks across five categories (non-compartmental analysis, average bioequivalence, dose proportionality, one-compartment structural PK, and steady-state exposure), each graded per target against a tolerance rule. A tool-calling reference agent scored 1.00 overall, confirming the tasks are well posed; a tool-free heuristic agent scored 0.78, establishing a discriminating floor. In an initial evaluation of two frontier-family models reasoning without tools (three runs each on the identical task set), Claude Opus 4.8 scored 0.93 ± 0.03 and Claude Haiku 4.5 scored 0.60 ± 0.04; both were near-exact on formula-driven categories but fell furthest on the task requiring iterative numerical estimation (one-compartment fitting: 0.74 ± 0.11 and 0.20 ± 0.11). When the same two models were instead given the validated tools by function-calling, both scored 1.00 in every category over three runs, showing that tool grounding, not model scale, is what closes the gap: a smaller model with tools (1.00) surpassed a larger model reasoning without them (0.93). The harness reports per-agent error and parse-gap counts so a truncated or malformed response is never scored as a wrong answer; this instrumentation caught, and we corrected, a measurement artifact that had spuriously halved one model's score. We release the benchmark, its reference agents, and a loader that assembles cited first-in-human PK answer keys from an open database, and we invite broader frontier-model evaluation.

---

## 1. Introduction

Language-model agents are moving from drafting text to performing quantitative analyses, including tasks in clinical pharmacology and pharmacometrics. In this setting a wrong output is not a matter of style: a mis-estimated clearance or a mis-computed exposure can propagate into a dosing recommendation. McCoy and McCoy argue that as agents take on interpretation, execution, and evaluation, the scientist's role shifts "from executor to orchestrator," and the central difficulty becomes evaluating recommendations that are hard to verify independently — a capability the field has not yet built (McCoy & McCoy, *Clin Pharmacol Ther* 2026, doi:10.1002/cpt.70380). Making "did the agent compute the right number" checkable, rather than a matter of trust, is therefore a prerequisite for using agents in regulated pharmacometrics.

General agent benchmarks exist, and the closest analogue in drug development, DrugDiscoveryBench, evaluates 82 expert-authored early-discovery tasks; the strongest agents solve only about half of them, and giving an agent an expert step-by-step plan lifts tasks that had scored zero unaided to near-perfect, which identifies unguided planning rather than raw model capability as the bottleneck (Scale Research Team and Siegel, 2026). These benchmarks share a scoring model: an agent's free-text answer is compared with a fixed key. That model rewards a correct-looking answer regardless of how it was produced, and it does not capture whether an agent used validated computation.

No published benchmark evaluates whether an agent can *execute* a pharmacometric analysis — non-compartmental analysis (NCA), bioequivalence (BE), dose proportionality, structural PK estimation — and return numbers that are both correct and reproducible. The gap matters because the pharmacometric question is not "can the model describe the method" but "can the agent run it and get the number right."

We address that gap with PharmacometricsBench. Each task's ground truth is the output of a validated deterministic compute tool run on the task's own data, so the tasks are generated rather than hand-labelled, and the score measures reproduction of a validated computation. We describe the benchmark and its grading, validate the harness with reference agents, report an initial evaluation of two real models that quantifies how far unaided reasoning falls short of a validated tool, and release a path to cited real-drug answer keys for a first-in-human prediction category.

## 2. Methods

### 2.1 Task model

A task is a prompt, a dataset the agent may use, and one or more typed targets. Each target carries an expected value and a tolerance rule: relative, absolute, a twofold (GMFE) band, or exact match for categorical and boolean outcomes. Every task is generated from a fixed random seed, so the task set is byte-reproducible and no answer key is stored by hand.

### 2.2 Ground truth as validated tool output

For each deterministic category the answer key is produced by running the corresponding analysis tool on the seeded data. The tools are those of a pharmacometric compute library (the PharmAgent platform; companion manuscript), and they carry their own reference-validation suite that checks them against consensus reference values for the standard Theophylline dataset (`tests/reference/test_theophylline.py`, 8 tests). That dataset (`datasets::Theoph` in R; originating in the NONMEM Users Guide of Boeckmann, Sheiner and Beal, from a study by Upton) has well-established one-compartment first-order parameters, as used by Pinheiro and Bates (2000). Ground truth is therefore agreement with an independently validated tool, not a per-task re-derivation from first principles; we return to this design choice in the limitations.

### 2.3 Categories (v0)

Version 0 contains five categories: NCA (Cmax, AUC0–inf, terminal half-life); average bioequivalence (geometric mean ratio, 90% confidence bounds, and the BE verdict); dose proportionality by the power model; one-compartment structural PK parameter recovery (CL, V, ka) by fitting; and steady-state exposure prediction by forward simulation (Cmax,ss and AUC,tau for a multiple-dose regimen). The exposure category is graded within twofold, the accuracy criterion used in human-PK-prediction method comparisons (Käser et al., *Mol Pharm* 2026, doi:10.1021/acs.molpharmaceut.6c00429). Categories that require a fitted population model or curated literature keys — population PK/TDM, physiologically based first-in-human prediction, and identifiability — are deferred; Section 2.7 describes the infrastructure now in place for the first of these.

### 2.4 Grading and scoring

Each target is scored pass or fail under its tolerance rule. A task's score is the fraction of its targets that pass. The overall score is the mean across categories rather than across tasks, so a large, easy category cannot dominate a small, hard one. Scores lie in [0, 1].

### 2.5 Reference agents

Three reference agents validate the harness before any paid model is evaluated. The oracle agent calls the compute tools and, by construction, reproduces ground truth; it verifies that tasks are well formed. A naïve agent applies standard textbook formulas without the tools (linear-trapezoid AUC, arithmetic-mean ratios, two-point terminal-slope estimation); it establishes the tool-free floor and confirms the benchmark discriminates. A keyless mock model routes a computed answer through the full prompt-to-parse pipeline, so the answer-extraction machinery is exercised and tested without an API key. Two further reference agents are pinned to specific models (Claude Opus 4.8 and Claude Haiku 4.5); these require an API key and never fall back to the keyless mock, so a keyless run cannot be mistaken for a real-model score. A final pair of agents gives those same models the validated compute tools by function-calling: each of five tools wraps one category's validated function, and the model must select the correct tool and supply the dataset to it. These tool-using agents test the benchmark's thesis directly — whether a real model, once it can call the tools, reaches the tool-grounded result.

### 2.6 Real-model evaluation and reliable answer extraction

A real model receives the task prompt and dataset and returns free text; the harness extracts a JSON answer object and grades it. Because a model may show working before committing an answer, the parser takes the last fenced JSON block, not the first, and the prompt requests that the answer block be last. The runner records, per agent, the number of responses that errored and the number that produced no parseable answer ("parse-gaps"), and reports these alongside the score. This instrumentation is not cosmetic: it distinguishes a wrong answer from an answer the harness failed to read, and it caught a real artifact during development (Section 3.3).

### 2.7 First-in-human PK category: cited real-drug answer keys

A deferred category predicts a drug's human PK from its properties, graded within twofold against observed clinical values — the design of the Käser method comparison. Its obstacle is answer keys: real, cited human PK values that must not be fabricated. We built a loader that assembles them from PK-DB, an open pharmacokinetics database (Grzegorzewski et al., *Nucleic Acids Res* 2020, doi:10.1093/nar/gkaa990). The loader harvests oral dosing and study metadata over the anonymous API and the drug-level PK parameters (the answer keys) once a free access token is supplied; it unit-harmonises and pools values across studies, pooling only dose-independent parameters (clearance, half-life, volume) for which cross-dose pooling is well defined, and it records the source study for every value. Without a token the harvest yields no answer keys and the builder writes zero tasks rather than any fabricated value. The loader and its task builder were hardened over two rounds of adversarial code review that fixed defects including a fail-open substance filter (which could have admitted a covariate as a drug's clearance) and an unanchored drug-name match (which could have attributed a congener's data to the target drug).

### 2.8 Reproducibility and software

The task set is seed-deterministic and the evaluation is a single command. The benchmark and its answer-parsing pipeline carry 20 regression tests (12 in `test_pharmacometricsbench.py`, 8 in `test_pmbench_llm.py`); the PK-DB loader carries 25 (`test_pkdb_loader.py`). The oracle-scores-1.00 assertion is itself a regression gate: a malformed task or a changed tool would break it. Every reported run is pinned to a captured log in the manuscript's `papers/` directory.

## 3. Results

### 3.1 The harness is well formed

The tool-calling oracle scored 1.00 overall and in every category, with no errored or parse-gapped tasks. Because ground truth is the tool's own output, this is a validity check rather than a performance claim: it confirms that all 30 tasks are correctly specified and gradeable.

### 3.2 The benchmark discriminates

The tool-free naïve agent scored 0.78 overall (NCA 0.89, BE 0.67, dose proportionality 0.92, one-compartment fitting 0.50, exposure 0.92), again with no errors or parse-gaps. The gap from the oracle is large and, because the agent applies correct textbook formulas, it is concentrated where a closed-form approximation departs from the validated computation — most of all in one-compartment fitting (0.50), which requires iterative estimation rather than a formula. The keyless mock model, routed through the full pipeline, reproduced this floor exactly (0.78), confirming that the answer-extraction and grading machinery is faithful independent of any paid model.

### 3.3 Real models: scale narrows the gap but does not close it

We evaluated two models reasoning without tools, three runs each on the identical 30-task set (Table 1; Figure 1). Claude Opus 4.8 scored **0.93 ± 0.03** overall and Claude Haiku 4.5 scored **0.60 ± 0.04**; both runs produced no parse-gaps. Opus was exact on bioequivalence and exposure (1.00 ± 0.00) and near-exact on NCA (0.98 ± 0.03), but fell to 0.92 ± 0.07 on dose proportionality and **0.74 ± 0.11** on one-compartment fitting — the largest and most variable gap, and the one category that requires iterative numerical optimisation rather than a formula. Haiku showed the same shape at a lower level, scoring 0.20 ± 0.11 on fitting. The pattern is consistent across both models: a capable model reasoning to a number matches a validated tool on formula-driven analyses but not where the answer must be obtained by iterative fitting. Model scale (Opus vs Haiku) narrows the shortfall substantially; it does not remove it.

Given the same tools by function-calling, both models scored 1.00 in every category over three runs, with zero variance and no parse-gaps (Table 1, "+ tools" rows). The gap the tool-free models could not close — most visibly one-compartment fitting, where tool-free Opus reached 0.74 and tool-free Haiku 0.20 — disappeared once the models could call the validated fitter. Grounding, not scale, is what closes the gap: the smaller model with tools (1.00) surpassed the larger model reasoning without them (0.93). This ceiling is bounded honestly by its construction. Because a tool returns the validated computation, a score of 1.00 means the model selected the correct tool and marshalled the dataset into it correctly on all 30 tasks — perfect tool orchestration — rather than an independent re-derivation. Orchestration is nonetheless a real capability: a model that called the wrong tool or supplied the wrong data would have scored below 1.00, and neither did across six runs.

### 3.4 Instrumentation prevents a measurement artifact

An initial Opus run scored 0.32 with 19 of 30 responses recorded as parse-gaps rather than wrong answers. Because the runner surfaces parse-gaps separately, the low score was not accepted at face value. The cause was a response-length limit that truncated the model's verbose working before it emitted the closing JSON block, so no answer could be parsed. Raising the limit, taking the last fenced block, and requesting the answer block last eliminated the artifact (0 parse-gaps), and the corrected score is the 0.93 above. We report this because it illustrates the benchmark's design principle in operation: a harness that conflates "unreadable" with "wrong" would have published a number roughly threefold too low, and single-model leaderboards without such instrumentation are exposed to exactly this failure.

### 3.5 Reproducibility

The task set regenerates deterministically from its seeds; the 45 regression tests (20 benchmark, 25 loader) pass, with 2–4 skipped only when an optional external engine is absent, by design. The reference-agent runs report 0 errored and 0 parse-gapped tasks, so the reported floors are genuinely wrong answers rather than harness failures. All runs quoted here are pinned to captured logs.

## 4. Discussion

Tool-grounded ground truth reframes the evaluation question from "did the model output the right string" to "did the agent produce a correct, reproducible number via validated computation." That is the question that matters for regulated pharmacometrics, and it is the question a fixed-key text benchmark cannot answer. The initial real-model results give the reframing empirical content: a frontier model unaided is already good at the formula-driven analyses (Opus scored 1.00 on bioequivalence and exposure), so the benchmark's value is not in showing that models fail broadly — they do not — but in locating precisely where unaided reasoning is unreliable, namely the tasks that require iterative numerical estimation, and in showing that giving the model those same tools removes the unreliability entirely: both models, tool-free, left a gap that both, tool-using, closed to 1.00.

This finding answers, on a narrow and testable ground, the call for verifiable agentic pharmacology. McCoy and McCoy (2026) hold that agentic AI helps only where its recommendations can be evaluated. A benchmark whose truth is a validated computation operationalises that evaluation for the execution layer: the score is a direct, reproducible measure of whether the agent reproduced the right number. We are explicit that this addresses verifiability of *execution* and not the deeper drivers of clinical success or failure.

Against prior work, the benchmark's scope is deliberately downstream of DrugDiscoveryBench, which stops at target and molecule; PharmacometricsBench begins at molecule and data (Scale Research Team and Siegel, 2026). Its tool-grounded truth also differs from general LLM-evaluation practice, where a static key cannot certify that computation, rather than recall, produced the answer.

Several limitations bound these claims. First, the real-model evaluation is small: two models, three runs each, on 30 tasks; the per-category standard deviations (up to 0.14) show meaningful run-to-run variance, and a publication-grade table would average more runs and more models. Second, the exposure category grades within twofold, which is the field standard but is deliberately generous, so it discriminates correct forward simulation from crude guesses rather than testing fine-grained accuracy. Third, ground truth is agreement with the platform's own (independently validated) tools, not an external re-derivation; the benchmark measures reproduction of those tools. Fourth, the first-in-human category's infrastructure is built but not yet populated: it produces real cited tasks only when a database access token is supplied, so no real-drug prediction result is reported here. Fifth, the tool-using rows measure grounding at the level of the whole task (tool-free versus tool-using score), and a score of 1.00 reflects correct tool orchestration and inheritance of the tool's exactness rather than independent derivation; a finer per-answer tool-fidelity instrument — the share of individual numeric answers traceable to a tool call rather than to generated text, which the platform's audit trail could support — is not yet implemented.

Future work is concrete. Broader and repeated model evaluation would turn the initial results into a stable leaderboard. Populating the first-in-human category with cited PK-DB answer keys would add the scientifically hardest, and most defensible, category and connect the benchmark to published human-PK method comparisons (Käser et al. 2026). Adding population-PK and identifiability categories, drawing on validated published models, would extend the benchmark into the analyses that most distinguish expert pharmacometrics.

## 5. Conclusion

A benchmark whose ground truth is the output of validated deterministic tools makes "can an agent run pharmacometrics" measurable and reproducible. Reference agents confirm the harness is well posed and discriminating; an initial evaluation shows a frontier model reasoning without tools reaches 0.93, with the shortfall concentrated on iterative fitting, while giving either the frontier or the smaller model the validated tools raises the score to 1.00 — grounding, not scale, closes the gap. The benchmark, its reference and tool-using agents, and a loader for cited real-drug answer keys are released for broader evaluation.

---

## Figure 1 (legend)

**Figure 1. Model scale narrows the gap to a validated tool; only tool grounding closes it.** Per-category scores on PharmacometricsBench v0 (30 tasks, five categories). A tool-grounded agent scores 1.00 in every category (exact by construction, since ground truth is the tool's output). Claude Opus 4.8 (mean of three runs; error bars = SD) and Claude Haiku 4.5 reason without tools; both are near-exact on the formula-driven categories (NCA, bioequivalence, exposure) but fall on the categories requiring iterative estimation, most of all one-compartment fitting (Opus 0.74 ± 0.11). Score = fraction of graded targets within tolerance; 0 parse-gaps across all runs. Source figure: `papers/figure_tool_fidelity.svg`.

## Table 1

**Table 1. Per-category and overall PharmacometricsBench v0 scores.** The oracle and heuristic are deterministic; the four model rows were run three times each on the identical 30-task set (mean ± SD). "+ tools" rows give the model the validated compute tools by function-calling; the tool-free rows do not. Score = fraction of graded targets within tolerance; 0 parse-gaps in all runs. Values pinned in `papers/PMBENCH_llm_run.txt` (tool-free) and `papers/PMBENCH_tools_run.txt` (tool-using).

| Agent | NCA | Bioequivalence | Dose-prop. | Compartmental | Exposure | **Overall** |
|---|---|---|---|---|---|---|
| Tool-grounded (oracle) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **1.00** |
| Heuristic (no tools) | 0.89 | 0.67 | 0.92 | 0.50 | 0.92 | **0.78** |
| Claude Opus 4.8, no tools (n = 3) | 0.98 ± 0.03 | 1.00 ± 0.00 | 0.92 ± 0.07 | 0.74 ± 0.11 | 1.00 ± 0.00 | **0.93 ± 0.03** |
| Claude Haiku 4.5, no tools (n = 3) | 0.59 ± 0.05 | 0.88 ± 0.09 | 0.42 ± 0.14 | 0.20 ± 0.11 | 0.89 ± 0.10 | **0.60 ± 0.04** |
| **Claude Opus 4.8 + tools (n = 3)** | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | **1.00 ± 0.00** |
| **Claude Haiku 4.5 + tools (n = 3)** | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | **1.00 ± 0.00** |

---

## References

1. McCoy and McCoy. *Clin Pharmacol Ther.* 2026. doi:10.1002/cpt.70380. `[verify exact title and author initials]`
2. Käser et al. *Mol Pharm.* 2026. doi:10.1021/acs.molpharmaceut.6c00429. `[verify exact title and full author list]`
3. Grzegorzewski J, et al. PK-DB: pharmacokinetics database for individualized and stratified computational modeling. *Nucleic Acids Res.* doi:10.1093/nar/gkaa990. `[verify year (2020 advance / 2021 issue), volume, pages]`
4. Scale Research Team, Siegel M. DrugDiscoveryBench. Scale AI and Phylo; 30 June 2026. https://scale.com/blog/drugdiscoverybench and https://labs.scale.com/leaderboard/drugdiscoverybench (accessed 13 July 2026).
5. Pinheiro JC, Bates DM. *Mixed-Effects Models in S and S-PLUS.* New York: Springer; 2000. (Consensus one-compartment parameters for the Theophylline dataset.)
6. Boeckmann AJ, Sheiner LB, Beal SL. *NONMEM Users Guide.* (Origin of the Theophylline dataset; data from a study by Upton.) `[verify edition/year]`

*Reference figures for DrugDiscoveryBench (ref. 4) verified against the source on 13 July 2026: 82 expert-authored tasks; strongest agents ≈50% pass; expert step-by-step plans lift zero-scoring tasks to near-perfect. The "76/82" figure from an earlier note was **not** found on the source and is not used.*

## Open items before submission (to-do, not prose)

- Confirm venue and author list; decide whether to add ≥1 non-Claude model and more runs before the Table hardens (strengthens generality — see reviewer note).
- Resolve the `[verify]` reference details above (titles, author initials, PK-DB pages, NONMEM guide edition).
- Decide whether to populate the first-in-human category (needs a free PK-DB token) before submission, or present it as released infrastructure.
