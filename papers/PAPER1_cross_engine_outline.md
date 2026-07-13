# Paper 1 — Cross-engine agentic pharmacometric orchestration

**Status:** outline + evidence map (pre-draft). No prose claims beyond verified artifacts.
**Assumed type/venue:** Methods / tutorial article, *CPT: Pharmacometrics & Systems Pharmacology* (redirectable — JPKPD or a short communication also fit).
**Integrity note:** all quantitative results below are from **simulated data** and are labelled as such. Nothing here is a validation on real clinical PK. Fill every `[CITATION NEEDED]` before submission.

---

## Working titles

1. *Can estimation engines talk to each other? An agent-orchestrated framework for cross-engine population-PK model comparison ranked on engine-agnostic prediction accuracy.*
2. *Beyond the objective function: fair cross-engine model selection in agentic pharmacometrics.*
3. *Orchestrating heterogeneous NLME engines: a reproducible framework that does not compare objective function values.*

## One-sentence thesis

> An AI agent can run the same candidate population-PK model across heterogeneous estimation engines and select a winner **fairly** — provided selection uses engine-agnostic prediction metrics computed by a single shared predictor, because native objective-function values (OFV/AIC/BIC) are not comparable across estimation algorithms.

## Central contribution (what is genuinely new)

1. A **normalized result schema + adapter protocol** that lets one agent fit a candidate across FOCE-I, SAEM, and an external R engine (nlmixr2), degrading gracefully when an engine is absent or cannot fit a model.
2. The **scientific correction**: ranking across engines must use engine-agnostic prediction metrics (prediction RMSE, VPC coverage), *not* native OFV/AIC/BIC — demonstrated by the same fit yielding very different native OFVs across engines.
3. A **uniform scoring construction** (`score_from_population`): individual predictions are recomputed on one side via a shared empirical-Bayes step from each engine's population parameters, so every engine is judged on identical footing.
4. A **software-integrity method**: a multi-round adversarial-review loop over the implementation surfaced and fixed a series of correctness defects before release — offered as a template for building trustworthy agentic pharmacometric tooling.
5. A **consensus (ensemble) engine**: the geometric-mean parameter set of the converged fits, scored on the same footing and competing in the ranking — motivated by ensembles beating single methods for human-PK prediction (Käser et al., *Mol. Pharm.* 2026). Presented as a mechanism, not an accuracy claim; whether it wins is data-dependent and decided by the same prediction ranking.

---

## IMRaD outline (topic sentence + evidence per paragraph)

### 1. Introduction
- **P1 (importance).** Model-informed drug development relies on several NLME engines (NONMEM, Monolix, nlmixr2, and others), each with strengths; practitioners increasingly want to combine them. `[CITATION NEEDED: MIDD / engine landscape]`
- **P2 (what is known / motivation).** Agentic AI now lets a single operator drive multi-tool analyses; recent agent benchmarks show frontier agents plateau on unguided planning in scientific workflows. `[CITATION NEEDED: Scale DrugDiscoveryBench — labs.scale.com/leaderboard/drugdiscoverybench]` McCoy and McCoy argue that this shift reframes the clinical-pharmacology scientist "from executor to orchestrator": as agents perform interpretation, execution, and evaluation, the human contribution moves toward specifying objectives precisely and evaluating recommendations that may be *difficult to verify independently* — and they caution that the field has not yet built the governance for the failure modes that delegation introduces (McCoy & McCoy, *Clin. Pharmacol. Ther.* 2026, doi:10.1002/cpt.70380). Verifiability is therefore not incidental to agentic pharmacometrics; it is the precondition for trusting it.
- **P3 (gap).** No published framework lets an agent run one candidate across engines and select a winner on a *valid* common footing; the naïve approach — comparing native OFV/AIC/BIC across engines — is statistically invalid because each engine's likelihood is on its own algorithm's scale.
- **P4 (aim).** We present a framework that (i) normalizes heterogeneous engine outputs, (ii) ranks on engine-agnostic prediction accuracy, and (iii) embeds the comparison in an audited, human-gated agent workflow; we demonstrate it on simulated data with a native engine and a live external R engine.

### 2. Methods
- **Architecture.** `EngineResult` normalized schema; `EngineAdapter` protocol (`available()`, `fit()`); adapters for native FOCE-I / SAEM, nlmixr2 (R shell-out), and a deterministic mock. Runner fans candidates × engines; absent/failed engines become data, not exceptions.
- **The non-comparability principle.** Native OFV/AIC/BIC are within-engine only; state explicitly and cite the estimation-method background. `[CITATION NEEDED: FOCE/Laplace — e.g. Wang 2007]` `[CITATION NEEDED: SAEM — Kuhn & Lavielle 2005]`
- **Engine-agnostic scoring.** Prediction RMSE (log-scale IPRED vs observed), VPC coverage (prediction-corrected VPC), R², bias; all computed by one shared predictor. `[CITATION NEEDED: pcVPC — Bergstrand et al. 2011]`
- **Uniform empirical-Bayes step (`score_from_population`).** Individual predictions recomputed from each engine's population parameters (θ, Ω, σ) via a common MAP/EBE step, so only the parameters differ between engines, never the prediction machinery. State the design tradeoff (an engine's own structural predictor is not used).
- **nlmixr2 adapter.** R model generated from the registry; NONMEM-style CSV writer; JSON round-trip; allometric weight scaling injected so estimates are weight-centred consistently with the native fitter. `[CITATION NEEDED: nlmixr2 — Fidler et al.]`
- **Reproducibility engineering.** Apple-silicon/Rosetta architecture handling for the R subprocess (documented so runs reproduce).
- **Agent integration.** Registered as an audited modeler tool + REST endpoint + UI + a human-gated `poppk_modeling` workflow (fit → cross-engine comparison → adversarial review gate).
- **Verification method.** Deterministic unit tests (N stated below) + a multi-round adversarial-review loop; report the loop as method, and the defect classes it caught as a validation result.
- **Data.** Simulated one- and two-compartment oral profiles (seeded, reproducible). *State clearly: no real clinical dataset is used.*

### 3. Results
- **R1 — same fit, incomparable OFVs (the headline).** On a simulated 6-subject one-compartment oral dataset (seed 7), native FOCE-I and nlmixr2 recover near-identical parameters yet report native OFVs of **−107.8 vs −210.5** — demonstrating that OFV cannot rank across engines. *(From the captured run in `papers/PAPER1_demo_run.txt`; reproducible via `python -m app.engines.demo` on a host with R + nlmixr2 and the arm64 handling described in Methods. Without those the nlmixr2 row is reported absent/failed and the demo ranks the remaining engines — so pin the captured run when quoting these figures.)*
- **R2 — engine-agnostic ranking.** On the same data, log-scale prediction RMSE was **0.070 (nlmixr2)** vs **0.084 (native FOCE-I)**, VPC coverage 1.00 for both; a deliberately biased mock engine scored 0.167 (VPC 0.375) — the ranking separates good from poor fits on a common footing.
- **R3 — graceful heterogeneity + honest gate.** In the end-to-end `poppk_modeling` workflow on a simulated set, structural selection chose a 2-compartment model; nlmixr2 (1-cmt-only adapter) skipped it and native won; the adversarial review then flagged, at the human gate, that only one engine produced a usable fit ("not cross-confirmed") — the framework surfaces its own limitation rather than hiding it.
- **R4 — software integrity.** The adversarial-review loop found and fixed a series of correctness defects (classes: allometric-scaling asymmetry between engines; a silent empirical-Bayes fallback masking a parameter-key mismatch; an AIC parameter-count error for categorical covariates; a degenerate-bin crash in VPC coverage; a rubber-stamp review over modeling state; an unguarded workflow state/audit race). Report as evidence that agentic tooling needs adversarial verification, not just tests.
- **R5 — consensus engine competes.** A geometric-mean ensemble of the converged fits was scored identically and entered the ranking; in the pinned run it placed mid-pack (pred_rmse 0.085, third of four), i.e. it did not beat the best single engine here. This is reported as evidence the *mechanism* works and is judged on the same footing — not as an accuracy claim, since ensemble benefit is data-dependent (and was the headline finding of Käser et al. 2026 on a real 40-molecule set). *(From `papers/PAPER1_demo_run.txt`.)*
- **R6 — reproducibility.** 28 framework-specific tests collect and pass across `test_engines.py`, `test_engine_tools.py`, and `test_engines_nlmixr2.py` (the last self-skips without R); the cross-engine review gate is additionally covered in `test_adversarial.py`. The demonstration is one command, and its output is pinned in `papers/PAPER1_demo_run.txt`.

### 4. Discussion
- **Interpretation.** The OFV result operationalizes a known caution (do not compare likelihoods across algorithms) into a concrete agent design constraint; the fix (shared-predictor prediction metrics) is simple and general.
- **Comparison to prior work.** Position against agent benchmarks (DrugDiscoveryBench: planning is the bottleneck) and against manual multi-engine workflows. `[CITATION NEEDED]`
- **Answering the call for verifiable agentic pharmacology.** McCoy and McCoy (2026) contend that agentic AI helps only if scientists can evaluate recommendations that are otherwise hard to verify, and that new failure modes accompany the delegation of scientific judgment. This framework is a concrete response on the narrow ground of model comparison: the ranking is computed from predictions rather than incomparable objective functions, every step is recorded in an append-only audit chain, the winner is presented at a human review gate, and the review itself raises calibrated caveats (e.g. flagging a "winner" confirmed by only one engine, or an information-criterion tie). We claim to address one failure mode — untrustworthy or unverifiable model selection — not the fundamental drivers of clinical failure (efficacy, safety, translation) that review identifies; that scope limit is stated plainly rather than blurred.
- **Limitations (honest, prominent).** (i) Simulated data only — no real clinical validation. (ii) Two real engines; Monolix/NONMEM/FeRx are stubs. (iii) nlmixr2 adapter covers 1-compartment models only. (iv) Prediction scoring uses one shared simulator — a deliberate fairness choice that does not exercise each engine's own predictor. (v) Single-dataset demonstration; no multi-dataset benchmarking. (vi) Any upstream *model or covariate selection* (e.g. the stepwise covariate search that can precede the comparison) carries its own post-selection-inference caveat — retained effects' standard errors and p-values are optimistic — which is separate from, and not corrected by, the cross-engine prediction ranking; the framework's adversarial-review gate now flags this rather than presenting a stepwise selection as settled.
- **Implications / future work.** Add engines and structures; validate on real datasets; connect to a benchmark (see companion paper); explore whether shared-predictor scoring biases toward the reference simulator.

### 5. Conclusion
- Restate: fair cross-engine agentic model comparison is achievable by ranking on shared-predictor prediction accuracy rather than native objective functions; the framework is reproducible and self-critical, but validation on real data remains future work.

---

## EVIDENCE MAP (claim → source → status)

| # | Claim in paper | Source artifact | Verified? |
|---|----------------|-----------------|-----------|
| R1 | Same fit → native OFV −107.8 vs −210.5 | `app/engines/demo.py` output (6-subj sim) | ✅ observed this session |
| R2 | pred_rmse 0.070 (nlmixr2) vs 0.084 (native); VPC 1.00 both; mock 0.167 | demo output | ✅ observed |
| R2 | live native vs nlmixr2 recover CL≈4.9, V≈51–53 | live run this session | ✅ observed (8-subj run; use ONE run consistently) |
| R3 | workflow picks 2-cmt; nlmixr2 skips; review flags single-engine | `poppk_modeling` end-to-end run | ✅ observed |
| R4 | adversarial loop found+fixed defect classes | 3 review workflows + fixes + regression tests | ✅ this session |
| R5 | 28 framework tests (engine + tool + nlmixr2 suites); review gate covered in test_adversarial.py | `pytest --collect-only` | ✅ verified (28 collected) |
| R1/R2 | captured demo numbers | `papers/PAPER1_demo_run.txt` | ✅ captured this session (requires R+nlmixr2 + arm64 handling for the nlmixr2 row) |
| Method | score_from_population / EngineResult / adapters | `app/engines/*.py` | ✅ code exists |
| Method | Rosetta arch handling | `app/engines/nlmixr2.py` `_r_cmd` | ✅ code exists |
| Intro | DrugDiscoveryBench motivation | Scale labs leaderboard | ⚠️ **[CITATION NEEDED — verify URL/authors/date]** |
| Method | nlmixr2 / pcVPC / FOCE / SAEM references | literature | ⚠️ **[CITATION NEEDED ×4]** |

## What we CANNOT claim (guardrails)
- Not "validated on clinical data." Not "outperforms Monolix/NONMEM" (those are not run). Not "nlmixr2 is worse/better than native" from one dataset. Not a benchmark result — that is the companion paper. The user's own Pi-agent Monolix/FeRx experiment is **motivation/anecdote**, not a result here (its numbers are not in this framework).

## Open decisions for the user
- Confirm type (methods vs tutorial vs perspective) and venue.
- Confirm authorship, and whether to fold in a real dataset before submission (strongly recommended — would move this from proof-of-concept to validated).
- Supply/verify the four `[CITATION NEEDED]` references.
