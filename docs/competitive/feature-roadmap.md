# Feature roadmap: closing the gap with Delineate

Companion to `delineate.md`. Based on a read of `backend/app`, `backend/pharmacometricsbench` and the README on 2026-10-01. File and function names below were spot-checked against the repo. Nothing here is built.

## Ground rule

PharmAgent's thesis is that agents decide and tools execute. Every new capability must keep it: a model may **propose** a structured value, a deterministic tool **computes** with it, and a person **verifies** before it is trusted. No number reaches a report because a language model wrote it.

## What exists today

| Capability | Status | Where |
|---|---|---|
| Submission-style outputs | Strong | `tools/report_tools.py` and `POST /report` (DOCX); `tools/regulatory_tools.py` and `/report/272` (ICH M4E 2.7.2); `core/cdisc.py` and `/cdisc` (ADPP, ADPC, define.xml); NONMEM and mrgsolve exports; VPC, GOF and forest in `compute/` |
| Simulation and dose selection | Partial | `compute/dose_sweep.py`; `compute/clinsim.py` (`clinical_trial_simulation`, `sample_theta_draws`, `reference_population`, single-threshold `_recommend`); Emax and sigmoid PD models in `compute/pk_models.py` |
| Population modelling | Strong | `compute/nlme.py` (FOCE-I, SAEM), SCM, bootstrap, SIR, cross-engine ranking in `engines/` |
| Real-data harvesting | Bench only | `pharmacometricsbench/pkdb/loader.py` (`PKDBClient`, licence-aware, `open_only`); not in the app |
| Evidence database | None | `core/store.py` holds sessions only |
| Document extraction (PDF, EPAR, FDA review) | None | uploads are CSV-only (`ALLOWED_UPLOAD_SUFFIXES = {".csv"}` in `main.py`); no PDF or OCR dependencies |
| Plot digitisation | None | no image upload or axis-calibration code |
| Meta-analysis (MBMA, indirect comparison) | None | no aggregate-data estimator |
| Binary and time-to-event exposure-response | None | only Emax-type fits |

## Roadmap

Ranked by value to a consultancy and by feasibility. Size: S under a week, M one to three weeks, L over three weeks (rough).

### Phase A: reuse what exists

**A1. Briefing memo and indirect comparison (S to M).**
- New `compute/indirect.py` (Bucher adjusted comparison first, a small network option later) and a `build_briefing_memo` tool extending `report_tools.py`.
- The memo is a template filled from PharmState values only. The model chooses sections; it never writes a number. Each number carries its audit-entry id.
- Risk: narrative drift. Mitigation: a Reviewer check that every digit in the memo traces to state.
- Determinism: closed-form maths, unit-tested against published worked examples.

**A2. Binary and time-to-event exposure-response, plus Project Optimus dose selection (M to L, very high value).**
- New `compute/er_models.py` (logistic E-R for response and adverse events, Kaplan-Meier and Cox with bootstrap intervals reusing `compute/bootstrap.py`) and `compute/optimus.py` (efficacy and toxicity curves with a user-declared clinical-utility index).
- New tools `fit_exposure_response` and `select_optimal_dose`, a new agent, and a new PharmState slot. Large bootstraps follow the existing `expensive` plus `proposable` pattern.
- Risk: statistical validity. Mitigation: golden-file tests against R (`glm`, `survival`) and new PharmacometricsBench tasks with tool-grounded ground truth.
- Determinism: seeded RNG recorded in the audit chain; utility weights are declared by the user and gated.

### Phase B: the evidence layer (the data moat)

**B1. Evidence store (M).** Tables for `evidence_source` (document hash, URL, licence, retrieval date) and `evidence_record` (drug, population, dose, metric, value, unit, n, source locator, extractor, verification status). Promote the PK-DB loader into the app so its licence gating carries over. Records are immutable and hashed; each ingestion batch is one audit entry. Everything below depends on this.

**B2. Document extraction from FDA reviews, EPARs and open-access papers (L, high value, highest risk).**
- New `ingest/` package: fetch, parse tables, and a model-assisted `propose_evidence_records` step.
- The model only proposes records, each tied to a verbatim snippet and a page or table locator. A deterministic check confirms the number literally occurs in the cited span and normalises units. A human gate verifies records before they enter the store. Runs as a job.
- Risks: misread numbers (mitigated by the literal-match check and a second-pass discrepancy flag), licensing (restrict to FDA and EMA public documents and open-access papers; store hashes and short excerpts), and fragile table extraction from PDFs.
- Start with FDA clinical pharmacology reviews only.

**B3. Class-level benchmarking (M, depends on B1 and A2).** A `benchmark_sponsor_data` tool places the sponsor's NCA exposures against class evidence records: percentile, overlap, and the sponsor's point on the class curve. Output lists the evidence-record ids it used.

### Phase C: heavier modelling

**C1. Model-based meta-analysis (L).** Random-effects meta-regression of arm-level dose-response with study weights, seeded, nested bootstrap intervals, marked expensive and proposable. Refuse or warn below minimum numbers of trials or arms. Optional R adapter in `engines/` modelled on the nlmixr2 one. Input fingerprint is the hash set of verified evidence records.

**C2. Plot digitisation (M, medium value, quick demo).** Image upload plus a `digitize_plot` tool. The user marks two reference points per axis (linear or log) on a frontend canvas and the backend does the exact pixel-to-data transform in numpy. A model may propose points but the user accepts or edits them. Report an error estimate. Store derived points with the citation, never the copyrighted figure. Calibration points and image hash go in the audit chain.

## Suggested order

1. A1 briefing memo and indirect comparison
2. A2 exposure-response and Optimus dose selection
3. B1 evidence store
4. B3 class benchmarking
5. B2 document extraction (start narrow)
6. C2 plot digitiser
7. C1 MBMA

A1 and A2 deliver visible value with no new data sources, and they strengthen the "submission outputs" and "dose optimisation" claims where PharmAgent already competes. B1 onward is the larger bet and is only worth starting if the agency decides to sell evidence work, not just analyses.

## Guardrails for every item

- Every extracted or digitised number is a proposal with a source locator and a verified or unverified status. Only verified records feed compute tools.
- Each new capability gets PharmacometricsBench tasks so the tool-grounded advantage stays measurable.
- Per `CLAUDE.md`: audit, review-gate and provenance code changes need human review; no real engine fits from automated loops.

## Decision needed from the owner

- Is the goal to match Delineate's evidence-database service, or to stay with own-data analysis and add only A1 and A2? The second is far smaller and plays to the product's strengths.
