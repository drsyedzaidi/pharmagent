# Competitor brief: Delineate (delineate.pro)

Source: delineate.pro homepage, read 2026-10-01. Every claim below is Delineate's own marketing copy, not independently verified. Funding and customer claims are theirs. No pricing, product demo or customer reference was available.

## What they sell

Tagline: "Quantitative evidence for consequential decisions." Positioning: "All available evidence in a single quantitative framework for your most consequential decisions."

Delineate is a services-plus-platform business that builds model-ready evidence databases from public sources, then models on top of them. The headline unit of work is a database, delivered NONMEM-ready, with QC documentation.

| Stream | What they say they do |
|---|---|
| Assemble evidence | Extract and structure FDA clinical pharmacology reviews, EMA EPARs, literature, AdCom documents and postmarket commitments into a queryable database |
| Model and simulate | Model-based meta-analysis (MBMA) across agents, indications and subgroups |
| Optimize dose | Efficacy and toxicity curves, aligned with FDA Project Optimus |
| Benchmark asset | Overlay a sponsor's Phase 1/2 PK and efficacy on class-level exposure-response |
| Submission outputs | Diagnostics, E-R plots, indirect comparison, formatted for FDA briefing, EMA scientific advice and HTA dossiers |
| Digitize plots | AI extraction of data points and error bars from published figures |

## Who they sell to

Four audiences on the page: clinical pharmacology and pharmacometrics; clinical development and trial design; commercial, BD and lifecycle; regulatory and market access. Note the buyer is broader than the modeller. The commercial and BD audience is where a "living model of the competitive class" is pitched.

## Their credibility claims (unverified)

- "4 of the top 10 pharma companies"
- Backed by Y Combinator, MIT Sandbox, NIH Seed
- Named case studies: obesity (GLP-1) MBMA database of 300 trials; antibody-drug conjugate decision engine for 15 approved ADCs; a hybrid-AI PK model corpus for Sanofi (900 publications screened, 4,000+ plots, 23,000+ points); a 16-asset pipeline intelligence database; a surrogate endpoint database with the Critical Path Institute

## Their quality story

"Dual independent extraction with formal arbitration," source-level verification, cross-study consistency checks, outlier flagging with documented rationale, and "full audit trail on every value."

## Where PharmAgent and Delineate overlap

- Both position around regulatory-grade, auditable quantitative work.
- Both end in submission-oriented outputs.
- Both care about NONMEM-compatible deliverables (PharmAgent exports `.ctl` and mrgsolve).
- Both sell a provenance story. This is the strongest shared theme.

## Where they differ

| | Delineate | PharmAgent |
|---|---|---|
| Data | Public literature and regulatory documents, across many drugs | The sponsor's own dataset |
| Unit of work | A curated evidence database, then MBMA | A single analysis, run end to end |
| Delivery | Service engagement with QC | Software a modeller operates |
| Provenance | Per-value audit of extraction | Per-tool-call SHA-256 audit chain and human gates |
| Buyer | Pharmacology plus development, BD, regulatory | Pharmacometricians, clinical pharmacologists, consultancies |

## What PharmAgent does not have (the gap)

1. Extraction from papers, FDA reviews and EMA EPARs into structured data.
2. Reading data points off published plots.
3. Cross-study model-based meta-analysis and class-level E-R benchmarking.
4. A briefing-memo style output (PharmAgent exports a DOCX analysis report and CDISC/NONMEM files).

See `docs/competitive/feature-roadmap.md` for how these could be built on the existing architecture.

## Where PharmAgent can differentiate

- **Own-data analysis.** Delineate starts from literature. PharmAgent starts from the data a client already holds. That is a different, more frequent workflow.
- **Deterministic compute with a visible seal.** "Agents decide, tools execute" plus a hash chain and human gates is a sharper claim than "audit trail on every value," and it is shown in the product (the review panel and audit trail).
- **Engine-agnostic checking.** Ranking NONMEM, nlmixr2, Monolix and the native estimator on prediction error is a verification story Delineate does not mention.
- **Open evaluation.** PharmacometricsBench measures tool-grounded agents against unaided models. A published, reproducible benchmark is an asset a services company is unlikely to match.
- **Agency fit.** PmatricsAI can sell analyses plus training (the courses) rather than databases.

## Risks to keep in view

- They have a head start in extraction at scale and named pharma customers (if the claims hold).
- Buyers who want an MBMA database will compare on breadth of sources, which PharmAgent cannot match without building extraction.
- Their plot-digitization capability is the most concrete technical moat; replicating it well needs a validated pipeline, not a prompt.

## Open questions to answer before positioning against them

- Pricing and engagement size (not public).
- How much of their extraction is automated versus analyst-reviewed (they describe dual independent extraction, which implies significant human work).
- Whether their platform is something a modeller operates directly or primarily a delivered service.
