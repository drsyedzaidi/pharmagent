# Context-engineering ideas for PharmAgent: design note

**Status:** proposal for discussion. Nothing here is built or decided. No application code is changed by this note.
**Date:** 2026-10-20. **Base:** `main` at `0bf3c56`.

## Source and licence
Ideas are adapted from the method in Pritchard-Bell, Lin, Holmes and Doshi, "Context Engineering for AI-Assisted
Pharmacometrics: A Practical Tutorial", *CPT: Pharmacometrics & Systems Pharmacology* 15(8), e70317 (2026),
doi:10.1002/psp4.70317 (CC BY-NC-ND 4.0, so the method is restated here in our own words and nothing is copied).
The paper's evidence is one synthetic scenario, 3 runs per platform, with harness and model confounded. It supports the
design pattern, not any performance claim for PharmAgent.

## Why these four
Four defect classes this cycle (PR #3) had one root cause: **a convention was applied implicitly and failed silently.**
A blank DV counted as a measurement; a dose row became a concentration; an excluded tau sample shrank AUC_tau while the
result still said "AUC_tau"; the rule text claimed lambda_z exclusion that the code did not perform. Green tests did not
catch these. Independent reviewers did. The four ideas make conventions explicit, checkable and visible.

## What already exists (verified in the code at `0bf3c56`)
- `StudyInfo` (`app/core/pharmstate.py`): drug, route, dose range, matrix, `assay_lloq` as a **free string**. It is read only
  by the regulatory report and the memo builder. The NCA tools do not use it.
- `detect_roles` (`app/core/schema_extractor.py`): infers ID/TIME/DV/AMT/II/ADDL/CENS roles from column names. This is the
  implicit inference behind the PR #3 defects.
- Conventions live in code, docstrings and tool rule text, in several places (for example the 3.84 cut-off in
  `app/compute/profile.py` and `app/tools/profile_tools.py`; rule M1 in `app/compute/dosing.py` and `app/tools/nca_tools.py`).
- `agent_eval/runner.py`: `--trials N` repeats a scenario but only checks whether the **score** is stable
  (`stable_fraction`), not which decisions differed. The live model is opt-in (`--allow-live`).
- `stats_advice` (statistician agent): a parametric vs non-parametric recommendation per exposure metric.

## Idea 1: declared study context that fails closed
**Problem.** Facts such as LLOQ, which column carries the BLQ flag, units, infusion duration and the analysis set are
inferred or absent. Inference fails silently.

**Proposal.** Give `StudyInfo` (or a sibling `StudyContext`) typed, optional fields: numeric LLOQ with unit, BLQ flag column
and coding, dose/concentration/time units, route, infusion duration, analysis-set definition, expected covariates.
A tool that needs a fact it does not have stops with a message naming the missing fact. It does not guess.

**Rollout (to avoid breaking existing sessions):** (a) record and display only; (b) warn when a needed fact is missing;
(c) fail closed behind an explicit strict mode; (d) default to strict only after a deprecation period.

**Constraints.**
- Context holds study-level facts only, never subject rows. It must stay inside the `SchemaExtractor` privacy boundary.
- Changes to it should be recorded through the **existing** audit API. This note does not touch audit code.
- The paper auto-extracts context from the protocol with an LLM. For a regulated app that is risky; if used at all, the
  extracted values must be shown for explicit human confirmation before they take effect.

**Acceptance tests.** Missing LLOQ with unflagged zeros and strict mode on stops with a named reason. A declared flag
column overrides role inference. Legacy sessions without context behave exactly as today.

## Idea 2: conventions as reviewable data, with drift tests
**Problem.** A reviewer must read code to learn which conventions apply. Rule text and behaviour have already drifted once
(the lambda_z claim).

**Proposal.** A read-only catalog of conventions, one entry each: id, plain-language rule, the tools it applies to, its
source (guidance, paper, or an owner decision), and the id of the test that pins it. Examples: M1 BLQ handling,
linear-up/log-down, delta-OFV 3.84, "estimate F when IV data are present", the C(tau) = C(0) steady-state convention.
Two checks: every entry must be referenced by at least one test, and the rule text shown in a tool result is generated from
the catalog so it cannot claim something the catalog does not list.

**Non-goal.** Runtime-editable rules (by the LLM or by users). That would create a new guarded configuration surface.

**Acceptance tests.** A rule with no test fails the catalog check. Changing a catalog entry changes the displayed rule text.
An export gives reviewers one document listing every convention in force.

## Idea 3: divergence reporting in the agent eval
**Problem.** `--trials` tells us whether scores are stable, not **where** the model's decisions differ. Divergence between
runs points at tool descriptions that under-specify a choice.

**Proposal.** Extend the runner to record per-turn decisions (routed agent, tool sequence, key arguments) across trials and
report agreement per decision point, flagging the ones that split. Keyless policies are deterministic, so this only means
something with the live model, behind the existing `--allow-live` flag. It must never run in CI.

**Cost.** LLM calls only; no model fits. The amount depends on the model and scenario count, which I have not measured.

**Acceptance tests.** With a stub policy that varies one argument across trials, the report names that decision point and its
agreement fraction. With identical trials it reports full agreement.

## Idea 4: branch-point reporting
**Problem.** Where uncertainty is real (parametric vs non-parametric bootstrap, whether a covariate is retained, tied
lambda_z windows), one answer is presented as if it were the only one.

**Proposal.** A common `branch_points` entry in results: the decision, what was chosen, the alternatives, the evidence, and
whether the alternative changes the conclusion. Start with two places: the statistician's parametric/non-parametric choice
and bootstrap method. Report an entry only when a cheap sensitivity check shows the conclusion would change, to avoid noise.
Entries surface in the review gate, the report and the memo.

**Constraint.** Memo numbers are traced to audit entries. Branch-point numbers must be traced the same way.

**Acceptance tests.** An entry appears only when the alternative changes the conclusion. The memo's untraced check passes.

## Suggested order and size
| Order | Idea | Size | Why |
|---|---|---|---|
| 1 | 2: conventions catalog and drift tests | small | Safe, read-only, and directly prevents text/behaviour drift |
| 2 | 1: study context, record and warn first | medium | Targets the most common defect class; needs your design decisions below |
| 3 | 3: divergence report | small | Isolated to the eval harness; live use only |
| 4 | 4: branch points | medium | Touches results, review gate and memo, so it carries the most risk |

## Decisions needed from the owner
1. Typed fields on `StudyInfo`, or a new `StudyContext`? Where is it entered (form, chat, upload)?
2. May the app ever auto-extract context from a protocol, and with what confirmation step?
3. Strict mode: opt-in per session, or a global default after a deprecation period?
4. Does the conventions catalog live in code or as a data file, and who may change an entry?
5. Which branch points matter most to the agency's reports?

## Out of scope
Anything in guarded code (auth/owner, audit trail, e-signature, review gate, provenance) without explicit human review.
Live model runs in CI. Any claim that the paper's results transfer to PharmAgent.
