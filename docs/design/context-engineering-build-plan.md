# Build plan: context-engineering ideas

**Status:** plan for review. Slice 1 exists as a local branch (`feat/conventions-catalog`, commit `1d1a6fd`), not pushed.
Decisions behind this plan: `context-engineering-ideas.md`, section "Owner decisions (2026-10-21)".

## Ground rules for every slice
- A new branch per slice, test-first, one concern per PR, no auto-merge.
- Guarded code (auth/owner, audit trail, e-signature, review gate, run provenance) is not edited without explicit human
  review. Where a slice needs a guarded change, it is split out as its own reviewed PR.
- A convention change needs: maintainer PR, pharmacometrics-owner approval, an authoritative source or recorded owner
  decision, a behavioural test, regenerated rule text, and the convention version in run provenance.
- No live R/PKNCA run unless the owner says "run".

## Slice 1: read-only conventions catalog (built, awaiting review)
Delivers: typed schema; 12 initial conventions; catalog validation; convention-to-test coverage check; rule text
generated from the data; markdown and JSON report; `python -m app.conventions --check`. No module outside
`app/conventions` imports it (pinned by a test), so runtime behaviour is unchanged.

Facts to know before reviewing:
- All 12 entries are `pending_owner_review`. None is marked approved, and none cites guidance that has not been fetched
  and checked; entries describing existing code are marked `implementation_convention`.
- Known gap recorded: the lambda_z tie rule has no dedicated test.
- Branch points are annotated on three entries (BLQ M1, steady-state closure, lambda_z window).

Owner action: review each entry's wording and approve or amend it (approval needs a name and a date in the file).

## Slice 2: use the catalog at runtime, still no behaviour change
- Generate the BLQ and steady-state rule text shown in tool results from the catalog, so text and data cannot drift.
- Validate the catalog at application startup (fail fast on a malformed file).
- Add the catalog version to tool results. **Run provenance is guarded code:** recording the version in provenance is a
  separate PR with explicit owner review.
- Test: result text equals the catalog's generated text.

## Slice 3: remaining initial branch points
Add catalog entries, each with tests, for: BE method and pass/fail result (the CI/limits rule is in slice 1; the design
choices around it are not), covariate inclusion and functional form, parametric vs non-parametric uncertainty, and
model-structure selection. Each needs an authoritative source or a recorded owner decision before it can be approved.

## Slice 4: `StudyContext` (new type), record-only
- Typed, versioned, immutable once confirmed, bound to the dataset content hash, analysis-governing facts only.
- Proposals come from upload and schema detection; a dedicated form confirms them; chat can propose but not activate.
- Record and display only; no tool reads it yet. Context changes use the existing audit API (no audit-code edits).
- Protocol extraction, if built, produces untrusted proposals with document hash, locator, evidence and confidence, and
  each consequential field needs explicit human confirmation.

## Slice 5: warn, then strict
- Warn when a tool needs a fact `StudyContext` lacks (LLOQ, BLQ flag column, units, route, dosing).
- Strict mode as a per-session opt-in; each result states strict or legacy mode.
- Default to strict for new sessions only after a documented deprecation period, with an explicit recorded opt-out for
  legacy. Production and regulatory exports eventually require strict.

## Slice 6: divergence report in the agent eval
Decision-level agreement across trials (routed agent, tool sequence, key arguments), live model only behind
`--allow-live`, never in CI. Cost is unmeasured.

## Slice 7: branch-point reporting
Surface a branch point in the review gate, report and memo only when a plausible alternative changes the conclusion or
materially changes an estimate. Memo numbers must stay traced to audit entries. Touches the review gate, so it needs its
own reviewed PR.

## Order and dependencies
1 → 2 → 3; 4 → 5; 6 independent; 7 after 2 and 3. Slices 2's provenance change and 7's review-gate change are the two
that touch guarded code and so need explicit human review.
