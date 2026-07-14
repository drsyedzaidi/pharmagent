<!-- kill switch: put `loop-pause-all` on the line below to halt all loops -->

# Loop State — PharmAgent

Last run: 2026-07-14T00:24:00Z (security fixes committed + pushed in `c9d2658`)

## Security-Finding queue

Current queue: **empty**.

The 2026-07-13 Codex Security scan `b8fc4338` produced 8 findings (3 high,
5 medium). They were adversarially re-verified, fixed with regression coverage,
and pushed as:

- `c9d2658` — `fix(security): address Codex audit findings (verified, with tests)`

Resolved findings from that scan:

- [x] **Raw bearer token used as owner** — fixed with non-secret hashed principal.
- [x] **Unbounded long-running jobs** — fixed with global/per-session admission caps and HTTP 429 mapping.
- [x] **Service-role lesson path control** — fixed with strict slug and filename allowlists before service-role storage calls.
- [x] **Non-finite PK simulation inputs** — fixed with finite/positive bounds and `n_doses <= 1000` at the tool boundary.
- [x] **Invalid steady-state tau accepted** — fixed by rejecting non-positive dosing intervals.
- [x] **Session mutators bypass lock** — fixed by running `chat` inside `session_lock(sid)`.
- [x] **Prefix-based static containment** — fixed with `is_relative_to` path-boundary check.
- [x] **Enrollment before paid status** — fixed by gating active enrollment on paid/no-payment-required status and async success.

Future security scans should append new unresolved items here as report-only work
queue entries. A loop may propose minimal fixes in an isolated worktree, but a
human must approve anything on the denylist.

## Watch List

- Promote CI Sweeper L1 -> L2 after one clean week of report-only runs.
- Wire `npm audit` into the Dependency Sweeper (only pip-audit today).
- Correlated-IIV / Cholesky-Omega still TODO (tracked in project notes, not a loop item).

## Recent Noise (ignored this run)

—

---
Run log: `loop-run-log.md`. Cadence and gates: `LOOP.md`. Constraints: `loop-constraints.md`.
