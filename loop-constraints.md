# Loop Constraints — PharmAgent

> The `loop-constraints` skill reads this file at the **start of every run** and
> enforces every rule. These constraints are **binding** — the agent MUST follow them.
> This skill runs BEFORE triage or any action.

## Kill switch (check first)
- If `loop-pause-all` appears at the top of `STATE.md`, **exit immediately**.

## Push & Merge
- Never push before telling the human.
- **Never auto-merge to main.** Always open a draft PR and wait for human review.
- Never self-close a security finding, issue, or PR.

## Paths — never edit unattended (denylist)
- `backend/app/main.py` — auth, bearer/owner, and SPA static-serving logic.
- Anything implementing **audit trail, e-signature, human-review gate, run provenance**.
- `backend/app/core/jobs.py` — job admission / concurrency logic.
- Stripe / enrollment functions (`dashboard/netlify/functions/*`).
- `.env`, `.env.*`, secrets, credentials, service-role keys.
- Infrastructure / CI config without human approval.

## Code
- Always run the gate before proposing a fix: `cd backend && ruff check app tests && pytest -q`.
- Never disable, skip, or `xfail` a test to make CI green.
- **One fix per run.** Never refactor unrelated code.
- Keep the diff minimal (use the `minimal-fix` skill).
- A separate `loop-verifier` (checker) must approve before anything is proposed —
  maker and checker are never the same role.

## Attempt limit & stall detection
- **Max 3 fix attempts per item**, then escalate to a human.
- **No-progress guard:** if two consecutive attempts yield the same failing test or
  the same error, stop — do not retry — and escalate.

## Tool scope (least-privilege)
- Triage runs read-only (`allowed-tools` in `.claude/skills/loop-triage/SKILL.md`).
- No loop is granted write-scoped or network MCP connectors without human approval.

## Communication / escalation
- Always state what you're about to do before doing it.
- Human-in-the-loop is mandatory for auth, audit, e-signature, jobs admission,
  payments/enrollment, or anything outside the CI lint/test allowlist.

## Budget
- At 80% of the daily token cap (`loop-budget.md`), switch to report-only.

---
<!-- Repo-specific rules above. Add project rules below. -->
