# Loop Safety — PharmAgent

PharmAgent runs unattended automation against a codebase that produces
**regulatory-style artifacts** (audit trails, e-signatures, run provenance,
reproducibility reports). A loop that corrupts those is worse than a loop that
does nothing. Treat every loop like a production operator under change control.

## Unattended automation risks

| Risk | Mitigation |
|------|------------|
| Loop edits auth / owner / audit / e-signature code | Denylist in `loop-constraints.md`; human gate; no auto-merge |
| Loop enqueues expensive NLME/SCM jobs | Admission control in `loop-budget.md`; loops never submit real fits |
| Loop weakens a test to go green | Constraint: never disable/skip/xfail tests; verifier rejects |
| Over-permissioned MCP / tool scope | Triage is read-only (`allowed-tools`); write/network scope needs human approval |
| Infinite fix loop burning budget | Max 3 attempts → escalate; no-progress guard; token cap → report-only |
| Secret exfiltration via prompts/state | Denylist `.env*`, secrets, service-role keys; never log secrets in STATE.md |
| Silent scope creep in a "small fix" | One fix per run; `minimal-fix` skill; verifier confirms diff scope |
| Payments/enrollment logic changed unattended | Stripe/enrollment functions are denylisted — human only |

## Gates before promoting a loop to L2 (assisted)
- [ ] Path denylist documented in `loop-constraints.md` — **done**
- [ ] Verifier (`loop-verifier`) runs `ruff` + `pytest` in an isolated worktree
- [ ] No auto-merge; draft PR + human review
- [ ] MCP / tool scope least-privilege (read-only triage) — **done**
- [ ] `loop-run-log.md` observability + `loop-budget.md` caps + kill switch — **done**
- [ ] One clean week of L1 report-only runs first

## Gates before L3 (unattended)
Do **not** run PharmAgent loops unattended (L3) while auth, audit, e-signature,
jobs-admission, or payments findings are open. L3 requires all of the above **plus**
proven, logged loop activity and a human-reviewed allowlist of exactly which paths a
loop may mutate without approval.

## Reporting
Security issues in the app itself: do not open public issues for exploitable bugs.
Track them in `STATE.md` under the finding queue and fix via the human-gated flow.
