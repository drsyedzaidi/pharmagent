# LOOP.md — PharmAgent Loop Operation

How PharmAgent is operated with loop-engineering patterns. PharmAgent is a
regulated-style agentic pharmacometrics app (FastAPI backend + React frontend,
deterministic compute + NLME/SCM, SHA-256 audit + human-review gate). Loops here
touch a codebase that produces **regulatory artifacts** — treat every loop like a
production operator under change control.

## Active Loops

### CI Sweeper (L1 → L2)
- **Cadence:** on push / PR, plus opportunistic manual `/loop`.
- **Trigger:** failing `.github/workflows/ci.yml` (ruff, pytest keyless MockLLM, frontend tsc/build).
- **Skill:** `loop-triage` (read-only) → report; `minimal-fix` in a worktree → `loop-verifier` gates.
- **Level:** **L2 (assisted auto-fix)** as of 2026-07-14 — owner decision, streak-1 override.
  Assisted-fix allowed ONLY for lint + obviously-scoped test breakage. `minimal-fix` in a
  worktree → `loop-verifier` runs ruff+pytest → **draft PR for human approval. No auto-merge.**
  Denylist (auth/audit/e-signature/jobs/payments) stays **human-only** — L2 never touches it.
- **State:** `STATE.md`.

### Security-Finding Triage (L1 — report-only)
- **Cadence:** after each security scan (`/security-scan` / codex scan) or weekly.
- **Input:** the findings queue in `STATE.md` when a scan leaves unresolved items.
- **Phase:** report + propose minimal fix per finding; **human approves every fix**
  touching auth, audit, e-signature, jobs admission, or Stripe/enrollment.
- **Never** self-close a finding. Verifier + human sign-off required.

### Dependency Sweeper (L1 — advisory)
- **Cadence:** weekly.
- **Source:** `pip-audit` (already advisory in CI) + `npm audit`.
- Patch + low-risk CVE only; majors and denylisted packages are a human gate.

## Multi-loop priority
CI Sweeper → Security-Finding Triage → Dependency Sweeper. Only one loop mutates
files at a time; each runs in an **isolated git worktree** and discards on REJECT.

## Worktrees
- Every unattended code change runs in a per-attempt git worktree.
- One fix per worktree; discard after verifier REJECT or human escalation.

## Connectors (MCP) — least privilege
- Loops get **read-only** tool scope by default (see `allowed-tools` in
  `.claude/skills/loop-triage/SKILL.md`).
- No MCP connector with write/network scope is granted to an unattended loop
  without explicit human approval. GitHub MCP, if used, is read + PR-comment only.

## Budget & Observability
- Token caps + kill switch: `loop-budget.md`.
- Run history: `loop-run-log.md` (append one JSON line per run).
- NLME / SCM / engine-comparison jobs are the expensive path — see the
  admission-control note in `loop-budget.md`.
- **Kill switch:** set `loop-pause-all` at the top of `STATE.md`; loops exit immediately.

## Safety & Gates
- Full risk model: [docs/safety.md](docs/safety.md).
- Binding constraints (read every run): [loop-constraints.md](loop-constraints.md).
- **Denylist (never edit unattended):** `backend/app/main.py` auth/owner logic,
  anything under audit / e-signature / human-review, `backend/app/core/jobs.py`
  admission logic, Stripe/enrollment functions, `.env*`, secrets.
- **No auto-merge to main.** Draft PR + human review always.

## Escalation / stall
- Max **3 fix attempts** per item, then **escalate to human** (stop and ask).
- Stall / no-progress guard: if two consecutive attempts produce the same failing
  test or the same error, stop — do not retry — and escalate.
- Human-in-the-loop is mandatory for any change outside the CI-lint/test allowlist.

## How to run locally
```bash
# score readiness
npx @cobusgreyling/loop-audit . --suggest
# backend gate (keyless — MockLLM)
cd backend && ruff check app tests && pytest -q
```

## Status (2026-07-13)
| Loop | Level | Automation | Notes |
|------|-------|------------|-------|
| CI Sweeper | **L2** | ⏸ manual `/loop` | promoted 2026-07-14 (owner, streak-1); assisted-fix lint+test only, draft-PR gated |
| Security-Finding Triage | L1 | ⏸ manual | 2026-07-13 findings fixed in c9d2658; no open queue |
| Dependency Sweeper | L1 | ✅ pip-audit advisory in CI | npm audit not yet wired |

---
*This file is documentation and the seed for the loops that maintain PharmAgent.*
