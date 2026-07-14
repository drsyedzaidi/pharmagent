# PharmAgent — Project Conventions

Agentic pharmacometrics web app. FastAPI backend + React frontend. Deterministic
compute (NCA, BE, compartmental, NLME FOCE-I/SAEM, SCM, VPC/GOF, MAP/TDM) with a
SHA-256 audit trail, human-review gate, and run provenance. Keyless tests via MockLLM.

## Build / test gate
```bash
cd backend && ruff check app tests && pytest -q     # keyless — MockLLM
cd frontend && npx tsc --noEmit && npm run build
```
- **Do not run uvicorn with `--reload`.** Restart uvicorn manually after backend edits.
- CI mirrors this: `.github/workflows/ci.yml`.

## Guardrails (regulated-style app)
- Never edit auth/owner, audit trail, e-signature, human-review gate, or run
  provenance code unattended — human review required.
- Never submit a real NLME/SCM/engine-comparison fit from an automated loop
  (expensive; see `loop-budget.md` admission control).
- One fix per change; keep diffs minimal; never disable tests to pass CI.

## Loop operation
This project is operated with loop-engineering patterns. Before any automated run:
1. Read `loop-constraints.md` (binding) and check `STATE.md` for `loop-pause-all`.
2. Triage read-only (`.claude/skills/loop-triage`) → propose via `minimal-fix` in a
   worktree → gate with `loop-verifier`. No auto-merge.
- Cadence, gates, budget, escalation: `LOOP.md`, `loop-budget.md`, `docs/safety.md`.
- Current work queue (8 security findings): `STATE.md`.
