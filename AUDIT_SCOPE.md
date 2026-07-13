# PharmAgent Audit Scope

This repository is the clean audit surface for PharmAgent: source only, with
dependencies excluded by `.gitignore`. Prefer auditing a fresh clone so review
tools do not spend time on virtual environments, package installs, build output,
or generated caches.

```bash
git clone git@github.com:drsyedzaidi/pharmagent.git
cd pharmagent
```

If auditing the existing local checkout instead, ignore:

- `**/.venv`
- `**/node_modules`
- `**/dist`
- generated `graphify-out/` cache artifacts

## Repository Map

| Area | Approx. files | Review purpose |
|---|---:|---|
| `backend/app/` | 67 | FastAPI service, deterministic compute, orchestration, persistence, audit trail |
| `backend/pharmacometricsbench/` | 14 | Evaluation harness and PK-DB loader |
| `backend/tests/` | 32 | Backend regression and adversarial tests |
| `frontend/src/` | 7 | Primary React/Vite UI |
| `dashboard/src/` | 17 | Dashboard/marketing React/Vite UI |

The highest-value audit surface is the Python backend core, especially security
and correctness boundaries where external input reaches files, subprocesses,
persistence, LLM calls, or numerical solvers.

## Priority 1: Security And Correctness Hotspots

Audit these first.

- `backend/app/main.py`
  - API endpoint input validation
  - bearer auth behavior
  - session/user ownership checks
  - error responses that could leak sensitive data

- `backend/app/core/audit.py`
  - SHA-256 hash-chain integrity
  - tamper-evidence assumptions
  - replay, truncation, and chain-reset edge cases

- `backend/app/core/orchestrator.py`
  - session locking
  - concurrency behavior
  - duplicate/overlapping job execution
  - failed-step rollback or recovery behavior

- `backend/app/core/provenance.py`
  - reproducibility metadata
  - provenance completeness
  - trust boundaries between tool output and reportable claims

- `backend/app/core/schema_extractor.py`
  - data-privacy stripping
  - accidental retention of PHI/PII-like fields
  - schema inference on malformed or adversarial datasets

- `backend/app/core/llm.py`
  - prompt construction
  - secret handling
  - model-provider error handling
  - separation between LLM text and deterministic tool output

- `backend/app/core/store.py`
  - SQLite persistence boundaries
  - path handling
  - ownership/session isolation
  - corruption and partial-write behavior

- `backend/app/core/jobs.py`
  - async job offload
  - lifecycle and cancellation semantics
  - exception capture
  - result isolation between sessions

- `backend/app/engines/nlmixr2.py`
  - subprocess invocation of R
  - command injection resistance
  - untrusted input passed to scripts, filenames, environment, or working dirs
  - timeout handling and cleanup of temp files

- `backend/app/tools/`
  - schema validation at tool boundaries
  - consistent error handling
  - unsafe assumptions about dataframe columns, units, and user-provided paths
  - distinction between user-facing messages and server-side diagnostics

- `backend/app/compute/nlme.py`
  - convergence and edge-case handling
  - numerically unstable likelihoods
  - impossible parameters, NaN/Inf propagation, and singular matrices
  - BLQ/M3 and covariate model correctness

- `backend/app/compute/compartmental.py`
  - solver stability
  - invalid dose/time/concentration inputs
  - parameter bounds
  - model selection edge cases

- `backend/app/core/exporters.py` and report/DOCX generation paths
  - path traversal resistance
  - safe output filenames
  - template/data injection into generated documents
  - overwrite and cleanup behavior

## Priority 2: API, Data, And Tool Boundary Review

After the hotspots, review the remaining backend modules for:

- Trust-boundary validation before any compute or persistence call.
- Consistent auth and session ownership enforcement across endpoints.
- Deterministic tool output being the only source for reported numeric claims.
- Strict separation between raw user uploads, derived datasets, reports, and
  audit/provenance records.
- Safe handling of malformed CSV/dataframe inputs, missing columns, duplicate
  subjects, mixed units, BLQ values, and non-monotone sampling times.
- Fail-closed behavior for filters, selectors, and dataset extraction.

## Known-Good / Recently Hardened Area

`backend/pharmacometricsbench/pkdb/` has already had two adversarial-review
rounds in this session. The fixes covered 20 defects, including fail-open
filters and substance-collision handling. Treat it as recently hardened, but
still regression-test it when changing shared parsing, filtering, or dataset
normalization code.

The broader `backend/app/` core has not had the same adversarial pass yet and
should receive the deepest review attention.

## Suggested Review Order

1. Threat-model `backend/app/main.py` request flows into core state, tools, and
   exports.
2. Review subprocess and file-path surfaces: `nlmixr2.py`, exporters, reports,
   temp files, and store paths.
3. Review session isolation and job concurrency: orchestrator, store, jobs, and
   audit chain.
4. Review tool and compute boundary validation, especially dataframe schemas and
   numerical edge cases.
5. Run the backend tests, then add targeted adversarial tests for any confirmed
   bug or ambiguous boundary.

## Verification Baseline

Use the keyless backend suite first:

```bash
cd backend
pytest -q
ruff check app tests
```

Frontend checks:

```bash
cd frontend
npm ci
npm run build
```

Dashboard checks:

```bash
cd dashboard
npm ci
npm run build
```

Do not require API keys for the baseline audit. Real-model tests should be
separate and opt-in because the default backend test suite is keyless.
