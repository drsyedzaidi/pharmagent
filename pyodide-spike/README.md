# P0 — Pyodide/WASM numerical-parity spike

The **de-risking prototype** for the browser-native port
(`docs/WASM_BROWSER_NATIVE_SPEC.md`, §7 Phase 0). It answers the one question that
gates the whole decision: **does PharmAgent's compute produce the same numbers
inside a browser (Pyodide/WASM) as it does on the shipping backend (CPython)?**

Everything else in the migration — swapping `api.ts` transport, Blob downloads,
IndexedDB persistence, lazy loading — is mechanical and known-good. The two things
that are *not* proven are (i) whether WASM + Pyodide's **older** scipy/pandas pins
reproduce the validated FOCE-I/SAEM estimates, and (ii) whether a fit finishes in
tolerable time. This spike measures both, with **no UI**.

## What it does

One shared harness (`harness.py`) runs under both interpreters, importing the
**exact shipping modules** (`app.compute.nlme`, `app.tools.pkmodel_tools`,
`app.compute.flexplot`) — the same validated path as
`backend/tests/reference/test_theophylline.py`. It computes five layers of
increasing optimizer-sensitivity on the 12-subject Theophylline cohort:

| Layer | Exercises | Purpose |
|-------|-----------|---------|
| `micro` | `scipy.linalg.expm`, `integrate.solve_ivp`, `stats.t`, `gaussian_kde` | raw float/BLAS parity |
| `nca` | numpy trapezoid + log-linear terminal slope | deterministic PK parity |
| `flex` | the custom loess + t-quantile CI | visualization-path parity |
| `focei` | validated FOCE-I fit (`max_iter=40`) | iterative optimizer parity |
| `saem` | validated seeded SAEM fit (`max_iter=120`, seed `20250614`) | MCMC parity |

`compare.py` diffs the two result JSONs with tolerances that widen as optimizer
sensitivity rises (near-exact for `micro`/`nca`; a few percent for the fits) and
prints a **GREEN / RED** verdict. The layering is diagnostic: if `micro` matches
but the fits drift, the cause is optimizer-path sensitivity (likely the pin
difference); if `micro` drifts, it's fundamental WASM float divergence.

## Run it

```bash
# 1. CPython baseline (shipping interpreter) — from the repo root:
backend/.venv/bin/python pyodide-spike/run_cpython.py          # or --quick

# 2. Pyodide/WASM (Node host):
cd pyodide-spike
npm install                                                     # once (~pulls pyodide)
node run_pyodide.mjs                                            # or --quick
#   first run downloads Pyodide + numpy/scipy/pandas (~30-40 MB) and is slow;
#   the FULL fits take minutes single-threaded — that IS the perf signal.

# 3. Verdict:
python compare.py     # (stdlib only — any python works)
```

Use `--quick` on both runners first for a fast wiring check (fewer fit
iterations), then a `full` run for the real parity + latency numbers. **Both
runners must use the same mode** (compare.py checks it).

## Reading the result

- **GREEN** — parity holds. The numerical risk is retired; the port is a green
  light (subject only to the fit *latency* you observed being acceptable).
- **RED** — a metric is out of tolerance. This is the spike doing its job: you
  learned it in an afternoon, not a quarter in. Note **which** layer drifted and
  the printed **version delta** (Pyodide typically ships pandas ~2.2 / scipy
  ~1.14-1.16 vs the backend's pandas 3.0 / scipy 1.17). Common culprit: pandas
  3.0→2.2 behavior (Copy-on-Write, PyArrow-string defaults). Pin-align or adjust
  before committing to the migration.

## Notes / caveats

- The spike mounts `backend/app/**.py` into the WASM virtual filesystem and imports
  only `app.compute.*` + `app.tools.pkmodel_tools` — none of the server blockers
  (FastAPI, SQLite, `subprocess`, `anthropic`) are touched, confirming the
  compute core is import-clean under Pyodide.
- No LLM, no network, no DB, no server — pure compute parity.
- Record the `pyodide` npm version you install; it determines the numpy/scipy/pandas
  pins and therefore the parity result. The runner prints the resolved versions.
- Artifacts (`results_*.json`, `node_modules/`) are gitignored.
