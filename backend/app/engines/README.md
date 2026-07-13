# Cross-engine orchestration (v0)

Run the same candidate PK models across **multiple estimation engines** (this
app's FOCE-I / SAEM today; nlmixr2 / FeRx / Monolix via adapters), normalize
every fit into one `EngineResult`, and pick a winner on an **engine-agnostic**
footing.

## The scientific crux

Each engine reports its likelihood (`ofv`, and the `aic`/`bic` derived from it) on
**its own algorithm's scale**. Comparing OFV/AIC/BIC *across* engines (FOCE vs
SAEM vs a NONMEM/Monolix/nlmixr fit) is invalid — different approximations, not a
common yardstick. So:

- **Cross-engine ranking uses only prediction metrics** — `pred_rmse`,
  `vpc_coverage90`, `pred_r2`, `|pred_bias|` — computed by pushing every engine's
  estimates through the *same* app simulator (`obs_vs_pred`, `pcvpc`) on the
  *same* data. Identical footing.
- **OFV/AIC/BIC are within-engine only** — reported in a separate table bucketed
  by engine, never consulted for the winner. `select_winner`'s ranking key
  structurally cannot read them.

## Files

| File | Role |
|------|------|
| `base.py` | `EngineResult`, `CandidateSpec`, `EngineAdapter` protocol, `aic_bic`/`k_from_result` (AIC/BIC derived — the native fit has no such key) |
| `scoring.py` | `score_predictions` — the engine-agnostic goodness fields; `vpc_coverage` |
| `native.py` | `PharmAgentAdapter(method=…)` — wraps the real `population_fit` (FOCE-I/SAEM as two engines) |
| `mock.py` | `MockEngineAdapter` — keyless, deterministic stand-in (oracle at `bias=0`) |
| `nlmixr2.py` | **Real** external R engine — shells out to nlmixr2, parses estimates, scores on our footing (`oral_1cmt`/`iv_1cmt`); `available()→False` without R |
| `dataset_io.py` | `subjects → NONMEM CSV` writer (ID/TIME/DV/AMT/EVID/CMT) for external engines |
| `scoring.py` | `score_from_population` — EBEs via the app's `map_estimate`, uniform across engines |
| `runner.py` | `run_matrix_subjects` / `run_matrix` — fan candidates × engines, tolerate absent/failed |
| `select.py` | `select_winner` — prediction ranking + separate within-engine likelihood table |
| `demo.py` | runnable end-to-end demo (uses real nlmixr2 when installed) |

Every engine is scored on **identical footing**: `score_from_population` recomputes
individual predictions on our side (via `app.compute.nlme.map_estimate`) from each
engine's population (θ, Ω%CV, σ), so an external fit that exposes only fixed
effects is judged exactly like the native fit — only the parameters differ.

## Run the demo

```bash
cd backend && source .venv/bin/activate
python -m app.engines.demo
```

Captured run (6 simulated `oral_1cmt` subjects, seed 7; `nlmixr2_focei` is the
real R engine, `monolix_like` a mock stand-in) — full output in
`papers/PAPER1_demo_run.txt`:

```
Cross-engine comparison — oral_1cmt, 6 subjects, 3/3 engines available

PREDICTION RANKING (engine-agnostic — this picks the winner)
  engine           pred_rmse  vpc_cov90  pred_r2   |bias|
  nlmixr2_focei       0.0699     1.0000   0.9866   0.0012
  pharmagent_focei    0.0843     1.0000   0.9805   0.0140
  ensemble            0.0849     0.7500   0.9803   0.0255
  monolix_like        0.1673     0.3750   0.9233   0.1111

WITHIN-ENGINE LIKELIHOOD (never compared across engines)
  pharmagent_focei   ofv=-107.8   nlmixr2_focei ofv=-210.5   monolix_like ofv=-82.8
WINNER: nlmixr2_focei
```

The `ensemble` row is the geometric-mean consensus of the converged fits
(`build_ensemble`), scored the same way and competing for the winner — motivated
by ensembles beating single methods for human-PK prediction (Käser et al.,
*Mol. Pharm.* 2026). It does **not** always win (here it does not); the ranking
decides, and the mechanism makes no accuracy claim of its own.

The two engines fit the same data yet report very different native OFVs
(−107.8 vs −210.5), which is exactly why ranking uses predictions, not OFV. The
`nlmixr2_focei` row requires R + nlmixr2 and the arm64 handling below; without
them that engine is reported absent/failed and the demo ranks the rest.

## Use it in code

```python
from app.engines import (CandidateSpec, PharmAgentAdapter, MockEngineAdapter,
                         run_matrix_subjects, select_winner)

specs = [CandidateSpec("oral_1cmt", iiv_params=["CL", "V"]),
         CandidateSpec("oral_2cmt", iiv_params=["CL", "V"])]
adapters = [PharmAgentAdapter(), MockEngineAdapter("nlmixr2_like", bias=0.05)]

matrix = run_matrix_subjects(subjects, specs, adapters)   # subjects: app NLME contract
sel = select_winner(matrix["results"])
print(sel["winner"].engine, sel["winner"].model_name)
```

`run_matrix(df, roles, ...)` is the dataframe entry point — it builds `subjects`
via the app's canonical `_build_subjects` converter (no second parser).

## As an agent tool

Wired as the modeler tool **`run_engine_comparison`** (`app/tools/engine_tools.py`,
registered in `builtins.default_registry`). It pulls the loaded dataset, fits the
candidate(s) across the requested engines, and writes the winner + ranking to
`PharmState.engine_comparison_results` (audit-safe via `EngineResult.to_audit_dict`).

```jsonc
// tool args
{ "candidates": [{ "model_key": "oral_1cmt", "iiv_params": ["CL","V"] }],
  "engines": ["pharmagent_focei", "nlmixr2"] }   // defaults to these two
```

## nlmixr2 adapter notes

- Supports `oral_1cmt` and `iv_1cmt` in v0; other keys return a `failed` result.
- `available()` gates on `Rscript` + the `nlmixr2` + `jsonlite` packages.
- When an **x86_64 Python runs on an Apple-silicon host**, it forces R to
  `arch -arm64` so the model compiles native arm64 — otherwise the arm64 R cannot
  load an x86_64 `.so` (the fit then fails with an architecture-mismatch error).
  Detection probes whether an arm64 binary can run (it does not rely on
  `sysctl.proc_translated`, which a sandbox can block); no-op on native
  arm64/Intel/non-macOS.
- To add another external engine (FeRx / Monolix), implement `EngineAdapter`:
  `available()` gates the runtime; `fit()` writes the CSV (`dataset_io`), shells
  out, parses estimates, and calls the **same** `score_from_population`.

## Tests

28 framework tests collect across `tests/test_engines.py`,
`tests/test_engine_tools.py`, and `tests/test_engines_nlmixr2.py` (the last
self-skips without R). The load-bearing one,
`test_selection_never_ranks_across_engines_by_ofv`, constructs an engine with far
lower OFV but worse predictions and asserts it does **not** win. Modeling-review
coverage (the cross-engine gate) lives in `tests/test_adversarial.py`.

## Tests

- `tests/test_engines.py` — core (keyless): schema, runner, selection invariants,
  `dataset_io` layout. Load-bearing: `test_selection_never_ranks_across_engines_by_ofv`.
- `tests/test_engine_tools.py` — tool wiring (mock engine, fast): registration,
  guards, JSON-safe audit payload, write-access.
- `tests/test_engines_nlmixr2.py` — **live** nlmixr2 fit; self-skips without R.

## Known open items

- Winner export / adversarial-review reuse assumes a native-shaped `raw`;
  external winners need a synthesized nl-dict first.
- PD-endpoint scoring is out of scope for v0 (scores concentration only).
- nlmixr2 adapter covers 1-compartment models; add 2-cmt templates as needed.
