# PharmacometricsBench v0

A reproducible eval for **agentic pharmacometrics**. Companion design doc:
`../../PHARMACOMETRICSBENCH.md`.

## The idea in one line

Ground truth for every task is **the output a validated PharmAgent compute tool
produces on the provided data**. So the benchmark measures exactly one thing:
does the agent reproduce the tool's numbers, or free-hand them? An agent that
*calls the tools* scores 1.0; an agent that eyeballs does not. That encodes the
platform thesis — *agents decide, tools execute* — as a number.

## What v0 covers

Five deterministic categories whose oracles need no fitted NLME model and no
external data:

| Category | Oracle | Graded targets |
|----------|--------|----------------|
| `nca` | `app.compute.nca.nca_subject` | Cmax, AUC0-inf, t1/2 |
| `be` | `app.compute.bioequivalence.be_one_parameter` | GMR%, 90% CI bounds, BE verdict |
| `dp` | `app.compute.dose_proportionality.power_model` | power-model slope, proportional? |
| `compartmental` | `app.compute.compartmental.fit_one_subject` | CL, V, ka (1-cmt oral) |
| `exposure` | `app.compute.pk_simulate.simulate_timecourse` | steady-state Cmax_ss, AUC_tau (2-fold) |

Deferred (need fixtures or literature keys): TDM/forecast, PK/PD, PBPK,
identifiability, regulatory, reporting — see the design doc.

### Real-drug data: PK-DB loader (`pkdb/`)

`pkdb/loader.py` pulls **real, cited** pharmacokinetic data from
[PK-DB](https://pk-db.com) (Grzegorzewski et al., *NAR* 2020, doi:10.1093/nar/gkaa990).
Verified against the live API (2026-07-12):

- Anonymous access exposes a real **dosing + study catalogue** (substance, dose,
  route, form, per-study reference + **licence** flag). It does **not** expose the
  drug-level PK answer keys — `outputs.csv` returns empty and `outputs__*` filters
  zero the result set. (The `clearance` rows that leak into `individuals.csv` are
  *creatinine* clearance, a covariate — the loader filters them out by requiring
  `substance == <drug sid>`.)
- The answer-key path (`harvest_pk_parameters` → `fih_tasks.build_fih_pk_tasks`,
  which unit-harmonises and per-study-pools observed values into 2-fold-graded
  `fih_pk` tasks) is built and unit-tested, and **activates once `PKDB_API_TOKEN`
  is set** (free PK-DB account). Only **dose-independent** parameters (CL, CL/F,
  t1/2, V, V/F) seed tasks; dose-dependent AUC/Cmax are harvested but not pooled
  into answer keys (pooling across doses would be ill-defined). No PK values are
  fabricated when the key is absent — the category simply yields zero tasks.

```bash
python -m pharmacometricsbench.pkdb.loader --drugs caffeine paracetamol --coverage
```

Offline tests run against a trimmed **real** fixture (`pkdb/fixtures/caffeine_filter.zip`);
`tests/test_pkdb_loader.py`. Live evidence: `../../papers/PKDB_loader_run.txt`.

## Run it

From `backend/` (so `app` imports resolve):

```bash
# (re)generate the reproducible task set, then score the reference agents
python -m pharmacometricsbench.runner --generate --per-category 6

# score only, against the committed task set
python -m pharmacometricsbench.runner --run oracle naive
```

Typical output:

```
PharmacometricsBench v0 — 30 tasks, 5 categories
  oracle   overall=1.000   [be=1.00  compartmental=1.00  dp=1.00  exposure=1.00  nca=1.00]
  naive    overall=0.778   [be=0.67  compartmental=0.50  dp=0.92  exposure=0.92  nca=0.89]
```

`oracle` (calls the tools) is the top-of-leaderboard reference and the harness
self-test. `naive` (tool-free, plausible-but-wrong) proves the benchmark
discriminates correct process from guessing.

## Grading

Per target, pass/fail under a tolerance rule (`rel`, `abs`, `twofold` GMFE band,
or `exact` for booleans/categories). Task score = mean over its targets. Overall
= mean over **categories** (equal weight, so a big easy category can't dominate).

## Plug in your own agent

An agent is any `Callable[[Task], dict]` returning a value per target name:

```python
from pharmacometricsbench.generators import build_taskset
from pharmacometricsbench.grading import grade_task, score_report

def my_agent(task):
    # task.prompt + task.dataset are the inputs; return {target_name: value}
    ...

tasks = build_taskset(per_category=6)
report = score_report([grade_task(t, my_agent(t)) for t in tasks])
print(report)
```

To evaluate PharmAgent or an LLM, wrap it in that signature (feed `task.prompt` +
`task.dataset`, parse its answer into the target dict). The audit trail then also
yields a **tool-fidelity** rate — the share of numeric answers that came from a
tool call rather than tokens — which only this platform can report.

## LLM adapter (keyless)

`llm.py` routes a task through **prompt → text → parse → grade**. It is keyless
by default: `MockLLM` reads the dataset embedded in the prompt, computes an
answer, and emits realistic prose + a fenced JSON block — so the fiddly part
(robustly parsing a model's free-form answer) is fully built and tested without
an API key.

```
  llm      overall=0.743   [be=0.67  compartmental=0.50  dp=0.92  nca=0.89]
```

The default `MockLLM(strategy="naive")` reproduces the tool-free floor through
the text channel (it matches `naive` exactly — the plumbing is faithful, not a
claim about any real model). `MockLLM(strategy="oracle")` proves the JSON
round-trip is lossless (scores 1.0).

**To score a real model**, just set `ANTHROPIC_API_KEY` — `default_client()` then
returns the built-in `AnthropicLLM` instead of `MockLLM`, and nothing else in the
pipeline moves. No key → keyless MockLLM (so tests/CI never need one).

```bash
export ANTHROPIC_API_KEY=sk-ant-...
# cheapest by default (Haiku 4.5, ~$0.11 for a 30-task run):
python -m pharmacometricsbench.runner --run oracle naive llm
# headline row on a frontier model:
PMBENCH_LLM_MODEL=claude-opus-4-8 python -m pharmacometricsbench.runner --run llm
```

`AnthropicLLM.complete(prompt)` calls the Messages API and returns the text;
`PMBENCH_LLM_MODEL` picks the model (default `claude-haiku-4-5`).

Wrapping **PharmAgent** the same way (its agent answers `task.prompt` +
`task.dataset`) yields the paper's headline: LLM-alone (near the floor) vs
PharmAgent (near the oracle), plus the **tool-fidelity** rate.

## Regression gate

`backend/tests/test_pharmacometricsbench.py` asserts the oracle scores a perfect
1.0 on every category (catches a malformed task or a changed tool) and that the
naive agent is materially worse. Wire it into CI alongside the reference-
validation suite.

## Reference LLM agents + FIH task builder

Two permanent, model-pinned reference agents score the figure's rows (they require
`ANTHROPIC_API_KEY` and never fall back to the keyless mock):

```bash
python -m pharmacometricsbench.runner --run oracle naive llm-opus llm-haiku
```

Real FIH answer-key tasks are built from PK-DB with one command (writes `tasks/fih_v0.jsonl`):

```bash
python -m pharmacometricsbench.pkdb.build_fih_taskset            # anonymous → 0 tasks (auth-gated)
PKDB_API_TOKEN=... python -m pharmacometricsbench.pkdb.build_fih_taskset   # real cited tasks
```
No token → the harvest is empty and it writes zero tasks (never a fabricated one).
