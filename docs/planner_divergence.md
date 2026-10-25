# Planner divergence harness

`backend/app/evals/planner_divergence.py` runs the goal planner's `poppk_goal`
leg several times on the same dataset and compares the runs decision by
decision. It never runs a fit.

## Purpose

The goal planner (an LLM) proposes each next tool. The deterministic policy
(`backend/app/core/policy.py`) then admits the proposal, gates it for a human,
or rejects it. When the same leg is replayed on the same data, the replicates
should make the same proposals wherever the rules leave only one sensible
choice. If they disagree at some point, there are two possible reasons:

- **A rule is missing or under-specified.** The planner has no basis for the
  choice, so it picks differently from run to run. The fix is a rule.
- **The decision is a genuine judgment branch.** Several choices are
  defensible, for example a structural model when the data cannot tell two
  models apart. Here the divergence is useful information for the human at
  the gate and should not be suppressed by a rule that pretends one answer is
  right.

The harness finds these points and names them. It does not decide which of the
two reasons applies, and it changes nothing.

**Converting a divergence into a policy rule or a mikebrain rule is an owner
decision.** The report lists *candidate* rule gaps only. Nothing in the
harness edits `policy.py`, the planner prompt, the gates, or any mikebrain
rule.

## The no-fit guarantee

- Each replicate calls the real `Orchestrator.start_workflow` in a fresh
  in-memory session (`SessionStore(":memory:")`, no audit key, nothing written
  to the app database).
- `allow_expensive` is never passed. It keeps its default (`False`), so
  `ToolRegistry.execute` refuses every long-running fit with
  `ExpensiveToolError`.
- The leg ends at its **first gated proposal**. The policy gates every
  long-running fit (`run_nlme`, `run_scm`, ...) and every scientific model
  choice, so a fit is at most *proposed*, never run. The harness never
  approves or resumes a proposal.
- If the policy ever admitted an expensive tool without a gate (a
  regression), the registry still raises `ExpensiveToolError`. The replicate
  then ends as `expensive_refused`, the run is reported `INCOMPLETE`, and the
  CLI exits 5.
- Every execution passes through a thin wrapper around that replicate's own
  registry. The wrapper records the `allow_expensive` flag each call actually
  received. A call that arrives with the flag set is refused before it
  reaches the registry, so a harness regression is stopped before a fit
  runs, not just detected after it. After the replicate, the harness also scans
  the session's audit chain for executed tools.
- If either check shows an expensive execution, or any call with
  `allow_expensive=True`, the harness raises `HarnessGuaranteeError`. The CLI
  then writes `planner_divergence.FAILED.txt` (with the offending replicate's
  record), writes no report, and exits 4.
  The report's guarantee line gives these observed counts; it is not a fixed
  sentence.

The autonomous steps that do run are cheap: load, profile, validate, the
spaghetti plot, and the structural comparison `fit_pk_model(compare=true)`,
about 1 s per replicate on the sample data.

## Running it

From `backend/`, with the project venv. The dataset must sit under the
PharmAgent data directory or `backend/sample_data`, the same confinement
`load_dataset` applies.

Self-test with the keyless mock planner. Every replicate must be identical; the
command exits 1 if they are not:

```bash
python -m app.evals.planner_divergence --dataset sample_data/oral_pk.csv --replicates 5 --provider mock --out /tmp/divergence_mock
```

A local model through Ollama (start Ollama and `ollama pull qwen3:8b` first):

```bash
python -m app.evals.planner_divergence --dataset sample_data/oral_pk.csv --replicates 5 --provider local --model qwen3:8b --out /tmp/divergence_qwen3
```

Options:

| flag | meaning |
|---|---|
| `--replicates N` | 2 to 50 (default 5) |
| `--provider` | `mock` (self-test), `local` (Ollama or LM Studio on loopback, the default), or `configured` (whatever the app's environment configures, for example Claude when an Anthropic key is set) |
| `--model` | local model name (default `gemma4:31b`) |
| `--base-url` | local server (default `http://127.0.0.1:11434/v1`); must be a loopback address |
| `--temperature T` | local only; default is the app's own setting, 0 |
| `--seed S` | local only, needs `--temperature` > 0; replicate *i* samples with seed *S + i* |
| `--timeout S` | local only: seconds per planner call (default 120, the client's own). gemma4:31b needed more: with other jobs loading the CPU, its first call ran past 120 s in 2 of 3 replicates. Use 600 to 900 for a 31B model |
| `--allow-hosted` | `configured` only: required when the configuration resolves to a hosted API (OpenAI, Anthropic), which then receives the planner's state summaries |
| `--out DIR` | where `planner_divergence.json` and `planner_divergence.md` are written |

**Choosing the sampling.** With the defaults the harness measures the planner
as the app runs it (temperature 0). Agreement there shows the planner's
preferred path is stable. It does not show how close the runner-up choice was,
so it is weak evidence that the rules are adequate. To surface near-ties, run
with sampling on, for example `--temperature 0.7 --seed 1`. Each replicate then
gets its own seed, and the run as a whole can be reproduced. Even at
temperature 0, local and hosted models are not guaranteed to be bit-for-bit
deterministic, so some run-to-run differences can appear without sampling.

Expect roughly one planner call per step, about six steps per replicate. On
this Mac, qwen3:8b took 60 to 145 s per replicate (~8 min for five), and
gemma4:31b took 383 to 695 s per replicate (~45 min for five, with
`--timeout 600`).
Hosted providers (`configured --allow-hosted`) receive the same compact state
summary the app's planner already sends. `--provider configured` refuses to
run when nothing is configured (it would silently be the mock).

Exit codes:

| code | meaning |
|---|---|
| 0 | report written (`ALL-CONVERGED` or `DIVERGED`) |
| 1 | mock self-test failed |
| 2 | bad arguments or dataset |
| 3 | `INCOMPLETE` |
| 4 | no-fit guarantee breached |
| 5 | policy regression: the policy admitted an expensive tool and the registry refused it | Each finished replicate is also appended to
`replicates.jsonl` in the output directory, so an interrupted run keeps what it
has done.

## Reading the report

`planner_divergence.md` is the short summary. `planner_divergence.json` holds
everything, including each replicate's full decision list. Neither contains
the dataset's path. The dataset appears by file name and SHA-256 (taken
before and after the run), and any occurrence of the path in arguments or
messages is replaced with `<dataset>`. All model text in the markdown
(rationales, argument keys and values) is flattened to one line, escaped, and
truncated. It cannot add headings, tables or HTML, and it cannot hide a link
behind other text. A bare URL in a rationale may still show as a link. The
CLI's progress lines drop terminal control characters. `stop_detail` keeps the
first 500 characters of an error message. A tool's error can quote a data
value, so treat the report with the same care as the dataset.

**Per replicate:** the ordered autonomous tools that ran, the gated proposal
(tool, arguments, rationale, gate reason), every proposal the policy rejected
(with its reason), the stop reason, the model, the sampling used, and the
elapsed and planner-only seconds. Stop reasons:

| stop reason | meaning | valid? |
|---|---|---|
| `gated_proposal` | the normal end: the planner proposed something the policy gates | yes |
| `planner_complete` | the planner stopped with a reason ("goal met") before any gate | yes |
| `silent_stop` | the planner proposed nothing and gave no text | yes |
| `repeat` | the policy stopped an identical repeated proposal | yes |
| `budget` | the step or gate budget ran out | yes |
| `tool_error` | an admitted tool refused after the data was loaded; the last decision's `outcome` is `tool_error` | yes |
| `dataset_error` | the injected `load_dataset` failed: an input problem, not a planner decision | **no** |
| `planner_error` | the planner call failed: network, timeout, server down | **no** |
| `harness_error` | the leg raised, for example on a malformed planner answer | **no** |
| `expensive_refused` | the policy admitted an expensive tool and the registry refused it: a policy regression | **no** |

Each decision also records its `outcome`: `ran`, `gated`, `rejected`,
`tool_error`, `refused_expensive`, or `ended` for the pseudo-decisions
below.

**Decision points.** Replicates are compared step by step, but only among
replicates whose earlier decisions were identical. Two proposals are therefore
compared only when the same calls on the same data led up to them. Once
replicates split, each branch is compared within itself. Calls are identified
by `policy.canonical_call(tool, args)`, the identity the policy's repeat guard
uses. A replicate that ended without proposing anything still counts as a
decision at that step: `(planner stop)` for a reasoned stop or
`(silent stop)`. Each point lists its replicates and the full call prefix
that led to it, so two branches at the same step can be told apart. A long
argument value is shown as its first characters plus a short hash, so two
values that differ never look the same.

| label | meaning |
|---|---|
| `CONVERGED` | every replicate at this point made the same call |
| `DIVERGED-ARGS` | same tool, different arguments; `varied_arg_keys` names the keys |
| `DIVERGED-TOOL` | different tools, or some replicates stopped while others proposed |
| `UNREPLICATED` | only one replicate reached this point, so there is nothing to compare; never counted as convergence |

**Verdict.** `ALL-CONVERGED` or `DIVERGED`, with the first step at which
replicates diverged. Replicates with an invalid stop are listed but left out
of the comparison. A network blip is not a rule gap. If any replicate is
invalid, the verdict is `INCOMPLETE`. The same applies when any of these
holds:

- fewer than two replicates are valid;
- no replicate made a planner decision beyond loading the data, for example
  a model without tool calling, empty replies, or a first-step failure. Five
  identical "stop" answers agree on nothing that tests a rule;
- the same call had different outcomes across replicates, such as a
  one-off I/O failure;
- the dataset file changed during the run. The report then
says why. When at least two replicates are valid, it also shows
`analysis_verdict` for those replicates alone. A
failed run never reads as convergence.

**Candidate rule gaps** (suggestions for the owner, never applied):

- *Argument keys that varied*, each classified by how the policy treats it:
  - **a policy scientific-choice key** (for example `fit_pk_model.model_key`
    or `run_nlme.error_model`). It is already gated, so a human sees the
    value. Either this is a judgment branch the gate exists for, or the owner
    may want to write down a default.
  - **not a scientific-choice key, but every variant carrying it was gated**
    for another reason (the tool is always gated, or another argument
    triggered the gate). A human saw the value. It needs a rule only if it
    should be gated on its own.
  - **not a scientific-choice key, and the policy rejected a variant carrying
    it** (and none ran without a gate). A rejected proposal never reaches the
    human gate, so no human is known to have seen that value. Check the policy
    rejections for why.
  - **not a scientific-choice key and admitted without a gate** in at least
    one replicate that carried it. The value ran with no human seeing it. If
    it changes the science, this is the strongest candidate for
    `SCIENTIFIC_CHOICE_KEYS` or a planner rule.
- *Tool choices*: points where different tools competed. Either nothing
  orders them (a candidate prerequisite or planner rule), or the order
  genuinely does not matter. When some replicates stopped where others
  proposed a call, the note says so instead. That pattern usually means the
  planner lacked information it needed, so check the state summary it sees.

**Policy rejections** lists the proposals the policy refused, grouped across
replicates. Repeated rejections of the same call (for example `run_vpc` before
any NLME fit) mean the planner does not know a prerequisite the policy
enforces.

## What to do with a divergence

1. Read the variants and their rationales at the first divergent step.
2. Decide whether the choice has one right answer for this kind of analysis.
   - If it does, the divergence marks a missing rule. The owner may add it to
     `policy.py` (gating or prerequisites) or to the mikebrain rule catalog,
     and then rerun the harness to confirm the point converges.
   - If it does not (data that cannot separate the candidates, weak
     identifiability, a defensible difference in judgment), keep the branch
     open. Its value is that it shows the human at the gate that a real
     choice exists.
3. Either way, the harness only reports. Rule changes go through the normal
   owner-reviewed change process.

## Limitations

- Only the first leg is replayed: everything up to the first gate. Decisions
  after a human approval, which would require running a fit, are out of
  scope by design.
- A handful of replicates gives a coarse picture. Five agreeing replicates do
  not prove that a sixth would agree.
- Under the mock provider the planner walks fixed rails, so the mock run
  tests the harness itself, not the rules.
- Calls are compared with `policy.canonical_call`, exactly as the policy does.
  Arguments that differ only in list order (`["a","b"]` vs `["b","a"]`) or
  number format (`1` vs `1.0`) therefore count as different.

## Method source

The replicate-divergence method is re-expressed (not copied) from
Pritchard-Bell A, Lin C-W, Holmes W, Doshi S. Context Engineering for
AI-Assisted Pharmacometrics: A Practical Tutorial. *CPT Pharmacometrics Syst
Pharmacol* 2026;15(8):e70317. doi:10.1002/psp4.70317. PMID 42557662.
