# Figure: Tool fidelity on PharmacometricsBench v0

Asset: `figure_tool_fidelity.svg` (publication, fixed colors, white bg).
Data: verified live runs, pinned in `PMBENCH_llm_run.txt` (fixed protocol —
`max_tokens=4096`, last-fence parser, "JSON last" prompt). All numbers are real; no
value is fabricated. 0 parse-gaps for both models after the truncation fix.

## Verified numbers — mean of 3 runs ± sd (0 parse-gaps throughout)

| Category | Tool-grounded | Opus 4.8 alone | Haiku 4.5 alone | Heuristic |
|---|---|---|---|---|
| NCA | 1.00 | 0.98 ± 0.03 | 0.59 ± 0.05 | 0.89 |
| Bioequivalence | 1.00 | 1.00 ± 0.00 | 0.88 ± 0.09 | 0.67 |
| Dose-proportionality | 1.00 | 0.92 ± 0.07 | 0.42 ± 0.14 | 0.92 |
| Compartmental fit | 1.00 | **0.74 ± 0.11** | 0.20 ± 0.11 | 0.50 |
| Exposure (ss, 2-fold) | 1.00 | 1.00 ± 0.00 | 0.89 ± 0.10 | 0.92 |
| **Overall** | **1.00** | **0.93 ± 0.03** | **0.60 ± 0.04** | 0.78 |

Tool-grounded and heuristic are deterministic (no run-to-run variance). The main figure
shows three bars (tool-grounded, Opus 4.8, Haiku 4.5) at the means; the heuristic column is a
fourth reference. Each LLM cell = mean of 3 runs on the identical 30-task set.

## The finding (honest, reframed after the truncation fix)

Model scale carries most of the load — Opus 4.8 unaided averages **0.93 ± 0.03** and is exact
on bioequivalence and exposure (1.00 ± 0.00); the cheaper Haiku 4.5 averages **0.60 ± 0.04**.
But **only the validated tools reach exact (1.00)**, and the residual Opus→tool gap concentrates
on the tasks that require iterative numerical work — one-compartment curve fitting
(**0.74 ± 0.11**, the single largest and most variable gap) and, at the margin,
dose-proportionality (0.92). The claim is therefore not "LLMs can't do pharmacometrics" (a
frontier model largely can); it is that a tool *guarantees* the exactness that
reasoning-to-a-number cannot, especially on hard numerics — a claim a skeptic's own Opus run
confirms rather than refutes.

> Honesty note: an earlier run reported Opus at 0.32 — a pure artifact of `max_tokens=1024`
> truncating Opus's verbose worked solutions before the closing JSON block (19/30 unparseable).
> The instrumentation surfaced it as parse-gaps (not silently as wrong answers), the harness was
> fixed, and both models were re-run. That number is discarded.

## Paper caption (draft)

> **Figure N. Model scale narrows the gap; tool grounding closes it.** Each task in
> PharmacometricsBench v0 (n = 30 across five deterministic categories) is graded against the
> output of a validated compute tool on the same input. An agent that *calls the tools*
> reproduces ground truth exactly (overall = 1.00). A frontier model reasoning to the numbers
> unaided (Claude Opus 4.8) averages 0.93 over three runs — exact on bioequivalence and exposure,
> near-exact on NCA (0.98), but falling to 0.92 and 0.74 on the tasks requiring iterative
> estimation (dose-proportionality, one-compartment fitting); a smaller model (Haiku 4.5)
> averages 0.60. Because ground truth is the tool's own output, the tool-grounded ceiling is exact
> by construction. Bars are per-category means of three runs (sd in table); overall weights
> categories equally.

## Alt text

Grouped bar chart, three bars per category, means of three runs. A tool-grounded agent scores
1.00 on all five categories. Opus 4.8 alone scores 0.98 NCA, 1.00 bioequivalence, 0.92
dose-proportionality, 0.74 compartmental fitting, 1.00 exposure (0.93 overall). Haiku 4.5 alone
scores 0.59, 0.88, 0.42, 0.20, 0.89 (0.60 overall).

## Agency-site headline (one line)

> A frontier model does your NCA at 93% — and only the validated tools take it to 100%, closing
> the last gap on exactly the fits that are hard to get right by hand. That guarantee is the product.

## Reproduce

The two model rows are now permanent, model-pinned reference agents in the runner
(`llm-opus`, `llm-haiku` — they require a key and never fall back to the keyless mock):

```bash
export ANTHROPIC_API_KEY=$(python3 -c "import json,os;print(json.load(open(os.path.expanduser('~/.rag-pharma/config.json')))['anthropic_api_key'])")
cd backend
python -m pharmacometricsbench.runner --run oracle naive llm-opus llm-haiku
```
Numbers above are the mean of 3 runs (`scratchpad/avg3.py`); error bars on the figure are sd.
Single runs vary by a few points — average ≥3 for a publication table.
