# Simulated peer review — PharmacometricsBench (Paper 2)

*Reviewer persona: CPT:PSP methods/resource reviewer with pharmacometrics + ML-eval expertise. Adversarial, constructive. This is a simulation to stress-test the draft before submission, not an actual review.*

**Recommendation: Major revision.** The idea — a benchmark whose ground truth is validated tool output — is genuinely useful and well motivated, and the manuscript is unusually candid about its limits. But the central empirical claim is currently supported by a tautology rather than an experiment, the real-model evaluation is statistically thin and single-family, and one framing overreaches. All are fixable.

## Major comments

**M1 — ✅ ADDRESSED (2026-07-13).** A tool-using agent was built (`make_tool_agent`: the model calls the five validated compute functions by function-calling) and run three times each for Opus 4.8 and Haiku 4.5. Both reached **1.00 ± 0.00** in every category (from tool-free 0.93 and 0.60), closing the compartmental-fitting gap (0.74/0.20 → 1.00). Draft updated: Table 1 gains two "+ tools" rows; abstract, §2.5, §3.3, Discussion, and Conclusion now report grounding-not-scale as demonstrated, with the honest bound that 1.00 reflects correct tool orchestration and inherited exactness. Pinned in `papers/PMBENCH_tools_run.txt`. Original finding retained below for the record.

**M1 (original) — The headline "only a tool-calling agent reaches the exact result" is not demonstrated by a real agent; it is true by construction of the oracle.**
The oracle *is* a scripted tool-caller, and ground truth *is* the tool's output, so oracle = 1.00 is definitional. The models tested never call tools. The paper therefore compares "tools, always" against "tools, never" and attributes the gap to grounding — but the scientifically interesting claim is that *a real LLM agent given the tools* closes the gap. That experiment is missing. **Required:** either (a) run a tool-using agent (the same Opus model with function-calling to the compute tools) and report whether it approaches 1.00 — this would be the paper's strongest result — or (b) reframe precisely: the oracle establishes an *attainable ceiling*; whether a real agent reaches it when given tools is stated as the immediate next experiment, and the abstract/§3.3/§4 wording is corrected so "tool-calling agent" is not read as an LLM result.

**M2 — Circularity of ground truth needs a stronger external anchor.**
Because truth = the platform's own tools, the oracle result proves internal consistency, not correctness. The manuscript concedes this, but the external-validity case rests entirely on one sentence about a Theophylline reference suite. Expand it: which parameters, what tolerance, against which independent estimates (NONMEM/nlmixr2/nlme consensus), and how close the tools land. Without this, a reader cannot judge whether "correct" means "correct" or "self-consistent."

**M3 — Statistical thinness; do not over-interpret per-category numbers.**
Two models, three runs, 30 tasks — six tasks per category, so a category score moves in ~0.17 increments and an SD from n = 3 is unstable. Report n per category explicitly. Either raise the run count or frame the real-model section as *preliminary* throughout (the abstract already says "initial"; carry that discipline into §3.3 and the Discussion). Avoid drawing conclusions from small per-category differences (e.g. dp 0.92 vs exposure 1.00).

**M4 — "Frontier models" but one model family.**
The title and abstract imply generality; only two Claude models are tested. Add at least one other family (a GPT, Gemini, or open-weights model) to support a general claim, or narrow the framing to "two models of differing capability."

**M5 — Justify the twofold exposure tolerance.**
With six tasks and a twofold band, exposure scores near ceiling for every agent and may not discriminate. State why twofold is the right criterion here (it is the human-PK-prediction field standard — cite Käser), and consider a tighter secondary tolerance to show the category still separates good from crude forward simulation.

## Minor comments

- **m1.** §3.4 (the parse-gap artifact) is a strength, but it invites the question "what else is mis-measured?" Add one sentence on a systematic safeguard — e.g. manual audit of a random sample of graded responses, confirming score integrity beyond the automated parse-gap count.
- **m2.** Ensure the abstract never lets the oracle 1.00 read as a performance result; §3.1 already frames it as a validity check — mirror that wording in the abstract.
- **m3.** Equal-category weighting is defensible but arbitrary; report the task-weighted overall alongside, for transparency.
- **m4.** Reproducibility: results are tied to specific API model versions that will change or retire. Pin the model version and evaluation date, and state that scores are not reproducible once a model is deprecated.
- **m5.** First-in-human category: infrastructure is described but produces no result; confirm the abstract and contributions do not imply one. (Currently handled correctly — keep it that way.)
- **m6.** Define GMFE and any BLQ/M3 shorthand at first use; the resource-article audience is broad.
- **m7.** Figure 1: consider overlaying the three individual run points per bar, not only mean ± SD, given n = 3.
- **m8.** Resolve the `[verify]` reference details (titles, author initials, PK-DB pages).

## What the paper does well (keep)

- The core contribution (deterministic, tool-grounded, contamination-resistant truth) is real and clearly argued.
- Limitations are prominent and honest; the disclosed measurement artifact builds rather than costs credibility.
- Reproducibility engineering (seed-determinism, regression gate on oracle = 1.00, pinned runs) is above the norm for this kind of paper.

## Single highest-value revision

Do **M1**: run the same model *with* tool access and report the score. If a tool-using Opus agent reaches ~1.00 where tool-free Opus scored 0.93, the paper's thesis moves from asserted to demonstrated, and the headline becomes a genuine result rather than a construction.
