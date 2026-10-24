# mikebrain — literature-grounded advisory critic

**Status:** M0 (corpus index), M1 (grounded critic tool) and M2 (owner-approved
rule catalog). The critic is advisory only. Nothing mikebrain says can block,
pass or alter a gate, and a rule-derived finding is capped at MEDIUM so it
cannot either. The deterministic refutations in
`backend/app/compute/adversarial.py` and the human-review gates remain the hard
floor.

## What it is

mikebrain is a private retrieval index over pharmacometrics books, papers and
guidelines that the owner supplies, plus one tool that consults it at a decision
point. Given a question about the current analysis ("is covariate screening on
EBEs valid at 45 % eta-shrinkage?") it retrieves the most relevant passages,
asks the configured LLM to answer **only from those passages**, and then checks
the answer in code before it counts for anything.

mikebrain **refutes or corroborates**. Its verdicts are `supported`,
`contradicted` or `no_evidence`, each with verbatim citations into the corpus.

## What it is not

- **Not a decider.** It writes advice to state; it never writes a finding
  severity, never changes `review_results.counts`, `goal_met` or `status`, and
  no gate reads it.
- **Not a general chat.** It answers only from retrieved chunks. With no
  corpus, or no matching chunk, it says so (`status: no_corpus`, `grounding:
  abstained`).
- **Not a rule author.** The critic's verdicts are advisory and carry no
  authority. A verdict can never become a rule: rules are written by the owner
  through a separate command, enter as `pending_owner_review`, and only the
  owner's reviewed edit makes one `approved` (see "The rule catalog (M2)").

## Default LLM

The critic uses `get_llm()`, i.e. whichever provider the app is configured for.
With no key set that is the **local** OpenAI-compatible provider (Ollama,
`gemma4:31b` by default); Claude is used only when an Anthropic key is
configured; the keyless `MockLLM` always abstains with `no_evidence`, which is
what the test suite exercises. One new method, `LLM.complete(system, user)`,
carries the plain-text (no tool) call.

**Latency (measured 2026-10-23, Apple-silicon Mac, `gemma4:31b` via Ollama):**
75–116 s per consult once the model is warm; the first call after idle, which
also loads the model, exceeded two minutes. The critic call therefore has its
own 300 s HTTP budget (`OpenAICompatLLM.COMPLETE_TIMEOUT_S`); a timeout is
recorded as `status: "llm_error"` and never raised. `adversarial_review` with
`consult_mikebrain=true` makes up to three such calls synchronously, so it is a
deliberate, opt-in choice, never the default of the review loop. A smaller local
model (`qwen2.5:7b`) answers in seconds.

Embeddings for the vector side of retrieval come from Ollama's
`nomic-embed-text` on this machine only. When Ollama is not reachable (CI,
tests, a laptop without it) retrieval runs **BM25-only** and every result says
so: `embedding: "unavailable"`.

**Where corpus text travels.** Retrieval and embeddings never leave the
machine. The critic call sends the retrieved excerpts (about six chunks of
≈500 tokens) to whichever LLM is configured: local Ollama by default, but if
the owner switches the app to Claude or ChatGPT in the UI, those excerpts go to
that hosted API. Choosing a provider is therefore also a choice about where
excerpts of the owner's library may be sent. Each record states `provider` and
`model`, derived from the client object, so the audit shows which it was.

## Data flow

```
owner: python -m app.knowledge add <book.pdf|paper.txt> --title ... --type book|paper|guideline
   │  extract text (pypdf / utf-8)   →  <data_dir>/knowledge/text/<sha256>.txt
   │  manifest.json: {path, sha256, title, year, source_type, added_by, added_at}
   │  corpus_version = sha256(sorted document sha256s)
   ▼
consult_mikebrain(question, context?)           [reviewer tool, not expensive]
   1. state summary  ← Agent._state_summary subset: nca/pk/nlme/review diagnostics,
                        pending proposal; caller context limited to scalars; never raw rows
   2. retrieval query = question + compact summary terms
   3. hybrid search   = BM25 ∪ vector (Ollama nomic-embed-text when reachable), RRF-fused, top-k (6)
   4. LLM.complete    = system prompt "answer ONLY from the excerpts, strict JSON"
   5. grounding       = deterministic validator (below)
   6. PharmState.knowledge_results += record   (bounded list of 20, newest last; owner "reviewer")
   7. audit entry     = registry.execute → outputs_hash over the full record
```

`adversarial_review(consult_mikebrain=true)` runs the same pipeline once per
unresolved finding (first three, severity order) and stores the answers under
`review_results["advisories"]` with `kind: "literature_advisory"` and **no
severity**. `advisory_status` is `ok`, `no_corpus`, `no_findings` or
`error: …` (an advisory crash never fails the review). The question put to the
literature is built from the finding with subject identifiers and numeric
values redacted ("CL/F = N is non-physiological"), so no individual value
leaves the state summary's aggregate posture. The flag defaults to false; the
review loop and the `/review` endpoint never set it, and it is honoured only
when a caller passes it explicitly — it is **not** listed in the tool schema the
chat LLM sees, because three synchronous local-LLM calls must not be something
an agent switches on by itself.

## The quote-grounding rule

The LLM must return

```json
{"verdict": "supported" | "contradicted" | "no_evidence",
 "rationale": "...",
 "citations": [{"chunk_id": "<retrieved id>", "quote": "<verbatim sentence>"}]}
```

`backend/app/knowledge/grounding.py` then enforces, deterministically:

| Rule | On violation |
|------|--------------|
| `chunk_id` must be one of the ids retrieved for this question | citation dropped (`chunk_id not among retrieved chunks`) |
| `quote` must be a verbatim substring of that chunk (only whitespace runs are normalised) | citation dropped (`quote is not a verbatim substring …`) |
| `quote` is 20–600 characters | dropped (`shorter than` / `longer than`) |
| no duplicate citations, objects only | dropped |
| zero valid citations while claiming `supported`/`contradicted` | verdict → `no_evidence`, `grounding: "failed"` |
| reply is not JSON | verdict `no_evidence`, `grounding: "unparseable"` |
| `no_evidence` with no citations offered | `grounding: "abstained"` (honest abstention, the MockLLM path) |
| all offered citations valid / some dropped | `grounding: "ok"` / `"partial"` |

Dropped citations are kept in the record with their reason so a reviewer can see
what the model tried to claim.

## Audit fields

Every consultation record (in `knowledge_results` and in the tool's audit entry
via `outputs_hash`) carries:

- `corpus_version`, `n_chunks`, `embedding` (`ollama:nomic-embed-text` or `unavailable`)
- `retrieved`: per chunk `chunk_id`, `doc_sha`, `chunk_hash` (sha256 of the chunk
  text), `title`, `section`, `offsets [start, end)`, `score`, ranks — **not the text**
- `chunk_hashes`: the evidence set, re-derivable from the stored text and offsets
- `provider`, `model`, `elapsed_s`, `status` (`ok` / `no_corpus` / `corpus_error` / `llm_error`), `error`
- `verdict`, `raw_verdict`, `rationale`, `citations`, `dropped_citations`, `grounding`, `parse_error`
- `state_summary`: exactly what the model was shown about the analysis

Chunking is deterministic (section-aware word windows of ≈500 tokens with
overlap, exact character offsets), so the same corpus version always yields the
same chunk ids and hashes.

## Adding documents (owner)

From `backend/` with the app environment:

```bash
python -m app.knowledge add ~/library/gabrielsson_weiner.pdf --title "Pharmacokinetic and Pharmacodynamic Data Analysis" --year 2016 --type book
python -m app.knowledge add ~/library/savic_karlsson_2009_shrinkage.pdf --title "Importance of shrinkage in EBEs" --year 2009 --type paper
python -m app.knowledge list
python -m app.knowledge search "eta shrinkage EBE diagnostics" -k 6
python -m app.knowledge remove <sha256 prefix>
```

The corpus root is `<PHARMAGENT_DATA_DIR>/knowledge/` (git-ignored). PDFs go
through pypdf only; an image-only PDF yields too little text and is refused
rather than indexed as empty (OCR / MinerU are deliberately not dependencies).
Adding or removing any document changes `corpus_version`, so earlier results
remain attributable to the corpus they were answered from.

`add` also warms the embedding cache when Ollama answers (`--no-embed` to
skip), so the first consult does not embed a whole book inside a request. The
cache under `<root>/embeddings/` is keyed by embedder name and chunk hash; after
updating the embedding model under the same name, delete that directory to
re-embed (a dimension mismatch is detected and falls back to BM25 with an
`embedding_error` until then). The in-process index is cached and re-read
whenever `corpus_version` changes.

## The rule catalog (M2)

M1 can say "the literature disagrees". M2 lets the owner turn a passage they
have read into a **deterministic check**, so the same finding appears on every
review without an LLM in the loop.

**What a rule is.** A closed, typed predicate over analysis diagnostics, plus a
claim, a rationale, a suggested action, a severity, and pointers to the passages
that support it. Example (the owner's own words; the numbers come from the
owner's reading):

```json
{"id": "mb.nlme.high_shrinkage_ebe_screening", "version": "1.0.0",
 "when": [{"path": "nlme_results.shrinkage_pct", "quantifier": "any", "op": "gt", "value": 30}],
 "severity": "MEDIUM", "target": "NLME covariate screening on EBEs",
 "claim": "...", "rationale": "...", "suggested_action": "...",
 "evidence": [{"doc_sha": "...", "document": "<title>", "year": 2009, "source_type": "paper",
               "section": "...", "start": 1364, "end": 2065, "text_sha256": "..."}],
 "proposed_by": "...", "proposed_on": "2026-10-24",
 "approval": {"status": "pending_owner_review"}}
```

**The vocabulary is closed.** `when` has one to four conditions, all of which
must hold. Each names one of a short allow-list of diagnostic paths
(`nlme_results.converged`, `.condition_number`, `.shrinkage_pct`,
`.theta_rse_pct`, `.omega_cv_pct`, `pk_model_results.mode`), one of six
operators (`gt ge lt le eq ne`) and a value. Per-parameter maps need an `any` or
`all` quantifier. There is no expression string, no `eval`, and no field a model
could use to smuggle logic in. The results block must report `status: ok`,
the same gate the built-in refutations use. A missing, `None`, boolean or
non-numeric value never satisfies a condition, so absent data never produces a
finding.

**Approval is the owner's act, not a tool's.**

| Property | How it is enforced |
|---|---|
| A rule enters as `pending_owner_review` with no approver | `python -m app.knowledge rules add` is the only writer, and it refuses to write anything else |
| A pending rule has no effect | only `approved` rules are evaluated; with none approved a review result (and its summary) equals that of an empty catalog (tested) |
| Approval names who and when | schema: `approval.by` and `approval.date` required when approved, forbidden when pending |
| An automated identity cannot approve | schema rejects `claude`, `llm`, `bot`, `agent`, `mikebrain`, `loop` and similar tokens as approver. This is a speed bump; the control is that approval is a reviewed pull request |
| No LLM path to the catalog | `knowledge_tools.py` cannot reach the writer (tested); predicates come from `--when` flags, never from model text |
| A rule can never block | schema allows only `MEDIUM`/`LOW`, and `adversarial.review` refuses any extra finding that is not, so `goal_met` cannot change |

"Approved" here means the owner reviewed the rule and its evidence in the pull
request that records the approval. It is **not** a cryptographic e-signature:
the e-signature and audit code is guarded and untouched. What binds a result to
the exact approved file is the catalog identity every finding carries:
`catalog_version` and `catalog_sha256` (SHA-256 of the parsed file as canonical
JSON, the same recipe as the conventions catalog, so reformatting does not
change it and any content change does).

**Evidence is bound by hash and holds no passage text.** `rules add --chunk`
takes chunk ids from `search` and records the document hash, bibliographic
title and year, the character span, and the SHA-256 of that span. Nothing from
the passage is copied, so the catalog is safe to commit. `rules verify`
re-derives each span from the stored text and reports `present`,
`text_changed`, `span_out_of_range`, `missing_text`, `missing_document` or
`corpus_empty`, and exits 1 on any problem. Review results deliberately do
**not** depend on the corpus being present: evaluation is a pure function of
the analysis state and the catalog, so it reproduces on CI and on a machine
without the owner's library.

**What a fired rule looks like.** An ordinary finding (`id` = the rule id,
`severity`, `target`, `claim`, `evidence` text with the observed values,
`suggested_action`) plus `origin: "mikebrain_rule"`, `rule_id`, `rule_version`,
`catalog_version`, `catalog_sha256`, `approved_by`, `approved_on` and
`literature` (title, year, document and span hashes). Findings are counted in
`counts` like any MEDIUM or LOW finding and never in a way that changes
`goal_met`. Once at least one rule is approved the result also carries a
`rule_catalog` block (`n_approved`, `n_pending_ignored`, `n_fired`, `fired`). If
the catalog file fails to load, the review still runs: rules are skipped and the
block says `status: "invalid"` with the error, rather than failing the review or
passing silently.

**Owner workflow.**

```bash
python -m app.knowledge search "eta shrinkage EBE covariate screening"        # find the chunk ids
python -m app.knowledge rules add --id mb.nlme.high_shrinkage_ebe_screening \
    --title "High eta-shrinkage makes EBE screening uninformative" --severity MEDIUM \
    --when "nlme_results.shrinkage_pct any gt 30" \
    --target "NLME covariate screening on EBEs" --claim "..." --rationale "..." --action "..." \
    --chunk 7cefffafb03f:2 --limit "30 % is a convention, not a law"
python -m app.knowledge rules list
python -m app.knowledge rules verify
```

Then read the cited passage, edit `approval` in
`backend/app/knowledge/rules/catalog.json` to
`{"status": "approved", "by": "<you>", "date": "YYYY-MM-DD"}`, and open the
change for review. The server reads the catalog once per process, so restart it
afterwards (never with `--reload`). The catalog ships **empty**: it contains no
rule, so nothing changes until the owner approves one.

**Deliberately not done in M2.** Raising a rule above MEDIUM (an owner decision;
see open question 3 in `docs/agentic_planner_design.md`), a cryptographic
signature on approval, rules over raw data or over paths outside the allow-list,
and any use of rules by the planner (M3).

## Copyright

The repository ships **no documents and no copyrighted text**; the three test
fixtures are short synthetic notes written for the suite. The corpus is the
owner's private library; retrieval happens on the owner's machine and excerpts
reach a hosted LLM only if the owner configures one (see "Where corpus text
travels"). Reports and state never reproduce passages: results store chunk
hashes and offsets, and citations are short verbatim excerpts (≤ 600 characters)
kept as evidence of grounding, not as content.

## Roadmap

- **M0 — corpus index** (#28): manifest, chunking, BM25 + optional vectors, CLI.
- **M1 — grounded critic tool** (#28): `consult_mikebrain`, quote-grounding
  validator, `knowledge_results`, advisory hook in `adversarial_review`.
- **M2 — owner-approved rule catalog** (stacked PR): typed rule schema with a closed
  predicate vocabulary, `pending_owner_review` by default, evidence bound by hash,
  deterministic evaluation inside `adversarial_review` capped at MEDIUM. The
  catalog ships empty.
- **M3 — planner gate.** mikebrain becomes the Phase B critic of
  `docs/agentic_planner_design.md`: each planner proposal is consulted and the
  advisory shown at the human gate. Whether a `contradicted` verdict may block on
  its own is still an owner decision; M2 rules are the only route by which
  literature can become a deterministic finding.

Out of scope so far: per-role LLM routing, planner integration, UI.
