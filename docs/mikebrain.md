# mikebrain — literature-grounded advisory critic

**Status:** M0 (corpus index) and M1 (grounded critic tool) — advisory only.
Nothing mikebrain says can block, pass or alter a gate. The deterministic
refutations in `backend/app/compute/adversarial.py` and the human-review gates
remain the hard floor.

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
- **Not a rule engine yet.** Until an owner-approved rule catalog exists (M2),
  every result is advisory and carries no authority, the same posture as the
  plausibility catalog's `pending_owner_review` status.

## Default LLM

The critic uses `get_llm()`, i.e. whichever provider the app is configured for.
With no key set that is the **local** OpenAI-compatible provider (Ollama,
`gemma4:31b` by default); Claude is used only when an Anthropic key is
configured; the keyless `MockLLM` always abstains with `no_evidence`, which is
what the test suite exercises. One new method, `LLM.complete(system, user)`,
carries the plain-text (no tool) call.

Embeddings for the vector side of retrieval come from Ollama's
`nomic-embed-text` on this machine only. When Ollama is not reachable (CI,
tests, a laptop without it) retrieval runs **BM25-only** and every result says
so: `embedding: "unavailable"`.

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
severity**. `advisory_status` is `ok`, `no_corpus` or `no_findings`. The flag
defaults to false; the review loop and the `/review` endpoint never set it.

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

## Copyright

The repository ships **no documents and no copyrighted text**; the three test
fixtures are short synthetic notes written for the suite. The corpus is the
owner's private library for private retrieval on the owner's machine. Reports
and state never reproduce passages: results store chunk hashes and offsets, and
citations are short verbatim excerpts (≤ 600 characters) kept as evidence of
grounding, not as content.

## Roadmap

- **M0 — corpus index** (this PR): manifest, chunking, BM25 + optional vectors, CLI.
- **M1 — grounded critic tool** (this PR): `consult_mikebrain`, quote-grounding
  validator, `knowledge_results`, advisory hook in `adversarial_review`.
- **M2 — owner-approved rule catalog.** Rules ("EBE covariate screening is
  uninformative above 30 % shrinkage") stored with their supporting chunk hashes
  and an owner signature; a rule is `pending_owner_review` until approved, and
  only approved rules may be promoted from advisory to a deterministic finding
  in `adversarial.py` — never the LLM's free-text verdict.
- **M3 — planner gate.** mikebrain becomes the Phase B critic of
  `docs/agentic_planner_design.md`: each planner proposal is consulted and the
  advisory shown at the human gate. Whether a `contradicted` verdict may block on
  its own is an owner decision deferred until M2 rules exist.

Out of scope so far: per-role LLM routing, planner integration, UI.
