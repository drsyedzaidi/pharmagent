"""Reviewer Agent tool: consult mikebrain, the literature-grounded critic.

``consult_mikebrain`` retrieves the most relevant chunks of the owner's corpus
(``app.knowledge``), asks the configured LLM — the LOCAL provider by default —
to answer ONLY from those chunks in a strict JSON schema, and then validates the
answer in code: every cited ``chunk_id`` must be one that was retrieved and every
``quote`` must be a verbatim substring of that chunk. Citations that fail are
dropped; an answer left with no valid citation is downgraded to ``no_evidence``.

mikebrain refutes or corroborates. It never decides: the result is written to
``PharmState.knowledge_results`` as advice, it carries no severity, and nothing
here can change ``review_results`` counts, ``goal_met`` or any gate. The
deterministic refutations in ``app.compute.adversarial`` and the human-review
gates remain the hard floor. See docs/mikebrain.md.
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from app.agents.base import Agent
from app.core.llm import LLM, MockLLM, OpenAICompatLLM, RealLLM, get_llm, is_loopback
from app.core.pharmstate import PharmState
from app.knowledge.grounding import GROUNDING_FAILED, NO_EVIDENCE, ground
from app.knowledge.index import DEFAULT_K, MAX_K, Embedder, SearchResult, default_embedder, load_index
from app.knowledge.manifest import ManifestError, load_manifest
from app.tools.base import Tool, ToolContext, ToolResult

log = logging.getLogger("pharmagent")

KNOWLEDGE_SUBDIR = "knowledge"
KNOWLEDGE_MAX = 20              # bounded knowledge_results history, newest last
MAX_QUESTION_CHARS = 2000
MAX_CONTEXT_ITEMS = 24          # caller-supplied context: scalar leaves only, bounded
MAX_QUERY_CHARS = 800           # retrieval query = question + compact state terms
MAX_ADVISORY_QUERIES = 3        # findings consulted per adversarial_review pass
LLM_MAX_TOKENS = 1024
ADVISORY_KIND = "literature_advisory"
#: the compact, privacy-safe parts of Agent._state_summary that mikebrain sees
SUMMARY_KEYS: tuple[str, ...] = ("nca_diagnostics", "pk_model_diagnostics", "nlme_diagnostics",
                                 "review_diagnostics", "pending_tool")

SYSTEM_PROMPT = (
    "You are mikebrain, a literature critic inside a pharmacometrics platform. You are given a "
    "question, a compact summary of the analysis state, and numbered excerpts from reference "
    "documents. Answer ONLY from the excerpts. Do not use outside knowledge. If the excerpts do not "
    "settle the question, say so.\n"
    "Reply with ONE JSON object and nothing else:\n"
    '{"verdict": "supported" | "contradicted" | "no_evidence", '
    '"rationale": "<= 3 sentences", '
    '"citations": [{"chunk_id": "<id of an excerpt>", "quote": "<a verbatim sentence copied exactly '
    'from that excerpt>"}]}\n'
    "Rules: 'supported' means the excerpts support the claim in the question; 'contradicted' means "
    "they contradict it; otherwise 'no_evidence'. Every citation must quote the excerpt word for word; "
    "a paraphrase is not a citation. Give no citation rather than an inexact one. You advise; you do "
    "not decide."
)


class _SilentAgent(Agent):
    """Only here to reuse Agent._state_summary (the privacy-safe view of state)."""


def knowledge_root(ctx: ToolContext) -> Path:
    return Path(ctx.data_dir) / KNOWLEDGE_SUBDIR


def compact_state_summary(state: PharmState, context: dict[str, Any] | None = None) -> dict[str, Any]:
    """The diagnostic summaries mikebrain may see — aggregates only, never raw rows —
    plus a bounded, scalar-only view of any caller-supplied ``context``."""
    full = _SilentAgent(name="reviewer", system_prompt="")._state_summary(state)
    out: dict[str, Any] = {k: full[k] for k in SUMMARY_KEYS if full.get(k) is not None}
    rev = out.get("review_diagnostics")
    if isinstance(rev, dict) and rev.get("blocking_ids"):
        # finding ids such as recompute-cmax-1771 end in a subject id: keep the kind, drop the subject
        out["review_diagnostics"] = {**rev, "blocking_ids": sorted({finding_kind(str(i)) for i in rev["blocking_ids"]})}
    if isinstance(context, dict) and context:
        extra = {str(k)[:60]: (v[:200] if isinstance(v, str) else v)
                 for k, v in list(context.items())[:MAX_CONTEXT_ITEMS]
                 if isinstance(v, (str, int, float, bool))}
        if extra:
            out["context"] = extra
    return out


def _terms(obj: Any, prefix: str = "") -> list[str]:
    if isinstance(obj, dict):
        return [t for k, v in obj.items() for t in _terms(v, str(k))]
    if isinstance(obj, (list, tuple)):
        return [t for v in obj for t in _terms(v, prefix)]
    if obj is None:
        return []
    return [f"{prefix} {obj}".strip()]


def build_query(question: str, summary: dict[str, Any]) -> str:
    q = " ".join(question.split())
    extra = " ".join(_terms(summary))
    return (q + (" " + extra if extra else ""))[:MAX_QUERY_CHARS]


def _render_excerpts(res: SearchResult) -> str:
    return "\n\n".join(f"[{h.chunk_id}] ({h.title}{'; ' + h.section if h.section else ''})\n{h.text}"
                       for h in res.hits)


def _llm_identity(llm: LLM) -> tuple[str, str]:
    """(provider, model) from the client object itself — never from ambient settings."""
    if isinstance(llm, MockLLM):
        return "mock", "mock"
    if isinstance(llm, RealLLM):
        return "anthropic", llm.model
    if isinstance(llm, OpenAICompatLLM):
        return ("local" if is_loopback(llm.base_url) else "openai"), llm.model
    return type(llm).__name__, str(getattr(llm, "model", "unknown"))


def consult(question: str, state: PharmState, root: Path, *, context: dict[str, Any] | None = None,
            k: int = DEFAULT_K, llm: LLM | None = None, embedder: Embedder | None = None) -> dict[str, Any]:
    """Retrieve → ask → ground. Returns a plain dict and never raises for corpus or
    LLM failures (they are recorded in ``status``); bad arguments raise ValueError."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("consult_mikebrain needs a non-empty 'question'")
    if context is not None and not isinstance(context, dict):
        raise ValueError("'context' must be an object when given")
    question = question.strip()[:MAX_QUESTION_CHARS]
    k = max(1, min(int(k), MAX_K))
    t0 = time.perf_counter()
    summary = compact_state_summary(state, context)
    base: dict[str, Any] = {
        "question": question, "k": k, "state_summary": summary,
        "verdict": NO_EVIDENCE, "rationale": "", "citations": [], "dropped_citations": [],
        "grounding": GROUNDING_FAILED, "raw_verdict": None, "parse_error": None,
        "corpus_version": None, "n_chunks": 0, "embedding": "unavailable",
        "retrieved": [], "chunk_hashes": [], "provider": None, "model": None,
        "elapsed_s": 0.0, "error": None,
    }

    def done(**upd: Any) -> dict[str, Any]:
        return {**base, **upd, "elapsed_s": round(time.perf_counter() - t0, 3)}

    try:
        manifest = load_manifest(root)
    except ManifestError as exc:
        return done(status="corpus_error", error=str(exc)[:300])
    if manifest.is_empty:
        return done(status="no_corpus", corpus_version=manifest.corpus_version,
                    rationale="no documents in the mikebrain corpus")
    try:
        index = load_index(root)
    except ManifestError as exc:
        return done(status="corpus_error", corpus_version=manifest.corpus_version, error=str(exc)[:300])

    res = index.search(build_query(question, summary), k=k, embedder=embedder)
    retrieved = {h.chunk_id: h.text for h in res.hits}
    evidence = {
        "corpus_version": res.corpus_version, "n_chunks": res.n_chunks, "embedding": res.embedding,
        "retrieved": [h.to_dict(with_text=False) for h in res.hits],
        "chunk_hashes": [h.chunk_hash for h in res.hits],
    }
    if res.embedding_error:
        evidence["embedding_error"] = res.embedding_error
    if not res.hits:
        return done(status="ok", grounding="abstained", rationale="no chunk matched the question", **evidence)

    llm = llm or get_llm()
    provider, model = _llm_identity(llm)
    user = (f"Question: {question}\n\nAnalysis state (aggregates only): {json.dumps(summary, default=str)}\n\n"
            f"Excerpts:\n\n{_render_excerpts(res)}")
    try:
        reply = llm.complete(SYSTEM_PROMPT, user, max_tokens=LLM_MAX_TOKENS)
    except Exception as exc:  # noqa: BLE001 - advisory path must never take the review down
        log.warning("mikebrain_llm_failed", extra={"provider": provider, "model": model, "error": str(exc)[:200]})
        return done(status="llm_error", provider=provider, model=model, error=str(exc)[:300], **evidence)
    answer = ground(reply, retrieved)
    return done(status="ok", provider=provider, model=model, **answer.to_dict(), **evidence)


def consult_mikebrain(state: PharmState, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    rec = consult(args.get("question"), state, knowledge_root(ctx), context=args.get("context"),
                  k=args.get("k") or DEFAULT_K, embedder=default_embedder())
    history = [*(state.knowledge_results or []), rec][-KNOWLEDGE_MAX:]
    summary = (f"mikebrain ({rec['status']}): {rec['verdict']} [grounding {rec['grounding']}; "
               f"{len(rec['citations'])} verbatim citation(s), {len(rec['dropped_citations'])} dropped] — "
               f"advisory only, nothing decided. {rec['rationale'][:200]}")
    if rec.get("error"):
        summary += f" Error: {rec['error']}"
    return ToolResult(
        summary=summary,
        action=f"consult_mikebrain(question={rec['question'][:80]!r}) -> {rec['verdict']} [{rec['grounding']}]",
        writes={"knowledge_results": history},
        result=rec,
    )


# ── adversarial_review integration (advisory only) ──────────────────────────────
_SUBJECT_RE = re.compile(r"\bsubjects?\s+[\w.-]+", re.IGNORECASE)
_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?", re.IGNORECASE)
MAX_REDACTED_CHARS = 400


def redact(text: Any) -> str:
    """Finding text with subject identifiers and numeric values removed: a literature
    question needs the *kind* of refutation, never an individual's value."""
    t = _SUBJECT_RE.sub("a subject", str(text or ""))
    t = _NUMBER_RE.sub("N", t)
    return " ".join(t.split())[:MAX_REDACTED_CHARS]


# Per-subject finding ids end in the subject id (nca-clf-sign-1771, recompute-cmax-S07). The last segment must
# contain a digit, so ordinary words such as "NCA-derived" or "recompute-from-raw-data" are left alone.
_SUBJECT_FINDING_ID = r"(?:nca|recompute)-(?:[a-z]+-)*[a-z]*\d[\w.]*"
_FINDING_ID_RE = re.compile(rf"\b{_SUBJECT_FINDING_ID}(?![\w-])", re.IGNORECASE)   # whole id only, never a prefix
_FINDING_ID_FULL_RE = re.compile(rf"^{_SUBJECT_FINDING_ID}$", re.IGNORECASE)
# "subject 12", "Subject: 12", "subject #12", "subject ID 12", "ID=1771", "patient 3", "subjects 1, 2 and 3"
_ID_WORDS = r"(?:subjects?|patients?|participants?|volunteers?|ids?)"
_ID_TOKEN = r"[\w.-]*\d[\w.-]*"
_SUBJECT_REF_RE = re.compile(
    rf"\b{_ID_WORDS}\b(?:\s+id\b)?\s*[:#=(\[]*\s*{_ID_TOKEN}(?:\s*(?:,|and|&)\s*{_ID_TOKEN})*[)\]]?",
    re.IGNORECASE)


def finding_kind(finding_id: str) -> str:
    """``nca-clf-sign-1771`` -> ``nca-clf-sign``; any other finding id is returned unchanged."""
    return finding_id.rsplit("-", 1)[0] if _FINDING_ID_FULL_RE.match(finding_id) else finding_id


def redact_subjects(text: Any) -> str:
    """``text`` with subject references removed ("subject 12", "ID=1771", "patients 1, 2") and per-subject
    finding ids reduced to their kind, whitespace folded. Numbers are kept: aggregate diagnostics are what a
    literature question is about. Best effort: an identifier with no digit in it is not recognised."""
    t = _SUBJECT_REF_RE.sub("a subject", str(text or ""))
    t = _FINDING_ID_RE.sub(lambda m: finding_kind(m.group(0)), t)
    return " ".join(t.split())


def _advisory_question(f: dict[str, Any]) -> str:
    return (f"Does the literature support this refutation of a {redact(f.get('target') or 'result')}? "
            f"The analysis claims: {redact(f.get('claim'))}. The independent evidence against it: "
            f"{redact(f.get('evidence'))}.")


def review_advisories(review: dict[str, Any], state: PharmState, ctx: ToolContext, *,
                      llm: LLM | None = None, embedder: Embedder | None = None) -> dict[str, Any]:
    """Literature advisories for the first ``MAX_ADVISORY_QUERIES`` unresolved findings
    of an adversarial review. Severity-free; never touches findings, counts or goal_met."""
    root = knowledge_root(ctx)
    try:
        if load_manifest(root).is_empty:
            return {"advisories": [], "advisory_status": "no_corpus"}
    except ManifestError as exc:
        return {"advisories": [], "advisory_status": f"corpus_error: {str(exc)[:120]}"}
    pending = [f for f in (review.get("findings") or []) if not f.get("resolved")][:MAX_ADVISORY_QUERIES]
    if not pending:
        return {"advisories": [], "advisory_status": "no_findings"}
    embedder = embedder if embedder is not None else default_embedder()
    llm = llm or get_llm()
    advisories = []
    for f in pending:
        rec = consult(_advisory_question(f), state, root, llm=llm, embedder=embedder)
        advisories.append({"kind": ADVISORY_KIND, "finding_id": f.get("id"),
                           **{key: rec[key] for key in ("question", "status", "verdict", "rationale", "citations",
                                                        "dropped_citations", "grounding", "corpus_version",
                                                        "chunk_hashes", "embedding", "provider", "model",
                                                        "elapsed_s", "error")}})
    return {"advisories": advisories, "advisory_status": "ok"}


TOOLS = [
    Tool(
        "consult_mikebrain",
        "Consult mikebrain, the literature-grounded critic: retrieve the most relevant passages of "
        "the owner's pharmacometrics corpus for a question about the current analysis and return a "
        "verdict (supported / contradicted / no_evidence) with verbatim citations validated in code. "
        "Advisory only — it refutes or corroborates, it never decides and cannot change a gate. "
        "Args: question (required string), context (optional object of scalars), k (optional, 1-12).",
        "reviewer",
        {"type": "object",
         "properties": {"question": {"type": "string"},
                        "context": {"type": "object"},
                        "k": {"type": "integer", "minimum": 1, "maximum": MAX_K}},
         "required": ["question"]},
        consult_mikebrain,
    ),
]
