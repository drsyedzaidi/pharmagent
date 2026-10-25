"""Deterministic quote-grounding validator for critic answers.

The critic LLM answers ONLY from retrieved chunks and must return strict JSON::

    {"verdict": "supported" | "contradicted" | "no_evidence",
     "rationale": "...",
     "citations": [{"chunk_id": "...", "quote": "..."}]}

Nothing about that request guarantees the model obeyed, so every citation is
checked here, in code, before the answer counts for anything:

* ``chunk_id`` must be one of the chunk ids that were actually retrieved;
* ``quote`` must be a verbatim substring of THAT chunk's text (whitespace runs
  are collapsed on both sides before matching — line breaks inside a PDF
  sentence must not fail a genuine quote — but no other normalisation);
* a quote must be at least ``MIN_QUOTE_CHARS`` long (a three-letter "quote" is
  always a substring of something) and at most ``MAX_QUOTE_CHARS`` (evidence is
  a short excerpt, never a reproduced passage).

A citation that violates any rule is dropped with its reason recorded. An
answer whose verdict claims evidence (supported / contradicted) but keeps zero
valid citations is downgraded to ``no_evidence`` with ``grounding: "failed"``.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

VERDICTS: tuple[str, ...] = ("supported", "contradicted", "no_evidence")
NO_EVIDENCE = "no_evidence"
MIN_QUOTE_CHARS = 20
MAX_QUOTE_CHARS = 600
MAX_RATIONALE_CHARS = 1500
MAX_CITATIONS = 12

#: grounding outcomes
GROUNDING_OK = "ok"                 # ≥1 citation, every offered citation validated
GROUNDING_PARTIAL = "partial"       # ≥1 valid citation, some dropped
GROUNDING_FAILED = "failed"         # evidence claimed but zero valid citations → downgraded
GROUNDING_ABSTAINED = "abstained"   # model said no_evidence and offered no citations
GROUNDING_UNPARSEABLE = "unparseable"  # the reply was not the required JSON → treated as no_evidence

_WS_RE = re.compile(r"\s+")
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def normalize_ws(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def quote_in_chunk(quote: str, chunk_text: str) -> bool:
    """Verbatim-substring test, tolerant only to whitespace layout."""
    q = normalize_ws(quote)
    return bool(q) and q in normalize_ws(chunk_text)


@dataclass(frozen=True)
class GroundedAnswer:
    verdict: str
    rationale: str
    citations: tuple[dict[str, str], ...]          # validated: {chunk_id, quote}
    dropped: tuple[dict[str, str], ...]            # {chunk_id, quote, reason}
    grounding: str
    raw_verdict: str | None = None                 # what the model said before any downgrade
    parse_error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"verdict": self.verdict, "rationale": self.rationale,
                "citations": list(self.citations), "dropped_citations": list(self.dropped),
                "grounding": self.grounding, "raw_verdict": self.raw_verdict,
                "parse_error": self.parse_error}


def parse_critic_json(text: str) -> dict[str, Any]:
    """The JSON object in a model reply. Accepts a bare object or one inside a ```json
    fence; otherwise takes the outermost ``{...}`` span. Raises ValueError."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("empty reply")
    candidates = [text.strip()]
    m = _FENCE_RE.search(text)
    if m:
        candidates.insert(0, m.group(1).strip())
    first, last = text.find("{"), text.rfind("}")
    if 0 <= first < last:
        candidates.append(text[first:last + 1])
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except ValueError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError("reply is not a JSON object")


def validate_citations(citations: Any, retrieved: dict[str, str]
                       ) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Split offered citations into (valid, dropped). ``retrieved`` maps chunk_id → text."""
    valid: list[dict[str, str]] = []
    dropped: list[dict[str, str]] = []
    if not isinstance(citations, list):
        return valid, [{"chunk_id": "", "quote": "", "reason": "citations is not a list"}] if citations else dropped
    seen: set[tuple[str, str]] = set()
    for item in citations[:MAX_CITATIONS]:
        if not isinstance(item, dict):
            dropped.append({"chunk_id": "", "quote": "", "reason": "citation is not an object"})
            continue
        cid = str(item.get("chunk_id") or "")
        quote = item.get("quote")
        quote = quote if isinstance(quote, str) else ""
        rec = {"chunk_id": cid, "quote": quote}
        if cid not in retrieved:
            dropped.append({**rec, "reason": "chunk_id not among retrieved chunks"})
        elif len(normalize_ws(quote)) < MIN_QUOTE_CHARS:
            dropped.append({**rec, "reason": f"quote shorter than {MIN_QUOTE_CHARS} characters"})
        elif len(quote) > MAX_QUOTE_CHARS:
            dropped.append({**rec, "reason": f"quote longer than {MAX_QUOTE_CHARS} characters"})
        elif not quote_in_chunk(quote, retrieved[cid]):
            dropped.append({**rec, "reason": "quote is not a verbatim substring of the cited chunk"})
        elif (cid, normalize_ws(quote)) in seen:
            dropped.append({**rec, "reason": "duplicate citation"})
        else:
            seen.add((cid, normalize_ws(quote)))
            valid.append(rec)
    return valid, dropped


def ground(reply: str, retrieved: dict[str, str]) -> GroundedAnswer:
    """Parse + validate a critic reply against the retrieved chunks. Never raises."""
    try:
        obj = parse_critic_json(reply)
    except ValueError as exc:
        return GroundedAnswer(NO_EVIDENCE, "", (), (), GROUNDING_UNPARSEABLE, None, str(exc))
    raw_verdict = str(obj.get("verdict") or "").strip().lower()
    rationale = obj.get("rationale")
    rationale = (rationale if isinstance(rationale, str) else "")[:MAX_RATIONALE_CHARS]
    valid, dropped = validate_citations(obj.get("citations"), retrieved)
    if raw_verdict not in VERDICTS:
        dropped = [*dropped, {"chunk_id": "", "quote": "", "reason": f"unknown verdict {raw_verdict!r}"}]
        return GroundedAnswer(NO_EVIDENCE, rationale, (), tuple(dropped), GROUNDING_FAILED, raw_verdict or None)
    if valid:
        grounding = GROUNDING_OK if not dropped else GROUNDING_PARTIAL
        return GroundedAnswer(raw_verdict, rationale, tuple(valid), tuple(dropped), grounding, raw_verdict)
    if raw_verdict == NO_EVIDENCE and not dropped:
        return GroundedAnswer(NO_EVIDENCE, rationale, (), (), GROUNDING_ABSTAINED, raw_verdict)
    return GroundedAnswer(NO_EVIDENCE, rationale, (), tuple(dropped), GROUNDING_FAILED, raw_verdict)
