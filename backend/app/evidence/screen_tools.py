"""Evidence agent tools: screen_publications (PROPOSAL per source) and decide_screening (the human's).

Screening per source: with a real LLM, the title + first two pages are wrapped as an untrusted
document and the model's answer must satisfy SCREEN_SCHEMA; every quote it cites must be found
literally on the cited page by the deterministic verifier, else that criterion is marked unmet
('quote not found') and a required criterion downgrades the verdict to 'unsure'. An invalid or
failed answer falls back to keyword screening with note 'llm_invalid'. Keyless runs use the
keyword screen directly. Rows are INSERT-OR-IGNORE on (source, criteria_hash): re-screening
with the same criteria returns the existing rows; nothing here ever changes a decided row.
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import asdict
from typing import Any

from app.core.pharmstate import PharmState
from app.evidence import ingest
from app.evidence.config_tools import summarise
from app.evidence.llm_access import current_llm, llm_available
from app.evidence.models import DECISIONS, Screen, Source, now_iso
from app.evidence.prompts import EVIDENCE_SYSTEM, SCREEN_SCHEMA, validate, wrap_document
from app.evidence.screen_rules import keyword_screen
from app.evidence.store import EvidenceStore, get_store
from app.evidence.verify import find_snippet_in_page
from app.tools.base import Tool, ToolContext, ToolResult

log = logging.getLogger(__name__)

AGENT = "evidence"
MAX_DOC_CHARS, PAGES_FOR_LLM, MAX_REASON = 6000, 2, 500
NOTE_LLM_INVALID, NOTE_QUOTE_NOT_FOUND, NOTE_NO_QUOTE, NOTE_NOT_ASSESSED = (
    "llm_invalid", "quote not found", "no quote", "not assessed")
_DECISION_STATUS = {"approve": "approved", "reject": "rejected"}
_TASK = (
    "Screen the document above against each criterion below. For every criterion return "
    "criterion_id, met (true / false / null when the document does not say), and when met is "
    "true a short exact quote from the document (verbatim, <= 300 characters) with the 1-based "
    "page it appears on. Then give verdict (include / exclude / unsure), score (0..1 confidence "
    "that the document should be included) and a brief reasoning.\n\nCriteria:\n{criteria}"
)


# ----------------------------------------------------------------- LLM path ---

def _criteria_block(criteria: list[dict[str, Any]]) -> str:
    return "\n".join(f"{c['id']} ({c['kind']}{', required' if c.get('required') else ''}): {c['text']}"
                     for c in criteria)


def _locate_quote(quote: str, page: dict[str, Any]) -> dict[str, Any] | None:
    span = find_snippet_in_page(quote, page.get("text") or "")
    if span is not None:
        return {"page": page["n"], "snippet": quote, "span": span, "column": None}
    for ci, col in enumerate(page.get("columns") or []):
        span = find_snippet_in_page(quote, col or "")
        if span is not None:
            return {"page": page["n"], "snippet": quote, "span": span, "column": ci}
    return None


def _check_entry(crit: dict[str, Any], entry: dict[str, Any] | None, pages: list[dict[str, Any]]) -> dict:
    """Normalise one LLM criterion answer against the config and the parsed pages."""
    base = {"criterion_id": crit["id"], "kind": crit["kind"], "required": bool(crit.get("required")),
            "met": None, "evidence": [], "note": ""}
    if entry is None:
        return {**base, "note": NOTE_NOT_ASSESSED}
    met, quote, page_n = entry.get("met"), entry.get("quote"), entry.get("page")
    if not quote:
        return {**base, "met": None if met else met, "note": NOTE_NO_QUOTE if met else ""}
    page = pages[page_n - 1] if isinstance(page_n, int) and 1 <= page_n <= len(pages) else None
    hit = _locate_quote(quote, page) if page is not None else None
    if hit is None:
        return {**base, "met": False, "note": NOTE_QUOTE_NOT_FOUND}
    return {**base, "met": met, "evidence": [hit]}


def _verified(answer: dict[str, Any], criteria: list[dict[str, Any]], pages: list[dict[str, Any]]) -> dict:
    by_id: dict[str, dict[str, Any]] = {}
    for e in answer.get("criteria") or []:
        by_id.setdefault(e["criterion_id"], e)
    entries = [_check_entry(c, by_id.get(c["id"]), pages) for c in criteria]
    problems = [f"{e['criterion_id']} {e['note']}" for e in entries if e["required"] and e["note"]]
    verdict, reasoning = answer["verdict"], answer["reasoning"]
    if problems:
        verdict = "unsure"
        reasoning = f"{reasoning} [verifier: {'; '.join(problems)}; verdict downgraded to unsure]"
    return {"verdict": verdict, "score": float(answer["score"]), "reasoning": reasoning[:2000],
            "criteria": entries}


def llm_screen(llm: Any, criteria: list[dict[str, Any]], source: Source,
               pages: list[dict[str, Any]]) -> dict[str, Any] | None:
    """A verified LLM screening proposal, or None (invalid answer / client failure)."""
    text = "\n\n".join((p.get("text") or "") for p in pages[:PAGES_FOR_LLM])
    user = wrap_document(text, _TASK.format(criteria=_criteria_block(criteria)), title=source.title,
                         max_chars=MAX_DOC_CHARS)
    try:
        raw = llm.extract_json(EVIDENCE_SYSTEM, user, SCREEN_SCHEMA)
    except Exception as exc:                      # a transport failure is a fallback, not a 500
        log.warning("evidence: llm screen failed for %s: %s", source.id, exc)
        return None
    answer = validate(SCREEN_SCHEMA, raw) if isinstance(raw, dict) else None
    return None if answer is None else _verified(answer, criteria, pages)


def _fallback(out: dict[str, Any]) -> dict[str, Any]:
    return {**out, "reasoning": f"[{NOTE_LLM_INVALID}: LLM answer rejected, keyword fallback] {out['reasoning']}",
            "criteria": [{**c, "note": NOTE_LLM_INVALID} for c in out["criteria"]]}


# ------------------------------------------------------------------- tool ---

def _select_sources(store: EvidenceStore, sid: str, ids: Any) -> list[Source]:
    sources = store.list_sources(sid)
    if ids is None:
        return sources
    if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
        raise ValueError("'source_ids' must be a list of source ids")
    by_id = {s.id: s for s in sources}
    unknown = [i for i in ids if i not in by_id]
    if unknown:
        raise ValueError(f"unknown evidence source(s) in this session: {unknown}")
    return [by_id[i] for i in dict.fromkeys(ids)]


def screening_results(store: EvidenceStore, sid: str, criteria_hash: str, method: str | None = None) -> dict:
    """PharmState mirror for the screens of one criteria version (counts only)."""
    rows = [s for s in store.list_screens(sid) if s.criteria_hash == criteria_hash]
    status, verdicts = Counter(s.status for s in rows), Counter(s.verdict for s in rows)
    methods = {s.method for s in rows}
    return {"criteria_hash": criteria_hash, "n_screened": len(rows),
            "verdicts": {v: verdicts.get(v, 0) for v in ("include", "exclude", "unsure")},
            "proposed": status.get("proposed", 0), "approved": status.get("approved", 0),
            "rejected": status.get("rejected", 0),
            "method": method or (methods.pop() if len(methods) == 1 else "mixed" if methods else "keyword"),
            "updated_at": now_iso()}


def _screen_one(src: Source, criteria: list[dict[str, Any]], llm: Any) -> tuple[dict[str, Any], str]:
    pages = (ingest.load_pages(src).get("pages")) or []
    if llm is not None:
        out = llm_screen(llm, criteria, src, pages)
        if out is not None:
            return out, "llm"
        return _fallback(keyword_screen(criteria, pages)), "keyword"
    return keyword_screen(criteria, pages), "keyword"


def screen_publications(state: PharmState, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    store, sid = get_store(), state.session_id
    cfg = store.get_config(sid, "criteria")
    if not cfg:
        raise ValueError("no screening criteria saved — save the inclusion/exclusion criteria first")
    criteria, chash = cfg["body"], cfg["body_hash"]
    llm = current_llm() if llm_available() else None
    rows, skipped = [], []
    for src in _select_sources(store, sid, args.get("source_ids")):
        if src.parse_status != "parsed":
            skipped.append({"source_id": src.id, "reason": f"parse_status {src.parse_status}"})
            continue
        try:
            out, method = _screen_one(src, criteria, llm)
        except (OSError, ValueError) as exc:                  # fail closed: changed/missing file
            skipped.append({"source_id": src.id, "reason": str(exc)})
            continue
        rows.append({"source_id": src.id, "criteria_hash": chash, "method": method, **out})
    if not rows:
        raise ValueError("no parsed publications to screen" + (f" ({len(skipped)} skipped)" if skipped else ""))
    before = {s.id for s in store.list_screens(sid)}
    stored = store.upsert_screens(sid, rows)
    n_new = sum(1 for s in stored if s.id not in before)
    methods = {r["method"] for r in rows}
    method = methods.pop() if len(methods) == 1 else "mixed"
    verdicts = Counter(s.verdict for s in stored)
    tally = ", ".join(f"{v} {verdicts.get(v, 0)}" for v in ("include", "exclude", "unsure"))
    results = screening_results(store, sid, chash, method)
    return ToolResult(
        summary=(f"Screened {len(stored)} publication(s) ({method}): {tally}; {n_new} new verdict(s) — "
                 f"{results['proposed']} await your decision in the Evidence workspace."),
        writes={"screening_results": results, "evidence_summary": summarise(store, sid)},
        result={"screens": [asdict(s) for s in stored], "skipped": skipped, "criteria_hash": chash,
                "n_new": n_new, "n_screened": len(stored), "method": method},
        action=f"screened {len(stored)} sources against criteria {chash[:12]} ({method}): {tally}; {n_new} new")


# ----------------------------------------------------------------- decide ---

def _owned_screen(store: EvidenceStore, sid: str, screen_id: Any) -> Screen:
    if not isinstance(screen_id, str) or not screen_id:
        raise ValueError("'screen_id' is required")
    for s in store.list_screens(sid):
        if s.id == screen_id:
            return s
    raise ValueError(f"unknown screening verdict in this session: {screen_id}")


def decide_screening(state: PharmState, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    store, sid = get_store(), state.session_id
    if args.get("confirm") is not True:
        raise ValueError("'confirm' must be true: a screening decision is the reviewer's, never the model's")
    status = _DECISION_STATUS.get(args.get("decision"))
    if status not in DECISIONS:
        raise ValueError("'decision' must be 'approve' or 'reject'")
    reason = args.get("reason")
    if not isinstance(reason, str) or not (1 <= len(reason.strip()) <= MAX_REASON):
        raise ValueError(f"'reason' is required (1..{MAX_REASON} characters)")
    actor = args.get("actor") if isinstance(args.get("actor"), str) and args.get("actor") else "anonymous"
    scr = _owned_screen(store, sid, args.get("screen_id"))
    decided = store.decide_screen(scr.id, status, actor, reason.strip())   # EvidenceStateError if decided
    cfg = store.get_config(sid, "criteria")
    results = screening_results(store, sid, cfg["body_hash"] if cfg else decided.criteria_hash)
    return ToolResult(
        summary=f"Screening verdict {decided.id} ({decided.verdict}) {status} by {actor}: {reason.strip()}",
        writes={"screening_results": results, "evidence_summary": summarise(store, sid)},
        result={"screen": asdict(decided)},
        action=f"decide_screening({decided.id}, {status})")


TOOLS = [
    Tool(name="screen_publications", agent=AGENT,
         description=("Screen the uploaded publications against the saved inclusion/exclusion criteria. "
                      "Each verdict is a PROPOSAL (include/exclude/unsure with per-criterion evidence "
                      "locators) that the reviewer approves or rejects; nothing is decided here. "
                      "Optional source_ids restricts the run; same criteria re-use existing verdicts."),
         input_schema={"type": "object",
                       "properties": {"source_ids": {"type": "array", "items": {"type": "string"}},
                                      "actor": {"type": "string"}}},
         run=screen_publications),
    Tool(name="decide_screening", agent=AGENT,
         description=("Record the reviewer's decision on one proposed screening verdict (approve or "
                      "reject, with a reason). Human-only: requires confirm=true from the workspace."),
         input_schema={"type": "object",
                       "properties": {"screen_id": {"type": "string"},
                                      "decision": {"type": "string", "enum": ["approve", "reject"]},
                                      "reason": {"type": "string"}, "actor": {"type": "string"},
                                      "confirm": {"type": "boolean"}},
                       "required": ["screen_id", "decision", "reason", "confirm"]},
         run=decide_screening),
]
