"""Evidence agent tools: extract_evidence (PROPOSALS per included source), verify_records (re-run the
literal verifier on proposed rows) and decide_records (the human's, approve refused on a failed row).

Pipeline per source: proposals (LLM when configured, else header-matched table cells + text rules)
-> deterministic literal verification against the parsed page -> scripted QC flags -> instruction-
text flag -> optional add-only LLM flags -> INSERT OR IGNORE on the claim hash. Only sources whose
screening verdict is an approved 'include' are read. Every value in the ledger therefore either
came from deterministic code or is a model proposal that a deterministic check found literally
on the cited page; the reviewer still decides each row.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict
from typing import Any

from app.core.pharmstate import PharmState
from app.evidence import ingest
from app.evidence import models as M
from app.evidence.config_tools import summarise
from app.evidence.extract_llm import MAX_PAGE_CHARS, llm_extract, llm_flags
from app.evidence.extract_rules import field_index, norm_key, rule_extract
from app.evidence.llm_access import current_llm, llm_available
from app.evidence.models import DECISIONS, Record, Source, now_iso
from app.evidence.normalize import normalise, numbers_in
from app.evidence.qc_flags import merge_llm_flags, scripted_flags
from app.evidence.screen_tools import _select_sources
from app.evidence.store import EvidenceStore, get_store
from app.evidence.verify import verify
from app.tools.base import Tool, ToolContext, ToolResult

__all__ = ["MAX_PAGE_CHARS", "check_proposals", "extract_evidence", "verify_records", "decide_records", "TOOLS"]

AGENT = "evidence"
MAX_REASON = 500
NOTE_LLM_INVALID = "llm_invalid"
_DECISION_STATUS = {"approve": "approved", "reject": "rejected"}
_INSTRUCTION_RE = re.compile(
    r"\b(ignore|disregard|override|forget)\b[^.\n]{0,60}\b(instructions?|rules?|prompts?|guidelines?)\b"
    r"|\bapprove (everything|all|every)\b|\bsystem prompt\b|\b(you are an? |as an ai\b|automated readers?)",
    re.IGNORECASE)


# ---------------------------------------------------------------- checking ---

def _suspicious(p: dict[str, Any]) -> list[dict[str, Any]]:
    text = f"{p.get('value') or ''} {p['locator'].get('snippet') or ''}"
    if not _INSTRUCTION_RE.search(text):
        return []
    return [{"code": M.FLAG_SUSPICIOUS_INSTRUCTION, "severity": "warn",
             "message": f"{p['field']}: the value or its snippet reads like an instruction to an automated "
                        "reader; it is document text, not a finding", "field": p["field"],
             "group": p.get("group_label") or "", "origin": "script"}]


def _composite(p: dict[str, Any], ftype: str) -> list[dict[str, Any]]:
    """A numeric table cell holding more than one number ('3,412 (812)'): the estimate was taken,
    the reviewer must confirm which number it is."""
    loc = p["locator"]
    if ftype == "text" or loc.get("kind") != "table" or any(f["code"] == M.FLAG_COMPOSITE_CELL for f in p["flags"]):
        return []
    if len(numbers_in(str(loc.get("snippet") or ""), loc.get("decimal_style") or ".")) <= 1:
        return []
    return [{"code": M.FLAG_COMPOSITE_CELL, "severity": "warn",
             "message": f"{p['field']}: cell {loc.get('snippet')!r} holds more than one number; "
                        f"{p['value']!r} was read as the estimate", "field": p["field"],
             "group": p.get("group_label") or "", "origin": "script"}]


def _field_type(p: dict[str, Any], index: dict[str, dict[str, Any]]) -> str:
    spec = index.get(norm_key(p.get("field")))
    if spec is not None:
        return str(spec.get("type") or "number")
    return "text" if p.get("value_num") is None else "number"


def _verify_one(p: dict[str, Any], doc: dict[str, Any], ftype: str) -> dict[str, Any]:
    res = verify(p, doc, field_type=ftype)
    status, reason, loc = res.status, res.reason, dict(res.locator or p["locator"])
    if status == "pass" and ftype == "text" and normalise(str(p.get("value") or ""))[0] not in normalise(
            str(loc.get("snippet") or ""))[0]:
        status, reason, loc = "fail", M.REASON_TEXT_NOT_IN_SNIPPET, {**loc, "span": None, "literal": None}
    return {**p, "locator": loc, "verify_status": status, "verify_reason": reason}


def check_proposals(proposals: list[dict[str, Any]], doc: dict[str, Any] | None, template: dict[str, Any],
                    approved: list[Any] | None = None) -> list[dict[str, Any]]:
    """Pure: literal verification of every proposal (all fail 'source_changed' when *doc* is None),
    then scripted QC flags and the instruction-text flag. Returns new dicts; inputs are untouched."""
    index = field_index(template)
    if doc is None:
        checked = [{**p, "locator": {**p["locator"], "span": None, "literal": None}, "verify_status": "fail",
                    "verify_reason": M.REASON_SOURCE_CHANGED} for p in proposals]
    else:
        checked = [_verify_one(p, doc, _field_type(p, index)) for p in proposals]
    flags = scripted_flags(checked, template, list(approved or []))
    out = [{**p, "flags": fl + _suspicious(p)} for p, fl in zip(checked, flags)]
    return [{**p, "flags": p["flags"] + _composite(p, _field_type(p, index))} for p in out]


# ----------------------------------------------------------------- extract ---

def _included_sources(store: EvidenceStore, sid: str, ids: Any, criteria_hash: str | None
                      ) -> tuple[list[Source], list[dict[str, Any]]]:
    """Sources whose screening verdict (current criteria, else the latest) is an approved 'include'."""
    screens: dict[str, Any] = {}
    for s in store.list_screens(sid):                      # ordered by created_at: later rows win
        if s.source_id not in screens or screens[s.source_id].criteria_hash != criteria_hash:
            screens[s.source_id] = s
    included, skipped = [], []
    for src in _select_sources(store, sid, ids):
        scr = screens.get(src.id)
        if scr is None or scr.status != "approved" or scr.verdict != "include":
            how = "not screened" if scr is None else f"screen {scr.id} is {scr.status} {scr.verdict}"
            skipped.append({"source_id": src.id, "reason": f"not an approved include verdict ({how})"})
        elif src.parse_status != "parsed":
            skipped.append({"source_id": src.id, "reason": f"parse_status {src.parse_status}"})
        else:
            included.append(src)
    return included, skipped


def _propose(llm: Any, template: dict[str, Any], src: Source, doc: dict[str, Any]) -> dict[str, Any]:
    if llm is not None:
        out = llm_extract(llm, template, src, doc)
        if out is not None:
            return {"proposals": out[0], "method": "llm", "unmatched_headers": [], "dropped": out[1], "note": ""}
    rules = rule_extract(doc, template, src.id)
    return {**rules, "method": "table", "dropped": 0, "note": NOTE_LLM_INVALID if llm is not None else ""}


def extraction_results(store: EvidenceStore, sid: str, template_hash: str, method: str | None = None) -> dict:
    """PharmState mirror for the records of one template version (counts only)."""
    rows = [r for r in store.list_records(sid) if r.template_hash == template_hash]
    status, extractors = Counter(r.status for r in rows), {r.extractor for r in rows}
    return {"template_hash": template_hash, "n_records": len(rows),
            "proposed": status.get("proposed", 0), "approved": status.get("approved", 0),
            "rejected": status.get("rejected", 0),
            "verify_fail": sum(1 for r in rows if r.status == "proposed" and r.verify_status == "fail"),
            "flagged": sum(1 for r in rows if r.flags),
            "method": method or (extractors.pop() if len(extractors) == 1 else "mixed" if extractors else "table"),
            "updated_at": now_iso()}


def _extract_source(store: EvidenceStore, sid: str, src: Source, template: dict[str, Any], thash: str,
                    llm: Any, approved: list[Record]) -> tuple[dict[str, Any], list[Record]]:
    doc = ingest.load_pages(src)
    out = _propose(llm, template, src, doc)
    checked = check_proposals(out["proposals"], doc, template, approved)
    if llm is not None and checked:
        checked = [{**p, "flags": f} for p, f in zip(checked, llm_flags(llm, template, checked))]
    inserted = store.insert_records(sid, src, thash, checked)
    statuses = Counter(p["verify_status"] for p in checked)
    row = {"source_id": src.id, "method": out["method"], "note": out["note"], "n_candidates": len(checked),
           "pass": statuses.get("pass", 0), "fail": statuses.get("fail", 0),
           "flagged": sum(1 for p in checked if p["flags"]), "n_new": len(inserted),
           "unmatched_headers": out["unmatched_headers"], "dropped": out["dropped"]}
    return row, inserted


def extract_evidence(state: PharmState, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    store, sid = get_store(), state.session_id
    tmpl = store.get_config(sid, "template")
    if not tmpl:
        raise ValueError("no extraction template saved — save the template fields first")
    template, thash = tmpl["body"], tmpl["body_hash"]
    crit = store.get_config(sid, "criteria")
    included, skipped = _included_sources(store, sid, args.get("source_ids"), crit["body_hash"] if crit else None)
    if not included:
        raise ValueError("no included publications to extract from — approve an 'include' screening verdict "
                         f"first ({len(skipped)} skipped)")
    llm = current_llm() if llm_available() else None
    approved = store.list_records(sid, status="approved")
    rows, inserted = [], []
    for src in included:
        try:
            row, new = _extract_source(store, sid, src, template, thash, llm, approved)
        except (OSError, ValueError) as exc:                  # fail closed: changed/missing file
            skipped.append({"source_id": src.id, "reason": str(exc)})
            continue
        rows.append(row)
        inserted.extend(new)
    if not rows:
        raise ValueError(f"no readable publications to extract from ({len(skipped)} skipped)")
    methods = {r["method"] for r in rows}
    method = methods.pop() if len(methods) == 1 else "mixed"
    tot = {k: sum(r[k] for r in rows) for k in ("n_candidates", "pass", "fail", "flagged", "n_new")}
    results = extraction_results(store, sid, thash, method)
    return ToolResult(
        summary=(f"Extracted {tot['n_candidates']} candidate record(s) from {len(rows)} publication(s) ({method}): "
                 f"{tot['pass']} verified literally, {tot['fail']} failed verification, {tot['flagged']} flagged; "
                 f"{tot['n_new']} new — {results['proposed']} await your decision in the Evidence workspace."),
        writes={"extraction_results": results, "evidence_summary": summarise(store, sid)},
        result={"records": [asdict(r) for r in inserted], "sources": rows, "skipped": skipped,
                "template_hash": thash, "method": method, **tot},
        action=(f"extracted {tot['n_candidates']} candidate records from {len(rows)} sources (template "
                f"{thash[:12]}, {method}): {tot['pass']} verified, {tot['fail']} failed, {tot['flagged']} flagged; "
                f"{tot['n_new']} new"))


# ------------------------------------------------------------------ verify ---

def _selected_ids(store: EvidenceStore, sid: str, args: dict[str, Any], proposed: list[Record]) -> set[str]:
    """Proposed ids to re-verify: all of them, or the listed ones (decided ids are skipped, unknown refused)."""
    ids = args.get("record_ids")
    if ids is None:
        return {r.id for r in proposed}
    if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
        raise ValueError("'record_ids' must be a list of record ids")
    unknown = [i for i in ids if (rec := store.get_record(i)) is None or rec.session_id != sid]
    if unknown:
        raise ValueError(f"unknown record(s) in this session: {unknown}")
    return {r.id for r in proposed if r.id in set(ids)}


def _llm_flags_of(rec: Record) -> list[dict[str, Any]]:
    return [f for f in rec.flags if f.get("origin") == "llm"]


def verify_records(state: PharmState, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    """Re-run the verifier and the scripted flags on proposed rows; decided rows are never touched."""
    store, sid = get_store(), state.session_id
    tmpl = store.get_config(sid, "template")
    template, thash = (tmpl["body"], tmpl["body_hash"]) if tmpl else ({"fields": []}, "")
    proposed = store.list_records(sid, status="proposed")
    selected = _selected_ids(store, sid, args, proposed)
    approved = store.list_records(sid, status="approved")
    refreshed: list[Record] = []
    changed = 0
    for src_id in dict.fromkeys(r.source_id for r in proposed):
        rows = [r for r in proposed if r.source_id == src_id]
        src = store.get_source(src_id)
        try:
            doc = ingest.load_pages(src) if src is not None else None
        except (OSError, ValueError):
            doc = None
        checked = check_proposals([asdict(r) for r in rows], doc, template, approved)
        for rec, c in zip(rows, checked):
            if rec.id not in selected:
                continue
            flags = merge_llm_flags(c["flags"], _llm_flags_of(rec))
            if (c["verify_status"], c["verify_reason"]) != (rec.verify_status, rec.verify_reason):
                changed += 1
            if store.reverify(rec.id, c["verify_status"], c["verify_reason"], c["locator"], flags):
                refreshed.append(store.get_record(rec.id))
    counts = Counter(r.verify_status for r in refreshed)
    results = extraction_results(store, sid, thash)
    return ToolResult(
        summary=(f"Re-verified {len(refreshed)} proposed record(s): {counts.get('pass', 0)} pass, "
                 f"{counts.get('fail', 0)} fail, {changed} changed."),
        writes={"extraction_results": results, "evidence_summary": summarise(store, sid)},
        result={"n_checked": len(refreshed), "pass": counts.get("pass", 0), "fail": counts.get("fail", 0),
                "changed": changed, "records": [asdict(r) for r in refreshed]},
        action=f"verify_records({len(refreshed)} checked: {counts.get('pass', 0)} pass, "
               f"{counts.get('fail', 0)} fail, {changed} changed)")


# ------------------------------------------------------------------ decide ---

def _owned_record(store: EvidenceStore, sid: str, record_id: Any) -> Record:
    if not isinstance(record_id, str) or not record_id:
        raise ValueError("'record_id' is required")
    rec = store.get_record(record_id)
    if rec is None or rec.session_id != sid:
        raise ValueError(f"unknown record in this session: {record_id}")
    return rec


def decide_records(state: PharmState, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    store, sid = get_store(), state.session_id
    if args.get("confirm") is not True:
        raise ValueError("'confirm' must be true: a record decision is the reviewer's, never the model's")
    status = _DECISION_STATUS.get(args.get("decision"))
    if status not in DECISIONS:
        raise ValueError("'decision' must be 'approve' or 'reject'")
    reason = args.get("reason")
    if not isinstance(reason, str) or not (1 <= len(reason.strip()) <= MAX_REASON):
        raise ValueError(f"'reason' is required (1..{MAX_REASON} characters)")
    actor = args.get("actor") if isinstance(args.get("actor"), str) and args.get("actor") else "anonymous"
    rec = _owned_record(store, sid, args.get("record_id"))
    if status == "approved" and rec.verify_status == "fail":
        raise ValueError("cannot approve a record that failed literal verification "
                         f"({rec.verify_reason or 'unknown'}); re-verify or reject it")
    decided = store.decide_record(rec.id, status, actor, reason.strip())   # EvidenceStateError if decided
    src = store.get_source(decided.source_id)
    tmpl = store.get_config(sid, "template")
    results = extraction_results(store, sid, tmpl["body_hash"] if tmpl else decided.template_hash)
    return ToolResult(
        summary=f"Record {decided.id} ({decided.field} = {decided.value}) {status} by {actor}: {reason.strip()}",
        writes={"extraction_results": results, "evidence_summary": summarise(store, sid)},
        result={"record": asdict(decided), "record_hash": decided.record_hash,
                "source_sha256": src.sha256 if src else None, "locator": decided.locator, "value": decided.value,
                "unit": decided.unit, "verify_status": decided.verify_status, "decision": status,
                "reason": reason.strip()},
        action=f"decide_records({decided.id}, {status})")


TOOLS = [
    Tool(name="extract_evidence", agent=AGENT,
         description=("Extract the saved template's fields from the publications whose screening verdict is an "
                      "approved 'include'. Every record is a PROPOSAL with a page/cell locator that a "
                      "deterministic verifier must find literally on the page; the reviewer approves or rejects "
                      "each one. Optional source_ids restricts the run; identical claims are never duplicated."),
         input_schema={"type": "object",
                       "properties": {"source_ids": {"type": "array", "items": {"type": "string"}},
                                      "actor": {"type": "string"}}},
         run=extract_evidence),
    Tool(name="verify_records", agent=AGENT,
         description=("Re-run the literal-match verifier and the scripted QC flags on proposed records "
                      "(optionally only record_ids). Decided records are never changed."),
         input_schema={"type": "object",
                       "properties": {"record_ids": {"type": "array", "items": {"type": "string"}},
                                      "actor": {"type": "string"}}},
         run=verify_records),
    Tool(name="decide_records", agent=AGENT,
         description=("Record the reviewer's decision on one proposed record (approve or reject, with a reason). "
                      "Human-only: requires confirm=true from the workspace; a record that failed literal "
                      "verification cannot be approved."),
         input_schema={"type": "object",
                       "properties": {"record_id": {"type": "string"},
                                      "decision": {"type": "string", "enum": ["approve", "reject"]},
                                      "reason": {"type": "string"}, "actor": {"type": "string"},
                                      "confirm": {"type": "boolean"}},
                       "required": ["record_id", "decision", "reason", "confirm"]},
         run=decide_records),
]
