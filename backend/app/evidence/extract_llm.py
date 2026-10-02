"""LLM extraction proposals (one wrapped, untrusted document per page) and the optional add-only
flag pass. Nothing an LLM returns reaches the ledger unverified: every record it proposes goes
through extract_tools.check_proposals (literal match on the cited page + scripted flags), and an
invalid answer for any page makes the whole source fall back to the deterministic rules.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from app.evidence.extract_rules import decimal_style, field_index, match_field, norm_key, value_number
from app.evidence.models import Source
from app.evidence.normalize import normalise
from app.evidence.prompts import (
    EVIDENCE_SYSTEM,
    EXTRACT_SCHEMA,
    FLAG_SCHEMA,
    MAX_FLAGS,
    validate,
    wrap_document,
)
from app.evidence.qc_flags import merge_llm_flags

log = logging.getLogger(__name__)

MAX_PAGE_CHARS, MAX_VALUE, MAX_UNIT, MAX_GROUP = 8000, 120, 40, 120
_TASK = (
    "The document above is page {page} of {total} of a publication. Extract every value it reports "
    "for the fields listed below. For each value return: field (the field name exactly as listed), "
    "group (the arm, dose level, population or table column the value belongs to; empty when none), "
    "value (one number exactly as printed, same digits and separators, no unit), unit (as printed, or "
    "null), page ({page}), kind ('table' when read from a table cell, else 'text') and snippet (an "
    "exact quote of the table cell or the sentence fragment containing the value, <= 300 characters). "
    "Omit fields this page does not report; never infer or convert.\n\nFields:\n{fields}"
)
_FLAG_TASK = (
    "The document above lists records proposed from a publication (field, group, value, unit, page, "
    "snippet, flags already raised). Propose additional QC flags only where a record looks wrong or "
    "needs the reviewer's attention (implausible magnitude, wrong row or column, dispersion instead of "
    "estimate, mixed units, inconsistent group labels). Return flags with a short code, severity "
    "(warn or error), message, and the field and group of the record concerned. Return an empty list "
    "when nothing stands out."
)


def _field_block(template: dict[str, Any]) -> str:
    lines = []
    for f in template.get("fields") or []:
        unit = f" unit {f['unit']}" if f.get("unit") else ""
        aliases = f"; also called {', '.join(f['aliases'])}" if f.get("aliases") else ""
        desc = f": {f['description']}" if f.get("description") else ""
        lines.append(f"- {f['name']} ({f.get('type', 'number')}{unit}){desc}{aliases}")
    return "\n".join(lines)


def pages_tables_first(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(pages, key=lambda p: 0 if p.get("tables") else 1)


def find_cell(page: dict[str, Any], snippet: str) -> dict[str, Any] | None:
    """The first table cell whose text contains the (normalised) snippet, as a cell locator."""
    ns, _ = normalise(snippet)
    if not ns:
        return None
    for table in page.get("tables") or []:
        rows = table.get("rows") or []
        for ri, row in enumerate(rows):
            for ci, cell in enumerate(row):
                text = str(cell or "")
                if text.strip() and ns in normalise(text)[0]:
                    header = str(rows[0][ci]) if rows and ci < len(rows[0]) else ""
                    return {"table": int(table.get("index") or 0), "row": ri, "col": ci, "header": header,
                            "row_label": str(row[0] or "") if row else ""}
    return None


def _to_proposal(rec: dict[str, Any], index: dict[str, dict[str, Any]], doc: dict[str, Any],
                 source_id: str) -> dict[str, Any] | None:
    spec = match_field(str(rec.get("field") or ""), index)
    if spec is None:
        return None
    pages = doc.get("pages") or []
    page_n = rec["page"]
    page = pages[page_n - 1] if 1 <= page_n <= len(pages) else None
    snippet = str(rec["snippet"])
    kind, cell = rec["kind"], None
    if kind == "table":
        cell = find_cell(page, snippet) if page is not None else None
        kind = "table" if cell is not None else "text"
    style = decimal_style(spec, doc)
    value = str(rec["value"]).strip()[:MAX_VALUE]
    unit = str(rec.get("unit") or "").strip()[:MAX_UNIT] or None
    locator = {"page": page_n, "kind": kind, "snippet": snippet, "span": None, "literal": None, "cell": cell,
               "column": None, "decimal_style": style}
    return {"source_id": source_id, "field": str(spec["name"]), "group_label": str(rec.get("group") or "")[:MAX_GROUP],
            "value": value, "value_num": value_number(value, style), "unit": unit, "locator": locator,
            "extractor": "llm"}


def llm_extract(llm: Any, template: dict[str, Any], source: Source,
                doc: dict[str, Any]) -> tuple[list[dict[str, Any]], int] | None:
    """(proposals, n_dropped) — or None when any page answer is invalid or the client fails."""
    index = field_index(template)
    fields = _field_block(template)
    pages = doc.get("pages") or []
    out: list[dict[str, Any]] = []
    dropped = 0
    for page in pages_tables_first(pages):
        text = page.get("text") or ""
        if not text.strip():
            continue
        title = f"{source.title or source.filename} (page {page['n']} of {len(pages)})"
        user = wrap_document(text, _TASK.format(page=page["n"], total=len(pages), fields=fields),
                             title=title, max_chars=MAX_PAGE_CHARS)
        try:
            raw = llm.extract_json(EVIDENCE_SYSTEM, user, EXTRACT_SCHEMA)
        except Exception as exc:                    # a transport failure is a fallback, not a 500
            log.warning("evidence: llm extraction failed for %s page %s: %s", source.id, page.get("n"), exc)
            return None
        answer = validate(EXTRACT_SCHEMA, raw) if isinstance(raw, dict) else None
        if answer is None:
            return None
        for rec in answer["records"]:
            p = _to_proposal(rec, index, doc, source.id)
            if p is None:
                dropped += 1
            else:
                out.append(p)
    return out, dropped


def _flag_rows(proposals: list[dict[str, Any]]) -> str:
    rows = [{"field": p["field"], "group": p.get("group_label") or "", "value": p["value"], "unit": p.get("unit"),
             "page": p["locator"].get("page"), "snippet": p["locator"].get("snippet"),
             "flags": [f.get("code") for f in p.get("flags") or []]} for p in proposals]
    return json.dumps(rows, ensure_ascii=False, indent=1)


def _valid_flags(raw: Any) -> list[dict[str, Any]]:
    """Each flag stands alone: well-formed items are kept even when a sibling is malformed."""
    items = raw.get("flags") if isinstance(raw, dict) else None
    if not isinstance(items, list):
        return []
    item_schema = FLAG_SCHEMA["properties"]["flags"]["items"]
    return [f for f in items[:MAX_FLAGS] if validate(item_schema, f) is not None]


def llm_flags(llm: Any, template: dict[str, Any], proposals: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """One merged flag list per proposal: scripted flags verbatim plus well-formed LLM flags (add-only).
    Unavailable, invalid or failing answers leave every list unchanged."""
    existing = [list(p.get("flags") or []) for p in proposals]
    if not proposals:
        return existing
    user = wrap_document(_flag_rows(proposals), _FLAG_TASK, title="proposed records", max_chars=MAX_PAGE_CHARS)
    try:
        raw = llm.extract_json(EVIDENCE_SYSTEM, user, FLAG_SCHEMA)
    except Exception as exc:
        log.warning("evidence: llm flag pass failed: %s", exc)
        return existing
    index = field_index(template)
    per: list[list[dict[str, Any]]] = [[] for _ in proposals]
    for flag in _valid_flags(raw):
        spec = match_field(str(flag.get("field") or ""), index)
        if spec is None:
            continue
        group = str(flag.get("group") or "")
        for i, p in enumerate(proposals):
            if norm_key(p["field"]) == norm_key(spec["name"]) and (not group or group == (p.get("group_label") or "")):
                per[i].append({**flag, "field": spec["name"], "group": p.get("group_label") or ""})
    return [merge_llm_flags(ex, add) for ex, add in zip(existing, per)]
