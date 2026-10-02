"""Literal-match verifier: a proposed value passes only when a deterministic check finds the
literal number inside the cited snippet AND the snippet inside the cited page (or column /
table cell). Offsets in the returned locator are RAW page-text offsets (raw[start:end] round-trips).

Check order: empty_snippet, snippet_too_long, page_out_of_range, non_ascii_digits, number
(no_number / number_mismatch / integer_expected), location (snippet_not_on_page for text,
table_cell_mismatch for table). A pass sets span, literal and (when used) column.
"""
from __future__ import annotations

import math
from typing import Any

from app.evidence import models as M
from app.evidence.normalize import (
    IdxMap,
    digits_only,
    has_non_ascii_digits,
    normalise,
    number_spans,
    parse_number,
)

MAX_SNIPPET_CHARS = 300
_REL_TOL = 1e-9


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=_REL_TOL, abs_tol=1e-12)


def _locate(snippet: str, text: str) -> tuple[int, int, IdxMap] | None:
    """(normalised start, normalised length, raw map) of the first normalised match, or None."""
    ns, _ = normalise(snippet)
    if not ns:
        return None
    nt, m = normalise(text)
    i = nt.find(ns)
    return None if i < 0 else (i, len(ns), m)


def _raw_span(start: int, length: int, m: IdxMap) -> list[int]:
    return [m[start], m[start + length - 1] + 1]


def find_snippet_in_page(snippet: str, page_text: str) -> list[int] | None:
    """Raw [start, end) offsets of *snippet* in *page_text* after normalisation, else None."""
    hit = _locate(snippet, page_text)
    return None if hit is None else _raw_span(*hit)


def find_number_in_snippet(value: str, snippet: str, decimal_style: str, integer: bool) -> str | None:
    """The literal token in *snippet* equal to *value* (parsed with *decimal_style*), else None."""
    literal, reason = _number_check(value, snippet, decimal_style, integer)
    return literal if reason == M.REASON_OK else None


def _number_check(value: str, snippet: str, decimal_style: str, integer: bool) -> tuple[str | None, str]:
    target = parse_number(value, decimal_style)
    if target is None:
        return None, M.REASON_NO_NUMBER
    # Proposed values usually arrive in canonical point-decimal form ('18.2' for a page that
    # prints '18,2'); both readings are tried, the page literal is what gets verified.
    targets = [target] + [t for t in [parse_number(value, ".")] if t is not None and not _close(t, target)]
    if integer and not all(float(t).is_integer() for t in targets[:1]):
        return None, M.REASON_INTEGER_EXPECTED
    spans = number_spans(snippet, decimal_style)
    for start, end, found in spans:
        if any(_close(found, t) for t in targets):
            return snippet[start:end], M.REASON_OK
    vdig = digits_only(value)
    fragment = any(vdig and vdig in digits_only(snippet[s:e]) for s, e, _ in spans)
    return None, (M.REASON_NO_NUMBER if not spans or fragment else M.REASON_NUMBER_MISMATCH)


def _page_literal(snippet: str, literal: str | None, text: str, start: int, m: IdxMap) -> str | None:
    """Map the snippet's matched token onto the raw page text (e.g. '-20' in the snippet -> '−20' on the page)."""
    if literal is None:
        return None
    ns, _ = normalise(snippet)
    nl, _ = normalise(literal)
    j = ns.find(nl)
    if j < 0 or not nl:
        return literal
    return text[m[start + j]:m[start + j + len(nl) - 1] + 1]


def _fail(loc: dict[str, Any], reason: str) -> M.VerifyResult:
    return M.VerifyResult("fail", reason, {**loc, "span": None, "literal": None})


def _pass(loc: dict[str, Any], span: list[int] | None, literal: str | None) -> M.VerifyResult:
    return M.VerifyResult("pass", M.REASON_OK, {**loc, "span": span, "literal": literal}, literal, span)


def _candidates(loc: dict[str, Any], page: dict[str, Any]) -> list[tuple[int | None, str]]:
    columns = [c for c in (page.get("columns") or []) if isinstance(c, str)]
    col = loc.get("column")
    if isinstance(col, int) and not isinstance(col, bool) and 0 <= col < len(columns):
        return [(col, columns[col])]
    return [(None, page.get("text") or "")] + list(enumerate(columns))


def _verify_text(loc: dict[str, Any], page: dict[str, Any], snippet: str, literal: str | None) -> M.VerifyResult:
    for column, text in _candidates(loc, page):
        hit = _locate(snippet, text)
        if hit is None:
            continue
        start, length, m = hit
        return _pass({**loc, "column": column}, _raw_span(start, length, m),
                     _page_literal(snippet, literal, text, start, m))
    return _fail(loc, M.REASON_SNIPPET_NOT_ON_PAGE)


def _cell_text(cell: Any, page: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    """(cell text, table) for a well-formed in-range cell locator, else None."""
    if not isinstance(cell, dict):
        return None
    idx = [cell.get(k) for k in ("table", "row", "col")]
    if any(not isinstance(i, int) or isinstance(i, bool) or i < 0 for i in idx):
        return None
    t, r, c = idx
    tables = page.get("tables") or []
    if t >= len(tables):
        return None
    rows = tables[t].get("rows") or []
    if r >= len(rows) or c >= len(rows[r]):
        return None
    return str(rows[r][c] or ""), tables[t]


def _verify_table(loc: dict[str, Any], page: dict[str, Any], snippet: str, literal: str | None) -> M.VerifyResult:
    found = _cell_text(loc.get("cell"), page)
    if found is None:
        return _fail(loc, M.REASON_TABLE_CELL_MISMATCH)
    cell_text, table = found
    ns, _ = normalise(snippet)
    nc, _ = normalise(cell_text)
    if not ns or ns not in nc or (literal is not None and normalise(literal)[0] not in nc):
        return _fail(loc, M.REASON_TABLE_CELL_MISMATCH)
    cell = {**loc["cell"], "strategy": table.get("strategy")}
    text = page.get("text") or ""
    hit = _locate(snippet, text)
    if hit is None:                       # cell verified but not locatable in the page text stream
        return _pass({**loc, "cell": cell}, None, literal)
    start, length, m = hit
    return _pass({**loc, "cell": cell}, _raw_span(start, length, m), _page_literal(snippet, literal, text, start, m))


def verify(record_like: Any, pages_doc: dict[str, Any], *, field_type: str = "number") -> M.VerifyResult:
    """Verify one proposal (dict or Record) against a parse document; never raises on bad input."""
    rec = M.as_mapping(record_like)
    loc = dict(rec.get("locator") or {})
    value = str(rec.get("value") if rec.get("value") is not None else "")
    snippet = str(loc.get("snippet") or "")
    style = loc.get("decimal_style") or pages_doc.get("decimal_style") or "."
    if not snippet.strip():
        return _fail(loc, M.REASON_EMPTY_SNIPPET)
    if len(snippet) > MAX_SNIPPET_CHARS:
        return _fail(loc, M.REASON_SNIPPET_TOO_LONG)
    pages = pages_doc.get("pages") or []
    page_n = loc.get("page")
    if not isinstance(page_n, int) or isinstance(page_n, bool) or not 1 <= page_n <= len(pages):
        return _fail(loc, M.REASON_PAGE_OUT_OF_RANGE)
    page = pages[page_n - 1]
    if has_non_ascii_digits(snippet) or has_non_ascii_digits(value):
        return _fail(loc, M.REASON_NON_ASCII_DIGITS)
    literal: str | None = None
    if field_type != "text":
        literal, reason = _number_check(value, snippet, style, field_type == "integer")
        if reason != M.REASON_OK:
            return _fail(loc, reason)
    kind = loc.get("kind") or "text"
    if kind == "table":
        return _verify_table(loc, page, snippet, literal)
    if kind != "text":
        return _fail(loc, M.REASON_SNIPPET_NOT_ON_PAGE)
    return _verify_text(loc, page, snippet, literal)
