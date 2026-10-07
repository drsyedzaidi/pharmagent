"""Where a record's cited value sits on its page: locator + parse + layout sidecar -> Region.

Pure: no I/O, no rendering. The answer is in PDF user-space boxes (parse_layout) and carries a
PRECISION the reviewer is told about:

- ``value``  the words holding the cited number (word-level: punctuation or a neighbouring character in
             the same word is inside the box),
- ``cell``   the table cell (ruled or layout table), with the column header and row label cells as context,
- ``line``   only the cited text (a text field, or a table cell whose box was not stored): the snippet's words,
- ``page``   not located; ``resolved`` is False and ``reason`` says why.

Nothing is guessed: the layout must be bound to the exact page text it indexes, the span must still hold
the snippet, the token must still be the literal, and every word involved must have a box. Any doubt
returns precision 'page' (never a crop around the wrong place).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.evidence.normalize import normalise
from app.evidence.parse_layout import sha256_text

Box = tuple[float, float, float, float]
MAX_VALUE_BOXES = 12


@dataclass(frozen=True)
class Region:
    precision: str                           # value | cell | line | page
    resolved: bool
    reason: str | None = None
    value_boxes: tuple[Box, ...] = ()
    snippet_box: Box | None = None
    context_boxes: tuple[Box, ...] = ()


def _page(reason: str) -> Region:
    return Region("page", False, reason)


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _pair(v: Any, size: int) -> tuple[int, int] | None:
    """(a, b) when *v* is a two-int list with 0 <= a < b <= size, else None."""
    if not isinstance(v, (list, tuple)) or len(v) != 2 or not all(_is_int(x) for x in v):
        return None
    a, b = v
    return (a, b) if 0 <= a < b <= size else None


def _union(boxes: list[Box]) -> Box:
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes),
            max(b[3] for b in boxes))


def _boxes(words: list[list[float]]) -> list[Box] | None:
    """The words' boxes, or None when there are none or any word has no box."""
    if not words or any(len(w) != 6 for w in words):
        return None
    return [(float(w[2]), float(w[3]), float(w[4]), float(w[5])) for w in words]


def _overlapping(words: list[list[float]], a: int, b: int) -> list[list[float]]:
    return [w for w in words if w[0] < b and w[1] > a]


def _same(a: str, b: str) -> bool:
    return normalise(a)[0] == normalise(b)[0]


def _box(v: Any) -> Box | None:
    if isinstance(v, (list, tuple)) and len(v) == 4 and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                                                            for x in v):
        return (float(v[0]), float(v[1]), float(v[2]), float(v[3]))
    return None


def _status_reason(layout: Any, page_doc: dict[str, Any]) -> str | None:
    """Why *layout* cannot be used for this page, or None."""
    if not isinstance(layout, dict) or not isinstance(layout.get("words"), list):
        return "layout_unavailable"
    if layout.get("status") != "ok":
        return {"unmapped": "word_unmapped"}.get(str(layout.get("status")), "no_boxes")
    if layout.get("text_sha256") != sha256_text(str(page_doc.get("text") or "")):
        return "layout_stale"
    return None


def _snippet_region(words: list[list[float]], text: str, loc: dict[str, Any], span: tuple[int, int],
                    precision: str) -> Region:
    """The snippet's words (every one boxed) as the located text."""
    if not _same(text[span[0]:span[1]], str(loc.get("snippet") or "")):
        return _page("span_stale")
    boxes = _boxes(_overlapping(words, *span))
    if boxes is None or len(boxes) > MAX_VALUE_BOXES * 4:
        return _page("word_unmapped")
    return Region(precision, True, None, (_union(boxes),), _union(boxes))


def _text(loc: dict[str, Any], page_doc: dict[str, Any], layout: dict[str, Any]) -> Region:
    col = loc.get("column")
    columns = [c for c in (page_doc.get("columns") or []) if isinstance(c, str)]
    if col is None:
        text, words = str(page_doc.get("text") or ""), layout["words"]
    elif _is_int(col) and 0 <= col < len(columns):
        shas, col_words = layout.get("col_sha256"), layout.get("col_words")
        if not (isinstance(shas, list) and len(shas) == len(columns) and shas[col] == sha256_text(columns[col])):
            return _page("layout_stale")
        if not (isinstance(col_words, list) and len(col_words) == len(columns)):
            return _page("word_unmapped")
        text, words = columns[col], col_words[col]
    else:
        return _page("no_span")
    if loc.get("span") is None:
        return _page("no_span")
    span = _pair(loc.get("span"), len(text))
    if span is None:
        return _page("span_stale")
    literal, token = loc.get("literal"), loc.get("token")
    if literal is None and token is None:                          # a text field: the snippet itself
        return _snippet_region(words, text, loc, span, "line")
    tok = _pair(token, len(text))
    if tok is None or not isinstance(literal, str) or text[tok[0]:tok[1]] != literal \
            or not (span[0] <= tok[0] and tok[1] <= span[1]):
        return _page("span_stale")
    if not _same(text[span[0]:span[1]], str(loc.get("snippet") or "")):
        return _page("span_stale")
    value = _boxes(_overlapping(words, *tok))
    if value is None or len(value) > MAX_VALUE_BOXES:
        return _page("word_unmapped")
    snippet = _boxes(_overlapping(words, *span))
    return Region("value", True, None, tuple(value), _union(snippet if snippet else value))


def _context(loc_cell: dict[str, Any], rows: list[list[str]], r: int, c: int,
             cells: dict[str, Any]) -> tuple[Box, ...]:
    """The column header cell and the row label cell, when they carry the text the locator vouches for."""
    out: list[Box] = []
    for key, expect, (rr, cc) in ((f"0:{c}", loc_cell.get("header"), (0, c)),
                                  (f"{r}:0", loc_cell.get("row_label"), (r, 0))):
        if (rr, cc) == (r, c) or not isinstance(expect, str) or not expect.strip():
            continue
        box = _box(cells.get(key))
        if box is not None and rr < len(rows) and cc < len(rows[rr]) and _same(str(rows[rr][cc] or ""), expect):
            out.append(box)
    return tuple(out)


def _table(loc: dict[str, Any], page_doc: dict[str, Any], layout: dict[str, Any]) -> Region:
    cell = loc.get("cell")
    idx = [cell.get(k) for k in ("table", "row", "col")] if isinstance(cell, dict) else []
    tables = page_doc.get("tables") or []
    if len(idx) != 3 or not all(_is_int(i) and i >= 0 for i in idx) or idx[0] >= len(tables):
        return _page("cell_unmapped")
    t, r, c = idx
    rows = tables[t].get("rows") or []
    if r >= len(rows) or c >= len(rows[r]):
        return _page("cell_unmapped")
    cell_text = str(rows[r][c] or "")
    entry = next((x for x in layout.get("tables") or [] if isinstance(x, dict) and x.get("index") == t), None)
    cells = entry.get("cells") if isinstance(entry, dict) and isinstance(entry.get("cells"), dict) else {}
    box = _box(cells.get(f"{r}:{c}"))
    literal, token = loc.get("literal"), loc.get("token")
    if token is not None or literal is not None:
        tok = _pair(token, len(cell_text))
        if tok is None or not isinstance(literal, str) or cell_text[tok[0]:tok[1]] != literal:
            return _page("span_stale")
    if box is not None:
        return Region("cell", True, None, (box,), box, _context(cell, rows, r, c, cells))
    text = str(page_doc.get("text") or "")
    span = _pair(loc.get("span"), len(text))
    if span is None:
        return _page("cell_unmapped")
    row = _snippet_region(layout["words"], text, loc, span, "line")
    return row if row.resolved else _page("cell_unmapped")


def locate(locator: dict[str, Any], page_doc: dict[str, Any], layout_page: dict[str, Any] | None) -> Region:
    """The Region of *locator* on its page (*page_doc*: the parse's page, *layout_page*: the sidecar's)."""
    reason = _status_reason(layout_page, page_doc)
    if reason is not None or layout_page is None:
        return _page(reason or "layout_unavailable")
    kind = locator.get("kind") or "text"
    if kind == "text":
        return _text(locator, page_doc, layout_page)
    if kind == "table":
        return _table(locator, page_doc, layout_page)
    return _page("no_span")
