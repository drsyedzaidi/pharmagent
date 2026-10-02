"""Literal-match verifier: a proposed value passes only when a deterministic check finds the
literal number inside the cited snippet AND the snippet inside the cited page (or column /
table cell). Offsets in the returned locator are RAW page-text offsets (raw[start:end] round-trips).

Check order: empty_snippet, snippet_too_long, page_out_of_range, non_ascii_digits, number
(no_number / number_mismatch / integer_expected), location (snippet_not_on_page for text,
table_cell_mismatch for table; partial_number when the snippet is there but its number is only
part of a longer page number: '2.4' in '12.4', '20' in '-20', '234' in '1,234'). A pass sets span,
literal (the WHOLE page / cell token) and number (that token's value in the page's decimal style,
which is what value_num must be), and (when used) column. A table record's span is the cited
cell's row in the page text, or None when the row cannot be found there.
"""
from __future__ import annotations

import math
import re
from collections.abc import Iterator
from typing import Any

from app.evidence import models as M
from app.evidence.normalize import (
    IdxMap,
    digits_only,
    has_non_ascii_digits,
    normalise,
    number_tokens,
    parse_number,
)

MAX_SNIPPET_CHARS = 300
_REL_TOL = 1e-9
_ASCII_DIGIT_RE = re.compile(r"[0-9]")


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=_REL_TOL, abs_tol=1e-12)


def _occurrences(needle: str, hay: str) -> Iterator[int]:
    i = hay.find(needle) if needle else -1
    while i >= 0:
        yield i
        i = hay.find(needle, i + 1)


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


def bare_ten_unresolved(page: dict[str, Any]) -> bool:
    """True when the page has small glyphs with digits, so a bare 'x 10' may have lost its exponent."""
    return bool(_ASCII_DIGIT_RE.search(str(page.get("small_text") or "")))


def find_number_in_snippet(value: str, snippet: str, decimal_style: str, integer: bool) -> str | None:
    """The literal token in *snippet* equal to *value* (parsed with *decimal_style*), else None."""
    span, reason = _number_check(value, snippet, decimal_style, integer)
    return snippet[span[0]:span[1]] if span is not None and reason == M.REASON_OK else None


def _number_check(value: str, snippet: str, decimal_style: str, integer: bool, bare_ten: bool = True
                  ) -> tuple[tuple[int, int] | None, str]:
    """(raw [start, end) of the snippet token equal to *value*, reason). A value equal only to the
    mantissa of scientific notation ('1.2' in '1.2e-3') is a partial number."""
    target = parse_number(value, decimal_style)
    if target is None:
        return None, M.REASON_NO_NUMBER
    # Proposed values usually arrive in canonical point-decimal form ('18.2' for a page that
    # prints '18,2'); both readings are tried, but the stored number is the PAGE token's value.
    targets = [target] + [t for t in [parse_number(value, ".")] if t is not None and not _close(t, target)]
    if integer and not all(float(t).is_integer() for t in targets[:1]):
        return None, M.REASON_INTEGER_EXPECTED
    tokens = number_tokens(snippet, decimal_style, bare_ten=bare_ten)
    for start, end, found, _ in tokens:
        if found is not None and any(_close(found, t) for t in targets):
            return (start, end), M.REASON_OK
    if any(man is not None and _close(man, t) for *_, man in tokens for t in targets):
        return None, M.REASON_PARTIAL_NUMBER
    spans = [(s, e, v) for s, e, v, _ in tokens]
    vdig = digits_only(value)
    # A fragment is a STRICT sub-run of a token's digits ('24' in '245.6'); equal digits with a
    # different reading ('3.412' vs '3,412') are a separator mismatch, i.e. number_mismatch.
    fragment = any(vdig and vdig != digits_only(snippet[s:e]) and vdig in digits_only(snippet[s:e])
                   for s, e, _ in spans)
    return None, (M.REASON_NO_NUMBER if not spans or fragment else M.REASON_NUMBER_MISMATCH)


def _norm_range(snippet: str, lit: tuple[int, int]) -> tuple[int, int] | None:
    """Normalised [j, k) of the snippet's raw literal range *lit*."""
    _, ms = normalise(snippet)
    idx = [i for i, r in enumerate(ms) if lit[0] <= r < lit[1]]
    return (idx[0], idx[-1] + 1) if idx else None


def _whole_token(text: str, a: int, b: int, style: str, bare_ten: bool) -> tuple[str, float] | None:
    """(raw literal, value) when text[a:b] is exactly one complete number token with a readable value
    (strict spaces; scientific notation is one token)."""
    for s, e, v, _ in number_tokens(text, style, strict=True, bare_ten=bare_ten):
        if (s, e) == (a, b):
            return None if v is None else (text[s:e], v)
        if s >= b:
            break
    return None


Hit = tuple[list[int], str | None, float | None, list[int] | None]   # raw span, literal, value, token


def _hits(snippet: str, lit: tuple[int, int] | None, text: str, style: str,
          bare_ten: bool = True) -> tuple[list[Hit], bool]:
    """Every occurrence of *snippet* in *text* whose cited number (raw snippet range *lit*) is a WHOLE
    number token of *text* -- never a run inside '12.4', '-20' or '1,234' -- and whether the snippet
    occurred at all. Without *lit* (text fields) every occurrence is a hit."""
    ns, _ = normalise(snippet)
    nt, m = normalise(text)
    rng = _norm_range(snippet, lit) if lit is not None else None
    hits: list[Hit] = []
    seen = False
    for i in _occurrences(ns, nt):
        seen = True
        span = _raw_span(i, len(ns), m)
        if lit is None:
            hits.append((span, None, None, None))
            continue
        a, b = (m[i + rng[0]], m[i + rng[1] - 1] + 1) if rng is not None else (0, 0)
        tok = None if rng is None else _whole_token(text, a, b, style, bare_ten)
        if tok is not None:
            hits.append((span, tok[0], tok[1], [a, b]))
    return hits, seen


def _fail(loc: dict[str, Any], reason: str) -> M.VerifyResult:
    return M.VerifyResult("fail", reason, {**loc, "span": None, "literal": None, "token": None})


def _pass(loc: dict[str, Any], span: list[int] | None, literal: str | None,
          number: float | None = None, token: list[int] | None = None) -> M.VerifyResult:
    """*token*: raw offsets of the literal in the text it was matched in (the cell for a table
    locator, else the page text or column *span* indexes); the unit and qualifier checks read there."""
    return M.VerifyResult("pass", M.REASON_OK, {**loc, "span": span, "literal": literal, "token": token},
                          literal, span, number)


def _candidates(loc: dict[str, Any], page: dict[str, Any]) -> list[tuple[int | None, str]]:
    columns = [c for c in (page.get("columns") or []) if isinstance(c, str)]
    col = loc.get("column")
    if isinstance(col, int) and not isinstance(col, bool) and 0 <= col < len(columns):
        return [(col, columns[col])]
    return [(None, page.get("text") or "")] + list(enumerate(columns))


def _split_lines(columns: list[str]) -> tuple[set[str], set[str]]:
    return ({normalise(x)[0] for x in columns[0].split("\n") if x.strip()},
            {normalise(x)[0] for x in columns[1].split("\n") if x.strip()})


def _cut(line: str, left: set[str], right: set[str]) -> tuple[int, int] | None:
    """(raw end of the left part, raw start of the right part) when *line* is a merged line: one
    column's line followed by the other column's line on the same baseline; else None."""
    nl, m = normalise(line)
    for lp in left:
        if len(lp) < len(nl) and nl.startswith(lp + " ") and nl[len(lp) + 1:] in right:
            return m[len(lp) - 1] + 1, m[len(lp) + 1]
    return None


def _crosses(span: list[int], text: str, columns: list[str]) -> bool:
    """True when a page-text span joins the two columns: it covers the boundary of a merged line, or
    runs over a line break while touching a merged line (two-column lines interleave the columns)."""
    left, right = _split_lines(columns)
    ls, cuts = text.rfind("\n", 0, span[0]) + 1, []
    while ls < span[1]:
        le = text.find("\n", ls)
        le = len(text) if le < 0 else le
        cut = _cut(text[ls:le], left, right)
        cuts.append(None if cut is None else (ls + cut[0], ls + cut[1]))
        ls = le + 1
    if len(cuts) > 1:
        return any(c is not None for c in cuts)
    return cuts[0] is not None and span[0] < cuts[0][0] and span[1] > cuts[0][1]


def _verify_text(loc: dict[str, Any], page: dict[str, Any], snippet: str, lit: tuple[int, int] | None,
                 style: str) -> M.VerifyResult:
    seen_any, crossed = False, False
    columns = [c for c in (page.get("columns") or []) if isinstance(c, str)]
    for column, text in _candidates(loc, page):
        hits, seen = _hits(snippet, lit, text, style, bare_ten_unresolved(page))
        seen_any = seen_any or seen
        if column is None and len(columns) == 2:
            kept = [h for h in hits if not _crosses(h[0], text, columns)]
            crossed, hits = crossed or len(kept) < len(hits), kept
        if hits:
            span, literal, number, token = hits[0]
            return _pass({**loc, "column": column}, span, literal, number, token)
    if crossed:
        return _fail(loc, M.REASON_CROSSES_COLUMNS)
    return _fail(loc, M.REASON_PARTIAL_NUMBER if seen_any else M.REASON_SNIPPET_NOT_ON_PAGE)


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


def _row_span(snippet: str, lit: tuple[int, int] | None, text: str, style: str,
              anchors: list[str], row_head: str, bare_ten: bool) -> list[int] | None:
    """Page-text span of the cited cell: the occurrence whose text line holds the most OTHER cells of
    the same table row (at least one), a line starting with the row's first cell winning a tie
    (prose can repeat a row's label and unit); first wins any remaining tie. None (highlight the
    rendered cell only) when no line holds one -- never the first match elsewhere on the page."""
    keys = [k for k in (normalise(a)[0] for a in anchors) if k]
    lead = normalise(row_head)[0]
    best: tuple[tuple[int, bool], list[int] | None] = ((0, False), None)
    for span, *_ in (_hits(snippet, lit, text, style, bare_ten)[0] if keys else []):
        ls = text.rfind("\n", 0, span[0]) + 1
        le = text.find("\n", span[1])
        line = normalise(text[ls:len(text) if le < 0 else le])[0]
        score = (sum(1 for k in keys if k in line), bool(lead) and line.startswith(lead))
        if score[0] and score > best[0]:
            best = (score, span)
    return best[1]


def _row_fragment(literal: str, text: str, anchors: list[str], style: str) -> bool:
    """True when a page-text line holding another cell of the row prints the cell's literal only as
    PART of a longer number ('2' of a row line reading '281'): a ruling split the number."""
    want = normalise(literal)[0]
    keys = [k for k in (normalise(a)[0] for a in anchors) if k]
    for line in text.split("\n"):
        if not any(k in normalise(line)[0] for k in keys):
            continue
        toks = [normalise(line[s:e])[0] for s, e, *_ in number_tokens(line, style, strict=True)]
        if want not in toks and any(want in t and digits_only(want) != digits_only(t) for t in toks):
            return True
    return False


def _verify_table(loc: dict[str, Any], page: dict[str, Any], snippet: str, lit: tuple[int, int] | None,
                  style: str) -> M.VerifyResult:
    found = _cell_text(loc.get("cell"), page)
    if found is None:
        return _fail(loc, M.REASON_TABLE_CELL_MISMATCH)
    cell_text, table = found
    bare_ten = bare_ten_unresolved(page)
    hits, seen = _hits(snippet, lit, cell_text, style, bare_ten)
    if not hits:
        return _fail(loc, M.REASON_PARTIAL_NUMBER if seen and lit is not None else M.REASON_TABLE_CELL_MISMATCH)
    _, literal, number, token = hits[0]
    cell = {**loc["cell"], "strategy": table.get("strategy")}
    r, c = loc["cell"]["row"], loc["cell"]["col"]
    row = (table.get("rows") or [])[r]
    anchors = [str(x or "") for i, x in enumerate(row) if i != c]
    span = _row_span(snippet, lit, page.get("text") or "", style, anchors, str(row[0] or "") if row else "",
                     bare_ten)
    if span is None and literal and _row_fragment(literal, page.get("text") or "", anchors, style):
        return _fail(loc, M.REASON_PARTIAL_NUMBER)
    return _pass({**loc, "cell": cell}, span, literal, number, token)


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
    lit: tuple[int, int] | None = None
    if field_type != "text":
        lit, reason = _number_check(value, snippet, style, field_type == "integer", bare_ten_unresolved(page))
        if reason != M.REASON_OK:
            return _fail(loc, reason)
    kind = loc.get("kind") or "text"
    if kind == "table":
        return _verify_table(loc, page, snippet, lit, style)
    if kind != "text":
        return _fail(loc, M.REASON_SNIPPET_NOT_ON_PAGE)
    return _verify_text(loc, page, snippet, lit, style)
