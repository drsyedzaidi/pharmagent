"""Literal-match verifier: a proposed value passes only when a deterministic check finds the
literal number inside the cited snippet AND the snippet inside the cited page (or column /
table cell). Offsets in the returned locator are RAW page-text offsets (raw[start:end] round-trips).

Check order: empty_snippet, snippet_too_long, page_out_of_range, readability (a character outside the
allowed number alphabet, or a pdfminer ``(cid:N)`` marker, in the snippet or the value: unresolved_context),
number (unresolved_context / unresolved_shape / no_number / number_mismatch / integer_expected / partial_number),
location (snippet_not_on_page for text, table_cell_mismatch for table; for a table cell also unresolved_context when the
cell before or after it holds a character outside the allowed alphabet; partial_number when the snippet is there
but its number is only part of a longer page number or expression: '2.4' in '12.4', '20' in '-20', '234' in
'1,234'; unresolved_context / unresolved_shape when the page prints that very number in a form the grammar
cannot read). A claim and a page token are the same number only when their canonical decimals are equal (no tolerance).
A pass sets span, literal (the WHOLE page / cell token) and number (that token's value in the
page's decimal style, which is what value_num must be), and (when used) column. A table record's span is the
cited cell's row in the page text, or None when the row cannot be found there. Numbers are read ONLY by
``numgrammar`` (an explicit allowed alphabet; every other character in or next to a number makes it unresolved);
the locator below (``normalise``) finds text, it never yields a value.
"""
from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterator
from decimal import Decimal
from typing import Any

from app.evidence import models as M
from app.evidence.normalize import IdxMap, normalise
from app.evidence.numgrammar import Tok, claim_decimal, digits_of, tokens, unreadable
from app.evidence.read_cache import read
from app.evidence.verify_columns import ColumnLines
from app.evidence.verify_messages import context_sentence, tok_sentence, unreadable_sentence

MAX_SNIPPET_CHARS = 300


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


def find_number_in_snippet(value: str, snippet: str, decimal_style: str, integer: bool) -> str | None:
    """The literal token in *snippet* equal to *value* (parsed with *decimal_style*), else None. (Test-only.)"""
    span, reason, _ = _number_check(value, snippet, decimal_style, integer)
    return snippet[span[0]:span[1]] if span is not None and reason == M.REASON_OK else None


def _targets(value: str, style: str) -> tuple[list[Decimal], str, str]:
    """(the exact decimals the proposal names, reason, detail): exactly ONE resolved number, in the page's style and
    (proposed values usually arrive in canonical point-decimal form: '18.2' for a page that prints '18,2') in point
    style. A number is the same number only when its canonical decimal is equal: no tolerance, so 3.2e-13 is never 0
    and a changed last digit of a long integer is never the page's number."""
    toks = tokens(value, style, mode="claim")
    if len(toks) != 1:
        return [], M.REASON_NO_NUMBER, ""
    if toks[0].value is None:
        return [], toks[0].reason, tok_sentence(toks[0], value)
    read_style = toks[0].decimal_value
    point = claim_decimal(value, ".")
    return [read_style] + ([point] if point is not None and point != read_style else []), "", ""


def _number_check(value: str, snippet: str, decimal_style: str, integer: bool
                  ) -> tuple[tuple[int, int] | None, str, str]:
    """(raw [start, end) of the snippet token equal to *value*, reason, reviewer detail). A value equal only to a
    part of an expression the grammar cannot read (the mantissa of '1.2e-3', the 12 of '12 3/4') is a partial
    number. The stored number is the PAGE token's value, never this one."""
    targets, reason, detail = _targets(value, decimal_style)
    if not targets:
        return None, reason, detail
    if integer and targets[0] != targets[0].to_integral_value():
        return None, M.REASON_INTEGER_EXPECTED, ""
    toks = tokens(snippet, decimal_style, mode="claim")
    for t in toks:
        if t.value is not None and t.decimal_value in targets:
            return (t.start, t.end), M.REASON_OK, ""
    floats = [float(x) for x in targets]
    vdig = digits_of(value)
    holding = [t for t in toks if any(p in floats for p in t.parts)]
    tainted = next((t for t in toks if t.reason == M.REASON_UNRESOLVED_CONTEXT
                    and (t in holding or (vdig and vdig in digits_of(snippet[t.start:t.end])))), None)
    if tainted is not None:                       # the cited number touches a character outside the alphabet
        return None, M.REASON_UNRESOLVED_CONTEXT, tok_sentence(tainted, snippet)
    if holding:                                   # ... or is only a part of an expression the grammar rejects
        return None, M.REASON_PARTIAL_NUMBER, ""
    # A fragment is a STRICT sub-run of a token's digits ('24' in '245.6'); equal digits with a
    # different reading ('3.412' vs '3,412') are a separator mismatch, i.e. number_mismatch.
    fragment = any(vdig and vdig != digits_of(snippet[t.start:t.end]) and vdig in digits_of(snippet[t.start:t.end])
                   for t in toks)
    return None, (M.REASON_NO_NUMBER if not toks or fragment else M.REASON_NUMBER_MISMATCH), ""


def _norm_range(ms: IdxMap, lit: tuple[int, int]) -> tuple[int, int] | None:
    """Normalised [j, k) of the snippet's raw literal range *lit* (*ms*: the snippet's normalise map)."""
    idx = [i for i, r in enumerate(ms) if lit[0] <= r < lit[1]]
    return (idx[0], idx[-1] + 1) if idx else None


Hit = tuple[list[int], str | None, float | None, list[int] | None]   # raw span, literal, value, token


def _hits(snippet: str, lit: tuple[int, int] | None, text: str, style: str,
          mode: str = "page") -> tuple[list[Hit], bool, tuple[Tok, str, tuple[int, int]] | None]:
    """Every occurrence of *snippet* in *text* whose cited number (raw snippet range *lit*) is a WHOLE
    number token of *text* -- never a run inside '12.4', '-20' or '1,234' -- whether the snippet occurred at all,
    and the first (expression, text, cited range) where the page prints the cited number in a form the grammar
    cannot read: an unresolved expression with exactly that span, or an unresolved_context expression (a word that
    holds a character outside the alphabet) that contains it. Without *lit* (text fields) every occurrence is a
    hit. *mode*: 'page' for page text and columns, 'cell' for a table cell."""
    ns, sm = normalise(snippet)
    nt, m = normalise(text)
    rng = _norm_range(sm, lit) if lit is not None else None
    hits: list[Hit] = []
    seen, blocked, reading = False, None, None
    for i in _occurrences(ns, nt):
        seen = True
        span = _raw_span(i, len(ns), m)
        if lit is None:
            hits.append((span, None, None, None))
            continue
        if rng is None:
            continue
        if reading is None:
            reading = read(text, style, mode)
        a, b = m[i + rng[0]], m[i + rng[1] - 1] + 1
        k = bisect_right(reading.starts, a) - 1
        tok = reading.toks[k] if k >= 0 and reading.toks[k].end >= b else None   # holds text[a:b]
        if tok is None:
            continue
        if tok.value is not None:
            if (tok.start, tok.end) == (a, b):
                hits.append((span, text[a:b], tok.value, [a, b]))
        elif blocked is None and ((tok.start, tok.end) == (a, b) or tok.reason == M.REASON_UNRESOLVED_CONTEXT):
            blocked = (tok, text, (a, b))
    return hits, seen, blocked


def _fail(loc: dict[str, Any], reason: str, detail: str = "") -> M.VerifyResult:
    return M.VerifyResult("fail", reason, {**loc, "span": None, "literal": None, "token": None}, detail=detail)


def _blocked(loc: dict[str, Any], blocked: tuple[Tok, str, tuple[int, int]]) -> M.VerifyResult:
    """The page prints the cited number as an expression the grammar cannot read: its code and reviewer sentence
    (naming the cited number, which may be only a part of the word that holds the offending character)."""
    tok, text, (a, b) = blocked
    return _fail(loc, tok.reason, tok_sentence(tok._replace(start=a, end=b) if tok.reason == M.REASON_UNRESOLVED_CONTEXT
                                                else tok, text))


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


def _verify_text(loc: dict[str, Any], page: dict[str, Any], snippet: str, lit: tuple[int, int] | None,
                 style: str) -> M.VerifyResult:
    seen_any, crossed, blocked = False, False, None
    columns = [c for c in (page.get("columns") or []) if isinstance(c, str)]
    lines: ColumnLines | None = None
    for column, text in _candidates(loc, page):
        hits, seen, block = _hits(snippet, lit, text, style)
        seen_any, blocked = seen_any or seen, blocked or block
        if column is None and len(columns) == 2 and hits:
            lines = lines or ColumnLines(columns)            # the columns are read once, not once per occurrence
            kept = next((h for h in hits if not lines.crosses(h[0], text)), None)
            crossed = crossed or kept is None
            hits = [] if kept is None else [kept]
        if hits:
            span, literal, number, token = hits[0]
            return _pass({**loc, "column": column}, span, literal, number, token)
    if crossed:
        return _fail(loc, M.REASON_CROSSES_COLUMNS)
    if blocked is not None:
        return _blocked(loc, blocked)
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
              anchors: list[str], row_head: str) -> list[int] | None:
    """Page-text span of the cited cell: the occurrence whose text line holds the most OTHER cells of
    the same table row (at least one), a line starting with the row's first cell winning a tie
    (prose can repeat a row's label and unit); first wins any remaining tie. None (highlight the
    rendered cell only) when no line holds one -- never the first match elsewhere on the page."""
    keys = [k for k in (normalise(a)[0] for a in anchors) if k]
    lead = normalise(row_head)[0]
    best: tuple[tuple[int, bool], list[int] | None] = ((0, False), None)
    for span, *_ in (_hits(snippet, lit, text, style)[0] if keys else []):
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
        toks = [normalise(line[t.start:t.end])[0] for t in tokens(line, style, mode="page")]
        if want not in toks and any(want in t and digits_of(want) != digits_of(t) for t in toks):
            return True
    return False


def _neighbour_offence(row: list[Any], col: int, literal: str) -> str:
    """The reviewer sentence when the cell right before or after the cited one holds a character outside the allowed
    alphabet or a ``(cid:N)`` marker (a qualifier, a sign or a multiplier the reader cannot read), else ''. A cell is
    one line; its neighbours are the words one separator away, so a look-alike ``≲`` in the next cell leaves the
    number as unresolved as it does on a page line."""
    for k, where in ((col - 1, "in the previous cell"), (col + 1, "in the next cell")):
        if 0 <= k < len(row) and (bad := unreadable(str(row[k] or ""))) is not None:
            return context_sentence(literal, bad[1], where)
    return ""


def _verify_table(loc: dict[str, Any], page: dict[str, Any], snippet: str, lit: tuple[int, int] | None,
                  style: str) -> M.VerifyResult:
    found = _cell_text(loc.get("cell"), page)
    if found is None:
        return _fail(loc, M.REASON_TABLE_CELL_MISMATCH)
    cell_text, table = found
    hits, seen, blocked = _hits(snippet, lit, cell_text, style, "cell")
    if not hits:
        if blocked is not None:
            return _blocked(loc, blocked)
        return _fail(loc, M.REASON_PARTIAL_NUMBER if seen and lit is not None else M.REASON_TABLE_CELL_MISMATCH)
    _, literal, number, token = hits[0]
    cell = {**loc["cell"], "strategy": table.get("strategy")}
    r, c = loc["cell"]["row"], loc["cell"]["col"]
    row = (table.get("rows") or [])[r]
    if literal and (offence := _neighbour_offence(row, c, literal)):
        return _fail(loc, M.REASON_UNRESOLVED_CONTEXT, offence)
    anchors = [str(x or "") for i, x in enumerate(row) if i != c]
    span = _row_span(snippet, lit, page.get("text") or "", style, anchors, str(row[0] or "") if row else "")
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
    for what, text in (("snippet", snippet), ("value", value)):     # readability: nothing outside the alphabet
        if (bad := unreadable(text)) is not None:
            return _fail(loc, M.REASON_UNRESOLVED_CONTEXT, unreadable_sentence(what, bad[1]))
    lit: tuple[int, int] | None = None
    if field_type != "text":
        lit, reason, detail = _number_check(value, snippet, style, field_type == "integer")
        if reason != M.REASON_OK:
            return _fail(loc, reason, detail)
    kind = loc.get("kind") or "text"
    if kind == "table":
        return _verify_table(loc, page, snippet, lit, style)
    if kind != "text":
        return _fail(loc, M.REASON_SNIPPET_NOT_ON_PAGE)
    return _verify_text(loc, page, snippet, lit, style)
