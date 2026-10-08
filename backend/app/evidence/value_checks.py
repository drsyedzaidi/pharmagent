"""Checks on a VERIFIED number that read the page around the matched token (locator['token']):

- censored_value (error): the token is qualified on the page ('<0.5', '≤(1.0)', '>4', '~=2', 'below
  0.5', 'BLQ 0.5', 'less than or equal to 3') or is one end of a range ('12 -15', '5.0- 12.4'): a bound or an
  approximation is never an exact value. For a table cell the cells right before and after it are read too.
- scaled_header (error): the column header, row label, unit column, the proposed unit or a cell right
  beside the number carries a multiplier ('AUC ×10³ (ng·h/mL)', 'x1000', '/1000', '(thousands)'), or a
  label number touches a character the reader cannot read: the cell is not the value.
- decimal_style_ambiguous (error): the token's reading depends on the page's decimal style (one
  separator followed by exactly three digits, '3,412') and the document's style evidence is weak
  (fewer than 3 decimals of the winning style, or the losing style at least a third of them),
  unless the template fixes the field's decimal style. The evidence is counted once per batch
  (check_proposals passes it as doc['decimal_evidence'], a (comma, point) tuple).
- glyph_text_mismatch (error) / glyph_unverified (warn) / overlapping_glyphs (error) /
  glyph_ink_mismatch (error): the literal holds a character whose drawn glyph the parse could not
  tie to the text (parse_fonts.py, parse_ink.py, parse_marks.py).

A pass without a token (a record verified before tokens were stored) reads nothing and raises none
of the token checks; every extraction and re-verification stores one.
"""
from __future__ import annotations

from typing import Any

from app.evidence import gaps
from app.evidence import models as M
from app.evidence.normalize import decimal_evidence
from app.evidence.numgrammar import (
    GAP,
    Tok,
    digit_group,
    is_data_cell,
    loose_in,
    scale_shapes,
    tokens,
    words_after,
    words_before,
)
from app.evidence.numgrammar import words as W
from app.evidence.numgrammar.alphabet import CLOSERS_BR, QUALIFIERS, RANGE_WORDS, TRAILING
from app.evidence.numgrammar.units import UNIT_RE
from app.evidence.qc_flags import is_interval_field
from app.evidence.read_cache import cached, read
from app.evidence.word_lists import (
    AFTER_SCALE_WORDS,
    BOUND_LEADS,
    BOUND_WORDS,
    EQUAL,
    OR,
    QUALIFIER_WORDS,
    SCALE_SYMBOLS,
    TO,
)

# the word lists of these checks (qualifier, bound and scale words) are in word_lists.py: every one is matched with
# ``loose_in``, so a look-alike spelling ('αbove') raises the flag the plain one does
_QUALIFIER_WORDS, _BOUND_WORDS, _BOUND_LEADS = QUALIFIER_WORDS, BOUND_WORDS, BOUND_LEADS
_TO, _OR, _EQUAL = TO, OR, EQUAL
_TRAIL = "".join(sorted(TRAILING - CLOSERS_BR - {"'"}))
DECISIVE_MIN, DECISIVE_RATIO = 3, 3
LOOK_BACK = 4                      # words before the token the qualifier look-back reads ('less than or equal to')


def _flag(code: str, severity: str, message: str, p: dict[str, Any]) -> dict[str, Any]:
    return {"code": code, "severity": severity, "message": message, "field": p["field"],
            "group": p.get("group_label") or "", "origin": "script"}


def token_context(loc: dict[str, Any], page: dict[str, Any]) -> tuple[str, int, int] | None:
    """(text, start, end) of the verified token: the cited cell for a table locator, else the page
    text or the column the span indexes; None when the locator has no usable token."""
    token = loc.get("token")
    if not (isinstance(token, list) and len(token) == 2 and all(isinstance(t, int) for t in token)):
        return None
    if loc.get("kind") == "table":
        cell = loc.get("cell") or {}
        try:
            text = str((page.get("tables") or [])[cell["table"]]["rows"][cell["row"]][cell["col"]] or "")
        except (KeyError, IndexError, TypeError):
            return None
    else:
        column, columns = loc.get("column"), page.get("columns") or []
        text = str(columns[column] if isinstance(column, int) and 0 <= column < len(columns)
                   else page.get("text") or "")
    a, b = token
    return (text, a, b) if 0 <= a < b <= len(text) else None


def _mode(loc: dict[str, Any]) -> str:
    """How the verified token's text is read: a table cell is one line, page text splits at line feeds."""
    return "cell" if loc.get("kind") == "table" else "page"


def verified_token(loc: dict[str, Any], page: dict[str, Any]) -> tuple[str, int, int, Tok | None] | None:
    """(text, start, end, the token the verifier matched there or None) from the locator's stored token."""
    ctx = token_context(loc, page)
    if ctx is None:
        return None
    text, a, b = ctx
    style = loc.get("decimal_style") or "."
    return text, a, b, next((t for t in read(text, style, _mode(loc)).toks if (t.start, t.end) == (a, b)), None)


def cell_row(loc: dict[str, Any], page: dict[str, Any]) -> tuple[list[Any], int] | None:
    """(the row of the cited table cell, its column), None for a text locator or an unusable cell locator."""
    if loc.get("kind") != "table":
        return None
    cell = loc.get("cell") or {}
    try:
        return (page.get("tables") or [])[cell["table"]]["rows"][cell["row"]], cell["col"]
    except (KeyError, IndexError, TypeError):
        return None


def _context(loc: dict[str, Any], page: dict[str, Any], text: str) -> tuple[str, int]:
    """(the text the words around the token are read in, the offset of *text* in it). A table cell is one line, but the
    cells right before and after it are the words one separator away: their qualifier, sign or range word counts."""
    found = cell_row(loc, page)
    if found is None:
        return text, 0
    row, col = found
    before = str(row[col - 1] or "") if 0 < col <= len(row) else ""
    after = str(row[col + 1] or "") if 0 <= col + 1 < len(row) else ""
    return f"{before} {text} {after}", len(before) + 1


def _numeric_word(word: str, style: str) -> bool:
    """A word that starts with a resolved number ('15', '15.4', '(15'): the other end of 'to'."""
    toks = tokens(word.rstrip(_TRAIL + ")"), style, mode="cell")
    return bool(toks) and toks[0].start <= 1 and toks[0].value is not None


def _own_word(text: str, a: int) -> str:
    """The whole word of *text* that holds the offset *a*."""
    k, e = a, a
    while k > 0 and text[k - 1] not in GAP:
        k -= 1
    while e < len(text) and text[e] not in GAP:
        e += 1
    return text[k:e]


def _own_prefix(text: str, a: int) -> str:
    """The characters of the token's own word before the token ('<' of '<0.5')."""
    k = a
    while k > 0 and text[k - 1] not in GAP:
        k -= 1
    return text[k:a]


def _qualifier(tok: Tok, text: str, a: int, prev: list[str]) -> str:
    """What stands before the token and makes it a bound or an approximation ('<', 'below', 'less than', 'less than
    or equal to'), '' when nothing does. *prev* holds up to four whole words before the token's word, in reading
    order. Every word list is matched with ``loose_in``: a look-alike spelling flags like the plain one."""
    w = [x.rstrip(_TRAIL) for x in prev]
    w1, w2, w3 = (w[-1] if w else ""), (w[-2] if len(w) > 1 else ""), (w[-3] if len(w) > 2 else "")
    if tok.qualified:
        return _own_prefix(text, a) or next((x for x in reversed(prev) if not QUALIFIERS.isdisjoint(x)),
                                            prev[-1] if prev else "")
    if W.qualifies(prev):
        return prev[-1]
    if loose_in(w1, _BOUND_WORDS) and loose_in(w2, _BOUND_LEADS):
        return " ".join(prev[-2:])
    if loose_in(w1, _TO) and loose_in(w2, _EQUAL) and loose_in(w3, _OR):
        return " ".join(prev[-3:])
    return prev[-1] if loose_in(w1, _QUALIFIER_WORDS) else ""


def censored_flags(p: dict[str, Any], page: dict[str, Any]) -> list[dict[str, Any]]:
    loc = p.get("locator") or {}
    got = verified_token(loc, page)
    if got is None:
        return []
    text, a, b, tok = got
    literal, field = loc.get("literal"), p["field"]
    if tok is None or tok.value is None:         # the stored token is not one readable number any more
        return [_flag(M.FLAG_CENSORED_VALUE, "error", f"{field}: {literal!r} is not one readable number next to "
                      "its neighbours, so a qualifier or a range bound cannot be ruled out; read the value on the "
                      "PDF", p)]
    style = loc.get("decimal_style") or "."
    ctx, off = _context(loc, page, text)
    prev, nxt = words_before(ctx, a + off, LOOK_BACK), words_after(ctx, b + off, 2)
    q = _qualifier(tok, ctx, a + off, prev)
    if q:
        return [_flag(M.FLAG_CENSORED_VALUE, "error", f"{field}: the page prints {q!r} before {literal!r}; a bound "
                      "or approximation is not an exact value", p)]
    to_before = bool(prev) and loose_in(prev[-1].rstrip(_TRAIL), _TO) and len(prev) > 1 \
        and _numeric_word(prev[-2], style)
    to_after = bool(nxt) and loose_in(nxt[0].rstrip(_TRAIL), _TO) and len(nxt) == 2 and _numeric_word(nxt[1], style)
    # the tokenizer marks a dash glued to the number, a lone '-' or '~' word before or after it; a lone one in the
    # neighbouring table cell is read here, and so is a dash glued to the end of the previous cell after its number
    # (r7 writer 3: the cells ['5.0-', '12.4'] are one range; the cell '5.0-' itself is unresolved: no right end)
    glued = bool(prev) and prev[-1].endswith("-") and W.ends_with_number(prev[-1][:-1]) \
        and W.digit_led(_own_word(ctx, a + off))
    ranged = tok.ranged or glued or bool(prev) and prev[-1] in RANGE_WORDS or bool(nxt) and nxt[0] in RANGE_WORDS
    if is_interval_field(str(field or "")) or not (ranged or to_before or to_after):
        return []
    return [_flag(M.FLAG_CENSORED_VALUE, "error", f"{field}: {literal!r} is one end of a printed range; a "
                  "range bound is not an exact value (use an interval field for it)", p)]


def _header_rows(rows: list[Any]) -> list[int]:
    """Indexes of the rows of a table that hold no data cell after their first cell (header rows), once per request."""
    return cached(("header rows", id(rows)),
                  lambda: [i for i, h in enumerate(rows)
                           if not any(is_data_cell(gaps.strip(str(x or ""))) for x in h[1:])],
                  sum(len(r) for r in rows))


def scale_after(loc: dict[str, Any], page: dict[str, Any]) -> str:
    """The scale printed right AFTER the verified token, '' when none: a glued bare capital M (``1.5M``: the unit
    production reads it as molar, but it may be millions), a scale word (``1.5 million``, ``1.5 bn``; a look-alike
    spelling
    counts) or a scale symbol (``1.5 k``, ``1.5 M``) as the next word (r7 round 2)."""
    got = verified_token(loc, page)
    if got is None or got[3] is None:
        return ""
    text, _, b, _ = got
    e = b
    while e < len(text) and text[e] not in GAP:
        e += 1
    glued = text[b:e]                                   # the rest of the token's own word ('M' of '1.5M')
    m = UNIT_RE.match(glued)
    if m is not None and glued[:m.end()] in SCALE_SYMBOLS:
        return glued[:m.end()]
    nxt = words_after(text, b, 1)
    word = nxt[0].rstrip(_TRAIL + "".join(sorted(CLOSERS_BR))) if nxt else ""
    return word if word in SCALE_SYMBOLS or (word and loose_in(word, AFTER_SCALE_WORDS)) else ""


def scaled_flags(p: dict[str, Any], page: dict[str, Any]) -> list[dict[str, Any]]:
    loc = p.get("locator") or {}
    after = scale_after(loc, page)
    if after:
        return [_flag(M.FLAG_SCALED_HEADER, "error", f"{p['field']}: the page prints {after!r} right after the number: "
                      "the printed number is not the value until it is scaled; compare with the page image", p)]
    labels = [str(p.get("unit") or "")]
    if loc.get("kind") == "table":
        cell = loc.get("cell") or {}
        try:
            rows = (page.get("tables") or [])[cell["table"]]["rows"]
            header, row, col = rows[0], rows[cell["row"]], cell["col"]
            block = [rows[i] for i in _header_rows(rows) if i < cell["row"]] or [header]
            labels += [str(h[col] or "") for h in block if col < len(h)] + [str(row[0] or "") if row else ""]
            labels += [str(row[i] or "") for i, h in enumerate(header)
                       if i < len(row) and gaps.strip(str(h or "")).casefold() in ("unit", "units")]
            labels += [str(row[k] or "") for k in (col - 1, col + 1) if 0 <= k < len(row)]   # a scale cell beside it
        except (KeyError, IndexError, TypeError):
            pass
    hit = next((lab for lab in labels if scale_shapes(lab)), None)
    if hit is None:
        return []
    return [_flag(M.FLAG_SCALED_HEADER, "error", f"{p['field']}: {hit!r} carries a multiplier, or holds a character "
                  "the reader cannot read next to a number; the printed number is not the value until it is scaled: "
                  "compare with the page image", p)]


def document_evidence(doc: dict[str, Any]) -> tuple[int, int]:
    """(decimal-comma, decimal-point) evidence over the document's page text (what its style was
    inferred from)."""
    return decimal_evidence("\n".join(str(pg.get("text") or "") for pg in doc.get("pages") or []))


def grouped_flags(p: dict[str, Any], page: dict[str, Any]) -> list[dict[str, Any]]:
    """Error when the token is one piece of digit groups the reader splits ('12' of '12 145', '1' and '234.5' of
    '1 234.5', '-1' of '-1 234'): the source may print ONE grouped number. The token is a head of 1-3 digits and the
    word beside it a group of 3 digits (2-3 when bare) that may carry a decimal tail and trailing punctuation, or the
    token is that group and the word before it the head; an opener, a sign or a closing mark never hides the group, a
    decimal on the head ('12.4 145') never makes one."""
    loc = p.get("locator") or {}
    got = verified_token(loc, page)
    if got is None or got[3] is None:
        return []
    text, a, b, _ = got
    own, before, after = text[a:b], words_before(text, a, 1), words_after(text, b, 1)
    head = bool(after) and digit_group(own, 1, 3, lead=True) and (
        digit_group(after[0], 2, 3) or digit_group(after[0], 3, 3, decimals=True))
    tail = bool(before) and digit_group(before[0], 1, 3, lead=True) and (
        digit_group(own, 2, 3) or digit_group(own, 3, 3, decimals=True))
    if not (head or tail):
        return []
    shown = f"{own} {after[0]}" if head else f"{before[0]} {own}"
    return [_flag(M.FLAG_COMPOSITE_CELL, "error", f"{p['field']}: {shown!r} may be one number grouped by a space; "
                  "read the value on the PDF", p)]


def cell_numbers(p: dict[str, Any], doc: dict[str, Any] | None) -> int | None:
    """How many numbers the cited cell / text holds (resolved tokens), None without a token."""
    pages, loc = (doc or {}).get("pages") or [], p.get("locator") or {}
    n = loc.get("page")
    if not (isinstance(n, int) and 1 <= n <= len(pages)):
        return None
    ctx = token_context(loc, pages[n - 1])
    style = loc.get("decimal_style") or "."
    return None if ctx is None else sum(1 for t in read(ctx[0], style, _mode(loc)).toks if t.value is not None)


def decimal_flags(p: dict[str, Any], page: dict[str, Any], doc: dict[str, Any],
                  spec: dict[str, Any] | None) -> list[dict[str, Any]]:
    loc = p.get("locator") or {}
    literal, style = str(loc.get("literal") or ""), loc.get("decimal_style") or "."
    if (spec or {}).get("decimal") in (".", ","):
        return []
    got = verified_token(loc, page)
    tok = got[3] if got is not None and got[3] is not None and got[3].value is not None else None
    shape = tok.shape if tok is not None else ""
    evidence = doc.get("decimal_evidence")
    comma, point = evidence if isinstance(evidence, tuple) else document_evidence(doc)
    if shape == "foreign_decimal" or (shape == "sci" and tok.decimal not in ("", style)):
        return [_flag(M.FLAG_DECIMAL_STYLE_AMBIGUOUS, "error", f"{p['field']}: {literal!r} uses the other decimal "
                      f"separator than this document ({style!r}): a list or a foreign decimal; set the field's "
                      "decimal style in the template or read the value on the PDF", p)]
    if shape == "grouped" and style in literal:
        return [_flag(M.FLAG_DECIMAL_STYLE_AMBIGUOUS, "error", f"{p['field']}: {literal!r} groups its digits with "
                      f"the character this document ({style!r}) prints its decimals with: a foreign-locale "
                      "thousands number, a version or a list; set the field's decimal style in the template or read "
                      "the value on the PDF", p)]
    if shape == "middot" and point >= DECISIVE_MIN:
        return [_flag(M.FLAG_DECIMAL_STYLE_AMBIGUOUS, "error", f"{p['field']}: {literal!r} puts a raised dot "
                      f"between digits in a document that prints its decimals with a point ({point} of them): it may "
                      "be a product, not a decimal; set the field's decimal style in the template or read the "
                      "value on the PDF", p)]
    if shape != "style_dependent":
        return []
    win, lose = (comma, point) if style == "," else (point, comma)
    if win >= DECISIVE_MIN and DECISIVE_RATIO * lose <= win:
        return []
    return [_flag(M.FLAG_DECIMAL_STYLE_AMBIGUOUS, "error", f"{p['field']}: {literal!r} reads differently with a "
                  f"decimal point or comma, and the document's style evidence is weak ({win} vs {lose}); set the "
                  "field's decimal style in the template", p)]


_GLYPH_MESSAGES = {
    M.FLAG_GLYPH_MISMATCH: ("error", "the text layer names {chars} for a glyph that may draw something else "
                                     "(font remap, Type3 program, /ActualText, rotated or mirrored glyph); "
                                     "read the value on the PDF"),
    M.FLAG_GLYPH_INK: ("error", "the ink drawn for {chars} is not the plain glyph (partly clipped or covered, "
                                "or non-text marks on or between the digits); read the value on the PDF"),
    M.FLAG_GLYPH_UNVERIFIED: ("warn", "the drawn glyph of {chars} could not be tied to the text; check the "
                                      "value on the PDF"),
    M.FLAG_OVERLAPPING_GLYPHS: ("error", "{chars} is over-struck by another glyph on this page, so the text "
                                         "may hold a character a reader does not see; read the value on the PDF"),
}


def glyph_flags(p: dict[str, Any], page: dict[str, Any]) -> list[dict[str, Any]]:
    check = page.get("glyph_check") or {}
    shown = str((p.get("locator") or {}).get("literal") or p.get("value") or "")
    out = []
    for code, (severity, message) in _GLYPH_MESSAGES.items():
        hit = sorted(set(shown) & set(str(check.get(code) or "")))
        if hit:
            out.append(_flag(code, severity, f"{p['field']}: " + message.format(chars=" ".join(map(repr, hit))), p))
    return out


def value_flags(p: dict[str, Any], page: dict[str, Any], doc: dict[str, Any],
                spec: dict[str, Any] | None) -> list[dict[str, Any]]:
    """All checks above for one passed proposal (none for a failed one)."""
    if p.get("verify_status") != "pass":
        return []
    numeric = p.get("value_num") is not None
    out = glyph_flags(p, page)
    if numeric:
        out = (censored_flags(p, page) + scaled_flags(p, page) + decimal_flags(p, page, doc, spec)
               + grouped_flags(p, page) + out)
    return out
