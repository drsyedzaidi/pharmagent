"""Checks on a VERIFIED number that read the page around the matched token (locator['token']):

- censored_value (error): the token is qualified on the page ('<0.5', '≤ 1.0', '>4', '~2', 'below
  0.5', 'BLQ 0.5'): a bound or an approximation is never an exact value.
- scaled_header (error): the column header, row label, unit column or the proposed unit carries a
  multiplier ('AUC ×10³ (ng·h/mL)', 'x1000', '/1000', '(thousands)'): the cell is not the value.
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

import re
from functools import lru_cache
from typing import Any

from app.evidence import models as M
from app.evidence.normalize import decimal_evidence
from app.evidence.numgrammar import (
    GAP,
    Tok,
    is_data_cell,
    scale_shapes,
    tokens,
    words_after,
    words_before,
)
from app.evidence.qc_flags import is_interval_field

# a word BEFORE the token that makes it a bound or an approximation ('below 12.4', 'BLQ 12.4', 'approx. 12.4')
_QUALIFIER_WORDS = frozenset({"below", "above", "under", "over", "approximately", "approx", "about", "circa", "ca",
                              "blq", "bloq", "lloq", "uloq"})
# the only two-word look-back: 'less than', 'more than', 'greater than', 'up to', 'at least', 'at most'
_BOUND_WORDS = frozenset({"than", "to", "least", "most"})
_BOUND_LEADS = frozenset({"less", "more", "greater", "up", "at"})
_TRAIL = ".,;:"
DECISIVE_MIN, DECISIVE_RATIO = 3, 3
_DIGITS = re.compile("[0-9]+")
_DIGITS_SET = frozenset("0123456789")


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


@lru_cache(maxsize=16)
def _read(text: str, style: str, mode: str) -> tuple[Tok, ...]:
    """tokens() of a page text or cell, kept for the next check of a record on the same page (a record is read by
    four checks and a page holds many records)."""
    return tuple(tokens(text, style, mode=mode))


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
    return text, a, b, next((t for t in _read(text, style, _mode(loc)) if (t.start, t.end) == (a, b)), None)


def _numeric_word(word: str, style: str) -> bool:
    """A word that starts with a resolved number ('15', '15.4', '(15'): the other end of 'to'."""
    toks = tokens(word.rstrip(_TRAIL + ")"), style, mode="cell")
    return bool(toks) and toks[0].start <= 1 and toks[0].value is not None


def _own_prefix(text: str, a: int) -> str:
    """The characters of the token's own word before the token ('<' of '<0.5')."""
    k = a
    while k > 0 and text[k - 1] not in GAP:
        k -= 1
    return text[k:a]


def _qualifier(tok: Tok, text: str, a: int, prev: list[str]) -> str:
    """What stands before the token and makes it a bound or an approximation ('<', 'below', 'less than'), '' when
    nothing does. *prev* holds the two whole words before the token's word, in reading order."""
    w1 = prev[-1].rstrip(_TRAIL).casefold() if prev else ""
    w2 = prev[0].casefold() if len(prev) == 2 else ""
    if tok.qualified:
        return _own_prefix(text, a) or (prev[-1] if prev else "")
    if w1 in _BOUND_WORDS and w2 in _BOUND_LEADS:
        return " ".join(prev)
    return prev[-1] if w1 in _QUALIFIER_WORDS else ""


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
    prev, nxt = words_before(text, a, 2), words_after(text, b, 2)
    q = _qualifier(tok, text, a, prev)
    if q:
        return [_flag(M.FLAG_CENSORED_VALUE, "error", f"{field}: the page prints {q!r} before {literal!r}; a bound "
                      "or approximation is not an exact value", p)]
    w1 = prev[-1].rstrip(_TRAIL).casefold() if prev else ""
    to_before = w1 == "to" and len(prev) == 2 and _numeric_word(prev[0], style)
    to_after = bool(nxt) and nxt[0].rstrip(_TRAIL).casefold() == "to" and len(nxt) == 2 and _numeric_word(nxt[1], style)
    # a dash that touches only ONE of the two numbers is a range as well ('12.5 -15.0', '5.0- 12.4')
    dash = bool(nxt) and len(nxt[0]) > 1 and nxt[0][0] == "-" and nxt[0][1] in _DIGITS_SET \
        or bool(prev) and len(prev[-1]) > 1 and prev[-1][-1] == "-" and prev[-1][-2] in _DIGITS_SET
    if is_interval_field(str(field or "")) or not (tok.ranged or to_before or to_after or dash):
        return []
    return [_flag(M.FLAG_CENSORED_VALUE, "error", f"{field}: {literal!r} is one end of a printed range; a "
                  "range bound is not an exact value (use an interval field for it)", p)]


def scaled_flags(p: dict[str, Any], page: dict[str, Any]) -> list[dict[str, Any]]:
    loc = p.get("locator") or {}
    labels = [str(p.get("unit") or "")]
    if loc.get("kind") == "table":
        cell = loc.get("cell") or {}
        try:
            rows = (page.get("tables") or [])[cell["table"]]["rows"]
            header, row, col = rows[0], rows[cell["row"]], cell["col"]
            block = [h for h in rows[:cell["row"]] if not any(is_data_cell(str(x or "").strip())
                                                              for x in h[1:])] or [header]
            labels += [str(h[col] or "") for h in block if col < len(h)] + [str(row[0] or "") if row else ""]
            labels += [str(row[i] or "") for i, h in enumerate(header)
                       if i < len(row) and str(h or "").strip().casefold() in ("unit", "units")]
        except (KeyError, IndexError, TypeError):
            pass
    hit = next((lab for lab in labels if scale_shapes(lab)), None)
    if hit is None:
        return []
    return [_flag(M.FLAG_SCALED_HEADER, "error", f"{p['field']}: {hit!r} carries a multiplier; the printed "
                  "number is not the value until it is scaled", p)]


def document_evidence(doc: dict[str, Any]) -> tuple[int, int]:
    """(decimal-comma, decimal-point) evidence over the document's page text (what its style was
    inferred from)."""
    return decimal_evidence("\n".join(str(pg.get("text") or "") for pg in doc.get("pages") or []))


def _bare_group(word: str, lo: int, hi: int) -> bool:
    """A word of nothing but lo..hi ASCII digits: a digit group a text layer may have split off a number."""
    return lo <= len(word) <= hi and _DIGITS.fullmatch(word) is not None


def grouped_flags(p: dict[str, Any], page: dict[str, Any]) -> list[dict[str, Any]]:
    """Error when the token is one piece of digit groups the reader splits ('12' of '12 145'): the source may
    print ONE grouped number. The token and the word beside it are bare digit runs ('145' after '12', '12' before
    '145'); a decimal, a bracket or a sign beside it never makes a group."""
    loc = p.get("locator") or {}
    got = verified_token(loc, page)
    if got is None or got[3] is None:
        return []
    text, a, b, _ = got
    own, before, after = text[a:b], words_before(text, a, 1), words_after(text, b, 1)
    head = bool(after) and _bare_group(own, 1, 3) and _bare_group(after[0], 2, 3)
    tail = bool(before) and _bare_group(own, 2, 3) and _bare_group(before[0], 1, 3)
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
    return None if ctx is None else sum(1 for t in _read(ctx[0], style, _mode(loc)) if t.value is not None)


def decimal_flags(p: dict[str, Any], page: dict[str, Any], doc: dict[str, Any],
                  spec: dict[str, Any] | None) -> list[dict[str, Any]]:
    loc = p.get("locator") or {}
    literal, style = str(loc.get("literal") or ""), loc.get("decimal_style") or "."
    if (spec or {}).get("decimal") in (".", ","):
        return []
    got = verified_token(loc, page)
    shape = got[3].shape if got is not None and got[3] is not None and got[3].value is not None else ""
    evidence = doc.get("decimal_evidence")
    comma, point = evidence if isinstance(evidence, tuple) else document_evidence(doc)
    if shape == "foreign_decimal":
        return [_flag(M.FLAG_DECIMAL_STYLE_AMBIGUOUS, "error", f"{p['field']}: {literal!r} uses the other decimal "
                      f"separator than this document ({style!r}): a list or a foreign decimal; set the field's "
                      "decimal style in the template or read the value on the PDF", p)]
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
