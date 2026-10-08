"""Checks on a VERIFIED number that read the page around the matched token (locator['token']):

- censored_value (error): the token is a bound, an approximation or one end of a printed range
  (``value_bounds.py``, split out in r7 round 3).
- scaled_header (error): the number is not the value until it is scaled (``value_scale.py``, split out in r7 round
  3: a scale word or symbol after it, or a label or a data cell beside it that is no inert production).
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

from app.evidence import models as M
from app.evidence.normalize import decimal_evidence
from app.evidence.numgrammar import digit_group, words_after, words_before
from app.evidence.numgrammar.alphabet import CLOSERS_BR, COMMA, DECIMAL_MARKS, GROUP_ONLY, LETTERS, TRAILING
from app.evidence.numgrammar.equivalents import THIN_SOURCES
from app.evidence.numgrammar.units import UNIT_RE
from app.evidence.read_cache import read
from app.evidence.value_bounds import censored_flags
from app.evidence.value_context import flag, mode, token_context, verified_token
from app.evidence.value_scale import scaled_flags

DECISIVE_MIN, DECISIVE_RATIO = 3, 3
GROUP_WIDTH = 3                    # the digits of a thousands group
_TRAIL = "".join(sorted(TRAILING - CLOSERS_BR - GROUP_ONLY))
_CLOSE = "".join(sorted(CLOSERS_BR))


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
        digit_group(after[0], 2, 3) or digit_group(after[0], 3, 3, decimals=True)
        or (text[b:b + 1] in THIN_SOURCES and _letter_group(after[0])))
    tail = bool(before) and digit_group(before[0], 1, 3, lead=True) and (
        digit_group(own, 2, 3) or digit_group(own, 3, 3, decimals=True))
    if not (head or tail):
        return []
    shown = f"{own} {after[0]}" if head else f"{before[0]} {own}"
    return [flag(M.FLAG_COMPOSITE_CELL, "error", f"{p['field']}: {shown!r} may be one number grouped by a space; "
                  "read the value on the PDF", p)]


def _letter_group(word: str) -> bool:
    """r7 round 3: a group of three letters that is no unit production ('OOO', 'lOO'), read after a GROUPING space (a
    thin or no-break space): the text layer may spell a digit group with look-alike letters ('1 OOO' for 1 000)."""
    core = word.rstrip(_TRAIL + _CLOSE)
    return len(core) == GROUP_WIDTH and all(c in LETTERS for c in core) and UNIT_RE.fullmatch(core) is None


def cell_numbers(p: dict[str, Any], doc: dict[str, Any] | None) -> int | None:
    """How many numbers the cited cell / text holds (resolved tokens), None without a token."""
    pages, loc = (doc or {}).get("pages") or [], p.get("locator") or {}
    n = loc.get("page")
    if not (isinstance(n, int) and 1 <= n <= len(pages)):
        return None
    ctx = token_context(loc, pages[n - 1])
    style = loc.get("decimal_style") or "."
    return None if ctx is None else sum(1 for t in read(ctx[0], style, mode(loc)).toks if t.value is not None)


def decimal_flags(p: dict[str, Any], page: dict[str, Any], doc: dict[str, Any],
                  spec: dict[str, Any] | None) -> list[dict[str, Any]]:
    loc = p.get("locator") or {}
    literal, style = str(loc.get("literal") or ""), loc.get("decimal_style") or "."
    if (spec or {}).get("decimal") in DECIMAL_MARKS:
        return []
    got = verified_token(loc, page)
    tok = got[3] if got is not None and got[3] is not None and got[3].value is not None else None
    shape = tok.shape if tok is not None else ""
    evidence = doc.get("decimal_evidence")
    comma, point = evidence if isinstance(evidence, tuple) else document_evidence(doc)
    if shape == "foreign_decimal" or (shape == "sci" and tok.decimal not in ("", style)):
        return [flag(M.FLAG_DECIMAL_STYLE_AMBIGUOUS, "error", f"{p['field']}: {literal!r} uses the other decimal "
                      f"separator than this document ({style!r}): a list or a foreign decimal; set the field's "
                      "decimal style in the template or read the value on the PDF", p)]
    if shape == "grouped" and style in literal:
        return [flag(M.FLAG_DECIMAL_STYLE_AMBIGUOUS, "error", f"{p['field']}: {literal!r} groups its digits with "
                      f"the character this document ({style!r}) prints its decimals with: a foreign-locale "
                      "thousands number, a version or a list; set the field's decimal style in the template or read "
                      "the value on the PDF", p)]
    if shape == "middot" and point >= DECISIVE_MIN:
        return [flag(M.FLAG_DECIMAL_STYLE_AMBIGUOUS, "error", f"{p['field']}: {literal!r} puts a raised dot "
                      f"between digits in a document that prints its decimals with a point ({point} of them): it may "
                      "be a product, not a decimal; set the field's decimal style in the template or read the "
                      "value on the PDF", p)]
    if shape != "style_dependent":
        return []
    win, lose = (comma, point) if style == COMMA else (point, comma)
    if win >= DECISIVE_MIN and DECISIVE_RATIO * lose <= win:
        return []
    return [flag(M.FLAG_DECIMAL_STYLE_AMBIGUOUS, "error", f"{p['field']}: {literal!r} reads differently with a "
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
            out.append(flag(code, severity, f"{p['field']}: " + message.format(chars=" ".join(map(repr, hit))), p))
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
