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
import unicodedata
from typing import Any

from app.evidence import models as M
from app.evidence.invisible import strip_invisible
from app.evidence.normalize import MIDDOTS, decimal_evidence, number_spans
from app.evidence.qc_flags import is_interval_field
from app.evidence.visible_window import visible_after, visible_before

QUALIFIER_CHARS = 24
_QUALIFIER_RE = re.compile(r"(?:<=|>=|=<|=>|[<>≤≥≦≧⩽⩾~≈∼≃≲≳≮≯≰≱≁≉]|\b(?:below|above|less than|more than|greater than|"
                           r"up to|at least|at most|approximately|approx\.?|about|circa|ca\.?|blq|bloq|"
                           r"lloq|uloq))\s*$", re.IGNORECASE)
# a multiplier of the printed number: 'x10^3', '×10³', 'x 10' (exponent lost), '10³ ng·h/mL', 'e3',
# 'x1000', '/1000' after a label, '(thousands)'. A power after '/' or 'per' is part of the unit
# ('µL/min/10^6 cells') and is skipped.
_SCALE_RE = re.compile(r"(?<![A-Za-z])[×xX*]\s*10(?![.,]?\d)|(?<![\d.,])10\s*(?:\^\s*[-−+]?\d+|[⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+)"
                       r"|\be[-−+]?\d+\b|(?<![A-Za-z])[×xX*]\s*1[ ,.']?000\b|/\s*1[ ,.']?000\b"
                       r"|\b(?:thousands?|millions?|billions?)\b", re.IGNORECASE)
_DENOMINATOR_RE = re.compile(r"(?:/|\bper)\s*$", re.IGNORECASE)
_STYLE_DEPENDENT = re.compile(r"^[-+]?\d{1,3}[.,]\d{3}$")
_RANGE_BEFORE = re.compile(r"\d\s*(?:[-‐‑‒–—]|\bto\b)\s*$", re.IGNORECASE)
_RANGE_AFTER = re.compile(r"^\s*(?:[-‐‑‒–—]|to\b)\s*[-+]?\d", re.IGNORECASE)
# a data cell: digits with number punctuation only (a header cell such as 'Day 1' has words)
_DATA_CELL = re.compile(r"[\s\d.,·'()±%\-–−+<>≤≥~×/]*\d[\s\d.,·'()±%\-–−+<>≤≥~×/eE]*")
# digit groups joined only by a space or by comma groups: '12 145', '1,23,456' (one grouped number?)
_GROUP_RUN = re.compile(r"(?<![\w.,])\d{1,3}(?:[ ,]\d{2,3})+(?:[.,]\d+)?(?!\w)")
DECISIVE_MIN, DECISIVE_RATIO = 3, 3


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


def _fold(text: str) -> str:
    """NFKC after invisible characters are dropped ('<' + ZWJ is still '<'; '<' + U+0338 composes to '≮');
    any combining mark left is then removed, so a marked qualifier is still a qualifier."""
    return strip_invisible(unicodedata.normalize("NFKC", strip_invisible(text)), marks=True)


def censored_flags(p: dict[str, Any], page: dict[str, Any]) -> list[dict[str, Any]]:
    ctx = token_context(p.get("locator") or {}, page)
    if ctx is None:
        return []
    text, a, b = ctx
    literal = (p.get("locator") or {}).get("literal")
    # the windows hold VISIBLE characters: padding with marks, invisible or unreadable characters or long
    # whitespace cannot push a qualifier or a range bound out of them
    prefix, capped_before = visible_before(text, a, QUALIFIER_CHARS)
    suffix, capped_after = visible_after(text, b, QUALIFIER_CHARS)
    before, after = _fold(prefix), _fold(suffix)
    m = _QUALIFIER_RE.search(before)
    if m is not None:
        return [_flag(M.FLAG_CENSORED_VALUE, "error", f"{p['field']}: the page prints {m.group(0).strip()!r} before "
                      f"{literal!r}; a bound or approximation is not an exact value", p)]
    if capped_before or capped_after:
        return [_flag(M.FLAG_CENSORED_VALUE, "error", f"{p['field']}: too many unreadable or invisible characters "
                      f"lie next to {literal!r} to rule out a qualifier or a range bound; read the value on the PDF",
                      p)]
    if is_interval_field(str(p.get("field") or "")) or not (_RANGE_BEFORE.search(before) or _RANGE_AFTER.match(after)):
        return []
    return [_flag(M.FLAG_CENSORED_VALUE, "error", f"{p['field']}: {literal!r} is one end of a printed range; a "
                  "range bound is not an exact value (use an interval field for it)", p)]


def _scaled(label: str) -> bool:
    label = strip_invisible(label, marks=True)         # 'x' + ZWJ + '10^3' and a struck-through 'x' still scale
    return any(not (_DENOMINATOR_RE.search(label[:m.start()]) and not m.group(0).startswith("/"))
               for m in _SCALE_RE.finditer(label))


def scaled_flags(p: dict[str, Any], page: dict[str, Any]) -> list[dict[str, Any]]:
    loc = p.get("locator") or {}
    labels = [str(p.get("unit") or "")]
    if loc.get("kind") == "table":
        cell = loc.get("cell") or {}
        try:
            rows = (page.get("tables") or [])[cell["table"]]["rows"]
            header, row, col = rows[0], rows[cell["row"]], cell["col"]
            block = [h for h in rows[:cell["row"]] if not any(_DATA_CELL.fullmatch(str(x or "").strip())
                                                              for x in h[1:])] or [header]
            labels += [str(h[col] or "") for h in block if col < len(h)] + [str(row[0] or "") if row else ""]
            labels += [str(row[i] or "") for i, h in enumerate(header)
                       if i < len(row) and str(h or "").strip().casefold() in ("unit", "units")]
        except (KeyError, IndexError, TypeError):
            pass
    hit = next((lab for lab in labels if _scaled(lab)), None)      # raw: NFKC turns '10³' into '103'
    if hit is None:
        return []
    return [_flag(M.FLAG_SCALED_HEADER, "error", f"{p['field']}: {hit!r} carries a multiplier; the printed "
                  "number is not the value until it is scaled", p)]


def document_evidence(doc: dict[str, Any]) -> tuple[int, int]:
    """(decimal-comma, decimal-point) evidence over the document's page text (what its style was
    inferred from)."""
    return decimal_evidence("\n".join(str(pg.get("text") or "") for pg in doc.get("pages") or []))


def _against_style(literal: str, style: str) -> bool:
    """True when the literal's decimal separator is not the page's ('1,2' or '0,412' on a '.' page)."""
    body = re.split(r"[eE×xX*^]", literal.lstrip("+-−"))[0].replace(" ", "").replace("'", "").replace("\u2009", "")
    seps = [c for c in body if c in ".,"]
    if not seps or any(d in body for d in MIDDOTS) or seps[-1] == style or seps.count(seps[-1]) > 1:
        return False
    whole, tail = body.rsplit(seps[-1], 1)
    return len(tail) != 3 or whole in ("0", "")


def grouped_flags(p: dict[str, Any], page: dict[str, Any]) -> list[dict[str, Any]]:
    """Error when the token is one piece of digit groups the strict reading splits ('12' of '12 145',
    '456' of '1,23,456'): the source may print ONE grouped number."""
    ctx = token_context(p.get("locator") or {}, page)
    if ctx is None:
        return []
    text, a, b = ctx
    style = (p.get("locator") or {}).get("decimal_style") or "."
    for m in _GROUP_RUN.finditer(text):
        if m.start() < b and a < m.end() and len(number_spans(m.group(0), style, strict=True)) > 1:
            return [_flag(M.FLAG_COMPOSITE_CELL, "error", f"{p['field']}: {m.group(0)!r} may be one number grouped "
                          "by a space or comma; read the value on the PDF", p)]
    return []


def cell_numbers(p: dict[str, Any], doc: dict[str, Any] | None) -> int | None:
    """How many numbers the cited cell / text holds (strict reading), None without a token."""
    pages, n = (doc or {}).get("pages") or [], (p.get("locator") or {}).get("page")
    if not (isinstance(n, int) and 1 <= n <= len(pages)):
        return None
    ctx = token_context(p.get("locator") or {}, pages[n - 1])
    style = (p.get("locator") or {}).get("decimal_style") or "."
    return None if ctx is None else len(number_spans(ctx[0], style, strict=True))


def decimal_flags(p: dict[str, Any], doc: dict[str, Any], spec: dict[str, Any] | None) -> list[dict[str, Any]]:
    literal = str((p.get("locator") or {}).get("literal") or "")
    evidence = doc.get("decimal_evidence")
    style = (p.get("locator") or {}).get("decimal_style") or "."
    if (spec or {}).get("decimal") in (".", ","):
        return []
    if _against_style(literal, style):
        return [_flag(M.FLAG_DECIMAL_STYLE_AMBIGUOUS, "error", f"{p['field']}: {literal!r} uses the other decimal "
                      f"separator than this document ({style!r}): a list or a foreign decimal; set the field's "
                      "decimal style in the template or read the value on the PDF", p)]
    if not _STYLE_DEPENDENT.match(literal):
        return []
    comma, point = evidence if isinstance(evidence, tuple) else document_evidence(doc)
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
        out = (censored_flags(p, page) + scaled_flags(p, page) + decimal_flags(p, doc, spec)
               + grouped_flags(p, page) + out)
    return out
