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
- glyph_text_mismatch (error) / glyph_unverified (warn) / overlapping_glyphs (error): the literal
  holds a character whose drawn glyph the parse could not tie to the text (parse_fonts.py,
  parse_ink.overlapping_chars).

A pass without a token (a record verified before tokens were stored) reads nothing and raises none
of the token checks; every extraction and re-verification stores one.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any

from app.evidence import models as M
from app.evidence.normalize import decimal_evidence

QUALIFIER_CHARS = 24
_QUALIFIER_RE = re.compile(r"(?:<=|>=|=<|=>|[<>≤≥≦≧⩽⩾~≈∼≃≲≳]|\b(?:below|above|less than|more than|greater than|"
                           r"up to|at least|at most|approximately|approx\.?|about|circa|ca\.?|blq|bloq|"
                           r"lloq|uloq))\s*$", re.IGNORECASE)
# a multiplier of the printed number: 'x10^3', '×10³', 'x 10' (exponent lost), '10³ ng·h/mL', 'e3',
# 'x1000', '/1000' after a label, '(thousands)'. A power after '/' or 'per' is part of the unit
# ('µL/min/10^6 cells') and is skipped.
_SCALE_RE = re.compile(r"(?<![A-Za-z])[×xX*]\s*10(?![.,]?\d)|(?<![\d.,])10\s*(?:\^\s*[-−+]?\d+|[⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+)"
                       r"|\be[-−+]?\d+\b|(?<![A-Za-z])[×xX*]\s*1[ ,.']?000\b|/\s*1[ ,.']?000\b"
                       r"|\b(?:thousands?|millions?|billions?)\b")
_DENOMINATOR_RE = re.compile(r"(?:/|\bper)\s*$", re.IGNORECASE)
_STYLE_DEPENDENT = re.compile(r"^[-+]?\d{1,3}[.,]\d{3}$")
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
    return unicodedata.normalize("NFKC", text)


def censored_flags(p: dict[str, Any], page: dict[str, Any]) -> list[dict[str, Any]]:
    ctx = token_context(p.get("locator") or {}, page)
    if ctx is None:
        return []
    text, a, _ = ctx
    before = _fold(text[max(0, a - QUALIFIER_CHARS):a])
    m = _QUALIFIER_RE.search(before)
    if m is None:
        return []
    return [_flag(M.FLAG_CENSORED_VALUE, "error", f"{p['field']}: the page prints {m.group(0).strip()!r} before "
                  f"{(p.get('locator') or {}).get('literal')!r}; a bound or approximation is not an exact value", p)]


def _scaled(label: str) -> bool:
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
            labels += [str(header[col] or "") if col < len(header) else "", str(row[0] or "") if row else ""]
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


def decimal_flags(p: dict[str, Any], doc: dict[str, Any], spec: dict[str, Any] | None) -> list[dict[str, Any]]:
    literal = str((p.get("locator") or {}).get("literal") or "")
    evidence = doc.get("decimal_evidence")
    if (spec or {}).get("decimal") in (".", ",") or not _STYLE_DEPENDENT.match(literal):
        return []
    comma, point = evidence if isinstance(evidence, tuple) else document_evidence(doc)
    style = (p.get("locator") or {}).get("decimal_style") or "."
    win, lose = (comma, point) if style == "," else (point, comma)
    if win >= DECISIVE_MIN and DECISIVE_RATIO * lose <= win:
        return []
    return [_flag(M.FLAG_DECIMAL_STYLE_AMBIGUOUS, "error", f"{p['field']}: {literal!r} reads differently with a "
                  f"decimal point or comma, and the document's style evidence is weak ({win} vs {lose}); set the "
                  "field's decimal style in the template", p)]


_GLYPH_MESSAGES = {
    M.FLAG_GLYPH_MISMATCH: ("error", "the font names {chars} for a glyph that draws something else (or draws "
                                     "it with an arbitrary Type3 program); read the value on the PDF"),
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
        out = censored_flags(p, page) + scaled_flags(p, page) + decimal_flags(p, doc, spec) + out
    return out
