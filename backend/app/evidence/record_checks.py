"""Checks on a VERIFIED proposal that need its parsed page: the proposed unit against the unit the
source prints for that number, and a flag when the cited page had invisible text removed.

Source unit, in order: the unit token printed right after the VERIFIED token on the page / in the
cell ('100 mcg', '45%'; read at locator['token'], never re-found in the snippet), else for a table
cell the column header's bracketed unit, the row label's bracketed unit or the row's 'Unit' column.
Units are compared for EQUALITY after normalisation and a small equivalence table: a printed 'mg'
never vouches for 'mg/kg' (a 1000x dose error otherwise passes). A unit the next token may continue
('mg kg-1') is undecidable. Different units are an error flag; no source unit, or an undecidable
one, is a warning (unit_unverified). Value checks (qualifiers, scale, style, glyphs) live in
value_checks.py.
"""
from __future__ import annotations

import re
from functools import cache
from typing import Any

from app.evidence import gaps
from app.evidence import models as M
from app.evidence.extract_rules import UNIT_HEADERS, norm_key, split_unit
from app.evidence.normalize import normalise
from app.evidence.numgrammar import GAP, words_after
from app.evidence.numgrammar.alphabet import (
    ASCII_LETTERS,
    CARET,
    CLOSERS_BR,
    DASH,
    DIGITS,
    GREEK,
    GROUP_ONLY,
    HYPHEN,
    LETTERS,
    MULT_LETTERS,
    MULTIPLIERS,
    OPENERS_BR,
    PER,
    PERCENTS,
    POINT,
    PRODUCT_DOTS,
    SLASH,
    STAR,
    TRAILING,
    char_class,
)
from app.evidence.numgrammar.equivalents import fold
from app.evidence.unit_norm import NOT_CONTINUATIONS, UNIT_SYMBOLS, canonical_unit, normalise_unit, prefix_conflict
from app.evidence.value_checks import token_context, value_flags

# a printed unit, from explicit classes (the alphabet's letters and digits; the text arrives folded, so the micro sign
# is already the Greek mu): a symbol led by a letter, or a percent sign; then up to three letters, a slash and more
_SPACE = char_class(GAP, negate=False)
_LETTER = char_class(LETTERS, negate=False)                       # a letter of the alphabet (r7: the accented ones too)
_WORD = char_class(ASCII_LETTERS | GREEK | DIGITS, negate=False)
# r7 round 4: the joiners inside a unit symbol from the role sets (a slash, the times sign and the middle dot, a star, a
# point, a caret, a hyphen: 'ng×h/mL' joins like 'ng·h/mL'), never a hand-typed class
_UNIT_JOINS = char_class(SLASH | PRODUCT_DOTS | {STAR, POINT, CARET} | HYPHEN, negate=False)
_EXP_JOINS = char_class({POINT, CARET} | HYPHEN, negate=False)
_SLASH = char_class(SLASH, negate=False)
_UNIT_CHARS = "(?:" + _WORD + "|" + _UNIT_JOINS + ")*"
_UNIT_TOKEN_RE = re.compile("^" + _SPACE + "*(" + char_class(PERCENTS, negate=False) + "|" + _LETTER + _UNIT_CHARS
                            + "(?:" + _SPACE + "+" + _LETTER
                            + "{1,3}(?:" + _SLASH + "(?:" + _WORD + "|" + _EXP_JOINS + ")+)+)?)")
_NEXT_TOKEN_RE = re.compile("^" + _SPACE + "*(" + _SLASH + "|" + PER + "(?!" + _WORD + ")|"
                            + char_class(GAP, negate=True) + "+)")
_TRAIL = "".join(sorted(TRAILING - CLOSERS_BR - GROUP_ONLY))       # '.,;:' after a unit
_TRAIL_CLOSE = "".join(sorted(TRAILING - GROUP_ONLY))             # the same and the closers
_OPEN = "".join(sorted(OPENERS_BR))
_INVERSE = (DASH + "1", CARET + DASH + "1")                       # a unit's inverse power ('h-1', 'h^-1')
UNIT_WORDS = 3          # whole words after the token a printed unit may span ('mg kg-1 h-1', 'ng h/mL')
_JOINERS = SLASH | {CARET} | (MULTIPLIERS - MULT_LETTERS)   # a next token made only of these continues a unit
# the unit comparison tables are single-sourced in unit_norm.py (r7 round 2): the printed-unit symbols and the words
# that never continue a unit; never reader data
_BASES, _NOT_CONTINUATIONS = UNIT_SYMBOLS, NOT_CONTINUATIONS


@cache
def _segmentable(piece: str) -> bool:
    """True when *piece* is a concatenation of base unit symbols ('ngh' = 'ng' + 'h'; 'healthy' is not)."""
    if not piece:
        return True
    return any(piece.startswith(b) and _segmentable(piece[len(b):]) for b in _BASES)


def recognised_unit(token: str | None) -> str | None:
    """normalise_unit(token) when every '/'-separated piece is made of known unit symbols, else None."""
    if not token:
        return None
    unit = normalise_unit(token).rstrip(_TRAIL)
    pieces = [p for p in re.split(_SLASH, unit) if p]
    return unit if pieces and all(_segmentable(p.replace(_INVERSE[0], "").replace(_INVERSE[1], ""))
                                  for p in pieces) else None


def _continues(tail: str) -> bool:
    """True when the text after a unit token may extend it ('/kg', 'per kg', 'kg-1', 'h')."""
    m = _NEXT_TOKEN_RE.match(tail)
    if m is None:
        return False
    token = m.group(1).rstrip(_TRAIL_CLOSE)
    if token.lstrip(_OPEN) in SLASH | {PER} or (token and set(token) <= _JOINERS):
        return True
    return token not in _NOT_CONTINUATIONS and recognised_unit(token) is not None


Printed = tuple[str | None, bool, str | None]      # (normalised unit, decided, unit as printed: case kept)


def _adjacent_unit(loc: dict[str, Any], page: dict[str, Any]) -> Printed:
    """(unit printed right after the VERIFIED token on the page / in the cell, decided, raw). Read from
    the page, never re-found in the snippet: '5' is not the '5' inside '15 mg/kg'. The text read is the glued rest of
    the token's own word ('%', '°C') and up to UNIT_WORDS whole words after it; unit matching only, a value is never
    read here (a record whose number touches a character outside the allowed alphabet has already failed
    verification). Undecided when the next token may continue the unit, or more words follow the last one read."""
    ctx = token_context(loc, page)
    if ctx is None:
        return None, True, None
    text, _, end = ctx
    folded = fold(text)                          # the words are read there: so is the end of the number's own word
    stop = end
    while stop < len(folded) and folded[stop] not in GAP:
        stop += 1
    words = words_after(text, end, UNIT_WORDS + 1)
    raw = text[end:stop] + "".join(" " + w for w in words[:UNIT_WORDS])
    norm, idx = normalise(raw)
    lead = " " if raw[:1] in GAP else ""
    rest = lead + norm
    m = _UNIT_TOKEN_RE.match(rest)
    unit = recognised_unit(m.group(1)) if m else None
    if m is None or unit is None:
        return None, True, None
    a, b = m.start(1) - len(lead), m.end(1) - len(lead)
    printed = raw[idx[a]:idx[b - 1] + 1] if 0 <= a < b <= len(idx) else None
    tail = rest[m.end():]
    if not gaps.strip(tail):
        return unit, len(words) <= UNIT_WORDS, printed
    return unit, not _continues(tail), printed


def _table_unit(loc: dict[str, Any], page: dict[str, Any]) -> tuple[str | None, str | None]:
    """(normalised unit, unit as printed) from the header, row label or Unit column of the cited cell."""
    cell = loc.get("cell") or {}
    try:
        rows = (page.get("tables") or [])[cell["table"]]["rows"]
        row, header = rows[cell["row"]], rows[0]
    except (KeyError, IndexError, TypeError):
        return None, None
    col = cell.get("col")
    candidates = [split_unit(header[col])[1] if isinstance(col, int) and col < len(header) else None,
                  split_unit(row[0])[1] if row else None]
    candidates += [row[i] for i, h in enumerate(header) if norm_key(h) in UNIT_HEADERS and i < len(row)]
    return next(((u, c) for u, c in ((recognised_unit(c), c) for c in candidates) if u), (None, None))


def source_unit(p: dict[str, Any], page: dict[str, Any]) -> Printed:
    """(unit the source prints for the cited number, decided, as printed) -- see the module docstring."""
    loc = p.get("locator") or {}
    unit, decided, printed = _adjacent_unit(loc, page)
    if unit is None and loc.get("kind") == "table":
        (unit, printed), decided = _table_unit(loc, page), True
    return unit, decided, printed


def _flag(code: str, severity: str, message: str, p: dict[str, Any]) -> dict[str, Any]:
    return {"code": code, "severity": severity, "message": message, "field": p["field"],
            "group": p.get("group_label") or "", "origin": "script"}


def unit_flags(p: dict[str, Any], page: dict[str, Any]) -> list[dict[str, Any]]:
    """For a passed numeric proposal with a unit: mismatch (error) or unverified (warn)."""
    proposed = p.get("unit")
    if p.get("verify_status") != "pass" or not proposed or p.get("value_num") is None:
        return []
    have, decided, printed = source_unit(p, page)
    if have is None:
        return [_flag(M.FLAG_UNIT_UNVERIFIED, "warn", f"{p['field']}: unit {proposed!r} is not printed next to "
                      "the cited number or in its table header; check it on the page", p)]
    want, have = canonical_unit(proposed), canonical_unit(have)
    case = prefix_conflict(proposed, printed) if want == have else None
    if case == "ambiguous":
        return [_flag(M.FLAG_UNIT_UNVERIFIED, "warn", f"{p['field']}: the source prints {printed!r} in capitals; "
                      f"check that it means {proposed!r} (milli) and not mega", p)]
    if case == "mismatch":
        return [_flag(M.FLAG_UNIT_SOURCE_MISMATCH, "error", f"{p['field']}: the source prints {printed!r} for this "
                      f"number, the record says {proposed!r} (mega and milli differ by case)", p)]
    if decided and want == have:
        return []
    if not decided and want.startswith(have):
        return [_flag(M.FLAG_UNIT_UNVERIFIED, "warn", f"{p['field']}: the source prints {have!r} followed by text "
                      f"that may continue the unit; check {proposed!r} on the page", p)]
    return [_flag(M.FLAG_UNIT_SOURCE_MISMATCH, "error", f"{p['field']}: the source prints {have!r} for this "
                  f"number, the record says {proposed!r}", p)]


def hidden_text_flags(p: dict[str, Any], page: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Warn when hidden glyphs were removed from the cited page; error when its visibility could not be
    checked from rendered ink at all (the page text may still hold hidden glyphs)."""
    page = page or {}
    if page.get("ink_check") == "unverified":
        return [_flag(M.FLAG_HIDDEN_TEXT, "error", f"page {page.get('n')} could not be rendered to check which "
                      "characters are visible, so hidden text may remain; confirm the value on the PDF", p)]
    n = int(page.get("hidden_chars") or 0)
    if n <= 0:
        return []
    return [_flag(M.FLAG_HIDDEN_TEXT, "warn", f"page {page.get('n')} holds {n} invisible or off-page "
                  "characters that were removed before verification; compare with the PDF", p)]


def page_checks(p: dict[str, Any], doc: dict[str, Any] | None,
                spec: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Unit, hidden-text and value (qualifier, scale, decimal style, glyph identity) flags for one
    checked proposal (none when the page is unavailable). *spec* is the field's template entry."""
    pages = (doc or {}).get("pages") or []
    n = (p.get("locator") or {}).get("page")
    page = pages[n - 1] if isinstance(n, int) and not isinstance(n, bool) and 1 <= n <= len(pages) else None
    if page is None:
        return []
    return unit_flags(p, page) + hidden_text_flags(p, page) + value_flags(p, page, doc or {}, spec)
