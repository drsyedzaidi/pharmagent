"""scaled_header (error): the printed number is not the value until it is scaled (split out of value_checks.py in r7
round 3).

* right AFTER the number: a glued bare capital M (``1.5M``: the unit production reads it as molar, but it may be
  millions), a scale word (``word_lists.AFTER_SCALE_WORDS``: '1.5 million', '1.5 bn', '1.5 mln'; a look-alike
  spelling, digits inside the word or OCR letter shapes included, counts: '1.5 mi11ion', '1.5 rnillion') or a scale
  symbol (``SCALE_SYMBOLS``: '1.5 k', '1.5 M', '1.5 B'; a lone Greek or accented letter too, '1.5 κ'), read as the
  leading letter run of the next word ('million[3]', 'million/year', 'M[3]') or of the next table cell, a word broken
  at a line end read whole ('mil-' + 'lion'; r7 round 4);
* in a LABEL of the number (the proposed unit, the column header rows above the cell, the row label, a unit-column
  cell): ``numgrammar.scale_shapes``, an allow-list over the label's words (r7 round 3: everything that is no inert
  production raises the flag, a character the reader cannot read too);
* in a DATA cell beside it (the cells right before and after): the same allow-list, where the cell's resolved numbers
  are data (``scale_shapes(..., data=True)``): a bare power, a multiplier or a scale word there raises the flag.
"""
from __future__ import annotations

from typing import Any

from app.evidence import gaps
from app.evidence import models as M
from app.evidence.extract_rules import UNIT_HEADERS, norm_key
from app.evidence.numgrammar import GAP, is_data_cell
from app.evidence.numgrammar.alphabet import ASCII_LETTERS, DIGITS, LETTERS, OPENERS_BR
from app.evidence.numgrammar.units import UNIT_RE
from app.evidence.read_cache import cached
from app.evidence.scale_label import scale_shapes
from app.evidence.value_context import context, flag, verified_token, words_after_joined
from app.evidence.word_lists import AFTER_SCALE_WORDS, SCALE_SYMBOLS
from app.evidence.word_match import loose_in

NON_ASCII_LETTERS = LETTERS - ASCII_LETTERS            # a lone one after a number may stand for a scale symbol


def _header_rows(rows: list[Any]) -> list[int]:
    """Indexes of the rows of a table that hold no data cell after their first cell (header rows), once per request."""
    return cached(("header rows", id(rows)),
                  lambda: [i for i, h in enumerate(rows)
                           if not any(is_data_cell(gaps.strip(str(x or ""))) for x in h[1:])],
                  sum(len(r) for r in rows))


def _head(word: str) -> str:
    """The leading letter run of a word (openers aside; a digit inside it is a look-alike letter, 'mi11ion'): the scale
    word or symbol however the next word goes on ('million' of 'million[3]', 'million/year', "million's", 'M' of
    'M[3]': r7 round 4)."""
    i = 0
    while i < len(word) and word[i] in OPENERS_BR:
        i += 1
    j = i
    while j < len(word) and (word[j] in LETTERS or word[j] in DIGITS):
        j += 1
    return word[i:j]


def _scale_word(word: str) -> bool:
    """A scale symbol (exact, case kept), a lone letter that is no ASCII letter (a look-alike of a symbol: a Greek kappa
    for k, r7 round 4) or a scale word (loose: look-alike spellings flag too)."""
    return word in SCALE_SYMBOLS or (len(word) == 1 and word in NON_ASCII_LETTERS) or (
        bool(word) and loose_in(word, AFTER_SCALE_WORDS))


def scale_after(loc: dict[str, Any], page: dict[str, Any]) -> str:
    """The scale printed right AFTER the verified token, '' when none (see the module docstring): the next word, or for
    a table cell the first word of the cells after it (r7 round 4: the reading of the bound checks), a word broken at a
    line end read whole."""
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
    ctx, off = context(loc, page, text)
    nxt = words_after_joined(ctx, b + off, 1)
    word = _head(nxt[0]) if nxt else ""
    return word if _scale_word(word) else ""


def _labels(loc: dict[str, Any], page: dict[str, Any]) -> tuple[list[str], list[str]]:
    """(the labels of a table record's number, the data cells right beside it)."""
    cell = loc.get("cell") or {}
    try:
        rows = (page.get("tables") or [])[cell["table"]]["rows"]
        header, row, col = rows[0], rows[cell["row"]], cell["col"]
    except (KeyError, IndexError, TypeError):
        return [], []
    block = [rows[i] for i in _header_rows(rows) if i < cell["row"]] or [header]
    labels = [str(h[col] or "") for h in block if col < len(h)] + ([str(row[0] or "")] if row and col else [])
    labels += [str(row[i] or "") for i, h in enumerate(header)
               if i < len(row) and norm_key(str(h or "")) in UNIT_HEADERS]
    return labels, [str(row[k] or "") for k in (col - 1, col + 1) if 0 < k < len(row)]   # the row label is a label


def scaled_flags(p: dict[str, Any], page: dict[str, Any]) -> list[dict[str, Any]]:
    loc = p.get("locator") or {}
    after = scale_after(loc, page)
    if after:
        return [flag(M.FLAG_SCALED_HEADER, "error", f"{p['field']}: the page prints {after!r} right after the number: "
                     "the printed number is not the value until it is scaled; compare with the page image", p)]
    labels, beside = _labels(loc, page) if loc.get("kind") == "table" else ([], [])
    hit = next((lab for lab in [str(p.get("unit") or ""), *labels] if scale_shapes(lab)), None)
    if hit is None:
        hit = next((lab for lab in beside if scale_shapes(lab, data=True)), None)
    if hit is None:
        return []
    return [flag(M.FLAG_SCALED_HEADER, "error", f"{p['field']}: {hit!r} carries a multiplier, a power of ten or a "
                 "scale word, or holds a character the reader cannot read; the printed number is not the value until "
                 "it is scaled: compare with the page image", p)]


__all__ = ["scale_after", "scaled_flags"]
