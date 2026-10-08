"""scaled_header (error): the printed number is not the value until it is scaled (split out of value_checks.py in r7
round 3).

* right AFTER the number: a glued bare capital M (``1.5M``: the unit production reads it as molar, but it may be
  millions), a scale word (``word_lists.AFTER_SCALE_WORDS``: '1.5 million', '1.5 bn', '1.5 mln'; a look-alike
  spelling, digits inside the word included, counts: '1.5 mi11ion') or a scale symbol (``SCALE_SYMBOLS``: '1.5 k',
  '1.5 M', '1.5 B');
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
from app.evidence.numgrammar import GAP, is_data_cell, loose_in, scale_shapes, words_after
from app.evidence.numgrammar.alphabet import CLOSERS_BR, GROUP_ONLY, TRAILING
from app.evidence.numgrammar.units import UNIT_RE
from app.evidence.read_cache import cached
from app.evidence.value_context import flag, verified_token
from app.evidence.word_lists import AFTER_SCALE_WORDS, SCALE_SYMBOLS

_STRIP = "".join(sorted(TRAILING - GROUP_ONLY | CLOSERS_BR))


def _header_rows(rows: list[Any]) -> list[int]:
    """Indexes of the rows of a table that hold no data cell after their first cell (header rows), once per request."""
    return cached(("header rows", id(rows)),
                  lambda: [i for i, h in enumerate(rows)
                           if not any(is_data_cell(gaps.strip(str(x or ""))) for x in h[1:])],
                  sum(len(r) for r in rows))


def scale_after(loc: dict[str, Any], page: dict[str, Any]) -> str:
    """The scale printed right AFTER the verified token, '' when none (see the module docstring)."""
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
    word = nxt[0].rstrip(_STRIP) if nxt else ""
    return word if word in SCALE_SYMBOLS or (word and loose_in(word, AFTER_SCALE_WORDS)) else ""


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
