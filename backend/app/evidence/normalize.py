"""The snippet LOCATOR (an index-mapped text normaliser), the page's decimal-style inference and the re-exports of
the number reader. This module never yields a value: numbers are read by ``numgrammar`` only (an explicit allowed
alphabet, one approved-equivalents table, every other character in or next to a number makes it unresolved).

Every transform returns (text, idx_map) with idx_map[i] = offset of text[i] in the ORIGINAL string, so a match
found in normalised text maps back to raw page offsets. Pipeline: per-char NFKC -> the approved equivalents
(``numgrammar.EQUIV``, 1:1) -> casefold -> de-hyphenate line wraps -> collapse whitespace. A character that renders
as nothing is NOT dropped: the page and the snippet must both hold it, or the snippet is not on the page.
"""
from __future__ import annotations

import re
import unicodedata

from app.evidence.numgrammar import EQUIV, number_spans, numbers_in

IdxMap = list[int]

# decimal-style evidence only (parse.py writes the style into pages.json; they never read a value)
_COMMA_DECIMAL = re.compile(r"\d,\d{1,2}(?!\d)")
_POINT_DECIMAL = re.compile(r"\d\.\d+")
_POINT_THOUSANDS = re.compile(r"\b\d{1,3}(?:\.\d{3})+,\d+")
_CITATIONS = re.compile(r"\[\s*\d+(?:\s*[,;\-–]\s*\d+)+\s*\]|\(\s*\d+(?:\s*[,;]\s*\d+){2,}\s*\)")


def decimal_evidence(text: str) -> tuple[int, int]:
    """(decimal-comma, decimal-point) evidence counts in *text*. Citation lists ('[1,2]', '[3-5]',
    '(1,4,7)') are not numbers and are removed first."""
    text = _CITATIONS.sub(" ", text)
    comma = len(_POINT_THOUSANDS.findall(text))
    text = _POINT_THOUSANDS.sub(" ", text)
    comma += len(_COMMA_DECIMAL.findall(text))
    point = len(_POINT_DECIMAL.findall(text))
    return comma, point


def infer_decimal_style(text: str) -> str:
    """',' when decimal-comma evidence outweighs decimal-point evidence, else '.'."""
    comma, point = decimal_evidence(text)
    return "," if comma > point else "."


# ------------------------------------------------- mapped transforms ---

def _identity(text: str, idx_map: IdxMap | None) -> IdxMap:
    return list(range(len(text))) if idx_map is None else idx_map


def _expand_with_map(text: str, idx_map: IdxMap | None, fn) -> tuple[str, IdxMap]:
    """Apply a per-character expansion fn(ch) -> str; every output char maps to the input char."""
    src_map = _identity(text, idx_map)
    out: list[str] = []
    out_map: IdxMap = []
    for ch, raw in zip(text, src_map):
        rep = fn(ch)
        out.append(rep)
        out_map.extend([raw] * len(rep))
    return "".join(out), out_map


def nfkc_with_map(text: str, idx_map: IdxMap | None = None) -> tuple[str, IdxMap]:
    return _expand_with_map(text, idx_map, lambda ch: unicodedata.normalize("NFKC", ch))


def equiv_with_map(text: str, idx_map: IdxMap | None = None) -> tuple[str, IdxMap]:
    """The approved equivalents (numgrammar.EQUIV), one character for one character."""
    return _expand_with_map(text, idx_map, lambda ch: EQUIV.get(ch, ch))


def casefold_with_map(text: str, idx_map: IdxMap | None = None) -> tuple[str, IdxMap]:
    return _expand_with_map(text, idx_map, str.casefold)


def dehyphenate_with_map(text: str, idx_map: IdxMap | None = None) -> tuple[str, IdxMap]:
    """Join words wrapped at a hyphen: letter '-' newline [ws] letter -> the hyphen and break are dropped."""
    src_map = _identity(text, idx_map)
    out: list[str] = []
    out_map: IdxMap = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch == "-" and i > 0 and text[i - 1].isalpha() and i + 1 < n and text[i + 1] == "\n":
            j = i + 1
            while j < n and text[j].isspace():
                j += 1
            if j < n and text[j].isalpha():
                i = j
                continue
        out.append(ch)
        out_map.append(src_map[i])
        i += 1
    return "".join(out), out_map


def strip_ws_with_map(text: str, idx_map: IdxMap | None = None) -> tuple[str, IdxMap]:
    """Collapse whitespace runs to one space and trim both ends; the space maps to the run's first char."""
    src_map = _identity(text, idx_map)
    out: list[str] = []
    out_map: IdxMap = []
    pending: int | None = None
    for ch, raw in zip(text, src_map):
        if ch.isspace():
            pending = raw if pending is None else pending
            continue
        if pending is not None and out:
            out.append(" ")
            out_map.append(pending)
        pending = None
        out.append(ch)
        out_map.append(raw)
    return "".join(out), out_map


def normalise(text: str) -> tuple[str, IdxMap]:
    """Full pipeline; returns (normalised text, map of each char to its raw offset)."""
    norm, m = nfkc_with_map(text)
    norm, m = equiv_with_map(norm, m)
    norm, m = casefold_with_map(norm, m)
    norm, m = dehyphenate_with_map(norm, m)
    return strip_ws_with_map(norm, m)


__all__ = ["IdxMap", "casefold_with_map", "decimal_evidence", "dehyphenate_with_map", "equiv_with_map",
           "infer_decimal_style", "nfkc_with_map", "normalise", "number_spans", "numbers_in", "strip_ws_with_map"]
