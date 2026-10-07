"""The snippet LOCATOR (an index-mapped text normaliser), the page's decimal-style inference and the re-exports of
the number reader. This module never yields a value: numbers are read by ``numgrammar`` only (an explicit allowed
alphabet, one approved-equivalents table, every other character in or next to a number makes it unresolved).

The locator has no normalisation table of its own. Pipeline: ``fold`` (the reader's approved equivalents, 1:1) -> the
lower case of the letters the alphabet lists (1:1) -> seven explicit ligature spellings (1 -> n) -> de-hyphenate line
wraps between two letters of the alphabet -> collapse runs of the GAP set and the thin space. Every other character is
kept as it is, so two texts locate each other only when the reader would read them the same way. A character that
renders as nothing is NOT dropped: the page and the snippet must both hold it, or the snippet is not on the page.

Every transform returns (text, idx_map) with idx_map[i] = offset of text[i] in the ORIGINAL string, so a match found in
normalised text maps back to raw page offsets. ``normalise`` is cached for the length of one request (read_cache).
"""
from __future__ import annotations

import re

from app.evidence.numgrammar import number_spans, numbers_in
from app.evidence.numgrammar.alphabet import ALL_LETTERS, GAP, char_class
from app.evidence.numgrammar.equivalents import CASE_PAIRS, LIGATURES, THIN, casefold_letters, fold
from app.evidence.numgrammar.style import style_evidence
from app.evidence.read_cache import cached

IdxMap = list[int]

_SPACES = GAP | frozenset({THIN})       # the spaces the locator collapses (the thin space: target of NBSP in ``fold``)
_SPACE_RUN = re.compile(char_class(_SPACES, negate=False) + "+")
_WRAP_LETTERS = ALL_LETTERS | frozenset(CASE_PAIRS.values())      # a letter, before or after the lower-casing
_HYPHEN_BREAK = re.compile("-\n")
_LIGATURE = re.compile(char_class(frozenset(LIGATURES), negate=False))


def decimal_evidence(text: str) -> tuple[int, int]:
    """(decimal-comma, decimal-point) evidence counts in *text*: the resolved numbers the reader reads with a decimal
    comma or a decimal point (``numgrammar.style``). A character the reader cannot read votes for nothing."""
    return style_evidence(text)


def infer_decimal_style(text: str) -> str:
    """',' when decimal-comma evidence outweighs decimal-point evidence, else '.'."""
    comma, point = decimal_evidence(text)
    return "," if comma > point else "."


# ------------------------------------------------- mapped transforms ---

def _identity(text: str, idx_map: IdxMap | None) -> IdxMap:
    return list(range(len(text))) if idx_map is None else idx_map


def ligature_with_map(text: str, idx_map: IdxMap | None = None) -> tuple[str, IdxMap]:
    """Each ligature of ``LIGATURES`` spelled with its letters (every letter maps to the ligature's offset)."""
    src = _identity(text, idx_map)
    if _LIGATURE.search(text) is None:
        return text, src
    out: list[str] = []
    out_map: IdxMap = []
    last = 0
    for m in _LIGATURE.finditer(text):
        a = m.start()
        rep = LIGATURES[m.group()]
        out.append(text[last:a] + rep)
        out_map.extend(src[last:a])
        out_map.extend([src[a]] * len(rep))
        last = a + 1
    out.append(text[last:])
    out_map.extend(src[last:])
    return "".join(out), out_map


def dehyphenate_with_map(text: str, idx_map: IdxMap | None = None) -> tuple[str, IdxMap]:
    """Join words wrapped at a hyphen: letter '-' newline [space] letter -> the hyphen and break are dropped. A letter
    is one the alphabet lists (a digit, a Cyrillic or a mathematical letter before the hyphen is a range or a code)."""
    src = _identity(text, idx_map)
    if "-\n" not in text:
        return text, src
    out: list[str] = []
    out_map: IdxMap = []
    last, n = 0, len(text)
    for m in _HYPHEN_BREAK.finditer(text):
        i = m.start()
        if i == 0 or text[i - 1] not in _WRAP_LETTERS:
            continue
        j = i + 1
        while j < n and text[j] in _SPACES:
            j += 1
        if j < n and text[j] in _WRAP_LETTERS:
            out.append(text[last:i])
            out_map.extend(src[last:i])
            last = j
    out.append(text[last:])
    out_map.extend(src[last:])
    return "".join(out), out_map


def strip_ws_with_map(text: str, idx_map: IdxMap | None = None) -> tuple[str, IdxMap]:
    """Collapse runs of the GAP set and the thin space to one space and trim both ends; the space maps to the run's
    first char."""
    src = _identity(text, idx_map)
    out: list[str] = []
    out_map: IdxMap = []
    last, n = 0, len(text)
    for m in _SPACE_RUN.finditer(text):
        a, b = m.span()
        if a > last:
            out.append(text[last:a])
            out_map.extend(src[last:a])
        if out and b < n:
            out.append(" ")
            out_map.append(src[a])
        last = b
    if last < n:
        out.append(text[last:])
        out_map.extend(src[last:])
    return "".join(out), out_map


def _normalise(text: str) -> tuple[str, IdxMap]:
    norm = casefold_letters(fold(text))                     # both 1:1: the offsets are still the page's own
    norm, m = ligature_with_map(norm)
    norm, m = dehyphenate_with_map(norm, m)
    return strip_ws_with_map(norm, m)


def normalise(text: str) -> tuple[str, IdxMap]:
    """Full pipeline; returns (normalised text, map of each char to its raw offset). Read-only result: it is shared
    by every call on the same text while a request cache is open."""
    return cached(("normalise", text), lambda: _normalise(text), len(text))


__all__ = ["IdxMap", "decimal_evidence", "dehyphenate_with_map", "infer_decimal_style", "ligature_with_map",
           "normalise", "number_spans", "numbers_in", "strip_ws_with_map"]
