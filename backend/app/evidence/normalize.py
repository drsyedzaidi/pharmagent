"""The snippet LOCATOR (an index-mapped text normaliser), the page's decimal-style inference and the re-exports of
the number reader. This module never yields a value: numbers are read by ``numgrammar`` only (an explicit allowed
alphabet, one approved-equivalents table, every other character in or next to a number makes it unresolved).

Every transform returns (text, idx_map) with idx_map[i] = offset of text[i] in the ORIGINAL string, so a match
found in normalised text maps back to raw page offsets. Pipeline: per-char NFKC (see ``nfkc_char``: never into or out
of the numeric alphabet) -> the approved equivalents (``numgrammar.EQUIV``, 1:1) -> casefold -> de-hyphenate line
wraps -> collapse whitespace. A character that renders as nothing is NOT dropped: the page and the snippet must both
hold it, or the snippet is not on the page. ``normalise`` is cached for the length of one request (read_cache).
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable

from app.evidence.numgrammar import EQUIV, number_spans, numbers_in
from app.evidence.numgrammar.alphabet import NUMBER_CHARS
from app.evidence.numgrammar.style import style_evidence
from app.evidence.read_cache import cached

IdxMap = list[int]

MAX_NFKC_EXPANSION = 4          # NFKC of one character longer than this stays as it is (U+FDFA expands to 18)
_NON_ASCII = re.compile("[^\x00-\x7f]")
_HYPHEN_BREAK = re.compile("-\n")
_SPACE_RUN = re.compile("\\s+")


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


def _rewrite(text: str, idx_map: IdxMap | None, char_fn: Callable[[str], str],
             run_fn: Callable[[str], str] | None = None) -> tuple[str, IdxMap]:
    """Rewrite every non-ASCII character with char_fn(ch) -> str (each output character maps to the input one); the
    ASCII runs between them are copied (through run_fn when it is not the identity: it must keep their length)."""
    src = _identity(text, idx_map)
    out: list[str] = []
    out_map: IdxMap = []
    last = 0
    for m in _NON_ASCII.finditer(text):
        a = m.start()
        if a > last:
            out.append(text[last:a] if run_fn is None else run_fn(text[last:a]))
            out_map.extend(src[last:a])
        rep = char_fn(m.group())
        out.append(rep)
        out_map.extend([src[a]] * len(rep))
        last = a + 1
    if last < len(text):
        out.append(text[last:] if run_fn is None else run_fn(text[last:]))
        out_map.extend(src[last:])
    return "".join(out), out_map


def nfkc_char(ch: str) -> str:
    """NFKC of one character, applied only when it does not touch the numeric alphabet and does not balloon: a
    superscript or fullwidth digit, a fullwidth sign, a circled letter ``ⓔ`` or ``½`` never becomes the ASCII character
    the reader reads (it would equate two different characters of a number), and a character whose NFKC is longer than
    MAX_NFKC_EXPANSION characters (``ﷺ``: 18) stays as it is."""
    nf = unicodedata.normalize("NFKC", ch)
    if nf == ch or len(nf) > MAX_NFKC_EXPANSION or ch in NUMBER_CHARS or not NUMBER_CHARS.isdisjoint(nf):
        return ch
    return nf


def nfkc_with_map(text: str, idx_map: IdxMap | None = None) -> tuple[str, IdxMap]:
    return _rewrite(text, idx_map, nfkc_char)


def equiv_with_map(text: str, idx_map: IdxMap | None = None) -> tuple[str, IdxMap]:
    """The approved equivalents (numgrammar.EQUIV), one character for one character."""
    return _rewrite(text, idx_map, lambda ch: EQUIV.get(ch, ch))


def casefold_with_map(text: str, idx_map: IdxMap | None = None) -> tuple[str, IdxMap]:
    return _rewrite(text, idx_map, str.casefold, str.casefold)


def dehyphenate_with_map(text: str, idx_map: IdxMap | None = None) -> tuple[str, IdxMap]:
    """Join words wrapped at a hyphen: letter '-' newline [ws] letter -> the hyphen and break are dropped."""
    src = _identity(text, idx_map)
    out: list[str] = []
    out_map: IdxMap = []
    last, n = 0, len(text)
    for m in _HYPHEN_BREAK.finditer(text):
        i = m.start()
        if i == 0 or not text[i - 1].isalpha():
            continue
        j = i + 1
        while j < n and text[j].isspace():
            j += 1
        if j < n and text[j].isalpha():
            out.append(text[last:i])
            out_map.extend(src[last:i])
            last = j
    out.append(text[last:])
    out_map.extend(src[last:])
    return "".join(out), out_map


def strip_ws_with_map(text: str, idx_map: IdxMap | None = None) -> tuple[str, IdxMap]:
    """Collapse whitespace runs to one space and trim both ends; the space maps to the run's first char."""
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
    norm, m = nfkc_with_map(text)
    norm, m = equiv_with_map(norm, m)
    norm, m = casefold_with_map(norm, m)
    norm, m = dehyphenate_with_map(norm, m)
    return strip_ws_with_map(norm, m)


def normalise(text: str) -> tuple[str, IdxMap]:
    """Full pipeline; returns (normalised text, map of each char to its raw offset). Read-only result: it is shared
    by every call on the same text while a request cache is open."""
    return cached(("normalise", text), lambda: _normalise(text), len(text))


__all__ = ["IdxMap", "casefold_with_map", "decimal_evidence", "dehyphenate_with_map", "equiv_with_map",
           "infer_decimal_style", "nfkc_char", "nfkc_with_map", "normalise", "number_spans", "numbers_in",
           "strip_ws_with_map"]
