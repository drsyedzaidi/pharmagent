"""Whitespace helpers built from the reader's GAP set (r7 round 2).

``str.split()`` and ``str.strip()`` without an argument use Unicode whitespace: an ideographic space, a line separator,
a vertical tab or an em space is ERASED by them, so a character the reader cannot read (OUTSIDE) vanishes before the
reader or a unit check ever sees it (a header 'Dose (mg<EM SPACE>/kg)' became the readable unit 'mg /kg'). The modules
whose text reaches the number reader, a unit or a glyph check use these instead: only the four GAP characters (space,
tab, line feed, carriage return) separate words, and every other character stays where it is.

``blank`` is the glyph-level question 'does this character put nothing a reader could take for a number character on the
page': a GAP character or a thin space (the readable spaces), nothing else. An OUTSIDE space is NOT blank: the glyph
checks judge it like any other character, and the reader leaves its numbers unresolved.
"""
from __future__ import annotations

import re

from app.evidence.numgrammar.alphabet import GAP, char_class
from app.evidence.numgrammar.equivalents import THIN_SOURCES

_GAP_RUN = re.compile(char_class(GAP, negate=False) + "+")
_GAPS = "".join(sorted(GAP))
_BLANK = GAP | THIN_SOURCES


def split(text: str) -> list[str]:
    """The words of *text*: maximal runs of non-GAP characters (``str.split()`` restricted to the GAP set)."""
    return [w for w in _GAP_RUN.split(text) if w]


def strip(text: str) -> str:
    """*text* without GAP characters at either end."""
    return text.strip(_GAPS)


def collapse(text: str) -> str:
    """*text* with every GAP run as one space, trimmed (``" ".join(text.split())`` restricted to the GAP set)."""
    return " ".join(split(text))


def blank(text: str) -> bool:
    """*text* holds only GAP characters and thin spaces (or nothing)."""
    return all(c in _BLANK for c in text)


__all__ = ["blank", "collapse", "split", "strip"]
