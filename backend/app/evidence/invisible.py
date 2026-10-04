"""Characters that render as nothing, or that change a neighbouring character, and how the number
reader treats them (fail closed).

Four kinds of character (see kind()):

- SOFT: renders as nothing (category Cf, Default_Ignorable_Code_Point: soft hyphen, zero-width
  space / joiners, word joiner, variation selectors, tag characters, Mongolian free variation
  selectors, Hangul fillers). Folded OUT of the text with an offset map, so the number grammar
  reads what a person sees: '1.2 x<ZWJ>10 3' is read exactly like '1.2 x 10 3'. Only where a soft
  character can stand for a sign or a digit-group break is the token unresolved: directly before
  a number (a soft hyphen is often a minus), between the digits / separators of one number, and
  right before an exponent sign or digits.
- HARD: bidirectional controls, interlinear annotation marks, and the Cf characters that DO draw
  (Arabic number signs): a number inside or adjacent to one may display differently from its
  logical order, so the token is unresolved.
- MARK: a combining or enclosing mark (Mn / Me / Mc). Removed from the text and remembered on the
  character it is attached to: a struck-through, dotted or enclosed digit, decimal point, sign
  or multiplier is not the plain character, so a token touching one is unresolved.
- anything else (kind 0) is ordinary.
"""
from __future__ import annotations

import re
import unicodedata
from bisect import bisect_left
from functools import lru_cache
from typing import NamedTuple

SOFT, HARD, MARK = 1, 2, 3

# Default_Ignorable_Code_Point (Unicode DerivedCoreProperties) plus every Cf character
_IGNORABLE = ((0x00AD, 0x00AD), (0x034F, 0x034F), (0x061C, 0x061C), (0x115F, 0x1160), (0x17B4, 0x17B5),
              (0x180B, 0x180F), (0x200B, 0x200F), (0x202A, 0x202E), (0x2060, 0x206F), (0x3164, 0x3164),
              (0xFE00, 0xFE0F), (0xFEFF, 0xFEFF), (0xFFA0, 0xFFA0), (0xFFF0, 0xFFF8), (0x1BCA0, 0x1BCA3),
              (0x1D173, 0x1D17A), (0xE0000, 0xE0FFF))
_HARD = ((0x0600, 0x0605), (0x061C, 0x061C), (0x06DD, 0x06DD), (0x070F, 0x070F), (0x0890, 0x0891),
         (0x08E2, 0x08E2), (0x200E, 0x200F), (0x202A, 0x202E), (0x2066, 0x2069), (0xFFF9, 0xFFFB),
         (0x110BD, 0x110BD), (0x110CD, 0x110CD))
_NON_ASCII = re.compile(r"[^\x00-\x7f]")
_NUMBER_CHARS = frozenset("0123456789.,'·‧˙٫")
_ASCII_DIGITS = frozenset("0123456789")
_SIGN_OR_SUP = frozenset("+-⁺⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def _within(cp: int, ranges: tuple[tuple[int, int], ...]) -> bool:
    return any(lo <= cp <= hi for lo, hi in ranges)


@lru_cache(maxsize=8192)
def kind(ch: str) -> int:
    """0 ordinary, SOFT, HARD or MARK (see the module docstring)."""
    cp = ord(ch)
    if cp < 0x80:
        return 0
    if _within(cp, _HARD):
        return HARD
    cat = unicodedata.category(ch)
    if cat == "Cf" or _within(cp, _IGNORABLE):
        return SOFT
    return MARK if cat in ("Mn", "Me", "Mc") else 0


def is_invisible(ch: str) -> bool:
    """True for a character that renders as nothing (SOFT) or only steers layout (HARD)."""
    return kind(ch) in (SOFT, HARD)


class Cleaned(NamedTuple):
    """Text with soft, hard and mark characters removed. cmap[i] = raw offset of text[i]; the other
    lists hold CLEANED offsets, sorted: *marked* = characters that carry a mark, *hard* / *soft* =
    gaps (offset of the character after the removed one)."""
    text: str
    cmap: list[int]
    marked: list[int]
    hard: list[int]
    soft: list[int]


def clean(text: str) -> Cleaned | None:
    """The cleaned view of *text*, or None when it holds none of those characters (the usual case)."""
    if text.isascii() or not any(kind(ch) for ch in _NON_ASCII.findall(text)):
        return None
    out: list[str] = []
    cmap: list[int] = []
    marked: set[int] = set()
    hard: list[int] = []
    soft: list[int] = []
    for i, ch in enumerate(text):
        k = kind(ch) if ch >= "\x80" else 0
        if k == 0:
            out.append(ch)
            cmap.append(i)
        elif k == SOFT:
            soft.append(len(out))
        elif k == HARD or not out:                   # a mark with nothing to attach to is a hard gap
            hard.append(len(out))
        else:
            marked.add(len(out) - 1)
    return Cleaned("".join(out), cmap, sorted(marked), hard, soft)


def _in_range(sorted_offsets: list[int], lo: int, hi: int) -> bool:
    i = bisect_left(sorted_offsets, lo)
    return i < len(sorted_offsets) and sorted_offsets[i] <= hi


def touches(c: Cleaned, start: int, end: int, mantissa_end: int) -> bool:
    """True when a mark, hard character or sign-ambiguous soft character sits on or next to the
    token cleaned[start:end] (whose mantissa ends at *mantissa_end*): the token is then unresolved."""
    if _in_range(c.marked, start - 1, end) or _in_range(c.hard, start, end):
        return True
    t = c.text
    for g in c.soft[bisect_left(c.soft, start):]:
        if g > end:
            break
        if g == start:                                  # a soft hyphen is often a minus: sign unknown
            return True
        if g >= end:
            continue
        prev, nxt = t[g - 1], t[g]
        if (prev in _NUMBER_CHARS and nxt in _NUMBER_CHARS and (prev in _ASCII_DIGITS or nxt in _ASCII_DIGITS)):
            return True                                 # inside the digits: a group break or a hyphen
        if g >= mantissa_end and (nxt in _SIGN_OR_SUP or (prev in "^eE" and nxt in _ASCII_DIGITS)):
            return True                                 # right before an exponent sign / digits
    return False


def strip_invisible(text: str, *, marks: bool = False) -> str:
    """*text* without soft and hard characters (and, with *marks*, without combining marks)."""
    return "".join(ch for ch in text if (k := kind(ch) if ch >= "\x80" else 0) not in (SOFT, HARD)
                   and not (marks and k == MARK))
