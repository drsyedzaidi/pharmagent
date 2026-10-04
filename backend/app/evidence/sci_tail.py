"""Scientific-notation tails the number grammar of normalize.py cannot read (fail closed).

normalize.py reads 'M x 10^e' / 'M x 10<superscript>' / 'M e-3' with ASCII digits and a short list of real
multiplication signs. Anything else that LOOKS like it, though, must not leave M as a plain number: a
reader sees 1.2 x 10^3 = 1200, never 1.2. This module finds the extent of such a tail so the token that
covers it is unresolved (value None) and none of its pieces ('10', '3') stands as a number of its own:

- a 'ten' that is not ASCII '10' after a multiplier: full-width / Arabic-Indic / mathematical / circled /
  roman / superscript digits, 'l0', '1O', Cyrillic 'І0';
- an exponent outside the grammar after '10': subscript or circled digits, '^' with a gap or a
  superscript, '**', parentheses, arrows, a sign before a non-ASCII digit;
- a symbol between the mantissa and the ten that is not a real multiplier (division sign, replacement
  character, private use, unassigned, junk next to the multiplier).
"""
from __future__ import annotations

import re
import unicodedata

GAPS = " \x00"                                    # a word gap in folded text (strict mode turns ASCII gaps into NUL)
MULT_CHARS = "xX*×·⋅∙•⨯⨉✕✖∗⋆⁎＊хХ"
_BOUNDARY = frozenset("±∓≤≥≦≧≈≃≅≠∼<>=")          # a comparison / plus-minus sign separates numbers
_ONE_LIKE = frozenset("Il|!ІіӀΙⅠⅼＩｌ")
_ZERO_LIKE = frozenset("0OoОоΟοＯｏ")
_EXP_GLYPHS = frozenset("⁺⁻⁽⁾ⁿⁱ₊₋₍₎↑∧⌃ˆ˄‸︿")
_RUN_MAX = 8
_ROUGH_TEN = re.compile(r"10(?!\d|[.,]\d)")
_PROPER_EXP = re.compile(r"(?:\^[ \x00]?[+-]?\d{1,4}(?!\d)|[⁺⁻]?[⁰¹²³⁴-⁹]{1,4}|[+-]\d{1,4}(?!\d))")
_TAIL_DIGITS = re.compile(r"[ \x00]{0,2}[+-]?[1-9]\d{0,3}(?!\d)")


def _digitlike(ch: str) -> bool:
    return not ch.isascii() and unicodedata.category(ch)[0] == "N"


def _exp_glyph(ch: str) -> bool:
    cp = ord(ch)
    return ch in _EXP_GLYPHS or 0x2070 <= cp <= 0x209F or (not ch.isascii() and _digitlike(ch))


def _exponent_starts(t: str, j: int) -> bool:
    """True when t[j] starts exponent text the grammar does not read ('^' after a gap, '**', a script
    glyph, a sign before a script digit)."""
    c = t[j]
    if c == "^":
        return True
    if c == "*":
        return t[j + 1:j + 2] == "*"
    if not c.isascii():
        return _exp_glyph(c)
    return c in "+-" and j + 1 < len(t) and not t[j + 1].isascii() and (_digitlike(t[j + 1]) or t[j + 1] in "⁽₍")


def exponent_end(t: str, i: int) -> int | None:
    """End of exponent-looking text outside the grammar right after a '10' ending at *i*, else None."""
    j = i
    while j < len(t) and t[j] in GAPS and j - i < 2:
        j += 1
    if j >= len(t) or not _exponent_starts(t, j):
        return None
    k = j
    while k < len(t) and k - j < _RUN_MAX and (t[k] in "^*+-()" or t[k].isdigit() and t[k].isascii()
                                              or t[k] in GAPS or (not t[k].isascii() and _exp_glyph(t[k]))):
        k += 1
    while t[k - 1] in GAPS:
        k -= 1
    return k


def _mult_candidate(c: str) -> bool:
    """A character a reader may take for a multiplication sign: a real one, or any non-ASCII symbol that
    is not a letter, digit, space or comparison."""
    if c in MULT_CHARS:
        return True
    return not c.isascii() and c not in _BOUNDARY and unicodedata.category(c)[0] not in "LNZ"


def _ten_end(t: str, j: int) -> tuple[int, bool] | None:
    """(end, is ASCII ten) of a 'ten' at t[j]: '10', or a look-alike (non-ASCII digits, 'l0', '1O')."""
    c0, c1 = t[j:j + 1], t[j + 1:j + 2]
    if not c0:
        return None
    if c0 == "1" and c1 == "0":
        m = _ROUGH_TEN.match(t, j)
        return (m.end(), True) if m else None
    if _digitlike(c0):
        k = j + 1
        while k < len(t) and k - j < 4 and (_digitlike(t[k]) or t[k] in "01"):
            k += 1
        return k, False
    pair_like = (c0 in _ONE_LIKE and c1 in _ZERO_LIKE) or (c0 == "1" and (c1 in _ZERO_LIKE or _digitlike(c1)))
    if pair_like and not t[j + 2:j + 3].isalpha():
        return j + 2, False
    return None


def scientific_end(t: str, end: int) -> int | None:
    """End of an unreadable scientific tail after the mantissa ending at *end*, else None: a multiplier
    candidate, up to two junk characters, then a 'ten' (any look-alike) and its exponent text."""
    i = end
    while i < len(t) and t[i] in GAPS and i - end < 2:
        i += 1
    if i >= len(t) or not _mult_candidate(t[i]):
        return None
    j, junk = i + 1, 0
    while j < len(t) and junk < 4 and (t[j] in GAPS or (not t[j].isascii() and _mult_candidate(t[j]))):
        j += 1
        junk += 1
    ten = _ten_end(t, j)
    if ten is None:
        return None
    k, ascii_ten = ten
    proper = _PROPER_EXP.match(t, k)
    if proper is not None:
        return proper.end()
    k = exponent_end(t, k) or k
    if not ascii_ten:
        tail = _TAIL_DIGITS.match(t, k)
        k = tail.end() if tail else k
    return k
