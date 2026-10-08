"""The round-2 specification of the spelling tables around the reader, written a SECOND time as bare code points (the
first copy is ``numgrammar/equivalents.py``; ``tests/evidence_r6_pins.py`` holds the alphabet).

The snippet locator, the unit comparison and the header key fold characters through ONE table, ``fold()`` (the approved
equivalents), plus three explicit tables that never touch a number: seven ligatures (locator only), the infinity sign of
a header key, and the case pairs of the letters the alphabet lists. ``spec_locator`` is the independent oracle of what
``normalise()`` does to one character, ``spec_unit_resolved`` of which characters a unit may hold, ``spec_key`` of what
a header key keeps: the layer-B sweep and the tests compare every code point against them.
"""
from __future__ import annotations

from tests import evidence_r6_pins as P

LIGATURE_PIN = {0xFB00: "ff", 0xFB01: "fi", 0xFB02: "fl", 0xFB03: "ffi", 0xFB04: "ffl", 0xFB05: "st", 0xFB06: "st"}
HEADER_SPELLING_PIN = {0x221E: "inf"}
# the capitals whose lower case the locator, a unit and a header key fold: ASCII A-Z, three Greek, five accented Latin
CASE_PIN = {**{c: c + 32 for c in range(0x41, 0x5B)}, 0x394: 0x3B4, 0x3A9: 0x3C9, 0x3A3: 0x3C3,
            0xC1: 0xE1, 0xC5: 0xE5, 0xD6: 0xF6, 0xD8: 0xF8, 0xC7: 0xE7}

# every letter the alphabet lists: the 52 ASCII letters, the 17 Greek letters and the 29 accented Latin ones (r7: all in
# INSIDE or CONTEXT; the LINE list is deleted and chi is outside)
LETTERS_PIN = frozenset(
    chr(cp) for cp in (*P.INSIDE_CPS, *P.CONTEXT_CPS) if chr(cp).isalpha())
SPACES_PIN = frozenset(" \t\n\r   ")                      # GAP plus the thin space and its two sources
KEY_STRIP_PIN = SPACES_PIN | frozenset("_-,.−–‐‑")   # what a header key drops (hyphens fold to '-')


def equiv_char(ch: str) -> str:
    return chr(P.EQUIV_PIN.get(ord(ch), ord(ch)))


def spec_locator(ch: str) -> str:
    """What ``normalise("a" + ch + "b")`` holds between the a and the b: a space for a space, the ligature's letters, the
    approved equivalent, the lower case of a listed capital; every other character as it is."""
    if ch in SPACES_PIN:
        return " "
    ch = equiv_char(ch)
    if ord(ch) in LIGATURE_PIN:
        return LIGATURE_PIN[ord(ch)]
    return chr(CASE_PIN.get(ord(ch), ord(ch)))


def spec_unit_token(ch: str) -> bool:
    """The printed-unit reader (``record_checks._UNIT_TOKEN_RE``) starts a unit with exactly one of: a percent sign of the
    alphabet (% or per mille) or a letter of INSIDE / CONTEXT (the 52 ASCII letters, the 17 Greek and the 29 accented
    Latin ones). The text
    arrives folded, so an approved equivalent's source (the micro sign) is not matched itself."""
    cp = ord(ch)
    return ch in "%\u2030" or (ch.isalpha() and (cp in P.INSIDE_SET or cp in P.CONTEXT_SET))


def spec_unit_resolved(ch: str) -> bool:
    """A unit may hold INSIDE, CONTEXT and GAP characters (and the approved equivalents' sources): nothing else."""
    cp = P.EQUIV_PIN.get(ord(ch), ord(ch))
    return P.expected_class(cp) in ("inside", "context", "gap")
