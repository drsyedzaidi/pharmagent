"""Unit identity: the comparison form of a unit, the equivalence table, and the SI prefix whose
case matters ('MIU' mega vs 'mIU' milli: a 10^9 error that casefolding would hide).

A unit is folded through the reader's own table (``fold``: the approved equivalents) and the alphabet, nothing else
(r6 round 2: it used NFKC and ``str.is*``, a second normalisation that equated a one-character compatibility sign for
a unit with its letters). A unit may hold only INSIDE, CONTEXT and GAP characters (letters, digits,
``/ . * ^ - % °``, brackets, superscripts, the approved equivalents' sources); one character outside makes the unit
UNRESOLVED: its comparison form is the unit itself behind a marker, so it equals no other unit (not even another case
of itself), is recognised as no unit and is never given a mega / milli verdict.

normalise_unit casefolds the alphabet's letters; prefix_conflict() is the case-sensitive check made beside it.
"""
from __future__ import annotations

import re

from app.evidence.letter_tables import CASE_PAIRS, UPPER_LETTERS, casefold_letters
from app.evidence.numgrammar.alphabet import (
    ASCII_LETTERS,
    DECIMAL_MARKS,
    DIGITS,
    GAP,
    GREEK,
    LETTERS,
    LIST_MARKS,
    MULT_LETTERS,
    MULTIPLIERS,
    OUTSIDE_RE,
    PERCENTS,
    SUP_TO_ASCII,
    THIN_GROUP,
    UNIT_BASES,
    UNIT_PREFIXES,
    char_class,
)
from app.evidence.numgrammar.equivalents import fold

UNRESOLVED = "\x00"            # starts the comparison form of a unit that holds a character outside the alphabet
(_MICRO,) = UNIT_PREFIXES & GREEK  # GREEK SMALL LETTER MU: the micro prefix (and the MICRO SIGN equivalent's target)
_WORD = char_class(ASCII_LETTERS | GREEK | DIGITS, negate=False)
_WHOLE = ("(?<!" + _WORD + ")", "(?!" + _WORD + ")")         # a whole word: no word character right before or after it
_UNIT_SYNONYMS = tuple((re.compile(_WHOLE[0] + pattern + _WHOLE[1]), rep) for pattern, rep in (
    ("(hrs?|hours?)", "h"), ("(liters?|litres?)", "l"), ("mcg", "ug"), ("(seconds?|sec)", "s")))
_STRIP = GAP | THIN_GROUP | (DECIMAL_MARKS - LIST_MARKS) | (MULTIPLIERS - MULT_LETTERS)   # space . * times dot
_UNIT_STRIP = re.compile(char_class(_STRIP, negate=False) + "+")
_EQUIV = {"mcg": "ug", "cc": "ml", "hr": "h", "hrs": "h", "mins": "min", "sec": "s", "lit": "l"}
# the printed-unit symbols a comparison form may be segmented into (record_checks.recognised_unit: 'ngh' = 'ng' + 'h')
# and the words that never continue a printed unit. Unit COMPARISON data, never reader data: the number reader's glued
# unit production (numgrammar/units.py) has its own corpus-counted terminals, and a symbol here never makes a number
# readable
UNIT_SYMBOLS = ("kg", "g", "mg", "ug", "ng", "pg", "mol", "mmol", "umol", "nmol", "pmol", "l", "ml", "dl", "ul",
                "h", "min", "s", "d", "wk", "iu", "miu", "u", *sorted(PERCENTS), "kda", "m2", "cells", "cell", "10^6",
                "10^9", "bpm", "mmhg", "x", "fold")
NOT_CONTINUATIONS = frozenset({casefold_letters(c) for c in MULT_LETTERS} | {"fold"} | PERCENTS)
# mega / milli: the terminals that are a prefix in one case and a unit base in the other (m vs M), derived from the
# unit production's terminals and the case pairs (r7 round 3: never typed)
_CASE_PREFIXES = frozenset(c for cap, low in CASE_PAIRS.items() if low in UNIT_PREFIXES and cap in UNIT_BASES
                           for c in (cap, low))


def _spell(unit: str | None) -> str | None:
    """The unit as the reader scans it (``fold``, superscripts spelled with ASCII digits and signs), or None when it
    holds a character outside INSIDE, CONTEXT and GAP."""
    folded = fold(unit or "")
    return None if OUTSIDE_RE.search(folded) else folded.translate(SUP_TO_ASCII)


def unit_resolved(unit: str | None) -> bool:
    """True when every character of *unit* is one the alphabet allows in a word (see the module docstring)."""
    return _spell(unit) is not None


def normalise_unit(unit: str | None) -> str:
    """Comparison form: equivalents (µ -> μ), superscripts as ASCII, casefold of the alphabet's letters, μ -> 'u',
    hr -> h, separators (space · * × .) removed. An unresolved unit is UNRESOLVED plus the folded unit, character for
    character."""
    spelled = _spell(unit)
    if spelled is None:
        return UNRESOLVED + fold(unit or "")
    u = casefold_letters(spelled).replace(_MICRO, "u")
    for pattern, rep in _UNIT_SYNONYMS:
        u = pattern.sub(rep, u)
    return _UNIT_STRIP.sub("", u)


def canonical_unit(unit: str | None) -> str:
    """normalise_unit with each '/'-piece mapped through the equivalence table (mcg = ug = µg, mL = ml)."""
    u = normalise_unit(unit)
    if u.startswith(UNRESOLVED):
        return u
    return "/".join(_EQUIV.get(piece, piece) for piece in u.split("/"))


def _pieces(unit: str | None) -> list[str] | None:
    spelled = _spell(unit)
    return None if spelled is None else [_UNIT_STRIP.sub("", p) for p in spelled.split("/")]


def _prefix(piece: str) -> tuple[str, str]:
    """(case-significant prefix letter or '', the rest) of one unit piece ('mIU' -> ('m', 'IU'))."""
    if len(piece) > 1 and piece[0] in _CASE_PREFIXES and piece[1] in LETTERS:
        return piece[0], piece[1:]
    return "", piece


def prefix_conflict(proposed: str | None, printed: str | None) -> str | None:
    """'mismatch' when the two units differ in a mega/milli prefix ('MIU' vs 'mIU', 'Mg' vs 'mg');
    'ambiguous' when an all-capitals printed unit could be the proposed one set in capitals ('MG',
    'MG/ML' for mg/mL; never 'MIU' for mIU, whose other letters are capitals anyway); else None. An unresolved unit
    (a character outside the alphabet) has no case to judge: None."""
    a, b = _pieces(proposed), _pieces(printed)
    if a is None or b is None or len(a) != len(b):
        return None
    pairs = [(_prefix(pa), _prefix(pb)) for pa, pb in zip(a, b)]
    if not any(xa and xb and xa != xb for (xa, _), (xb, _) in pairs):
        return None
    letters = [ch for piece in b for ch in piece if ch in LETTERS]
    caps_page = bool(letters) and all(ch in UPPER_LETTERS for ch in letters)
    lower_rest = any(ch in LETTERS and ch not in UPPER_LETTERS for (_, ra), _ in pairs for ch in ra)
    return "ambiguous" if caps_page and lower_rest else "mismatch"


def same_unit(a: str | None, b: str | None) -> bool:
    """True when two record units name the same unit (equivalence table, prefix case); a unit
    present on one side only is a different unit, two missing units are the same."""
    return canonical_unit(a) == canonical_unit(b) and prefix_conflict(a, b) is None
