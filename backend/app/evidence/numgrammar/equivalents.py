"""The ONE table of approved equivalents of the numeric reader (r6 owner decision).

A character listed here is replaced 1:1 by its target before anything is scanned (``str.translate``), so every
offset stays native: ``text[start:end]`` of a token is the literal as printed, equivalents included. Nothing
else is normalised in the reader: no NFKC, no case folding, no category tests. Ten entries, each one
corpus- or test-motivated (see docs/evidence-pipeline.md, "How numbers are matched"):

====== ============================= ====== ================================================================
source name                           target why
====== ============================= ====== ================================================================
U+2212 MINUS SIGN                     -      199 hits / 19 real papers; ``-0.45`` written with a true minus
U+2013 EN DASH                        -      sign (``-11.8``) and range dash (``12-15``) in real tables
U+2010 HYPHEN                         -      NEJM sets every hyphen as U+2011; ``Physiologically-based``
U+2011 NON-BREAKING HYPHEN            -      same glyph and meaning as ``-``
U+2019 RIGHT SINGLE QUOTATION MARK    '      Swiss digit grouping ``24'560`` and apostrophes in prose
U+202F NARROW NO-BREAK SPACE          U+2009 thin-space digit grouping (parse_guard folds both to U+2009)
U+00A0 NO-BREAK SPACE                 U+2009 same
U+00B5 MICRO SIGN                     U+03BC ``245 ug`` units written with the micro sign
U+223C TILDE OPERATOR                 ~      34 real hits as "approximately" before a number: a qualifier
U+25E6 WHITE BULLET                   U+00B0 LaTeX papers print the degree sign ``37 C`` as U+25E6
====== ============================= ====== ================================================================

Rejected candidates stay OUTSIDE (alphabet.py): U+22C5 / U+2219 (look-alike multipliers: every near-digit hit is
an attack fixture or a lecture equation, and as a multiplier they would be fail-open), U+2014 EM DASH (a list
dash or "not available" in tables), U+2012 FIGURE DASH, U+02BC, U+2032 and the other look-alikes.

Round 2: the three explicit tables AROUND the reader. The reader itself (``tokens``) reads ONLY ``fold`` / ``EQUIV``.
The snippet locator (``normalize.py``), the unit comparison (``unit_norm.py``) and the header key
(``extract_rules.norm_key``) used to carry NFKC and Unicode case folding of their own, which equated characters the
reader keeps apart. They now fold through ``fold`` plus these tables, and a character none of them lists stays as it is
(so it equals nothing else and, in a unit, makes the unit unresolved):

``LIGATURES``        locator only, 1 -> n letters: the seven Latin presentation forms U+FB00..U+FB06. NO hit in the 449
                     parse caches (pdfminer spells them out through the font's ToUnicode map). Kept because a snippet
                     typed with the letters for a page that prints the ligature is the commonest honest drift, and
                     because the expansion is ASCII letters only: it cannot make a number readable, so the locator
                     stays a finder. The owner may delete the table: a ligature page then fails
                     ``snippet_not_on_page`` (fail closed).
``HEADER_SPELLINGS`` header keys only: the infinity sign (U+221E) is the word ``inf`` (``AUC0-<infinity>`` is
                     ``AUC0-inf``; 2 corpus headers; template aliases are written either way).
``CASE_PAIRS``       the 34 capitals of the letters the alphabet lists (ASCII A-Z, Greek Delta Omega Sigma, five
                     accented Latin capitals) and their lower case, 1:1. Explicit pairs, not ``str.casefold``: the
                     Kelvin sign, the long s and the other characters Unicode folds into ASCII letters are not
                     equated with them.
"""
from __future__ import annotations

import re
from functools import cache

THIN = "\u2009"          # THIN SPACE: the digit-group separator; a gap anywhere else

# source -> target, applied once per character. The test pins this dict entry by entry.
EQUIV: dict[str, str] = {
    "\u2212": "-",       # MINUS SIGN
    "\u2013": "-",       # EN DASH
    "\u2010": "-",       # HYPHEN
    "\u2011": "-",       # NON-BREAKING HYPHEN
    "\u2019": "'",       # RIGHT SINGLE QUOTATION MARK
    "\u202f": THIN,      # NARROW NO-BREAK SPACE
    "\u00a0": THIN,      # NO-BREAK SPACE
    "\u00b5": "\u03bc",  # MICRO SIGN -> GREEK SMALL LETTER MU
    "\u223c": "~",       # TILDE OPERATOR
    "\u25e6": "\u00b0",  # WHITE BULLET -> DEGREE SIGN
}

_TRANSLATE = str.maketrans(EQUIV)
# the characters that are a thin space: U+2009 itself and every source the table maps to it (parse_guard keeps exactly
# these inside a number; any other space character is a plain word break)
THIN_SOURCES = frozenset(k for k, v in EQUIV.items() if v == THIN) | {THIN}
# a thin space groups digits only BETWEEN two digits; anywhere else it is a plain gap (1:1 rewrite)
_THIN_GAP = re.compile("(?<![0-9])" + THIN + "|" + THIN + "(?![0-9])")


@cache
def _claim_group() -> re.Pattern[str]:
    """Claim mode only (a snippet or value a model typed with plain spaces): an ASCII space between a 1-3 digit head
    and a 3-digit group reads as a thin space ('18 900', '1 234 567'); '12 1500' and '0.123 4' stay two tokens. The
    head must itself start a number: it follows the start of the text, a gap, a starter or a sign, never a digit, a
    separator or a letter ('229.64 233.64' are two decimals, 'CYP2B6 125.0' is a label and a number). The class is
    built from the alphabet's GAP, STARTERS and SIGNS (imported here, not at module level: alphabet imports this
    module)."""
    from app.evidence.numgrammar.alphabet import GAP, SIGNS, STARTERS, char_class
    before = char_class(GAP | STARTERS | SIGNS, negate=True)
    return re.compile("(?<!" + before + ")([0-9]{1,3}) (?=[0-9]{3}(?![0-9]))")


def fold(text: str, *, claim: bool = False) -> str:
    """The text as the reader scans it: equivalents translated 1:1, a thin space not between two digits
    rewritten to U+0020, and (claim mode) a thousands-position ASCII space read as a thin space. Same length
    as *text*: offsets are native."""
    out = text.translate(_TRANSLATE)
    if THIN in out:
        out = _THIN_GAP.sub(" ", out)
    if claim and " " in out:
        out = _claim_group().sub("\\1" + THIN, out)
    return out


# ------------------------------------------------------- the tables around the reader (round 2) ---
# locator only: a presentation-form ligature is its letters (one character -> several: the locator's index map repeats
# the offset). U+FB05 is the long-s ligature and is spelled with a plain s.
LIGATURES: dict[str, str] = {
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl", "\ufb05": "st", "\ufb06": "st",
}

# header keys only (extract_rules.norm_key): the infinity sign of AUC0-inf / AUC0-infinity
HEADER_SPELLINGS: dict[str, str] = {"\u221e": "inf"}

# capital -> lower case, 1:1, for the letters of the alphabet that have a case (ASCII, three Greek, five accented Latin)
CASE_PAIRS: dict[str, str] = {
    **{chr(c): chr(c + 32) for c in range(0x41, 0x5B)},                   # A-Z -> a-z
    "\u0394": "\u03b4", "\u03a9": "\u03c9", "\u03a3": "\u03c3",         # Delta Omega Sigma
    "\u00c1": "\u00e1", "\u00c5": "\u00e5", "\u00d6": "\u00f6",         # A-acute A-ring O-diaeresis
    "\u00d8": "\u00f8", "\u00c7": "\u00e7",                             # O-stroke C-cedilla
}
_CASE_TABLE = str.maketrans(CASE_PAIRS)


def casefold_letters(text: str) -> str:
    """*text* with the capitals of ``CASE_PAIRS`` in lower case; every other character (a Kelvin sign, a fullwidth or
    mathematical letter, a Cyrillic capital) is unchanged. Same length: offsets are native."""
    return text.translate(_CASE_TABLE)
