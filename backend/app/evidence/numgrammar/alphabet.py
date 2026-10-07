"""The allowed alphabet of the numeric reader: five classes, every character in exactly one (r6 owner decision).

A number is readable only when it matches a small explicit grammar made of an explicit alphabet. There is no
Unicode category, no code-point blocklist and no ``str.is*`` test anywhere in the reader: ``cls(ch)`` is a
membership test on the literal sets below. Everything not listed is OUTSIDE.

``inside``  may stand inside a numeric token in a grammar position (membership means "may touch the digits"; the
            literal of a token stays the number itself: ``45`` of ``45%``, ``0.5`` of ``<0.5``).
``gap``     one visual separator: a run of any length and mix of the four counts as ONE gap.
``context`` allowed in a word within one visual separator of a token (units, words, brackets, Greek letters).
``line``    allowed anywhere on the token's line, but NOT inside the token's word nor in the word before or
            after it (footnote marks, typographic symbols, subscripts, accented author names).
``outside`` everything else, and the pdfminer marker ``(cid:N)`` (the text layer's U+FFFD): taints the whole line.

Every set is frozen data with its code points written as ``\\u`` escapes, and
``tests/test_evidence_r6_sets.py`` prints and pins their exact contents: an addition is a visible diff.
The admission procedure for a new character is one corpus hit on a digit-bearing line in ordinary prose
(never a mis-mapped glyph), written here with its code point, plus a regenerated oracle golden file.

Counts: inside 45, gap 4, context 74, line 76, equivalents 10 (equivalents.py). Design A's count of 78 for the
context set listed the four ASCII letters e E x X twice (they are INSIDE: exponent marker and multiplier);
a character belongs to exactly one class, so the context set holds the other 48 ASCII letters.
"""
from __future__ import annotations

import re

from app.evidence.numgrammar.equivalents import EQUIV, THIN

# ------------------------------------------------------------------- INSIDE (45) ---
DIGITS = frozenset("0123456789")                       # U+0030..U+0039
SIGNS = frozenset("+-")                                # U+002B U+002D
SEPARATORS = frozenset(".,'" + THIN + "\u00b7")        # U+002E U+002C U+0027 U+2009 THIN SPACE, U+00B7 MIDDLE DOT
EXP_MARKERS = frozenset("eE")                          # U+0065 U+0045
MULTIPLIERS = frozenset("xX*\u00d7\u00b7")             # U+0078 U+0058 U+002A U+00D7 (U+00B7 doubles as one)
CARET = "^"                                            # U+005E
SUP_DIGITS = frozenset(                                # U+2070 U+00B9 U+00B2 U+00B3 U+2074..U+2079
    "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079")
SUP_SIGNS = frozenset("\u207a\u207b")                  # U+207A SUPERSCRIPT PLUS, U+207B SUPERSCRIPT MINUS
PERCENTS = frozenset("%\u2030")                        # U+0025 U+2030 PER MILLE
PLUS_MINUS = "\u00b1"                                  # U+00B1
QUALIFIERS = frozenset("<>\u2264\u2265~\u2248")        # U+003C U+003E U+2264 U+2265 U+007E U+2248

INSIDE = frozenset().union(DIGITS, SIGNS, SEPARATORS, EXP_MARKERS, MULTIPLIERS, CARET, SUP_DIGITS, SUP_SIGNS,
                           PERCENTS, PLUS_MINUS, QUALIFIERS)

# -------------------------------------------------------------------- GAP (4) ---
GAP = frozenset(" \t\n\r")                             # U+0020 U+0009 U+000A U+000D

# ----------------------------------------------------------------- CONTEXT (74) ---
CONTEXT_LETTERS = frozenset(                           # the 48 ASCII letters that are not e E x X (INSIDE)
    "ABCDFGHIJKLMNOPQRSTUVWYZ"                         # U+0041..U+005A without E (U+0045) and X (U+0058)
    "abcdfghijklmnopqrstuvwyz")                        # U+0061..U+007A without e (U+0065) and x (U+0078)
GREEK = frozenset(                                     # the 17 Greek letters seen within one word of a digit
    "\u03bc\u03c9\u03b2\u03b1\u03c4\u03bb\u03b7\u03b8"  # mu omega beta alpha tau lambda eta theta
    "\u03b5\u03b3\u03c3\u03c7\u03c1\u03c0"             # epsilon gamma sigma chi rho pi
    "\u0394\u03a9\u03a3")                               # Delta Omega Sigma
OPENERS_BR = frozenset("([")                           # U+0028 U+005B
CLOSERS_BR = frozenset(")]")                           # U+0029 U+005D
CONTEXT_PUNCT = frozenset(";:/=\u00b0")                # U+003B U+003A U+002F U+003D U+00B0 DEGREE SIGN
CONTEXT = frozenset().union(CONTEXT_LETTERS, GREEK, OPENERS_BR, CLOSERS_BR, CONTEXT_PUNCT)

# -------------------------------------------------------------------- LINE (76) ---
LINE_ASCII = frozenset(                                # 13: remaining printable ASCII
    "!\"#$&?@\\_`{|}")                                 # U+0021 22 23 24 26 3F 40 5C 5F 60 7B 7C 7D
LINE_TYPOGRAPHIC = frozenset(                          # 21
    "\u2018\u201c\u201d"                               # left single, left and right double quotation mark
    "\u2014\u2026\u2022"                               # em dash, ellipsis, bullet
    "\u00a9\u00ae\u2122"                               # copyright, registered, trade mark
    "\u2020\u2021\u00a7\u00b6\u2016\u2032"             # dagger, double dagger, section, pilcrow, double bar, prime
    "\u2192\u2190\u2191\u2193"                         # arrows
    "\u221e\u00bd")                                    # infinity, one half
LINE_SUBSCRIPTS = frozenset(                           # 12: U+2080..U+2089, U+208A, U+208B
    "\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089\u208a\u208b")
LINE_KAPPA = frozenset("\u03ba")                       # 1: U+03BA GREEK SMALL LETTER KAPPA
LINE_ACCENTED = frozenset(                             # 29 accented Latin letters seen in author names
    "\u00e9\u00e8\u00eb\u00e2\u00e1\u00c1\u00e3\u00e6\u00c5\u00e4\u00f6\u00d6"   # e-acute e-grave e-diaeresis ...
    "\u00fc\u00ed\u00ef\u00f3\u00f8\u00d8\u00fa\u0171\u00f1\u0144\u00e7\u00c7"   # u-diaeresis i-acute o-acute ...
    "\u0107\u010d\u011f\u0142\u015b")                  # c-acute c-caron g-breve l-stroke s-acute
LINE = frozenset().union(LINE_ASCII, LINE_TYPOGRAPHIC, LINE_SUBSCRIPTS, LINE_KAPPA, LINE_ACCENTED)

# ------------------------------------------------------------------ role sets used by the reader ---
LETTERS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz") | GREEK   # a word led by one is a label
ASCII_LETTERS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
STARTERS = frozenset("([<>\u2264\u2265~\u2248=\u00b1")  # a token may start right after one of these in a word
JOINERS = frozenset("-\u00b1xX*\u00d7\u00b7")            # a token may start right after one that follows a token
TRAILING = frozenset(")],;:.'")                          # may follow a token glued (not followed by a digit)
SUP_CHARS = SUP_DIGITS | SUP_SIGNS
SUP_TO_ASCII = str.maketrans("\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079\u207a\u207b",
                             "0123456789+-")

# the pdfminer marker for a glyph it could not map (a lost >= +- x deg micro minus): one OUTSIDE character
CID_MARKER = re.compile(r"\(cid:[0-9]+\)")


def cls(ch: str) -> str:
    """'inside' | 'context' | 'line' | 'gap' | 'outside' for ONE character. A character in EQUIV classifies
    as its target; the "(cid:" marker is a string rule (CID_MARKER) and is not a character."""
    if len(ch) != 1:
        raise ValueError("cls() classifies exactly one character")
    ch = EQUIV.get(ch, ch)
    if ch in INSIDE:
        return "inside"
    if ch in GAP:
        return "gap"
    if ch in CONTEXT:
        return "context"
    if ch in LINE:
        return "line"
    return "outside"


def char_class(chars: frozenset[str], *, negate: bool) -> str:
    body = "".join(re.escape(c) for c in sorted(chars))
    return "[" + ("^" if negate else "") + body + "]"


# word tier: a character that may not touch a token's word (LINE or OUTSIDE); line tier: OUTSIDE only. Both are
# allowlist classes built from the literal sets (never hand-written) and the tests prove they agree with cls()
# on every code point.
_WORD_OK = INSIDE | CONTEXT | GAP | frozenset(EQUIV)
_LINE_OK = _WORD_OK | LINE
WORD_TIER_RE = re.compile(char_class(_WORD_OK, negate=True))
LINE_TIER_RE = re.compile(char_class(_LINE_OK, negate=True))
INSIDE_ONLY_RE = re.compile(char_class(INSIDE, negate=False) + "+")
GAP_CLASS = char_class(GAP, negate=False)
WORD_RE = re.compile(char_class(GAP, negate=True) + "+")
