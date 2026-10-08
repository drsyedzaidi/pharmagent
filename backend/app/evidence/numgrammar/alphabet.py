"""The allowed alphabet of the numeric reader: four classes, every character in exactly one (r6 owner decision; r7
deleted the fifth, LINE, and derives the line rule from the four permitted lists).

A number is readable only when it matches a small explicit grammar made of an explicit alphabet. There is no
Unicode category, no code-point blocklist and no ``str.is*`` test anywhere in the reader: ``cls(ch)`` is a
membership test on the literal sets below. Everything not listed is OUTSIDE.

``inside``  may stand inside a numeric token in a grammar position (membership means "may touch the digits"; the
            literal of a token stays the number itself: ``45`` of ``45%``, ``0.5`` of ``<0.5``).
``gap``     one visual separator: a run of any length and mix of the four counts as ONE gap.
``context`` allowed in a word within one visual separator of a token (units, words, brackets, letters, prose marks).
``outside`` everything else, and the pdfminer marker ``(cid:N)`` (the text layer's U+FFFD).

THE DERIVED LINE RULE (r7, owner decision 5). There is no fifth list. A text is READABLE where every character is
INSIDE, CONTEXT, GAP or an approved equivalent's source (``READABLE``; ``OUTSIDE_RE`` matches every other character).
A character that is not readable taints its own word, the words one separator away and every number of its line: the
word, window and line tiers of ``taint.py`` read the one allow-list. The 77 characters of the r6 LINE list were folded
in by one test each: a character goes to CONTEXT only when it can never change the reading of a number beside it (no
digit, sign, dash, separator, multiplier, exponent, fraction, range or comparison look-alike), it is never seen as a
mis-mapped glyph in the corpus, and the owner's decision 4 does not keep it fail-closed. Corpus evidence (the 449
parse caches; digit-bearing lines of the 60-paper design corpus):

* 42 to CONTEXT. Kappa (Greek, ``Between-occasion \u03ba parameters``) and the 29 accented Latin letters (author
  names, ``opioid-na\u00efve subjects``; none between two digits): letters, they lead a label like any letter.
  ``& @`` (``Beal & Sheiner, 1980``, an e-mail address), the quotes U+201C U+201D (``\u201chypo-responsive\u201d
  (Group 1); 9 (21%)``), ``\u00a9 \u00ae \u2122`` (``PK-Sim\u00ae database [58]``), the infinity sign of the
  ``AUC0-\u221e`` header (MacRoman prints a degree sign as it; the degree sign is CONTEXT, so the reading is the same)
  and the footnote marks ``\u00a7 \u00b6 \u2020 \u2021`` (NEJM rows ``Pre-eclampsia\u00a7 688/11 225 (6.1)``,
  ``(0.05 to 0.56)\u00a7``): none of the twelve is ever between two digits in the corpus.
* 35 OUTSIDE. ``# ? _ |`` (decision 4: fail-closed), ``$`` (the Elsevier font's mis-mapped ``\u2265``: ``Patients
  $18 years``), ``\\`` (a mis-mapped ``<``: ``(\\1% of dose)``), U+2018 (a mis-mapped infinity: ``AUC(0\u2013\u2018)``),
  the em dash, the ellipsis (a dash and a range look-alike), the bullet (the middle dot's look-alike), U+2032 and the
  backtick (look-alikes of the apostrophe separator), U+2016 (the bar of decision 4), ``" ! { }`` (code, LaTeX,
  factorial, ditto), the arrows (a relation between two numbers, ``240\u2192184``; the markers of rotated tables read
  backwards, ``51\u2191 551\u2191``), the one-half sign (``12 \u00bd`` is 12.5), the 12 subscript digits and signs
  (digit look-alikes) and chi (decision 8: only in the chi-square production of ``productions.py``).

Counts: inside 45, gap 4, context 115, equivalents 10 (equivalents.py). Design A's count of 78 for the context set
listed the four ASCII letters e E x X twice (they are INSIDE: exponent marker and multiplier); a character belongs to
exactly one class, so the context set holds the other 48 ASCII letters.

Round-1 change: GREEK SMALL LETTER CHI (U+03C7) left CONTEXT. Chi is the one Greek letter that draws like a
multiplier sign: ``1.2 \u03c7 10^3`` is a look-alike multiplier, and as a CONTEXT word it read as a unit and left
``1.2`` standing as the whole number (1000x low). It is outside (r7), so it taints its word, its window and its line.
"""
from __future__ import annotations

import re

from app.evidence.numgrammar.equivalents import EQUIV, THIN

# ------------------------------------------------------------------- INSIDE (45) ---
DIGITS = frozenset("0123456789")                       # U+0030..U+0039
SIGNS = frozenset("+-")                                # U+002B U+002D
SEPARATORS = frozenset(".,'" + THIN + "\u00b7")        # U+002E U+002C U+0027 U+2009 THIN SPACE, U+00B7 MIDDLE DOT
EXP_MARKERS = frozenset("eE")                          # U+0065 U+0045
MULT_LETTERS = frozenset("xX")                         # U+0078 U+0058: the two multipliers that are letters
MULTIPLIERS = MULT_LETTERS | frozenset("*\u00d7\u00b7")   # U+002A U+00D7 (U+00B7 doubles as one)
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

# ----------------------------------------------------------------- CONTEXT (115) ---
CONTEXT_LETTERS = frozenset(                           # the 48 ASCII letters that are not e E x X (INSIDE)
    "ABCDFGHIJKLMNOPQRSTUVWYZ"                         # U+0041..U+005A without E (U+0045) and X (U+0058)
    "abcdfghijklmnopqrstuvwyz")                        # U+0061..U+007A without e (U+0065) and x (U+0078)
GREEK = frozenset(                                     # the 17 Greek letters seen within one word of a digit
    "\u03bc\u03c9\u03b2\u03b1\u03c4\u03bb\u03b7\u03b8"  # mu omega beta alpha tau lambda eta theta
    "\u03b5\u03b3\u03c3\u03c1\u03c0"                    # epsilon gamma sigma rho pi (chi is outside)
    "\u03ba"                                            # kappa (r7, from the deleted LINE list)
    "\u0394\u03a9\u03a3")                               # Delta Omega Sigma
ACCENTED = frozenset(                                  # 29 accented Latin letters seen in author names (r7, from LINE)
    "\u00e9\u00e8\u00eb\u00e2\u00e1\u00c1\u00e3\u00e6\u00c5\u00e4\u00f6\u00d6"   # e-acute e-grave e-diaeresis ...
    "\u00fc\u00ed\u00ef\u00f3\u00f8\u00d8\u00fa\u0171\u00f1\u0144\u00e7\u00c7"   # u-diaeresis i-acute o-acute ...
    "\u0107\u010d\u011f\u0142\u015b")                  # c-acute c-caron g-breve l-stroke s-acute
CONTEXT_MARKS = frozenset(                             # 12 prose and footnote marks (r7, from LINE)
    "&@\u201c\u201d"                                    # ampersand, commercial at, left and right double quotation mark
    "\u00a9\u00ae\u2122\u221e"                          # copyright, registered, trade mark, infinity
    "\u00a7\u00b6\u2020\u2021")                         # section, pilcrow, dagger, double dagger
OPENERS_BR = frozenset("([")                           # U+0028 U+005B
CLOSERS_BR = frozenset(")]")                           # U+0029 U+005D
CONTEXT_PUNCT = frozenset(";:/=\u00b0")                # U+003B U+003A U+002F U+003D U+00B0 DEGREE SIGN
CONTEXT = frozenset().union(CONTEXT_LETTERS, GREEK, ACCENTED, CONTEXT_MARKS, OPENERS_BR, CLOSERS_BR, CONTEXT_PUNCT)

# ------------------------------------------- the unit production's terminals (r7, owner decision 2) ---
# Grammar data of the CONTEXT allow-list, not a new class: every character of a terminal is a CONTEXT letter (a test
# pins it). ``units.py`` parses ``atom {("/" | "per") atom}``, ``atom = ([prefix] base | prefix prefixed) [exponent]``,
# glued right after a number ('150mg', '24h', '10mL', '1.73m2'). Each terminal is admitted by its count in a COMPLETE
# production glued to a number at a former glued_letters rejection of the 449 parse caches (r7/w2_unit_census3.txt; all
# in the 60-paper design corpus, none elsewhere):
#   prefixes  m 172 (mg 122, mL 23, ml 14, mM 13), n 29 (ng 25, nM 4), k 16 (kg), mu 12 (ug 9, uM 3), c 6 (cm)
#   bases     g 181, h 46, L 44, y 29, M 27, min 24, m 20, w 8, mmHg 8, Hz 4, IU 2
#   prefixed  l 14, every one after a prefix (ml, ng/ml): no bare 'l' is glued to a number anywhere, so the lower-case
#             l is a base only after a prefix and '23l' stays unresolved (it may be '231'; tests/test_evidence_r3_*)
# Refused by the same census: d (17 of 21 bare uses are figure panels '2d', and 'nd' of '2nd' would parse as n + d),
# s (all 8 bare uses are decades: '1970s'), U, hr, wk (1 use each), the prefixes p u f M G d (no use), and words
# (days 5, years 12, weeks 21, hours 5: a word is not a unit symbol). A glued word that is not a full production
# ('1OmL', '150mgtid', '2nd', '5th', '1970s', '23l') stays unresolved.
UNIT_PREFIXES = frozenset({"m", "\u03bc", "n", "k", "c"})              # milli micro nano kilo centi
UNIT_BASES = frozenset({"g", "h", "L", "y", "M", "min", "m", "w", "mmHg", "Hz", "IU"})
UNIT_PREFIXED_BASES = frozenset({"l"})                                  # a base only after a prefix: 'ml'

# ------------------------------------------------------------------ role sets used by the reader ---
# Each role set is defined ONCE here, as members of the four lists (an intersection or a union of them: a role set
# can never hold a character the lists do not; tests/test_evidence_r7_round2_hygiene.py pins every one as a subset of
# INSIDE, CONTEXT and GAP). No other module of the reader types a character set of its own (r7 round 2).
ASCII_LETTERS = CONTEXT_LETTERS | EXP_MARKERS | MULT_LETTERS        # the 52 ASCII letters
# every letter the alphabet lists (a word led by one is a label): the letters whose case the locator, a unit and a
# header key fold, and between which a hyphen at a line break joins a wrapped word
LETTERS = ASCII_LETTERS | GREEK | ACCENTED
SUP_CHARS = SUP_DIGITS | SUP_SIGNS
SIGN_CHARS = SIGNS | SUP_SIGNS                           # every sign a number or an exponent can start with
(MIDDOT,) = SEPARATORS & MULTIPLIERS                     # U+00B7: the one separator that doubles as a multiplier
STAR = "*"                                               # the multiplier that is also a footnote star run
SUP_MINUS = "\u207b"                                    # the superscript minus of SUP_SIGNS (a negative power)
DEGREE = "\u00b0"                                       # the degree sign of CONTEXT_PUNCT ('37°C')
PRODUCT_DOTS = MULTIPLIERS - MULT_LETTERS - {STAR}       # the times sign and the middle dot
DECIMAL_MARKS = SEPARATORS & frozenset(".,")             # the separators that may be a decimal point
GROUP_ONLY = SEPARATORS - DECIMAL_MARKS - {MIDDOT}       # ' and the thin space: they only ever group thousands
EQUALS = CONTEXT_PUNCT & frozenset("=")
RATIO_MARKS = CONTEXT_PUNCT & frozenset("/:")            # a lone word of one between two numbers: a ratio mark
SLASH = CONTEXT_PUNCT & frozenset("/")                   # a ratio's, a unit denominator's and a divisor's slash
N_LETTERS = CONTEXT_LETTERS & frozenset("nN")            # the 'n' of the sample-size production 'n = 12'
# a token may start right after one of these in its word ('n=12', '<5', '(12', '±3')
STARTERS = OPENERS_BR | QUALIFIERS | EQUALS | {PLUS_MINUS}
# a run of these before a token qualifies it when it holds a qualifier ('<=', '~=', '<(', '<-')
OPERATORS = QUALIFIERS | EQUALS | OPENERS_BR | (CONTEXT_PUNCT & frozenset(":;")) | (MULTIPLIERS & {STAR}) | SIGNS
HYPHEN = SIGNS & frozenset("-")                          # the hyphen-minus (a sign, a range dash, a word's hyphen)
RANGE_WORDS = HYPHEN | (QUALIFIERS & frozenset("~"))      # a lone one between two numbers: a range
# may follow a token glued to it (not followed by a digit)
TRAILING = CLOSERS_BR | (CONTEXT_PUNCT & frozenset(";:")) | DECIMAL_MARKS | (SEPARATORS & frozenset("'"))
NUMBER_CLOSERS = TRAILING | PERCENTS | (MULTIPLIERS & {STAR})   # may follow the last digit of a number in its word
JOINERS = (SIGNS & frozenset("-")) | {PLUS_MINUS} | MULTIPLIERS  # a token may start right after one after a token
REF_SEPARATORS = (DECIMAL_MARKS & frozenset(",")) | (SIGNS & frozenset("-"))   # between the references of a citation
NAME_LETTER = "q"                                        # the CONTEXT letter a label digit reads as (decision 3)
LIST_MARKS = (DECIMAL_MARKS & frozenset(",")) | (CONTEXT_PUNCT & frozenset(";"))   # end a list item ('1/2, 3/4')
# r7 round 3: the single terminals the grammar compares against, each one member of a list (no module of the reader
# types a one-character literal: tests/test_evidence_r7_round3_hygiene.py)
(POINT,) = DECIMAL_MARKS - LIST_MARKS                     # the decimal point (a page style, a unit separator)
(COMMA,) = DECIMAL_MARKS & LIST_MARKS                     # the decimal comma (a page style, a list mark)
(ZERO,) = DIGITS & frozenset("0")                         # a leading zero; the zeros of a written-out power of ten
(SQUARE_OPEN,) = OPENERS_BR & frozenset("[")              # the opener of a citation group
(LF,) = GAP & frozenset("\n")                     # a line break: a word wrapped at a hyphen continues past it
APOSTROPHE = SEPARATORS & frozenset("'")                 # the apostrophe that groups thousands ('24'560')
THIN_GROUP = GROUP_ONLY - APOSTROPHE                     # the thin space: groups digits only between two digits
(SUP_TWO,) = SUP_DIGITS & frozenset("\u00b2")             # the superscript two of a square and of chi-square
# the chi-square form's letter (owner decision 8): grammar data of the production in productions.py, NOT a member of
# any list (chi is outside; the form masks it 1:1, every other chi stays outside)
CHI = "\u03c7"                                           # GREEK SMALL LETTER CHI
SUP_TO_ASCII = str.maketrans("\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079\u207a\u207b",
                             "0123456789+-")

# every character the reader can resolve as part of a number (INSIDE plus the sources of the approved equivalents):
# the one list the glyph-identity checks (parse_fonts, parse_marks, parse_ink, parse) compare against
NUMBER_CHARS = INSIDE | frozenset(EQUIV)

# the pdfminer marker for a glyph it could not map (a lost >= +- x deg micro minus): one OUTSIDE character
CID_MARKER = re.compile(r"\(cid:[0-9]+\)")


def cls(ch: str) -> str:
    """'inside' | 'context' | 'gap' | 'outside' for ONE character. A character in EQUIV classifies as its target; the
    "(cid:" marker is a string rule (CID_MARKER) and is not a character."""
    if len(ch) != 1:
        raise ValueError("cls() classifies exactly one character")
    ch = EQUIV.get(ch, ch)
    if ch in INSIDE:
        return "inside"
    if ch in GAP:
        return "gap"
    if ch in CONTEXT:
        return "context"
    return "outside"


def char_class(chars: frozenset[str], *, negate: bool) -> str:
    body = "".join(re.escape(c) for c in sorted(chars))
    return "[" + ("^" if negate else "") + body + "]"


# the derived line rule (r7): ONE allow-list for the word, window and line tiers, built from the four permitted lists
# (never hand-written); the tests prove it agrees with cls() on every code point
READABLE = INSIDE | CONTEXT | GAP | frozenset(EQUIV)
OUTSIDE_RE = re.compile(char_class(READABLE, negate=True))
INSIDE_ONLY_RE = re.compile(char_class(INSIDE, negate=False) + "+")
WORD_RE = re.compile(char_class(GAP, negate=True) + "+")
