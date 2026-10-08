"""Word-level shapes the multi-word forms of the tokenizer recognise (explicit classes only, no ``\\d \\w \\s \\b``)."""
from __future__ import annotations

import re

from app.evidence.numgrammar.alphabet import (
    DIGITS,
    LETTERS,
    MULTIPLIERS,
    NUMBER_CLOSERS,
    OPENERS_BR,
    OPERATORS,
    QUALIFIERS,
    SIGNS,
    STARTERS,
    SUP_CHARS,
    SUP_DIGITS,
    SUP_SIGNS,
    TRAILING,
    char_class,
)

MULT_CHARS = "".join(sorted(MULTIPLIERS))               # for str.strip
_MULT = char_class(MULTIPLIERS, negate=False)
_POWER = ("(?:\\^[-+]?[0-9]{1,4}|" + char_class(SUP_SIGNS, negate=False) + "?"
          + char_class(SUP_DIGITS, negate=False) + "{1,4})")
BARE_POWER_RE = re.compile("10" + _POWER)               # '10^3', '10⁻³': a ten with a power, no mantissa
MULT_POWER_RE = re.compile(_MULT + "10" + _POWER)       # '×10^3', 'x10³': a multiplier glued to a power of ten
MULT_TEN_RE = re.compile(_MULT + "10(?!0)[0-9]*")       # 'x103', '×10': a multiplier glued to a ten, no power
E_WORD_RE = re.compile("[eE][-+]?[0-9]{1,4}")           # a spaced exponent: 'E-03', 'e3'
FRACTION_RE = re.compile("[0-9]{1,3}/[0-9]{1,3}")       # '3/4'
CARET_OR_SUP_RE = re.compile("[\\^" + "".join(sorted(SUP_CHARS)) + "]")
DIGIT_RE = re.compile("[0-9]")

# special words
E_WORD, FRACTION, MULT_POWER, MULT_TEN = "eword", "fraction", "multpow", "multten"


def special(w: str) -> str:
    """The special shape of the whole word *w*, '' for an ordinary word."""
    c = w[0]
    if c == "e" or c == "E":
        return E_WORD if E_WORD_RE.fullmatch(w) else ""
    if c in DIGITS:
        return FRACTION if "/" in w and FRACTION_RE.fullmatch(w) else ""
    if c in MULTIPLIERS:
        if MULT_POWER_RE.fullmatch(w):
            return MULT_POWER
        return MULT_TEN if MULT_TEN_RE.fullmatch(w) else ""
    return ""


def is_mult_word(w: str) -> bool:
    """A word that is ONE multiplier character (``x``, ``X``, ``*``, ``×``, ``·``): ``xx`` and ``**`` are
    not multipliers (``1.2 ** 10^3`` must not read as 1.2 x 10^3)."""
    return len(w) == 1 and w in MULT_CHARS


def ten_shape(w: str) -> bool:
    """``10`` not followed by ``0`` at the start of the word (after one sign): a ten whose exponent may have been
    raised (``10``, ``103``, ``10(3)``, ``10mg``); ``100`` and ``1000`` are ordinary numbers."""
    i = 1 if w[0] in SIGNS else 0
    return w[i:i + 2] == "10" and w[i + 2:i + 3] != "0"


def letter_led(w: str) -> bool:
    """The first character after at most one opener is a letter: a LABEL word."""
    i = 1 if w[0] in STARTERS else 0
    return i < len(w) and w[i] in LETTERS


def exp_like(w: str) -> bool:
    """A non-letter-led word that is ``^``, starts with ``^``, is made of superscripts, or holds ``^`` or a
    superscript without being an accepted power or multiplier power: an exponent detached from its number."""
    if letter_led(w) or CARET_OR_SUP_RE.search(w) is None:
        return False
    return BARE_POWER_RE.fullmatch(w) is None and MULT_POWER_RE.fullmatch(w) is None


RATIO_CHARS = frozenset("/:")                           # a lone word of one of these between two numbers is a ratio
RATIO_TRAIL = "".join(sorted(TRAILING))                  # closers a spaced ratio's denominator word may end with


def operator_tail(w: str) -> str:
    """The trailing run of OPERATORS characters of *w* ('' when it ends in another character)."""
    k = len(w)
    while k > 0 and w[k - 1] in OPERATORS:
        k -= 1
    return w[k:]


def qualifies(words: list[str]) -> bool:
    """True when the words before a number (nearest LAST, at most three read) hold a qualifier right before it: the
    nearest word ends in a run of operator characters that holds one (``<``, ``<=``, ``~=``, ``\u2264(``, ``<-``), or is
    made only of operator characters without one (``=``, ``(``) and the word before it does, so ``< = 5`` and ``< ( 5``
    qualify like ``<5`` and nothing a gap, an opener or an equals sign adds can hide the qualifier."""
    for w in reversed(words[-3:]):
        run = operator_tail(w)
        if any(c in QUALIFIERS for c in run):
            return True
        if not run or len(run) < len(w):
            return False
    return False


def ends_with_number(w: str) -> bool:
    """The word, without the closers and punctuation that may follow a number, ends in an ASCII digit."""
    k = len(w)
    while k > 0 and w[k - 1] in NUMBER_CLOSERS:
        k -= 1
    return k > 0 and w[k - 1] in DIGITS


def digit_led(w: str) -> bool:
    """The word starts with an ASCII digit, or with one opener or sign right before one."""
    return bool(w) and (w[0] in DIGITS or (len(w) > 1 and (w[0] in OPENERS_BR or w[0] in SIGNS) and w[1] in DIGITS))


def signed_digits(w: str) -> bool:
    """The word starts with a sign and a digit (``-15``): a number that may be the right end of ``12 -15``."""
    return len(w) > 1 and w[0] in SIGNS and w[1] in DIGITS
