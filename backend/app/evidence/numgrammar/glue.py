"""The inert productions around a number (r7 writer 3, the neighbour-word allow-list): what may stand next to a number
inside its own word (the GLUE before, between and after its tokens) and in a word of its window that holds no number.

Every class is built from the alphabet's lists and role sets; nothing here names a bad character. A character may stand
next to a number only inside one of these productions::

    prose   = a word without a digit whose MODIFIER characters are all joined to a letter (below)
    label   = a word led by a letter (after openers) whose first letter run is not a lone operator letter
    glue    = [PUNCT*] [prose | label] [PUNCT*]                         the text between the tokens of one word
    prefix  = glue [starter run]                                        before the first token of a word

The MODIFIERS are the characters that change a number they stand beside (the owner's list): a sign, a multiplier
(``* x X × ·``), an exponent letter (``e E``), a caret, a superscript, a qualifier, the plus-minus and the percent sign.
Inside a prose word one is inert only where it is joined to a letter the way prose and unit symbols join them:

    sign, plus-minus, percent, star run   a letter on at least one side    once-daily, -fold, %CV, mg%, Cmax*, *P
    times sign, middle dot                a letter on both sides            ng·h/mL, mg×h
    superscript run                       right after a letter              m², h⁻¹, L⁻¹
    caret                                 a letter on its left, a letter or sign on its right (a unit exponent in a
                                          label, its digits read as letters: 'cells/10^6', 'h^-1'; never alone)
    qualifier                             never (a qualifier is its own production, see neighbours.py)
    e E x X                               inside a letter run of two or more letters (exposure, Cmax, e.g.); a LONE one
                                          (no letter on either side) is an OPERATOR LETTER when it touches a modifier,
                                          a digit, '=' or an opener, or is the whole word ('1.2 e 3', '(x', 'E-3',
                                          'x10^3', 'e3,', 'E2'), and every word holding one is no prose and no label
"""
from __future__ import annotations

import re

from app.evidence.numgrammar.alphabet import (
    CARET,
    CLOSERS_BR,
    DIGITS,
    EXP_MARKERS,
    LETTERS,
    MULTIPLIERS,
    OPENERS_BR,
    PERCENTS,
    PLUS_MINUS,
    QUALIFIERS,
    SIGNS,
    STARTERS,
    SUP_CHARS,
    TRAILING,
    char_class,
)

OP_LETTERS = EXP_MARKERS | (MULTIPLIERS & LETTERS)                       # e E x X: a letter or an operator
STAR = "*"
PRODUCT_DOTS = MULTIPLIERS - LETTERS - {STAR}                            # the times sign and the middle dot
MODIFIERS = SIGNS | (MULTIPLIERS - LETTERS) | {CARET} | SUP_CHARS | QUALIFIERS | {PLUS_MINUS} | PERCENTS
# a word made only of these is punctuation: skipped by the window (it then belongs to it), never a value's neighbour
PUNCT = OPENERS_BR | CLOSERS_BR | TRAILING | frozenset("/=")
SEPARATOR_PUNCT = frozenset(".,'")                     # a word of only these between two numbers splits one number
_HALF_JOINED = SIGNS | {PLUS_MINUS} | PERCENTS          # inert with a letter on at least one side
# what makes a lone e E x X an operator letter: on its left ('-e', '1x', '=x'), on its right ('e-', 'x1', 'x(', 'E=')
_TOUCH_LEFT = MODIFIERS | DIGITS | {"="}
_TOUCH_RIGHT = _TOUCH_LEFT | OPENERS_BR
STARTER_RUN = QUALIFIERS | {"=", PLUS_MINUS} | OPENERS_BR   # may stand right before a token in its own word
_AS_LETTER = str.maketrans(dict.fromkeys(sorted(DIGITS), "q"))   # a label digit read as a letter (a CONTEXT letter)
_PLAIN_PROSE = re.compile(char_class(MODIFIERS | DIGITS | OP_LETTERS, negate=True) + "*")


def _letter(w: str, i: int) -> bool:
    return 0 <= i < len(w) and w[i] in LETTERS


def core(w: str) -> tuple[int, int]:
    """[lo, hi) of *w* without its leading openers and trailing closers or punctuation."""
    lo, hi = 0, len(w)
    while lo < hi and w[lo] in OPENERS_BR:
        lo += 1
    while hi > lo and (w[hi - 1] in TRAILING or w[hi - 1] in CLOSERS_BR):
        hi -= 1
    return lo, hi


def operator_letter(w: str, i: int) -> bool:
    """w[i] (e E x X) stands alone where an exponent marker or a multiplier stands."""
    if w[i] not in OP_LETTERS or _letter(w, i - 1) or _letter(w, i + 1):
        return False
    lo, hi = core(w)
    if (lo, hi) == (i, i + 1):
        return True
    return (i > 0 and w[i - 1] in _TOUCH_LEFT) or (i + 1 < len(w) and w[i + 1] in _TOUCH_RIGHT)


def _run(w: str, i: int, chars: frozenset[str] | str) -> tuple[int, int]:
    a, b = i, i + 1
    while a > 0 and w[a - 1] in chars:
        a -= 1
    while b < len(w) and w[b] in chars:
        b += 1
    return a, b


def _joined(w: str, i: int) -> bool:
    """The modifier w[i] of a word without a digit is joined to a letter the way prose and unit symbols join one."""
    c = w[i]
    if c in _HALF_JOINED:
        return _letter(w, i - 1) or _letter(w, i + 1)
    if c == STAR:
        a, b = _run(w, i, STAR)
        return _letter(w, a - 1) or _letter(w, b)
    if c in PRODUCT_DOTS:
        return _letter(w, i - 1) and _letter(w, i + 1)
    if c in SUP_CHARS:
        a, _ = _run(w, i, SUP_CHARS)
        return _letter(w, a - 1)
    if c == CARET:
        return _letter(w, i - 1) and (_letter(w, i + 1) or (i + 1 < len(w) and w[i + 1] in SIGNS))
    return False                                          # a qualifier


def prose(w: str) -> bool:
    """*w* holds no digit and every modifier and operator letter in it is inert (see the module docstring)."""
    if _PLAIN_PROSE.fullmatch(w) is not None:
        return True                                       # no digit, no modifier, no e E x X: nothing to judge
    for i, c in enumerate(w):
        if c in DIGITS:
            return False
        if c in OP_LETTERS:
            if operator_letter(w, i):
                return False
        elif c in MODIFIERS and not _joined(w, i):
            return False
    return True


def label(w: str) -> bool:
    """*w* is a LABEL: led by a letter after a run of openers or starters, that letter is not a lone operator letter
    ('E-3', 'x10^3', 'e3,', '(x10^3', 'E2' are none), and with its digits read as the letters of a name (decision 3:
    label digits never act by themselves) it is prose ('AUC0-24', 'CYP2D6*4', 't1/2', 'h-1', 'L\u00b7h\u207b\u00b9',
    '(AUC0-inf'; never 'AUC0^-24')."""
    i = 0
    while i < len(w) and (w[i] in OPENERS_BR or w[i] in STARTERS):
        i += 1
    if i >= len(w) or w[i] not in LETTERS or operator_letter(w, i):
        return False
    return prose(w[i:].translate(_AS_LETTER))


def glue(run: str) -> bool:
    """The characters between the tokens of one word (or after its last one) are inert: punctuation around one prose
    stretch, or around label digits of a label word ('', ')', '),(' , '-fold', '/day', ';n', ',44.83', 'and77.43');
    never a sign, a multiplier, a lone operator letter or a superscript that joins no letter ('-', 'x', '(x)', '-e',
    '*', ',e3')."""
    a, b = 0, len(run)
    while a < b and run[a] in PUNCT:
        a += 1
    while b > a and run[b - 1] in PUNCT:
        b -= 1
    mid = run[a:b]
    return prose(mid.translate(_AS_LETTER))               # label digits are letters of a name (decision 3)


def variable(w: str) -> bool:
    """*w* (an equals sign follows it) is a variable name: led by a letter after openers, any letters ('E', 'x', 'E0',
    '(E', 'AUC0-24'): the assignment production 'E = 0.9', 'x=12', 'E0 = 0'."""
    i = 0
    while i < len(w) and w[i] in OPENERS_BR:
        i += 1
    return i < len(w) and w[i] in LETTERS


def prefix(run: str) -> bool:
    """The characters of a word before its first token: glue, then a run of starters ('(', '<=', 'n=', '(p<',
    'AUC0-24=', '\u00b1('), or a variable name then '=' ('x=', 'E0='). A modifier right before the starters ('-(', '-=',
    '+[', 'Cmax-(', '%(', '*(') is none: a sign before an opener belongs to no production."""
    k = len(run)
    while k > 0 and run[k - 1] in STARTER_RUN:
        k -= 1
    head = run[:k]
    if head and (head[-1] in SIGNS or (head[-1] in MODIFIERS and not _joined(head, len(head) - 1))):
        return False                                      # a sign before an opener is never glue; 'c\u00b2(' is
    return glue(head) or (run[k:k + 1] == "=" and variable(head))


__all__ = ["MODIFIERS", "OP_LETTERS", "PUNCT", "SEPARATOR_PUNCT", "core", "glue", "label", "operator_letter", "prefix",
           "prose", "variable"]
