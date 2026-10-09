"""The inert productions around a number (r7 writer 3, the neighbour-word allow-list): what may stand next to a number
inside its own word (the GLUE before, between and after its tokens) and in a word of its window that holds no number.

Every class is built from the alphabet's lists and role sets; nothing here names a bad character. A character may stand
next to a number only inside one of these productions::

    prose   = a word without a digit whose MODIFIER characters are all joined to a letter (below)
    label   = a word led by a letter (after openers) that holds no SCALE production (below) and that is prose with its
              digits read as letters (decision 3)
    scale   = on the PRINTED text (r7 round 4, before any label digit reads as a letter): a multiplier (x X * × ·) right
              before a digit or a signed digit, unless a label digit stands on its left ('CYP2D6*4'; the allele star
              also after a slash that follows one, 'CYP2C19*1/*17'; a times sign or middle dot wherever it stands:
              'AUC0-24×1000'), or a LONE e E x X (no letter beside it) that is an operator letter, unless it stands
              between label digits ('CYP2E1'): 'AUC×1000', 'AUCx1000', 'AUC*1e3', 'AUC:E3', 'CLx10', 'AUC1000x' and
              '(x' are scales; it replaces the round-3 carve-outs for a bracket and for the times sign only, and fails
              closed for names of the same shape ('Cmax1')
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

r7 round 2: every check is linear in the word (the core of a word is found once, a star or superscript run is judged
once), and ``unit_powers`` is the power rule of a neighbour word: a caret or superscript run is inert beside a number
only after a unit symbol (``m²``, ``h⁻¹``, ``h^-1``) or as the power of ten of a unit denominator (``cells/10^6``).
"""
from __future__ import annotations

import re

from app.evidence.numgrammar.alphabet import (
    CARET,
    CLOSERS_BR,
    DECIMAL_MARKS,
    DIGITS,
    EQUALS,
    EXP_MARKERS,
    LETTERS,
    MULTIPLIERS,
    NAME_LETTER,
    OPENERS_BR,
    PERCENTS,
    PLUS_MINUS,
    PRODUCT_DOTS,
    QUALIFIERS,
    SEPARATORS,
    SIGNS,
    SLASH,
    STAR,
    STARTERS,
    SUP_CHARS,
    TRAILING,
    char_class,
)
from app.evidence.numgrammar.units import ATOM_RE

OP_LETTERS = EXP_MARKERS | (MULTIPLIERS & LETTERS)                       # e E x X: a letter or an operator
MODIFIERS = SIGNS | (MULTIPLIERS - LETTERS) | {CARET} | SUP_CHARS | QUALIFIERS | {PLUS_MINUS} | PERCENTS
# a word made only of these is punctuation: skipped by the window (it then belongs to it), never a value's neighbour
PUNCT = OPENERS_BR | CLOSERS_BR | TRAILING | SLASH | EQUALS
SEPARATOR_PUNCT = SEPARATORS & TRAILING                  # a word of only these between two numbers splits one number
_HALF_JOINED = SIGNS | {PLUS_MINUS} | PERCENTS          # inert with a letter on at least one side
# what makes a lone e E x X an operator letter: on its left ('-e', '1x', '=x'), on its right ('e-', 'x1', 'x(', 'E=')
_TOUCH_LEFT = MODIFIERS | DIGITS | EQUALS
_TOUCH_RIGHT = _TOUCH_LEFT | OPENERS_BR
STARTER_RUN = QUALIFIERS | EQUALS | {PLUS_MINUS} | OPENERS_BR   # may stand right before a token in its own word
_AS_LETTER = str.maketrans(dict.fromkeys(sorted(DIGITS), NAME_LETTER))   # a label digit read as a letter
_PLAIN_PROSE = re.compile(char_class(MODIFIERS | DIGITS | OP_LETTERS, negate=True) + "*")
_STRETCH = LETTERS | DIGITS | DECIMAL_MARKS              # the characters of a unit stretch before its power
SCALE_CHARS = MULTIPLIERS | EXP_MARKERS                   # a multiplier or an exponent marker: the SCALE production


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


def operator_letter(w: str, i: int, bounds: tuple[int, int] | None = None) -> bool:
    """w[i] (e E x X) stands alone where an exponent marker or a multiplier stands. *bounds*: ``core(w)`` when the
    caller already has it (one word is walked once, never once per letter)."""
    if w[i] not in OP_LETTERS or _letter(w, i - 1) or _letter(w, i + 1):
        return False
    if (bounds or core(w)) == (i, i + 1):
        return True
    return (i > 0 and w[i - 1] in _TOUCH_LEFT) or (i + 1 < len(w) and w[i + 1] in _TOUCH_RIGHT)


def _run_end(w: str, i: int, chars: frozenset[str] | str) -> int:
    b = i + 1
    while b < len(w) and w[b] in chars:
        b += 1
    return b


def _joined(w: str, i: int) -> bool:
    """The modifier w[i] (not a star or a superscript: their runs are judged once by ``prose``) of a word without a
    digit is joined to a letter the way prose and unit symbols join one."""
    c = w[i]
    if c in _HALF_JOINED:
        return _letter(w, i - 1) or _letter(w, i + 1)
    if c in PRODUCT_DOTS:
        return _letter(w, i - 1) and _letter(w, i + 1)
    if c == CARET:
        return _letter(w, i - 1) and (_letter(w, i + 1) or (i + 1 < len(w) and w[i + 1] in SIGNS))
    if c == STAR:                                         # a run, from its first star
        return _letter(w, i - 1) or _letter(w, _run_end(w, i, STAR))
    if c in SUP_CHARS:                                    # a run, from its first character
        return _letter(w, i - 1)
    return False                                          # a qualifier


def prose(w: str) -> bool:
    """*w* holds no digit and every modifier and operator letter in it is inert (see the module docstring). Linear: a
    star or superscript run is judged once, at its first character."""
    if _PLAIN_PROSE.fullmatch(w) is not None:
        return True                                       # no digit, no modifier, no e E x X: nothing to judge
    bounds = core(w)
    i, n = 0, len(w)
    while i < n:
        c = w[i]
        if c in DIGITS:
            return False
        if c in OP_LETTERS:
            if operator_letter(w, i, bounds):
                return False
        elif c == STAR or c in SUP_CHARS:
            if not _joined(w, i):
                return False
            i = _run_end(w, i, STAR if c == STAR else SUP_CHARS)
            continue
        elif c in MODIFIERS and not _joined(w, i):
            return False
        i += 1
    return True


def scaled_at(w: str, j: int, bounds: tuple[int, int]) -> bool:
    """w[j] is the SCALE production on the printed text (see the module docstring): a multiplier or an exponent marker
    right before a digit or a signed digit with no digit on its left, or a lone operator letter. *bounds*:
    ``core(w)``, computed once per word."""
    c = w[j]
    if c not in SCALE_CHARS:
        return False
    k = j + 1 if j + 1 < len(w) and w[j + 1] in SIGNS else j
    number_after = k + 1 < len(w) and w[k + 1] in DIGITS
    if number_after and c in MULTIPLIERS:                 # MULT [sign] digit: 'AUCx1000', 'AUC*1000', 'AUC×1000'
        return c in PRODUCT_DOTS or not _in_name(w, j)
    if c in OP_LETTERS:                                   # a lone e E x X: 'AUC:E3', 'AUC-E3', '(E3)', 'AUC1000x'
        return operator_letter(w, j, bounds) and not (number_after and _in_name(w, j))
    return False


def _in_name(w: str, j: int) -> bool:
    """The letter or allele star w[j] (e E x X *) before a digit continues a name: a digit of the label stands on its
    left ('CYP2E1', 'CYP2D6*4'), or for the allele star a slash after such a digit ('CYP2C19*1/*17': a diplotype)."""
    left = w[j - 1] if j > 0 else ""
    if left in DIGITS:
        return True
    return w[j] == STAR and left in SLASH and j > 1 and w[j - 2] in DIGITS


def scaled(w: str, start: int = 0) -> bool:
    """Some character of *w* from *start* on is the SCALE production (linear: the core is found once)."""
    bounds = core(w)
    return any(scaled_at(w, j, bounds) for j in range(start, len(w)))


def label(w: str) -> bool:
    """*w* is a LABEL: led by a letter after a run of openers or starters, no SCALE production on its printed text
    ('E-3', 'x10^3', 'e3,', '(x10^3', 'E2', 'Count(E3)', 'AUC(x1000)', 'AUC×1000', 'AUCx1000', 'AUC*1000', 'AUC1000x'
    are none: r7 round 4), and with its digits read as the letters of a name (decision 3: label digits never act by
    themselves) it is prose ('AUC0-24', 'CYP2D6*4', 'CYP2E1', 't1/2', 'h-1', 'L·h⁻¹', '(AUC0-inf'; never 'AUC0^-24')."""
    i = 0
    while i < len(w) and (w[i] in OPENERS_BR or w[i] in STARTERS):
        i += 1
    if i >= len(w) or w[i] not in LETTERS or scaled(w, i):
        return False
    return prose(w[i:].translate(_AS_LETTER))


def unit_powers(w: str) -> bool:
    """Every caret and superscript run of *w* is the exponent of a unit symbol: the stretch right before it (letters,
    digits and decimal marks: a label's digits are letters of its name, decision 3) is one unit atom ('m²', 'cm²',
    'h⁻¹', 'mL⁻¹', 'h^-1'), or that stretch follows a slash: a unit denominator ('cells/10^6', 'mL/min/1.73m²'). A
    neighbour word with any other power (a look-alike of a detached power: 'l0³', 'I0³', 'lO³', 'Cmax²') is no inert
    neighbour."""
    i, n = 0, len(w)
    while i < n:
        c = w[i]
        if c != CARET and c not in SUP_CHARS:
            i += 1
            continue
        a = i
        while a > 0 and (w[a - 1] in _STRETCH):
            a -= 1
        if not (a > 0 and w[a - 1] in SLASH) and (a == i or ATOM_RE.fullmatch(w, a, i) is None):
            return False
        i = _run_end(w, i, SUP_CHARS) if c in SUP_CHARS else i + 1
    return True


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
    if scaled(mid):
        return False                                      # r7 round 4: judged on the printed text ('(x10^3)', '(E3)')
    return prose(mid.translate(_AS_LETTER))               # label digits are letters of a name (decision 3)


def variable(w: str) -> bool:
    """*w* (an equals sign follows it) is a variable name: ONE letter after openers ('E', 'x', '(E') or a LABEL whose
    powers follow unit symbols ('Emax', 'AUC0-24', 'k12'): the assignment production 'E = 0.9', 'x=12', 'Emax = 8'. r7
    round 3: an operator letter before digits, a sign or a power is a scale, never a name ('x10^3', '(E3)', 'e3',
    'x1000', 'E0')."""
    i = 0
    while i < len(w) and w[i] in OPENERS_BR:
        i += 1
    if i >= len(w) or w[i] not in LETTERS:
        return False
    return i + 1 == len(w) or (label(w) and unit_powers(w))


def prefix(run: str) -> bool:
    """The characters of a word before its first token: glue, then a run of starters ('(', '<=', 'n=', '(p<',
    'AUC0-24=', '±('), or a variable name then '=' ('x=', 'E0='). A modifier right before the starters ('-(', '-=',
    '+[', 'Cmax-(', '%(', '*(') is none: a sign before an opener belongs to no production."""
    k = len(run)
    while k > 0 and run[k - 1] in STARTER_RUN:
        k -= 1
    head = run[:k]
    if head and (head[-1] in SIGNS or (head[-1] in MODIFIERS and not _modifier_joined_at_end(head))):
        return False                                      # a sign before an opener is never glue; 'c²(' is
    return glue(head) or (run[k:k + 1] in EQUALS and variable(head))


def _modifier_joined_at_end(head: str) -> bool:
    """The modifier that ends *head* is joined to a letter (a star or superscript run judged from its first
    character)."""
    i = len(head) - 1
    run = STAR if head[i] == STAR else (SUP_CHARS if head[i] in SUP_CHARS else "")
    while run and i > 0 and head[i - 1] in run:
        i -= 1
    return _joined(head, i)


__all__ = ["MODIFIERS", "OP_LETTERS", "PUNCT", "SEPARATOR_PUNCT", "core", "glue", "label", "operator_letter", "prefix",
           "prose", "scaled", "unit_powers", "variable"]
