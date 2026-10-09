"""Productions the reader applies to a whole folded text before it reads its words (r7 owner decisions 7 and 8).

CITATION (decision 7). An unmistakable bracketed citation is a NON-VALUE token::

    group    = "[" gap* ref {gap* ("," | "-") gap* ref} gap* "]"     ref = [1-9] digit{0,2}; no digit right after "]"
    citation = an unmistakable group (below)                         '[46]', '[1, 2, 3]', '[12-15]', '[ 12 ]'
    bracket_list = a group with a comma that is NOT unmistakable but starts a word: '12 [1,2]', a cell '[1, 5]'

r7 round 2: the gaps of a spaced Vancouver list ('efflux [12, 13]') belong to the group (they were read as three
numbers),
and the one other bracket production is the ``bracket_list``: a comma list of reference-like integers in square brackets
is a list, never a decimal or a thousands number (it replaces scan.py's detector of the one shape ``[n,m]``). Both are
masked and emitted as unresolved expressions with the references as parts.

It is unmistakable when it is glued to a word (after a letter, or after a trailing closer that follows a letter:
``studies[47-49]``, ``al.[1]``, ``(ADC)[4]``), when it continues a citation (``[1][2]``, ``[27] [28]``,
``[21]-[23]``), or when the word before it ends with a letter, trailing closers aside (``efflux [46].``); a label whose
last digits are no token (``CYP2D6 [12]``, ``AUC0-24[3]``) counts as such a word (decision 3). After a number
(``89 [46]``, the range of a median ``9[4-9]``), after a lone multiplier or exponent letter, or with no word before it
(a cell ``[62]``) a bracket keeps its ordinary reading. A citation is masked 1:1 with ``MASK_LETTER`` (a CONTEXT letter)
before the words are read, its opening bracket kept (r7 round 3: a unit glued before it ends at that opener, '12mg[3]'),
so the words around it see one more label: it never taints a number, never ends a range and is never an operand. The
reader then emits it as one unresolved expression (``unresolved_shape:citation``, parts = the reference numbers):
citing a number inside it fails, with a sentence that says it is a citation.

CHI-SQUARE (decision 8). Greek chi (U+03C7) is outside except in the explicit statistical form::

    chi_square = CHI (SUP_TWO | "2") ["(" ["df" [gap] "=" [gap]] digit{1,3} ")"] [gap] "=" [gap]  followed by a digit

(``CHI`` and ``SUP_TWO`` are grammar data of alphabet.py since r7 round 3: Greek small chi, the superscript two.)

Each chi of the form and its square are masked 1:1 with ``MASK_LETTER``: the statistic reads as a label (a variable
name before its '=': r7 round 3, a variable is a label or one letter, never a letter before a power), its degrees of
freedom and the number after ``=`` read as usual. Every other chi, and every look-alike (a capital chi, a Cyrillic ha,
a Latin x), is left as it is: outside, or what the alphabet says it is. Corpus (449 parse caches): chi occurs on 3
lines of 2 papers, none of them in this form (chi-square ``df=1, p<0.01`` in brackets, chi-square ``threshold``), so
they stay unresolved.

All classes come from the alphabet's lists and role sets (``LETTERS``, ``GAP``, ``TRAILING``, ``MULTIPLIERS``,
``EXP_MARKERS``) and the literal terminals of the two forms; nothing here blocks a character. Letters are never told
apart by case (a label glued to a citation, ``AUC[12-24]``, is read as one: it fails closed).
"""
from __future__ import annotations

import re
from bisect import bisect_left
from typing import NamedTuple

from app.evidence.numgrammar.alphabet import (
    CHI,
    DECIMAL_MARKS,
    DF,
    DIGITS,
    EQUALS,
    EXP_MARKERS,
    GAP,
    HYPHEN,
    INSIDE,
    LETTERS,
    MULTIPLIERS,
    REF_SEPARATORS,
    ROUND_CLOSE,
    ROUND_OPEN,
    SQUARE_CLOSE,
    SQUARE_OPEN,
    STARTERS,
    SUP_TWO,
    TRAILING,
    char_class,
)
from app.evidence.numgrammar.body import to_float

MASK_LETTER = "c"                                 # a CONTEXT letter: a masked citation or statistic reads as a label
_GAPS = char_class(GAP, negate=False) + "*"
_EQ = char_class(EQUALS, negate=False)
_CHI_SQUARE = re.compile(CHI + "(?:" + SUP_TWO + "|2)(?=(?:" + re.escape(ROUND_OPEN) + "(?:" + DF + _GAPS + _EQ + _GAPS
                         + ")?[0-9]{1,3}" + re.escape(ROUND_CLOSE) + ")?" + _GAPS + _EQ + _GAPS + "[0-9])")
_REF_SEP = char_class(REF_SEPARATORS, negate=False)
_REF_NUM = "[1-9][0-9]{0,2}"
_GROUP = re.compile(re.escape(SQUARE_OPEN) + _GAPS + _REF_NUM + "(?:" + _GAPS + _REF_SEP + _GAPS + _REF_NUM + ")*"
                    + _GAPS + re.escape(SQUARE_CLOSE) + "(?![0-9])")
_LIST_SEP = (REF_SEPARATORS & DECIMAL_MARKS)                 # the comma that makes a bracket group a list
_REF = re.compile("[0-9]+")
_GAP_RE = re.compile(char_class(GAP, negate=False))
_NON_GAP_RE = re.compile(char_class(GAP, negate=True))
_CLOSE = "".join(sorted(TRAILING))                # closers that may stand between a word and its citation
_LONE = MULTIPLIERS | EXP_MARKERS                 # a one-character multiplier or exponent word: never before a citation


class Cite(NamedTuple):
    start: int
    end: int
    refs: tuple[float, ...]
    why: str = "citation"                         # or "bracket_list" (r7 round 2)


def _previous_word(t: str, s: int) -> tuple[int, int] | None:
    j = s
    while j > 0 and t[j - 1] in GAP:
        j -= 1
    if j == 0:
        return None
    i = j
    while i > 0 and t[i - 1] not in GAP:
        i -= 1
    return i, j


def _continues(t: str, s: int, chain: int) -> bool:
    """The group at t[s] follows the citation ending at *chain* with only gaps and at most one range dash between
    (``[1][2]``, ``[27] [28]``, ``[21]-[23]``, ``[21] -\n[22]``)."""
    k = s
    while k > 0 and t[k - 1] in GAP:
        k -= 1
    if k > 0 and t[k - 1] in HYPHEN:
        k -= 1
        while k > 0 and t[k - 1] in GAP:
            k -= 1
    return k == chain


class _WordStarts:
    """The start of the word that holds a position, from the gap positions of the text (built once, on first use: a
    long word with many glued groups is never scanned back once per group)."""
    __slots__ = ("t", "gaps")

    def __init__(self, t: str) -> None:
        self.t, self.gaps = t, None

    def __call__(self, p: int) -> int:
        if self.gaps is None:
            self.gaps = [m.start() for m in _GAP_RE.finditer(self.t)]
        k = bisect_left(self.gaps, p) - 1
        return self.gaps[k] + 1 if k >= 0 else 0


def _label_end(t: str, ws: int, end: int) -> bool:
    """r7, owner decision 3: the word t[ws:end] (its trailing closers left out) is a label (led by a letter after at
    most one opener) whose last digit run is no token: the run of INSIDE characters that ends it holds no starter and
    follows none, since a token of a label word starts only after a starter (``AUC=12`` ends in the number 12;
    ``CYP2D6``, ``AUC0-24``, ``(t1/2`` end in label digits). It walks back over that run only, so a long word with many
    glued groups is read in linear time."""
    if end <= ws or t[end - 1] not in DIGITS:
        return False
    lead = ws + 1 if t[ws] in STARTERS else ws
    if lead >= end or t[lead] not in LETTERS:
        return False
    k = end
    while k > lead and t[k - 1] in INSIDE:
        if t[k - 1] in STARTERS:
            return False
        k -= 1
    return not (k > lead and t[k - 1] in STARTERS)


def _unmistakable(t: str, s: int, chain: int, word_start: _WordStarts) -> bool:
    """The group at t[s] is a citation: glued to a word, continuing a citation, or after a word ending in a letter. A
    label whose last digits are no number counts as a word ending in a letter (r7, owner decision 3: ``CYP2D6 [12]``,
    ``AUC0-24[3]`` read like ``efflux [46]``; ``(n=7)[48]`` still ends in the number 7)."""
    if chain >= 0 and _continues(t, s, chain):
        return True
    if s > 0 and t[s - 1] not in GAP:
        k = s - 1
        if t[k] in _CLOSE and k > 0 and t[k - 1] in LETTERS:
            return True
        if t[k] in LETTERS:
            return True
        e = s
        while e > 0 and t[e - 1] in _CLOSE:
            e -= 1
        return _label_end(t, word_start(k), e)
    prev = _previous_word(t, s)
    if prev is None:
        return False
    word = t[prev[0]:prev[1]]
    core = word.rstrip(_CLOSE)
    return word not in _LONE and bool(core) and (core[-1] in LETTERS or _label_end(t, prev[0], prev[0] + len(core)))


def citations(t: str) -> list[Cite]:
    """Every unmistakable citation and every bracket list of the folded text *t*, in order."""
    out: list[Cite] = []
    chain = -1
    word_start = _WordStarts(t)
    for m in _GROUP.finditer(t):
        refs = tuple(to_float(r) for r in _REF.findall(m.group()))
        if _unmistakable(t, m.start(), chain, word_start):
            out.append(Cite(m.start(), m.end(), refs))
            chain = m.end()
        elif (m.start() == 0 or t[m.start() - 1] in GAP) and not _LIST_SEP.isdisjoint(m.group()):
            out.append(Cite(m.start(), m.end(), refs, "bracket_list"))
    return out


def bracket_groups(t: str) -> list[tuple[int, int]]:
    """[start, end) of every bracketed reference group of the folded text *t* (the ``group`` of the citation
    production, whatever stands around it): the flag checks read the words around a number with them taken out (r7
    round 4: '[3]' between 'less than' and the number never hides the bound)."""
    return [m.span() for m in _GROUP.finditer(t)]


def mask_citations(t: str, cites: list[Cite]) -> str:
    """*t* with every character of every citation but its opening bracket and its gaps replaced 1:1 by ``MASK_LETTER``:
    offsets never move and a line break inside a spaced list stays a line break. r7 round 3: the opener stays, so a
    citation glued to a unit ('12mg[3]') ends the unit production at an opener like any bracket (it read 'mgccc')."""
    if not cites:
        return t
    parts, at = [], 0
    for c in cites:
        parts += [t[at:c.start], t[c.start], _NON_GAP_RE.sub(MASK_LETTER, t[c.start + 1:c.end])]
        at = c.end
    return "".join(parts) + t[at:]



def mask_chi_square(t: str) -> str:
    """*t* with the chi and the square of every chi-square form replaced 1:1 by ``MASK_LETTER`` (offsets never move: the
    statistic reads as a two-letter label, a variable name before its '=', r7 round 3); every other chi stays, and is
    outside."""
    return _CHI_SQUARE.sub(MASK_LETTER * 2, t) if CHI in t else t


__all__ = ["CHI", "MASK_LETTER", "Cite", "bracket_groups", "citations", "mask_chi_square", "mask_citations"]
