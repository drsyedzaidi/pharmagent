"""Productions the reader applies to a whole folded text before it reads its words (r7 owner decisions 7 and 8).

CITATION (decision 7). An unmistakable bracketed citation is a NON-VALUE token::

    citation = "[" ref {("," | "-") ref} "]"      ref = [1-9] digit{0,2}; no digit right after "]"

It is unmistakable when it is glued to a word (after a letter, or after a trailing closer that follows a letter:
``studies[47-49]``, ``al.[1]``, ``(ADC)[4]``), when it continues a citation (``[1][2]``, ``[27] [28]``,
``[21]-[23]``), or when the word before it ends with a letter, trailing closers aside (``efflux [46].``); a label whose
last digits are no token (``CYP2D6 [12]``, ``AUC0-24[3]``) counts as such a word (decision 3). After a number
(``89 [46]``, the range of a median ``9[4-9]``), after a lone multiplier or exponent letter, or with no word before it
(a cell ``[62]``) a bracket keeps its ordinary reading. A citation is masked 1:1 with ``MASK_LETTER`` (a CONTEXT letter)
before the words are read, so the words around it see one more label: it never taints a number, never ends a range and
is never an operand. The reader then emits it as one unresolved expression (``unresolved_shape:citation``, parts = the
reference numbers): citing a number inside it fails, with a sentence that says it is a citation.

CHI-SQUARE (decision 8). Greek chi (U+03C7) is outside except in the explicit statistical form::

    chi_square = chi ("\u00b2" | "2") ["(" ["df" [gap] "=" [gap]] digit{1,3} ")"] [gap] "=" [gap]  followed by a digit

Each chi of the form is masked 1:1 with ``MASK_LETTER``: the statistic reads as a label, its degrees of freedom and the
number after ``=`` read as usual. Every other chi, and every look-alike (a capital chi, a Cyrillic ha, a Latin x), is
left as it is: outside, or what the alphabet says it is. Corpus (449 parse caches): chi occurs on 3 lines of 2 papers,
none of them in this form (``(\u03c7\u00b2 df=1, p<0.01)``, ``\u03c7\u00b2 threshold``), so they stay unresolved.

All classes come from the alphabet's lists and role sets (``LETTERS``, ``GAP``, ``TRAILING``, ``MULTIPLIERS``,
``EXP_MARKERS``) and the literal terminals of the two forms; nothing here blocks a character. Letters are never told
apart by case (a label glued to a citation, ``AUC[12-24]``, is read as one: it fails closed).
"""
from __future__ import annotations

import re
from bisect import bisect_left
from typing import NamedTuple

from app.evidence.numgrammar.alphabet import (
    DIGITS,
    EXP_MARKERS,
    GAP,
    INSIDE,
    LETTERS,
    MULTIPLIERS,
    STARTERS,
    TRAILING,
    char_class,
)
from app.evidence.numgrammar.body import to_float

MASK_LETTER = "c"                                 # a CONTEXT letter: a masked citation or statistic reads as a label
CHI = "\u03c7"                                     # GREEK SMALL LETTER CHI: a terminal of the chi-square form only
_GAPS = char_class(GAP, negate=False) + "*"
_CHI_SQUARE = re.compile(CHI + "(?=(?:\u00b2|2)(?:\\((?:df" + _GAPS + "=" + _GAPS + ")?[0-9]{1,3}\\))?" + _GAPS + "="
                         + _GAPS + "[0-9])")
_GROUP = re.compile("\\[[1-9][0-9]{0,2}(?:[,-][1-9][0-9]{0,2})*\\](?![0-9])")
_REF = re.compile("[0-9]+")
_GAP_RE = re.compile(char_class(GAP, negate=False))
_CLOSE = "".join(sorted(TRAILING))                # closers that may stand between a word and its citation
_LONE = MULTIPLIERS | EXP_MARKERS                 # a one-character multiplier or exponent word: never before a citation


class Cite(NamedTuple):
    start: int
    end: int
    refs: tuple[float, ...]


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
    if k > 0 and t[k - 1] == "-":
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
    """Every unmistakable citation of the folded text *t*, in order."""
    out: list[Cite] = []
    chain = -1
    word_start = _WordStarts(t)
    for m in _GROUP.finditer(t):
        if _unmistakable(t, m.start(), chain, word_start):
            out.append(Cite(m.start(), m.end(), tuple(to_float(r) for r in _REF.findall(m.group()))))
            chain = m.end()
    return out


def mask_citations(t: str, cites: list[Cite]) -> str:
    """*t* with every citation replaced 1:1 by ``MASK_LETTER``: offsets never move."""
    if not cites:
        return t
    parts, at = [], 0
    for c in cites:
        parts += [t[at:c.start], MASK_LETTER * (c.end - c.start)]
        at = c.end
    return "".join(parts) + t[at:]


def mask_chi_square(t: str) -> str:
    """*t* with the chi of every chi-square form replaced 1:1 by ``MASK_LETTER`` (offsets never move); every other chi
    stays, and is outside."""
    return _CHI_SQUARE.sub(MASK_LETTER, t) if CHI in t else t


__all__ = ["CHI", "MASK_LETTER", "Cite", "citations", "mask_chi_square", "mask_citations"]
