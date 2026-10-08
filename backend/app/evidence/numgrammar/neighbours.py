"""The neighbour-word allow-list (r7 writer 3; a mixin of ``Reader``): the words around a resolved number.

A resolved number (the tokens of one word, or one multi-word token such as ``1.2 x 10^3``) keeps its value only when the
word on each side of it is an ACCEPTED PRODUCTION. Punctuation-only words (openers, closers, ``, . ; : ' / =``) are
skipped and belong to the window, so ``5.2 ( x 10^3 )`` is judged by its ``x``. The productions (glue.py holds the
character-level ones)::

    prose       no digit, every modifier joined to a letter               mg  ng·h/mL  h⁻¹  -fold  Cmax*  e.g.
    label       led by a letter that is no lone operator letter           AUC0-24  CYP2D6  t1/2  h-1  n=12
    number      not led by a letter, read as resolved numbers, no         12  (3.1)  45%  -15  <0.5  3/4  10^3
                dangling multiplier, not across a lone separator word     ('12 . 5' is one number split in two)
    citation    a reference pair, the reader's own production             [26,27]
    variable    (before the number, behind '=') any letter-led word       E = 0.9   x = 12   E0 = 0
    qualifier   (before the number) operators holding a qualifier         <  <=  ≤(  ~  p<  (p<  <-
    dispersion  a lone plus-minus                                         12 ± 3
    percent     a lone percent sign after the number, or after the        45 %   16.9 %, 13.1
                number before it
    infix       a lone - ~ x X * × · (or, after the number, a qualifier)  12 - 15   20 mg – 40 mg   2 x 3   1.25 > 1.0
                directly beside it, a number (or a unit then a number)
                on its other side; a lone multiplier before a ten is a
                scale, never a product
    range end   a word ending in a dash after its number: directly        0.5-\\n1.5   10- fold   150- and 300-mg
                followed by a number, or by a prose word (a compound)

Everything else beside a number (``1.2 e-3.``, ``(x 10^3)``, ``- ( 12 )``, ``1.2 e 3``, ``12 x3``, ``12 5th``,
``Table 3 -``, a cell ``5.0-``) makes it unresolved: one expression from the number through the refused word, named
``unresolved_shape:spaced_exponent`` / ``bracketed_multiplier`` / ``neighbour`` (the name only describes the refused
word; the allow-list decides). Each word is judged by its own shape, never by whether its own numbers resolved, so
nothing cascades and the cost is linear: a punctuation run is walked only from the two words that bound it.
"""
from __future__ import annotations

import re

from app.evidence.numgrammar import glue as G
from app.evidence.numgrammar import words as W
from app.evidence.numgrammar.alphabet import (
    DIGITS,
    EXP_MARKERS,
    LETTERS,
    MULTIPLIERS,
    OPENERS_BR,
    PERCENTS,
    PLUS_MINUS,
    QUALIFIERS,
    RANGE_WORDS,
    STARTERS,
    char_class,
)
from app.evidence.numgrammar.scan import TAIL_MULT

LEFT, RIGHT = -1, 1
SPACED_EXPONENT, BRACKETED, NEIGHBOUR = "spaced_exponent", "bracketed_multiplier", "neighbour"
_PUNCT_RE = re.compile(char_class(G.PUNCT, negate=False) + "+")
_PLAIN = re.compile("0|[1-9][0-9]{0,39}")                # a plain integer word: scan_word reads it as one token
# a word that is inert in every window position without further reading: a plain integer, or ASCII letters and slashes
# that start and end with a letter (two or more letters: no lone operator letter), after openers, before closers
_SIMPLE = re.compile("0|[1-9][0-9]{0,39}|[(\\[]*[A-Za-z][A-Za-z/]*[A-Za-z][)\\],;:.']*")


def punct_only(w: str) -> bool:
    return _PUNCT_RE.fullmatch(w) is not None


def lead_letter(w: str) -> bool:
    """The first character after a run of openers and starters is a letter."""
    i = 0
    while i < len(w) and (w[i] in OPENERS_BR or w[i] in STARTERS):
        i += 1
    return i < len(w) and w[i] in LETTERS


def qualifier_word(w: str) -> bool:
    """A run of operators holding a qualifier ends the word and what stands before it is inert glue ('<', '<=', '≤(',
    'p<', '(p<', '<-'): the qualifier production before a number."""
    tail = W.operator_tail(w)
    return any(c in QUALIFIERS for c in tail) and G.glue(w[:len(w) - len(tail)])


def _why(w: str, opened: bool) -> str:
    """The name of a refused word: an exponent letter, a multiplier behind an opener, or a plain refused neighbour."""
    lo, hi = G.core(w)
    if any(c in EXP_MARKERS and G.operator_letter(w, i) for i, c in enumerate(w)):
        return SPACED_EXPONENT
    if (opened or lo > 0) and lo < hi and w[lo] in MULTIPLIERS:
        return BRACKETED
    return NEIGHBOUR


class Window:
    """The window rule; ``Reader`` provides ``n``, ``t``, ``starts``, ``ends``, ``words``, ``text``, ``has_digit`` and
    ``scan``."""
    __slots__ = ()

    def number_word(self, k: int) -> bool:
        """Word *k* is a number: not led by a letter, read as resolved tokens, no dangling multiplier."""
        if not 0 <= k < self.n:
            return False
        w = self.text(k)
        if _PLAIN.fullmatch(w) is not None:
            return True
        if not self.has_digit(k) or lead_letter(w):
            return False
        sc = self.scan(k)
        return sc.kind == "numeric" and sc.tail != TAIL_MULT

    def far_number(self, k: int, side: int) -> bool:
        """The word beyond the infix word *k* is a number, or a prose word (a unit) and then a number."""
        j = k + side
        if self.number_word(j):
            return True
        if not 0 <= j < self.n or self.has_digit(j) or punct_only(self.text(j)):
            return False
        return G.prose(self.text(j)) and self.number_word(j + side)

    def _digit_word(self, k: int, split: bool) -> bool:
        """A word with a digit: a label, a word whose own numbers follow a starter ('n=12'), a number, a reference
        pair."""
        if lead_letter(self.text(k)):
            return self.scan(k).kind == "numeric" or G.label(self.text(k))
        if not split and self.number_word(k):
            return True
        sc = self.scan(k)
        return sc.kind == "rejected" and sc.why == "citation_list"

    def accepts(self, k: int, side: int, adjacent: bool, split: bool, assigned: bool) -> bool:
        """Word *k*, the first word of the window on *side* that is not punctuation, is an accepted production.
        *assigned*: an equals sign stands between it and the number (a variable name before it: 'E = 0.9')."""
        w = self.text(k)
        if _PLAIN.fullmatch(w) is not None:
            return not split
        if side == LEFT and assigned and G.variable(w):
            return True
        if self.has_digit(k):
            return self._digit_word(k, split)
        if G.prose(w):
            return True
        lo, hi = G.core(w)
        if w[lo:hi] == PLUS_MINUS:
            return True
        if hi - lo == 1 and w[lo] in PERCENTS and (side == RIGHT or self.number_word(k - 1)):
            return True                                   # '45 %', and '16.9 %, 13.1' (the number before's percent)
        if side == LEFT and qualifier_word(w):
            return True
        if side == LEFT and W.is_mult_word(w) and W.ten_shape(self.text(k + 1)):
            return False                                  # 'x 10^3' not joined to a mantissa: a scale, never a product
        infix = w in RANGE_WORDS or W.is_mult_word(w) or (side == RIGHT and all(c in QUALIFIERS for c in w))
        return infix and adjacent and self.far_number(k, side)

    def window_fault(self, fw: int, lw: int, dash: bool) -> tuple[int, int, str] | None:
        """(side, index of the refused word, why) for the number over words fw..lw, None when its window is inert.
        *dash*: word lw ends in a dash ('5.0-', '45%-'): a number directly after it (a range) or a prose word (a
        compound the text layer split) completes it, nothing else does."""
        for side, k0 in ((LEFT, fw - 1), (RIGHT, lw + 1)):
            if 0 <= k0 < self.n and _SIMPLE.fullmatch(self.words[k0]) is not None:
                continue                                  # a common neighbour: 'mg', 'ng/mL', '(CV', 'after', '12'
            k, split, opened, assigned = k0, False, False, False
            edge = self.t[self.starts[fw]] if side == LEFT else self.t[self.ends[lw] - 1]
            while 0 <= k < self.n and punct_only(self.text(k)):
                w = self.text(k)
                # a lone separator word between two digits splits one number ('12 . 5', '12 , 5'; not '24.5% , 17.8')
                split = split or (edge in DIGITS and G.SEPARATOR_PUNCT.issuperset(w))
                opened = opened or not OPENERS_BR.isdisjoint(w)
                assigned = assigned or "=" in w
                k += side
            need_end = side == RIGHT and dash
            if not 0 <= k < self.n:
                if need_end:
                    return side, lw, NEIGHBOUR
                continue
            w = self.text(k)
            if need_end:
                ok = k == k0 and (self.number_word(k) or (lead_letter(w) and G.prose(w)))
            else:
                facing = w[-1 if side == LEFT else 0]
                ok = self.accepts(k, side, k == k0, split and facing in DIGITS, assigned or w.endswith("="))
            if not ok:
                return side, k, _why(w, opened)
        return None


__all__ = ["LEFT", "RIGHT", "Window", "lead_letter", "punct_only", "qualifier_word"]
