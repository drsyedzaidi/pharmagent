"""The neighbour-word allow-list (r7 writer 3; a mixin of ``Reader``): the words around a resolved number.

A resolved number (the tokens of one word, or one multi-word token such as ``1.2 x 10^3`` or ``12 / 15``) keeps its
value only when the word on each side of it is an ACCEPTED PRODUCTION. Punctuation-only words (openers, closers, ``, . ;
: ' / =``) are skipped and belong to the window, so ``5.2 ( x 10^3 )`` is judged by its ``x``; the taint tiers walk the
SAME window (``window_taint``: r7 round 2), so a character the reader cannot read past a punctuation word, across a line
break too, leaves the number unresolved. The productions (glue.py holds the character-level ones)::

    prose       no digit, every modifier joined to a letter, every power   mg  ng·h/mL  h⁻¹  -fold  Cmax*  e.g.  m²
                after a unit symbol (glue.unit_powers); r7 round 4: never between a plain mantissa and a power of ten
                with no mantissa of its own ('1.2 times 10^3', '1.2 by 1e3': a spelled-out multiplier)
    label       led by a letter, no SCALE production on its printed text   AUC0-24  CYP2D6  t1/2  h-1  n=12
                (glue.py: never 'AUCx1000', 'AUC*1000', 'AUC:E3')
    number      a PLAIN number word: resolved tokens, no dangling          12  (3.1)  45%  -15  <0.5  3/4,
                multiplier, never a pure power (a detached scale: '1.2 10^3', r7 round 4 '1.2 1E+03'), never a raised
                ten after a number ending in a digit (r7 round 4: '1.2 10−3', '2.5 109/L', a dropped multiplier), never
                a ten behind a multiplier ('x 10'), never a fraction right after the number ('12 3/4') unless a list
                mark (, ; ) ]) separates them
    variable    (before the number, behind '=') any letter-led word       E = 0.9   x = 12   E0 = 0
    qualifier   (before the number) operators holding a qualifier         <  <=  ≤(  ~  p<  (p<  <-
    dispersion  a lone plus-minus                                         12 ± 3
    percent     a lone percent sign after the number, or after the        45 %   16.9 %, 13.1
                number before it
    infix       a lone - ~ x X * × · (or, after the number, a qualifier)  12 - 15   20 mg – 40 mg   2 x 3   1.25 > 1.0
                directly beside it, a plain number (or a unit then one) on its other side; a ten behind a multiplier is
                a scale, never a product
    range end   a word ending in a dash after its number: directly        0.5-\\n1.5   5- mg
                followed by a number, or (r7 round 4) by a unit production; never '10- fold' or '12- lOO' (decision 2)

Never accepted: a lone ':' or '/' word between the number and another number (``1 : 2``, ``5 / 12 / 2020``; the ratio
production ``12 / 15`` consumes its own slash), r7 round 4 such a word, or a word led by one, after the number and
before anything but a unit production (``12 / lS``, ``12 :lS``; ``12 / kg`` reads), a word of one letter, a sign and
digits right after the number unless that letter is a unit symbol (``1.2 é-3``; ``ka 0.12 h-1`` reads), a word led by a
multiplier right after the number (``12 xenon``, ``12 x3/4``). A RATIO is dimensionless and never an operand: a unit or
a multiplier after it, a multiplier before it, or a number before it without a list mark refuses it. A number whose
glued multiplier ends its word (``2x``) needs a plain number right after it.

Everything else beside a number makes it unresolved: one expression from the number through the refused word (and the
operand behind a refused multiplier, or the INSIDE words a ten or a detached power absorbs), named for the reviewer
(``mixed_number``, ``fraction``, ``detached_exponent``, ``detached_decimal``, ``ten_after_multiplier``,
``multiplier_operand``, ``spaced_exponent``, ``bracketed_multiplier`` or ``neighbour``): the name only
describes the refused word; the allow-list decides. Each word is judged by its own shape, never by whether its own
numbers resolved, so nothing cascades and the cost is linear: a punctuation run is walked only from the two words that
bound it.
"""
from __future__ import annotations

import re

from app.evidence.numgrammar import glue as G
from app.evidence.numgrammar import product as P
from app.evidence.numgrammar import words as W
from app.evidence.numgrammar.alphabet import (
    DIGITS,
    EQUALS,
    EXP_MARKERS,
    LETTERS,
    LIST_MARKS,
    MULTIPLIERS,
    OPENERS_BR,
    PERCENTS,
    PLAIN_POWERS,
    PLUS_MINUS,
    QUALIFIERS,
    RANGE_WORDS,
    RATIO_MARKS,
    SIGNS,
    STARTERS,
    char_class,
)
from app.evidence.numgrammar.product import RAISED_TEN_RE
from app.evidence.numgrammar.product_window import (
    _SIMPLE,
    LEFT,
    PLAIN_INT,
    POWER_MARK_RE,
    POWER_SHAPES,
    RIGHT,
    ProductWindow,
)
from app.evidence.numgrammar.scan import TAIL_MULT, TAIL_STAR
from app.evidence.numgrammar.units import ATOM_RE, unit_led

SPACED_EXPONENT, BRACKETED, NEIGHBOUR = "spaced_exponent", "bracketed_multiplier", "neighbour"
MIXED, FRACTION, DETACHED, FRAGMENT = "mixed_number", "fraction", "detached_exponent", "detached_decimal"
TEN, OPERAND = "ten_after_multiplier", "multiplier_operand"
_PUNCT_RE = re.compile(char_class(G.PUNCT, negate=False) + "+")
# one letter, a sign and digits: a spaced exponent unless the letter is a unit symbol with a unit exponent ('h-1', 'm2')
_ONE_LETTER_EXP = re.compile("(" + char_class(LETTERS, negate=False) + ")(" + char_class(SIGNS, negate=False)
                             + "?[0-9]{1,4})")
_ONE_LETTER_MAX = 12                                     # a letter, a sign, 4 digits and the openers/closers around
_UNIT_EXPONENT = re.compile(char_class(SIGNS, negate=False) + "[0-9]{1,2}|" + char_class(PLAIN_POWERS, negate=False))


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


def spaced_exponent(w: str) -> bool:
    """*w* (its openers and closers aside) is one letter, an optional sign and digits, and is no unit symbol with a unit
    exponent: a spaced exponent or a look-alike of one ('é-3', 'ε-03', 'ë3', 'q-3'; never 'h-1', 'L-1', 'm2')."""
    if len(w) > _ONE_LETTER_MAX or not any(c in DIGITS for c in w):
        return False
    lo, hi = G.core(w)
    m = _ONE_LETTER_EXP.fullmatch(w, lo, hi)
    return m is not None and not (ATOM_RE.fullmatch(m.group(1)) and _UNIT_EXPONENT.fullmatch(m.group(2)))


def _why(w: str, opened: bool) -> str:
    """The name of a refused word, for the reviewer only (the allow-list has already refused it)."""
    lo, hi = G.core(w)
    if any(c in EXP_MARKERS and G.operator_letter(w, i, (lo, hi)) for i, c in enumerate(w)) or spaced_exponent(w):
        return SPACED_EXPONENT
    lead = w[lo:lo + 1]
    if (opened or lo > 0) and lo < hi and lead in MULTIPLIERS:
        return BRACKETED
    if lead in MULTIPLIERS:
        return TEN if W.special(w[lo:hi]) == W.MULT_TEN else OPERAND
    if W.CARET_OR_SUP_RE.search(w) is not None:
        return DETACHED
    if w[hi - 1:hi] in RATIO_MARKS or w[-1:] in RATIO_MARKS:
        return FRACTION                                   # '12/ 15': the first part of a ratio
    if lead in G.SEPARATOR_PUNCT and w[lo + 1:lo + 2] in DIGITS:
        return FRAGMENT
    return FRACTION if lead in RATIO_MARKS else NEIGHBOUR


class Window(ProductWindow):
    """The window rule; ``Reader`` provides ``n``, ``t``, ``starts``, ``ends``, ``words``, ``text``, ``has_digit``,
    ``scan``, ``clean``, ``mult_word``, ``number_end``, ``marks`` and ``target`` (the tokens being judged)."""
    __slots__ = ()

    # ------------------------------------------------------------------ number words ---
    def number_word(self, k: int) -> bool:
        """Word *k* is a number word: not led by a letter, read as resolved tokens, no dangling multiplier."""
        if not 0 <= k < self.n:
            return False
        w = self.words[k]
        if PLAIN_INT.fullmatch(w) is not None:
            return True
        if not self.has_digit(k) or lead_letter(w) or w[0] in RATIO_MARKS:
            return False                                  # ':4', '/15': the second part of a ratio, never a number
        sc = self.scan(k)
        return sc.kind == "numeric" and sc.tail != TAIL_MULT and not sc.need_unit   # r7 round 3: "2x100" needs a unit

    def _shapes(self, k: int) -> set[str]:
        return {t.shape for t in self.scan(k).toks} if self.has_digit(k) else set()

    def plain_number(self, k: int) -> bool:
        """A number word that is no ratio, no pure power and no refused amount behind a multiplier: an infix operand."""
        return self.number_word(k) and "ratio" not in self._shapes(k) and not self.power_word(k) \
            and not self.refused_amount(k)

    def number_fault(self, k: int, side: int, listed: bool, kind: set[str], tight: bool = True,
                     adjacent: bool = True) -> str:
        """'' when the number word *k* may stand on *side* of the target, else why not."""
        if k in self.power_words and not kind & {"ratio", "power"}:
            return ""                                     # a scientific token's power word: a number ('4.5 x 10^9 3.2')
        if not self.number_word(k):
            if "ratio" in kind and side == LEFT and self.number_end(k) and not listed:
                return MIXED                              # '1O5 3/4': a rejected whole part is still a whole part
            return _why(self.words[k], False)
        if adjacent and (refused := self.refused_amount(k)):
            return refused                                # '1.2 x 10 3': the 3 beside a refused product's amount
        if listed:
            return ""
        toks = self.scan(k).toks
        if P.pure_power(toks[0]) or "power" in kind:
            return DETACHED                               # '1.2 10^3', '1.2\n10^9', '1.2 1E+03': a detached scale
        if tight and ((side == RIGHT and self.raised_word(k)) or (side == LEFT and "raised" in kind
                                                                   and self.t[self.ends[k] - 1] in DIGITS)):
            return DETACHED                               # r7 round 4: '1.2 10−3', '2.5 109/L': a dropped multiplier
        if "ratio" in kind and side == LEFT:
            return MIXED                                  # '12 3/4': a whole part before a fraction
        if tight and side == RIGHT and toks[0].start == self.starts[k] and any(t.shape == "ratio" for t in toks):
            return MIXED                                  # '12 3/4': an integer that ends its word, then a fraction
        return ""

    def far_fault(self, k: int, side: int) -> str:
        """'' when the word beyond the infix word *k* is a plain number, or a prose word (a unit) then a plain
        number."""
        j, mult = k + side, W.is_mult_word(self.words[k])
        if mult:                                          # the product production: count MULT amount (product.py)
            if side == RIGHT:
                return self.product_fault(self.target[-1] if self.target else None, self.amount_of(j), j)
            why = self.product_fault(self.count_before(j), self.target[0] if self.target else None, k + 1)
            return NEIGHBOUR if why == OPERAND else why   # before the number: the reviewer's name of r7 writer 3
        if 0 <= j < self.n:
            if self.plain_number(j):
                return ""
            if not (mult and side == RIGHT) and not self.has_digit(j) and not punct_only(self.words[j]) \
                    and G.prose(self.words[j]) and self.plain_number(j + side):
                return ""                                 # '20 mg – 40 mg'; a multiplier's operand: its next word
        return OPERAND if mult and side == RIGHT else NEIGHBOUR

    # ------------------------------------------------------------------ one judged word ---
    def word_fault(self, k: int, side: int, adjacent: bool, split: bool, assigned: bool, kind: set[str],
                   listed: bool, lw: int) -> str:
        """'' when word *k* (the first word of the window on *side* that is not punctuation) is an accepted production,
        else the name of the refusal."""
        w = self.words[k]
        if not self.clean(k):
            return NEIGHBOUR                              # never reached: the taint tiers read the same window first
        lo, hi = G.core(w)
        if side == RIGHT:
            tail = self.scan(lw).tail
            if tail == TAIL_MULT:                         # '2x 5': a glued multiplier is the product's
                return self.product_fault(self.target[-1], self.amount_of(k), k) if adjacent else OPERAND
            amount = self.amount_of(k) if tail == TAIL_STAR and adjacent else None
            if amount is not None and P.scale(amount):    # '12* 10', '12* 1000': the star may multiply a scale
                return P.name(amount.canonical)
            if adjacent and w[:1] in MULTIPLIERS and not W.is_mult_word(w):
                return _why(w, False)                     # '12 xenon', '12 x3/4', '12 x103': its operand is this word
            if "ratio" in kind and (unit_led(w[lo:hi]) or w[lo:lo + 1] in MULTIPLIERS):
                return FRACTION                           # '150/300 mg', '12/15 x 1000': a ratio is dimensionless
            if adjacent and w[lo:lo + 1] in RATIO_MARKS and not self.has_digit(k) and not unit_led(w[lo + 1:hi]):
                return FRACTION                           # r7 round 4: '12 /lS', '12 :lS'; '12 /kg' is a unit
            if spaced_exponent(w):
                return SPACED_EXPONENT                    # '1.2 é-3', '1.2 ε-03'
        tight = side == LEFT or self.t[self.ends[lw] - 1] in DIGITS
        if PLAIN_INT.fullmatch(w) is not None:
            return NEIGHBOUR if split else self.number_fault(k, side, listed, kind, tight, adjacent)
        if side == LEFT and assigned and G.variable(w):
            return ""
        if self.has_digit(k):
            if lead_letter(w):
                if "ratio" in kind and side == LEFT and self.number_end(k) and not listed:
                    return MIXED                          # 'AUC=12 3/4': 12 is a token, a whole part
                numeric = self.scan(k).kind == "numeric"
                if not (numeric or G.label(w)):
                    return _why(w, False)
                return "" if numeric or G.unit_powers(w) else DETACHED
            return NEIGHBOUR if split else self.number_fault(k, side, listed, kind, tight, adjacent)
        if G.prose(w):
            if adjacent and ("power" in kind if side == LEFT else k + 1 < self.n and POWER_MARK_RE.search(
                    self.words[k + 1]) is not None) and self._power_across(k, side, kind):
                return DETACHED                           # r7 round 4: '1.2 times 10^3', '1.2 by 1e3'
            return "" if G.unit_powers(w) else DETACHED  # 'm²' reads; 'lO³' is a look-alike of a detached power
        return self._operator_fault(k, w, lo, hi, side, adjacent, kind)

    def _operator_fault(self, k: int, w: str, lo: int, hi: int, side: int, adjacent: bool, kind: set[str]) -> str:
        """A word of operators beside the number: dispersion, percent, qualifier or infix, else refused."""
        if w[lo:hi] == PLUS_MINUS:
            return ""
        if hi - lo == 1 and w[lo] in PERCENTS and (side == RIGHT or self.number_word(k - 1)):
            return ""                                     # '45 %', and '16.9 %, 13.1' (the number before's percent)
        if side == LEFT and qualifier_word(w):
            return ""
        mult = W.is_mult_word(w)
        if mult and "ratio" in kind:
            return OPERAND                                # a ratio is never an operand
        infix = w in RANGE_WORDS or mult or (side == RIGHT and all(c in QUALIFIERS for c in w))
        if infix and adjacent:
            return self.far_fault(k, side)
        return OPERAND if mult and side == RIGHT else _why(w, False)

    # ------------------------------------------------------------------ the window ---
    def _walk(self, k: int, side: int) -> int:
        """The first word from *k* on *side* that is not punctuation-only (-1 or n past the text)."""
        while 0 <= k < self.n and punct_only(self.words[k]):
            k += side
        return k

    def window_taint(self, fw: int, lw: int) -> str:
        """The reviewer fragment when a word of the window (the punctuation words skipped and the first word past them,
        on each side: the very words the allow-list judges) holds a character the reader cannot read; '' otherwise."""
        marks = self.marks
        for side, k, where in ((LEFT, fw - 1, "one word before"), (RIGHT, lw + 1, "one word after")):
            end = self._walk(k, side)
            for j in range(k, end + side, side):
                if 0 <= j < self.n and (p := marks.first_bad(self.starts[j], self.ends[j])) >= 0:
                    return f"{marks.label(p)} {where}"
        return ""

    def window_fault(self, fw: int, lw: int, dash: bool) -> tuple[int, int, str] | None:
        """(side, index of the refused word, why) for the number over words fw..lw, None when its window is inert.
        *dash*: word lw ends in a dash ('5.0-', '45%-'): a number directly after it (a range) or a prose word (a
        compound the text layer split) completes it, nothing else does."""
        kind = {t.shape for t in self.target}
        if len(self.target) == 1 and (kind & POWER_SHAPES or RAISED_TEN_RE.fullmatch(self.target[0].canonical)):
            kind |= self.alone_kind(self.target[0], fw)   # a power or a raised ten ('10' of '1.2 10')
        if self.scan(lw).need_unit and not (lw + 1 < self.n and unit_led(self.words[lw + 1])):
            return RIGHT, lw, P.name(self.target[-1].canonical)   # '2x100': a power-of-ten amount needs its unit
        if (split := self.detached_decimal(fw, lw)) is not None:
            return split
        plain = not kind & {"ratio", "power", "raised"} and not self.scan(lw).tail and not dash
        for side, k0 in ((LEFT, fw - 1), (RIGHT, lw + 1)):
            if plain and 0 <= k0 < self.n and (_SIMPLE.fullmatch(w0 := self.words[k0]) is not None and (
                    side == LEFT or not self.power_next(k0)) or PLAIN_INT.fullmatch(w0) is not None and (
                    not self.refused_amount(k0) if side == LEFT else RAISED_TEN_RE.fullmatch(w0) is None)):
                continue                                  # a common neighbour: 'mg', 'ng/mL', '(CV', 'after', '12'
            k, split, opened, assigned, ratio_mark, listed = k0, False, False, False, False, False
            edge = self.t[self.starts[fw]] if side == LEFT else self.t[self.ends[lw] - 1]
            while 0 <= k < self.n and punct_only(self.words[k]):
                w = self.words[k]
                # a lone separator word between two digits splits one number ('12 . 5', '12 , 5'; not '24.5% , 17.8')
                split = split or (edge in DIGITS and G.SEPARATOR_PUNCT.issuperset(w))
                opened = opened or not OPENERS_BR.isdisjoint(w)
                assigned = assigned or not EQUALS.isdisjoint(w)
                ratio_mark = ratio_mark or w in RATIO_MARKS
                listed = listed or not LIST_MARKS.isdisjoint(w)
                k += side
            fault = self._side_fault(k, k0, side, fw, lw, dash, (split, opened, assigned, ratio_mark, listed), kind)
            if fault is not None:
                return fault
        return None

    def detached_decimal(self, fw: int, lw: int) -> tuple[int, int, str] | None:
        """r7 round 3: a number without a decimal mark whose word ENDS in the page's decimal mark right after it, then a
        word that starts with a digit ('0. 5', '12.\n5', '1,234. 5'), is one number the text layer split, refused like
        '12 .5' (a comma on a point page is a list mark, a point on a comma page a sentence end, '12.5. 15' and
        '35. [35]' a sentence end)."""
        t, last, first = self.t, self.target[-1], self.target[0]
        if lw + 1 < self.n and t[self.ends[lw] - 1] == self.style and last.end == self.ends[lw] - 1 \
                and not last.decimal and t[self.starts[lw + 1]] in DIGITS:
            return RIGHT, lw + 1, FRAGMENT
        if fw >= 1 and t[self.starts[fw]] in DIGITS and first.start == self.starts[fw] \
                and t[self.ends[fw - 1] - 1] == self.style and self.number_end(fw - 1, 1):
            before = self.scan(fw - 1).toks
            if before and not before[-1].decimal:
                return LEFT, fw - 1, FRAGMENT
        return None

    def _side_fault(self, k: int, k0: int, side: int, fw: int, lw: int, dash: bool, flags: tuple[bool, ...],
                    kind: set[str]) -> tuple[int, int, str] | None:
        split, opened, assigned, ratio_mark, listed = flags
        need_end = side == RIGHT and dash
        if not 0 <= k < self.n:
            if need_end:
                return side, lw, NEIGHBOUR
            if side == RIGHT and self.scan(lw).tail == TAIL_MULT:
                return side, lw, OPERAND                  # '1.2 ×' at the end of the text
            return None
        w = self.words[k]
        if need_end:                                      # r7 round 4: a number or a unit production ('5- mg'),
            ok = k == k0 and (self.number_word(k) or unit_led(w[G.core(w)[0]:]))   # never '10- fold', '12- lOO'
            return None if ok else (side, k, _why(w, opened))
        if ratio_mark and self.has_digit(k) and not lead_letter(w):
            return side, k, FRACTION                      # '1 : 2', '5 / 12 / 2020': no ratio production took it
        if ratio_mark and side == RIGHT and lead_letter(w) and not unit_led(w[G.core(w)[0]:]):
            return side, k, FRACTION                      # r7 round 4: '12 / lS', '12 : lS'; '12 / kg' is a unit
        facing = self.words[lw if side == RIGHT else k][-1]
        listed = listed or facing in LIST_MARKS
        why = self.word_fault(k, side, k == k0, split and w[-1 if side == LEFT else 0] in DIGITS,
                              assigned or w[-1] in EQUALS, kind, listed, lw)
        if why and opened and w[G.core(w)[0]:G.core(w)[0] + 1] in MULTIPLIERS:
            why = BRACKETED                               # '5.2 ( x 10^3 )': a scale in a bracket after the number
        return (side, k, why) if why else None


__all__ = ["LEFT", "RIGHT", "Window", "lead_letter", "punct_only", "qualifier_word", "spaced_exponent"]
