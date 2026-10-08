"""The multi-word forms of the tokenizer (a mixin of ``Reader``): the expressions a number forms with the words one
gap run away from it. ``Reader`` (tokenizer.py) owns the words and the emission; this module owns the rules.

ACCEPT   ``1.2 x 10^3``  ``1.2 ×10⁻³``  ``1.2× 10^3``  ``1.2 ×\\n10^3``      -> one ``sci`` token
REJECT   ``2 x 10 mg`` ``1.2 x 10 3`` ``× 103``          ten_after_multiplier
         ``2 × daily`` ``1.2 ×`` ``1.2 × l0³``            multiplier_operand
         ``10 ^ 3`` ``10 ³`` ``10^ 3``                    detached_exponent
         ``3.4 E-03`` ``2012 e1553``                      spaced_exponent
         ``12 3/4``                                       mixed_number (a lone ``3/4`` is a ratio)
         ``12 /15`` ``12/ 15`` ``1 : 2`` ``5 / 12 / 2020`` fraction (a ratio, a date, a score: never two exact values)
ACCEPT   ``12 / 15``                                      one ``ratio`` token (r7, owner decision 6: value 12/15, parts
                                                          (12, 15); the slash spaced on both sides, nothing else)
         ``12 .5`` ``12 ,5``                              detached_decimal (a text layer split the number at its point)
         ``5.2 (x10^3 cells)`` ``5.2 [× 10^3]``           bracketed_multiplier (the scale hides in a bracket)
"""
from __future__ import annotations

from app.evidence.numgrammar import words as W
from app.evidence.numgrammar.alphabet import BRACKET_MULTIPLIERS, DIGITS, MULTIPLIERS, OPENERS_BR
from app.evidence.numgrammar.body import decimal_mantissa, to_float
from app.evidence.numgrammar.scan import RATIO_INT, TAIL_MULT, TAIL_STAR, Scan, ratio_parts, read_power, scan_word
from app.evidence.numgrammar.tok import MAX_PARTS, Tok

_FRAGMENT_LEADS = frozenset(".,")


class Forms:
    """The form rules; every method returns the new ``consumed`` index or None when the form does not apply."""
    __slots__ = ()

    def forms(self, a: int, sc: Scan, consumed: int, out: list[Tok], nxt: str, prev: str) -> int | None:
        """The multi-word forms; the new ``consumed`` index of the first that applies, None when none does."""
        if sc.tail:
            done = self.form_tail(a, sc, out, nxt)
            if done is not None:
                return done
        if nxt:
            c = nxt[0]
            if c in MULTIPLIERS:
                done = self.form_mult(a, sc, out, nxt)
            elif c == "e" or c == "E":
                done = self.form_exponent(a, sc, out, nxt)
            elif c in DIGITS:
                done = self.form_fraction(a, sc, out, nxt)
            elif c in OPENERS_BR:
                done = self.form_bracket_mult(a, sc, out, nxt)
            elif c in _FRAGMENT_LEADS:
                done = self.form_fragment(a, sc, out, nxt)
            else:
                done = None
            if done is None and c in W.RATIO_CHARS:
                done = self.form_ratio(a, sc, out, nxt)
            if done is not None:
                return done
        return self.form_detached(a, sc, consumed, out, nxt, prev)

    # ------------------------------------------------------------------ forms ---
    def joinable(self, a: int, sc: Scan) -> bool:
        """One plain token that ends the word (before a glued multiplier): a mantissa a power can join."""
        t = sc.toks[0]
        tail = 1 if sc.tail == TAIL_MULT else 0
        return len(sc.toks) == 1 and t.shape not in ("sci", "bare_power") and t.end == self.ends[a] - tail

    def sci(self, out: list[Tok], a: int, sc: Scan, last: int, power_text: str, at: int) -> int:
        """One scientific token from the mantissa of word *a* and the power at ``power_text[at:]`` (word *last*)."""
        t = sc.toks[0]
        mant, mark = t.canonical, t.decimal
        if t.shape == "style_dependent" and not mark:             # '1.234 x 10^3' on a comma page: a decimal point
            mant, mark = decimal_mantissa(self.t[t.start:t.end])
        pw = read_power(power_text, at, len(power_text))
        value, canon = None, ""
        if pw is not None and pw[0] is not None:
            canon = f"{mant}e{pw[0]}"
            value = to_float(canon)
        if value is None:
            return self.expr(out, a, last, t.start, self.ends[last], "exponent_range", (t.value,))
        tok = Tok(t.start, self.ends[last], value, "sci", (to_float(mant),), t.qualified, t.ranged, canon, mark)
        self.put(out, a, last, self.flagged((tok,), a, last)[0])
        return last

    def operand(self, out: list[Tok], a: int, sc: Scan, last: int) -> int:
        """A multiplier without a number to its right: the left token and the operand word are one expression."""
        return self.expr(out, a, last, sc.toks[0].start, self.ends[last], "multiplier_operand",
                         self.parts_of(a, last))

    def ten(self, out: list[Tok], a: int, sc: Scan, ten_word: int) -> int:
        return self._ten(out, a, ten_word, sc.toks[0].start, consumed_from=a)

    def form_tail(self, a: int, sc: Scan, out: list[Tok], nxt: str) -> int | None:
        """A multiplier or star run glued to the end of the word (``1.2×``, ``2x``, ``12*``)."""
        j = a + 1
        if sc.tail == TAIL_STAR:                                           # '12* 10^3': a star is not a multiplier here
            if nxt and (W.BARE_POWER_RE.fullmatch(nxt) or W.ten_shape(nxt)):
                return self.operand(out, a, sc, j)
            return None
        if not nxt:
            return self.operand(out, a, sc, a)
        if self.joinable(a, sc) and W.BARE_POWER_RE.fullmatch(nxt):
            return self.sci(out, a, sc, j, nxt, 2)
        if W.ten_shape(nxt):
            return self.ten(out, a, sc, j)
        return None if self.numeric_word(j) else self.operand(out, a, sc, j)

    def form_mult(self, a: int, sc: Scan, out: list[Tok], t1: str) -> int | None:
        """A word led by a multiplier character right after the number (``x``, ``×10^3``, ``xenon``, ``x1O³``)."""
        j = a + 1
        if W.is_mult_word(t1):
            return self._mult_word(a, sc, j, out)
        sp = W.special(t1)
        if sp == W.MULT_POWER:
            return self.sci(out, a, sc, j, t1, 3) if self.joinable(a, sc) else self.operand(out, a, sc, j)
        if sp == W.MULT_TEN or W.ten_shape(t1[1:]):                         # 'x103', 'x10mg': a ten, no power
            return self.ten(out, a, sc, j)
        rest = scan_word(t1[1:], self.starts[j] + 1, self.style) if len(t1) > 1 and t1[1] in DIGITS else None
        product = rest is not None and rest.kind == "numeric" and rest.toks[0].shape != "ratio"   # 'x3/4': an operand
        return None if product else self.operand(out, a, sc, j)

    def _mult_word(self, a: int, sc: Scan, j: int, out: list[Tok]) -> int | None:
        """``N x <operand>``: the operand is the word after the multiplier word."""
        k = j + 1
        if k >= self.n:
            return self.operand(out, a, sc, j)
        t2 = self.text(k)
        if self.joinable(a, sc) and W.BARE_POWER_RE.fullmatch(t2):
            return self.sci(out, a, sc, k, t2, 2)
        if W.ten_shape(t2):
            return self.ten(out, a, sc, k)
        return None if self.numeric_word(k) else self.operand(out, a, sc, k)

    def form_exponent(self, a: int, sc: Scan, out: list[Tok], nxt: str) -> int | None:
        """``3.4 E-03``: an exponent in the next word."""
        if W.E_WORD_RE.fullmatch(nxt):
            return self.expr(out, a, a + 1, sc.toks[0].start, self.ends[a + 1], "spaced_exponent",
                             self.parts_of(a, a))
        return None

    def form_fraction(self, a: int, sc: Scan, out: list[Tok], nxt: str) -> int | None:
        """``12 3/4``: an integer followed by a fraction word is a mixed number."""
        t = sc.toks[0]
        if len(sc.toks) == 1 and t.shape == "int" and t.end == self.ends[a] and "/" in nxt \
                and W.FRACTION_RE.fullmatch(nxt):
            return self.expr(out, a, a + 1, t.start, self.ends[a + 1], "mixed_number", (t.value,))
        return None

    def form_detached(self, a: int, sc: Scan, consumed: int, out: list[Tok], nxt: str, prev: str) -> int | None:
        """``10 ^ 3``, ``10 ³``, ``10^ 3``: an exponent word next to a number that is not a power."""
        if nxt and W.exp_like(nxt):
            last = self.absorb(a + 1)
            return self.expr(out, a, last, sc.toks[0].start, self.ends[last], "detached_exponent",
                             self.parts_of(a, a))
        if prev and W.exp_like(prev):
            if consumed == a - 1 and out and out[-1].value is None and out[-1].end == self.ends[a - 1]:
                before = out.pop()                                  # the rejected word before holds the caret
                return self.expr(out, a - 1, a, before.start, self.ends[a], "detached_exponent",
                                 before.parts + self.parts_of(a, a))
            if consumed < a - 1:
                return self.expr(out, a - 1, a, self.starts[a - 1], self.ends[a], "detached_exponent",
                                 self.parts_of(a, a))
        return None

    def form_ratio(self, a: int, sc: Scan, out: list[Tok], nxt: str) -> int | None:
        """``12 / 15``, ``12 /15``, ``1 : 2``, ``5 / 12 / 2020``: a ratio, a score or a date. A slash or colon word
        between two numbers (spaced or glued to the right one) makes the whole run one unresolved expression, the way
        an unspaced ``12/15`` already is: two exact values are never the reading. (``12/ 15``, the slash glued to the
        left number, is a rejected word: ``Reader.process`` continues it with ``ratio_run``.)"""
        j = a + 1
        if nxt in W.RATIO_CHARS:
            if j + 1 >= self.n or not W.digit_led(self.text(j + 1)):
                return None
            last = j + 1
            tok = self.spaced_ratio(a, sc, last) if nxt == "/" else None
            if tok is not None:
                self.put(out, a, last, self.flagged((tok,), a, last)[0])
                return last
        elif len(nxt) > 1 and nxt[0] in W.RATIO_CHARS and nxt[1] in DIGITS:
            last = j
        else:
            return None
        return self.ratio_run(out, a, last, sc.toks[0].start, self.parts_of(a, a))

    def spaced_ratio(self, a: int, sc: Scan, last: int) -> Tok | None:
        """``12 / 15``: one unsigned integer token ending word *a*, a lone ``/`` word, and word *last* an unsigned
        integer (trailing closers allowed) that no further ``/ N`` continues; None for any other shape."""
        t = sc.toks[0]
        word = self.text(last)
        den = word.rstrip(W.RATIO_TRAIL)
        if len(sc.toks) != 1 or t.shape != "int" or t.end != self.ends[a] or not RATIO_INT.fullmatch(t.canonical) \
                or not RATIO_INT.fullmatch(den):
            return None
        if last + 2 < self.n and self.text(last + 1) in W.RATIO_CHARS and W.digit_led(self.text(last + 2)):
            return None                                              # '5 / 12 / 2020': a date or a list
        if a and self.number_end(a - 1):
            return None                                              # '4 12 / 15': a whole part may stand before
        rp = ratio_parts(t.canonical, den)
        if rp is None:
            return None
        return Tok(t.start, self.starts[last] + len(den), rp[0], "ratio", rp[2], t.qualified, False, rp[1], "")

    def ratio_run(self, out: list[Tok], a: int, last: int, start: int, first: tuple[float, ...]) -> int:
        """The ratio expression of words a..last (*first*: the numbers of word a), continued over further ``/ N`` pairs
        (``5 / 12 / 2020``)."""
        while last + 2 < self.n and self.text(last + 1) in W.RATIO_CHARS and W.digit_led(self.text(last + 2)):
            last += 2
        parts = (first + self.parts_of(a + 1, last))[:MAX_PARTS]
        return self.expr(out, a, last, start, self.ends[last], "fraction", parts)

    def form_fragment(self, a: int, sc: Scan, out: list[Tok], nxt: str) -> int | None:
        """``12 .5``, ``12 ,5``: a whole number whose decimal tail stands one word away (a text layer split the number
        at its point): the number is only a part of what the page shows."""
        t = sc.toks[-1]
        if len(nxt) > 1 and nxt[1] in DIGITS and t.shape == "int" and t.end == self.ends[a]:
            return self.expr(out, a, a + 1, sc.toks[0].start, self.ends[a + 1], "detached_decimal", self.parts_of(a, a))
        return None

    def form_bracket_mult(self, a: int, sc: Scan, out: list[Tok], nxt: str) -> int | None:
        """``5.2 (x10^3 cells/mL)``, ``5.2 [× 10^3]``: a scale in a bracket after the number. The multiplier opens the
        next word (glued to its ten, or alone with the ten one word further)."""
        if len(nxt) < 2 or nxt[1] not in BRACKET_MULTIPLIERS:
            return None
        rest = nxt[2:]
        if rest:
            return self.expr(out, a, a + 1, sc.toks[0].start, self.ends[a + 1], "bracketed_multiplier",
                             self.parts_of(a, a)) if W.ten_shape(rest) else None
        if a + 2 < self.n and (W.ten_shape(self.text(a + 2)) or W.BARE_POWER_RE.match(self.text(a + 2))):
            return self.expr(out, a, a + 2, sc.toks[0].start, self.ends[a + 2], "bracketed_multiplier",
                             self.parts_of(a, a))
        return None
