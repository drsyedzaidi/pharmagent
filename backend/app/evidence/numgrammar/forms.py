"""The multi-word PRODUCTIONS of the tokenizer (a mixin of ``Reader``): the two ways a number reads across one gap run.

    multiplier_power = mantissa [GAP] MULT [GAP] "10" power      -> one ``sci`` token
                       ``1.2 x 10^3``  ``1.2 ×10⁻³``  ``1.2× 10^3``  ``1.2 ×\\n10^3``  ``1.2 x 10^3.``  ``1.2 x
                       10^3/mL``
    ratio            = int GAP "/" GAP int                         -> one ``ratio`` token (r7, owner decision 6)
                       ``12 / 15`` (value 12/15, parts (12, 15)); the glued ``12/15`` is read inside its word (scan.py)

The mantissa is ONE plain token that ends its word: never a ratio, a power or a thousands-grouped number (a thousands
group never carries an exponent: ``1.234.567 x 10^3`` is unresolved, r7 round 2). The power word starts with the power
and may go on only the way a glued token may (``scan.lead_power``: a trailing closer, a unit, a denominator), so
``1.2 x 10^3.`` reads like ``1.2x10^3.`` (r7 round 2).

r7 round 2: this module holds ONLY the accepting productions. Every shape they do not accept (a mixed number ``12 3/4``,
a detached exponent ``10 ^ 3``, a detached decimal ``12 .5``, a slash or colon between numbers ``1 : 2``, a ten behind a
multiplier ``2 x 10 mg``, a multiplier without a number ``2 × daily``) is refused by the neighbour-word allow-list
(neighbours.py): a number keeps its value only beside an accepted production, so no detector of a bad shape is needed.
"""
from __future__ import annotations

from app.evidence.numgrammar import words as W
from app.evidence.numgrammar.alphabet import MULTIPLIERS, SLASH
from app.evidence.numgrammar.body import decimal_mantissa, to_float
from app.evidence.numgrammar.scan import RATIO_INT, TAIL_MULT, Scan, lead_power, ratio_parts, read_power
from app.evidence.numgrammar.tok import Tok

# a mantissa is one plain number: never a ratio, a power, a scientific token or a thousands-grouped number
_NO_MANTISSA = frozenset({"sci", "bare_power", "ratio", "grouped", "grouped_decimal"})


class Forms:
    """The two productions; ``forms`` returns the new ``consumed`` index, or None when neither applies."""
    __slots__ = ()

    def forms(self, a: int, sc: Scan, out: list[Tok], nxt: str) -> int | None:
        """The multiplier-power or the spaced-ratio production starting at word *a*, when it applies."""
        if not nxt:
            return None
        j = a + 1
        if sc.tail == TAIL_MULT:                                       # '1.2× 10^3'
            at = lead_power(nxt, 0) if self.joinable(a, sc) else -1
            return self.sci(out, a, sc, j, nxt, at) if at >= 0 else None
        if W.is_mult_word(nxt):                                        # '1.2 x 10^3'
            if j + 1 >= self.n or not self.joinable(a, sc):
                return None
            power = self.text(j + 1)
            at = lead_power(power, 0)
            return self.sci(out, a, sc, j + 1, power, at) if at >= 0 else None
        if nxt[0] in MULTIPLIERS and self.joinable(a, sc):            # '1.2 ×10^3', '1.2 x10^3.'
            at = lead_power(nxt, 1)
            return self.sci(out, a, sc, j, nxt, at) if at >= 0 else None
        if nxt in SLASH and j + 1 < self.n and W.digit_led(self.text(j + 1)):   # '12 / 15'
            tok = self.spaced_ratio(a, sc, j + 1)
            if tok is not None:
                return self.put(out, a, j + 1, self.flagged((tok,), a, j + 1))
        return None

    def joinable(self, a: int, sc: Scan) -> bool:
        """One plain token that ends the word (before a glued multiplier): a mantissa a power can join."""
        t = sc.toks[0]
        tail = 1 if sc.tail == TAIL_MULT else 0
        return len(sc.toks) == 1 and t.shape not in _NO_MANTISSA and t.end == self.ends[a] - tail

    def sci(self, out: list[Tok], a: int, sc: Scan, last: int, power_text: str, at: int) -> int:
        """One scientific token from the mantissa of word *a* and the power at ``power_text[at:]`` (word *last*); the
        token ends where the power ends, whatever trailing closer or unit the power word goes on with."""
        t = sc.toks[0]
        mant, mark = t.canonical, t.decimal
        if t.shape == "style_dependent" and not mark:             # '1.234 x 10^3' on a comma page: a decimal point
            mant, mark = decimal_mantissa(self.t[t.start:t.end])
        pw = read_power(power_text, at, len(power_text))
        if pw is None or pw[0] is None or (value := to_float(f"{mant}e{pw[0]}")) is None:
            end = self.starts[last] + (pw[1] if pw is not None else len(power_text))
            return self.expr(out, a, last, t.start, end, "exponent_range", (t.value,))
        tok = Tok(t.start, self.starts[last] + pw[1], value, "sci", (to_float(mant),), t.qualified, t.ranged,
                  f"{mant}e{pw[0]}", mark)
        return self.put(out, a, last, self.flagged((tok,), a, last))

    def spaced_ratio(self, a: int, sc: Scan, last: int) -> Tok | None:
        """``12 / 15``: one unsigned integer token ending word *a*, a lone ``/`` word, and word *last* an unsigned
        integer (trailing closers allowed) that no further ``/ N`` continues and no whole part precedes; None for any
        other shape (the allow-list then judges the words one by one)."""
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
