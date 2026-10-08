"""The product production on the WINDOW (r7 round 3; a mixin of ``neighbours.Window``): a lone multiplier word
between two number words, a glued multiplier that ends a word, a footnote star before a scale (product.py states the
production).

The count is the last token of the word before the multiplier (closers aside), or of the word before a prose word that
is the count's unit ('8 weeks x 3', '20 mg x 2'); the amount is the first token of the word after it; a written-out
power of ten as the amount needs a unit production right after it, glued ('100mg') or as the next word ('100 mg').
"""
from __future__ import annotations

import re

from app.evidence.numgrammar import glue as G
from app.evidence.numgrammar import product as P
from app.evidence.numgrammar import words as W
from app.evidence.numgrammar.alphabet import OPENERS_BR, TRAILING, char_class
from app.evidence.numgrammar.tok import Tok
from app.evidence.numgrammar.units import unit_end, unit_led

_PUNCT_RE = re.compile(char_class(G.PUNCT, negate=False) + "+")


class ProductWindow:
    """``Reader`` provides ``n``, ``starts``, ``ends``, ``text``, ``has_digit``, ``scan``, ``mult_word`` and
    ``number_word``."""
    __slots__ = ()

    def unit_after(self, k: int, tok_end: int) -> bool:
        """A unit production follows the token ending at *tok_end* in word *k*: glued to it ('100mg'), or, the word
        ending there (closers aside), as the start of the next word ('100 mg', '(2 x 100) mg')."""
        w, e = self.text(k), tok_end - self.starts[k]
        if unit_end(w, e) > e:
            return True
        return all(c in TRAILING for c in w[e:]) and k + 1 < self.n and unit_led(self.text(k + 1))

    def product_fault(self, count: Tok | None, amount: Tok | None, ak: int) -> str:
        """'' when ``count MULT amount`` (the amount a token of word *ak*) is the product production, else the
        reviewer's name of the refusal."""
        if count is None or amount is None:
            w = self.text(ak) if 0 <= ak < self.n else ""
            return P.TEN if w and W.ten_shape(w) else P.OPERAND
        unit = P.plain(amount.shape, amount.decimal) and amount.end <= self.ends[ak] and self.unit_after(ak, amount.end)
        return P.fault(count, (amount.shape, amount.canonical, amount.decimal), unit)

    def amount_of(self, k: int) -> Tok | None:
        """The first token of number word *k* when it starts the word (one opener aside): a product's amount."""
        if not (0 <= k < self.n and self.number_word(k)):
            return None
        tok, w = self.scan(k).toks[0], self.text(k)
        return tok if tok.start <= self.starts[k] + (w[0] in OPENERS_BR) else None

    def count_of(self, k: int) -> Tok | None:
        """The last token of word *k* when it ends the word (closers aside): the count of a product."""
        if not (0 <= k < self.n and self.number_word(k)):
            return None
        tok, w = self.scan(k).toks[-1], self.text(k)
        return tok if all(c in TRAILING for c in w[tok.end - self.starts[k]:]) else None

    def count_before(self, j: int) -> Tok | None:
        """The count of a product whose multiplier follows word *j*: word *j*'s last token, or, when word *j* is a prose
        word (the count's unit: '8 weeks x 3', '20 mg x 2'), the last token of the word before it."""
        tok = self.count_of(j)
        if tok is not None or not 0 <= j < self.n or self.has_digit(j) or _PUNCT_RE.fullmatch(self.text(j)):
            return tok
        return self.count_of(j - 1) if G.prose(self.text(j)) else None

    def refused_amount(self, k: int) -> str:
        """Word *k* stands right after a lone multiplier and is no product's amount: the refusal's name, else ''."""
        if not (k >= 1 and self.mult_word(k - 1)):
            return ""
        return self.product_fault(self.count_before(k - 2), self.amount_of(k), k)


__all__ = ["ProductWindow"]
