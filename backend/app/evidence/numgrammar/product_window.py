"""The product production on the WINDOW (r7 round 3; a mixin of ``neighbours.Window``): a lone multiplier word
between two number words, a glued multiplier that ends a word, a footnote star before a scale (product.py states the
production); r7 round 4: the DETACHED SCALES of the window, a pure power ('1E+03'), a raised ten after a number
('1.2 10−3', '2.5 109/L') and a power of ten one prose word after a number ('1.2 times 10^3').

The count is the last token of the word before the multiplier (closers aside), or of the word before a prose word that
is the count's unit ('8 weeks x 3', '20 mg x 2'); the amount is the first token of the word after it; a written-out
power of ten as the amount needs a unit production right after it, glued ('100mg') or as the next word ('100 mg').
"""
from __future__ import annotations

import re

from app.evidence.numgrammar import glue as G
from app.evidence.numgrammar import product as P
from app.evidence.numgrammar import words as W
from app.evidence.numgrammar.alphabet import (
    ASCII_LETTERS,
    CLOSERS_BR,
    CONTEXT_LETTERS,
    OPENERS_BR,
    PERCENTS,
    SLASH,
    TRAILING,
    char_class,
)
from app.evidence.numgrammar.tok import Tok
from app.evidence.numgrammar.units import unit_end, unit_led

_PUNCT_RE = re.compile(char_class(G.PUNCT, negate=False) + "+")
PLAIN_INT = re.compile("0|[1-9][0-9]{0,39}")                # a plain integer word: scan_word reads it as one token
# a word that is inert beside a plain number without further reading: ASCII letters and slashes that start with a letter
# that is no operator letter and end with a letter (two or more letters), after openers, before closers
_SIMPLE = re.compile(char_class(OPENERS_BR, negate=False) + "*"
                     + char_class(CONTEXT_LETTERS & ASCII_LETTERS, negate=False)
                     + char_class(ASCII_LETTERS | SLASH, negate=False) + "*" + char_class(ASCII_LETTERS, negate=False)
                     + char_class(TRAILING | CLOSERS_BR, negate=False) + "*")
LEFT, RIGHT = -1, 1


class ProductWindow:
    """``Reader`` provides ``n``, ``t``, ``starts``, ``ends``, ``words``, ``text``, ``has_digit``, ``scan``,
    ``mult_word``, ``number_word`` and ``refused_amount``."""
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


    # ------------------------------------------------------------- r7 round 4: detached scales ---
    def power_word(self, k: int) -> bool:
        """Word *k* starts with a power of ten that has no mantissa of its own ('10^3', '(10^-4)', '10^9,', '10^6/L',
        r7 round 4: '1e3', '10E+03')."""
        sc = self.scan(k) if 0 <= k < self.n and self.has_digit(k) else None
        return sc is not None and sc.kind == "numeric" and P.pure_power(sc.toks[0])

    def raised_word(self, k: int) -> bool:
        """r7 round 4: word *k* starts with a raised ten ('10', '103', '109/L', '10-3'; product.raised_ten)."""
        sc = self.scan(k) if 0 <= k < self.n and self.has_digit(k) else None
        return sc is not None and sc.kind == "numeric" and self.raised(sc.toks[0], k)

    def raised(self, tok: Tok, k: int) -> bool:
        """*tok* starts word *k* and is a raised ten that no percent sign follows (a percentage is a value: '10%')."""
        return P.raised_ten(tok) and tok.start == self.starts[k] and self.t[tok.end:tok.end + 1] not in PERCENTS

    def _power_across(self, k: int, side: int, kind: set[str]) -> bool:
        """r7 round 4: the prose word *k* stands between a MANTISSA (one plain number: an int or a decimal, never a
        scientific, power, ratio or grouped number, as in the multiplier-power production) and a power of ten with no
        mantissa of its own: '1.2 times 10^3', '1.2 by 1e3' (a spelled-out multiplier): the scale is no value and the
        number no exact one. '1.2E+03 and 1e5' is two values (the left number carries its own power), and a written-out
        power such as '1000' is not taken ('1 in 10', '5 to 100' are its shape)."""
        if side == RIGHT:
            return len(self.target) == 1 and P.plain(self.target[0].shape, self.target[0].decimal) \
                and self.power_word(k + 1)
        if "power" not in kind or not self.number_word(k - 1):
            return False
        last = self.scan(k - 1).toks[-1]
        return P.plain(last.shape, last.decimal)

    def _common(self, k: int, side: int) -> bool:
        """Word *k* is a common neighbour of a plain number that needs no further reading: a simple letter word (not
        before a power of ten: '1.2 times 10^3') or a plain integer (not a refused amount, no raised ten after it)."""
        w = self.words[k]
        if _SIMPLE.fullmatch(w) is not None:
            return not (side == RIGHT and self.power_word(k + 1))
        return PLAIN_INT.fullmatch(w) is not None and not (side == LEFT and self.refused_amount(k)) \
            and not (side == RIGHT and self.raised_word(k))


__all__ = ["LEFT", "RIGHT", "ProductWindow"]
