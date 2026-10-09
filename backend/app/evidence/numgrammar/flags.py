"""The qualifier and range MARKS of a resolved number (a mixin of ``Reader``; moved out of tokenizer.py in r7 round 3).

``Tok.qualified``: a qualifier stands before the number through a run of operators ('<5', '<= 5', '< ( 5', '< , 12').
``Tok.ranged``: the number is one end of a printed range: a dash glued to it or a lone range word ('-' '~') beside it
('12-15', '12 - 15', '12 -15', '5.0- 12.4', '12.5 ~ 15.0'), a label that ends in a joiner before it ('AUC0-' line break
'24'), and (r7 round 3, the round-3 audit) a unit or a percent between the number and the range word: '20 mg - 40 mg',
'12 h ~ 15 h', '45 % - 55 %', '20 mg -40 mg', and a unit glued to the dash and the other end ('20 mg-40 mg': the unit,
a dash and digits that are no unit exponent, two digits at most, unless another unit follows them). These marks only
raise the censored_value flag (value_bounds.py); they never change a value.
"""
from __future__ import annotations

from app.evidence.numgrammar import glue as G
from app.evidence.numgrammar import words as W
from app.evidence.numgrammar.alphabet import DIGITS, HYPHEN, OPERATORS, PERCENTS, RANGE_WORDS, SLASH, TRAILING
from app.evidence.numgrammar.glue import PUNCT
from app.evidence.numgrammar.neighbours import lead_letter, punct_only, qualifier_word
from app.evidence.numgrammar.tok import Tok
from app.evidence.numgrammar.units import ATOM_RE, unit_led

_QUALIFY_ENDS = OPERATORS | PUNCT              # a word before a qualified number ends in one of these
_JOINER_END = HYPHEN | SLASH                  # a word led by a letter that ends in one of these may continue over a gap
UNIT_EXPONENT_DIGITS = 2                      # a unit's signed exponent ('h-1', 'mL-1') has at most two digits
_TRAIL = "".join(sorted(TRAILING))


class Flags:
    """``Reader`` provides ``n``, ``t``, ``starts``, ``ends``, ``text`` and ``number_end``."""
    __slots__ = ()

    def unit_word(self, k: int) -> bool:
        """Word *k* is a unit or a noun between a number and a range word: prose without a digit, or percent signs."""
        if not 0 <= k < self.n:
            return False
        w = self.words[k].rstrip(_TRAIL)
        return bool(w) and not any(c in DIGITS for c in w) and (G.prose(w) or all(c in PERCENTS for c in w))

    def unit_range_after(self, b: int) -> bool:
        """r7 round 3: after the number's word *b* a unit word, then a lone range word and a number, or a signed number
        ('20 mg - 40 mg', '20 mg -40 mg'); or a unit glued to a dash and digits that are no unit exponent ('20
        mg-40 mg', 'mg-400'): the number is the left end of a range."""
        if b + 2 < self.n:
            r = self.words[b + 2]
            if ((r in RANGE_WORDS and b + 3 < self.n and W.digit_led(self.words[b + 3])) or (
                    r[0] in HYPHEN and W.signed_digits(r))) and self.unit_word(b + 1):
                return True
        if b + 1 >= self.n or HYPHEN.isdisjoint(u := self.words[b + 1]) or not unit_led(u):
            return False
        m = ATOM_RE.match(u)
        rest = u[m.end():].rstrip(_TRAIL) if m is not None else ""
        if not (rest[:1] in HYPHEN and len(rest) > 1 and all(c in DIGITS for c in rest[1:])):
            return False
        return len(rest) - 1 > UNIT_EXPONENT_DIGITS or (b + 2 < self.n and unit_led(self.words[b + 2]))

    def unit_range_before(self, a: int, first: Tok) -> bool:
        """r7 round 3: a number led by a dash after a unit word that follows a number ('20 mg -40 mg'): a right end."""
        return self.t[first.start] in HYPHEN and a >= 2 and self.unit_word(a - 1) and self.number_end(a - 2)

    def continued(self, a: int, first: Tok) -> bool:
        """r7 round 2: a word led by a letter that ends in '-' or '/' ('AUC0-\\n24', 't1/\\n2', and 'docu-\\n2' alike:
        decision 3) may continue over the gap into the number that starts word *a*: a range end, never a clean pass."""
        prev = self.words[a - 1]
        return prev[-1] in _JOINER_END and first.start == self.starts[a] and lead_letter(prev)

    def qualified_before(self, a: int) -> bool:
        """A qualifier stands right before word *a*: up to three words back through operator words (``words.qualifies``:
        '< = 5'), or as the first word past the punctuation words the window skips (r7 round 2: '< , 12', '< ) 12',
        '<\n.\n12', 'LLOQ < ( ( ( 0.5' are qualified like '< 12'; the walk is the window's own)."""
        if a == 0 or self.t[self.ends[a - 1] - 1] not in _QUALIFY_ENDS:
            return False                                   # the word before ends in no operator or punctuation
        if W.qualifies([self.words[k] for k in range(max(0, a - 3), a)]):
            return True
        k = a - 1
        while k >= 0 and self.t[self.ends[k] - 1] in PUNCT and punct_only(self.words[k]):
            k -= 1
        return 0 <= k < a - 1 and qualifier_word(self.words[k])

    def flagged(self, toks: tuple[Tok, ...], a: int, b: int | None = None) -> tuple[Tok, ...]:
        """Tokens of the words a..b with the qualifier and range flags the neighbouring words add. A lone range word
        (``-`` ``~``) before or after marks every token; a dash glued to the word edge marks the one token it joins to
        the next word: the right end of ``12 -15`` and ``12- 15``, the left end of ``12 -15`` and ``5.0- 12.4``."""
        b = a if b is None else b
        prev = self.words[a - 1] if a else ""
        nxt = self.words[b + 1] if b + 1 < self.n else ""
        last = self.words[b]
        q = self.qualified_before(a)
        lone = prev in RANGE_WORDS or nxt in RANGE_WORDS
        left = bool(nxt) and ((nxt[0] in HYPHEN and W.signed_digits(nxt) and W.ends_with_number(last))
                              or (last[-1] in HYPHEN and W.digit_led(nxt)) or self.unit_range_after(b))
        right = bool(prev) and ((self.t[toks[0].start] in HYPHEN and self.number_end(a - 1))
                                or (prev[-1] in HYPHEN and self.number_end(a - 1, 1) and W.digit_led(self.words[a]))
                                or self.continued(a, toks[0]) or self.unit_range_before(a, toks[0]))
        if not (q or lone or left or right):
            return toks
        end = len(toks) - 1
        return tuple(t._replace(qualified=t.qualified or q, ranged=t.ranged or lone or (left and k == end)
                                or (right and k == 0)) for k, t in enumerate(toks))


__all__ = ["Flags"]
