"""``tokens(text, style, mode=...)``: every numeric expression of a text, resolved or not, with native offsets.

Pipeline: the text is folded 1:1 (equivalents.fold: no offset ever moves), split into WORDS (maximal runs of
non-gap characters), and each word that holds a digit is read by scan.scan_word. The multi-word forms then join
the words one gap run apart (any length and mix of space, tab, CR, LF: line breaks included):

ACCEPT   ``1.2 x 10^3``  ``1.2 ×10⁻³``  ``1.2× 10^3``  ``1.2 ×\\n10^3``  -> one ``sci`` token
REJECT   ``2 x 10 mg`` ``1.2 x 10 3`` ``× 103``          ten_after_multiplier
         ``2 × daily`` ``1.2 ×`` ``1.2 × l0³``            multiplier_operand
         ``10 ^ 3`` ``10 ³`` ``10^ 3``                    detached_exponent
         ``3.4 E-03`` ``2012 e1553``                      spaced_exponent
         ``12 3/4``                                       mixed_number (a lone ``3/4`` is ``fraction``)

then the taint tiers (taint.py) turn an expression with an outside character in its word, its neighbouring words
or its line into ``unresolved_context``. Decimal style decides only the ``A,BBB`` / ``A.BBB`` cases.
"""
from __future__ import annotations

from bisect import bisect_right

from app.evidence.numgrammar import words as W
from app.evidence.numgrammar.alphabet import (
    CID_MARKER,
    DIGITS,
    INSIDE_ONLY_RE,
    LETTERS,
    MULTIPLIERS,
    QUALIFIERS,
    WORD_RE,
)
from app.evidence.numgrammar.body import to_float
from app.evidence.numgrammar.equivalents import fold
from app.evidence.numgrammar.scan import (
    TAIL_MULT,
    TAIL_NONE,
    TAIL_STAR,
    Scan,
    read_power,
    scan_word,
)
from app.evidence.numgrammar.taint import Marks
from app.evidence.numgrammar.tok import MAX_PARTS, REASON_CONTEXT, REASON_SHAPE, Tok

MODES = ("page", "claim", "cell")
_DOUBLE_QUAL = frozenset({"<=", ">=", "=<", "=>"})
ABSORB_WORDS = 2          # words of INSIDE characters a ten after a multiplier swallows ('1.2 x 10 3', '10 ^ 3')
# a word led by one of these is a unit or a plain label: no multi-word form starts with it
_UNIT_LEADS = frozenset(LETTERS - {"e", "E", "x", "X"})
_SPECIAL_LEADS = frozenset("eE") | MULTIPLIERS


MASK = "\ufffd"          # stands for a (cid:N) marker, character for character: offsets never move


def _mask(m) -> str:
    return MASK * len(m.group())


def _demote(tok: Tok, detail: str) -> Tok:
    """A resolved token (or unresolved expression) turned into ``unresolved_context``; its value stays a part."""
    parts = (tok.value,) if tok.value is not None else tok.parts
    return Tok(tok.start, tok.end, None, REASON_CONTEXT, parts, tok.qualified, tok.ranged, "", "", detail)


class Reader:
    """The words of one folded text and the forms that read them. One instance per ``tokens`` call."""
    __slots__ = ("t", "style", "starts", "ends", "n", "marks", "cache", "anchors")

    def __init__(self, folded: str, style: str, whole_line: bool, cid: dict[int, str]) -> None:
        self.t, self.style = folded, style
        self.starts: list[int] = []
        self.ends: list[int] = []
        for m in WORD_RE.finditer(folded):
            self.starts.append(m.start())
            self.ends.append(m.end())
        self.n = len(self.starts)
        self.marks = Marks(folded, self.starts, self.ends, whole_line, self.mult_word, cid)
        self.cache: dict[int, Scan] = {}
        self.anchors = self._anchors()

    # ------------------------------------------------------------------ words ---
    def _anchors(self) -> list[int]:
        """Indexes of the words that hold an ASCII digit, in order."""
        out: list[int] = []
        pos, ends, find, t = 0, self.ends, W.DIGIT_RE.search, self.t
        while (m := find(t, pos)) is not None:
            wi = bisect_right(ends, m.start())
            out.append(wi)
            pos = ends[wi]
        return out

    def text(self, i: int) -> str:
        return self.t[self.starts[i]:self.ends[i]]

    def mult_word(self, i: int) -> bool:
        return W.is_mult_word(self.text(i))

    def has_digit(self, i: int) -> bool:
        return W.DIGIT_RE.search(self.t, self.starts[i], self.ends[i]) is not None

    def scan(self, i: int) -> Scan:
        """The reading of word *i* (a word without a digit holds no tokens)."""
        sc = self.cache.get(i)
        if sc is None:
            sc = scan_word(self.text(i), self.starts[i], self.style) if self.has_digit(i) else Scan("label")
            self.cache[i] = sc
        return sc

    def clean(self, i: int) -> bool:
        """No tainting character in word *i*."""
        return not self.marks.bad or self.marks.first_bad(self.starts[i], self.ends[i]) < 0

    def numeric_word(self, i: int) -> bool:
        """Word *i* is clean, NUMERIC-LED (a label word never is: ``2 x n=12``, ``2 x <a>10^3``) and holds
        resolved numbers: an ordinary numeric word."""
        if i >= self.n or not self.clean(i) or not self.has_digit(i):
            return False
        w = self.text(i)
        return not W.letter_led(w) and not W.special(w) and self.scan(i).kind == "numeric"

    def absorb(self, j: int) -> int:
        """Last word index after swallowing up to ABSORB_WORDS following words made only of INSIDE characters."""
        k = j
        while k - j < ABSORB_WORDS and k + 1 < self.n and INSIDE_ONLY_RE.fullmatch(self.text(k + 1)):
            k += 1
        return k

    def parts_of(self, a: int, b: int) -> tuple[float, ...]:
        """Values of the resolved numbers inside words a..b (what a citation of one of them is a part of)."""
        out: list[float] = []
        for i in range(a, b + 1):
            if not self.has_digit(i) or not self.clean(i) or W.special(self.text(i)):
                continue
            sc = self.scan(i)
            out.extend(t.value for t in sc.toks[:MAX_PARTS]) if sc.kind == "numeric" else out.extend(sc.parts)
            if len(out) >= MAX_PARTS:
                break
        return tuple(out[:MAX_PARTS])

    # --------------------------------------------------------------- emission ---
    def put(self, out: list[Tok], fw: int, lw: int, tok: Tok) -> None:
        d = self.marks.detail(fw, lw, tok.start, tok.end)
        out.append(_demote(tok, d) if d else tok)

    def expr(self, out: list[Tok], fw: int, lw: int, start: int, end: int, why: str, parts: tuple[float, ...]) -> int:
        d = self.marks.detail(fw, lw, start, end)
        out.append(Tok(start, end, None, REASON_CONTEXT if d else f"{REASON_SHAPE}:{why}", parts, detail=d or why))
        return lw

    def _word_expr(self, out: list[Tok], a: int, why: str, parts: tuple[float, ...] = ()) -> int:
        return self.expr(out, a, a, self.starts[a], self.ends[a], why, parts)

    # ------------------------------------------------------------------- main ---
    def run(self) -> list[Tok]:
        out: list[Tok] = []
        consumed = -1
        for a in self.anchors:
            if a > consumed:
                consumed = self.process(a, consumed, out)
        return out

    def process(self, a: int, consumed: int, out: list[Tok]) -> int:
        s0, e0 = self.starts[a], self.ends[a]
        w = self.t[s0:e0]
        marks = self.marks
        if marks.bad and marks.first_bad(s0, e0) >= 0:                     # an outside character in the word itself
            out.append(Tok(s0, e0, None, REASON_CONTEXT, (), detail=marks.detail(a, a, s0, e0)))
            return a
        c = w[0]
        if c in _SPECIAL_LEADS or (c in DIGITS and "/" in w):
            sp = W.special(w)
            if sp == W.E_WORD or sp == W.MULT_POWER:
                return a
            if sp == W.FRACTION:
                return self._word_expr(out, a, "fraction")
            if sp == W.MULT_TEN:
                return self._ten(out, a, a, s0, consumed_from=a)
        if a - 1 > consumed and a >= 1 and W.ten_shape(w) and self.mult_word(a - 1):   # '× 103', no mantissa
            return self._ten(out, a - 1, a, self.starts[a - 1], consumed_from=a)
        sc = self.cache.get(a)
        if sc is None:
            sc = self.cache[a] = scan_word(w, s0, self.style)
        if sc.kind == "rejected":
            if sc.why == "ten_after_multiplier":
                return self._ten(out, a, a, s0, consumed_from=a)
            return self._word_expr(out, a, sc.why, sc.parts)
        if sc.kind != "numeric":
            return a
        return self.numeric(a, sc, consumed, out)

    def _ten(self, out: list[Tok], fw: int, ten_word: int, start: int, *, consumed_from: int) -> int:
        """ten_after_multiplier over the words fw..(ten word + the INSIDE words it absorbs)."""
        last = self.absorb(ten_word)
        parts = self.parts_of(consumed_from, last)
        return self.expr(out, fw, last, start, self.ends[last], "ten_after_multiplier", parts)

    def numeric(self, a: int, sc: Scan, consumed: int, out: list[Tok]) -> int:
        """The forms read first (a tainted neighbour can never complete one: its characters match no form), and the
        expression they produce is tainted as a whole; a plain word is tainted by its own window."""
        toks = sc.toks
        n, t, starts, ends = self.n, self.t, self.starts, self.ends
        nxt = t[starts[a + 1]:ends[a + 1]] if a + 1 < n else ""
        prev = t[starts[a - 1]:ends[a - 1]] if a else ""
        if sc.tail or (nxt and nxt[0] not in _UNIT_LEADS) or (prev and prev[0] not in LETTERS):
            done = self.forms(a, sc, consumed, out, nxt, prev)
            if done is not None:
                return done
        if self.marks.bad:
            d = self.marks.detail(a, a, toks[0].start, toks[-1].end)
            if d:
                out.extend(_demote(x, d) for x in toks)
                return a
        out.extend(self.flagged(toks, prev, nxt))
        return a

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
            else:
                done = None
            if done is not None:
                return done
        return self.form_detached(a, sc, consumed, out, nxt, prev)

    @staticmethod
    def flagged(toks: tuple[Tok, ...], prev: str, nxt: str) -> tuple[Tok, ...]:
        """Tokens of a word with the qualifier / range flags the neighbouring words add."""
        q = bool(prev) and (prev[-1] in QUALIFIERS or prev[-2:] in _DOUBLE_QUAL)
        r = prev == "-" or nxt == "-"
        if not (q or r):
            return toks
        return tuple(t._replace(qualified=t.qualified or q, ranged=t.ranged or r) for t in toks)

    # ------------------------------------------------------------------ forms ---
    def joinable(self, a: int, sc: Scan) -> bool:
        """One plain token that ends the word (before a glued multiplier): a mantissa a power can join."""
        t = sc.toks[0]
        tail = 1 if sc.tail == TAIL_MULT else 0
        return len(sc.toks) == 1 and t.shape not in ("sci", "bare_power") and t.end == self.ends[a] - tail

    def sci(self, out: list[Tok], a: int, sc: Scan, last: int, power_text: str, at: int) -> int:
        """One scientific token from the mantissa of word *a* and the power at ``power_text[at:]`` (word *last*)."""
        t = sc.toks[0]
        pw = read_power(power_text, at, len(power_text))
        value, canon = None, ""
        if pw is not None and pw[0] is not None:
            canon = f"{t.canonical}e{pw[0]}"
            value = to_float(canon)
        if value is None:
            return self.expr(out, a, last, t.start, self.ends[last], "exponent_range", (t.value,))
        prev = self.text(a - 1) if a else ""
        nxt = self.text(last + 1) if last + 1 < self.n else ""
        tok = Tok(t.start, self.ends[last], value, "sci", (t.value,), t.qualified, t.ranged, canon, t.decimal)
        self.put(out, a, last, self.flagged((tok,), prev, nxt)[0])
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
        return None if rest is not None and rest.kind == "numeric" else self.operand(out, a, sc, j)

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


def tokens(text: str, style: str = ".", *, mode: str = "page") -> list[Tok]:
    """Every numeric expression of *text*, in order; ``value`` None marks an unresolved one.

    ``mode="page"`` is for page text and column strings (an ASCII space never groups digits), ``"cell"`` for a
    table cell (page reading, the whole string is one line), ``"claim"`` for a snippet or a proposed value (a
    thousands-position ASCII space reads as a thin space, the whole string is one line)."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if not text or W.DIGIT_RE.search(text) is None:
        return []
    folded = fold(text, claim=mode == "claim")
    cid: dict[int, str] = {}
    if "(cid:" in folded:                      # the pdfminer marker is ONE outside character: mask it 1:1
        cid = {m.start(): m.group() for m in CID_MARKER.finditer(folded)}
        folded = CID_MARKER.sub(_mask, folded)
    return Reader(folded, "," if style == "," else ".", mode != "page", cid).run()


__all__ = ["MODES", "Reader", "TAIL_NONE", "tokens"]
