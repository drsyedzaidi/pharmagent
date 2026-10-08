"""``tokens(text, style, mode=...)``: every numeric expression of a text, resolved or not, with native offsets.

Pipeline: the text is folded 1:1 (equivalents.fold: no offset ever moves), split into WORDS (maximal runs of
non-gap characters), and each word that holds a digit is read by scan.scan_word. The multi-word forms then join
the words one gap run apart (any length and mix of space, tab, CR, LF: line breaks included):

ACCEPT   ``1.2 x 10^3``  ``1.2 ×10⁻³``  ``1.2× 10^3``  ``1.2 ×\\n10^3``  -> one ``sci`` token
REJECT   ``2 x 10 mg`` ``1.2 x 10 3`` ``× 103``          ten_after_multiplier
         ``2 × daily`` ``1.2 ×`` ``1.2 × l0³``            multiplier_operand
         ``10 ^ 3`` ``10 ³`` ``10^ 3``                    detached_exponent
         ``3.4 E-03`` ``2012 e1553``                      spaced_exponent
         ``12 3/4``                                       mixed_number (a lone ``3/4`` is a ratio, r7)
         ``12 /15`` ``12 .5`` ``5.2 (x10^3 cells)``       fraction, detached_decimal, bracketed_multiplier
ACCEPT   ``12 / 15`` (and the glued ``12/15`` of scan.py)  -> one ``ratio`` token (r7, owner decision 6)
         (the rules live in forms.py)

then the taint tiers (taint.py) turn an expression with an outside character in its word, its neighbouring words
or its line into ``unresolved_context``. Decimal style decides only the ``A,BBB`` / ``A.BBB`` cases.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right

from app.evidence.numgrammar import words as W
from app.evidence.numgrammar.alphabet import (
    CID_MARKER,
    DIGITS,
    INSIDE_ONLY_RE,
    LETTERS,
    MULTIPLIERS,
    RANGE_WORDS,
    WORD_RE,
)
from app.evidence.numgrammar.equivalents import fold
from app.evidence.numgrammar.forms import Forms
from app.evidence.numgrammar.productions import Cite, citations, mask_chi_square, mask_citations
from app.evidence.numgrammar.scan import TAIL_NONE, Scan, scan_word
from app.evidence.numgrammar.taint import Marks
from app.evidence.numgrammar.tok import MAX_PARTS, REASON_CONTEXT, REASON_SHAPE, Tok

MODES = ("page", "claim", "cell")
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


class Reader(Forms):
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
            if w.endswith("/") and sc.why == "glued_symbol" and a + 1 < self.n and W.digit_led(self.text(a + 1)):
                return self.ratio_run(out, a, a + 1, s0, sc.parts)              # '12/ 15'
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
        if toks[0].shape == "ratio" and toks[0].start == starts[a] and W.ends_with_number(prev):
            return self._word_expr(out, a, "mixed_number", toks[0].parts)   # '...12 3/4': a whole part may stand before
        if sc.tail or (nxt and nxt[0] not in _UNIT_LEADS) or (prev and prev[0] not in LETTERS):
            done = self.forms(a, sc, consumed, out, nxt, prev)
            if done is not None:
                return done
        if self.marks.bad:
            d = self.marks.detail(a, a, toks[0].start, toks[-1].end)
            if d:
                out.extend(_demote(x, d) for x in toks)
                return a
        out.extend(self.flagged(toks, a))
        return a


    def with_citations(self, out: list[Tok], cites: list[Cite]) -> list[Tok]:
        """The citations (r7, decision 7) as unresolved expressions of their own, tainted like any expression, merged
        in order; one inside a wider unresolved expression (its word holds an outside character) is part of that."""
        starts = [t.start for t in out]
        extra: list[Tok] = []
        for c in cites:
            k = bisect_left(starts, c.end)
            if (k > 0 and out[k - 1].end > c.start) or (k < len(out) and out[k].start < c.end):
                continue
            wi = bisect_right(self.starts, c.start) - 1
            d = self.marks.detail(wi, wi, c.start, c.end)
            extra.append(Tok(c.start, c.end, None, REASON_CONTEXT if d else f"{REASON_SHAPE}:citation", c.refs,
                             detail=d or "citation"))
        return sorted(out + extra, key=lambda t: t.start) if extra else out

    def qualified_before(self, a: int) -> bool:
        """A qualifier stands right before word *a* (up to three words back; see ``words.qualifies``)."""
        return W.qualifies([self.text(k) for k in range(max(0, a - 3), a)])

    def flagged(self, toks: tuple[Tok, ...], a: int, b: int | None = None) -> tuple[Tok, ...]:
        """Tokens of the words a..b with the qualifier and range flags the neighbouring words add. A lone range word
        (``-`` ``~``) before or after marks every token; a dash glued to the word edge marks the one token it joins to
        the next word: the right end of ``12 -15`` and ``12- 15``, the left end of ``12 -15`` and ``5.0- 12.4``."""
        b = a if b is None else b
        prev = self.text(a - 1) if a else ""
        nxt = self.text(b + 1) if b + 1 < self.n else ""
        last = self.text(b)
        q = self.qualified_before(a)
        lone = prev in RANGE_WORDS or nxt in RANGE_WORDS
        left = bool(nxt) and ((nxt[0] == "-" and W.signed_digits(nxt) and W.ends_with_number(last))
                              or (last.endswith("-") and W.digit_led(nxt)))
        right = bool(prev) and ((W.ends_with_number(prev) and self.t[toks[0].start] == "-")
                                or (prev.endswith("-") and W.ends_with_number(prev[:-1]) and W.digit_led(self.text(a))))
        if not (q or lone or left or right):
            return toks
        end = len(toks) - 1
        return tuple(t._replace(qualified=t.qualified or q, ranged=t.ranged or lone or (left and k == end)
                                or (right and k == 0)) for k, t in enumerate(toks))



def tokens(text: str, style: str = ".", *, mode: str = "page") -> list[Tok]:
    """Every numeric expression of *text*, in order; ``value`` None marks an unresolved one.

    ``mode="page"`` is for page text and column strings (an ASCII space never groups digits), ``"cell"`` for a
    table cell (page reading, the whole string is one line), ``"claim"`` for a snippet or a proposed value (a
    thousands-position ASCII space reads as a thin space, the whole string is one line)."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if not text or W.DIGIT_RE.search(text) is None:
        return []
    folded = mask_chi_square(fold(text, claim=mode == "claim"))     # r7: chi of the chi-square form is a label
    cid: dict[int, str] = {}
    if "(cid:" in folded:                      # the pdfminer marker is ONE outside character: mask it 1:1
        cid = {m.start(): m.group() for m in CID_MARKER.finditer(folded)}
        folded = CID_MARKER.sub(_mask, folded)
    cites = citations(folded) if "[" in folded else []     # r7: an unmistakable citation is masked, then emitted
    reader = Reader(mask_citations(folded, cites), "," if style == "," else ".", mode != "page", cid)
    return reader.with_citations(reader.run(), cites) if cites else reader.run()


__all__ = ["MODES", "Reader", "TAIL_NONE", "tokens"]
