"""``tokens(text, style, mode=...)``: every numeric expression of a text, resolved or not, with native offsets.

Pipeline: the text is folded 1:1 (equivalents.fold: no offset ever moves), split into WORDS (maximal runs of
non-gap characters), and each word that holds a digit is read by scan.scan_word. The two multi-word productions of
forms.py then join words one gap run apart (line breaks included): ``1.2 x 10^3`` -> one ``sci`` token and ``12 / 15``
-> one ``ratio`` token; a glued unit the next word continues is ``split_unit`` (``10m L``). Then the taint tiers
(taint.py, and the window walk of neighbours.py since r7 round 2) turn an expression with an outside character in its
word, its window or its line into ``unresolved_context``, and the neighbour-word allow-list (neighbours.py) turns a
resolved number whose window holds a word that is no accepted production (``2 x 10 mg``, ``12 3/4``, ``10 ^ 3``,
``1.2 10^3``, ``3.4 E-03``, ``5.2 ( x 10^3 )``, ``- ( 12 )``) into one unresolved expression through that word.
Decimal style decides only the ``A,BBB`` / ``A.BBB`` cases.
"""
from __future__ import annotations

import re
from bisect import bisect_left, bisect_right
from itertools import islice

from app.evidence.numgrammar import words as W
from app.evidence.numgrammar.alphabet import (
    CARET,
    CID_MARKER,
    COMMA,
    DIGITS,
    EXP_MARKERS,
    INSIDE_ONLY_RE,
    LETTERS,
    MULT_LETTERS,
    MULTIPLIERS,
    NUMBER_CLOSERS,
    POINT,
    RATIO_MARKS,
    SEPARATORS,
    SLASH,
    SQUARE_OPEN,
    SUP_CHARS,
    WORD_RE,
)
from app.evidence.numgrammar.body import MAX_DIGITS, to_float
from app.evidence.numgrammar.equivalents import fold
from app.evidence.numgrammar.flags import Flags
from app.evidence.numgrammar.forms import Forms
from app.evidence.numgrammar.neighbours import LEFT, NEIGHBOUR, Window
from app.evidence.numgrammar.productions import Cite, citations, mask_chi_square, mask_citations
from app.evidence.numgrammar.scan import TAIL_NONE, Scan, scan_word
from app.evidence.numgrammar.taint import Marks
from app.evidence.numgrammar.tok import MAX_PARTS, REASON_CONTEXT, REASON_SHAPE, Tok
from app.evidence.numgrammar.units import splits

MODES = ("page", "claim", "cell")
ABSORB_WORDS = 2          # words of INSIDE characters a ten after a multiplier swallows ('1.2 x 10 3', '10 ^ 3')
# a word led by one of these is a unit or a plain label: no multi-word form starts with it
_UNIT_LEADS = LETTERS - EXP_MARKERS - MULT_LETTERS
_SPECIAL_LEADS = EXP_MARKERS | MULTIPLIERS


# a word that ends in one of these continues into the next word ('12/ 15', '10^ 3'), one that starts with one of these
# continues the word before ('12 .5', '10 ^3', '12 /15', '12 x3'): the refused words and the number are one expression
_JOIN_ENDS = RATIO_MARKS | {CARET} | SUP_CHARS
_JOIN_STARTS = _JOIN_ENDS | SEPARATORS | MULTIPLIERS
MASK = "\ufffd"          # stands for a (cid:N) marker, character for character: offsets never move
_DIGIT_RUN = re.compile("[0-9]+")


def _mask(m) -> str:
    return MASK * len(m.group())


def _own_parts(tok: Tok) -> tuple[float, ...]:
    """What citing a piece of *tok* inside a wider expression may cite: its value and parts (a ratio: its numerator and
    denominator only, never its quotient)."""
    return tok.parts if tok.value is None or tok.shape == "ratio" else (tok.value, *tok.parts)


def _demote(tok: Tok, detail: str) -> Tok:
    """A resolved token (or unresolved expression) turned into ``unresolved_context``; its value stays a part."""
    parts = (tok.value,) if tok.value is not None else tok.parts
    return Tok(tok.start, tok.end, None, REASON_CONTEXT, parts, tok.qualified, tok.ranged, "", "", detail)


class Reader(Forms, Flags, Window):
    """The words of one folded text and the forms that read them. One instance per ``tokens`` call."""
    __slots__ = ("t", "style", "starts", "ends", "words", "n", "marks", "cache", "anchors", "target", "power_words")

    def __init__(self, folded: str, style: str, whole_line: bool, cid: dict[int, str]) -> None:
        self.t, self.style = folded, style
        self.starts: list[int] = []
        self.ends: list[int] = []
        self.words: list[str] = []
        for m in WORD_RE.finditer(folded):
            self.starts.append(m.start())
            self.ends.append(m.end())
            self.words.append(m.group())
        self.n = len(self.starts)
        self.marks = Marks(folded, self.starts, self.ends, whole_line, self.mult_word, cid)
        self.cache: dict[int, Scan] = {}
        self.anchors = self._anchors()
        self.target: tuple[Tok, ...] = ()           # the tokens the window is judging (neighbours.Window)
        self.power_words: set[int] = set()          # r7 round 3: the power words that scientific tokens consumed

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
        return self.words[i]

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

    def number_end(self, i: int, drop: int = 0) -> bool:
        """Word *i* (without its last *drop* characters) ends in a NUMBER: its last digit, closers aside, belongs to a
        token, or the word is numeric-led (a rejected one too: '1O5 3/4' stays a mixed number). The digits of a label
        (r7, owner decision 3: ``AUC0-24``, ``CYP2D6``, ``m12``) are not a number: they never make the next word a
        range end, the fraction of a mixed number or the second part of a spaced ratio."""
        w = self.text(i)[:len(self.text(i)) - drop]
        if not W.ends_with_number(w):
            return False
        if not W.letter_led(w):
            return True
        sc = self.scan(i)
        k = len(w)
        while w[k - 1] in NUMBER_CLOSERS:
            k -= 1
        return sc.kind == "numeric" and sc.toks[-1].end == self.starts[i] + k

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
            sp = W.special(self.text(i))
            if not self.has_digit(i) or not self.clean(i) or (sp and sp != W.FRACTION):
                continue
            sc = self.scan(i)
            if sc.kind == "numeric":                  # a ratio cites its numerator and denominator, never its quotient
                for t in sc.toks[:MAX_PARTS]:
                    out.extend(t.parts if t.shape == "ratio" else (t.value,))
            else:
                out.extend(sc.parts)
            if len(out) >= MAX_PARTS:
                break
        return tuple(out[:MAX_PARTS])

    def ratios_of(self, a: int, b: int) -> tuple[str, ...]:
        """Canonical ``n/d`` of the valid ratios read inside words a..b: what an unresolved expression holding them
        lets a ratio claim cite as a partial number (a ratio key is a string, not one of ``parts``)."""
        out: list[str] = []
        for i in range(a, b + 1):
            if not self.has_digit(i) or not self.clean(i) or SLASH.isdisjoint(self.text(i)):
                continue
            sc = self.scan(i)
            if sc.kind == "numeric":
                out.extend(t.canonical for t in sc.toks if t.shape == "ratio")
            if len(out) >= MAX_PARTS:
                break
        return tuple(out[:MAX_PARTS])

    # --------------------------------------------------------------- emission ---
    def put(self, out: list[Tok], fw: int, lw: int, toks: tuple[Tok, ...]) -> int:
        """Emit the resolved (and flagged) tokens of words fw..lw; returns the last word consumed. A tainting character
        in the words or their window (the words the allow-list judges: r7 round 2) demotes each token (a multiplier
        right after the number takes its operand into the same expression, as it does when the window refuses it); a
        window word that is no accepted production (neighbours.py) turns them into one unresolved expression through
        that word."""
        d = (self.marks.detail(fw, lw, toks[0].start, toks[-1].end) or self.window_taint(fw, lw)) if self.marks.bad \
            else ""
        if d:
            if lw + 1 < self.n and self.text(lw + 1)[:1] in MULTIPLIERS:
                final = self.consume(lw + 1, lw)
                parts = tuple(p for x in toks for p in _own_parts(x)) + self.parts_of(lw + 1, final)
                return self.expr(out, fw, final, toks[0].start, self.ends[final], NEIGHBOUR, parts[:MAX_PARTS])
            out.extend(_demote(x, d) for x in toks)
            return lw
        # a range dash ends the word right after its last number or that number's suffix ('5.0-', '45%-'); a dash after
        # letters is the hyphen of a word the line broke ('...complica-'), judged with the rest of the word's glue
        self.target = toks
        fault = self.window_fault(fw, lw, self.scan(lw).dash)
        self.target = ()
        if fault is None:
            out.extend(toks)
            return lw
        side, k, why = fault
        start, end, first, final = toks[0].start, toks[-1].end, fw, lw
        if side == LEFT:
            own = self.scan(k).kind in ("numeric", "rejected")
            if k < fw and not own and (not out or out[-1].end <= self.starts[k]):
                start, first = self.starts[k], k                       # a refused word with no expression of its own
            elif k == fw - 1 and out and out[-1].value is None and out[-1].end == self.ends[k] \
                    and self.t[self.ends[k] - 1] in _JOIN_ENDS:
                before = out.pop()                                     # '12/ 15', '10^ 3': the refused word's own
                start, first = before.start, bisect_right(self.starts, before.start) - 1   # expression and the number
                toks = (before, *toks)                                 # are one expression
        elif k > lw:
            final = self.consume(k, lw)
            end = self.ends[final] if final > lw else end
        parts = tuple(p for x in toks for p in _own_parts(x))
        parts += self.parts_of(lw + 1, final)
        return self.expr(out, first, final, start, end, why, parts[:MAX_PARTS])

    def consume(self, k: int, lw: int) -> int:
        """The last word of an expression refused at the word *k* after it: the word itself (unless it is a rejected
        word that no joining character ties to the number: '12 +3/4', '12 5th' keep their own expression), the operand
        behind a refused multiplier, and the INSIDE words a ten or a detached power absorbs ('1.2 x 10 3', '10 ^ 3')."""
        w = self.text(k)
        if W.is_mult_word(w) and k + 1 < self.n:
            return self.absorb(k + 1) if W.ten_shape(self.text(k + 1)) else k + 1
        if W.ten_shape(w) or (not self.has_digit(k) and W.CARET_OR_SUP_RE.search(w)):
            return self.absorb(k)
        if self.scan(k).kind == "rejected" and w[0] not in _JOIN_STARTS:
            return lw
        return k

    def expr(self, out: list[Tok], fw: int, lw: int, start: int, end: int, why: str, parts: tuple[float, ...]) -> int:
        d = (self.marks.detail(fw, lw, start, end) or self.window_taint(fw, lw)) if self.marks.bad else ""
        out.append(Tok(start, end, None, REASON_CONTEXT if d else f"{REASON_SHAPE}:{why}", parts, detail=d or why,
                       ratios=self.ratios_of(fw, lw)))
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
        if c in _SPECIAL_LEADS or (c in DIGITS and not SLASH.isdisjoint(w)):
            sp = W.special(w)
            if sp == W.E_WORD or sp == W.MULT_POWER:
                return a
            if sp == W.MULT_TEN:
                return self._ten(out, a, a, s0, consumed_from=a)
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
        """The two multi-word productions read first (a tainted neighbour can never complete one: its characters match
        neither), and the token they produce is tainted and judged as a whole; a plain word by its own window."""
        toks = sc.toks
        n, t, starts, ends = self.n, self.t, self.starts, self.ends
        nxt = t[starts[a + 1]:ends[a + 1]] if a + 1 < n else ""
        if sc.unit and nxt and splits(sc.unit, nxt):                       # '10m L': the next word continues the unit
            return self.expr(out, a, a + 1, toks[0].start, ends[a + 1], "split_unit", self.parts_of(a, a))
        if sc.tail or (nxt and nxt[0] not in _UNIT_LEADS):
            done = self.forms(a, sc, out, nxt)
            if done is not None:
                return done
        return self.put(out, a, a, self.flagged(toks, a))

    def with_citations(self, out: list[Tok], cites: list[Cite]) -> list[Tok]:
        """The citations (r7, decision 7) and bracket lists (r7 round 2) as unresolved expressions of their own, tainted
        like any expression, merged in order; one inside a wider unresolved expression (its word holds an outside
        character) is part of that."""
        starts = [t.start for t in out]
        extra: list[Tok] = []
        for c in cites:
            k = bisect_left(starts, c.end)
            if (k > 0 and out[k - 1].end > c.start) or (k < len(out) and out[k].start < c.end):
                continue
            wi, wj = bisect_right(self.starts, c.start) - 1, bisect_right(self.starts, c.end - 1) - 1
            d = self.marks.detail(wi, wj, c.start, c.end)
            shape = REASON_CONTEXT if d else f"{REASON_SHAPE}:{c.why}"
            extra.append(Tok(c.start, c.end, None, shape, c.refs[:MAX_PARTS], detail=d or c.why))
        return sorted(out + extra, key=lambda t: t.start) if extra else out

    def with_labels(self, out: list[Tok]) -> list[Tok]:
        """The digits of every LABEL (r7, owner decision 3: a word led by a letter, ``AUC0-24``, ``CYP2D6``, ``t1/2``,
        ``C24h``) as unresolved expressions ``unresolved_shape:label``: each stretch of a label word outside the
        expressions already read that holds a digit (parts = its digit runs). They are added AFTER the reading, so they
        never change another expression; a record that cites one fails with the label sentence."""
        starts = [t.start for t in out]
        extra: list[Tok] = []
        for wi in self.anchors:
            if not W.letter_led(self.text(wi)):
                continue
            lo, hi = self.starts[wi], self.ends[wi]
            k = max(bisect_right(starts, lo) - 1, 0)
            while k < len(out) and out[k].start < hi:          # the expressions that overlap the word, in order
                if out[k].end > lo:
                    self._label(extra, lo, min(out[k].start, hi))
                    lo = max(lo, out[k].end)
                k += 1
            self._label(extra, lo, hi)
        return sorted(out + extra, key=lambda t: t.start) if extra else out

    def _label(self, extra: list[Tok], lo: int, hi: int) -> None:
        """One label expression over t[lo:hi] when it holds a digit; parts = its first MAX_PARTS digit runs."""
        if lo < hi and W.DIGIT_RE.search(self.t, lo, hi) is not None:
            runs = islice((to_float(m.group()) for m in _DIGIT_RUN.finditer(self.t, lo, hi)
                           if m.end() - m.start() <= MAX_DIGITS), MAX_PARTS)
            extra.append(Tok(lo, hi, None, f"{REASON_SHAPE}:label", tuple(runs), detail="label"))


def tokens(text: str, style: str = ".", *, mode: str = "page", labels: bool = False) -> list[Tok]:
    """Every numeric expression of *text*, in order; ``value`` None marks an unresolved one.

    ``mode="page"`` is for page text and column strings (an ASCII space never groups digits), ``"cell"`` for a
    table cell (page reading, the whole string is one line), ``"claim"`` for a snippet or a proposed value (a
    thousands-position ASCII space reads as a thin space, the whole string is one line). ``labels=True`` (the
    verifier) also returns the digits of label words as ``unresolved_shape:label`` expressions (r7, decision 3)."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if not text or W.DIGIT_RE.search(text) is None:
        return []
    folded = mask_chi_square(fold(text, claim=mode == "claim"))     # r7: chi of the chi-square form is a label
    cid: dict[int, str] = {}
    if "(cid:" in folded:                      # the pdfminer marker is ONE outside character: mask it 1:1
        cid = {m.start(): m.group() for m in CID_MARKER.finditer(folded)}
        folded = CID_MARKER.sub(_mask, folded)
    cites = citations(folded) if SQUARE_OPEN in folded else []    # r7: an unmistakable citation is masked, then emitted
    reader = Reader(mask_citations(folded, cites), COMMA if style == COMMA else POINT, mode != "page", cid)
    out = reader.with_citations(reader.run(), cites) if cites else reader.run()
    return reader.with_labels(out) if labels else out


__all__ = ["MODES", "Reader", "TAIL_NONE", "tokens"]
