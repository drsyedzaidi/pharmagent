"""Where a character outside the allowed alphabet touches an expression (the owner's one-separator rule).

Three tiers, all allow-list tests (alphabet.py):

1. WORD   a character that is not INSIDE or CONTEXT (so LINE or OUTSIDE, or a ``(cid:N)`` marker) in the
          expression's own word.
2. WINDOW the same character in the word before or after (one gap run of any length is ONE separator: padding
          never widens the window), or in the right operand behind a multiplier word.
3. LINE   an OUTSIDE character (or marker) anywhere on the expression's line: LINE-class characters are allowed
          there, nothing else is. A line is the text between line feeds; for a table cell, a snippet or a value
          it is the whole string.

``Marks.detail`` returns the fixed fragment the reviewer message is built from (``U+200B inside the number``,
``(cid:3) one word before``, ``U+2028 elsewhere on the same line``), or '' when nothing taints the expression.
"""
from __future__ import annotations

import re
from bisect import bisect_left, bisect_right
from collections.abc import Callable

from app.evidence.numgrammar.alphabet import LINE_TIER_RE, WORD_TIER_RE

_NEWLINE = re.compile("\n")
INSIDE_THE_NUMBER = "inside the number"
GLUED_BEFORE = "glued before the digits"
GLUED_AFTER = "glued after the digits"
ONE_BEFORE = "one word before"
ONE_AFTER = "one word after"
BEHIND_MULT = "two words after, behind a multiplier"
SAME_LINE = "elsewhere on the same line"


class Marks:
    """Positions of the characters that taint, built once per text (empty for clean text: no cost).

    A ``(cid:N)`` marker arrives masked: the tokenizer replaced each marker 1:1 by MASK, an OUTSIDE character, and
    passes ``cid`` (marker start -> marker text) so the reviewer fragment can name the marker."""
    __slots__ = ("text", "bad", "outside", "cid", "cid_starts", "whole_line", "_nl", "_lines", "starts", "ends",
                 "mult_word")

    def __init__(self, text: str, starts: list[int], ends: list[int], whole_line: bool,
                 mult_word: Callable[[int], bool], cid: dict[int, str]) -> None:
        self.text, self.starts, self.ends, self.whole_line, self.mult_word = text, starts, ends, whole_line, mult_word
        self.bad = [m.start() for m in WORD_TIER_RE.finditer(text)]
        self.outside = [m.start() for m in LINE_TIER_RE.finditer(text)]
        self.cid, self.cid_starts = cid, sorted(cid)
        self._nl: list[int] | None = None
        self._lines: dict[int, int] | None = None

    def first_bad(self, lo: int, hi: int) -> int:
        """The first tainting position in [lo, hi), -1 when none."""
        bad = self.bad
        k = bisect_left(bad, lo)
        return bad[k] if k < len(bad) and bad[k] < hi else -1

    def label(self, p: int) -> str:
        """``U+200B`` or, for a position inside a masked marker, the marker text ``(cid:3)``."""
        if self.cid:
            k = bisect_right(self.cid_starts, p) - 1
            if k >= 0 and p < self.cid_starts[k] + len(self.cid[self.cid_starts[k]]):
                return self.cid[self.cid_starts[k]]
        return f"U+{ord(self.text[p]):04X}"

    def _line_of(self, pos: int) -> int:
        if self.whole_line:
            return 0
        if self._nl is None:
            self._nl = [m.start() for m in _NEWLINE.finditer(self.text)]
        return bisect_left(self._nl, pos)

    def _bad_lines(self) -> dict[int, int]:
        """line number -> its first OUTSIDE position."""
        if self._lines is None:
            lines: dict[int, int] = {}
            for p in self.outside:
                lines.setdefault(self._line_of(p), p)
            self._lines = lines
        return self._lines

    def line_offender(self, fw: int, lw: int) -> int:
        """First OUTSIDE position on the line(s) of words fw..lw, -1 when none."""
        lines = self._bad_lines()
        for line in {self._line_of(self.starts[fw]), self._line_of(self.ends[lw] - 1)}:
            if line in lines:
                return lines[line]
        return -1

    def detail(self, fw: int, lw: int, a: int, b: int) -> str:
        """Reviewer fragment for an expression at [a, b) spanning words fw..lw, '' when nothing taints it."""
        if not self.bad:
            return ""
        starts, ends = self.starts, self.ends
        p = self.first_bad(starts[fw], ends[lw])
        if p >= 0:
            where = INSIDE_THE_NUMBER if a <= p < b else (GLUED_BEFORE if p < a else GLUED_AFTER)
            return f"{self.label(p)} {where}"
        if fw > 0 and (p := self.first_bad(starts[fw - 1], ends[fw - 1])) >= 0:
            return f"{self.label(p)} {ONE_BEFORE}"
        n = len(starts)
        if lw + 1 < n:
            p = self.first_bad(starts[lw + 1], ends[lw + 1])
            if p >= 0:
                return f"{self.label(p)} {ONE_AFTER}"
            if self.mult_word(lw + 1) and lw + 2 < n and (p := self.first_bad(starts[lw + 2], ends[lw + 2])) >= 0:
                return f"{self.label(p)} {BEHIND_MULT}"
        if self.outside and (p := self.line_offender(fw, lw)) >= 0:
            return f"{self.label(p)} {SAME_LINE}"
        return ""
