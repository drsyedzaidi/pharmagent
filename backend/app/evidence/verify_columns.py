"""Two-column pages: does a page-text span join the two columns? (``verify`` rejects it as ``crosses_columns``.)

A page text of a two-column paper holds the merged lines (one line of the left column followed by one line of the right
column on the same baseline). ``ColumnLines`` normalises the lines of both columns ONCE per verification, so testing
many occurrences of a snippet never reads the columns again.

r7 round 2: a number on a COLUMN line (a merged line, or a line that is one column's own line) is read in its column,
never in the page text: the page text puts the other column's words around it (the line after '... was 2.5' starts
with the other column, so a scale 'x 10^9' on the column's next line, or a qualifier 'less than' at the end of the
column's previous line, is invisible there). ``column_line`` says when a page-text hit is such a number; ``verify``
then reads it in the columns only. A line that spans both columns (a title) is the page text's own.

r7 round 3 (the round-3 audit): every page-text LINE is read once per verification (its bounds from one sorted index
of the line breaks, its normalised text and its split memoised by line start), so a long merged line holding the snippet
thousands of times costs one read, not one per occurrence; and the split of a merged line no longer depends on the
order of a set: the left lines are tried in a fixed order, and a line with more than one valid split is AMBIGUOUS, so
every span on it crosses the columns (the page-text hit is refused and the columns are read instead).
"""
from __future__ import annotations

import re
from bisect import bisect_right

from app.evidence import gaps
from app.evidence.normalize import COLLAPSED, normalise
from app.evidence.numgrammar.alphabet import LF

AMBIGUOUS = (-1, -1)                                    # a merged line with more than one valid split
_LF_RE = re.compile(re.escape(LF))


class ColumnLines:
    """The normalised lines of a two-column page's columns, and the test for a span that joins them."""
    __slots__ = ("left", "right", "_order", "_breaks", "_lines")

    def __init__(self, columns: list[str]) -> None:
        self.left = {normalise(x)[0] for x in columns[0].split(LF) if not gaps.blank(x)}
        self.right = {normalise(x)[0] for x in columns[1].split(LF) if not gaps.blank(x)}
        self._order = sorted(self.left, key=lambda lp: (len(lp), lp))     # a fixed order, never a set's
        self._breaks: dict[str, list[int]] = {}
        self._lines: dict[tuple[str, int], tuple[int, str, tuple[int, int] | None]] = {}

    def cut(self, line: str) -> tuple[int, int] | None:
        """(raw end of the left part, raw start of the right part) when *line* is a merged line: one column's line
        followed by the other column's line on the same baseline; ``AMBIGUOUS`` when two splits are valid; else None."""
        nl, m = normalise(line)
        found = [lp for lp in self._order
                 if len(lp) < len(nl) and nl.startswith(lp + COLLAPSED) and nl[len(lp) + 1:] in self.right]
        if not found:
            return None
        return AMBIGUOUS if len(found) > 1 else (m[len(found[0]) - 1] + 1, m[len(found[0]) + 1])

    def _line(self, text: str, pos: int) -> tuple[int, int, str, tuple[int, int] | None]:
        """(start, end, normalised text, cut) of the page-text line holding *pos*, read once per line."""
        breaks = self._breaks.get(text)
        if breaks is None:
            breaks = self._breaks[text] = [m.start() for m in _LF_RE.finditer(text)]
        k = bisect_right(breaks, pos - 1)
        ls = breaks[k - 1] + 1 if k else 0
        if (text, ls) not in self._lines:
            le = breaks[k] if k < len(breaks) else len(text)
            self._lines[(text, ls)] = (le, normalise(text[ls:le])[0], self.cut(text[ls:le]))
        le, nl, cut = self._lines[(text, ls)]
        return ls, le, nl, cut

    def column_line(self, pos: int, text: str) -> bool:
        """True when the page-text line holding *pos* is a merged line or one column's own line."""
        _, _, nl, cut = self._line(text, pos)
        return nl in self.left or nl in self.right or cut is not None

    def crosses(self, span: list[int], text: str) -> bool:
        """True when a page-text span joins the two columns: it covers the boundary of a merged line, runs over a line
        break while touching a merged line (two-column lines interleave the columns), or lies on an ambiguous line."""
        cuts: list[tuple[int, int] | None] = []
        pos = span[0]
        while True:
            ls, le, _, cut = self._line(text, pos)
            cuts.append(None if cut is None else cut if cut == AMBIGUOUS else (ls + cut[0], ls + cut[1]))
            if le + 1 >= span[1] or le >= len(text):
                break
            pos = le + 1
        if any(c == AMBIGUOUS for c in cuts):
            return True
        if len(cuts) > 1:
            return any(c is not None for c in cuts)
        return cuts[0] is not None and span[0] < cuts[0][0] and span[1] > cuts[0][1]


__all__ = ["AMBIGUOUS", "ColumnLines"]
