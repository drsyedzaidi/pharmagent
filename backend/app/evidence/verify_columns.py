"""Two-column pages: does a page-text span join the two columns? (``verify`` rejects it as ``crosses_columns``.)

A page text of a two-column paper holds the merged lines (one line of the left column followed by one line of the right
column on the same baseline). ``ColumnLines`` normalises the lines of both columns ONCE per verification, so testing
many occurrences of a snippet never reads the columns again.
"""
from __future__ import annotations

from app.evidence.normalize import normalise


class ColumnLines:
    """The normalised lines of a two-column page's columns, and the test for a span that joins them."""
    __slots__ = ("left", "right")

    def __init__(self, columns: list[str]) -> None:
        self.left = {normalise(x)[0] for x in columns[0].split("\n") if x.strip()}
        self.right = {normalise(x)[0] for x in columns[1].split("\n") if x.strip()}

    def cut(self, line: str) -> tuple[int, int] | None:
        """(raw end of the left part, raw start of the right part) when *line* is a merged line: one column's line
        followed by the other column's line on the same baseline; else None."""
        nl, m = normalise(line)
        for lp in self.left:
            if len(lp) < len(nl) and nl.startswith(lp + " ") and nl[len(lp) + 1:] in self.right:
                return m[len(lp) - 1] + 1, m[len(lp) + 1]
        return None

    def crosses(self, span: list[int], text: str) -> bool:
        """True when a page-text span joins the two columns: it covers the boundary of a merged line, or runs over a
        line break while touching a merged line (two-column lines interleave the columns)."""
        ls, cuts = text.rfind("\n", 0, span[0]) + 1, []
        while ls < span[1]:
            le = text.find("\n", ls)
            le = len(text) if le < 0 else le
            cut = self.cut(text[ls:le])
            cuts.append(None if cut is None else (ls + cut[0], ls + cut[1]))
            ls = le + 1
        if len(cuts) > 1:
            return any(c is not None for c in cuts)
        return cuts[0] is not None and span[0] < cuts[0][0] and span[1] > cuts[0][1]
