"""Briefing-memo building blocks: document blocks and the traced-value recorder.

Every number the memo prints goes through ``Cited`` (a recorder view bound to
the audit entry that produced the value). It returns the rendered text with a
trace tag (``8.65 [#5]``) AND records a ``TracedValue`` so the memo can later be
checked against the table (app.compute.memo). A value that is missing or not a
finite number renders as ``-`` and is never recorded.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from app.compute.levels import level_percent
from app.compute.memo import TracedValue

MISSING = "-"
_HAS_DIGIT = re.compile(r"\d")


# ── document blocks (immutable) ────────────────────────────────────────────────
@dataclass(frozen=True)
class Heading:
    text: str
    level: int = 1


@dataclass(frozen=True)
class Para:
    text: str


@dataclass(frozen=True)
class Bullet:
    text: str


@dataclass(frozen=True)
class Table:
    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


Block = Heading | Para | Bullet | Table


@dataclass(frozen=True)
class Section:
    """One memo section: its blocks and a one-line headline for the summary."""
    key: str
    title: str
    blocks: tuple[Block, ...]
    headline: str = ""


def block_lines(block: Block) -> list[str]:
    """Plain-text lines of a block; identical to what a DOCX re-read yields."""
    if isinstance(block, Table):
        return [" | ".join(block.headers), *(" | ".join(r) for r in block.rows)]
    return [block.text]


def blocks_text(blocks: tuple[Block, ...] | list[Block]) -> str:
    return "\n".join(line for b in blocks for line in block_lines(b))


# ── recorder ───────────────────────────────────────────────────────────────────
def is_number(value: Any) -> bool:
    """A finite int/float (bool and out-of-range ints such as 10**400 are not)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def trace_tag(src: int | None) -> str:
    return f" [#{src}]" if src is not None else " [#?]"


def _clean(text: str) -> str:
    """'-0.00' -> '0.00': a rounded zero carries no sign."""
    return text[1:] if text.startswith("-") and float(text) == 0.0 else text


class Recorder:
    """Append-only collector of the values a memo prints."""

    def __init__(self) -> None:
        self._values: list[TracedValue] = []

    @property
    def values(self) -> tuple[TracedValue, ...]:
        return tuple(self._values)

    def at(self, src: int | None) -> Cited:
        return Cited(self, src)

    def add(self, value: TracedValue) -> None:
        self._values.append(value)


class Cited:
    """A recorder view bound to the audit entry (``src``) that produced the values."""

    def __init__(self, rec: Recorder, src: int | None, kind: str = "number") -> None:
        self._rec, self.src, self._kind = rec, src, kind

    def inputs(self) -> Cited:
        """The same view, recording values as kind ``input``: numbers a user (or the chat
        assistant on the user's behalf) SUPPLIED to a tool, reproduced as given and not
        computed by it. The memo must label them as such."""
        return Cited(self._rec, self.src, "input")

    def _number(self, key: str, text: str) -> str:
        self._rec.add(TracedValue(key, text, self.src, kind=self._kind))
        return text

    def num(self, key: str, value: Any, decimals: int = 2, unit: str = "") -> str:
        """Fixed-point number (+ optional unit suffix such as ``%``) plus tag, or ``-``
        when the value is not a finite number."""
        if not is_number(value):
            return MISSING
        return self._number(key, _clean(f"{float(value):.{decimals}f}")) + unit + trace_tag(self.src)

    def level(self, key: str, level: Any) -> str:
        """A confidence level as a percentage at its own precision (0.975 -> ``97.5%``)."""
        if not is_number(level):
            return MISSING
        return self._number(key, level_percent(level)) + "%" + trace_tag(self.src)

    def sig(self, key: str, value: Any, digits: int = 3) -> str:
        """Significant-digit number (p-values: may print as 1.97e-09) plus tag."""
        if not is_number(value):
            return MISSING
        return self._number(key, f"{float(value):.{digits}g}") + trace_tag(self.src)

    def count(self, key: str, value: Any, unit: str = "") -> str:
        """Whole-number count plus tag (a float count that is integral prints without decimals)."""
        if not is_number(value) or float(value) != int(value):
            return MISSING
        return self._number(key, str(int(value))) + unit + trace_tag(self.src)

    def span(self, key: str, lo: Any, hi: Any, decimals: int = 2, unit: str = "") -> str:
        """``lo to hi`` plus one tag; both ends recorded."""
        if not (is_number(lo) and is_number(hi)):
            return MISSING
        a = self._number(f"{key}.lower", _clean(f"{float(lo):.{decimals}f}"))
        b = self._number(f"{key}.upper", _clean(f"{float(hi):.{decimals}f}"))
        return f"{a} to {b}{unit}" + trace_tag(self.src)

    def frac(self, key: str, num: Any, den: Any) -> str:
        """``a/b`` (e.g. converged/total) plus one tag."""
        if not (is_number(num) and is_number(den)) or float(num) != int(num) or float(den) != int(den):
            return MISSING
        a, b = self._number(f"{key}.n", str(int(num))), self._number(f"{key}.of", str(int(den)))
        return f"{a}/{b}" + trace_tag(self.src)

    def lit(self, key: str, text: Any) -> str:
        """A verbatim state string; registered as a literal so digits inside it are
        accounted for, and tagged when it contains any."""
        s = " ".join(str(text).split()) if text is not None else ""
        if not s:
            return MISSING
        self._rec.add(TracedValue(key, s, self.src, kind="literal"))
        return s + (trace_tag(self.src) if _HAS_DIGIT.search(s) else "")
