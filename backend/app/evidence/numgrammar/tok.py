"""One numeric expression of a text: a resolved number or an unresolved expression, with NATIVE offsets."""
from __future__ import annotations

from decimal import Decimal
from typing import NamedTuple

REASON_CONTEXT = "unresolved_context"
REASON_SHAPE = "unresolved_shape"
MAX_PARTS = 64          # resolved pieces an unresolved expression remembers (a 500 000-number word keeps 64)


class Tok(NamedTuple):
    """``text[start:end]`` is the literal as printed (equivalents included: ``-20`` written with a minus sign, a
    thin-space group). No cleaned view and no index map exist: offsets index the string that was passed in.

    resolved   ``value`` is a float, ``shape`` one of int decimal foreign_decimal style_dependent grouped
               grouped_decimal middot sci bare_power, ``canonical`` the ASCII text the value was read from
               (``-1.2e3``), ``decimal`` the character used as the decimal point ('' when none).
    unresolved ``value`` is None and ``shape`` is ``unresolved_context`` (a character outside the allowed
               alphabet touches the number, sits one word away, or lies elsewhere on its line) or
               ``unresolved_shape:<why>`` (allowed characters in an order the grammar rejects). ``parts`` holds
               the values of the resolved numbers inside the expression (a citation of one of them is a partial
               number), ``detail`` a fixed sentence fragment for the reviewer message: ``U+200B inside the
               number``, ``(cid:3) one word before``, ``glued_letters``.
    """
    start: int
    end: int
    value: float | None
    shape: str
    parts: tuple[float, ...] = ()
    qualified: bool = False      # a qualifier stands before it through a run of operators (<, <=, ~=, <(, < =)
    ranged: bool = False         # a range dash joins it to another number (12-15, 12 -15, 5.0- 12.4, 12 ~ 15)
    canonical: str = ""
    decimal: str = ""
    detail: str = ""

    @property
    def reason(self) -> str:
        """``unresolved_context``, ``unresolved_shape`` or '' for a resolved token."""
        return "" if self.value is not None else self.shape.partition(":")[0]

    @property
    def offender(self) -> str:
        """The character (``U+200B``) or marker (``(cid:3)``) that made an ``unresolved_context`` expression, ''
        otherwise. The caller names it (``U+200B ZERO WIDTH SPACE``): this package never reads Unicode names."""
        return self.detail.partition(" ")[0] if self.reason == REASON_CONTEXT else ""

    @property
    def where(self) -> str:
        """Where the offender sits (``inside the number``, ``one word before``, ``elsewhere on the same line``...),
        '' for anything but an ``unresolved_context`` expression."""
        return self.detail.partition(" ")[2] if self.reason == REASON_CONTEXT else ""

    @property
    def why(self) -> str:
        """The shape reason of an unresolved expression (``ten_after_multiplier``), '' otherwise."""
        return self.shape.partition(":")[2] if self.value is None else ""

    @property
    def decimal_value(self) -> Decimal | None:
        """The exact decimal reading of the canonical string, None when unresolved."""
        return Decimal(self.canonical) if self.canonical else None
