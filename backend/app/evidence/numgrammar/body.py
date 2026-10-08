"""The body of a number: digit groups and the separators between them -> a canonical ASCII string and a shape.

``read_body(groups, seps, style)`` is pure: it receives the digit runs and the separator characters the
tokenizer found (``'12.5'`` -> ``['12', '5'], ['.']``) and returns ``(canonical, shape, decimal_mark)`` or the
name of the reason the body is unreadable. The canonical string is built here from ASCII digits only, so
``to_float`` (the only ``float()`` call of the package) never sees raw text.

Shapes (``Tok.shape``): ``int`` ``decimal`` ``foreign_decimal`` ``style_dependent`` ``grouped``
``grouped_decimal`` ``middot``. A shape other than ``int``/``decimal``/``grouped`` feeds the decimal-style flags.
"""
from __future__ import annotations

from app.evidence.numgrammar.alphabet import DECIMAL_MARKS, GROUP_ONLY, MIDDOT

MAX_DIGITS = 40          # digits of one number (a 200-digit run is not a measurement)
MAX_EXPONENT = 300       # |e| of a power of ten


def to_float(canonical: str) -> float | None:
    """float() of a canonical ASCII string the reader built; None when it overflows (``9e999``)."""
    value = float(canonical)
    return None if value in (float("inf"), float("-inf")) else value


def exp_int(digits: str, negative: bool) -> int | None:
    """The exponent of a 1-4 digit ASCII run, None when it has more digits or |e| > MAX_EXPONENT."""
    if len(digits) > 4:
        return None
    e = int(digits)
    return None if e > MAX_EXPONENT else (-e if negative else e)


def valid_groups(groups: list[str], seps: list[str]) -> bool:
    """One separator kind, a head of 1-3 digits without a leading zero, every other group exactly 3 digits."""
    return (len(set(seps)) == 1 and 1 <= len(groups[0]) <= 3 and groups[0][0] != "0"
            and all(len(g) == 3 for g in groups[1:]))


def _single(groups: list[str], sep: str, style: str) -> tuple[str, str, str] | str:
    """One separator kind used once: ``A S B``."""
    head, tail = groups
    if len(tail) != 3 or head == "0":                       # '0,412', '18,2', '1234,5': a decimal
        return head + "." + tail, ("decimal" if sep == style else "foreign_decimal"), sep
    if len(head) <= 3:                                      # '3,412': thousands or decimal, by the page style
        return (head + "." + tail, "style_dependent", sep) if sep == style else (head + tail, "style_dependent", "")
    return (head + "." + tail, "decimal", sep) if sep == style else "ambiguous_tail"   # '21741,229'


def _middot(groups: list[str], seps: list[str], style: str) -> tuple[str, str, str] | str:
    """A raised dot between digits is a decimal point on a point-decimal page only ('12·4' = 12.4)."""
    if style != "." or seps[-1] != MIDDOT or seps.count(MIDDOT) != 1:
        return "middot"
    if len(seps) > 1 and not valid_groups(groups[:-1], seps[:-1]):
        return "groups"
    return "".join(groups[:-1]) + "." + groups[-1], "middot", MIDDOT


def read_body(groups: list[str], seps: list[str], style: str) -> tuple[str, str, str] | str:
    """(canonical, shape, decimal_mark) for the digit groups joined by *seps*, else the reason it is unreadable
    (``too_many_digits`` ``leading_zero`` ``middot`` ``groups`` ``ambiguous_tail`` ``separators``)."""
    if sum(len(g) for g in groups) > MAX_DIGITS:
        return "too_many_digits"
    if len(groups[0]) > 1 and groups[0][0] == "0":
        return "leading_zero"                                # '007', '05', '00.5'
    if not seps:
        return groups[0], "int", ""
    kinds = set(seps)
    if MIDDOT in kinds:
        return _middot(groups, seps, style)
    if len(kinds) == 1:
        sep = seps[0]
        if sep in GROUP_ONLY or len(seps) > 1:              # '24'560', '1.250.000', '18 900'
            return ("".join(groups), "grouped", "") if valid_groups(groups, seps) else "groups"
        return _single(groups, sep, style)
    last = seps[-1]
    if len(kinds) == 2 and last in DECIMAL_MARKS and seps.count(last) == 1 and valid_groups(groups[:-1], seps[:-1]):
        # '1,234,567.89', '1.234.567,89', '1'234.5': thousands groups, then one decimal separator
        return ("".join(groups[:-1]) + "." + groups[-1],
                "grouped_decimal" if last == style else "foreign_decimal", last)
    return "groups" if len(kinds) == 2 and last in GROUP_ONLY else "separators"      # '0.123 4' groups, '1.2,3' not


def mantissa(groups: list[str], seps: list[str], style: str) -> tuple[str, str, str] | str | None:
    """The body of a number an exponent follows (``1.234e+03``, ``1.234 x 10^3``): ONE separator is always a decimal
    point, whatever the page's style. A thousands group never carries an exponent, so ``1.234e+03`` on a decimal-comma
    page is 1.234e3 (a foreign decimal: the flags ask for a look), never 1 234 000. None when the body has not exactly
    one ``.`` or ``,``; else (canonical, shape, mark) or the reason it is unreadable."""
    if len(seps) != 1 or seps[0] not in DECIMAL_MARKS:
        return None
    head, tail = groups
    if len(head) + len(tail) > MAX_DIGITS:
        return "too_many_digits"
    if len(head) > 1 and head[0] == "0":
        return "leading_zero"
    return head + "." + tail, ("decimal" if seps[0] == style else "foreign_decimal"), seps[0]


def decimal_mantissa(literal: str) -> tuple[str, str]:
    """(canonical, mark) of a one-separator literal read as a decimal (``1,234`` -> ``1.234``): the mantissa of a power
    form whose number the page style read as thousands."""
    sign = "-" if literal[:1] == "-" else ""
    body = literal[len(sign):]
    mark = next(c for c in body if c in DECIMAL_MARKS)
    head, _, tail = body.partition(mark)
    return f"{sign}{head}.{tail}", mark
