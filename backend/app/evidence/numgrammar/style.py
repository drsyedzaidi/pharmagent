"""The decimal-style evidence of a text, counted from the grammar (never from a pattern over raw characters).

A document prints its decimals with a point or with a comma. The evidence is the resolved numbers whose reading uses
each mark: ``tokens(text, '.')`` and its shapes (``decimal``, ``style_dependent`` and ``grouped_decimal`` with a
point; ``foreign_decimal`` with a comma; the mantissa of a ``sci`` token). A character the reader cannot read (a
non-ASCII digit, a look-alike dot, a line with a character outside the alphabet) leaves its numbers unresolved, so it
can never vote; a reference pair ``[1,2]`` is an unresolved expression and votes for nothing either. ``A.BBB`` read
as thousands (``3,412``) and every grouped number is no evidence.
"""
from __future__ import annotations

from app.evidence.numgrammar.alphabet import COMMA, POINT
from app.evidence.numgrammar.tokenizer import tokens


def style_evidence(text: str) -> tuple[int, int]:
    """(decimal-comma, decimal-point) evidence counts in *text* (page reading: an ASCII space never groups)."""
    comma = point = 0
    for tok in tokens(text, POINT, mode="page"):
        if tok.value is None:
            continue
        if tok.decimal == POINT:
            point += 1
        elif tok.decimal == COMMA:
            comma += 1
    return comma, point
