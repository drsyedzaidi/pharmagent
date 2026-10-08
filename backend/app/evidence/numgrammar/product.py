"""The PRODUCT production (r7 round 3): when a multiplier between two numbers is a product, never a scale.

EBNF (explicit; the round-3 audit replaced a refusal of one 'ten' operand shape, which let every other power of ten
behind a multiplier pass as a clean value)::

    product = count MULT amount             spaced (a lone multiplier word) or glued (x X * x-sign; a middle dot glued
                                            between digits is the separator production, never a product)
    count   = plain                         an unsigned int or decimal ('2', '4.6', '0.0015')
    amount  = [sign] plain                  ('100', '250', '-3', '1.5'), with two conditions:
              not a raised ten              its integer digits are not '10' followed by a digit 1-9 or by nothing
                                            ('10', '103', '10.5': a ten whose exponent the text layer may have lost)
              a written-out power of ten    ('100', '1000', '0.1'; one scales nothing) only after an integer count of
                                            one or two digits AND before a unit production: '2 x 100 mg', '3 x 1000 mg'
    plain   = an int or a decimal token (a one-separator number the page style reads as a decimal included): never a
              thousands-grouped, scientific, power-of-ten or ratio token

Every other ``number MULT number`` is one unresolved expression (``1.2 x 10000``, ``2.5 x 1e6``, ``1.2 x 1,000,000``,
``3.1 . 1000``, ``2 x 10000 cells``, ``2 x 103``): its name for the reviewer is ``ten_after_multiplier`` when the
amount's digits start with a ten, else ``multiplier_operand`` (the name only describes; the production decides).
"""
from __future__ import annotations

import re

from app.evidence.numgrammar.alphabet import HYPHEN, POINT, ZERO
from app.evidence.numgrammar.body import to_float
from app.evidence.numgrammar.tok import Tok

PLAIN_SHAPES = frozenset({"int", "decimal", "foreign_decimal"})
_STYLE = "style_dependent"                                  # '1.018': plain only when the page style reads a decimal
TEN, OPERAND = "ten_after_multiplier", "multiplier_operand"
_COUNT = re.compile("[1-9][0-9]?")                          # an integer count of one or two digits
_RAISED_TEN = re.compile("10(?:[1-9][0-9]*)?")              # integer digits '10', '103', '1012'
_POWER_DIGITS = re.compile("10*")                           # 1, 10, 100 ... once the point and leading zeros go
_ONES = (1.0, -1.0)                                         # ten to the zero scales nothing


def _unsigned(canonical: str) -> str:
    return canonical[1:] if canonical[:1] in HYPHEN else canonical


def power_of_ten(canonical: str) -> bool:
    """The canonical number is a power of ten written out that scales ('10', '100', '1000', '0.1'; never one: '1', '1.0'
    scale nothing, so they read like any other digit)."""
    digits = _unsigned(canonical).replace(POINT, "").lstrip(ZERO)
    return _POWER_DIGITS.fullmatch(digits) is not None and to_float(canonical) not in _ONES


def name(amount_canonical: str) -> str:
    """The reviewer's name of a refused product: a ten when the amount's digits start with one."""
    return TEN if _unsigned(amount_canonical)[:2] == "10" else OPERAND


def plain(shape: str, mark: str) -> bool:
    """An int or a decimal: a ``style_dependent`` number is plain only when the page style reads it as a decimal."""
    return shape in PLAIN_SHAPES or (shape == _STYLE and mark != "")


def scale(amount: Tok) -> bool:
    """The amount scales what stands before it: a raised ten, a power of ten written out, a bare power or a
    scientific token whose mantissa is one. A FOOTNOTE star run that ends a number's word ('12.4*') is a suffix; one
    such number word before a scale may be a multiplier, so the pair is refused ('12* 10', '12* 1000')."""
    if amount.shape == "bare_power" or (amount.shape == "sci" and amount.parts[:1] == (1.0,)):
        return True
    return plain(amount.shape, amount.decimal) and (
        _RAISED_TEN.fullmatch(_unsigned(amount.canonical).partition(POINT)[0]) is not None
        or power_of_ten(amount.canonical))


def fault(count: Tok, amount: tuple[str, str, str], unit_after: bool) -> str:
    """'' when ``count MULT amount`` is the product production (*amount*: its shape, canonical and decimal mark;
    *unit_after*: a unit production follows the amount, or the caller checks the next word), else the reviewer's name
    of the refusal."""
    amount_shape, amount_canonical, amount_mark = amount
    why = name(amount_canonical)
    if not plain(count.shape, count.decimal) or not plain(amount_shape, amount_mark) or count.canonical[:1] in HYPHEN:
        return why
    if _RAISED_TEN.fullmatch(_unsigned(amount_canonical).partition(POINT)[0]):
        return why
    if power_of_ten(amount_canonical) and not (_COUNT.fullmatch(count.canonical) and unit_after):
        return why
    return ""


__all__ = ["OPERAND", "PLAIN_SHAPES", "TEN", "fault", "name", "plain", "power_of_ten", "scale"]
