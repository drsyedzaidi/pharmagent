"""The unit production (r7, owner decision 2): a unit glued to a number is read by grammar, never by a letter list.

EBNF (the terminals are ``alphabet.UNIT_PREFIXES``, ``UNIT_BASES`` and ``UNIT_PREFIXED_BASES``, grammar data of the
CONTEXT allow-list)::

    unit      = atom {("/" | "per") atom}                 parsed directly after the number's last character
    atom      = ([prefix] base | prefix prefixed_base) [exponent]       (prefixed_base: the 'l' of 'ml', never bare)
    exponent  = "2" | "3"                                  (a plain square or cube: '1.73m2', 'cm3'; not before a digit)
              | "^" ["-" | "+"] digit{1,2}                 ('m^2', 'h^-1')
              | [superscript sign] superscript digit{1,2}  ('m²', 'h⁻¹')
    boundary  = end of word | trailing closer not before a digit | opener

A glued unit is accepted only when the whole production ends at a boundary: '150mg', '24h', '10mL', '(20mg)', '1.73m2',
'2.5mg/kg/h', '0.6ng/ml,'. Everything else stays unresolved ``glued_letters``: a letter that is no terminal ('1OmL',
'5th', '1970s', '150mgtid', '23l'), a unit that runs into a sign, a digit or another symbol ('5h-1', '9.9mg*',
'5mg,10mg'), a signed plain exponent (it would be a range: '5h-1' is refused, '5h^-1' and '5h⁻¹' are read). The
unit is context: it never changes the number, which is the token's literal as before ('150' of '150mg').

ODD SPACING. A glued unit that the next word continues ('10m L', '81ng/m L,', '15m in', '10mg /kg') is a text layer that
broke the unit: the number and the next word are one unresolved expression ``split_unit`` (``splits``).
"""
from __future__ import annotations

import re

from app.evidence.numgrammar.alphabet import (
    CARET,
    OPENERS_BR,
    PER,
    PLAIN_POWERS,
    PRODUCT_DOTS,
    SIGNS,
    SLASH,
    SUP_CHARS,
    SUP_DIGITS,
    SUP_SIGNS,
    TRAILING,
    UNIT_BASES,
    UNIT_PREFIXED_BASES,
    UNIT_PREFIXES,
    char_class,
)

SPLIT_LOOK = 64            # characters of the next word a split unit is looked for in (a unit is never longer)


def _alt(terms: frozenset[str]) -> str:
    """One alternation of literal terminals, longest first."""
    return "(?:" + "|".join(re.escape(t) for t in sorted(terms, key=lambda s: (-len(s), s))) + ")"


_EXPONENT = ("(?:" + char_class(PLAIN_POWERS, negate=False) + "(?![0-9])|" + re.escape(CARET)
             + char_class(SIGNS, negate=False) + "?[0-9]{1,2}(?![0-9])|" + char_class(SUP_SIGNS, negate=False) + "?"
             + char_class(SUP_DIGITS, negate=False) + "{1,2})")
_ATOM_BODY = ("(?:" + _alt(UNIT_PREFIXES) + "?" + _alt(UNIT_BASES) + "|" + _alt(UNIT_PREFIXES)
              + _alt(UNIT_PREFIXED_BASES) + ")")
_ATOM = _ATOM_BODY + _EXPONENT + "?"
_UNIT = _ATOM + "(?:(?:" + char_class(SLASH, negate=False) + "|" + PER + ")" + _ATOM + ")*"
_BOUNDARY = ("(?=\\Z|" + char_class(TRAILING, negate=False) + "(?![0-9])|" + char_class(OPENERS_BR, negate=False)
             + ")")
UNIT_RE = re.compile(_UNIT + _BOUNDARY)
ATOM_RE = re.compile(_ATOM_BODY)                     # one unit symbol without its exponent ('m', 'mL', 'h', 'cm')
# a unit symbol, then what ends it inside a word: the word end, a denominator, a product dot, an exponent, a closer
_UNIT_LED = re.compile(_ATOM_BODY + "(?=\\Z|" + char_class(SLASH | PRODUCT_DOTS | SUP_CHARS | TRAILING | {CARET}
                                                           | SIGNS, negate=False) + ")")


def unit_led(w: str) -> bool:
    """The word starts with a unit symbol that no letter continues ('mg', 'mg/kg', 'mmHg,', 'h-1', 'mL\u207b\u00b9';
    never 'months', 'Lung', 'events'): the unit after a ratio refuses the ratio (r7 round 2: a ratio is
    dimensionless)."""
    return _UNIT_LED.match(w) is not None


def unit_end(w: str, e: int) -> int:
    """End index of the unit production glued at w[e] (it ends at a boundary), -1 when w[e:] starts none."""
    m = UNIT_RE.match(w, e)
    return -1 if m is None else m.end()


def splits(unit: str, nxt: str) -> bool:
    """True when the word *nxt* one gap after a glued *unit* that ends its word continues the unit ('m' + 'L,AU',
    'm' + 'in', 'mg' + '/kg'): the unit and a head of the next word form a longer production."""
    m = UNIT_RE.match(unit + nxt[:SPLIT_LOOK])
    return m is not None and m.end() > len(unit)


__all__ = ["ATOM_RE", "UNIT_RE", "splits", "unit_end", "unit_led"]
