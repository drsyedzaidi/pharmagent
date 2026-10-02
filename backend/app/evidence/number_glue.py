"""What touches a number token: a character a reader may take as PART of the number makes the
token unreadable (fail closed), so a fragment of the printed number can never pass as the number.

- A letter glued to the digits ('12З' drawn like 123, '1O5', '23l', 'б12') unless it is a unit
  written without a space ('150mg', '24h', '1.73m2', '3MIU').
- A non-ASCII character between ASCII digits ('12․4', '24‘560', '1⁄2'); '±' / '∓' separate an
  estimate from its dispersion and stay a boundary. A backtick counts as such a character.
- A minus / plus / hyphen look-alike, or an invisible format character, directly before the
  token ('˗20', a soft hyphen): the sign of the number is unknown.
- A vulgar fraction or combining mark glued after the digits ('12½', '3́').
- Strict (page) text only: a thin / no-break space with a digit on its other side that the
  number grammar did not take as a 3-digit group ('0.123 4' ISO decimal grouping).
"""
from __future__ import annotations

import re
import unicodedata

_ASCII_DIGITS = frozenset("0123456789")
_SEPARATES = frozenset("±∓")
_LETTERS = re.compile(r"[^\W\d_]+")
# units (and ordinals) a paper may glue to the digits; case matters ('l', 'I', 'O' look like digits)
GLUED_UNITS = frozenset({
    "g", "mg", "µg", "μg", "ug", "mcg", "ng", "pg", "fg", "kg",
    "L", "mL", "ml", "dL", "dl", "µL", "μL", "uL", "nL",
    "mol", "mmol", "µmol", "μmol", "umol", "nmol", "pmol", "M", "mM", "µM", "μM", "uM", "nM", "pM",
    "IU", "MIU", "mIU", "U", "kU", "mU", "kDa", "Da",
    "h", "hr", "hrs", "min", "mins", "s", "sec", "d", "wk", "wks", "y", "yr", "yrs", "mo", "days",
    "m", "cm", "mm", "kcal", "bpm", "mmHg", "x", "X", "fold", "am", "pm", "AM", "PM",
    "st", "nd", "rd", "th",
})
_SQUARED = frozenset({"m2", "m3"})          # '1.73m2' body surface area


def _signlike(ch: str) -> bool:
    """True for an unfolded minus / plus / hyphen / dash look-alike or an invisible format char."""
    if ch.isascii() or ch in _SEPARATES:
        return False
    if unicodedata.category(ch) == "Cf":
        return True
    name = unicodedata.name(ch, "")
    return any(w in name for w in ("MINUS", "HYPHEN", "DASH", "PLUS"))


def _odd_between(ch: str) -> bool:
    """A character between two ASCII digits that the number grammar did not read."""
    return ch == "`" or (not ch.isascii() and ch not in _SEPARATES)


def _letters_after(text: str, end: int) -> bool:
    m = _LETTERS.match(text, end)
    if m is None:
        return False
    run, nxt = m.group(0), text[m.end():m.end() + 1]
    if nxt.isdigit():
        tail = text[m.end() + 1:m.end() + 2]
        return not (run + nxt in _SQUARED and not tail.isdigit())
    return run not in GLUED_UNITS


def _fraction_or_mark(ch: str) -> bool:
    cat = unicodedata.category(ch)
    return cat.startswith("M") or (cat == "No" and "⁄" in unicodedata.normalize("NFKC", ch))


def glued(text: str, start: int, end: int, *, strict: bool) -> bool:
    """True when text[start:end] (a folded number token) touches a character a reader may read as
    part of it (see the module docstring)."""
    before = text[start - 1] if start > 0 else ""
    before2 = text[start - 2] if start > 1 else ""
    after = text[end] if end < len(text) else ""
    after2 = text[end + 1] if end + 1 < len(text) else ""
    if before and (_signlike(before) or (unicodedata.category(before).startswith("L") and not before.isascii())):
        return True
    if before and before2 in _ASCII_DIGITS and _odd_between(before):
        return True
    if after and after2 in _ASCII_DIGITS and _odd_between(after):
        return True
    if after and (_fraction_or_mark(after) or _letters_after(text, end)):
        return True
    if strict and ((after == " " and after2 in _ASCII_DIGITS) or (before == " " and before2 in _ASCII_DIGITS)):
        return True
    return False
