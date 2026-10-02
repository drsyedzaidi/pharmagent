"""Unit identity: the comparison form of a unit, the equivalence table, and the SI prefix whose
case matters ('MIU' mega vs 'mIU' milli: a 10^9 error that casefolding would hide).

normalise_unit casefolds; prefix_conflict() is the case-sensitive check made beside it.
"""
from __future__ import annotations

import re
import unicodedata

_UNIT_SYNONYMS = ((re.compile(r"\b(hrs?|hours?)\b"), "h"), (re.compile(r"\b(liters?|litres?)\b"), "l"),
                  (re.compile(r"\bmcg\b"), "ug"), (re.compile(r"\b(seconds?|sec)\b"), "s"))
_UNIT_STRIP = re.compile(r"[\s·*×.]+")
_EQUIV = {"mcg": "ug", "cc": "ml", "hr": "h", "hrs": "h", "mins": "min", "sec": "s", "lit": "l"}
_CASE_PREFIXES = frozenset("Mm")           # mega / milli: the only prefix pair told apart by case alone


def normalise_unit(unit: str | None) -> str:
    """Comparison form: NFKC, casefold, µ/μ -> 'u', hr -> h, separators (space · * × .) removed."""
    u = unicodedata.normalize("NFKC", unit or "").casefold().replace("μ", "u").replace("µ", "u")
    for pattern, rep in _UNIT_SYNONYMS:
        u = pattern.sub(rep, u)
    return _UNIT_STRIP.sub("", u)


def canonical_unit(unit: str | None) -> str:
    """normalise_unit with each '/'-piece mapped through the equivalence table (mcg = ug = µg, mL = ml)."""
    return "/".join(_EQUIV.get(piece, piece) for piece in normalise_unit(unit).split("/"))


def _pieces(unit: str | None) -> list[str]:
    text = unicodedata.normalize("NFKC", unit or "")
    return [_UNIT_STRIP.sub("", p) for p in text.split("/")]


def _prefix(piece: str) -> tuple[str, str]:
    """(case-significant prefix letter or '', the rest) of one unit piece ('mIU' -> ('m', 'IU'))."""
    if len(piece) > 1 and piece[0] in _CASE_PREFIXES and piece[1].isalpha():
        return piece[0], piece[1:]
    return "", piece


def prefix_conflict(proposed: str | None, printed: str | None) -> str | None:
    """'mismatch' when the two units differ in a mega/milli prefix ('MIU' vs 'mIU', 'Mg' vs 'mg');
    'ambiguous' when an all-capitals printed unit could be the proposed one set in capitals ('MG',
    'MG/ML' for mg/mL; never 'MIU' for mIU, whose other letters are capitals anyway); else None."""
    a, b = _pieces(proposed), _pieces(printed)
    if len(a) != len(b):
        return None
    pairs = [(_prefix(pa), _prefix(pb)) for pa, pb in zip(a, b)]
    if not any(xa and xb and xa != xb for (xa, _), (xb, _) in pairs):
        return None
    letters = [ch for piece in b for ch in piece if ch.isalpha()]
    caps_page = bool(letters) and all(ch.isupper() for ch in letters)
    lower_rest = any(ch.islower() for (_, ra), _ in pairs for ch in ra)
    return "ambiguous" if caps_page and lower_rest else "mismatch"


def same_unit(a: str | None, b: str | None) -> bool:
    """True when two record units name the same unit (equivalence table, prefix case); a unit
    present on one side only is a different unit."""
    if not (a or "").strip() and not (b or "").strip():
        return True
    return canonical_unit(a) == canonical_unit(b) and prefix_conflict(a, b) is None
