"""The candidates of the keyless DOSE rule (r7 round 4: moved out of the reader package, which never reads a word list).

``find_dose`` lists every number the reader resolves that is followed by a dose unit word of the extraction rule
(``word_lists.DOSE_UNITS``: mg, ug, g, mg/kg ...). It only proposes a record for a human to verify: the number is the
reader's token, the unit word is compared after the alphabet's case fold, and nothing here makes a number readable.
"""
from __future__ import annotations

from app.evidence.letter_tables import casefold_letters
from app.evidence.numgrammar import tokens
from app.evidence.numgrammar.alphabet import DIGITS, GAP, POINT, TRAILING
from app.evidence.numgrammar.equivalents import fold
from app.evidence.word_lists import DOSE_UNITS

DOSE_SHAPES = frozenset({"int", "decimal", "foreign_decimal", "style_dependent"})
MAX_WHOLE, MAX_FRACTION = 5, 3                           # at most 5 integer digits and 3 decimals
_TRAIL = "".join(sorted(TRAILING))


def _word_end(t: str, i: int) -> int:
    while i < len(t) and t[i] not in GAP:
        i += 1
    return i


def _next_word(t: str, i: int) -> str:
    while i < len(t) and t[i] in GAP:
        i += 1
    return t[i:_word_end(t, i)]


def find_dose(text: str, units: frozenset[str] = DOSE_UNITS) -> list[tuple[int, int, float, str]]:
    """(start, end, value, unit word) of every number of at most 5 integer digits and 3 decimals followed by a unit
    word of an extraction rule. A unit glued to the digits (``150mg``) is read by the unit production but is no
    candidate here (the rule reads the next word); the caller applies its own context pattern to the surrounding line.
    The text is folded once: the next word of every number is found by a forward scan."""
    t = fold(text)
    out: list[tuple[int, int, float, str]] = []
    for tok in tokens(text, POINT, mode="page"):
        whole, _, frac = tok.canonical.partition(POINT)
        if tok.shape not in DOSE_SHAPES or not whole or len(whole) > MAX_WHOLE or len(frac) > MAX_FRACTION \
                or not all(c in DIGITS for c in whole + frac):
            continue
        unit = _next_word(t, _word_end(t, max(tok.end - 1, 0))).rstrip(_TRAIL)
        if casefold_letters(unit) in units:
            out.append((tok.start, tok.end, tok.value, unit))
    return out


__all__ = ["find_dose"]
