"""The RANGE half of censored_value (split out of value_bounds.py in r7 round 4): a verified number that is one end of
a printed range.

* the reader's range mark (``Tok.ranged``: a dash glued to the number, a lone '-' or '~' word, a unit or percent between
  the number and the range word: '20 mg - 40 mg'), a lone '-' or '~' word or cell beside it, a dash glued to the end of
  the previous cell after its number (the cells ['5.0-', '12.4']);
* a RANGE WORD (``word_lists.TO``: 'to', and r7 round 4 'through' 8 corpus uses, 'thru', 'until', 'till') with a number
  on its other side, a unit word allowed between ('12 to 15', '20 mg to 40 mg', 'days 1 through 5'): read as the head of
  its word, openers and hyphen wrappers aside ('(to 15)', '-to-'), and glued to its number ('12 to15');
* 'between 12 and 24'.
"""
from __future__ import annotations

from app.evidence.numgrammar import GAP, Tok, tokens
from app.evidence.numgrammar import glue as G
from app.evidence.numgrammar import words as W
from app.evidence.numgrammar.alphabet import (
    CLOSERS_BR,
    DIGITS,
    GROUP_ONLY,
    HYPHEN,
    LETTERS,
    OPENERS_BR,
    PERCENTS,
    RANGE_WORDS,
    TRAILING,
)
from app.evidence.word_lists import AND, BETWEEN, TO
from app.evidence.word_match import loose_in

TRAIL = "".join(sorted(TRAILING - CLOSERS_BR - GROUP_ONLY))
CLOSE = "".join(sorted(CLOSERS_BR))
_HYPHENS = "".join(sorted(HYPHEN))
_OPEN = "".join(sorted(OPENERS_BR))


def numeric_word(word: str, style: str) -> bool:
    """A word that starts with a resolved number ('15', '15.4', '(15'): the other end of 'to'."""
    toks = tokens(word.rstrip(TRAIL + CLOSE), style, mode="cell")
    return bool(toks) and toks[0].start <= 1 and toks[0].value is not None


def unit_word(word: str) -> bool:
    """A word without a digit between a number and a range or bound word: a unit or a noun ('mg', 'h', 'patients',
    'years') or a lone percent sign."""
    core = word.rstrip(TRAIL + CLOSE)
    return bool(core) and not any(c in DIGITS for c in core) and (G.prose(core) or all(c in PERCENTS for c in core))


def to_split(word: str) -> tuple[bool, str]:
    """(the word is a range word, the number glued after it): the head of *word* once its openers, hyphen wrappers and
    trailing marks go ('to', '(to', '-to-', 'through'), or a range word glued to digits ('to15' -> 'to', '15')."""
    core = word.lstrip(_OPEN).rstrip(TRAIL + CLOSE).strip(_HYPHENS)
    if core and loose_in(core, TO):
        return True, ""
    k = 0
    while k < len(core) and core[k] in LETTERS:
        k += 1
    if 0 < k < len(core) and core[k] in DIGITS and loose_in(core[:k], TO):
        return True, core[k:]
    return False, ""


def _to_number(words: list[str], i: int, style: str) -> bool:
    """words[i] is a range word and a number stands after it, glued ('to15') or as the next word ('to 15')."""
    is_to, rest = to_split(words[i])
    if not is_to:
        return False
    return numeric_word(rest, style) if rest else i + 1 < len(words) and numeric_word(words[i + 1], style)


def _own_word(text: str, a: int) -> str:
    """The whole word of *text* that holds the offset *a*."""
    k, e = a, a
    while k > 0 and text[k - 1] not in GAP:
        k -= 1
    while e < len(text) and text[e] not in GAP:
        e += 1
    return text[k:e]


def ranged(tok: Tok, ctx: str, a: int, prev: list[str], nxt: list[str], style: str) -> bool:
    """One end of a printed range (see the module docstring); *prev* and *nxt* are the words before and after the
    token's word, in reading order."""
    to_before = len(prev) > 1 and to_split(prev[-1]) == (True, "") and (numeric_word(prev[-2], style) or (
        len(prev) > 2 and unit_word(prev[-2]) and numeric_word(prev[-3], style)))
    to_after = bool(nxt) and (_to_number(nxt, 0, style) or (len(nxt) > 1 and unit_word(nxt[0])
                                                              and _to_number(nxt, 1, style)))
    between = (bool(prev) and loose_in(prev[-1], BETWEEN) and len(nxt) > 1 and loose_in(nxt[0], AND)
               and numeric_word(nxt[1], style)) or (len(prev) > 2 and loose_in(prev[-1], AND)
                                                   and numeric_word(prev[-2], style) and loose_in(prev[-3], BETWEEN))
    # the tokenizer marks a dash glued to the number, a lone '-' or '~' word before or after it (a unit between them
    # too); a lone one in the neighbouring table cell is read here, and so is a dash glued to the end of the previous
    # cell after its number (r7 writer 3: the cells ['5.0-', '12.4'] are one range)
    glued = bool(prev) and prev[-1][-1] in HYPHEN and W.ends_with_number(prev[-1][:-1]) \
        and W.digit_led(_own_word(ctx, a))
    lone = bool(prev) and prev[-1] in RANGE_WORDS or bool(nxt) and nxt[0] in RANGE_WORDS
    return tok.ranged or glued or lone or to_before or to_after or between


__all__ = ["CLOSE", "TRAIL", "numeric_word", "ranged", "to_split", "unit_word"]
