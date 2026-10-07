"""The surface the callers use (verify, value_checks, record_checks, extract_rules, qc_flags): thin helpers over
``tokens``. None of them reads a number the tokenizer did not resolve, and none keeps a cleaned view of the text.
"""
from __future__ import annotations

import re

from app.evidence.numgrammar import words as W
from app.evidence.numgrammar.alphabet import (
    DIGITS,
    GAP,
    INSIDE,
    LETTERS,
    MULTIPLIERS,
    STARTERS,
    WORD_RE,
    WORD_TIER_RE,
    char_class,
)
from app.evidence.numgrammar.equivalents import THIN, fold
from app.evidence.numgrammar.tok import Tok
from app.evidence.numgrammar.tokenizer import tokens

_TRAIL_STRIP = ".,;:)"
_DOSE_UNITS = frozenset({"mg", "\u03bcg", "\u00b5g", "ug", "g", "mg/kg", "\u03bcg/kg", "\u00b5g/kg"})
_DATA_CELL = re.compile(char_class(INSIDE | GAP | frozenset("()/"), negate=False) + "+")
_THOUSAND_WORD = re.compile("[" + "".join(sorted(MULTIPLIERS)) + "/]1[,.'" + THIN + "]?000")
_THOUSAND_PLAIN = re.compile("1[,.'" + THIN + "]?000")
_SCALE_WORDS = frozenset({"thousand", "thousands", "million", "millions", "billion", "billions"})
_SCALE_WHY = frozenset({"ten_after_multiplier", "multiplier_operand", "detached_exponent", "bare_superscript_power"})


def parse_claim(value: str, style: str = ".") -> float | None:
    """The one resolved number a proposed value names, else None (no number, several, or unresolved)."""
    toks = tokens(value, style, mode="claim")
    return toks[0].value if len(toks) == 1 else None


def number_spans(text: str, style: str = ".", *, strict: bool = False) -> list[tuple[int, int, float]]:
    """(start, end, value) of every resolved number; *strict* is the page reading (an ASCII space never groups)."""
    toks = tokens(text, style, mode="page" if strict else "claim")
    return [(t.start, t.end, t.value) for t in toks if t.value is not None]


def numbers_in(text: str, style: str = ".") -> list[tuple[str, float]]:
    """(literal as written, value) of every resolved number of a snippet or proposed value."""
    return [(text[s:e], v) for s, e, v in number_spans(text, style)]


def token_at(text: str, a: int, b: int, style: str = ".", *, mode: str = "page") -> Tok | None:
    """The token or unresolved expression whose span is exactly text[a:b], else None."""
    return next((t for t in tokens(text, style, mode=mode) if (t.start, t.end) == (a, b)), None)


# ------------------------------------------------------------------ words ---

def _word_bounds(t: str, pos: int) -> tuple[int, int]:
    """[a, b) of the word that holds or touches the offset *pos* (a word edge when *pos* sits in a gap)."""
    a = pos
    while a > 0 and t[a - 1] not in GAP:
        a -= 1
    b = pos
    while b < len(t) and t[b] not in GAP:
        b += 1
    return a, b


def _next_word(t: str, i: int) -> tuple[int, int] | None:
    while i < len(t) and t[i] in GAP:
        i += 1
    if i >= len(t):
        return None
    j = i
    while j < len(t) and t[j] not in GAP:
        j += 1
    return i, j


def _prev_word(t: str, i: int) -> tuple[int, int] | None:
    while i > 0 and t[i - 1] in GAP:
        i -= 1
    if i <= 0:
        return None
    j = i
    while j > 0 and t[j - 1] not in GAP:
        j -= 1
    return j, i


def words_around(text: str, start: int, end: int) -> tuple[str, str]:
    """(W-, W+): the whole words one gap run before and after the word that holds text[start:end] ('' at an edge).
    A run of gap characters of any length and mix is ONE separator, so padding never widens the window."""
    t = fold(text)
    a, _ = _word_bounds(t, start)
    _, b = _word_bounds(t, max(end - 1, start))
    prev, nxt = _prev_word(t, a), _next_word(t, b)
    return (t[prev[0]:prev[1]] if prev else "", t[nxt[0]:nxt[1]] if nxt else "")


def words_after(text: str, end: int, n: int) -> list[str]:
    """The next *n* whole words after the word that holds the offset *end* - 1 (fewer at the end of the text)."""
    t = fold(text)
    _, i = _word_bounds(t, max(end - 1, 0))
    out: list[str] = []
    while len(out) < n and (nxt := _next_word(t, i)) is not None:
        out.append(t[nxt[0]:nxt[1]])
        i = nxt[1]
    return out


# ------------------------------------------------------------ labels, cells ---

def _scale_word(core: str, prev: str, nxt: str) -> bool:
    """One label word that scales the printed number."""
    if W.special(core) in (W.MULT_POWER, W.MULT_TEN, W.E_WORD) or _THOUSAND_WORD.fullmatch(core) \
            or core.casefold() in _SCALE_WORDS:
        return True
    denominator = prev.endswith("/") or prev.casefold() == "per"          # 'cells/10^6 cells' is a unit
    if not denominator and W.BARE_POWER_RE.fullmatch(core):
        return True
    return (W.is_mult_word(core) or core == "/") and _THOUSAND_PLAIN.fullmatch(nxt.rstrip(_TRAIL_STRIP)) is not None


def scale_shapes(label: str) -> bool:
    """True when a header, row label, unit or proposed unit carries a multiplier of the printed number: an accepted
    or broken power-of-ten word (``×10³``, ``x10^3``, ``× 10 3``, ``10³``), a spaced exponent (``E+03``), a ``×`` or
    ``/`` before 1000 in any grouping, or thousand(s)/million(s)/billion(s). A power after ``/`` or ``per`` is a
    unit denominator and does not count; letter-led words never count (``AUC0-inf``, ``ng.h/mL``). Characters outside
    the alphabet are dropped first: this check may only raise a flag, never hide one."""
    t = WORD_TIER_RE.sub("", fold(label))
    ws = [m.group() for m in WORD_RE.finditer(t)]
    for i, w in enumerate(ws):
        core = w[1:] if w[0] in STARTERS and len(w) > 1 else w
        core = core.rstrip(_TRAIL_STRIP) or core
        prev = ws[i - 1] if i else ""
        nxt = ws[i + 1] if i + 1 < len(ws) else ""
        if W.letter_led(core) and not (W.special(core) or core.casefold() in _SCALE_WORDS
                                       or _THOUSAND_WORD.fullmatch(core)):
            continue
        if _scale_word(core, prev, nxt):
            return True
    return any(tok.value is None and tok.why in _SCALE_WHY for tok in tokens(t, ".", mode="cell"))


def is_data_cell(text: str) -> bool:
    """A cell of numbers and number punctuation only (digits, signs, separators, brackets, ``/``): ``Day 1`` is a
    header, ``12.4 (3.1)`` a data cell."""
    return _DATA_CELL.fullmatch(fold(text)) is not None and bool(tokens(text, ".", mode="cell"))


# ------------------------------------------------------------- text rules ---

def _before_equals(t: str, i: int) -> int | None:
    """Index of the ``n`` of ``n =`` / ``N=`` right before offset *i* (one gap run either side of ``=``), else None."""
    j = i
    while j > 0 and t[j - 1] in GAP:
        j -= 1
    if j == 0 or t[j - 1] != "=":
        return None
    j -= 1
    while j > 0 and t[j - 1] in GAP:
        j -= 1
    if j == 0 or t[j - 1] not in ("n", "N"):
        return None
    k = j - 1
    return None if k > 0 and t[k - 1] in LETTERS else k


def find_n(text: str) -> list[tuple[int, int, int]]:
    """(start, end, n) of every ``n = <integer of 1-6 digits>``: ``n = 12``, ``n=12``, ``(n=12)``, ``N=12``."""
    t = fold(text)
    out: list[tuple[int, int, int]] = []
    for tok in tokens(text, ".", mode="page"):
        digits = tok.canonical
        if tok.shape == "int" and 1 <= len(digits) <= 6 and all(c in DIGITS for c in digits) \
                and _before_equals(t, tok.start) is not None:
            out.append((tok.start, tok.end, round(tok.value)))
    return out


def find_dose(text: str) -> list[tuple[int, int, float, str]]:
    """(start, end, value, unit word) of every number of at most 5 integer digits and 3 decimals followed by a unit
    word of an extraction rule (mg, ug, g, mg/kg...). A unit glued to the digits (``150mg``) is unresolved, so it
    is never a candidate; the caller applies its own context pattern to the surrounding line."""
    out: list[tuple[int, int, float, str]] = []
    for tok in tokens(text, ".", mode="page"):
        whole, _, frac = tok.canonical.partition(".")
        if tok.shape not in ("int", "decimal", "foreign_decimal", "style_dependent") or not whole \
                or len(whole) > 5 or len(frac) > 3 or not all(c in DIGITS for c in whole + frac):
            continue
        nxt = words_after(text, tok.end, 1)
        unit = nxt[0].rstrip(_TRAIL_STRIP) if nxt else ""
        if unit.casefold() in _DOSE_UNITS:
            out.append((tok.start, tok.end, tok.value, unit))
    return out


__all__ = ["find_dose", "find_n", "is_data_cell", "number_spans", "numbers_in", "parse_claim", "scale_shapes",
           "token_at", "words_after", "words_around"]
