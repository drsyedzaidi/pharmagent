"""The surface the callers use (verify, value_checks, record_checks, extract_rules, qc_flags): thin helpers over
``tokens``. None of them reads a number the tokenizer did not resolve, and none keeps a cleaned view of the text.

Every character class here is a role set of alphabet.py (r7 round 2: no ``str.isascii``, no hand-typed set); the word
lists of the flag checks and of the dose rule are in ``app/evidence/word_lists.py``, outside the reader.
"""
from __future__ import annotations

import re
from decimal import Decimal

from app.evidence.letter_tables import casefold_letters
from app.evidence.numgrammar import words as W
from app.evidence.numgrammar.alphabet import (
    ASCII_LETTERS,
    CID_MARKER,
    CLOSERS_BR,
    CONTEXT_PUNCT,
    DECIMAL_MARKS,
    DIGITS,
    GAP,
    INSIDE,
    LETTERS,
    MULTIPLIERS,
    N_LETTERS,
    NUMBER_CLOSERS,
    OPENERS_BR,
    OUTSIDE_RE,
    SEPARATORS,
    SIGNS,
    SLASH,
    STARTERS,
    SUP_CHARS,
    TRAILING,
    WORD_RE,
    char_class,
)
from app.evidence.numgrammar.body import MIDDOT
from app.evidence.numgrammar.equivalents import fold
from app.evidence.numgrammar.productions import mask_chi_square
from app.evidence.numgrammar.tokenizer import tokens
from app.evidence.word_lists import DOSE_UNITS, SCALE_WORDS

_DATA_CELL = re.compile(char_class(INSIDE | GAP | OPENERS_BR | CLOSERS_BR | SLASH, negate=False) + "+")
_THOUSAND_PLAIN = re.compile("1" + char_class(SEPARATORS - {MIDDOT}, negate=False) + "?000")
_SCALE_WHY = frozenset({"ten_after_multiplier", "multiplier_operand", "detached_exponent", "bare_superscript_power"})
# the characters a word list reads as themselves: ASCII letters, numbers and their punctuation, brackets, the CONTEXT
# punctuation and the gaps; every other character (a Greek or accented letter, a mark, a superscript, a character the
# reader cannot read) may be a look-alike in a flag word and stands for any one character or for none (``loose_in``)
_PLAIN = (INSIDE - SUP_CHARS) | ASCII_LETTERS | OPENERS_BR | CLOSERS_BR | CONTEXT_PUNCT | GAP


def _foreign(c: str) -> bool:
    return c not in _PLAIN


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


def words_after(text: str, end: int, n: int) -> list[str]:
    """The next *n* whole words after the word that holds the offset *end* - 1 (fewer at the end of the text)."""
    t = fold(text)
    _, i = _word_bounds(t, max(end - 1, 0))
    out: list[str] = []
    while len(out) < n and (nxt := _next_word(t, i)) is not None:
        out.append(t[nxt[0]:nxt[1]])
        i = nxt[1]
    return out


def words_before(text: str, start: int, n: int) -> list[str]:
    """The *n* whole words before the word that holds the offset *start*, in reading order (fewer at the start)."""
    t = fold(text)
    a, _ = _word_bounds(t, start)
    out: list[str] = []
    while len(out) < n and (prev := _prev_word(t, a)) is not None:
        out.insert(0, t[prev[0]:prev[1]])
        a = prev[0]
    return out


def unreadable(text: str) -> tuple[int, str] | None:
    """(offset, 'U+200B' or '(cid:3)') of the first character of *text* outside the readable alphabet (INSIDE,
    CONTEXT, GAP and the equivalents: the derived rule of r7) or the first pdfminer unmapped-glyph marker; None when the
    whole string is readable (the chi of a chi-square form is: r7). Offsets index *text* itself: the fold is 1:1."""
    t = mask_chi_square(fold(text))
    ch, cid = OUTSIDE_RE.search(t), CID_MARKER.search(t)
    if ch is not None and (cid is None or ch.start() < cid.start()):
        return ch.start(), f"U+{ord(t[ch.start()]):04X}"
    return None if cid is None else (cid.start(), cid.group())


# ------------------------------------------------------------ labels, cells ---

def loose_in(word: str, targets: frozenset[str]) -> bool:
    """*word* (the alphabet's capitals lower-cased) is one of the lowercase ASCII *targets* where every FOREIGN
    character (a Greek or accented letter, a mark, a superscript, a character the reader cannot read: everything outside
    ``_PLAIN``) stands for any one character (``τo`` is ``to``, ``αbove`` is ``above``, ``thоusands`` is ``thousands``)
    or for none (a zero-width character inside ``thousands``). Only ever used to RAISE a flag (a word list of the value
    checks), never to read a number: a look-alike or padded spelling must flag like the plain spelling does."""
    w = casefold_letters(word)
    if w in targets:
        return True
    if not any(_foreign(c) for c in w):
        return False
    if "".join(c for c in w if not _foreign(c)) in targets:        # a foreign character inside the word
        return True
    return any(len(t) == len(w) and all(x == y or _foreign(x) for x, y in zip(w, t, strict=True)) for t in targets)


_TRAILING = "".join(sorted(TRAILING))


def _strip_trailing(w: str) -> str:
    """*w* without its trailing closers and punctuation (``alphabet.TRAILING``)."""
    return w.rstrip(_TRAILING)


def _core(w: str) -> str:
    """The word without one leading starter or sign and its trailing closers."""
    core = w[1:] if (w[0] in STARTERS or w[0] in SIGNS) and len(w) > 1 else w
    return _strip_trailing(core) or core


def _stray_multiplier(core: str) -> bool:
    """A word that is one multiplier character, or one foreign character where a multiplier would stand."""
    return W.is_mult_word(core) or (len(core) == 1 and _foreign(core))


def _thousand_word(core: str) -> bool:
    """``x1000``, ``/1,000``, ``×1000`` or a stray symbol glued to ``1000``: a multiplier or divisor of 1000."""
    return len(core) > 1 and (core[0] in MULTIPLIERS or core[0] in SLASH or _foreign(core[0])) \
        and _THOUSAND_PLAIN.fullmatch(core[1:]) is not None


def _unreadable_scale(core: str) -> bool:
    """A multiplier character glued to a foreign character (``x1０００``, ``×１０００``): a scale the label prints in a
    form the reader cannot read."""
    return len(core) > 1 and core[0] in MULTIPLIERS and any(_foreign(c) for c in core[1:])


def _power_word(core: str) -> bool:
    """``×10^3``, ``x10³`` or a stray symbol glued to a power of ten."""
    return W.special(core) in (W.MULT_POWER, W.MULT_TEN) or (
        len(core) > 1 and _foreign(core[0]) and W.BARE_POWER_RE.fullmatch(core[1:]) is not None)


def _scale_word(core: str, prev: str, nxt: str) -> bool:
    """One label word that scales the printed number."""
    if W.special(core) == W.E_WORD or _power_word(core) or _thousand_word(core) or loose_in(core, SCALE_WORDS) \
            or _unreadable_scale(core):
        return True
    denominator = prev.endswith(tuple(SLASH)) or casefold_letters(prev) == "per"   # 'cells/10^6 cells' is a unit
    if not denominator and W.BARE_POWER_RE.fullmatch(core):
        return True
    return (_stray_multiplier(core) or core in SLASH) and _THOUSAND_PLAIN.fullmatch(_strip_trailing(nxt)) is not None


def _scale_view(t: str) -> bool:
    """The scale reading of one folded label (see ``scale_shapes``)."""
    ws = [m.group() for m in WORD_RE.finditer(t)]
    for i, w in enumerate(ws):
        core = _core(w)
        prev = ws[i - 1] if i else ""
        nxt = ws[i + 1] if i + 1 < len(ws) else ""
        if W.letter_led(core) and not (W.special(core) or loose_in(core, SCALE_WORDS) or _stray_multiplier(core)
                                       or _thousand_word(core) or _power_word(core) or _unreadable_scale(core)):
            continue
        if _scale_word(core, prev, nxt):
            return True
    return any(tok.value is None and (tok.why in _SCALE_WHY or (
        tok.reason == "unresolved_context" and not W.letter_led(t[tok.start:tok.end])))
        for tok in tokens(t, ".", mode="cell"))


def scale_shapes(label: str) -> bool:
    """True when a header, row label, unit or proposed unit carries a multiplier of the printed number or cannot be
    read for one: an accepted or broken power-of-ten word (``×10³``, ``x10^3``, ``× 10 3``, ``10³``), a spaced
    exponent (``E+03``), a multiplier (or a stray foreign symbol where one stands) before 1000 in any grouping,
    a scale word (``word_lists.SCALE_WORDS``; a look-alike spelling counts: ``loose_in``), or a number of the label that
    touches a character outside the alphabet. A power after ``/`` or ``per`` is a unit denominator and does not
    count; a label word led by a letter never counts (``AUC0-inf``, ``ng.h/mL``). The label is read twice, as it
    stands and with every character the reader cannot read removed (``OUTSIDE_RE``), and either reading raises the flag:
    a character the reader cannot read, put anywhere in the label, never hides a scale. Neither reading can lower a
    flag."""
    t = fold(label)
    return _scale_view(t) or (OUTSIDE_RE.search(t) is not None and _scale_view(OUTSIDE_RE.sub("", t)))


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
    if j == 0 or t[j - 1] not in N_LETTERS:
        return None
    k = j - 1
    return None if k > 0 and t[k - 1] in LETTERS else k


def find_n_lead(text: str) -> list[tuple[int, int, int, int]]:
    """(offset of the ``n``, start, end, n) of every ``n = <integer of 1-6 digits>``; the span of the integer is
    text[start:end], the span of the whole phrase text[lead:end]."""
    t = fold(text)
    out: list[tuple[int, int, int, int]] = []
    for tok in tokens(text, ".", mode="page"):
        digits = tok.canonical
        lead = _before_equals(t, tok.start) if tok.shape == "int" and 1 <= len(digits) <= 6 else None
        if lead is not None and all(c in DIGITS for c in digits):
            out.append((lead, tok.start, tok.end, round(tok.value)))
    return out


def find_dose(text: str, units: frozenset[str] = DOSE_UNITS) -> list[tuple[int, int, float, str]]:
    """(start, end, value, unit word) of every number of at most 5 integer digits and 3 decimals followed by a unit
    word of an extraction rule (*units*: mg, ug, g, mg/kg...). A unit glued to the digits (``150mg``) is read by the
    unit production since r7 but is no candidate here (the rule reads the next word); the caller applies its own context
    pattern to the surrounding line. The text is folded
    once: the next word of every number is found by a forward scan, never by folding the page again."""
    t = fold(text)
    out: list[tuple[int, int, float, str]] = []
    for tok in tokens(text, ".", mode="page"):
        whole, _, frac = tok.canonical.partition(".")
        if tok.shape not in ("int", "decimal", "foreign_decimal", "style_dependent") or not whole \
                or len(whole) > 5 or len(frac) > 3 or not all(c in DIGITS for c in whole + frac):
            continue
        nxt = _next_word(t, _word_bounds(t, max(tok.end - 1, 0))[1])
        unit = _strip_trailing(t[nxt[0]:nxt[1]]) if nxt else ""
        if casefold_letters(unit) in units:
            out.append((tok.start, tok.end, tok.value, unit))
    return out


# ------------------------------------------------------------ digits, claims ---

def digits_of(text: str) -> str:
    """The ASCII digits of *text*, in order: how a literal is compared with the digits of a longer one."""
    return "".join(c for c in text if c in DIGITS)


def is_digit_run(word: str, lo: int, hi: int) -> bool:
    """*word* is made only of ASCII digits, lo to hi of them."""
    return lo <= len(word) <= hi and all(c in DIGITS for c in word)


def digit_group(word: str, lo: int, hi: int, *, decimals: bool = False, lead: bool = False) -> bool:
    """*word* starts with lo..hi ASCII digits and holds nothing but punctuation, closers or stars after them (and, with
    *decimals*, a ``.`` or ``,`` and more digits first): a digit group a text layer may have split off a number
    (``145`` of ``12 145``, ``234.5`` of ``1 234.5``, ``234)`` of ``(1 234)``). With *lead* one opener and one sign
    may stand before the digits: the head of a group (``(1``, ``-1``), never the group itself."""
    k = 0
    if lead:
        k = 1 if word[:1] in OPENERS_BR else 0
        k += 1 if word[k:k + 1] in SIGNS else 0
    e = len(word)
    while e > k and word[e - 1] in NUMBER_CLOSERS:
        e -= 1
    body = word[k:e]
    i = 0
    while i < len(body) and body[i] in DIGITS:
        i += 1
    if not lo <= i <= hi:
        return False
    return i == len(body) or (decimals and body[i] in DECIMAL_MARKS and is_digit_run(body[i + 1:], 1, len(body)))


def claim_decimal(value: str, style: str = ".") -> Decimal | None:
    """The exact decimal the one number of a proposed value names, else None (no number, several, unresolved)."""
    toks = tokens(value, style, mode="claim")
    return toks[0].decimal_value if len(toks) == 1 else None


__all__ = ["claim_decimal", "digit_group", "digits_of", "find_dose", "find_n_lead", "is_data_cell", "loose_in",
           "number_spans", "numbers_in", "parse_claim", "scale_shapes", "unreadable", "words_after", "words_before"]
