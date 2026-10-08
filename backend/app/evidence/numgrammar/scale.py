"""Does a label scale the printed number? (r7 round 3: an ALLOW-LIST over the label's words, never a list of scale
shapes; the round-3 audit found every unlisted scale form passing: '×10−3', 'x100', '×1e3', 'x 10000', '(hundreds)').

A label (a header cell, a row label, a unit cell, a proposed unit, or a data cell beside the number) is SCALE-FREE only
when every PART of every word (a word split at its brackets and slashes) is one of these productions::

    punctuation    nothing left once the trailing closers and punctuation go
    operators      signs, plus-minus, qualifiers and percent signs only ('-', '±', '<', '%'), a footnote star run
    exponent       a lone e or E (an exponent marker with no operand)
    prose          glue.prose with every power after a unit symbol (glue.unit_powers) and no scale word
    label          glue.label (a letter-led name, its digits letters of the name: 'AUC0-24', 'CYP2E1') with the same two
                   conditions ('AUC10^3' and 'th0usands' are none)
    number         digit-led, every number resolved, and none a scale: a bare power ('10^3') unless it is a unit
                   denominator (after a slash or the word 'per': 'cells/10^6', 'per 10^6'), a scientific number or a
                   written-out power of ten ('1000', '1,000', '0.001'; never '1'); in a DATA cell beside the number
                   (data=True) every resolved number but a bare power is data

A sample size ('n = 45', 'N=1208') and a percentage ('95%') are counts, read as names first. Everything else raises the
flag: a multiplier (lone 'x', '×', glued 'x1000'), an exponent letter before digits ('E3', 'e-3'), a scale word of
``word_lists.SCALE_WORDS`` (a look-alike spelling too: ``loose_in``, where a foreign character or a digit inside a
letter word stands for a letter), a number the reader cannot resolve, and any character the reader cannot read.
"""
from __future__ import annotations

import re

from app.evidence.letter_tables import casefold_letters
from app.evidence.numgrammar import glue as G
from app.evidence.numgrammar import product as P
from app.evidence.numgrammar import words as W
from app.evidence.numgrammar.alphabet import (
    ASCII_LETTERS,
    CID_MARKER,
    CLOSERS_BR,
    CONTEXT_PUNCT,
    DECIMAL_MARKS,
    DIGITS,
    EXP_MARKERS,
    GAP,
    INSIDE,
    LETTERS,
    N_LETTERS,
    NAME_LETTER,
    OPENERS_BR,
    OUTSIDE_RE,
    PERCENTS,
    PLUS_MINUS,
    POINT,
    QUALIFIERS,
    SIGNS,
    SLASH,
    STAR,
    SUP_CHARS,
    TRAILING,
    WORD_RE,
    char_class,
)
from app.evidence.numgrammar.equivalents import fold
from app.evidence.numgrammar.tokenizer import tokens
from app.evidence.word_lists import SCALE_WORDS

# the characters a word list reads as themselves: ASCII letters, numbers and their punctuation, brackets, the CONTEXT
# punctuation and the gaps; every other character (a Greek or accented letter, a mark, a superscript, a character the
# reader cannot read) may be a look-alike in a flag word and stands for any one character or for none (``loose_in``)
_PLAIN = (INSIDE - SUP_CHARS) | ASCII_LETTERS | OPENERS_BR | CLOSERS_BR | CONTEXT_PUNCT | GAP
_OPERATORS = SIGNS | {PLUS_MINUS} | QUALIFIERS | PERCENTS          # a part of only these is inert ('-', '±', '<', '%')
_SPLIT = re.compile(char_class(OPENERS_BR | CLOSERS_BR | SLASH, negate=False))
_STRIP = "".join(sorted(TRAILING))
_GAPS = char_class(GAP, negate=False) + "*"
_COUNTS = re.compile(char_class(N_LETTERS, negate=False) + _GAPS + "=" + _GAPS + "[0-9]+|[0-9]+(?:"
                     + char_class(DECIMAL_MARKS, negate=False) + "[0-9]+)?" + char_class(PERCENTS, negate=False))
_AS_NAME = str.maketrans(dict.fromkeys(sorted(DIGITS), NAME_LETTER))   # digits read as the letters of a name
_PER = "per"                                                        # the word that makes the next power a denominator


def _foreign(w: str) -> list[bool]:
    """Per character of *w*: does it stand for any one character (or none) in a word list? A character outside
    ``_PLAIN``, and (r7 round 3) a digit run with a letter on both sides ('bel0w', 'mi11ions', 'BL0Q'; never 'c1')."""
    out = [c not in _PLAIN for c in w]
    i = 0
    while i < len(w):
        if w[i] not in DIGITS:
            i += 1
            continue
        j = i
        while j < len(w) and w[j] in DIGITS:
            j += 1
        if i > 0 and w[i - 1] in LETTERS and j < len(w) and w[j] in LETTERS:
            out[i:j] = [True] * (j - i)
        i = j
    return out


def loose_in(word: str, targets: frozenset[str]) -> bool:
    """*word* (the alphabet's capitals lower-cased) is one of the lowercase ASCII *targets* where every FOREIGN
    character (``_foreign``) stands for any one character (``τo`` is ``to``, ``αbove`` is ``above``, ``bel0w`` is
    ``below``) or for none (a zero-width character inside ``thousands``). Only ever used to RAISE a flag (a word list
    of the value checks), never to read a number: a look-alike or padded spelling must flag like the plain one does."""
    w = casefold_letters(word)
    if w in targets:
        return True
    foreign = _foreign(w)
    if not any(foreign):
        return False
    if "".join(c for c, f in zip(w, foreign, strict=True) if not f) in targets:      # a foreign character inside it
        return True
    return any(len(t) == len(w) and all(x == y or f for x, y, f in zip(w, t, foreign, strict=True)) for t in targets)


def _mask(t: str, data: bool) -> str:
    """*t* with the digits of a sample size ('n = 45') and of a percentage ('95%') read as the letters of a name, 1:1;
    in a DATA cell also every resolved number but a bare power (they are the cell's data, not a scale)."""
    t = _COUNTS.sub(lambda m: m.group().translate(_AS_NAME), t)
    if not data:
        return t
    for tok in tokens(t, POINT, mode="cell"):
        if tok.value is not None and tok.shape != "bare_power":
            t = t[:tok.start] + NAME_LETTER * (tok.end - tok.start) + t[tok.end:]
    return t


def _number_inert(part: str, denominator: bool, data: bool) -> bool:
    toks = tokens(part, POINT, mode="cell")
    if not toks:
        return False
    for tok in toks:
        if tok.value is None:
            return False
        if tok.shape == "bare_power" and not denominator:
            return False
        if not data and (tok.shape == "sci" or P.power_of_ten(tok.canonical)):
            return False                    # a scientific number or a written-out power of ten ('1000', '1,000')
    return True


def _part_inert(part: str, denominator: bool, data: bool) -> bool:
    """One part of a label word is an inert production (see the module docstring)."""
    core = part.rstrip(_STRIP)
    if not core:
        return True
    if all(c in _OPERATORS for c in core) or all(c == STAR for c in core) or core in EXP_MARKERS:
        return True
    if any(c in DIGITS for c in core) and not W.letter_led(core):
        return _number_inert(core, denominator, data)
    named = G.prose(core) if not any(c in DIGITS for c in core) else G.label(core)
    return named and G.unit_powers(core) and not loose_in(core, SCALE_WORDS)


def scale_shapes(label: str, *, data: bool = False) -> bool:
    """True when *label* scales the printed number, or cannot be read for one (see the module docstring); *data*: the
    label is a data cell beside the number, whose resolved numbers are data."""
    t = fold(label)
    if OUTSIDE_RE.search(t) is not None or CID_MARKER.search(t) is not None:
        return True
    prev = ""
    for m in WORD_RE.finditer(_mask(t, data)):
        w, at, after_slash = m.group(), 0, casefold_letters(prev) == _PER or prev[-1:] in SLASH
        for sep in _SPLIT.finditer(w):
            if not _part_inert(w[at:sep.start()], after_slash, data):
                return True
            at, after_slash = sep.end(), sep.group() in SLASH
        if not _part_inert(w[at:], after_slash, data):
            return True
        prev = w
    return False


__all__ = ["loose_in", "scale_shapes"]
