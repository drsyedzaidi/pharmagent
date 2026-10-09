"""The LOOSE word match of the flag checks (r7 round 4: moved out of the reader package with the scale label check;
the reader never reads a word list or a letter table).

``loose_in(word, targets)`` says whether a word printed on the page is one of the lowercase words of a flag list
(``word_lists.py``: qualifier, bound, range and scale words) spelled the way a damaged text layer spells it. It only
ever RAISES a flag (censored_value, scaled_header); it never makes a number readable. Two explicit readings, applied to
the page word and to every list word alike:

* the OCR SKELETON (``letter_tables.OCR_SPELLINGS``, ``OCR_LETTERS``, ``OCR_DIGITS``): the alphabet's capitals in lower
  case, then 'rn' -> m, 'cl' -> d, 'vv' -> w, then i -> l, e -> c, h -> b, and a digit run that ends a letter word read
  as letters (0 -> o, 1 -> l): 'beIow', 'rnillion', 'thousancls', 'betvveen', 'abovc', 'up t0';
* FOREIGN characters (``_foreign``): a character a word list never holds (a Greek or accented letter, a mark, a
  superscript, a character the reader cannot read) and a digit run with a letter on both sides stand for any one
  character or for none ('τo', 'αbove', 'bel0w', 'mi11ion', a zero-width character inside 'thousands').

Wrapped words: ``join_wrap`` joins a word broken at a line end (a letter, a hyphen or a character the reader cannot read
such as a soft hyphen, a line break, a letter) the way the snippet locator does, so 'thou-' + 'sands' is 'thousands'.
"""
from __future__ import annotations

import re
from functools import lru_cache

from app.evidence.letter_tables import OCR_DIGITS, OCR_LETTERS, OCR_SPELLINGS, casefold_letters
from app.evidence.numgrammar.alphabet import (
    ASCII_LETTERS,
    CLOSERS_BR,
    CONTEXT_PUNCT,
    DIGITS,
    GAP,
    HYPHEN,
    INSIDE,
    LETTERS,
    LF,
    OPENERS_BR,
    OUTSIDE_RE,
    SUP_CHARS,
    char_class,
)

# the characters a word list reads as themselves: ASCII letters, numbers and their punctuation, brackets, the CONTEXT
# punctuation and the gaps; every other character may be a look-alike in a flag word (``_foreign``)
_PLAIN = (INSIDE - SUP_CHARS) | ASCII_LETTERS | OPENERS_BR | CLOSERS_BR | CONTEXT_PUNCT | GAP
_SPELLINGS = re.compile("|".join(re.escape(a) for a, _ in OCR_SPELLINGS))
_SPELL_TO = dict(OCR_SPELLINGS)
_LETTER_TABLE = str.maketrans(OCR_LETTERS)
# a word broken at a line end: a letter, a hyphen (or a character the reader cannot read: a soft hyphen), a line break
_WRAP = re.compile("(?<=" + char_class(LETTERS, negate=False) + ")(?:" + char_class(HYPHEN, negate=False) + "|"
                   + OUTSIDE_RE.pattern + ")" + char_class(GAP - {LF}, negate=False) + "*" + LF
                   + char_class(GAP, negate=False) + "*(?=" + char_class(LETTERS, negate=False) + ")")


def skeleton(word: str) -> str:
    """The OCR skeleton of *word* (see the module docstring)."""
    w = _SPELLINGS.sub(lambda m: _SPELL_TO[m.group()], casefold_letters(word)).translate(_LETTER_TABLE)
    k = len(w)
    while k > 0 and w[k - 1] in DIGITS:
        k -= 1
    if 0 < k < len(w) and w[k - 1] in LETTERS and all(c in OCR_DIGITS for c in w[k:]):
        w = w[:k] + "".join(OCR_DIGITS[c] for c in w[k:]).translate(_LETTER_TABLE)
    return w


@lru_cache(maxsize=64)
def _skeletons(targets: frozenset[str]) -> frozenset[str]:
    return frozenset(skeleton(t) for t in targets)


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
    """*word* is one of the lowercase ASCII *targets*, read through the OCR skeleton, where every FOREIGN character
    stands for any one character or for none (see the module docstring). Only ever used to RAISE a flag."""
    w = casefold_letters(word)
    if w in targets:
        return True
    sk, skeletons = skeleton(w), _skeletons(targets)
    if sk in skeletons:
        return True
    foreign = _foreign(sk)
    if not any(foreign):
        return False
    if "".join(c for c, f in zip(sk, foreign, strict=True) if not f) in skeletons:     # a foreign character inside it
        return True
    return any(len(t) == len(sk) and all(x == y or f for x, y, f in zip(sk, t, foreign, strict=True))
               for t in skeletons)


def join_wrap(text: str) -> str:
    """*text* with every word broken at a line end joined (the hyphen, or the unreadable character, and the break
    dropped): 'thou-' + line break + 'sands' is 'thousands'. Offsets move: callers read words, never positions."""
    return _WRAP.sub("", text) if LF in text else text


__all__ = ["join_wrap", "loose_in", "skeleton"]
