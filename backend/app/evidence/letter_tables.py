"""The explicit tables AROUND the number reader (r6 round 2; moved out of ``numgrammar`` in r7 round 2 so the reader
package holds only the four permitted lists and the grammar). The reader never reads them: ``tokens`` folds through
``numgrammar.equivalents.fold`` alone, and no module of the ``numgrammar`` package imports this one (r7 round 4: the
scale label check and the loose word match moved out of the package, ``scale_label.py`` and ``word_match.py``).

The snippet locator (``normalize.py``), the unit comparison (``unit_norm.py``), the header key
(``extract_rules.norm_key``) and the group key (``qc_flags``) used to carry NFKC and Unicode case folding of their own,
which equated characters the reader keeps apart. They fold through ``fold`` plus these tables, and a character none of
them lists stays as it is (so it equals nothing else and, in a unit, makes the unit unresolved):

``LIGATURES``        locator only, 1 -> n letters: the seven Latin presentation forms U+FB00..U+FB06. NO hit in the 449
                     parse caches (pdfminer spells them out through the font's ToUnicode map). Kept because a snippet
                     typed with the letters for a page that prints the ligature is the commonest honest drift, and
                     because the expansion is ASCII letters only: it cannot make a number readable, so the locator
                     stays a finder. The owner may delete the table: a ligature page then fails
                     ``snippet_not_on_page`` (fail closed).
``HEADER_SPELLINGS`` header keys only: the infinity sign (U+221E) is the word ``inf`` (``AUC0-<infinity>`` is
                     ``AUC0-inf``; 2 corpus headers; template aliases are written either way).
``CASE_PAIRS``       the 34 capitals of the letters the alphabet lists (ASCII A-Z, Greek Delta Omega Sigma, five
                     accented Latin capitals) and their lower case, 1:1. Explicit pairs, not ``str.casefold``: the
                     Kelvin sign, the long s and the other characters Unicode folds into ASCII letters are not
                     equated with them. (Two lower-case targets, U+03B4 and U+00E5, are not letters of the alphabet: a
                     key or unit holding them compares only with another that folds to them.)
``OCR_SPELLINGS``,   flag word lists only (r7 round 4, ``word_match.loose_in``): the letter shapes an OCR'd
``OCR_LETTERS``,     or mis-mapped text layer confuses (``rn`` for m, ``cl`` for d, ``vv`` for w, a capital I
``OCR_DIGITS``       or a lower-case i for l, c for e, b for h; a digit that ends a letter word, 0 for o and 1
                     for l), applied to BOTH the word on the page and the list word, so 'beIow', 'rnillion',
                     'thousancls', 'betvveen', 'abovc' and 'up t0' raise the flag the plain spelling raises. They
                     only ever RAISE a flag: a number is never read through them.
"""
from __future__ import annotations

# locator only: a presentation-form ligature is its letters (one character -> several: the locator's index map repeats
# the offset). U+FB05 is the long-s ligature and is spelled with a plain s.
LIGATURES: dict[str, str] = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st",
}

# header keys only (extract_rules.norm_key): the infinity sign of AUC0-inf / AUC0-infinity
HEADER_SPELLINGS: dict[str, str] = {"∞": "inf"}

# capital -> lower case, 1:1, for the letters of the alphabet that have a case (ASCII, three Greek, five accented Latin)
CASE_PAIRS: dict[str, str] = {
    **{chr(c): chr(c + 32) for c in range(0x41, 0x5B)},                   # A-Z -> a-z
    "Δ": "δ", "Ω": "ω", "Σ": "σ",         # Delta Omega Sigma
    "Á": "á", "Å": "å", "Ö": "ö",         # A-acute A-ring O-diaeresis
    "Ø": "ø", "Ç": "ç",                             # O-stroke C-cedilla
}
UPPER_LETTERS = frozenset(CASE_PAIRS)                                   # the capitals of the alphabet's letters

# flag word lists only (word_match.loose_in), after the case fold: multi-letter spellings first, then single letters
OCR_SPELLINGS: tuple[tuple[str, str], ...] = (("rn", "m"), ("cl", "d"), ("vv", "w"))
OCR_LETTERS: dict[str, str] = {"i": "l", "e": "c", "h": "b"}               # I and i look like l, c like e, b like h
OCR_DIGITS: dict[str, str] = {"0": "o", "1": "l"}                          # a digit run that ends a letter word ('t0')
_CASE_TABLE = str.maketrans(CASE_PAIRS)


def casefold_letters(text: str) -> str:
    """*text* with the capitals of ``CASE_PAIRS`` in lower case; every other character (a Kelvin sign, a fullwidth or
    mathematical letter, a Cyrillic capital) is unchanged. Same length: offsets are native."""
    return text.translate(_CASE_TABLE)


__all__ = ["CASE_PAIRS", "HEADER_SPELLINGS", "LIGATURES", "OCR_DIGITS", "OCR_LETTERS", "OCR_SPELLINGS", "UPPER_LETTERS",
           "casefold_letters"]
