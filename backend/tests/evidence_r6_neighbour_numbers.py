"""The NUMBER words of the neighbour-shape sweep's specification (r7 round 4: split out of
``evidence_r6_neighbour_rules`` to keep each file under 400 lines; written independently of the reader, the tokens of
one word aside): a number word, a power of ten with no mantissa (a bare power or a pure power in e-notation), a raised
ten, a ratio-led word, a word that ends in a number, and the split decimal."""
from __future__ import annotations

import re

from app.evidence.numgrammar import tokens
from tests import evidence_r6_oracle as O

DIG = frozenset("0123456789")
LET = O._ASCII_LETTERS | O._GREEK | O._ACCENTED
OPEN, CLOSE = frozenset("(["), frozenset(")]")
TRAIL = CLOSE | frozenset(",;:.'")
QUAL = frozenset("<>≤≥~≈")
STARTERS = OPEN | QUAL | frozenset("=±")


def _lead(w: str) -> int:
    i = 0
    while i < len(w) and (w[i] in OPEN or w[i] in STARTERS):
        i += 1
    return i


def lead_letter(w: str) -> bool:
    return _lead(w) < len(w) and w[_lead(w)] in LET


def number(w: str) -> bool:
    if lead_letter(w) or not any(c in DIG for c in w) or w[0] in "/:":     # r7 round 2: '/15', ':4' are ratio parts
        return False
    body = w[:-1] if w.endswith("-") else w
    got = tokens(body, mode="cell")
    return bool(got) and all(t.value is not None for t in got)


def _has_tokens(w: str) -> bool:
    return any(t.value is not None for t in tokens(w[:-1] if w.endswith("-") else w, mode="cell"))


def _pure(t) -> bool:
    """r7 round 4: a power of ten with no mantissa of its own: a bare power, or a scientific number whose mantissa is a
    power of ten ('1e3', '10E+03', '1.0E+03')."""
    if t.value is None or t.shape not in ("bare_power", "sci"):
        return False
    mant = t.canonical.split("e")[0].lstrip("-").replace(".", "").lstrip("0")
    return t.shape == "bare_power" or re.fullmatch("10*", mant) is not None


def _bare_power(w: str) -> bool:
    """The word starts with a power of ten that has no mantissa ('10^3', '(10^-4)', '10^6/L', r7 round 4 '1e3')."""
    got = tokens(w, mode="cell")
    return bool(got) and _pure(got[0])


def _raised(w: str) -> bool:
    """r7 round 4: the word starts with a raised ten: an unsigned int '10', '103', '109' (of '109/L') at its start, no
    percent sign after it."""
    got = tokens(w, mode="cell")
    return bool(got) and got[0].value is not None and got[0].shape == "int" and got[0].start == 0 \
        and re.fullmatch("10(?:[1-9][0-9]*)?", got[0].canonical) is not None and w[got[0].end:got[0].end + 1] != "%"


def _ratio_led(w: str) -> bool:
    got = tokens(w, mode="cell")
    return bool(got) and got[0].shape == "ratio" and got[0].start == 0


def _number_end(w: str) -> bool:
    """The word ends in a number (closers aside): numeric-led, or a label whose last digit belongs to a token."""
    k = len(w)
    while k > 0 and w[k - 1] in TRAIL | frozenset("%‰*"):
        k -= 1
    if k == 0 or w[k - 1] not in DIG:
        return False
    if not lead_letter(w):
        return True
    got = [t for t in tokens(w, mode="cell") if t.value is not None]
    return bool(got) and got[-1].end == k


def _mark_free(w: str) -> bool:
    """*w* ends in a number (closers aside) whose last token holds no decimal mark: the integer part of a split decimal."""
    got = [t for t in tokens(w, mode="cell") if t.value is not None]
    return _number_end(w) and bool(got) and not got[-1].decimal


def _split_decimal(words: list[str], first: int, last: int) -> bool:
    """r7 round 3: a mark-free number whose word ends in the decimal point, then a word that starts with a digit."""
    after = last + 1 < len(words) and words[last].endswith(".") and _mark_free(words[last][:-1]) \
        and words[last + 1][:1] in DIG
    return after or (first >= 1 and words[first][:1] in DIG and words[first - 1].endswith(".")
                     and _mark_free(words[first - 1][:-1]))
