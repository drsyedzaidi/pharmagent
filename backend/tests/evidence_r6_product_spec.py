"""The PRODUCT production of r7 round 3, written again for the neighbour-shape sweep (independent of the reader's
``numgrammar/product.py`` and ``product_window.py``; its terminals come from the second copy of the lists).

    product = count MULT amount      count: the last token of the word before the multiplier (a prose word between them
                                     is the count's unit: '8 weeks x 3'); amount: the first token of the word after it
    count, amount: plain             an int or a decimal ('1.018' read as a decimal too), never grouped, scientific, a
                                     power or a ratio; the count unsigned
    amount: no raised ten            its integer digits are not '10' followed by a digit 1-9 or by nothing
    amount: a power of ten           written out ('100', '1000'): only after a count of one or two digits [1-9][0-9]? and
                                     before a unit production, glued or as the next word
"""
from __future__ import annotations

import re

from app.evidence.numgrammar import tokens
from tests import evidence_r6_pins as P

PLAIN = frozenset({"int", "decimal", "foreign_decimal"})
CLOSERS = frozenset(")],;:.'")
_ALT = "|".join(re.escape(t) for t in sorted(P.UNIT_PREFIX_PIN | P.UNIT_BASE_PIN | P.UNIT_PREFIXED_BASE_PIN, key=len,
                                                 reverse=True))
_PRE = "(?:" + "|".join(re.escape(t) for t in sorted(P.UNIT_PREFIX_PIN, key=len, reverse=True)) + ")"
_BASE = "(?:" + "|".join(re.escape(t) for t in sorted(P.UNIT_BASE_PIN, key=len, reverse=True)) + ")"
_PREFIXED = "(?:" + "|".join(re.escape(t) for t in sorted(P.UNIT_PREFIXED_BASE_PIN, key=len, reverse=True)) + ")"
ATOM = "(?:" + _PRE + "?" + _BASE + "|" + _PRE + _PREFIXED + ")"
_UNIT_LED = re.compile(ATOM + "(?=$|[/·×⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻)\\],;:.'^+-])")
_GLUED_UNIT = re.compile(ATOM + "(?:[23](?![0-9])|\\^[-+]?[0-9]{1,2}(?![0-9])|[⁺⁻]?[⁰¹²³⁴⁵⁶⁷⁸⁹]{1,2})?"
                         + "(?:(?:/|per)" + ATOM + ")*(?=$|[)\\],;:.'](?![0-9])|[(\\[])")


def plain(t) -> bool:
    return t.value is not None and (t.shape in PLAIN or (t.shape == "style_dependent" and bool(t.decimal)))


def _digits(canonical: str) -> str:
    return canonical[1:] if canonical[:1] == "-" else canonical


def power_of_ten(canonical: str) -> bool:
    """A written-out power of ten that scales: never one."""
    return re.fullmatch("10*", _digits(canonical).replace(".", "").lstrip("0")) is not None and abs(float(canonical)) != 1


def raised_ten(canonical: str) -> bool:
    return re.fullmatch("10(?:[1-9][0-9]*)?", _digits(canonical).split(".")[0]) is not None


def count_tok(w: str):
    """The last token of *w* when only closers follow it."""
    got = tokens(w, mode="cell")
    if not got or got[-1].value is None or not all(c in CLOSERS for c in w[got[-1].end:]):
        return None
    return got[-1]


def amount_tok(w: str):
    """The first token of *w* when it starts the word (one opener aside; a dangling range dash read as the reader's
    number word does: '0-' before the other end)."""
    got = tokens(w[:-1] if w.endswith("-") else w, mode="cell")
    if not got or got[0].value is None or got[0].start > (1 if w[:1] in "([" else 0):
        return None
    return got[0]


def unit_after(words: list[str], k: int, tok) -> bool:
    w = words[k]
    if _GLUED_UNIT.match(w, tok.end) and tok.end < len(w):
        return True
    return all(c in CLOSERS for c in w[tok.end:]) and k + 1 < len(words) and _UNIT_LED.match(words[k + 1]) is not None


def count_index(words: list[str], j: int, prose) -> int:
    """The word holding the count when a multiplier follows word *j*: *j*, or the word before a prose unit word."""
    if 0 <= j < len(words) and count_tok(words[j]) is None and not any(c in "0123456789" for c in words[j]) \
            and not all(c in ")](,;:.'/=" for c in words[j]) and prose(words[j]):
        return j - 1
    return j


def product_ok(words: list[str], ck: int, ak: int) -> bool:
    if not (0 <= ck < len(words) and 0 <= ak < len(words)):
        return False
    c, a = count_tok(words[ck]), amount_tok(words[ak])
    if c is None or a is None or not plain(c) or not plain(a) or c.canonical.startswith("-"):
        return False
    if raised_ten(a.canonical):
        return False
    return not power_of_ten(a.canonical) or (re.fullmatch("[1-9][0-9]?", c.canonical) is not None
                                             and unit_after(words, ak, a))


_POWER_WORD = re.compile("10(?:\\^[-+]?[0-9]{1,4}|[⁺⁻]?[⁰¹²³⁴⁵⁶⁷⁸⁹]{1,4})[)\\],;:.']*")


def power_word(words: list[str], k: int) -> bool:
    """r7 round 3: word *k* is the power of a multiplier-power token ('4.5 x 10^9'): a plain mantissa that ends its word,
    a lone multiplier, then the power: it belongs to that scientific number."""
    if not (k >= 2 and len(words[k - 1]) == 1 and words[k - 1] in "xX*×·" and _POWER_WORD.fullmatch(words[k])):
        return False
    m = count_tok(words[k - 2])
    return m is not None and m.shape in PLAIN | {"style_dependent"} and m.end == len(words[k - 2])


def refused_amount(words: list[str], k: int, prose) -> bool:
    """Word *k* stands right after a lone multiplier and is no product's amount (nor a power word of a scientific
    number)."""
    if not (k >= 1 and len(words[k - 1]) == 1 and words[k - 1] in "xX*×·") or power_word(words, k):
        return False
    return not product_ok(words, count_index(words, k - 2, prose), k)
