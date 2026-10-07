"""Which code points the per-commit tier of the r6 exhaustive tests samples (the full run takes every one).

``stratified_outside(n)`` is a deterministic subset of the outside code points: all of C0 and Latin-1, every Cc, Cf, Zs,
Zl, Zp and Me, a stride through every other general category, and each block that holds a listed or look-alike character.
"""
from __future__ import annotations

import unicodedata
from functools import cache

from tests import evidence_r6_pins as P


def class_of(cp: int) -> str:
    return P.expected_class(cp)


@cache
def non_outside_cps() -> list[int]:
    """Every listed code point and every equivalent's source (209 + 10 sources already counted among the listed)."""
    return [cp for cp in range(0x110000) if not 0xD800 <= cp <= 0xDFFF and class_of(cp) != "outside"]


@cache
def outside_cps() -> list[int]:
    return [cp for cp in range(0x110000) if not 0xD800 <= cp <= 0xDFFF and class_of(cp) == "outside"]


@cache
def categories() -> dict[int, str]:
    return {cp: unicodedata.category(chr(cp)) for cp in outside_cps()}


# every k-th code point of a general category (k = 1 takes them all): the rest of the outside space is identical by
# construction and is covered by the exhaustive run, not by the per-commit subset
_STRIDE = {"Cc": 1, "Cf": 1, "Zs": 1, "Zl": 1, "Zp": 1, "Me": 1, "Nd": 3, "No": 3, "Nl": 3, "Mc": 5, "Mn": 16,
           "Co": 4096, "Cn": 4096, "Cs": 1, "Lu": 701, "Ll": 701, "Lt": 1, "Lm": 29, "Lo": 701, "Pc": 1, "Pd": 1,
           "Ps": 5, "Pe": 5, "Pi": 1, "Pf": 1, "Po": 11, "Sm": 17, "Sc": 1, "Sk": 3, "So": 53}
_RANGES = [(0x2000, 0x20FF, 1), (0x2100, 0x218F, 1), (0xFE00, 0xFE6F, 1), (0xFF00, 0xFFEF, 7), (0x1D400, 0x1D7FF, 29),
           (0xE0000, 0xE007F, 3), (0xE0100, 0xE01EF, 9), (0x2460, 0x24FF, 5), (0x2700, 0x27BF, 7), (0x2190, 0x22FF, 11),
           (0x0300, 0x036F, 5), (0x3000, 0x303F, 1), (0x2800, 0x28FF, 13), (0x0370, 0x04FF, 4), (0x0600, 0x06FF, 5)]


# the look-alikes and invisible characters of the r2-r5b attack tests: always in the subset
ALWAYS = (0xFFFC, 0xFFFD, 0xFFF9, 0xFFFA, 0xFFFB, 0x2800, 0x3164, 0xFFA0, 0x115F, 0x1160, 0x17B4, 0x17B5, 0x180E,
          0x061C, 0x034F, 0xFE0F, 0xFE00, 0xE0001, 0xE0020, 0xE007F, 0xFF11, 0x0661, 0x2460, 0x00BE, 0x2012, 0xFE63,
          0xFF0D, 0x02D7, 0x2043, 0x22C5, 0x2219, 0x2A2F, 0x2715, 0x2217, 0x0445, 0x0425, 0x2031, 0xFE62, 0xFF0B,
          0x2027, 0x02D9, 0x066B, 0x2024, 0x02BC, 0x00B4, 0x00A8, 0x0300, 0x0338, 0x20D0, 0x20E3, 0x2155, 0x2189,
          0x2070 + 0x10, 0x00F0, 0x00DE, 0x00FE, 0x00BC, 0x00A5, 0x00A3, 0x00AA, 0x014B, 0x0263, 0x2206, 0x2191 + 0x1,
          0x2264 + 0x20, 0x2A7D, 0x2A7E, 0x2243, 0x2272, 0x2273, 0x226E, 0x226F, 0x2270, 0x2271, 0x2241, 0x2249,
          0x226A, 0x226B, 0x2260, 0x00F7, 0x221A, 0x2211, 0x2202, 0x2194, 0x25CF, 0x2605, 0x2713, 0x2500, 0x2503,
          0x1D44E, 0x20AC, 0x1D62, 0xFB00, 0xFB06, 0x200B, 0x200C, 0x200D, 0x2060, 0x00AD, 0x2028, 0x2029, 0x0085)


def stratified_outside(limit: int) -> list[int]:
    """A deterministic sample of outside code points: all of C0 and Latin-1, every Cc, Cf, Zs, Zl, Zp and Me, a
    stride through every other general category, and each block that holds a listed or look-alike character."""
    cats = categories()
    picked: set[int] = {cp for cp in range(0x100) if cp in cats}
    by_cat: dict[str, list[int]] = {}
    for cp, cat in cats.items():
        by_cat.setdefault(cat, []).append(cp)
    for cat, cps in by_cat.items():
        picked |= set(cps[::_STRIDE.get(cat, 97)])
    for lo, hi, step in _RANGES:
        picked |= {cp for cp in range(lo, hi + 1, step) if cp in cats}
    picked |= {cp for cp in ALWAYS if cp in cats}
    must = {cp for cp in picked if cats[cp] in ("Cc", "Cf", "Zs", "Zl", "Zp", "Me") or cp < 0x100} | \
        {cp for cp in ALWAYS if cp in cats}
    rest = sorted(picked - must)
    room = max(limit - len(must), 0)
    if len(rest) > room:
        rest = rest[:: -(-len(rest) // max(room, 1))][:room]
    return sorted(must | set(rest))
