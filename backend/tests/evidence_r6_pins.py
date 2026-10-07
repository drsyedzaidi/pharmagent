"""The five literal sets of the numeric reader, written a SECOND time as bare code points (r6 spec).

``app/evidence/numgrammar/alphabet.py`` is the first copy. ``tests/test_evidence_r6_sets.py`` compares the two
entry by entry and prints this table, and the exhaustive classification test classifies every code point from THIS
copy: an edit in one place and not the other fails. An addition to the alphabet is a visible diff here, with its
code point, in the same commit (the admission rule: one corpus hit on a digit-bearing line in ordinary prose, never
a mis-mapped glyph, plus a regenerated oracle golden file).
"""
from __future__ import annotations

# ---------------------------------------------------------------- INSIDE (45) ---
INSIDE_CPS = (
    *range(0x30, 0x3A),                       # digits 0-9
    0x2B, 0x2D,                               # + -
    0x2E, 0x2C, 0x27, 0x2009, 0xB7,           # . , ' THIN SPACE, MIDDLE DOT
    0x65, 0x45,                               # e E
    0x78, 0x58, 0x2A, 0xD7,                   # x X * MULTIPLICATION SIGN
    0x5E,                                     # ^
    0x2070, 0xB9, 0xB2, 0xB3, *range(0x2074, 0x207A),   # superscript digits 0-9
    0x207A, 0x207B,                           # superscript plus, minus
    0x25, 0x2030,                             # % PER MILLE
    0xB1,                                     # PLUS-MINUS
    0x3C, 0x3E, 0x2264, 0x2265, 0x7E, 0x2248,  # < > <= >= ~ ~=
)
# ------------------------------------------------------------------ GAP (4) ---
GAP_CPS = (0x20, 0x09, 0x0A, 0x0D)
# ---------------------------------------------------------------- CONTEXT (74) ---
CONTEXT_CPS = (
    *[c for c in range(0x41, 0x5B) if c not in (0x45, 0x58)],      # A-Z without E X
    *[c for c in range(0x61, 0x7B) if c not in (0x65, 0x78)],      # a-z without e x
    0x3BC, 0x3C9, 0x3B2, 0x3B1, 0x3C4, 0x3BB, 0x3B7, 0x3B8,       # mu omega beta alpha tau lambda eta theta
    0x3B5, 0x3B3, 0x3C3, 0x3C7, 0x3C1, 0x3C0, 0x394, 0x3A9, 0x3A3,  # epsilon gamma sigma chi rho pi Delta Omega Sigma
    0x28, 0x29, 0x5B, 0x5D,                   # ( ) [ ]
    0x3B, 0x3A, 0x2F, 0x3D, 0xB0,             # ; : / = DEGREE SIGN
)
# ------------------------------------------------------------------ LINE (76) ---
LINE_CPS = (
    0x21, 0x22, 0x23, 0x24, 0x26, 0x3F, 0x40, 0x5C, 0x5F, 0x60, 0x7B, 0x7C, 0x7D,        # ! " # $ & ? @ \ _ ` { | }
    0x2018, 0x201C, 0x201D, 0x2014, 0x2026, 0x2022,                                     # quotes, em dash, ellipsis, bullet
    0xA9, 0xAE, 0x2122, 0x2020, 0x2021, 0xA7, 0xB6, 0x2016, 0x2032,                      # (c) (R) TM dagger ddagger sect para ||  prime
    0x2192, 0x2190, 0x2191, 0x2193, 0x221E, 0xBD,                                       # arrows, infinity, one half
    *range(0x2080, 0x208C),                   # subscripts 0-9, plus, minus
    0x3BA,                                    # kappa
    0xE9, 0xE8, 0xEB, 0xE2, 0xE1, 0xC1, 0xE3, 0xE6, 0xC5, 0xE4, 0xF6, 0xD6, 0xFC, 0xED, 0xEF,
    0xF3, 0xF8, 0xD8, 0xFA, 0x171, 0xF1, 0x144, 0xE7, 0xC7, 0x107, 0x10D, 0x11F, 0x142, 0x15B,   # accented Latin
)
# ------------------------------------------------------------- EQUIVALENTS (10) ---
EQUIV_PIN = {
    0x2212: 0x2D, 0x2013: 0x2D, 0x2010: 0x2D, 0x2011: 0x2D,     # minus sign, en dash, hyphen, non-breaking hyphen -> -
    0x2019: 0x27,                                               # right single quotation mark -> '
    0x202F: 0x2009, 0xA0: 0x2009,                               # narrow no-break space, no-break space -> thin space
    0xB5: 0x3BC,                                                # micro sign -> mu
    0x223C: 0x7E,                                               # tilde operator -> ~
    0x25E6: 0xB0,                                               # white bullet -> degree sign
}
COUNTS = {"inside": 45, "gap": 4, "context": 74, "line": 76, "equiv": 10}


def expected_class(cp: int) -> str:
    """The class of one code point, from this copy of the sets (an equivalent classifies as its target)."""
    cp = EQUIV_PIN.get(cp, cp)
    if cp in INSIDE_SET:
        return "inside"
    if cp in GAP_SET:
        return "gap"
    if cp in CONTEXT_SET:
        return "context"
    if cp in LINE_SET:
        return "line"
    return "outside"


INSIDE_SET, GAP_SET = frozenset(INSIDE_CPS), frozenset(GAP_CPS)
CONTEXT_SET, LINE_SET = frozenset(CONTEXT_CPS), frozenset(LINE_CPS)
