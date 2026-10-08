"""The four literal lists of the numeric reader, written a SECOND time as bare code points (r6 spec; r7 deleted LINE).

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
# ---------------------------------------------------------------- CONTEXT (115) ---
# r7 (owner decision 5): there is no LINE list. 42 of its 77 characters moved here (letters, prose marks and footnote
# marks that never change the reading of a number beside them); the other 35 are outside like every unlisted character.
CONTEXT_CPS = (
    *[c for c in range(0x41, 0x5B) if c not in (0x45, 0x58)],      # A-Z without E X
    *[c for c in range(0x61, 0x7B) if c not in (0x65, 0x78)],      # a-z without e x
    0x3BC, 0x3C9, 0x3B2, 0x3B1, 0x3C4, 0x3BB, 0x3B7, 0x3B8,       # mu omega beta alpha tau lambda eta theta
    0x3B5, 0x3B3, 0x3C3, 0x3C1, 0x3C0, 0x394, 0x3A9, 0x3A3,        # epsilon gamma sigma rho pi Delta Omega Sigma
    0x3BA,                                    # kappa (r7, from LINE; chi is outside)
    0x28, 0x29, 0x5B, 0x5D,                   # ( ) [ ]
    0x3B, 0x3A, 0x2F, 0x3D, 0xB0,             # ; : / = DEGREE SIGN
    0xE9, 0xE8, 0xEB, 0xE2, 0xE1, 0xC1, 0xE3, 0xE6, 0xC5, 0xE4, 0xF6, 0xD6, 0xFC, 0xED, 0xEF,
    0xF3, 0xF8, 0xD8, 0xFA, 0x171, 0xF1, 0x144, 0xE7, 0xC7, 0x107, 0x10D, 0x11F, 0x142, 0x15B,   # accented Latin (r7)
    0x26, 0x40, 0x201C, 0x201D, 0xA9, 0xAE, 0x2122, 0x221E,           # & @ quotes (c) (R) TM infinity (r7)
    0xA7, 0xB6, 0x2020, 0x2021,               # section, pilcrow, dagger, double dagger: footnote marks (r7)
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
COUNTS = {"inside": 45, "gap": 4, "context": 115, "equiv": 10}


# ------------------------------------------- the unit production's terminals (r7, owner decision 2) ---
def _terms(*cps: tuple[int, ...]) -> frozenset[str]:
    return frozenset("".join(map(chr, t)) for t in cps)


UNIT_PREFIX_PIN = _terms((0x6D,), (0x3BC,), (0x6E,), (0x6B,), (0x63,))                  # m mu n k c
UNIT_BASE_PIN = _terms((0x67,), (0x68,), (0x4C,), (0x79,), (0x4D,), (0x6D, 0x69, 0x6E), (0x6D,), (0x77,),
                       (0x6D, 0x6D, 0x48, 0x67), (0x48, 0x7A), (0x49, 0x55))        # g h L y M min m w mmHg Hz IU
UNIT_PREFIXED_BASE_PIN = _terms((0x6C,))                                              # l (only after a prefix: ml)
# the 42 characters of the deleted LINE list that are CONTEXT since r7 (the layer-B sweeps keep checking them as they did)
FOLDED_FROM_LINE = frozenset((0x3BA, 0xE9, 0xE8, 0xEB, 0xE2, 0xE1, 0xC1, 0xE3, 0xE6, 0xC5, 0xE4, 0xF6, 0xD6, 0xFC, 0xED,
                              0xEF, 0xF3, 0xF8, 0xD8, 0xFA, 0x171, 0xF1, 0x144, 0xE7, 0xC7, 0x107, 0x10D, 0x11F, 0x142,
                              0x15B, 0x26, 0x40, 0x201C, 0x201D, 0xA9, 0xAE, 0x2122, 0x221E, 0xA7, 0xB6, 0x2020, 0x2021))


def expected_class(cp: int) -> str:
    """The class of one code point, from this copy of the sets (an equivalent classifies as its target)."""
    cp = EQUIV_PIN.get(cp, cp)
    if cp in INSIDE_SET:
        return "inside"
    if cp in GAP_SET:
        return "gap"
    if cp in CONTEXT_SET:
        return "context"
    return "outside"                          # the derived line rule (r7): every character of no list


INSIDE_SET, GAP_SET = frozenset(INSIDE_CPS), frozenset(GAP_CPS)
CONTEXT_SET = frozenset(CONTEXT_CPS)
