"""r7 owner decision 5: the LINE list is gone; the line rule is DERIVED from the four permitted lists.

A character is readable on a number's line, in its word and in the words one separator away exactly when it is INSIDE,
CONTEXT, GAP or an approved equivalent's source. Every other character is OUTSIDE: it taints its own word, the words one
separator away (a line break included) and every number of its line. The 77 characters of the deleted r6 LINE list were
folded into CONTEXT where they never change the reading of a number beside them (letters, prose marks and footnote
marks, each with corpus evidence in ``alphabet.py``) and are OUTSIDE otherwise (look-alikes of a sign, a dash, a
separator, a multiplier or a fraction, mis-mapped glyphs, code characters, arrows, subscripts and chi).
"""
from __future__ import annotations

import pytest

from app.evidence.numgrammar import alphabet as A
from app.evidence.numgrammar import tokens, unreadable
from app.evidence.numgrammar.equivalents import EQUIV
from tests import evidence_r6_pins as P
from tests.evidence_r6_round1_support import cell

# the deleted r6 LINE list, written once more so the test can say where each character went
FORMER_LINE = (
    0x21, 0x22, 0x23, 0x24, 0x26, 0x3F, 0x40, 0x5C, 0x5F, 0x60, 0x7B, 0x7C, 0x7D,
    0x2018, 0x201C, 0x201D, 0x2014, 0x2026, 0x2022, 0xA9, 0xAE, 0x2122, 0x2020, 0x2021, 0xA7, 0xB6, 0x2016, 0x2032,
    0x2192, 0x2190, 0x2191, 0x2193, 0x221E, 0xBD, *range(0x2080, 0x208C), 0x3BA, 0x3C7,
    0xE9, 0xE8, 0xEB, 0xE2, 0xE1, 0xC1, 0xE3, 0xE6, 0xC5, 0xE4, 0xF6, 0xD6, 0xFC, 0xED, 0xEF,
    0xF3, 0xF8, 0xD8, 0xFA, 0x171, 0xF1, 0x144, 0xE7, 0xC7, 0x107, 0x10D, 0x11F, 0x142, 0x15B,
)
ACCENTED = (0xE9, 0xE8, 0xEB, 0xE2, 0xE1, 0xC1, 0xE3, 0xE6, 0xC5, 0xE4, 0xF6, 0xD6, 0xFC, 0xED, 0xEF,
            0xF3, 0xF8, 0xD8, 0xFA, 0x171, 0xF1, 0x144, 0xE7, 0xC7, 0x107, 0x10D, 0x11F, 0x142, 0x15B)
MARKS = (0x26, 0x40, 0x201C, 0x201D, 0xA9, 0xAE, 0x2122, 0x221E, 0xA7, 0xB6, 0x2020, 0x2021)
TO_CONTEXT = frozenset((*ACCENTED, 0x3BA, *MARKS))
TO_OUTSIDE = frozenset(FORMER_LINE) - TO_CONTEXT


def V(text: str, mode: str = "page") -> list[float | None]:
    return [t.value for t in tokens(text, mode=mode)]


# ------------------------------------------------------------------ the lists ---
def test_the_alphabet_has_no_line_list_and_four_classes():
    assert not [name for name in vars(A) if name == "LINE" or name.startswith("LINE_")]
    assert {A.cls(chr(cp)) for cp in range(0x110000) if not 0xD800 <= cp <= 0xDFFF} == {
        "inside", "context", "gap", "outside"}
    assert len(FORMER_LINE) == 77 and len(TO_CONTEXT) == 42 and len(TO_OUTSIDE) == 35


def test_the_derived_rule_is_the_union_of_the_four_permitted_lists():
    assert A.READABLE == A.INSIDE | A.CONTEXT | A.GAP | frozenset(EQUIV)
    text = "".join(chr(cp) for cp in range(0x110000) if not 0xD800 <= cp <= 0xDFFF)
    flagged = {ord(text[m.start()]) for m in A.OUTSIDE_RE.finditer(text)}
    assert flagged == {cp for cp in map(ord, text) if P.expected_class(cp) == "outside"}


def test_every_former_line_character_is_now_context_or_outside():
    assert {cp for cp in FORMER_LINE if A.cls(chr(cp)) == "context"} == TO_CONTEXT
    assert {cp for cp in FORMER_LINE if A.cls(chr(cp)) == "outside"} == TO_OUTSIDE
    assert all(chr(cp) in A.LETTERS for cp in (*ACCENTED, 0x3BA))       # letters lead a label like any letter
    assert chr(0x3C7) not in A.LETTERS and chr(0x3C7) not in A.CONTEXT     # chi: decision 8, outside


# ------------------------------------------------------------------ outside former-LINE characters ---
@pytest.mark.parametrize("cp", sorted(TO_OUTSIDE))
def test_an_outside_former_line_character_anywhere_on_the_line_taints_every_number_of_it(cp):
    c = chr(cp)
    for text in (f"AUC was 12.5 mg in all subjects {c} ok", f"{c} ok ok AUC 12.5 mg", f"AUC 12.5 mg ok ok ok{c}ok"):
        toks = tokens(text)
        assert toks and all(t.value is None and t.reason == "unresolved_context" for t in toks), (text, toks)
        assert toks[0].detail == f"U+{cp:04X} elsewhere on the same line"
    assert unreadable(f"snippet {c} ok") == (8, f"U+{cp:04X}")


@pytest.mark.parametrize("cp", sorted(TO_OUTSIDE))
def test_an_outside_former_line_character_on_another_line_two_words_away_taints_nothing(cp):
    c = chr(cp)
    assert V(f"{c}\nok 12 mg") == [12.0] and V(f"12 mg\nok {c}") == [12.0]
    assert V(f"ok {c}\n12 mg") == [None] and V(f"12\n{c} ok") == [None]     # one separator away, across the break


# ------------------------------------------------------------------ the characters folded into CONTEXT ---
@pytest.mark.parametrize("cp", sorted(TO_CONTEXT))
def test_a_folded_context_character_beside_a_number_leaves_it_resolved_and_never_joins_it(cp):
    c = chr(cp)
    assert V(f"AUC{c} 30 mg") == [30.0] and V(f"dose {c} 30 mg") == [30.0] and V(f"30 {c} mg") == [30.0]
    assert V(f"Smith{c} reported 12.4 mg and 5 ng {c}") == [12.4, 5.0]
    (glued,) = tokens(f"312{c} mg")                                       # glued to the digits: rejected, never read
    assert glued.value is None and glued.why in ("glued_letters", "glued_symbol")
    assert tokens(f"{c}312 mg") == []                                     # a word led by it is a label
    assert unreadable(f"AUC{c} 30 mg") is None


def test_the_r6_line_examples_of_the_corpus_read_by_the_derived_rule():
    assert V("Final concept map \u2014 linking Data \u2192 Model \u2192 Decision: 20 %") == [None]
    assert set(V("evaluation. Total AUC(0\u2013\u2018) (567.6 mgh/l for CORAL")) == {None}   # a mis-mapped infinity
    assert V("770 51\u2191 551\u2191 002\u2191 921 1 790") == [None] * 7          # a reversed (rotated) table row
    assert V("Mean no. of days\u2021 2.8\u00b11.7 2.9\u00b11.7") == [2.8, 1.7, 2.9, 1.7]
    assert V("Placental abruption\u00a7 41 (7.3) 2.3 (1.7 to 3.1)") == [41.0, 7.3, 2.3, 1.7, 3.1]
    assert V("FO was the original NONMEM method (Beal & Sheiner, 1980). The linearization") == [1980.0]
    assert V("AUC0-\u221e ng h/mL 18 900") == [18.0, 900.0]
    assert V("depression the plateau occurs at 16 mg in opioid-na\u00efve subjects.") == [16.0]


def test_a_table_cell_beside_a_folded_context_cell_reads_and_beside_an_outside_one_does_not():
    rows = [["Parameter", "Value"], ["AUC0\u2013\u221e", "1234"], ["\u2014", "1234"], ["CL\u2020", "5.2"]]
    assert cell(rows, 1, 1, "1234").status == "pass"
    got = cell(rows, 2, 1, "1234")
    assert (got.status, got.reason) == ("fail", "unresolved_context") and "in the previous cell" in got.detail
    assert cell(rows, 3, 1, "5.2").status == "pass"
