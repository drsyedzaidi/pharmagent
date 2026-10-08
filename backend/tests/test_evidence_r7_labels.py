"""r7 owner decision 3: digits embedded in label words (``AUC0-24``, ``CYP2D6``, ``t1/2``, ``C24h``) are LABELS.

They never resolve as numbers, they never taint a neighbour by themselves (no mixed-number whole part, no spaced-ratio
whole part, no range end, no 'after a number' reading of a bracket), and a record that cites a digit inside a label
fails with a clear reason (``unresolved_shape``, the label sentence). The verifier asks ``tokens(..., labels=True)`` for
the label expressions; every other caller reads exactly what it read before.
"""
from __future__ import annotations

import pytest

from app.evidence import models as M
from app.evidence.numgrammar import tokens
from tests.evidence_r5b_support import check

LABELS = ("AUC0-24", "CYP2D6", "t1/2", "C24h", "mL/min/1.73m2", "m12", "AUC0–24h", "(CYP3A4)", "Level2-Acero,1")


def outside_first_word(text: str) -> list[tuple]:
    """The reading of everything after the first word (offsets relative to the text)."""
    cut = text.index(" ")
    return [(t.start, t.end, t.value, t.shape, t.qualified, t.ranged, t.parts) for t in tokens(text) if t.start > cut]


def digits_to_q(word: str) -> str:
    return "".join("q" if c in "0123456789" else c for c in word)


@pytest.mark.parametrize("label", LABELS)
def test_the_digits_of_a_label_never_resolve(label):
    assert tokens(label) == [] and tokens(f"the {label} was") == []
    (lab,) = tokens(label, labels=True)
    assert (lab.start, lab.end, lab.value, lab.shape, lab.why) == (0, len(label), None, "unresolved_shape:label",
                                                                   "label")
    assert lab.parts and all(p == int(p) for p in lab.parts)


@pytest.mark.parametrize("label", LABELS)
@pytest.mark.parametrize("tail", ["3/4", "12 / 15", "-0.5", "[12]", "- 12", "-15 mg", "– 12", "[3] 5", "x 4"])
def test_a_label_never_taints_its_neighbour_by_its_digits(label, tail):
    text = f"{label} {tail}"
    assert outside_first_word(text) == outside_first_word(f"{digits_to_q(label)} {tail}"), text


def test_the_neighbour_readings_decision_3_changes():
    assert [(t.value, t.shape) for t in tokens("m12 3/4")] == [(0.75, "ratio")]          # was mixed_number (r7 w1)
    assert [(t.value, t.shape) for t in tokens("AUC0-24 3/4")] == [(0.75, "ratio")]
    assert [(t.value, t.shape) for t in tokens("CYP2D6 12 / 15")] == [(0.8, "ratio")]
    assert [(t.value, t.ranged) for t in tokens("AUC0-24 -0.5")] == [(-0.5, False)]
    assert [t.shape for t in tokens("CYP2D6 [38]; 20 trials")] == ["unresolved_shape:citation", "int"]
    assert [t.shape for t in tokens("AUC0-24[3]")] == ["unresolved_shape:citation"]
    assert [t.shape for t in tokens("(CYP2D6)[4]")] == ["unresolved_shape:citation"]


def test_a_number_before_its_neighbour_keeps_every_fail_closed_reading():
    assert [t.why for t in tokens("12 3/4")] == ["mixed_number"]
    assert [t.why for t in tokens("1O5 3/4")] == ["glued_letters", "mixed_number"]            # a rejected number word
    assert [t.why for t in tokens("AUC=12 3/4")] == ["mixed_number"]                          # 12 is a token, not a label
    assert [t.why for t in tokens("4 12 / 15")][-1] == "fraction"
    assert [(t.value, t.ranged) for t in tokens("12 -15")] == [(12.0, True), (-15.0, True)]
    assert [t.value for t in tokens("data(n=7)[48]")] == [7.0, 48.0]                       # ends in the number 7
    assert [t.value for t in tokens("CYP3A4 Vmax 89 [46] and")] == [89.0, 46.0]


def test_label_expressions_are_added_after_the_reading_and_change_nothing_else():
    for text in ("AUC0-24 was 12.5 h", "CYP2D6 [12] 5", "AUC0-24=12.5", "2 x CYP2D6", "t1/2 3/4 and C24h -5",
                 "AUC(0-24) 1234", "BMI≥35kg/m", "Cmax—AUC0-24 12", "x12 y3 (cid:3) CYP2D6 4"):
        plain, labelled = tokens(text), tokens(text, labels=True)
        assert [t for t in labelled if t.why != "label"] == plain, text
        spans = [(t.start, t.end) for t in labelled]
        assert spans == sorted(spans) and all(a[1] <= b[0] for a, b in zip(spans, spans[1:], strict=False)), text
    got = tokens("AUC0-24=12.5", labels=True)
    assert [("AUC0-24=12.5"[t.start:t.end], t.why) for t in got] == [("AUC0-24=", "label"), ("12.5", "")]
    assert [t.why for t in tokens("2 x CYP2D6", labels=True)] == ["multiplier_operand"]   # inside a wider expression


# ------------------------------------------------------------------ the verify() path ---
@pytest.mark.parametrize("value, snippet, page, label", [
    ("24", "AUC0-24 was 12.5", "The AUC0-24 was 12.5 ng/mL.", "AUC0-24"),
    ("6", "by CYP2D6 mostly", "Metabolised by CYP2D6 mostly.", "CYP2D6"),
    ("1.73", "mL/min/1.73m2", "GFR 90 mL/min/1.73m2 in adults", "mL/min/1.73m2"),
    ("24", "24", "The AUC0-24 was 7.", "AUC0-24"),
    ("2", "t1/2", "Its t1/2 was 11 h.", "t1/2"),
])
def test_a_record_citing_a_digit_inside_a_label_fails_with_the_label_sentence(value, snippet, page, label):
    res = check(value, snippet, page)
    assert (res.status, res.reason) == ("fail", M.REASON_UNRESOLVED_SHAPE)
    assert f"is part of the label {label!r}" in res.detail and "never a value" in res.detail


def test_a_table_cell_citing_a_label_digit_fails_with_the_label_sentence():
    res = check("24", "24", "AUC0-24", cell=True)
    assert (res.status, res.reason) == ("fail", M.REASON_UNRESOLVED_SHAPE) and "'AUC0-24'" in res.detail


def test_a_record_citing_the_number_beside_a_label_passes():
    assert check("12.5", "AUC0-24 was 12.5", "The AUC0-24 was 12.5 ng/mL.").number == 12.5
    assert check("3/4", "m12 3/4", "dose m12 3/4 tablet").status == "pass"
    assert check("-0.5", "AUC0-24 -0.5", "change AUC0-24 -0.5 units").number == -0.5


def test_label_expressions_and_the_citation_rule_stay_linear_on_a_hostile_page():
    import time
    t0 = time.perf_counter()
    many = tokens("CYP2D6 [12] " * 20_000, labels=True)
    glued = tokens("a1[1]" * 40_000, labels=True)
    chain = tokens("AUC=0-24[1]" * 20_000, labels=True)
    assert time.perf_counter() - t0 < 6.0                           # quadratic paths took minutes
    assert len(many) == 40_000 and {t.why for t in many} == {"label", "citation"}
    assert len(glued) == 80_000 and len(chain) == 60_000
