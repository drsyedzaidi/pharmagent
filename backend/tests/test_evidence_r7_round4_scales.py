"""r7 round 4: scales the text layer detached from their number (the round-4 audit, findings 6, 7, 15 and 16).

* a RAISED TEN (an integer whose digits are 10 and then the exponent digits the text layer flattened: '10', '103',
  '109/L', '10−3', '10 3') right after a number is a detached scale, refused like the bare power '1.2 10^3' (the
  multiplier glyph was dropped and the superscript set at full size): in running text and in a table cell's window;
* a PURE power of ten in e-notation ('10E+03', '1E+03', '1e3') beside a number is refused like '10^3';
* a power of ten printed without a mantissa one prose word after a number ('1.2 times 10^3', '1.2 by 1e3') is no value,
  and the number before the prose word is unresolved too;
* a header or unit label holding a flattened raised ten ('WBC (109/L)') raises scaled_header like '10⁹/L'.
"""
from __future__ import annotations

import pytest

from app.evidence.numgrammar import tokens
from app.evidence.scale_label import scale_shapes
from tests.evidence_r6_round1_support import cell, run

PAGE = "Clearance 12.4 L/h, volume 45.6 L, half-life 3.5 h, ka 1.25 /h. "


def _resolved(text: str, literal: str, nth: int = 0) -> bool:
    s = -1
    for _ in range(nth + 1):
        s = text.index(literal, s + 1)
    return any(t.value is not None and t.start < s + len(literal) and s < t.end for t in tokens(text))


# ------------------------------------------------------------------- finding 6: a dropped multiplier ---
@pytest.mark.parametrize("text,literal", (
    ("Count was 1.2 10−3 M", "1.2"), ("WBC 2.5 109/L", "2.5"), ("Count was 1.2 10 3 cells/mL", "1.2"),
    ("Count was 1.2 103 cells/mL", "1.2"), ("Count was 1.2 10^3 cells/mL", "1.2"), ("WBC 2.5 × 109/L", "2.5"),
    ("Count was 1.2 10 cells", "1.2"), ("Count was 1.2 1012/L", "1.2")))
def test_a_raised_ten_after_a_number_is_a_detached_scale(text, literal):
    assert not _resolved(text, literal), (text, tokens(text))


@pytest.mark.parametrize("text,literal", (("Count was 1.2 10−3 M", "3"), ("Count was 1.2 10 3 cells/mL", "3"),
                                          ("Count was 1.2 10 cells", "10"), ("WBC 2.5 109/L", "109")))
def test_the_raised_ten_and_its_exponent_are_no_value_either(text, literal):
    assert not _resolved(text, literal, 0 if literal != "3" else 0), (text, tokens(text))


@pytest.mark.parametrize("p,s,v", (("Count was 1.2 10−3 M", "1.2 10−3 M", "1.2"), ("WBC 2.5 109/L", "2.5 109/L", "2.5"),
                                   ("Count was 1.2 10 3 cells/mL", "1.2 10 3 cells/mL", "1.2"),
                                   ("Count was 1.2 103 cells/mL", "1.2 103 cells/mL", "1.2")))
def test_a_dropped_multiplier_fails_end_to_end(p, s, v):
    assert run(p, s, v, field="AUC").status == "fail"


def test_a_raised_ten_in_the_next_cell_refuses_the_value():
    assert cell([["Test", "Value", "Unit"], ["WBC", "2.5", "109/L"]], 1, 1, "2.5", field="AUC").status == "fail"
    assert cell([["Test", "Value", "Unit"], ["WBC", "2.5", "10⁹/L"]], 1, 1, "2.5", field="AUC").status == "fail"


@pytest.mark.parametrize("text,literal", (("Cmax 12.5 10.2 15.3", "12.5"), ("Cmax 12.5 10.2 15.3", "10.2"),
                                          ("doses 12 100 150", "12"), ("Arm 10 12", "12"), ("Arm 10 12", "10"),
                                          ("values 1.2 1100", "1.2"), ("values 1.2, 10 and 12", "1.2")))
def test_a_decimal_a_written_out_hundred_and_a_ten_before_a_number_still_read(text, literal):
    assert _resolved(text, literal), (text, tokens(text))


# --------------------------------------------------------------- finding 16: e-notation pure powers ---
@pytest.mark.parametrize("text,literal", (("The titre was 1.2 10E+03 copies/mL.", "1.2"),
                                          ("The titre was 1.2 10E+03 copies/mL.", "10E+03"),
                                          ("The titre was 1.2 1E+03 copies/mL.", "1.2"), ("titre 1.2 10e3", "1.2"),
                                          ("titre 1.2 1e3", "1.2"), ("titre 1.2 1.0E+03", "1.2"),
                                          ("titre 1.2 1e3", "1e3")))
def test_a_pure_power_of_ten_in_e_notation_beside_a_number_is_a_detached_scale(text, literal):
    assert not _resolved(text, literal), (text, tokens(text))


def test_a_scientific_number_that_is_no_pure_power_is_an_ordinary_neighbour():
    assert _resolved("titre 1.2 5e3 copies", "1.2") and _resolved("titre 1.2 5e3 copies", "5e3")


def test_an_e_notation_detached_scale_fails_end_to_end():
    assert run(PAGE + "The titre was 1.2 10E+03 copies/mL.", "1.2 10E+03", "1.2", field="AUC").status == "fail"
    assert run(PAGE + "The titre was 1.2 10E+03 copies/mL.", "10E+03 copies", "10E+03", field="AUC").status == "fail"
    assert run(PAGE + "The titre was 1.2 1E+03 copies/mL.", "1.2 1E+03", "1.2", field="AUC").status == "fail"
    assert run(PAGE + "The titre was 1.2 10^3 copies/mL.", "1.2 10^3", "1.2", field="AUC").status == "fail"


# ------------------------------------------------------------- finding 15: a spelled-out multiplier ---
@pytest.mark.parametrize("text,literal", (("The count was 1.2 times 10^3 cells/mL.", "1.2"),
                                          ("The count was 1.2 times 10^3 cells/mL.", "10^3"),
                                          ("The count was 1.2 by 10^3 cells.", "1.2"),
                                          ("The count was 1.2 times 1e3 cells.", "1.2"),
                                          ("The count was 1.2 times 1e3 cells.", "1e3"),
                                          ("The count was 1.2 times 10E+03 cells.", "1.2")))
def test_a_power_of_ten_one_prose_word_after_a_number_is_no_value(text, literal):
    assert not _resolved(text, literal), (text, tokens(text))


def test_a_bare_power_after_prose_alone_still_reads():
    assert _resolved("cells were seeded at 10^6 per well", "10^6")
    assert _resolved("dose 12 mg given to 10 patients", "12") and _resolved("dose 12 mg given to 10 patients", "10")


def test_a_spelled_out_multiplier_fails_end_to_end():
    s = "The count was 1.2 times 10^3 cells/mL."
    assert run(PAGE + s, s, "1.2", field="AUC").status == "fail"
    assert run(PAGE + s, s, "10^3", field="AUC").status == "fail"
    s2 = "The count was 1.2 times 1e3 cells."
    assert run(PAGE + s2, s2, "1.2", field="AUC").status == "fail"


# --------------------------------------------------------- finding 7: a flattened raised ten in a label ---
@pytest.mark.parametrize("label", ("WBC (109/L)", "Platelets (103/μL)", "Count (106 cells/mL)", "WBC (10⁹/L)",
                                   "Count (10^6 cells/mL)", "Count 1012/L"))
def test_a_flattened_raised_ten_in_a_header_is_a_scale(label):
    assert scale_shapes(label), label


@pytest.mark.parametrize("label", ("Count (cells/106)", "events per 106 cells", "Dose 12 mg", "Day 3", "Arm 2.5"))
def test_a_raised_ten_as_a_denominator_and_ordinary_numbers_stay_scale_free(label):
    assert not scale_shapes(label), label


@pytest.mark.parametrize("label", ("WBC (109/L)", "Platelets (103/μL)", "Count (106 cells/mL)"))
def test_a_flattened_raised_ten_header_flags_the_cell(label):
    assert "scaled_header" in cell([["Arm", label], ["A", "2.5"]], 1, 1, "2.5", field="AUC").flags
