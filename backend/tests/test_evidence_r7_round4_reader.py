"""r7 round 4: the reader findings of the round-4 audit, each PoC a regression (owner decisions 2, 3 and 6; THE RULE).

* a multiplier or exponent character printed right before a digit or a sign inside a letter-led word is the multiplier
  or exponent production, never a letter of a name, unless a digit of the label stands on its left (CYP2E1, CYP2D6*4);
  an operator letter is judged on the printed text (no carve-out after a bracket or for the times sign only);
* a number's dash, slash or colon joins letters only as a unit production ('24-h', '12/kg'); '12-lOO', '10-fold',
  '12/lS' and '12:lS' are no production; a lone slash or colon word after a number needs a unit after it;
* a degree sign before a digit is no glue ('T=37°5');
* the spaced ratio is unsigned like the glued one ('+12 / 15');
* a raised ten whose multiplier the text layer dropped ('1.2 10−3 M', '2.5 109/L', '1.2 10 3') and a pure power of ten in
  e-notation ('1.2 10E+03') beside a number are detached scales; a power of ten one prose word after a number
  ('1.2 times 10^3') is no value and leaves the number unresolved.
"""
from __future__ import annotations

import pytest

from app.evidence.numgrammar import glue as G
from app.evidence.numgrammar import tokens
from app.evidence.scale_label import scale_shapes
from tests.evidence_r6_round1_support import cell, run


def _resolved(text: str, literal: str, mode: str = "page") -> bool:
    s = text.index(literal)
    return any(t.value is not None and (t.start, t.end) == (s, s + len(literal)) for t in tokens(text, mode=mode))


def _any_resolved_over(text: str, literal: str) -> bool:
    s = text.index(literal)
    return any(t.value is not None and t.start < s + len(literal) and s < t.end for t in tokens(text))


# ----------------------------------------------------------------------- finding 1: the label production ---
GLUED_SCALES = ("AUC*1000", "AUCx1000", "AUCX1000", "AUC-x1000", "AUC,x1000", "AUC.x1000", "AUC;x1000", "AUC:E3",
                "AUC=E3", "AUC-E3", "AUC*1e3", "CLx10", "AUC1000x", "Conc*1000 (ng/mL)", "AUC×1000", "AUC·1000",
                "AUC(x1000)", "Count(E3)")


@pytest.mark.parametrize("header", GLUED_SCALES)
def test_a_scale_glued_to_a_header_word_is_never_a_name(header):
    assert scale_shapes(header), header


@pytest.mark.parametrize("header", ("AUC0-24", "CYP2E1", "CYP2D6*4", "CYP3A4*22", "AUC (ng·h/mL)", "mg/kg", "Cmax,ss",
                                    "t1/2", "mL/min/1.73m2", "L·h⁻¹", "AUC(0-24)", "ng×h"))
def test_names_with_digits_stay_scale_free(header):
    assert not scale_shapes(header), header


def test_the_label_production_is_judged_on_the_printed_text():
    for w in ("AUCx1000", "AUC*1000", "AUC-E3", "Count(E3)", "AUC×1000", "AUC1000x", "Cmax1"):
        assert not G.label(w), w                       # 'Cmax1' fails closed: x before a digit (the finding accepts it)
    for w in ("CYP2E1", "CYP2D6*4", "AUC0-24", "t1/2", "Cmax,ss", "C24h", "h-1", "ng·h/mL", "L·h⁻¹"):
        assert G.label(w), w
    assert not hasattr(G, "_product_before_digit") and not hasattr(G, "_bracketed_operator")


@pytest.mark.parametrize("header", ("AUC*1000 (ng·h/mL)", "AUCx1000 (ng·h/mL)", "Conc*1000 (ng/mL)", "CLx10 (L/h)",
                                    "AUC1000x"))
def test_a_glued_scale_header_flags_the_cell_end_to_end(header):
    r = cell([["Arm", header], ["A", "12.5"]], 1, 1, "12.5", field="AUC")
    assert "scaled_header" in r.flags, (header, r)


@pytest.mark.parametrize("text", ("Countx1000 250", "Count*1000 250", "Count-E3 250", "Count(E3) 250", "Count×1000 250"))
def test_a_number_beside_a_glued_scale_word_is_unresolved(text):
    assert not _resolved(text, "250"), text


# --------------------------------------------------------------- finding 2: a number's dash joins a unit only ---
@pytest.mark.parametrize("word", ("12-lOO", "12-IOO", "12-ISO", "12-lSO", "12-OOO", "1.5-SOO", "12-SOl", "5-lOmg",
                                  "10-fold", "3-day", "12-week"))
def test_letters_after_a_numbers_dash_that_are_no_unit_production_leave_it_unresolved(word):
    toks = tokens(word)
    assert toks and all(t.value is None for t in toks), (word, toks)
    assert toks[0].why == "glued_letters", toks


@pytest.mark.parametrize("word,literal", (("24-h", "24"), ("5-mg", "5"), ("2.5-mg/kg", "2.5"), ("12-15", "12")))
def test_a_unit_production_or_a_number_after_the_dash_still_reads(word, literal):
    assert _resolved(word, literal), word


def test_a_look_alike_range_fails_end_to_end_and_the_digit_range_is_flagged():
    assert run("doses of 12-lOO mg were given", "doses of 12", "12", field="dose").status == "fail"
    ctl = run("doses of 12-100 mg were given", "doses of 12", "12", field="dose")
    assert ctl.status == "pass" and "censored_value" in ctl.flags


# ------------------------------------------------------------ finding 3: a slash or colon before letters ---
@pytest.mark.parametrize("word", ("12/lS", "12/IS", "5/lO", "1/OO", "12/lOO", "12:lS", "5:lO", "12/day", "12/(kg"))
def test_letters_after_a_glued_slash_or_colon_that_are_no_unit_production_leave_the_number_unresolved(word):
    assert not _any_resolved_over(word, word.split("/")[0].split(":")[0]), (word, tokens(word))


@pytest.mark.parametrize("word,literal", (("12/kg", "12"), ("5/min", "5"), ("12/h", "12"), ("0.6ng/ml", "0.6"),
                                          ("12/15", "12/15"), ("12:", "12")))
def test_a_unit_denominator_after_a_glued_slash_still_reads(word, literal):
    assert _resolved(word, literal), (word, tokens(word))


@pytest.mark.parametrize("text", ("12 / lS", "12 /lS", "12 : lS", "12 :lS", "12 / IS mg"))
def test_a_lone_slash_or_colon_after_a_number_needs_a_unit_after_it(text):
    assert not _resolved(text, "12"), (text, tokens(text))


@pytest.mark.parametrize("text", ("12 / kg body weight", "12 /kg", "5 / min", "Cmax : 12.5 ng/mL"))
def test_a_unit_after_a_lone_slash_and_a_colon_before_the_number_still_read(text):
    literal = "12.5" if "12.5" in text else text.split()[0]
    assert _resolved(text, literal), (text, tokens(text))


def test_a_look_alike_ratio_fails_end_to_end():
    assert run("responders 12/lS in arm A", "responders 12", "12").status == "fail"
    assert run("ratio 5:lO observed", "ratio 5", "5").status == "fail"
    assert run("responders 12/15 in arm A", "responders 12", "12").reason == "partial_number"


# ------------------------------------------------------------ finding 11: a degree sign before a digit ---
@pytest.mark.parametrize("text,literal", (("T=37°5", "37"), ("fever(37°5)", "37"), ("lat=45°30'", "45"),
                                          ("T=37◦5", "37"), ("37°5", "37"), ("T = 37°5", "37")))
def test_a_degree_sign_before_a_digit_is_no_glue(text, literal):
    assert not _any_resolved_over(text, literal), (text, tokens(text))


@pytest.mark.parametrize("text", ("37°C", "T=37°C", "at 37° for", "(37°)"))
def test_a_degree_sign_before_a_letter_or_at_the_end_still_reads(text):
    assert _resolved(text, "37"), (text, tokens(text))


def test_a_french_decimal_temperature_fails_end_to_end():
    assert run("On admission T=37°5 and stable", "T=37°5", "37").status == "fail"
    assert run("Fever (T=37◦5) was noted", "T=37◦5", "37").status == "fail"
    assert cell([["Arm", "Temp"], ["A", "T=37°5"]], 1, 1, "37").status == "fail"


# --------------------------------------------------------------------- finding 31: an unsigned spaced ratio ---
@pytest.mark.parametrize("text", ("+12 / 15", "-12 / 15", "(+12 / 15)"))
def test_a_signed_spaced_ratio_is_refused_like_the_glued_one(text):
    assert all(t.shape != "ratio" for t in tokens(text)), tokens(text)
    assert not _any_resolved_over(text, "12"), tokens(text)


def test_a_signed_spaced_ratio_fails_end_to_end():
    assert run("ratio +12 / 15 here", "+12 / 15", "12/15").status == "fail"
    assert run("ratio 12 / 15 here", "12 / 15", "12/15").status == "pass"
