"""r7 round 3: shapes made only of allowed characters that read a wrong value (findings of the round-3 audit), each
refused through a production, never a detector of the bad shape:

* letters glued after a decimal mark or a group separator are no glue ('1,OOO', '0.OS', '12.l'): after '.', ',' or "'"
  only a closer, the end of the word or a unit production may follow; a dash after a number joins a compound only
  when a unit production or a word of three letters or more follows it ('12-lS' is no compound);
* the detached decimal is symmetric: a number word that ENDS in the page's decimal mark, then a digit-led word ('0. 5',
  '12.\\n5'), is one unresolved expression like '12 .5' (a comma on a point page stays a list mark: '12, 15');
* a variable name before '=' is a LABEL or one letter: an operator letter before digits ('x10^3', 'E3', 'x1000') is
  no variable; a letter run after an opener or closer inside a label word gets the operator-letter test on the
  printed text ('Count(E3) 250', 'Count(x1000) 250');
* a citation glued to a glued unit ('12mg[3]') leaves the number read: the masked citation keeps its opener.
"""
from __future__ import annotations

import pytest

from app.evidence import models as M
from app.evidence.dose_rule import find_dose
from app.evidence.numgrammar import tokens
from tests.evidence_r6_round1_support import cell, run


def values(text: str, style: str = ".") -> list[float | None]:
    return [t.value for t in tokens(text, style)]


# -------------------------------------------------------------------- letters after a separator ---
@pytest.mark.parametrize("page, value", [
    ("A total of 1,OOO patients", "1"), ("A total of 1,ØØØ patients", "1"), ("p < 0.OS", "0"),
    ("Cmax 12.l ng/mL", "12"), ("A total of 1'OOO patients", "1"), ("A total of 1’OOO patients", "1"),
    ("dose 10.OOO units", "10"), ("Cmax 12-lS ng/mL", "12"), ("Cmax 12-l ng/mL", "12")])
def test_letters_after_a_separator_or_a_lone_dash_are_no_glue(page, value):
    out = run(page, page, value, field="dose")
    assert out.status == "fail", (page, out)
    assert all(t.value is None for t in tokens(page) if t.start <= page.index(value) < t.end)


def test_the_keyless_dose_rule_never_proposes_a_look_alike_group():
    assert find_dose("Each subject received 1,OOO mg daily") == []
    assert find_dose("Each subject received 1,000 mg daily") == [(22, 27, 1000.0, "mg")]


def test_a_cell_with_a_look_alike_group_fails():
    assert cell([["Arm", "N"], ["A", "1,OOO"]], 1, 1, "1").status == "fail"


@pytest.mark.parametrize("text, want", [
    ("10-fold", [None]), ("24-h", [24.0]), ("5-mg tablets", [5.0]), ("12-week study", [None]), ("3-day", [None]),
    ("12.5.", [12.5]), ("12.5,", [12.5]), ("(12.5).", [12.5]), ("150mg,", [150.0]), ("0.5,mg", [0.5])])
def test_a_compound_or_a_unit_after_the_mark_still_reads(text, want):
    """r7 round 4 (owner decision 2, the round-4 audit): a number's dash joins letters only as a unit production; the
    compounds '10-fold', '12-week', '3-day' (read in round 3 through a three-letter minimum, a heuristic) are no
    production and stay unresolved like '12-lOO'."""
    assert values(text) == want, (text, tokens(text))


@pytest.mark.parametrize("sep", [" ", " ", " "])
def test_a_letter_group_after_a_thin_space_is_flagged_like_a_digit_group(sep):
    page = f"A total of 1{sep}OOO patients"
    out = run(page, page, "1", field="dose")
    assert out.status == "pass" and M.FLAG_COMPOSITE_CELL in out.flags, out
    assert M.FLAG_COMPOSITE_CELL not in run("A total of 1 mg", "1 mg", "1", field="dose").flags


# ------------------------------------------------------------------------- the detached decimal ---
@pytest.mark.parametrize("page, value, style", [
    ("the dose was 0. 5 mg daily", "5", "."), ("the dose was 0. 5 mg daily", "0", "."), ("Cmax 12.\n5 ng/mL", "5", "."),
    ("Cmax (12. 5) ng/mL", "5", "."), ("the dose was 0, 5 mg daily", "5", ","), ("GA<0. 6", "6", ".")])
def test_a_number_that_ends_in_the_decimal_mark_before_a_digit_word_is_one_detached_decimal(page, value, style):
    out = run(page, page, value, field="dose", style=style)
    assert out.status == "fail", (page, out)
    hit = [t for t in tokens(page, style) if t.why == "detached_decimal"]
    assert hit and hit[0].value is None, tokens(page, style)


def test_a_detached_decimal_in_a_cell_fails():
    assert cell([["Parameter", "Dose (mg)"], ["Drug A", "0. 5"]], 1, 1, "5").status == "fail"


@pytest.mark.parametrize("text, style, want", [
    ("at 12, 15 and 18 h", ".", [12.0, 15.0, 18.0]), ("1/2, 3/4", ".", [0.5, 0.75]), ("12. 15", ",", [12.0, 15.0]),
    ("was 12.\nThe next", ".", [12.0]), ("Table 2. Results", ".", [2.0]), ("the age of 35. [35]", ".", [35.0, 35.0]),
    ("ratios of 1.8 \u00b1 0.28,respectively", ".", [1.8, 0.28]), ("AUC of 2.40andC of 1.40.The", ".", [None, 1.40])])
def test_a_list_comma_and_a_sentence_end_are_no_detached_decimal(text, style, want):
    assert values(text, style) == want, (text, tokens(text, style))


# -------------------------------------------------------------------------------- variable names ---
@pytest.mark.parametrize("page", ["Platelets (x10^3) = 250", "Platelets (x10^3)=250", "Platelets x10^3 = 250",
                                  "Platelets (x10^3/uL) = 250", "Count (x10^-3) = 250", "Platelets (E3)= 250",
                                  "Count (e3) = 250", "AUC (x1000) = 250", "Count(E3) 250", "Count(E3): 250",
                                  "Count(x1000) 250"])
def test_a_scale_word_is_never_a_variable_name_or_a_label(page):
    assert run(page, page, "250", field="dose").status == "fail", page
    assert all(t.value is None for t in tokens(page) if t.start <= page.index("250") < t.end), tokens(page)


@pytest.mark.parametrize("text, want", [("(E = 0.9)", [0.9]), ("x = 12", [12.0]), ("Emax = 8", [8.0]),
                                        ("AUC0-24 = 12", [12.0]), ("Platelets = 250", [250.0]), ("k12 = 0.5", [0.5]),
                                        ("CYP2E1 12.5", [12.5]), ("Cmax(ng/mL) 12", [12.0])])
def test_a_label_or_a_single_letter_stays_a_variable_and_a_label(text, want):
    assert values(text) == want, (text, tokens(text))


# ---------------------------------------------------------------------- a citation glued to a unit ---
@pytest.mark.parametrize("page, value", [("dose 12mg[3] daily", "12"), ("sampled at 24h[12] after dosing", "24"),
                                         ("dose 12 mg[3] daily", "12"), ("dose 12mg [3] daily", "12")])
def test_a_citation_glued_to_a_glued_unit_leaves_the_number_read(page, value):
    out = run(page, page, value, field="dose")
    assert out.status == "pass", (page, out)
    cites = [t for t in tokens(page) if t.why == "citation"]
    assert len(cites) == 1 and page[cites[0].start:cites[0].end].startswith("["), tokens(page)
