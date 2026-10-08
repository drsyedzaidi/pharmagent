"""r7 round 2: the neighbour WINDOW is one walk for the taint tiers and the allow-list (findings of the round-2 review).

* the taint window follows the allow-list window: an outside character in the first word past a run of punctuation words
  (which the allow-list skips) taints the number, across a line break too ('12.5\\n( <U+22C5> 10^3');
* a bare power beside a number is a detached scale, never a second number ('1.2 10^3');
* a word of one letter, a sign and digits right after a number is a spaced exponent unless the letter is a unit symbol
  ('1.2 é-3' fails, 'ka 0.12 h-1' reads); a power inside a neighbour word is inert only after a unit atom ('m²', 'h⁻¹';
  never 'l0³', 'lO³');
* a table cell's neighbour check walks past empty, blank and punctuation-only cells;
* a qualifier is read through the same walk ('< , 12' is qualified);
* a label that ends in a joiner continues over the gap ('AUC0-\\n24', 't1/\\n2');
* the in-word checks are linear (a 200 000-glyph page of '(x)' runs or superscript runs).
"""
from __future__ import annotations

import time

import pytest

from app.evidence import models as M
from app.evidence.numgrammar import tokens
from tests import evidence_r6_oracle as O
from tests.evidence_r5b_support import check
from tests.evidence_r6_round1_support import cell, run
from tests.evidence_timing import limit


def status(value: str, snippet: str, page: str) -> tuple[str, str]:
    res = check(value, snippet, page)
    return res.status, res.reason


# ------------------------------------------------------- the taint window follows the allow-list window ---
TEMPLATES = [("12.5\n( {c} 10³ cells/L)", "12.5"), ("{c}\n( 12.5 )", "12.5"), ("12.5 ,\n{c} mg", "12.5"),
             ("{c} =\n12.5", "12.5")]


@pytest.mark.parametrize("template, value", TEMPLATES)
@pytest.mark.parametrize("ch", ["⋅", "‒", "$", "(cid:2)", "​", "х"])
def test_an_outside_character_one_punctuation_word_away_across_a_line_break_taints_the_number(template, value, ch):
    page = template.format(c=ch)
    got = [t for t in tokens(page) if t.start <= page.index(value) < t.end]
    assert got and all(t.value is None and t.reason == "unresolved_context" for t in got), (page, got)


@pytest.mark.parametrize("template, value", TEMPLATES)
def test_the_four_templates_hold_for_a_stratified_sample_of_every_outside_code_point(template, value):
    for cp in O.stratified_outside(512):
        page = template.format(c=chr(cp))
        hit = [t for t in tokens(page) if t.start <= page.index(value) < t.end]
        assert hit and all(t.value is None for t in hit), (hex(cp), page)


@pytest.mark.parametrize("page, snippet, value, control", [
    ("Cmax 12.5\n( ⋅ 10³ cells/L)", "Cmax 12.5", "12.5", "Cmax 12.5\n( x 10³ cells/L)"),
    ("Cmax 12.5\n( (cid:2) 10³ cells/L)", "Cmax 12.5", "12.5", "Cmax 12.5\n( x 10³ cells/L)"),
    ("change ‒ (\n12.5 )", "12.5 )", "12.5", "change - (\n12.5 )"),
    ("Patients $ =\n18 years", "18 years", "18", None)])
def test_the_reported_pages_fail_and_the_ascii_controls_fail_too(page, snippet, value, control):
    assert status(value, snippet, page)[0] == "fail", page
    assert control is None or status(value, snippet, control)[0] == "fail", control


def test_a_tainted_neighbour_names_the_character_for_the_reviewer():
    res = check("12.5", "Cmax 12.5", "Cmax 12.5\n( ⋅ 10³ cells/L)")
    assert (res.status, res.reason) == ("fail", M.REASON_UNRESOLVED_CONTEXT) and "U+22C5" in res.detail, res


def test_the_window_without_an_outside_character_reads_as_before():
    assert [t.value for t in tokens("Cmax 12.5\n( cells/L)")] == [12.5]
    assert [t.value for t in tokens("Dose\n( 12.5 ) mg")] == [12.5]
    assert [t.value for t in tokens("Cmax  =\n12.5 mg")] == [12.5]


# ------------------------------------------------------------------- a bare power beside a number ---
@pytest.mark.parametrize("page, value", [("1.2 10^3", "1.2"), ("5.6 10^-4 cm/s", "5.6"), ("WBC 1.2\n10^9 cells", "1.2"),
                                         ("1.2 10^3,", "1.2"), ("1.2 (10^3)", "1.2")])
def test_a_bare_power_beside_a_number_is_a_detached_scale_and_neither_number_passes(page, value):
    assert status(value, value, page)[0] == "fail", page
    power = page[page.index("10^"):].split()[0].strip("(),")
    assert status(str(10.0 ** int(power[3:])), power, page)[0] == "fail", page
    assert all(t.value is None for t in tokens(page)), tokens(page)


def test_a_bare_power_beside_a_number_in_a_cell_fails():
    assert cell([["P", "V"], ["Cmax", "1.2 10^3"]], 1, 1, "1.2").status == "fail"


def test_a_bare_power_on_its_own_still_reads():
    assert [t.value for t in tokens("count 10^6 cells/mL")] == [1e6]
    assert [t.value for t in tokens("1.2 x 10^3 cells")] == [1200.0]


# ------------------------------------------------------------- letter look-alikes of an exponent or a power ---
@pytest.mark.parametrize("word", ["é-3", "è+03", "ë3", "ε-03", "l0³", "I0³", "lO³",
                                  "q-3", "é3", "o³"])
def test_a_letter_look_alike_of_an_exponent_or_a_power_never_passes_its_mantissa(word):
    page = f"1.2 {word}"
    assert status("1.2", page, page)[0] == "fail", page
    assert cell([["P", "V"], ["Cmax", page]], 1, 1, "1.2").status == "fail", page


@pytest.mark.parametrize("page, want", [("ka 0.12 h-1", [0.12]), ("BSA 1.73 m²", [1.73]), ("k 0.3 h⁻¹", [0.3]),
                                        ("CL 5 L h-1", [5.0]), ("CL 5 mL⁻¹", [5.0]), ("V 12 L-1", [12.0]),
                                        ("12 cells/10^6 cells", [12.0]), ("AUC 12 cm²", [12.0])])
def test_a_unit_symbol_with_an_exponent_is_still_an_inert_neighbour(page, want):
    assert [t.value for t in tokens(page)] == want


# ------------------------------------------------------------- the table-cell neighbours walk like the window ---
@pytest.mark.parametrize("row, col", [(["Cmax", "12.5", "", "⋅10³"], 1), (["Cmax", "12.5", " ", "⋅10³"], 1),
                                      (["‒", "(", "12.5", ")"], 2), (["‒", "", "12.5"], 2),
                                      (["Cmax", "12.5", "]", "⋅10³"], 1)])
def test_a_cell_neighbour_check_walks_past_empty_blank_and_punctuation_cells(row, col):
    got = cell([["P"] * len(row), row], 1, col, "12.5")
    assert (got.status, got.reason) == ("fail", M.REASON_UNRESOLVED_CONTEXT), (row, got)


def test_a_cell_with_clean_neighbours_past_a_gap_cell_passes():
    assert cell([["P", "V", "x", "y"], ["Cmax", "12.5", "", "ng/mL"]], 1, 1, "12.5").status == "pass"


# ------------------------------------------------------------------------- the hidden qualifier ---
@pytest.mark.parametrize("page, value", [
    ("concentration < , 12 ng/mL", "12"), ("p < . 0.05", "0.05"), ("concentration < ) 12 ng/mL", "12"),
    ("concentration < / 12 ng/mL", "12"), ("x < ' 12", "12"), ("x < ] 12", "12"), ("x <\n.\n12", "12"),
    (">= , 18 years", "18"), ("LLOQ < ( ( ( 0.5", "0.5"), ("x < = = = 12", "12")])
def test_a_qualifier_behind_punctuation_words_still_qualifies_the_number(page, value):
    hit = next(t for t in tokens(page) if t.start <= page.rindex(value) < t.end)
    assert hit.value is None or hit.qualified, (page, hit)
    got = run(page, page, value)
    assert got.status == "fail" or M.FLAG_CENSORED_VALUE in got.flags, (page, got)


def test_a_qualified_number_names_the_qualifier_for_the_reviewer():
    got = run("concentration < , 12 ng/mL", "concentration < , 12 ng/mL", "12")
    assert M.FLAG_CENSORED_VALUE in got.flags


# ----------------------------------------------------------- a label that continues over the gap (decision 3) ---
@pytest.mark.parametrize("page, value", [("the AUC0-\n24 was 120 ng.h/mL", "24"), ("the t1/\n2 was 5 h", "2"),
                                         ("the AUC0- 24 was 120", "24"), ("other docu-\n2 weeks", "2")])
def test_a_word_that_ends_in_a_joiner_may_continue_over_the_gap_so_the_number_never_passes_clean(page, value):
    """The label digits decide nothing (decision 3: 'AUC0-' reads like 'docu-'), so the number after a word that ends in
    a hyphen or a slash is marked as one end of a range: the record carries the censored_value error flag."""
    hit = next(t for t in tokens(page) if page[t.start:t.end] == value)
    assert hit.value is not None and hit.ranged, (page, hit)
    got = run(page, page, value)
    assert got.status == "pass" and M.FLAG_CENSORED_VALUE in got.flags, (page, got)


def test_decision_3_still_reads_a_number_after_a_complete_label():
    assert [t.value for t in tokens("AUC0-24 -0.5")] == [-0.5] and [t.value for t in tokens("t1/2 4.5 h")] == [4.5]
    assert check("-0.5", "AUC0-24 -0.5", "change AUC0-24 -0.5 units").number == -0.5


# ------------------------------------------------------------------------------------- linear cost ---
@pytest.mark.parametrize("text", [
    "5 " + "(" * 66_666 + "x)" * 66_666,                       # 200 000 glyphs: operator letters inside one prose word
    "5 A" + "x)" * 66_666 + ")" * 66_666,                      # the same inside a label word
    "a" + "²⁻" * 100_000 + " 5",                     # one superscript run
    "a" + "*" * 200_000 + " 5",                                # one star run
    "A" + "²" * 200_000 + "=5",                           # a superscript run inside a label word before '='
])
def test_the_neighbour_and_glue_checks_are_linear_on_a_200k_glyph_page(text):
    t0 = time.perf_counter()
    tokens(text)
    assert time.perf_counter() - t0 < limit(3.0), text[:20]          # 252 s and 56.8 s (quadratic) at c55fcaf


# ----------------------------------------------- spaced powers with trailing punctuation, ':' labels, ratio lists ---
@pytest.mark.parametrize("page", ["The count was 1.2 x 10^3.", "(1.2 x 10^3)", "1.2 x 10^3,", "1.2 x 10^3;",
                                  "[1.2 x 10^3]", "1.2 x10^3.", "1.2x 10^3.", "clearance 1.2 x 10^3/mL"])
def test_a_spaced_multiplier_power_reads_with_the_closers_and_units_the_glued_form_accepts(page):
    assert [(t.value, t.shape) for t in tokens(page)] == [(1200.0, "sci")], page
    assert status("1.2x10^3", page, page)[0] == "pass", page


@pytest.mark.parametrize("page", ["1.2 x 10^3x", "1.2 x 10^3-", "1.2 x 10^3e", "1.2 x 10^3/5", "1.2 x 10^3.5"])
def test_a_spaced_multiplier_power_glued_to_anything_else_stays_unresolved(page):
    assert all(t.value is None for t in tokens(page)), (page, tokens(page))


@pytest.mark.parametrize("page, value", [("Cmax:12 ng/mL", "12"), ("AUC0-24:12.5 ng.h/mL", "12.5"), ("t1/2:12h", "12")])
def test_a_number_glued_to_a_name_by_a_colon_fails_with_a_sentence_that_says_so(page, value):
    """The colon is no starter (measured on the corpus: as one, 330 reference-list numbers became readable, abbreviated
    page ends '258-9' and DOI prefixes 'doi:10.1016' among them), so the reason must not call the value a label digit."""
    res = check(value, page, page)
    assert (res.status, res.reason) == ("fail", M.REASON_UNRESOLVED_SHAPE), (page, res)
    assert "glued to the name" in res.detail and "by ':'" in res.detail and "part of the label" not in res.detail


@pytest.mark.parametrize("text, want", [("Hypertension.2011;58(1):63-", [1.0]), ("time 8:00", [None]),
                                        ("at:8:00 h", []), ("ratio 1:2", [None])])
def test_a_colon_after_a_closer_or_inside_a_number_starts_nothing(text, want):
    assert [t.value for t in tokens(text)] == want


def test_a_comma_separated_ratio_list_reads_each_ratio():
    assert [(t.value, t.shape) for t in tokens("grade 1/2, 3/4 events")] == [(0.5, "ratio"), (0.75, "ratio")]
    assert [(t.value, t.shape) for t in tokens("12/15, 3/4")] == [(0.8, "ratio"), (0.75, "ratio")]
    assert status("3/4", "grade 1/2, 3/4 events", "grade 1/2, 3/4 events")[0] == "pass"
    assert all(t.value is None for t in tokens("grade 1/2 3/4 events"))          # no list mark: a mixed number
