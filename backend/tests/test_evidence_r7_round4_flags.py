"""r7 round 4: the flag findings of the round-4 audit (raise-only checks around a verified number; each PoC a regression).

* scaled_header: a scale word broken at a line-end hyphen ('thou-' + 'sands'), spelled with OCR letter look-alikes
  ('rnillion', 'miIlion', 'thousancls', a Greek kappa for k), or read as the head of the next word or cell
  ('million[3]', 'million/year', a cell 'M');
* censored_value: qualifier and bound words with look-alikes ('beIow', 'up t0', 'betvveen'), a citation between the
  qualifier and the number, a unit between the number and 'or older', range words 'through' / 'thru' / 'until' / 'till'
  and a glued 'to15', bound phrases split over table cells, the bound words 'not exceeding', 'maximum', 'GE', and sign
  words ('minus 12%');
* composite_cell: a grouped mantissa that carries an exponent ('1 234 x 10^3'), a head behind a prefix ('n=1 234'), and a
  grouping space before anything but a unit ('1 OO0 mg').
"""
from __future__ import annotations

import pytest

from app.evidence.scale_label import scale_shapes
from tests.evidence_r6_round1_support import cell, run

PAGE = "Clearance 12.4 L/h, volume 45.6 L, half-life 3.5 h, ka 1.25 /h. "


def _text(s: str, value: str, snippet: str | None = None, unit: str | None = None):
    return run(PAGE + s, snippet or s, value, field="AUC", unit=unit)


# --------------------------------------------------------------- findings 8, 9 and 14: scale words ---
@pytest.mark.parametrize("label", ("AUC (thou-\nsands)", "AUC (thou‐\nsands)", "AUC (mil-\nlions)", "AUC (rnillions)",
                                   "AUC (miIlions)", "AUC (thousancls)"))
def test_a_wrapped_or_look_alike_scale_word_in_a_header_is_a_scale(label):
    assert scale_shapes(label), label
    assert "scaled_header" in cell([["Arm", label], ["A", "2.5"]], 1, 1, "2.5", field="AUC").flags


@pytest.mark.parametrize("s", ("Exposure was 1.5 mil-\nlion person-years", "Exposure was 1.5 thou‐\nsand person-years",
                               "Exposure was 1.5 rnillion person-years", "Exposure was 1.5 miIlion person-years",
                               "Exposure was 1.5 thousancl person-years", "Sales were 1.5 κ units",
                               "There were 1.5 million[3] cases worldwide.", "There were 1.5 billion[3] cases.",
                               "There were 1.5 thousand[3] cases.", "There were 1.5 M[3] cases.",
                               "There were 1.5 million[3-5] cases.", "There were 1.5 million.[3] cases.",
                               "Cases 1.5 million/year rose.", "Cases 1.5 million* rose.", "Cases 1.5 million(3) rose.",
                               "Cases 1.5 million- rose.", "Cases 1.5 million's rose.", "Cases 1.5 mi11ion rose."))
def test_a_scale_after_the_number_flags_however_it_is_glued_wrapped_or_spelled(s):
    r = _text(s, "1.5", snippet=s)
    assert r.status == "pass" and "scaled_header" in r.flags, (s, r)


def test_a_scale_symbol_in_the_next_cell_flags_like_the_next_word():
    assert "scaled_header" in cell([["Item", "Count", "Scale"], ["Cases", "1.5", "M"]], 1, 1, "1.5", field="AUC").flags


@pytest.mark.parametrize("s", ("Exposure was 1.5 mg in millions of cells.", "The mill was 1.5 km away.",
                               "Volume 1.5 mL in 3 mice."))
def test_ordinary_words_after_a_number_raise_no_scale(s):
    assert "scaled_header" not in _text(s, "1.5", snippet=s).flags, s


# ------------------------------------------------------------- finding 10: look-alike qualifier words ---
@pytest.mark.parametrize("s,v", (("LLOQ was beIow 0.5 ng/mL", "0.5"), ("LLOQ was Iess than 0.5 ng/mL", "0.5"),
                                 ("Cmax was approxirnately 12.5 ng/mL", "12.5"), ("Cmax was abovc 12.5 ng/mL", "12.5"),
                                 ("LLOQ was at rnost 0.5", "0.5"), ("LLOQ was at Ieast 0.5", "0.5"),
                                 ("LLOQ was rnore than 0.5", "0.5"), ("LLOQ was up t0 0.5", "0.5"),
                                 ("Cmax was betvveen 12 and 15 ng/mL", "12"), ("LLOQ was bel0w 0.5 ng/mL", "0.5")))
def test_a_look_alike_qualifier_word_flags_censored_value(s, v):
    r = _text(s, v, snippet=s)
    assert r.status == "pass" and "censored_value" in r.flags, (s, r)


# -------------------------------------------------- finding 18: a citation between a bound word and a number ---
@pytest.mark.parametrize("s,v", (("Cmax was less than [3] 1.5 ng/mL.", "1.5"), ("Cmax was approximately [3] 1.5 ng/mL.", "1.5"),
                                 ("Cmax was below [3, 4] 1.5 ng/mL.", "1.5"), ("Cmax was approximately[3] 12 ng/mL.", "12"),
                                 ("Cmax was less than[3] 12 ng/mL.", "12"), ("Cmax was up to[3] 12 ng/mL.", "12"),
                                 ("Cmax was between[3] 12 and 15 ng/mL.", "12"), ("Cmax was 12 or more[3] ng/mL.", "12"),
                                 ("Cmax ranged 1.5 to[3] 4.5 ng/mL.", "1.5"), ("Cmax ranged 1.5 to[3] 4.5 ng/mL.", "4.5"),
                                 ("Cmax ranged 1.5 to [3] 4.5 ng/mL.", "4.5"), ("Cmax ranged 1.5 [3] to 4.5 ng/mL.", "1.5")))
def test_a_citation_never_hides_a_bound_or_range_word(s, v):
    r = _text(s, v, snippet=s, unit="ng/mL")
    assert r.status == "pass" and "censored_value" in r.flags, (s, v, r)


@pytest.mark.parametrize("row,col", ((["Cmax", "approximately[3]", "12"], 2), (["Cmax", "<[3]", "12"], 2)))
def test_a_citation_in_a_qualifier_cell_never_hides_it(row, col):
    assert "censored_value" in cell([["Param", "A", "B"], row], 1, col, "12").flags


# --------------------------------------------------- finding 19: a unit between the number and the bound ---
@pytest.mark.parametrize("s,v", (("Patients aged 18 years or older were enrolled.", "18"), ("Cmax was 0.5 ng/mL or less.", "0.5"),
                                 ("eGFR 60 mL/min or more", "60"), ("Patients aged 65 years and above.", "65"),
                                 ("Cmax was 0.5 ng/mL or lower.", "0.5"), ("Tmax was 12 h or longer.", "12"),
                                 ("Tmax was 12 h or shorter.", "12")))
def test_a_bound_after_the_unit_flags_censored_value(s, v):
    r = _text(s, v, snippet=s)
    assert r.status == "pass" and "censored_value" in r.flags, (s, r)


# ------------------------------------------------------------------- finding 20: range words ---
@pytest.mark.parametrize("s", ("Cmax ranged 12 through 15 ng/mL.", "Cmax ranged 12 thru 15 ng/mL.",
                               "Cmax ranged 12 until 15 ng/mL.", "Cmax ranged 12 till 15 ng/mL.", "Cmax ranged 12 to15 ng/mL.",
                               "Cmax ranged 12 -to- 15 ng/mL.", "Cmax 12 (to 15) ng/mL.", "days 1 through 5 of dosing"))
def test_every_range_word_marks_the_left_end(s):
    v = "1" if s.startswith("days") else "12"
    r = _text(s, v, snippet=s)
    assert r.status == "pass" and "censored_value" in r.flags, (s, r)


def test_the_right_end_after_a_range_word_is_flagged_too():
    s = "Cmax ranged 12 through 15 ng/mL."
    assert "censored_value" in _text(s, "15", snippet=s).flags


# --------------------------------------------------- finding 21: bound and range words in their own cells ---
@pytest.mark.parametrize("row,col,v", ((["Cmax", "12", "to", "15"], 1, "12"), (["Cmax", "12", "to", "15"], 3, "15"),
                                       (["Cmax", "12", "or", "less"], 1, "12"), (["Cmax", "less", "than", "12"], 3, "12"),
                                       (["Cmax", "between", "12", "and", "15"], 2, "12"),
                                       (["Cmax", "between", "12", "and", "15"], 4, "15"),
                                       (["Cmax", "at", "least", "12"], 3, "12")))
def test_a_bound_phrase_spread_over_cells_is_read_whole(row, col, v):
    header = ["Param"] + [f"C{i}" for i in range(1, len(row))]
    assert "censored_value" in cell([header, row], 1, col, v).flags, (row, col)


def test_an_ordinary_row_raises_no_bound():
    assert "censored_value" not in cell([["Param", "A", "B", "C"], ["Cmax", "12", "15", "18"]], 1, 2, "15").flags


# ---------------------------------------------------- finding 22: bound words and operator abbreviations ---
@pytest.mark.parametrize("s,v", (("Doses not exceeding 12 mg were given.", "12"), ("Doses exceeding 12 mg were toxic.", "12"),
                                 ("Doses in excess of 12 mg were toxic.", "12"), ("A dose of max. 12 mg was given.", "12"),
                                 ("A dose of maximum 12 mg was given.", "12"), ("A minimum of 12 mg was given.", "12"),
                                 ("Values upwards of 12 mg.", "12"), ("Values almost 12 mg.", "12"),
                                 ("Doses beyond 12 mg.", "12"), ("Age GE 18 years.", "18"), ("Age LT 18 years.", "18"),
                                 ("Age LE 18 years.", "18"), ("Age gte 18 years.", "18"), ("Age GT 18 years.", "18"),
                                 ("Age lte 18 years.", "18"), ("Age &gt; 18 years.", "18"), ("Age &lt; 18 years.", "18")))
def test_a_bound_word_or_operator_abbreviation_flags_censored_value(s, v):
    r = _text(s, v, snippet=s)
    assert r.status == "pass" and "censored_value" in r.flags, (s, r)


# ---------------------------------------------------------------------------- finding 17: sign words ---
@pytest.mark.parametrize("s,v", (("The change from baseline was minus 12% at week 4.", "12"), ("The temperature was minus 20 C.", "20"),
                                 ("The z-score was negative 1.5 overall.", "1.5"), ("The shift was plus 3 mm.", "3")))
def test_a_spelled_sign_word_flags_the_unsigned_number(s, v):
    r = _text(s, v, snippet=s)
    assert r.status == "pass" and "censored_value" in r.flags, (s, r)


# ------------------------------------------------------- findings 4, 12 and 13: composite numbers ---
@pytest.mark.parametrize("s,v,snip", (("The count was 1 234 x 10^3 cells.", "234000", "234 x 10^3"),
                                      ("The count was 1 234e3 cells.", "234e3", "234e3 cells"),
                                      ("The count was 12 345e3 cells.", "345e3", "345e3 cells"),
                                      ("The count was 1 234.5 x 10^3 cells.", "234500", "234.5 x 10^3"),
                                      ("The count was (1 234 x 10^3) cells.", "234000", "234 x 10^3")))
def test_a_grouped_mantissa_with_an_exponent_flags_composite_cell(s, v, snip):
    r = _text(s, v, snippet=snip)
    assert r.status == "pass" and "composite_cell" in r.flags, (s, r)


@pytest.mark.parametrize("s,v,snip", (("A total of n=1 234 patients were enrolled.", "234", "234 patients"),
                                      ("A total of n=12 345 patients.", "345", "345 patients"),
                                      ("Counts were <1 234 cells.", "234", "234 cells"),
                                      ("Counts were ~1 234 cells.", "234", "234 cells"),
                                      ("A total of (n=1 234) patients.", "234", "234)"),
                                      ("A total of n:1 234 patients.", "234", "234 patients")))
def test_a_head_behind_a_prefix_flags_composite_cell(s, v, snip):
    r = _text(s, v, snippet=snip)
    assert r.status == "pass" and "composite_cell" in r.flags, (s, r)


def test_a_grouped_table_mantissa_is_an_error():
    r = cell([["Test", "Value"], ["WBC", "1 234 x 10^3"]], 1, 1, "234000", snippet="234 x 10^3")
    assert ("composite_cell" in r.flags) and r.status == "pass", r


@pytest.mark.parametrize("page", ("Dose 1 OO0 mg daily", "Dose 1 lO0 mg daily", "Dose 1 OOO mg daily",
                                  "Dose 1 lOO mg daily", "Dose 12 patients daily"))
def test_a_grouping_space_before_anything_but_a_unit_flags_composite_cell(page):
    r = run(page, page, "12" if "12" in page else "1", field="dose")
    assert r.status == "pass" and "composite_cell" in r.flags, (page, r)


@pytest.mark.parametrize("page", ("Dose 1 mg daily", "Dose 12 mg/kg daily", "Dose 1 OOO mg daily"))
def test_a_grouping_space_before_a_unit_raises_no_composite_cell(page):
    assert "composite_cell" not in run(page, page, "12" if "12" in page else "1", field="dose").flags
