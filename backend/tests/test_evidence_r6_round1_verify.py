"""r6 round-1 regressions through the whole proposal path (verify, then the value, unit and glyph checks): every case
is a record the round-1 audit passed with a wrong value or without a flag. Reduced to ``Outcome`` by
``evidence_r6_round1_support.run``."""
from __future__ import annotations

import pytest

from app.evidence import models as M
from app.evidence.scale_label import scale_shapes
from tests.evidence_r6_round1_support import COMMA_PAGE, POINT_PAGE, Outcome, cell, doc, run


def ok(o: Outcome) -> bool:
    return o.status == "pass"


# ------------------------------------------- an exact number is the same number, never nearly the same ---

@pytest.mark.parametrize("page, snippet, claim", [
    ("Ki was 3.2e-13 M", "3.2e-13", "0"), ("Ki was 3.2e-13 M", "3.2e-13", "-5e-13"), ("Ki was 3.2e-13 M", "3.2e-13", "7.1e-13"),
    ("Count was 1234567890123 cells", "1234567890123", "1234567890124"),
    ("Total exposures 12345678902 events", "12345678902", "12345678901"),
    ("p was 1.0e-12", "1.0e-12", "0"), ("Tiny 5e-13 M", "5e-13", "1e-13"),
])
def test_a_claim_that_differs_in_value_never_passes_however_small_or_long_the_number(page, snippet, claim):
    got = run(page, snippet, claim)
    assert got.status == "fail" and got.reason in (M.REASON_NUMBER_MISMATCH, M.REASON_NO_NUMBER), got


@pytest.mark.parametrize("page, snippet, claim, num", [
    ("Ki was 3.2e-13 M", "3.2e-13", "3.2e-13", 3.2e-13), ("Ki was 3.2e-13 M", "3.2e-13", "3.20e-13", 3.2e-13),
    ("Count was 1234567890123 cells", "1234567890123", "1234567890123", 1234567890123.0),
    ("Dose 12.50 mg", "12.50", "12.5", 12.5),
])
def test_the_same_decimal_in_another_spelling_still_passes_and_stores_the_page_number(page, snippet, claim, num):
    got = run(page, snippet, claim)
    assert ok(got) and got.value_num == num, got


# --------------------------------------- characters the reader cannot read never vote on the decimal style ---

@pytest.mark.parametrize("decoy", ["٣,٥ ٤,٥", "３.５ ４.５", "۳,۵ ۴,۵"])
def test_padding_of_non_ascii_digits_never_turns_3412_into_3_412(decoy):
    page = "AUC 3,412 ng.h/mL\n" + "\n".join([decoy] * 8) + "\n"
    wrong = run(page, "AUC 3,412 ng.h/mL", "3.412")
    assert (wrong.status, wrong.reason) == ("fail", M.REASON_NUMBER_MISMATCH)      # the old reader passed 3.412 with no flag
    right = run(page, "AUC 3,412 ng.h/mL", "3412")
    assert ok(right) and right.value_num == 3412.0 and M.FLAG_DECIMAL_STYLE_AMBIGUOUS in right.flags
    page = "AUC was 12.345 mg.h/L in healthy volunteers.\n" + "\n".join([decoy] * 8) + "\n"
    got = run(page, "AUC was 12.345 mg.h/L", "12.345")
    assert ok(got) and got.value_num == 12.345


# ------------------------------------------------ a text field is read by the grammar, not by NFKC ---

@pytest.mark.parametrize("page, snippet", [
    ("Dose 10⁰ mg once daily", "Dose 100 mg"), ("Inoculum: 10⁶ cells/mL were infused.", "106 cells/mL"),
    ("Inoculum: 106 cells/mL were infused.", "10⁶ cells/mL"), ("Dose １００ mg", "Dose 100 mg"),
    ("Dose 100 mg", "Dose １００ mg"), ("Dose 1½ mg", "Dose 11/2 mg"), ("Dose 2₂ mg", "Dose 22 mg"),
    ("Dose 5 ⓔ mg", "Dose 5 e mg"), ("Dose 5．5 mg", "Dose 5.5 mg"),
])
def test_a_text_field_never_equates_a_character_of_the_numeric_alphabet_with_another_one(page, snippet):
    got = run(page, snippet, snippet, field="Cell dose")
    assert got.status == "fail", (page, got)


@pytest.mark.parametrize("page, snippet", [("ﬁnal dose 100 mg", "final dose 100 mg"), ("Dose 10² mg", "Dose 10² mg"),
                                           ("dose  100 mg", "dose 100 mg"), ("Dose ≤ 5 mg", "Dose ≤ 5 mg")])
def test_a_ligature_a_space_and_the_same_characters_still_match(page, snippet):
    assert ok(run(page, snippet, snippet, field="Cell dose")), (page, snippet)


# ------------------------------------------- a cell next to the number is read, not skipped ---

HEAD = ["Parameter", "Flag", "Result (ng/mL)"]


@pytest.mark.parametrize("neighbour", ["≤", "<", ">=", "~", "≈", "∼", "below", "less than", "−", "-"])
def test_a_qualifier_or_sign_in_the_previous_cell_flags_the_number(neighbour):
    got = cell([HEAD, ["Cmax", neighbour, "12.5"]], 1, 2, "12.5")
    assert ok(got) and M.FLAG_CENSORED_VALUE in got.flags, (neighbour, got)


@pytest.mark.parametrize("neighbour", ["≲", "⩽", "≠", "х", "✕", "(cid:3)", "​"])
def test_a_character_outside_the_alphabet_in_a_neighbouring_cell_leaves_the_number_unresolved(neighbour):
    for rows, col in (([HEAD, ["Cmax", neighbour, "12.5"]], 2), ([["Parameter", "Result", "Flag"], ["Cmax", "12.5", neighbour]], 1)):
        got = cell(rows, 1, col, "12.5")
        assert (got.status, got.reason) == ("fail", M.REASON_UNRESOLVED_CONTEXT), (neighbour, got)
    assert "neighbouring" in got.detail or "cell" in got.detail


@pytest.mark.parametrize("neighbour", ["×10^3", "x1000", "(thousands)", "x10³"])
def test_a_multiplier_in_a_neighbouring_cell_raises_the_scale_flag(neighbour):
    """r7 round 3: was 'ok(got) and scaled_header in got.flags' for every neighbour. The cells one separator away are
    the number's window (verify_table.window_fault): a multiplier or power word there refuses the number as it does in
    text ('x1000 12.5'), so the record fails; a scale WORD beside it is prose and still raises the flag."""
    got = cell([HEAD, ["Cmax", neighbour, "12.5"]], 1, 2, "12.5")
    if neighbour == "(thousands)":
        assert ok(got) and M.FLAG_SCALED_HEADER in got.flags, (neighbour, got)
    else:
        assert got.status == "fail" and got.reason == M.REASON_UNRESOLVED_SHAPE, (neighbour, got)


def test_a_clean_neighbour_cell_and_a_row_label_with_a_line_character_change_nothing():
    for rows, c in (([HEAD, ["Cmax", "ng/mL", "12.5"]], 2), ([["Parameter", "Result"], ["AUC0–∞", "12.5"]], 1)):
        got = cell(rows, 1, c, "12.5")
        assert ok(got) and not set(got.flags) & {M.FLAG_CENSORED_VALUE, M.FLAG_SCALED_HEADER}, got


# --------------------------------------------- digit groups split by a space, with decimals and signs ---

@pytest.mark.parametrize("page, snippet, style", [
    ("Cmax was 1 234.5 ng/mL", "234.5", "."), ("Cmax was 1 234.5 ng/mL", "1", "."), ("Cmax was 12 345.67 ng/mL", "345.67", "."),
    (COMMA_PAGE + "AUC war 1 234,56 ng/mL", "234,56", ","), ("Cmax was 1 234. ", "1", "."), ("Cmax was 1 234. ", "234", "."),
    ("Cmax (1 234) ng/mL", "1", "."), ("Cmax was -1 234 ng/mL", "-1", "."), ("Cmax was -1 234 ng/mL", "234", "."),
    ("Cmax was 1 234* ng/mL", "1", "."), ("Cmax was 1 234 ng/mL", "234", "."), ("Cmax was 12 145 ng/mL", "12", "."),
])
def test_a_number_a_text_layer_may_have_split_at_a_space_is_flagged_on_either_piece(page, snippet, style):
    got = run(page, snippet, snippet, style=style)
    assert ok(got) and M.FLAG_COMPOSITE_CELL in got.flags, (page, snippet, got)


@pytest.mark.parametrize("page, snippet", [("Cmax was 12.4 145 ng/mL", "12.4"), ("n 12 and 5.5 mg", "12"), ("Cmax 12 ng/mL 1 mg", "12")])
def test_a_decimal_head_and_unrelated_neighbours_are_not_a_group(page, snippet):
    got = run(page, snippet, snippet)
    assert ok(got) and M.FLAG_COMPOSITE_CELL not in got.flags, (page, got)


# -------------------------------------------------------------- ranges, qualifiers, ratios ---

@pytest.mark.parametrize("page, snippet", [
    ("AUC was 12 -15 mg", "-15"), ("AUC was 12 -15 mg", "12"), ("AUC was 12.5 -15.0 mg", "-15.0"), ("dose 5.0- 12.4 mg", "5.0"),
    ("dose 0.5-\n1.5 mg", "0.5"), ("Cmax 12.5 ∼ 15.0 ng/mL", "12.5"), ("Cmax 12.5 ∼ 15.0 ng/mL", "15.0"),
])
def test_both_ends_of_a_range_written_with_one_dash_are_flagged(page, snippet):
    got = run(page, snippet, snippet)
    assert ok(got) and M.FLAG_CENSORED_VALUE in got.flags, (page, snippet, got)


@pytest.mark.parametrize("page", [
    "Cmax ≤(12.5) ng/mL", "Cmax ≈(12.5) ng/mL", "Cmax ≥[12.5] ng/mL", "Cmax ~=12.5 ng/mL", "Cmax >=(12.5) ng/mL",
    "Cmax <(12.5) ng/mL", "Cmax ~= 12.5 ng/mL", "Cmax < = 12.5 ng/mL", "Cmax <- 12.5 ng/mL", "Cmax <+ 12.5", "Cmax <: 12.5",
    "Cmax < ( 12.5 ) ng/mL", "Cmax less than or equal to 12.5 ng/mL", "Cmax greater than or equal to 12.5",
    "Cmax no more than 12.5", "Cmax at or below 12.5",
])
def test_no_qualifier_hides_behind_an_opener_an_equals_sign_or_a_phrase(page):
    got = run(page, "12.5", "12.5")
    assert ok(got) and M.FLAG_CENSORED_VALUE in got.flags, (page, got)


@pytest.mark.parametrize("page, value", [
    ("Cmax αbove 12.5 ng/mL", "12.5"), ("Cmax μnder 12.5 ng/mL", "12.5"), ("Cmax lεss thαn 12.5", "12.5"),
    ("Cmax 12 τo 15 ng/mL", "12"), ("Cmax αpprox 12.5", "12.5"),
])
def test_a_look_alike_spelling_of_a_qualifier_flags_like_the_plain_one(page, value):
    got = run(page, value, value)
    assert ok(got) and M.FLAG_CENSORED_VALUE in got.flags, (page, got)


@pytest.mark.parametrize("page", ["Cmax abοve 12.5 ng/mL", "Cmax ΑBOVE 12.5", "Cmax аbove 12.5 ng/mL"])
def test_a_look_alike_letter_outside_the_alphabet_leaves_the_number_unresolved(page):
    got = run(page, "12.5", "12.5")
    assert (got.status, got.reason) == ("fail", M.REASON_UNRESOLVED_CONTEXT), (page, got)


@pytest.mark.parametrize("page, snippet", [("Cmax 12 / 15 patients", "12"), ("Cmax 12 /15 patients", "12"),
                                           ("Cmax 12/ 15 patients", "15"), ("Date 5 / 12 / 2020 here", "5"),
                                           ("Ratio 1 : 2 dilution", "2"), ("Cmax 12 .5 ng/mL", "12"), ("Cmax 12 ,5 ng/mL", "12")])
def test_a_ratio_or_a_split_decimal_is_never_one_exact_number(page, snippet):
    got = run(page, snippet, snippet)
    assert (got.status, got.reason) == ("fail", M.REASON_PARTIAL_NUMBER), (page, got)


@pytest.mark.parametrize("page", ["Cmax 5.2 (x10^3 cells/mL)", "Cmax 5.2 [x10^3] cells/mL", "Cmax 5.2 (x 10^3 cells/mL)"])
def test_a_scale_in_a_bracket_after_the_number_is_never_an_unscaled_value(page):
    got = run(page, "5.2", "5.2")
    assert (got.status, got.reason) == ("fail", M.REASON_PARTIAL_NUMBER), (page, got)
    got = cell([["P", "V"], ["x", "5.2 (x10^3)"]], 1, 1, "5.2")
    assert (got.status, got.reason) == ("fail", M.REASON_PARTIAL_NUMBER), got


# ---------------------------------------------- a mantissa, a foreign grouping and a reference pair ---

@pytest.mark.parametrize("snippet, want", [("1.234e+03", 1234.0), ("2.345E-3", 0.002345), ("5.678e-4", 0.0005678)])
def test_a_mantissa_on_a_decimal_comma_page_is_a_decimal_point_and_is_flagged_for_a_look(snippet, want):
    got = cell([["Gruppe", "Cmax (ng/mL)"], ["A", snippet]], 1, 1, snippet, style=",", page=COMMA_PAGE)
    assert ok(got) and got.value_num == want and M.FLAG_DECIMAL_STYLE_AMBIGUOUS in got.flags, got
    assert not ok(run(COMMA_PAGE + f"k = {snippet} 1/h", snippet, str(want * 1000), style=","))


def test_a_multi_word_power_on_a_comma_page_reads_the_mantissa_as_a_decimal():
    got = run(COMMA_PAGE + "Zellen 1.234 x 10^3 /mL", "1.234 x 10^3", "1234", style=",")
    assert ok(got) and got.value_num == 1234.0 and M.FLAG_DECIMAL_STYLE_AMBIGUOUS in got.flags
    assert not ok(run(COMMA_PAGE + "Zellen 1.234 x 10^3 /mL", "1.234 x 10^3", "1234000", style=","))
    got = run("Dose 1,234E3 mg", "1,234E3", "1234", style=".")
    assert ok(got) and got.value_num == 1234.0 and M.FLAG_DECIMAL_STYLE_AMBIGUOUS in got.flags


@pytest.mark.parametrize("snippet, value", [("3.101.102", "3101102"), ("192.168.001.001", "192168001001"), ("1.234.567", "1234567")])
def test_a_number_grouped_with_the_documents_decimal_point_is_flagged(snippet, value):
    got = run(POINT_PAGE + f"Version {snippet} of", snippet, value)
    assert ok(got) and M.FLAG_DECIMAL_STYLE_AMBIGUOUS in got.flags, got
    got = run(POINT_PAGE + "doses 1,234,567 mg", "1,234,567", "1234567")
    assert ok(got) and M.FLAG_DECIMAL_STYLE_AMBIGUOUS not in got.flags            # the usual thousands comma on a point page


def test_a_reference_pair_is_not_a_decimal_on_a_comma_page():
    got = run(COMMA_PAGE + "Siehe Referenzen [1,2] fuer", "1,2", "1.2", style=",")
    # r7 (decision 7): the pair is a citation; citing a number inside it fails with the citation sentence
    assert (got.status, got.reason) == ("fail", M.REASON_UNRESOLVED_SHAPE) and "citation" in got.detail, got
    got = run(COMMA_PAGE + "Wert 7 [1,2] fuer", "1,2", "1.2", style=",")          # after a number: the r6 pair rule
    assert (got.status, got.reason) == ("fail", M.REASON_PARTIAL_NUMBER), got


def test_greek_chi_is_a_look_alike_multiplier_and_the_number_stays_unresolved():
    for page in ("AUC was 1.2 χ 10^3 ng.h/mL", "AUC was 1.2 χ 10³ ng.h/mL", "AUC was 1.2 χ10^3 ng.h/mL"):
        got = run(page, "1.2", "1.2")
        assert (got.status, got.reason) == ("fail", M.REASON_UNRESOLVED_CONTEXT), (page, got)
    assert (run("AUC was 1.2 x 10^3 ng.h/mL", "1.2", "1.2").reason == M.REASON_PARTIAL_NUMBER)


# ------------------------------------------------------------------------ headers and units ---

@pytest.mark.parametrize("label", ["AUC (x1000)", "AUC (✕ 1000)", "AUC (⨯1000)", "AUC (⋅1000)", "AUC (∗ 1000)",
                                   "AUC x 1000", "AUC X 1000", "AUC x 1,000", "AUC × 1000", "Cmax х 1000", "Cmax x1０００",
                                   "Cmax ×１０００", "Cmax thоusands", "AUC (τhousands)", "AUC (in μillions)",
                                   "Cmax (✕ 10^3)", "AUC (✕✕ 10^3)", "Cmax • 1000"])
def test_a_label_that_scales_the_number_is_never_read_as_unscaled(label):
    assert scale_shapes(label) is True, label


@pytest.mark.parametrize("label", ["AUC0–∞ (ng·h/mL)", "Cmax (ng/mL)", "CLint (µL/min/10^6 cells)",
                                   "Day 1", "AUC0-24", "t1/2 (h)", "Cmax,ss †", "Visit"])
def test_an_ordinary_label_is_not_a_scale(label):
    assert scale_shapes(label) is False, label


def test_a_lone_multiplier_in_a_label_is_a_scale():
    """r7 round 3: was a row of test_an_ordinary_label_is_not_a_scale ('Dose (mg) x 2' is False). scale_shapes is an
    allow-list (numgrammar/scale.py): a lone multiplier word is no inert production, and '× 2' does scale the value."""
    assert scale_shapes("Dose (mg) x 2") is True


def test_the_unit_next_to_a_number_separated_by_a_thin_or_no_break_space_is_read_once():
    for sp in (" ", " ", " ", " "):
        got = run(f"AUC was 12.5{sp}ng/mL at steady state.", f"AUC was 12.5{sp}ng/mL", "12.5", unit="ng/mL")
        assert ok(got) and not set(got.flags) & {M.FLAG_UNIT_SOURCE_MISMATCH, M.FLAG_UNIT_UNVERIFIED}, (sp, got)
        got = run(f"Dose was 12{sp}mg given daily here.", f"Dose was 12{sp}mg given", "12", unit="mg", field="dose")
        assert ok(got) and not set(got.flags) & {M.FLAG_UNIT_SOURCE_MISMATCH, M.FLAG_UNIT_UNVERIFIED}, (sp, got)
        wrong = run(f"AUC was 12.5{sp}mg/kg total.", f"AUC was 12.5{sp}mg/kg", "12.5", unit="mg")
        assert M.FLAG_UNIT_SOURCE_MISMATCH in wrong.flags, (sp, wrong)


def test_document_helper_builds_what_the_page_says():
    assert doc("x")["pages"][0]["text"] == "x"
