"""r7 owner decision 6: ``12 / 15`` and ``12/15`` are read only through an explicit RATIO production.

    ratio = int "/" int                       glued in one word: 12/15, 3/4, 140/90, (12/15), 12/15.
          | int GAP "/" GAP int               spaced on both sides: 12 / 15 (a gap run is one separator)
    int   = "0" | [1-9][0-9]{0,5}             no sign, no separator, no leading zero
    value = numerator / denominator           (a zero denominator is refused)
    literal = the printed span; canonical = "n/d"; parts = (n, d)

A ratio is equal only to a claim that prints the same ratio (``12/15`` or ``12 / 15``): never to its quotient (``0.8``),
never to a reduced or other ratio (``4/5``). Citing either part alone fails ``partial_number``. Every other slash or
colon shape is refused (``unresolved_shape:fraction``): ``12 /15``, ``12/ 15``, ``1 : 2``, ``8:00``, ``5/12/2020``,
``5 / 12 / 2020``, ``1.5/2``, ``-12/15``, ``05/12``, ``12/0``, ``12/15%``, ``12/15mg``, and a mixed number ``12 3/4``.
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from app.evidence import models as M
from app.evidence.numgrammar import parse_claim, tokens
from tests.evidence_r6_round1_support import cell, run


def one(text: str):
    (t,) = tokens(text)
    return t


@pytest.mark.parametrize("text, span, value, parts, canonical", [
    ("12/15", (0, 5), 0.8, (12.0, 15.0), "12/15"),
    ("3/4", (0, 3), 0.75, (3.0, 4.0), "3/4"),
    ("BP 140/90 at rest", (3, 9), 140 / 90, (140.0, 90.0), "140/90"),
    ("responders (12/15).", (12, 17), 0.8, (12.0, 15.0), "12/15"),
    ("0/15 responded", (0, 4), 0.0, (0.0, 15.0), "0/15"),
    ("12 / 15", (0, 7), 0.8, (12.0, 15.0), "12/15"),
    ("score 12\n/\n15 at", (6, 13), 0.8, (12.0, 15.0), "12/15"),
    ("1/2 of subjects", (0, 3), 0.5, (1.0, 2.0), "1/2"),
])
def test_the_ratio_production_defines_value_literal_and_parts(text, span, value, parts, canonical):
    t = one(text)
    assert ((t.start, t.end), t.value, t.shape, t.parts, t.canonical) == (span, value, "ratio", parts, canonical)
    assert t.key == canonical and t.decimal_value is None and t.decimal == ""


@pytest.mark.parametrize("text", [
    "Cmax 12 /15 patients", "Cmax 12/ 15 patients", "Ratio 1 : 2 dilution", "score 3 :4", "12 : 13", "time 8:00",
    "Date 5 / 12 / 2020 here", "Date 5/12/2020 here", "dose 1.5/2 mg", "change -12/15", "on 05/12 we", "12/0 cases",
    "12/15% of", "12/15mg", "12/15-18/20", "12/15/kg", "1234567/2",
])
def test_every_other_slash_or_colon_shape_is_refused(text):
    got = tokens(text)
    assert got and all(t.value is None and t.reason == "unresolved_shape" for t in got if t.parts or "/" in
                       text[t.start:t.end] or ":" in text[t.start:t.end]), (text, got)
    assert not any(t.shape == "ratio" for t in got), (text, got)


def test_a_ratio_with_a_unit_after_it_is_refused_with_both_parts():
    """r7 round 2 (P0 finding): a ratio is dimensionless; '140/90 mmHg' prints two pressures, never their quotient 1.56
    (it was a resolved ratio of value 140/90 until round 2)."""
    got = tokens("BP 140/90 mmHg")
    assert [(t.value, t.why, t.parts) for t in got] == [(None, "fraction", (140.0, 90.0))]


def test_a_mixed_number_and_a_ratio_inside_a_label_word_are_never_a_ratio():
    assert [t.why for t in tokens("n = 12 3/4 pts")] == ["mixed_number"]
    assert tokens("t1/2 was long") == [] and tokens("CL/F 12/kg")[0].value == 12.0


@pytest.mark.parametrize("part", ["3", "4", "3/4"])
def test_every_fraction_part_of_a_mixed_number_fails_partial_number(part):
    """A rejected mixed number must not make any of its constituent numbers look absent."""
    got = run("dose was 12 3/4 mg", "12 3/4", part)
    assert (got.status, got.reason) == ("fail", M.REASON_PARTIAL_NUMBER), got


def test_a_ratio_equals_only_the_same_printed_ratio():
    assert parse_claim("12/15") == 0.8 and parse_claim("12 / 15") == 0.8
    for page in ("Responders 12/15 (80%) overall", "Responders 12 / 15 (80%) overall"):
        snippet = page[:-8]
        for value in ("12/15", "12 / 15"):
            got = run(page, snippet, value)
            assert (got.status, got.value_num) == ("pass", 0.8), (page, value, got)
        for part in ("12", "15"):
            got = run(page, snippet, part)
            assert (got.status, got.reason) == ("fail", M.REASON_PARTIAL_NUMBER), (page, part, got)
        for other in ("0.8", "4/5", "12/16", "15/12"):
            got = run(page, snippet, other)
            assert got.status == "fail" and got.reason in (M.REASON_NUMBER_MISMATCH, M.REASON_NO_NUMBER), (page, other)


def test_a_ratio_claim_never_matches_a_decimal_and_a_ratio_is_not_an_integer():
    got = run("ratio was 0.8 overall", "ratio was 0.8", "4/5")
    assert got.status == "fail"
    rows = [["Arm", "Responders"], ["A", "12/15"]]
    assert cell(rows, 1, 1, "12/15").status == "pass"
    part = cell(rows, 1, 1, "12", snippet="12/15")
    assert (part.status, part.reason) == ("fail", M.REASON_PARTIAL_NUMBER)


def test_the_key_of_a_number_is_its_exact_decimal():
    t = one("12.50")
    assert t.key == Decimal("12.50") and t.key == t.decimal_value and t.key != "12.50"
    assert tokens("12 / 15")[0].key != tokens("4/5")[0].key


@pytest.mark.parametrize("text, why", [
    ("m 12 3/4", "mixed_number"), ("Week 4 12/15 responders", "mixed_number"), ("4 12 / 15", "fraction"),
    ("12 x3/4", "multiplier_operand"), ("12 *3/4", "multiplier_operand"), ("12 x 3/4", "multiplier_operand"),
    ("+3/4", "glued_symbol"), ("12 +3/4", "glued_symbol"),
])
def test_a_ratio_that_may_be_part_of_a_mixed_number_a_product_or_a_signed_shape_is_refused(text, why):
    got = tokens(text)
    assert any(t.why == why for t in got) and not any(t.shape == "ratio" for t in got), (text, got)
    assert not any(t.value is not None and t.start <= text.find("/") < t.end for t in got), (text, got)


@pytest.mark.parametrize("part", ["3", "4", "3/4"])
def test_every_fraction_part_of_a_mixed_number_in_a_table_cell_fails_partial_number(part):
    rows = [["Arm", "Dose"], ["A", "12 3/4"]]
    got = cell(rows, 1, 1, part, snippet="12 3/4")
    assert (got.status, got.reason) == ("fail", M.REASON_PARTIAL_NUMBER), got
    assert cell(rows, 1, 1, "12", snippet="12 3/4").reason == M.REASON_PARTIAL_NUMBER


@pytest.mark.parametrize("printed", ["12 3/4", "12  3/4", "12\n3/4", "AUC=12 3/4"])
@pytest.mark.parametrize("part", ["3", "4", "3/4", "12"])
def test_spacing_forms_of_the_mixed_number_cite_partial_number(printed, part):
    got = run(f"dose was {printed} mg", printed, part)
    assert (got.status, got.reason) == ("fail", M.REASON_PARTIAL_NUMBER), (printed, part, got)


@pytest.mark.parametrize("part", ["6/8", "0.75", "34", "1", "12 3/4", "3 / 4", "75%"])
def test_negative_controls_a_mixed_number_never_passes_for_any_value(part):
    got = run("dose was 12 3/4 mg", "12 3/4", part)
    assert got.status == "fail" and got.value_num is None, got


def test_the_quotient_and_equivalent_ratio_of_a_mixed_number_part_do_not_become_partial_or_pass():
    for part in ("6/8", "0.75"):
        got = run("dose was 12 3/4 mg", "12 3/4", part)
        assert got.status == "fail" and got.reason in (M.REASON_NO_NUMBER, M.REASON_NUMBER_MISMATCH), (part, got)


def test_an_adjacent_list_item_is_not_a_part_of_a_mixed_number():
    got = run("doses 5, 12 3/4 and 7 mg", "5, 12 3/4 and 7", "7")
    assert got.status == "pass" and got.value_num == 7.0, got
    got = run("doses 5, 12 3/4 and 7 mg", "5, 12 3/4 and 7", "5")
    assert got.status == "pass" and got.value_num == 5.0, got


def test_a_standalone_ratio_beside_a_mixed_number_still_matches_only_itself():
    page = "n 12 3/4 and 3/4 here"
    got = run(page, "3/4 here", "3/4")
    assert got.status == "pass", got


def test_mixed_number_offsets_are_native_and_cite_the_ratio_once():
    text = "dose was 12 3/4 mg"
    (t,) = tokens(text, mode="claim", labels=True)
    assert text[t.start:t.end] == "12 3/4" and t.value is None and t.why == "mixed_number"
    assert t.parts == (12.0, 3.0, 4.0) and t.ratios == ("3/4",)


def test_ratio_provenance_is_linear_on_a_long_list_of_mixed_numbers():
    import time
    text = " ".join("12 3/4" for _ in range(20000))
    t0 = time.perf_counter()
    got = tokens(text, mode="claim", labels=True)
    assert time.perf_counter() - t0 < 10 and len(got) >= 1
    assert all(len(x.parts) <= 64 and len(x.ratios) <= 64 for x in got)
