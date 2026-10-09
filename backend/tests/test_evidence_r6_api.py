"""r6 numeric grammar: the helpers the callers will use (number_spans, words_after, scale_shapes ...). ``token_at``,
``words_around`` and ``find_n`` had no caller in the application (r6 round-2 audit): they are test helpers now
(``tests/evidence_r6_helpers.py``) and are exercised here through it."""
from __future__ import annotations

import pytest

from app.evidence.dose_rule import find_dose
from app.evidence.numgrammar import (
    is_data_cell,
    number_spans,
    numbers_in,
    parse_claim,
    tokens,
    words_after,
)
from app.evidence.scale_label import scale_shapes
from tests.evidence_r6_helpers import find_n, token_at, words_around

T = "\u2009"


def test_number_spans_and_numbers_in_are_wrappers_over_resolved_tokens():
    assert number_spans("Cmax 12.5 and 3,412 (n=12) and 2 x 10 mg", ".") == [(5, 9, 12.5), (14, 19, 3412.0),
                                                                           (23, 25, 12.0)][:3]
    assert numbers_in("Cmax 12.5 and 18 900 ng", ".") == [("12.5", 12.5), ("18 900", 18900.0)]
    assert numbers_in("<0.5 and −20", ".") == [("0.5", 0.5), ("−20", -20.0)]
    assert numbers_in("", ".") == [] and numbers_in("no digits", ".") == []


def test_strict_number_spans_split_on_ascii_space_only():
    assert [v for *_, v in number_spans("12 145", ".", strict=True)] == [12.0, 145.0]
    assert [v for *_, v in number_spans("12 145", ".", strict=False)] == [12145.0]
    assert [v for *_, v in number_spans(f"12{T}145", ".", strict=True)] == [12145.0]
    assert [v for *_, v in number_spans("18,2", ",", strict=True)] == [18.2]


def test_token_at_finds_the_token_or_expression_with_exactly_that_span():
    text = "AUC 12.5 and 1.2 x 10 3 cells"
    assert token_at(text, 4, 8, ".").value == 12.5
    assert token_at(text, 13, 23, ".").why == "ten_after_multiplier" and token_at(text, 13, 16, ".") is None
    assert token_at(text, 0, 3, ".") is None
    assert token_at("3,412 (812)", 0, 5, ".", mode="cell").value == 3412.0


def test_words_around_returns_the_whole_words_one_gap_run_away():
    text = "AUC <   12.5 mg/L"
    assert words_around(text, 9, 13) == ("<", "mg/L")
    assert words_around("12.5", 0, 4) == ("", "")
    assert words_around("a 12.5-15 b", 2, 6) == ("a", "b")                    # the token's WORD is 12.5-15
    assert words_around("x\n\n\t 12 \r\n y", 6, 8) == ("x", "y")
    assert words_around("AUC <" + " " * 5000 + "12.5", 5006, 5010)[0] == "<"      # padding never widens the window
    assert words_around("− 12 –", 2, 4) == ("-", "-")                   # words are read after the 1:1 folding


def test_words_after_lists_the_following_words():
    assert words_after("AUC 12.5 to 15 mg", 8, 2) == ["to", "15"]
    assert words_after("AUC 12.5 to 15 mg", 8, 9) == ["to", "15", "mg"]
    assert words_after("12.5", 4, 3) == [] and words_after("a 12.5\n\nb c", 6, 2) == ["b", "c"]
    assert words_after("12.5-15 x", 4, 1) == ["x"]                              # the rest of the token's word is skipped


@pytest.mark.parametrize("label, expected", [
    ("AUC ×10³ (ng·h/mL)", True), ("AUC (ng·h/mL) E+03", True), ("Cmax (x10^3)", True),
    ("AUC x 10^3", True), ("AUC × 10 3", True), ("n × 10", True), ("x1000", True), ("× 1000", True),
    ("/1000", True), ("/ 1,000", True), ("(thousands)", True), ("in Millions", True), ("billion cells", True),
    ("count 10^3", True), ("count 10³", True), ("AUC e3", True), ("AUC E-03", True), ("x 10 ^ 3", True),
    ("AUC\u0336 ×\u200d10^3", True), ("AUC x\u00ad10^3", True), ("AUC ×\u200e10³", True),
    ("AUC ×\u0336 10^3", True),
    ("AUC0–∞ (ng·h/mL)", False), ("AUC0-inf", False), ("Cmax (ng/mL)", False), ("Day 1", False),
    ("CLint (µL/min/10^6 cells)", False), ("mL/10^6 cells", False), ("per 10^6 cells", False),
    ("cells / 10^6", False), ("Dose (mg)", False), ("t1/2 (h)", False), ("", False), ("ng.h/mL", False),
    ("150 mg", False), ("e", False), ("Test", False),
    # r7 round 3: were ("Group 10", False) and ("AUC\u202eabc", False). scale_shapes is an allow-list
    # (numgrammar/scale.py): a written-out power of ten in a label may be a scale, and a word the reader cannot read
    # (U+202E) raises the flag: the reviewer compares with the page image
    ("Group 10", True), ("AUC\u202eabc", True),
])
def test_scale_shapes_flags_a_multiplier_and_ignores_unit_denominators_and_labels(label, expected):
    assert scale_shapes(label) is expected, label


def test_a_tainted_auc_header_does_not_raise_scaled_header_but_a_hidden_multiplier_does():
    assert scale_shapes("AUC0–∞") is False and scale_shapes("AUC† (ng/mL)") is False
    assert scale_shapes("×\u200d10^3") is True and scale_shapes("x† 10^3") is True


@pytest.mark.parametrize("text, expected", [
    ("12", True), ("12.4 (3.1)", True), ("3,412 (812)", True), ("−14.2***", True), ("<0.5", True),
    ("18 900 ± 4 100", True), ("5.3 ± 0.4", True), ("45%", True), ("1.2e-3", True), ("12/24", True),
    ("1.2 × 10^3", True), ("·", False), ("Day 1", False), ("n=12", False), ("12 mg", False),
    ("", False), ("-", False), ("(n)", False), ("AUC", False), ("12†", False),
])
def test_is_data_cell(text, expected):
    assert is_data_cell(text) is expected


def test_find_n_reads_n_equals_an_integer():
    assert find_n("The study (n=12) and N = 40; also n = 1234567 and n=5.5 and an=7 and Fn=3") == [(13, 15, 12),
                                                                                                    (25, 27, 40)]
    assert [x[2] for x in find_n("n \n= \n 6, n=7")] == [6, 7] and find_n("n = 12†") == []
    assert find_n("n=0") == [(2, 3, 0)] and find_n("n = 007") == [] and find_n("n=-3") == []


def test_find_dose_reads_a_number_followed_by_a_dose_unit_word():
    text = "a dose of 150 mg, then 2.5 mg/kg and 20 µg and 1,5 g and 150mg and 3 ml and 100000 mg"
    got = [(text[s:e], v, u) for s, e, v, u in find_dose(text)]
    assert got == [("150", 150.0, "mg"), ("2.5", 2.5, "mg/kg"), ("20", 20.0, "μg"), ("1,5", 1.5, "g")]
    assert find_dose("5 mg.") == [(0, 1, 5.0, "mg")] and find_dose("5 mg†") == []


def test_parse_claim_is_the_one_token_reading_of_a_proposed_value():
    assert parse_claim("12.4 ng/mL") == 12.4 and parse_claim("3,412 (812)") is None
    assert parse_claim("1.2 x 10^3") == 1200.0 and parse_claim("") is None


def test_every_export_resolves_and_the_package_surface_is_complete():
    import app.evidence.numgrammar as NG
    assert not hasattr(NG, "LINE") and "LINE" not in NG.__all__                 # r7: the LINE list is deleted
    for name in ("INSIDE", "CONTEXT", "GAP", "EQUIV", "CID_MARKER", "cls", "Tok", "tokens", "parse_claim",
                 "number_spans", "numbers_in", "words_after", "words_before", "is_data_cell", "find_n_lead"):
        assert hasattr(NG, name), name
    for gone in ("words_around", "token_at", "find_n", "is_digit_run"):          # no caller in app/: test helpers now
        assert not hasattr(NG, gone) and gone not in NG.__all__, gone
    # r7 round 4: the helpers that read a word list or a letter table live outside the reader package
    from app.evidence import dose_rule, scale_label, word_match
    for gone, home in (("scale_shapes", scale_label), ("loose_in", word_match), ("find_dose", dose_rule)):
        assert not hasattr(NG, gone) and gone not in NG.__all__ and callable(getattr(home, gone)), gone
    assert tokens("1") and NG.cls("a") == "context"
