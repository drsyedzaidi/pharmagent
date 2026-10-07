"""r6 round-1 regressions at the grammar level: what the tokenizer, the style evidence and the word helpers promise for
the shapes the round-1 audit found (each case is a defect found by a PoC that failed before the fix)."""
from __future__ import annotations

import pytest

from app.evidence.normalize import decimal_evidence, infer_decimal_style
from app.evidence.numgrammar import (
    NUMBER_CHARS,
    claim_decimal,
    digit_group,
    digits_of,
    is_digit_run,
    loose_in,
    tokens,
)
from app.evidence.numgrammar import alphabet as A
from app.evidence.numgrammar import words as W


def vals(text: str, style: str = ".", mode: str = "page") -> list[float | None]:
    return [t.value for t in tokens(text, style, mode=mode)]


def why(text: str, style: str = ".") -> list[str]:
    return [t.shape for t in tokens(text, style)]


# --------------------------------------------- style evidence is counted from the grammar ---

DECOYS = {
    "arabic-indic": "٣,٥ ٤,٥",
    "fullwidth": "３.５ ４.５",
    "extended arabic-indic": "۳,۵ ۴,۵",
    "mathematical": "\U0001d7d1,\U0001d7d3 \U0001d7d2,\U0001d7d3",
}


@pytest.mark.parametrize("name", sorted(DECOYS))
def test_digits_outside_the_alphabet_vote_for_no_decimal_style(name):
    """Decoy lines of non-ASCII digits (OUTSIDE the alphabet) never flip 12.345 to 12345 or 12,345 to 12.345."""
    pad = "\n".join([DECOYS[name]] * 8) + "\n"
    point = "AUC was 12.345 mg.h/L in healthy volunteers.\n"
    comma = "Die AUC betrug 12,345 mg.h/L; Cmax 3,4 und 5,6 und 7,8 und 9,1 mg/L.\n"
    assert decimal_evidence(point + pad) == decimal_evidence(point) == (0, 1)
    assert infer_decimal_style(point + pad) == "."
    assert decimal_evidence(comma + pad) == decimal_evidence(comma) and infer_decimal_style(comma + pad) == ","


def test_the_style_evidence_comes_from_the_resolved_numbers_of_the_page_reading():
    assert decimal_evidence("CL 2.31 L/h V 45.2 L ka 1.05 per h") == (0, 3)
    assert decimal_evidence("CLint 18,2 ± 3,4 and fu 0,12") == (3, 0)
    assert decimal_evidence("values 1.234,5 and 2.345,6") == (2, 0)           # a point group and a decimal comma
    assert decimal_evidence("counts 1,200 and 3,400 only") == (0, 0)          # thousands or decimals: no evidence
    assert decimal_evidence("1.250.000 and 24'560 and 1.2.3 and 12.05.2020") == (0, 0)   # groups, versions, dates
    assert decimal_evidence("k = 1.2e-3 and 2,5E4") == (1, 1)                  # the mantissa of a power votes too
    assert decimal_evidence("12.5 ng/mL \u200b and 3.4 h") == (0, 0)         # one line with an outside character: none
    assert decimal_evidence("12.5 ng/mL \u200b\nand 3.4 h") == (0, 1)       # the next line is its own


def test_a_reference_pair_in_square_brackets_is_no_number_and_no_evidence():
    assert decimal_evidence("Drug X [1,2] and [3,4] and [5,6], t1/2 was 11.4 h") == (0, 1)
    for style in (".", ","):
        assert [(t.value, t.why) for t in tokens("see [1,2] here", style)] == [(None, "citation_list")]
    assert vals("[0,5] mg", ",") == [0.5]                                   # a bracketed decimal with head 0 is no pair


def test_decimal_evidence_is_linear_in_a_run_of_digit_groups():
    import time
    t0 = time.perf_counter()
    assert decimal_evidence("1" + ".234" * 40_000) == (0, 0)
    assert time.perf_counter() - t0 < 5.0                                       # the old pattern took 40 s on 128 KB


# --------------------------------------------------- Greek chi is a look-alike multiplier ---

def test_greek_chi_is_a_line_character_and_taints_the_number_it_stands_beside():
    assert "χ" in A.LINE and "χ" not in A.CONTEXT and len(A.GREEK) == 16
    for text in ("AUC was 1.2 χ 10^3 ng.h/mL", "AUC was 1.2 χ10^3 ng.h/mL", "AUC 1.2 χ 10³ ng.h/mL"):
        assert all(t.value is None for t in tokens(text)), text
    assert vals("AUC was 1.2 x 10^3 ng.h/mL") == [1200.0]                       # the real multiplier is unchanged
    assert vals("χ² = 3.4") == [3.4]                                           # an operator between: not within a word


# ----------------------------------------------------------- qualifiers hidden by a starter ---

QUALIFIED = ["Cmax ≤(12.5) ng/mL", "Cmax ≈(12.5)", "Cmax ≥[12.5]", "Cmax ~=12.5", "Cmax >=(12.5)",
             "Cmax <(12.5)", "Cmax ~= 12.5", "Cmax < = 12.5", "Cmax >  = 12.5", "Cmax <- 12.5", "Cmax <( 12.5)",
             "Cmax <+ 12.5", "Cmax <: 12.5", "Cmax <* 12.5", "Cmax <; 12.5", "Cmax < ( 12.5 ) ng/mL", "Cmax =~ 12.5",
             "Cmax (<12.5)", "Cmax <= 12.5", "Cmax < (12.5)", "Cmax ∼ 12.5", "Cmax <\n12.5", "Cmax <(\n12.5)"]


@pytest.mark.parametrize("text", QUALIFIED)
def test_a_qualifier_is_never_hidden_by_an_opener_an_equals_sign_or_a_gap(text):
    got = [t for t in tokens(text) if t.value == 12.5]
    assert len(got) == 1 and got[0].qualified, text


@pytest.mark.parametrize("text", ["Cmax = 12.5", "Cmax ( 12.5 )", "Cmax: 12.5", "n=12.5", "Cmax (12.5)", "Cmax 12.5 <"])
def test_an_operator_run_without_a_qualifier_does_not_qualify(text):
    assert [t.qualified for t in tokens(text) if t.value == 12.5] == [False]


def test_the_word_helper_reads_runs_of_operator_words():
    assert W.qualifies(["<", "="]) and W.qualifies(["Cmax<"]) and W.qualifies(["a", "≤("])
    assert not W.qualifies(["=", "("]) and not W.qualifies(["Cmax", "="]) and not W.qualifies([])
    assert not W.qualifies(["<", "word", "="])                       # a word between: the qualifier is not the last run


# ------------------------------------------------------- ranges written with one dash ---

def _ranged(text: str, style: str = ".") -> list[tuple[float, bool]]:
    return [(t.value, t.ranged) for t in tokens(text, style) if t.value is not None]


@pytest.mark.parametrize("text, want", [
    ("AUC was 12 -15 mg", [(12.0, True), (-15.0, True)]),
    ("AUC was 12.5 -15.0 mg", [(12.5, True), (-15.0, True)]),
    ("dose 5.0- 12.4 mg", [(5.0, True), (12.4, True)]),
    ("dose 0.5-\n1.5 mg", [(0.5, True), (1.5, True)]),
    ("Cmax 12.5 ∼ 15.0 ng/mL", [(12.5, True), (15.0, True)]),
    ("Cmax 12.5 - 15.0 ng/mL", [(12.5, True), (15.0, True)]),
    ("range 12–15", [(12.0, True), (15.0, True)]),
])
def test_both_ends_of_a_range_are_marked_whatever_the_spacing(text, want):
    assert _ranged(text) == want


@pytest.mark.parametrize("text", ["AUC was 12.5 mg", "Cmax -15.0 ng/mL", "12 mg and 15 mg", "Day 5- fold change", "5-year 12"])
def test_a_lone_number_a_negative_first_word_and_prose_with_a_dash_are_not_ranges(text):
    assert not any(r for _, r in _ranged(text))


# --------------------------------------------------------- ratios, fragments and brackets ---

@pytest.mark.parametrize("text", ["Cmax 12 / 15 patients", "Cmax 12 /15 patients", "Cmax 12/ 15 patients", "Ratio 1 : 2 dilution",
                                  "Date 5 / 12 / 2020 here", "score 3 :4", "n 12 \n/\n 15", "12 : 13"])
def test_a_slash_or_colon_between_two_numbers_makes_them_one_unresolved_ratio(text):
    """'12 : 13' used to sit in test_evidence_r6_line_tier's list of 'context characters one word away are fine' (the colon
    is a CONTEXT character and never taints): it still never taints, and it is no longer two exact values either."""
    got = tokens(text)
    assert got and all(t.value is None and t.shape == "unresolved_shape:fraction" for t in got), (text, got)


@pytest.mark.parametrize("text, want", [("dose 12 mg / 15 mL", [12.0, 15.0]), ("Cmax 12 ng / mL", [12.0]),
                                        ("Table 3: 12 patients", [3.0, 12.0]), ("n = 12 / kg", [12.0])])
def test_a_slash_or_colon_that_joins_no_two_numbers_changes_nothing(text, want):
    assert vals(text) == want
    assert vals("time 12:30") == [None]                       # glued: unresolved before and after the round


@pytest.mark.parametrize("text", ["Cmax 12 .5 ng/mL", "Cmax 12 ,5 ng/mL", "Cmax 12 .50 ng/mL", "Cmax 12\n.5"])
def test_a_whole_number_beside_its_detached_decimal_tail_is_one_unresolved_expression(text):
    got = tokens(text)
    assert [(t.value, t.shape) for t in got] == [(None, "unresolved_shape:detached_decimal")], text


@pytest.mark.parametrize("text", ["Cmax 5.2 (x10^3 cells/mL)", "Cmax 5.2 [x10^3] cells/mL", "Cmax 5.2 (x 10^3 cells/mL)",
                                  "Cmax 5.2 (×10^3)", "Cmax 5.2 (X10³ cells)", "Cmax 5.2 [x\n10^3]"])
def test_a_scale_in_a_bracket_after_the_number_leaves_the_number_unresolved(text):
    got = tokens(text)
    assert got and got[0].value is None and got[0].shape == "unresolved_shape:bracketed_multiplier", (text, got)


@pytest.mark.parametrize("text", ["Cmax 5.2 (ng/mL)", "Cmax 5.2 (xenon)", "Cmax 5.2 (x ng/mL)", "Cmax 5.2 [n=12]"])
def test_a_bracket_without_a_power_of_ten_is_not_a_scale(text):
    assert tokens(text)[0].value == 5.2


# --------------------------------------------- a mantissa is never a thousands group ---

def test_one_separator_before_an_exponent_is_always_a_decimal_point():
    for style, text, want in [(",", "k = 1.234e+03", 1234.0), (",", "k = 2.345E-3", 0.002345), (".", "k = 1,234E3", 1234.0),
                              (".", "k = 1.234e3", 1234.0), (",", "k = 5.678e-4", 0.0005678), (",", "k = 1,234e3", 1234.0)]:
        assert vals(text, style) == [want], (style, text)
    assert vals("Zellen 1.234 x 10^3 /mL", ",") == [1234.0]
    assert vals("cells 1,234 x 10^3 /mL", ".") == [1234.0]
    assert vals("Zellen 1.234 ×10^3", ",") == [1234.0]
    assert vals("cells 1.234 x 10^3", ".") == [1234.0]


def test_a_foreign_mantissa_carries_its_separator_so_the_style_flag_can_ask():
    got = [t for t in tokens("k = 1.234e+03", ",") if t.value is not None]
    assert len(got) == 1 and got[0].shape == "sci" and got[0].decimal == "."
    assert [t.decimal for t in tokens("k = 1,234e+03", ",") if t.value is not None] == [","]


# --------------------------------------------------------- helpers for the callers ---

def test_digit_helpers_read_only_ascii_digits():
    assert digits_of("1٢₃3,4５") == "13" + "4" and digits_of("") == ""
    assert is_digit_run("123", 1, 3) and not is_digit_run("1234", 1, 3) and not is_digit_run("١", 1, 1)
    assert is_digit_run("", 1, 3) is False


@pytest.mark.parametrize("word, lo, hi, kw, want", [
    ("145", 2, 3, {}, True), ("234.5", 3, 3, {"decimals": True}, True), ("234,56", 3, 3, {"decimals": True}, True),
    ("234.", 3, 3, {"decimals": True}, True), ("234)", 3, 3, {}, True), ("234*", 3, 3, {}, True),
    ("234.5", 3, 3, {}, False), ("2345", 3, 3, {"decimals": True}, False), ("-15", 2, 3, {}, False),
    ("-1", 1, 3, {"lead": True}, True), ("(1", 1, 3, {"lead": True}, True), ("(-1", 1, 3, {"lead": True}, True),
    ("12.4", 1, 3, {"lead": True}, False), ("١٢٣", 3, 3, {}, False), ("", 1, 3, {}, False),
])
def test_digit_group_matches_a_group_a_text_layer_may_have_split(word, lo, hi, kw, want):
    assert digit_group(word, lo, hi, **kw) is want


def test_loose_in_lets_every_non_ascii_character_stand_for_any_one():
    words = frozenset({"to", "above", "thousands"})
    for w in ("to", "TO", "τo", "αbove", "abοve", "thоusands", "το"):
        assert loose_in(w, words), w
    for w in ("too", "abov", "toe", "τ", "", "thousandz"):
        assert not loose_in(w, words), w


def test_claim_decimal_is_the_exact_decimal_of_the_one_number():
    from decimal import Decimal
    assert claim_decimal("12.50") == Decimal("12.5") and claim_decimal("1,234", ".") == Decimal("1234")
    assert claim_decimal("3.2e-13") != claim_decimal("0") and claim_decimal("12 13") is None and claim_decimal("") is None
    assert claim_decimal("1234567890123") != claim_decimal("1234567890124")


def test_number_chars_are_the_inside_class_and_the_sources_of_the_equivalents():
    assert NUMBER_CHARS == A.INSIDE | frozenset(A.EQUIV) and "−" in NUMBER_CHARS and "x" in NUMBER_CHARS
    assert "١" not in NUMBER_CHARS and "‒" not in NUMBER_CHARS and "a" not in NUMBER_CHARS


# ------------------------------------------------- invariants with the round-1 shapes in the mix ---

_ATOMS = (list("0123456789") * 5 + list(".,'-+ \t\n") * 3 + ["[1,2]", " / ", " : ", " .5", " ,5", "(x10^3", "[x 10^3]", "<=", "~=",
                                                            " -", "- ", "χ", "∼", "1.234e+03", "e", "x", "×", "10^3",
                                                            "≤(", "(", ")", "%", "±", "mg", "to", "12 -15", " "])


@pytest.mark.parametrize("style", [".", ","])
def test_random_text_with_the_round_1_shapes_stays_well_formed_and_gap_invariant(style):
    import math
    import random
    import re
    rnd = random.Random(31)
    for _ in range(1500):
        s = "".join(rnd.choice(_ATOMS) for _ in range(rnd.randint(1, 30)))
        got = tokens(s, style)
        end = -1
        for t in got:
            assert 0 <= t.start < t.end <= len(s) and t.start >= end, (s, t)
            end = t.end
            assert (t.value is None) == (t.canonical == "" and t.shape.startswith("unresolved")), (s, t)
            if t.value is not None:
                assert math.isfinite(t.value) and t.canonical, (s, t)
        padded = re.sub("[ \t\n\r]+", lambda m: "".join(rnd.choice(" \t\n\r") for _ in range(rnd.randint(1, 5))), s)
        assert [(t.value, t.shape, t.qualified, t.ranged) for t in tokens(padded, style)] == \
            [(t.value, t.shape, t.qualified, t.ranged) for t in got], (s, padded)
        for mode in ("cell", "claim"):
            tokens(s, style, mode=mode)                                  # never raises


def test_the_style_evidence_of_two_texts_is_the_sum_of_their_evidence_when_prose_lies_between():
    import random
    rnd = random.Random(5)
    for _ in range(300):
        a = "".join(rnd.choice(_ATOMS) for _ in range(rnd.randint(1, 14))).replace("\n", " ")
        b = "".join(rnd.choice(_ATOMS) for _ in range(rnd.randint(1, 14))).replace("\n", " ")
        ea, eb, both = decimal_evidence(a), decimal_evidence(b), decimal_evidence(a + "\nok ok\n" + b)
        assert both == (ea[0] + eb[0], ea[1] + eb[1]), (a, b)             # two words of prose end every form
