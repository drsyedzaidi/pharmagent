"""r5b: scientific notation the grammar cannot read never passes as its mantissa, and a symbol that is
not a real multiplier never turns two numbers into one.

A look-alike or non-ASCII 'ten', an exponent outside the grammar (subscript, circled, '^' with a
gap, '**', arrows), a multiplier that is not on the allow-list and junk between multiplier and ten
all leave the token UNRESOLVED, whatever the page's small glyphs say (bare_ten). Line and paragraph
separators are gaps, not multipliers.
"""
from __future__ import annotations

import pytest

from app.evidence.normalize import number_tokens
from tests.evidence_r5b_support import check

TENS = ["\uff11\uff10", "\U0001d7cf\U0001d7ce", "\u0661\u0660", "\u2469", "\u2169", "l0", "1O", "\u04060",
        "\u00b9\u2070", "\u2081\u2080", "I0", "1o", "\u0399\u039f"]
EXPONENTS = ["\u2083", "\u2462", "^\u00b3", "^\u207b\u00b3", "\u207d\u00b3\u207e", "\u2212\u00b3", " ^3",
             " ^ 3", "**3", " \u00b3", "\u207f", "\u21913", "\u02c63", "\u22273", "^\uff13", "\u2074"]
SEPARATORS = ["\u2028", "\u2029", "\u0085", "\u1680", "\u2800", "\uffff", "\ue000", "\ufffd", "\u0378"]


def _unresolved(page, small, cell=False):
    res = check("1.2", "1.2", page, small, cell)
    assert res.status == "fail", (page, small, res)


@pytest.mark.parametrize("small", ["", "3"])
@pytest.mark.parametrize("cell", [False, True])
@pytest.mark.parametrize("ten", TENS)
def test_a_lookalike_ten_after_a_multiplier_is_unresolved(ten, cell, small):
    for exp in ("\u00b3", "^3", " 3", ""):
        _unresolved(f"Cmax 1.2 \u00d7 {ten}{exp} ng/mL", small, cell)
        _unresolved(f"Cmax 1.2 x{ten}{exp} ng/mL", small, cell)


@pytest.mark.parametrize("small", ["", "3"])
@pytest.mark.parametrize("cell", [False, True])
@pytest.mark.parametrize("exp", EXPONENTS)
def test_an_exponent_outside_the_grammar_is_unresolved_even_without_small_glyphs(exp, cell, small):
    _unresolved(f"Cmax 1.2 \u00d7 10{exp} ng/mL", small, cell)
    _unresolved(f"Cmax 1.2 x 10{exp} ng/mL", small, cell)


def test_the_exponent_digits_are_part_of_the_unresolved_token():
    for page in ("Cmax 1.2 \u00d7 10 ^ 3 ng/mL", "Cmax 1.2 \u00d7 10**3 ng/mL", "Cmax 1.2 \u00d7 10\u2083 ng/mL"):
        for small in ("", "3"):
            res = check("3", "3", page, small)
            assert res.status == "fail", (page, small, res)
            res = check("10", "10", page, small)
            assert res.status == "fail", (page, small, res)


@pytest.mark.parametrize("sep", SEPARATORS)
@pytest.mark.parametrize("cell", [False, True])
def test_a_separator_or_unreadable_symbol_is_not_a_multiplier(sep, cell):
    page = f"Cmax 1.2{sep}10^3 ng/mL"
    res = check("1200", "1.2" + sep + "10^3", page, "3", cell)
    assert res.status == "fail", (sep, res)
    res = check("1200", "1.2" + sep + "10^3", page, "", cell)
    assert res.status == "fail", (sep, res)


@pytest.mark.parametrize("junk", ["\uffff", "\ufffd", "\ue000", "\u0378", "\u2028", "\u2029", "\u2800"])
@pytest.mark.parametrize("cell", [False, True])
def test_junk_between_multiplier_and_ten_is_unresolved(junk, cell):
    for small in ("", "3"):
        _unresolved(f"Cmax 1.2 \u00d7 {junk}10\u00b3 ng/mL", small, cell)
        _unresolved(f"Cmax 1.2 \u00d7{junk}10^3 ng/mL", small, cell)


def test_line_separators_stay_two_numbers_like_a_plain_newline():
    for sep in ("\n", "\u2028", "\u2029", "\u0085"):
        toks = number_tokens(f"1.2{sep}10^3", strict=True)
        assert [t.value for t in toks] == [1.2, 1000.0]


def test_line_separators_are_not_digit_group_separators():
    for sep in ("\u2028", "\u2029", "\u0085", "\u1680", "\u2800"):
        toks = number_tokens(f"12{sep}145", strict=True)
        assert [t.value for t in toks] == [12.0, 145.0], sep


@pytest.mark.parametrize("mult", ["\u00d7", "x", "X", "*", "\u00b7", "\u22c5", "\u2219", "\u2022", "\u2a2f",
                                  "\u2715", "\u2716", "\u0445", "\u0425", "\u2217"])
def test_real_multipliers_still_read_the_exponent(mult):
    res = check("1200", f"1.2 {mult} 10^3", f"Cmax 1.2 {mult} 10^3 ng/mL", "3")
    assert res.status == "pass" and res.number == pytest.approx(1200.0), (mult, res)


def test_honest_scientific_notation_and_plain_text_still_pass():
    for page, snip, value, want in (("Cmax 1.2 \u00d7 10^3 ng/mL", "1.2 \u00d7 10^3", "1200", 1200.0),
                                    ("Cmax 1.2e3 ng/mL", "1.2e3", "1200", 1200.0),
                                    ("Cmax 1.2 x 10^-3 ng/mL", "1.2 x 10^-3", "0.0012", 0.0012),
                                    ("Cmax 2 x 100 mg", "2", "2", 2.0), ("n = 12 x 10 subjects", "12", "12", 12.0),
                                    ("Cmax 1.2 mg and 10 subjects", "1.2", "1.2", 1.2)):
        res = check(value, snip, page, "")
        assert res.status == "pass" and res.number == pytest.approx(want), (page, res)
