"""r5b: scientific notation the grammar cannot read never passes as its mantissa, and a symbol that is
not a real multiplier never turns two numbers into one.

A look-alike or non-ASCII 'ten', an exponent outside the grammar (subscript, circled, '^' with a
gap, '**', arrows), a multiplier that is not on the allow-list and junk between multiplier and ten
all leave the token UNRESOLVED, whatever the page's small glyphs say. Line and paragraph separators are
characters outside the allowed alphabet (r6): a number next to one is unresolved_context; only a line feed, a
carriage return, a tab and a space are gaps.
"""
from __future__ import annotations

import pytest

from app.evidence import models as M
from app.evidence.numgrammar import tokens
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


def test_a_plain_newline_keeps_two_numbers_and_the_line_separators_leave_one_unresolved_expression():
    """Was test_line_separators_stay_two_numbers_like_a_plain_newline: U+2028 / U+2029 / U+0085 read as [1.2, 1000.0]
    like a newline. They are outside the allowed alphabet, so the numbers touching them are ONE unresolved
    expression. r7 round 2 (P0 finding): the line feed no longer reads two numbers either: a bare power one gap after a
    number is a detached scale (a multiplier the text layer dropped: 'WBC 1.2\n10^9 cells' is 1.2 x 10^9), so both are
    one unresolved expression that keeps both values as its parts."""
    got = tokens("1.2\n10^3", ".")
    assert [(t.value, t.why, t.parts) for t in got] == [(None, "detached_exponent", (1.2, 1000.0))]
    for sep in ("\u2028", "\u2029", "\u0085"):
        toks = tokens(f"1.2{sep}10^3", ".")
        assert [(t.value, t.reason) for t in toks] == [(None, "unresolved_context")], sep


def test_line_separators_are_not_digit_group_separators():
    """Was [12.0, 145.0] for all five: now one unresolved expression each (outside the alphabet)."""
    for sep in ("\u2028", "\u2029", "\u0085", "\u1680", "\u2800"):
        assert [(t.value, t.reason) for t in tokens(f"12{sep}145", ".")] == [(None, "unresolved_context")], sep


@pytest.mark.parametrize("mult", ["\u00d7", "x", "X", "*", "\u00b7"])
def test_real_multipliers_still_read_the_exponent(mult):
    res = check("1200", f"1.2 {mult} 10^3", f"Cmax 1.2 {mult} 10^3 ng/mL", "3")
    assert res.status == "pass" and res.number == pytest.approx(1200.0), (mult, res)


@pytest.mark.parametrize("mult", ["\u22c5", "\u2219", "\u2022", "\u2a2f", "\u2715", "\u2716", "\u0445", "\u0425",
                                  "\u2217"])
def test_a_lookalike_multiplier_is_outside_the_alphabet_and_never_a_power(mult):
    """Was in test_real_multipliers_still_read_the_exponent (pass, 1200.0): the look-alikes are not in the alphabet
    and not in the equivalents table, so a power they would multiply is unresolved_context, never read."""
    res = check("1200", f"1.2 {mult} 10^3", f"Cmax 1.2 {mult} 10^3 ng/mL", "3")
    # the bullet U+2022 is a LINE character (a word of its own): it taints both neighbours, and the cited 1200 then
    # matches no resolved number (number_mismatch); every other look-alike is outside the alphabet itself
    assert res.status == "fail" and res.reason in (M.REASON_UNRESOLVED_CONTEXT, M.REASON_NUMBER_MISMATCH), (mult, res)
    assert [t.value for t in tokens(f"1.2 {mult} 10^3", ".")] == [None, None], mult


def test_honest_scientific_notation_and_plain_text_still_pass():
    for page, snip, value, want in (("Cmax 1.2 \u00d7 10^3 ng/mL", "1.2 \u00d7 10^3", "1200", 1200.0),
                                    ("Cmax 1.2e3 ng/mL", "1.2e3", "1200", 1200.0),
                                    ("Cmax 1.2 x 10^-3 ng/mL", "1.2 x 10^-3", "0.0012", 0.0012),
                                    ("Cmax 2 x 100 mg", "2", "2", 2.0),
                                    ("Cmax 1.2 mg and 10 subjects", "1.2", "1.2", 1.2)):
        res = check(value, snip, page, "")
        assert res.status == "pass" and res.number == pytest.approx(want), (page, res)


def test_n_equals_twelve_times_ten_is_not_a_plain_twelve():
    """Was a row of test_honest_scientific_notation_and_plain_text_still_pass ('n = 12 x 10 subjects' cites 12:
    pass). '12 x 10 subjects' is a bare ten after a multiplier (unresolved_shape:ten_after_multiplier): the 12 is only
    a part of it."""
    res = check("12", "12", "n = 12 x 10 subjects", "")
    assert (res.status, res.reason) == ("fail", M.REASON_PARTIAL_NUMBER), res
