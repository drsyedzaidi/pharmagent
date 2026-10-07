"""r5b, rewritten for the r6 numeric grammar: the qualifier / range window is one WORD, and the reader is linear.

A '<' or a range bound pushed away from its number by combining marks, invisible characters or unreadable characters
used to be flagged through a cleaned 24-character window. Now there is no cleaning and no character window: the
qualifier is the word before the token (a gap run of ANY length and mix is ONE separator, so whitespace padding never
widens it), and a character outside the allowed alphabet between a qualifier and its number, or anywhere on the number's
line, makes the number unresolved: verification fails (unresolved_context) and the reviewer compares with the page image.
tokens() on a page with tens of thousands of invisible characters stays near the cost of the same page without them.
"""
from __future__ import annotations

import time

import pytest

from app.evidence import models as M
from app.evidence.numgrammar import tokens
from app.evidence.record_checks import source_unit
from app.evidence.value_checks import censored_flags
from tests.evidence_r5b_support import check, doc
from tests.evidence_timing import limit

ZWSP = "​"


def _marks(n: int) -> str:
    return "".join(chr(0x300 + i % 0x70) for i in range(n))


def _unresolved(value: str, page: str, snippet: str | None = None) -> None:
    """The record fails closed with unresolved_context (was: passes flagged censored, or fails: either)."""
    res = check(value, snippet or value, page)
    assert (res.status, res.reason) == ("fail", M.REASON_UNRESOLVED_CONTEXT), (page[:80], res)


def _flagged(value: str, page: str) -> None:
    res = check(value, value, page)
    assert res.status == "pass", (page[:60], res)
    p = {"field": "Cmax", "locator": res.locator, "verify_status": "pass"}
    assert censored_flags(p, doc(page)["pages"][0]), repr(page[:80])


@pytest.mark.parametrize("n", [1, 5, 15, 22, 23, 24, 25, 30, 40, 200])
def test_marks_between_a_qualifier_and_the_number_fail_closed(n):
    _unresolved("12.4", f"Cmax was <{_marks(n)} 12.4 ng/mL")
    _unresolved("12.4", f"Cmax was ≤{_marks(n)} 12.4 ng/mL")


@pytest.mark.parametrize("n", [1, 22, 23, 24, 30, 40])
def test_marks_inside_a_range_fail_closed(n):
    _unresolved("12.4", f"Cmax was 5.0 -{_marks(n)} 12.4 ng/mL")
    _unresolved("12.4", f"Cmax was 5.0 - {_marks(n)} 12.4 ng/mL")


@pytest.mark.parametrize("n", [24, 30, 100])
def test_invisible_padding_inside_a_range_fails_closed(n):
    _unresolved("12.5", f"AUC 12.5{ZWSP * n}-15.0 mg/L")
    _unresolved("12.5", f"AUC 12.5{_marks(n)} -15.0 mg/L")


@pytest.mark.parametrize("n", [24, 30, 100])
def test_long_whitespace_does_not_hide_a_qualifier_but_invisible_padding_fails(n):
    _flagged("12.5", "AUC <" + " " * n + "12.5 mg/L")
    _flagged("12.5", "AUC <" + " \n\t" * n + "12.5 mg/L")
    _unresolved("12.5", "AUC <" + ZWSP * n + " 12.5 mg/L")


@pytest.mark.parametrize("ch", ["￿", "�", "", "͸", "\x7f", "\x00", "￼", "\U000f0000"])
def test_an_unreadable_character_between_qualifier_and_number_fails_closed(ch):
    _unresolved("12.5", f"AUC <{ch}12.5 mg/L")
    _unresolved("12.5", f"AUC <{ch} 12.5 mg/L")
    _unresolved("12.5", f"AUC 12.5{ch}-15.0 mg/L")
    _unresolved("12.5", f"AUC 12.5 {ch} - 15.0 mg/L")


def test_a_qualifier_buried_under_more_padding_than_can_be_read_fails_closed():
    _unresolved("12.5", "AUC <" + ZWSP * 5000 + " 12.5 mg/L")
    _unresolved("12.5", "AUC 12.5" + ZWSP * 5000 + " -15.0 mg/L")


def test_whitespace_padding_of_any_length_is_one_separator():
    """5000 spaces between a qualifier and its number are ONE gap run: the qualifier is still the word before."""
    _flagged("12.5", "AUC <" + " " * 5000 + "12.5 mg/L")
    _flagged("12.5", "AUC 12.5" + " " * 5000 + "-15.0 mg/L")


def test_plain_values_are_not_flagged_by_the_wider_window():
    for page in ("AUC 12.4 ng/mL", "p < 0.05 and AUC 12.4 ng/mL", "n < " + "a" * 30 + " 12.4"):
        res = check("12.4", "12.4", page)
        assert res.status == "pass", page[:50]
        p = {"field": "Cmax", "locator": res.locator, "verify_status": "pass"}
        assert censored_flags(p, doc(page)["pages"][0]) == [], page[:50]


@pytest.mark.parametrize("page", ["AUC​ 12.4​ ng/mL", "AUC was" + _marks(30) + " 12.4 ng/mL",
                                  "Cmax" + _marks(60) + " 12.4"])
def test_a_plain_value_with_invisible_characters_or_marks_fails_closed(page):
    """Was test_plain_values_are_not_flagged_by_the_wider_window rows (pass, no flag): the marks and invisible
    characters touch the number's word or its neighbour, so the number is unresolved_context."""
    _unresolved("12.4", page)


def test_qualifier_is_the_word_before_the_token():
    """Was test_window_is_twenty_four_visible_characters (QUALIFIER_CHARS == 24): there is no character window. The
    qualifier is the word (or the two words 'less than', 'up to', 'at least') before the token."""
    _flagged("12.4", "Cmax ≤ " + " " * 3 + "12.4")
    _flagged("12.4", "AUC was below" + " " * 2 + "12.4")
    _flagged("12.4", "AUC was less than 12.4")
    _flagged("12.4", "dose up to 12.4")
    for page in ("AUC was below the LLOQ of 12.4", "AUC increased to 12.4", "AUC was 12.4"):
        res = check("12.4", "12.4", page)
        assert res.status == "pass", page
        p = {"field": "Cmax", "locator": res.locator, "verify_status": "pass"}
        assert censored_flags(p, doc(page)["pages"][0]) == [], page


def test_the_printed_unit_behind_invisible_padding_fails_closed():
    """Was test_the_printed_unit_is_found_behind_invisible_padding (70 ZWSP then 'mg/L' still read as the unit)."""
    _unresolved("12.4", "AUC 12.4" + ZWSP * 70 + " mg/L")
    res = check("12.4", "12.4", "AUC 12.4 mg/L")
    unit, decided, _ = source_unit({"locator": res.locator}, doc("AUC 12.4 mg/L")["pages"][0])
    assert unit == "mg/l" and decided


# ---------------------------------------------------------------- linear ---

def test_tokens_is_linear_on_a_page_of_invisible_characters():
    text = ("1 " + ZWSP) * 60_000 + " Cmax was 12.4 ng/mL"
    control = ("1 " + " ") * 60_000 + " Cmax was 12.4 ng/mL"
    assert len(text) < 200_000
    t0 = time.monotonic()
    toks = tokens(text, ".")
    slow = time.monotonic() - t0
    t0 = time.monotonic()
    tokens(control, ".")
    fast = time.monotonic() - t0
    assert toks[-1].value is None and toks[-1].reason == "unresolved_context"       # the line tier: ZWSP is on the line
    assert slow < limit(max(1.5, 8 * fast)), (slow, fast)


def test_verify_of_one_record_on_a_padded_page_is_fast_and_fails_closed():
    page = ("1 " + ZWSP) * 60_000 + " Cmax was 12.4 ng/mL"
    t0 = time.monotonic()
    res = check("12.4", "Cmax was 12.4", page)
    assert (res.status, res.reason) == ("fail", M.REASON_UNRESOLVED_CONTEXT), res
    assert time.monotonic() - t0 < limit(3.0)
