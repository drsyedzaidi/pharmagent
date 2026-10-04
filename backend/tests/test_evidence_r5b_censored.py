"""r5b: the qualifier / range window is read from the CLEANED text, and the cleaner is linear.

A '<' or a range bound pushed out of a raw 24-character window by combining marks, invisible
characters or long whitespace must still flag the value as censored (or the record must fail);
unreadable characters between a qualifier and its number are skipped. number_tokens on a page with
tens of thousands of invisible characters stays near the cost of the same page without them.
"""
from __future__ import annotations

import time

import pytest

from app.evidence.invisible import clean
from app.evidence.normalize import number_tokens
from app.evidence.record_checks import source_unit
from app.evidence.value_checks import QUALIFIER_CHARS, censored_flags
from tests.evidence_r5b_support import check, doc
from tests.evidence_timing import limit

ZWSP = "\u200b"


def _marks(n: int) -> str:
    return "".join(chr(0x300 + i % 0x70) for i in range(n))


def _flagged_or_failed(value: str, page: str, snippet: str | None = None) -> None:
    res = check(value, snippet or value, page)
    if res.status != "pass":
        return
    p = {"field": "Cmax", "locator": res.locator, "verify_status": "pass"}
    assert censored_flags(p, doc(page)["pages"][0]), repr(page[:80])


def _flagged(value: str, page: str) -> None:
    res = check(value, value, page)
    assert res.status == "pass", (page[:60], res)
    p = {"field": "Cmax", "locator": res.locator, "verify_status": "pass"}
    assert censored_flags(p, doc(page)["pages"][0]), repr(page[:80])


@pytest.mark.parametrize("n", [1, 5, 15, 22, 23, 24, 25, 30, 40, 200])
def test_marks_between_a_qualifier_and_the_number_do_not_hide_it(n):
    _flagged_or_failed("12.4", f"Cmax was <{_marks(n)} 12.4 ng/mL")
    _flagged_or_failed("12.4", f"Cmax was \u2264{_marks(n)} 12.4 ng/mL")


@pytest.mark.parametrize("n", [1, 22, 23, 24, 30, 40])
def test_marks_inside_a_range_do_not_hide_the_lower_bound(n):
    _flagged("12.4", f"Cmax was 5.0 -{_marks(n)} 12.4 ng/mL")
    _flagged("12.4", f"Cmax was 5.0 - {_marks(n)} 12.4 ng/mL")


@pytest.mark.parametrize("n", [24, 30, 100])
def test_invisible_padding_does_not_hide_a_range_upper_bound(n):
    _flagged("12.5", f"AUC 12.5{ZWSP * n}-15.0 mg/L")
    _flagged_or_failed("12.5", f"AUC 12.5{_marks(n)} -15.0 mg/L")      # marks on the digit: fails closed


@pytest.mark.parametrize("n", [24, 30, 100])
def test_long_whitespace_does_not_hide_a_qualifier(n):
    _flagged("12.5", "AUC <" + " " * n + "12.5 mg/L")
    _flagged("12.5", "AUC <" + ZWSP * n + " 12.5 mg/L")
    _flagged("12.5", "AUC <" + " \n\t" * n + "12.5 mg/L")


@pytest.mark.parametrize("ch", ["\uffff", "\ufffd", "\ue000", "\u0378", "\x7f", "\x00", "\ufffc", "\U000f0000"])
def test_an_unreadable_character_between_qualifier_and_number_is_skipped(ch):
    _flagged_or_failed("12.5", f"AUC <{ch}12.5 mg/L")
    _flagged_or_failed("12.5", f"AUC <{ch} 12.5 mg/L")
    _flagged_or_failed("12.5", f"AUC 12.5{ch}-15.0 mg/L")
    _flagged_or_failed("12.5", f"AUC 12.5 {ch} - 15.0 mg/L")


def test_a_qualifier_buried_under_more_padding_than_can_be_read_is_flagged():
    _flagged("12.5", "AUC <" + ZWSP * 5000 + " 12.5 mg/L")
    _flagged("12.5", "AUC 12.5" + ZWSP * 5000 + " -15.0 mg/L")


def test_plain_values_are_not_flagged_by_the_wider_window():
    for page in ("AUC 12.4 ng/mL", "p < 0.05 and AUC 12.4 ng/mL", "n < " + "a" * 30 + " 12.4", "AUC\u200b 12.4\u200b ng/mL",
                 "AUC was" + _marks(30) + " 12.4 ng/mL", "Cmax" + _marks(60) + " 12.4"):
        res = check("12.4", "12.4", page)
        assert res.status == "pass", page[:50]
        p = {"field": "Cmax", "locator": res.locator, "verify_status": "pass"}
        assert censored_flags(p, doc(page)["pages"][0]) == [], page[:50]


def test_window_is_twenty_four_visible_characters():
    assert QUALIFIER_CHARS == 24
    _flagged("12.4", "Cmax \u2264 " + " " * 3 + "12.4")
    p_far = "AUC was below" + " " * 2 + "12.4"
    _flagged("12.4", p_far)


def test_the_printed_unit_is_found_behind_invisible_padding():
    page = "AUC 12.4" + ZWSP * 70 + " mg/L"
    res = check("12.4", "12.4", page)
    assert res.status == "pass"
    unit, decided, _ = source_unit({"locator": res.locator}, doc(page)["pages"][0])
    assert unit == "mg/l" and decided


# ---------------------------------------------------------------- linear ---

def test_clean_is_memoised_per_text_and_number_tokens_is_not_quadratic():
    text = ("1 " + ZWSP) * 60_000 + " Cmax was 12.4 ng/mL"
    control = ("1 " + " ") * 60_000 + " Cmax was 12.4 ng/mL"
    assert len(text) < 200_000
    t0 = time.monotonic()
    toks = number_tokens(text, strict=True)
    slow = time.monotonic() - t0
    t0 = time.monotonic()
    number_tokens(control, strict=True)
    fast = time.monotonic() - t0
    assert toks[-1].value == 12.4
    assert slow < limit(max(1.5, 8 * fast)), (slow, fast)
    before = clean.cache_info().hits
    number_tokens(text, strict=True)
    assert clean.cache_info().hits == before + 1


def test_verify_of_one_honest_record_on_a_padded_page_is_fast():
    page = ("1 " + ZWSP) * 60_000 + " Cmax was 12.4 ng/mL"
    t0 = time.monotonic()
    res = check("12.4", "Cmax was 12.4", page)
    assert res.status == "pass"
    assert time.monotonic() - t0 < limit(3.0)
