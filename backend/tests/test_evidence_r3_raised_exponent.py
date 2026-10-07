"""A power of ten whose exponent is set at full size but raised ('1.2 x 10' + raised '3') reads '1.2 × 103'
in the text layer. The exponent cannot be read from the text, so the whole shape is unresolved: citing the
mantissa, the 10 or the digits fails and the whole literal has no value. Explicit exponents (caret,
e-notation) and plain dose products ('2 x 10 mg', '2 x 100 mg') are unchanged."""
from __future__ import annotations

import pytest

from app.evidence import models as M
from app.evidence.extract_rules import value_number
from app.evidence.numgrammar import parse_claim, tokens
from app.evidence.verify import verify


def _doc(text: str, small: str = "") -> dict:
    return {"sha256": "x" * 64, "decimal_style": ".",
            "pages": [{"n": 1, "text": text, "small_text": small, "columns": None, "tables": []}]}


def _prop(value: str, snippet: str) -> dict:
    loc = M.Locator(page=1, kind="text", snippet=snippet, decimal_style=".").to_json()
    return {"source_id": "S1", "field": "Km", "group_label": "", "value": value,
            "value_num": value_number(value, "."), "unit": None, "locator": loc, "extractor": "llm"}


RAISED_PAGES = ["Peak 1.2 × 103 cells", "Peak 1.2 x 103 cells", "Peak 1.2 · 103 cells",
                "Peak 1.2 * 103 cells", "Peak 1.2 x 10 3 cells", "Peak 1.2 × 1012 cells"]


@pytest.mark.parametrize("page", RAISED_PAGES)
def test_full_size_raised_exponent_has_no_readable_value(page):
    toks = tokens(page, ".")
    assert toks[0].value is None and toks[0].parts[0] == 1.2, toks      # the mantissa leads the resolved parts


@pytest.mark.parametrize("page", RAISED_PAGES)
def test_citing_the_mantissa_of_the_raised_shape_fails_partial_number(page):
    for snippet in ("Peak 1.2", page):
        res = verify(_prop("1.2", snippet), _doc(page))
        assert (res.status, res.reason) == ("fail", M.REASON_PARTIAL_NUMBER), (snippet, res)


@pytest.mark.parametrize("page, value, snippet", [
    ("Peak 1.2 × 103 cells", "10", "1.2 × 103"),
    ("Peak 1.2 × 103 cells", "103", "× 103 cells"),
    ("Peak 1.2 x 103 cells", "103", "x 103 cells"),
    ("Peak 1.2 × 1012 cells", "1012", "× 1012 cells"),
    ("Peak 1.2 x 10 3 cells", "3", "10 3 cells"),
])
def test_citing_the_ten_or_the_digits_of_the_raised_shape_fails(page, value, snippet):
    res = verify(_prop(value, snippet), _doc(page))
    assert res.status == "fail", res
    if value != "10":
        assert res.reason == M.REASON_PARTIAL_NUMBER, res


@pytest.mark.parametrize("page, literal", [("Peak 1.2 × 103 cells", "1.2 × 103"),
                                           ("Peak 1.2 x 10 3 cells", "1.2 x 10 3")])
def test_whole_literal_of_the_raised_shape_never_passes(page, literal):
    res = verify(_prop(literal, f"Peak {literal}"), _doc(page))
    assert (res.status, res.reason) == ("fail", M.REASON_UNRESOLVED_SHAPE), res     # was no_number / partial_number


@pytest.mark.parametrize("page, value, snippet, number", [
    ("Km 1.2 x 10^3 M", "1.2 x 10^3", "Km 1.2 x 10^3", 1200.0),
    ("Km 1.2 × 10^3 M", "1.2 × 10^3", "Km 1.2 × 10^3", 1200.0),
    ("Km 1.2e3 M", "1.2e3", "Km 1.2e3 M", 1200.0),
])
def test_explicit_exponents_still_pass(page, value, snippet, number):
    res = verify(_prop(value, snippet), _doc(page))
    assert res.status == "pass" and res.number == pytest.approx(number)


def test_small_text_exponent_stays_unresolved_as_documented():
    res = verify(_prop("1.2", "Peak 1.2"), _doc("Peak 1.2 x 10 cells", small="3"))
    assert (res.status, res.reason) == ("fail", M.REASON_PARTIAL_NUMBER)


@pytest.mark.parametrize("page, value, snippet", [
    ("Patients took 2 x 100 mg tablets daily.", "100", "2 x 100 mg"),
    ("Patients took 3 x 1000 mg tablets daily.", "1000", "3 x 1000 mg"),
    ("Patients took 2 x 100 mg tablets daily.", "2", "took 2 x 100 mg"),
])
def test_dose_products_stay_plain_numbers(page, value, snippet):
    assert verify(_prop(value, snippet), _doc(page)).status == "pass"


def test_a_snippet_cut_at_a_dangling_multiplier_is_a_partial_number():
    """Was a row of test_dose_products_stay_plain_numbers: snippet 'took 2 x' (cut inside '2 x 100 mg') citing 2
    passed. Read alone, '2 x' ends in a multiplier without a number (unresolved_shape:multiplier_operand), so the
    cited 2 is only a part of it: stricter, and the snippet that holds the whole product still passes (above)."""
    res = verify(_prop("2", "took 2 x"), _doc("Patients took 2 x 100 mg tablets daily."))
    assert (res.status, res.reason) == ("fail", M.REASON_PARTIAL_NUMBER), res


def test_a_bare_ten_after_a_multiplier_is_never_a_plain_number():
    """Was a row of test_dose_products_stay_plain_numbers: ('2 x 10 mg', cite '10') passed. A ten after a multiplier
    with no readable power may be a raised exponent the text layer lost, so the whole expression is unresolved and
    the cited ten is a part of it."""
    res = verify(_prop("10", "2 x 10 mg"), _doc("Patients took 2 x 10 mg tablets daily."))
    assert (res.status, res.reason) == ("fail", M.REASON_PARTIAL_NUMBER), res


def test_parse_claim_of_raised_shape_is_none():
    assert parse_claim("1.2 × 103") is None


SIGNED_GAP_PAGES = ["Peak 1.2 x 10 -3 cells per mL", "Peak 1.2 x 10 −3 cells per mL", "Peak 1.2 x 10 +3 cells per mL",
                    "Peak 1.2 × 10 -3 cells per mL"]


@pytest.mark.parametrize("page", SIGNED_GAP_PAGES)
def test_signed_exponent_after_a_word_gap_is_unresolved_even_without_small_glyphs(page):
    toks = tokens(page, ".")
    assert toks[0].value is None and toks[0].parts[0] == 1.2, toks      # the mantissa leads the resolved parts
    res = verify(_prop("1.2", "Peak 1.2"), _doc(page, small=""))
    assert (res.status, res.reason) == ("fail", M.REASON_PARTIAL_NUMBER), res


def test_signed_exponent_whole_literal_never_passes():
    page = "Peak 1.2 x 10 -3 cells per mL"
    res = verify(_prop("1.2 x 10 -3", "Peak 1.2 x 10 -3"), _doc(page))
    assert res.status == "fail", res


def test_dose_product_of_a_hundred_still_passes_beside_signed_gap_rule():
    page = "Patients took 2 x 100 mg tablets daily."
    assert verify(_prop("100", "2 x 100 mg"), _doc(page)).status == "pass"


@pytest.mark.parametrize("value, snippet", [("10", "2 x 10 mg"), ("2", "took 2 x 10 mg")])
def test_bare_ten_dose_products_are_unresolved_beside_signed_gap_rule(value, snippet):
    """Was test_dose_products_still_pass_beside_signed_gap_rule rows ('2 x 10 mg', 10) and ('2 x 10 mg', 2): both
    passed. '2 x 10 mg' is one unresolved expression (ten_after_multiplier); each number in it is only a part."""
    res = verify(_prop(value, snippet), _doc("Patients took 2 x 10 mg tablets daily."))
    assert (res.status, res.reason) == ("fail", M.REASON_PARTIAL_NUMBER), res
