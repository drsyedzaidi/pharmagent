"""Scientific notation is one number: 'Km 1.2e-3 M', '1.2E+03', '1.2 x 10^3' and '4.5 x 10' whose
superscript exponent went to small_text are single tokens. A proposal citing only the mantissa fails
partial_number (never a value off by 10^3..10^6); a full-literal proposal carries the exponent."""
from __future__ import annotations

import pytest

from app.evidence import models as M
from app.evidence.extract_rules import leading_number, value_number
from app.evidence.extract_tools import check_proposals
from app.evidence.normalize import parse_number
from app.evidence.verify import verify

TEMPLATE = {"fields": [{"name": "Km", "type": "number"}, {"name": "dose", "type": "number"}]}


def _doc(text: str, small: str = "", style: str = ".") -> dict:
    return {"sha256": "x" * 64, "decimal_style": style,
            "pages": [{"n": 1, "text": text, "small_text": small, "columns": None, "tables": []}]}


def _prop(value: str, snippet: str, field: str = "Km", style: str = ".") -> dict:
    loc = M.Locator(page=1, kind="text", snippet=snippet, decimal_style=style).to_json()
    return {"source_id": "S1", "field": field, "group_label": "", "value": value,
            "value_num": value_number(value, style), "unit": None, "locator": loc, "extractor": "llm"}


@pytest.mark.parametrize("tok, style, expected", [
    ("1.2e-3", ".", 0.0012), ("1.2E+03", ".", 1200.0), ("1.2 x 10^3", ".", 1200.0),
    ("1.2×10^-3", ".", 0.0012), ("2.5·10^4", ".", 25000.0), ("4.5 X 10^6", ".", 4.5e6),
    ("10^6", ".", 1e6), ("1,2e-3", ",", 0.0012), ("-3.1e2", ".", -310.0), ("1.2 × 10⁻³", ".", 0.0012),
])
def test_full_scientific_literal_parses_with_its_exponent(tok, style, expected):
    assert parse_number(tok, style) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("tok", ["4.5 x 10", "1.2e999"])
def test_scientific_literal_without_a_readable_exponent_has_no_value(tok):
    assert parse_number(tok, ".") is None


@pytest.mark.parametrize("page, value, snippet, small", [
    ("Km 1.2e-3 M", "1.2", "Km 1.2e-3 M", ""),
    ("Km 1.2E+03 M", "1.2", "Km 1.2E+03", ""),
    ("Km 1.2 x 10^3 M", "1.2", "Km 1.2 x 10^3", ""),
    ("count 4.5 x 10 cells/L", "4.5", "count 4.5 x 10 cells", "6"),
    ("Km 1.2e-3 M", "1.2", "Km 1.2", ""),
    ("Km 1.2 x 10^3 M", "1.2", "Km 1.2", ""),
    ("Km 1.2 × 10⁻³ M", "1.2", "Km 1.2", ""),
    ("count 4.5 x 10 cells/L", "4.5", "count 4.5", "6"),
])
def test_mantissa_only_proposal_fails_partial_number(page, value, snippet, small):
    res = verify(_prop(value, snippet), _doc(page, small))
    assert res.status == "fail" and res.reason == M.REASON_PARTIAL_NUMBER, res


@pytest.mark.parametrize("page, value, snippet", [
    ("Km 1.2 x 10^3 M", "10", "x 10^3 M"), ("Km 1.2 x 10^3 M", "3", "10^3 M"), ("Km 1.2e-3 M", "-3", "e-3 M"),
    ("count 4.5 x 10 cells/L", "4.5 x 10", "count 4.5 x 10 cells"), ("count 4.5 x 10 cells/L", "45", "4.5 x 10"),
])
def test_exponent_parts_never_pass(page, value, snippet):
    assert verify(_prop(value, snippet), _doc(page, "6")).status == "fail"


@pytest.mark.parametrize("page, value, snippet, literal, number", [
    ("Km 1.2e-3 M", "1.2e-3", "Km 1.2e-3 M", "1.2e-3", 0.0012),
    ("Km 1.2E+03 M", "1.2E+03", "Km 1.2E+03 M", "1.2E+03", 1200.0),
    ("Km 1.2 x 10^3 M", "1.2 x 10^3", "Km 1.2 x 10^3 M", "1.2 x 10^3", 1200.0),
    ("Km 1.2 x 10^3 M", "1200", "Km 1.2 x 10^3 M", "1.2 x 10^3", 1200.0),
])
def test_full_literal_proposal_passes_with_the_exponent(page, value, snippet, literal, number):
    res = verify(_prop(value, snippet), _doc(page))
    assert res.status == "pass" and res.literal == literal and res.number == pytest.approx(number, rel=1e-12)
    out = check_proposals([_prop(value, snippet)], _doc(page), TEMPLATE)[0]
    assert out["verify_status"] == "pass" and out["value_num"] == pytest.approx(number, rel=1e-12)


def test_bare_times_ten_is_plain_multiplication_only_without_small_digits():
    page = "Patients took 2 x 10 mg tablets daily."
    assert verify(_prop("10", "2 x 10 mg", "dose"), _doc(page)).status == "pass"
    res = verify(_prop("10", "2 x 10 mg", "dose"), _doc(page, small="3"))
    assert res.status == "fail"


def test_rule_reader_never_takes_a_mantissa_or_skips_to_a_later_number():
    assert leading_number("1.2 x 10^3 (0.4)", ".")[:2] == ("1.2 x 10^3", 1200.0)
    assert leading_number("4.5 x 10 (1.2)", ".") is None
