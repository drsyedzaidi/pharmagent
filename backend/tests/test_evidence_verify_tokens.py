"""Whole-token verification: the cited number must coincide with a complete number token on the page
(or in the cell), sign and all; value_num is the matched page token's value; an ASCII space never
joins two numbers on the page; a table record's page span is the cited row, never an earlier match."""
from __future__ import annotations

import pytest

from app.evidence import models as M
from app.evidence.extract_rules import value_number
from app.evidence.extract_tools import check_proposals
from app.evidence.normalize import number_spans
from app.evidence.verify import verify

THIN = "\u2009"
TEMPLATE = {"fields": [{"name": "CL", "type": "number", "unit": "L/h"},
                       {"name": "dose", "type": "number", "unit": "mg"},
                       {"name": "AUC", "type": "number", "unit": "ng*h/mL"},
                       {"name": "change", "type": "number"},
                       {"name": "n", "type": "integer"},
                       {"name": "CL_lower", "type": "number"}, {"name": "CL_upper", "type": "number"}]}


def _doc(text: str, style: str = ".", tables: list | None = None) -> dict:
    return {"sha256": "x" * 64, "decimal_style": style,
            "pages": [{"n": 1, "text": text, "small_text": "", "columns": None, "tables": tables or []}]}


def _prop(field: str, value: str, snippet: str, *, style: str = ".", kind: str = "text", cell: dict | None = None,
          unit: str | None = None, group: str = "") -> dict:
    loc = M.Locator(page=1, kind=kind, snippet=snippet, cell=cell, decimal_style=style).to_json()
    return {"source_id": "S1", "field": field, "group_label": group, "value": value,
            "value_num": value_number(value, style), "unit": unit, "locator": loc, "extractor": "llm"}


@pytest.mark.parametrize("page, value, snippet", [
    ("Clearance was 12.4 L/h in adults.", "2.4", "2.4 L/h"),
    ("Mean change was -20 mg from baseline.", "20", "20 mg from baseline"),
    ("Mean change was −20 mg from baseline.", "20", "20 mg from baseline"),
    ("AUC was 1,234 ng*h/mL.", "234", "234 ng*h/mL"),
    ("fraction absorbed 0.45 overall", "45", "45 overall"),
    ("Clearance was 12.45 L/h.", "12.4", "Clearance was 12.4"),
    ("A dose of 1250 mg was given.", "250", "250 mg was given"),
    ("CL/F was 2.31 L/h (n = 10)", "31", "31 L/h"),
])
def test_truncated_number_fails(page, value, snippet):
    res = verify(_prop("CL", value, snippet), _doc(page))
    assert res.status == "fail" and res.reason == M.REASON_PARTIAL_NUMBER, res


def test_truncated_table_cell_fails():
    cell = {"table": 0, "row": 1, "col": 1, "header": "CL (L/h)", "row_label": "Adults"}
    tables = [{"index": 0, "strategy": "lines", "rows": [["Group", "CL (L/h)"], ["Adults", "12.4"]]}]
    res = verify(_prop("CL", "2.4", "2.4", kind="table", cell=cell), _doc("Group CL (L/h)\nAdults 12.4", tables=tables))
    assert res.status == "fail" and res.reason in (M.REASON_PARTIAL_NUMBER, M.REASON_TABLE_CELL_MISMATCH)


def test_whole_tokens_still_pass_with_sign_and_later_occurrence():
    assert verify(_prop("change", "-20", "-20 mg"), _doc("change was −20 mg")).status == "pass"
    # first '31' occurrence is inside 2.31; the second is a whole token and is the one cited
    page = "CL/F was 2.31 L/h; in renal impairment 31 L/h"
    res = verify(_prop("CL", "31", "31 L/h"), _doc(page))
    assert res.status == "pass" and res.literal == "31" and res.number == 31.0
    start, end = res.span
    assert page[start:end] == "31 L/h" and start > page.index("renal")


@pytest.mark.parametrize("page, value, snippet", [
    ("CL 12\n345 subjects enrolled", "12345", "CL 12 345 subjects"),
    ("Adults 12 145 0.85", "12145", "Adults 12 145 0.85"),
])
def test_ascii_space_never_joins_page_numbers(page, value, snippet):
    assert verify(_prop("CL", value, snippet), _doc(page)).status == "fail"


def test_thin_space_thousands_still_pass():
    res = verify(_prop("AUC", "18900", "AUC 18 900 ng"), _doc("AUC 18\u2009900 ng h/mL"))
    assert res.status == "pass" and res.number == 18900.0


def test_strict_number_spans_split_on_ascii_space_only():
    assert [v for _, _, v in number_spans("12 145 and 1\u2009234", ".", strict=True)] == [12.0, 145.0, 1234.0]
    assert [v for _, _, v in number_spans("12 145", ".")] == [12145.0]


def test_value_num_comes_from_the_page_token_on_a_decimal_comma_page():
    out = check_proposals([_prop("AUC", "1,234", "AUC 1234", style=",")], _doc("AUC 1234 ng*h/mL", ","), TEMPLATE)[0]
    assert out["verify_status"] == "pass" and out["value_num"] == 1234.0 and out["value"] == "1234"
    out = check_proposals([_prop("AUC", "1.234", "AUC 1,234", style=",")], _doc("AUC 1,234 ng*h/mL", ","), TEMPLATE)[0]
    assert out["verify_status"] == "pass" and out["value_num"] == pytest.approx(1.234)


def test_ci_inversion_seen_through_the_page_reading():
    page = _doc("CI lower 1234 upper 100", ",")
    props = [_prop("CL_lower", "1,234", "lower 1234", style=","), _prop("CL_upper", "100", "upper 100", style=",")]
    out = check_proposals(props, page, TEMPLATE)
    assert all(any(f["code"] == M.FLAG_CI_INVERTED for f in o["flags"]) for o in out)


def test_table_span_is_the_cited_row_not_an_earlier_match():
    text = ("Twelve volunteers (n = 12) received 150 mg.\nDose n Cmax\n150 mg 12 3,412\n")
    tables = [{"index": 0, "strategy": "lines", "rows": [["Dose", "n", "Cmax"], ["150 mg", "12", "3,412"]]}]
    cell = {"table": 0, "row": 1, "col": 1, "header": "n", "row_label": "150 mg"}
    res = verify(_prop("n", "12", "12", kind="table", cell=cell), _doc(text, tables=tables), field_type="integer")
    assert res.status == "pass"
    start, end = res.span
    assert text[start:end] == "12" and start > text.index("Dose n Cmax")
    # a prose line can name the same row cells ('Cmax,ss ... ng/mL'): the line that starts with the row label wins
    prose = f"Results. Cmax,ss was 1{THIN}234 ng/mL.\nParameter Unit Value\nCmax,ss ng/mL 1{THIN}234\n"
    t2 = [{"index": 0, "strategy": "lines", "rows": [["Parameter", "Unit", "Value"], ["Cmax,ss", "ng/mL", f"1{THIN}234"]]}]
    c2 = {"table": 0, "row": 1, "col": 2, "header": "Value", "row_label": "Cmax,ss"}
    res = verify(_prop("AUC", "1234", "1 234", kind="table", cell=c2), _doc(prose, tables=t2))
    assert res.status == "pass" and res.span[0] > prose.index("Parameter Unit")
    cell0 = {"table": 0, "row": 1, "col": 0, "header": "Dose", "row_label": "150 mg"}
    res = verify(_prop("dose", "150", "150 mg", kind="table", cell=cell0), _doc(text, tables=tables))
    assert res.status == "pass" and (res.span is None or res.span[0] > text.index("Dose n Cmax"))
