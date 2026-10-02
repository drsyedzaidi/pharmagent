"""The printed unit must EQUAL the proposed one (after normalisation and a small equivalence table),
never merely prefix it: printed 'mg' does not vouch for 'mg/kg' or 'mg/L', nor 'L' for 'L/h'. A
snippet cut short is read on in the page; a unit that may continue on the next token ('mg kg-1')
cannot be decided and is a unit_unverified warning, never a silent pass."""
from __future__ import annotations

import pytest

from app.evidence import models as M
from app.evidence.extract_rules import value_number
from app.evidence.extract_tools import check_proposals

TEMPLATE = {"fields": [{"name": "dose", "type": "number"}, {"name": "CL", "type": "number"},
                       {"name": "conc", "type": "number"}]}
UNIT_CODES = {M.FLAG_UNIT_SOURCE_MISMATCH, M.FLAG_UNIT_UNVERIFIED}


def _doc(text: str, tables: list | None = None) -> dict:
    return {"sha256": "x" * 64, "decimal_style": ".",
            "pages": [{"n": 1, "text": text, "small_text": "", "columns": None, "tables": tables or []}]}


def _check(page: str, value: str, snippet: str, unit: str, *, kind: str = "text", cell: dict | None = None,
           tables: list | None = None) -> dict:
    loc = M.Locator(page=1, kind=kind, snippet=snippet, cell=cell).to_json()
    p = {"source_id": "S1", "field": "dose", "group_label": "", "value": value, "value_num": value_number(value, "."),
         "unit": unit, "locator": loc, "extractor": "llm"}
    out = check_proposals([p], _doc(page, tables), TEMPLATE)[0]
    assert out["verify_status"] == "pass", out
    return {f["code"]: f["severity"] for f in out["flags"] if f["code"] in UNIT_CODES}


@pytest.mark.parametrize("page, snippet, unit", [
    ("A dose of 100 mg was given.", "dose of 100 mg was given", "mg/kg"),
    ("A dose of 100 mg in water.", "dose of 100 mg in water", "mg/L"),
    ("Volume 2.3 L in adults.", "Volume 2.3 L in adults", "L/h"),
    ("A dose of 100 mg/kg was given.", "dose of 100 mg/kg was given", "mg"),
])
def test_a_printed_unit_never_vouches_for_a_longer_or_shorter_one(page, snippet, unit):
    value = "2.3" if "2.3" in page else "100"
    assert _check(page, value, snippet, unit) == {M.FLAG_UNIT_SOURCE_MISMATCH: "error"}


@pytest.mark.parametrize("page, snippet, unit, expected", [
    ("CL 2.3 L/h in adults", "CL 2.3 L", "L/h", {}),                       # read on in the page: 'L/h'
    ("CL 2.3 L/h in adults", "CL 2.3 L", "L", {M.FLAG_UNIT_SOURCE_MISMATCH: "error"}),
    ("A dose of 100 mg given.", "dose of 100 mg", "mg/kg", {M.FLAG_UNIT_SOURCE_MISMATCH: "error"}),
    ("A dose of 100 mg given.", "dose of 100 mg", "mg", {}),
])
def test_a_snippet_ending_at_the_unit_is_read_on_in_the_page(page, snippet, unit, expected):
    assert _check(page, "2.3" if "2.3" in page else "100", snippet, unit) == expected


@pytest.mark.parametrize("unit, expected", [
    ("mg/kg", {M.FLAG_UNIT_UNVERIFIED: "warn"}),
    ("mg", {M.FLAG_UNIT_UNVERIFIED: "warn"}),
    ("ug/kg", {M.FLAG_UNIT_SOURCE_MISMATCH: "error"}),
])
def test_a_unit_that_may_continue_on_the_next_token_is_unverified(unit, expected):
    page = "Patients received 100 mg kg-1 daily."
    assert _check(page, "100", "received 100 mg kg-1 daily", unit) == expected
    assert _check(page, "100", "received 100 mg", unit) == expected


@pytest.mark.parametrize("printed, unit", [("mcg", "µg"), ("µg", "ug"), ("ug", "mcg"), ("mL", "ml"),
                                           ("hr", "h"), ("ng·h/mL", "ng*h/mL")])
def test_equivalent_spellings_match(printed, unit):
    assert _check(f"The value 100 {printed} was seen.", "100", f"value 100 {printed} was seen", unit) == {}


def test_table_cell_unit_is_read_to_the_end_of_the_cell():
    tables = [{"index": 0, "strategy": "lines", "rows": [["Group", "Dose"], ["Adults", "100 mg/kg"]]}]
    cell = {"table": 0, "row": 1, "col": 1, "header": "Dose", "row_label": "Adults"}
    page = "Group Dose\nAdults 100 mg/kg"
    assert _check(page, "100", "100 mg", "mg/kg", kind="table", cell=cell, tables=tables) == {}
    assert _check(page, "100", "100 mg", "mg", kind="table", cell=cell, tables=tables) == {
        M.FLAG_UNIT_SOURCE_MISMATCH: "error"}
