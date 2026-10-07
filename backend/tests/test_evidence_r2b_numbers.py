"""Attack round 2 regressions on number reading (verifier level, no model needed):

- a unit continued by a non-ASCII slash or product dot ('5 mg⁄kg', '5 mg⋅kg−1', '5 mg · kg−1',
  '5 mg (per kg)') is never read as a finished 'mg';
- a leading-zero group ('0,412' on a point page, '0.412' on a comma page) is never 412;
- spaced E-notation ('3.4 E-03') is one token: never its mantissa 3.4;
- one fragment of a grouped cell ('12' of '12 145', '456' of '1,23,456') carries an error flag;
- a multiplier in the second header row, or an upper-case 'E+03' header suffix, is scaled_header;
- a range bound ('15' of '12–15 ng/mL') and a decimal separator against the page style ('1,2' on a
  point page) carry an error flag.
"""
from __future__ import annotations

import pytest

from app.evidence import models as M
from app.evidence.extract_rules import rule_extract, value_number
from app.evidence.extract_tools import check_proposals

STRONG = "Clearance 12.4 L/h, volume 45.6 L, half-life 3.5 h, ka 1.25 /h. "
STRONG_COMMA = "Clearance 12,4 L/h, volume 45,6 L, half-life 3,5 h, ka 1,25 /h. "
NUM = {"fields": [{"name": "Cmax", "type": "number", "unit": "ng/mL"}]}
AUC = {"fields": [{"name": "AUC", "type": "number", "unit": "ng·h/mL"}]}
DOSE = {"fields": [{"name": "dose", "type": "number", "unit": "mg"}]}
CELL = {"table": 0, "row": 1, "col": 1}


def _doc(text: str, *, tables: list | None = None, style: str = ".") -> dict:
    return {"sha256": "x" * 64, "decimal_style": style,
            "pages": [{"n": 1, "text": text, "small_text": "", "columns": None, "hidden_chars": 0, "ink_check": "ok",
                       "tables": [{"index": i, "strategy": "lines", "rows": r} for i, r in enumerate(tables or [])]}]}


def _prop(field: str, value: str, snippet: str, *, unit: str | None = None, cell: dict | None = None,
          style: str = ".") -> dict:
    loc = M.Locator(page=1, kind="table" if cell else "text", snippet=snippet, cell=cell,
                    decimal_style=style).to_json()
    return {"source_id": "S1", "field": field, "group_label": "", "value": value,
            "value_num": value_number(value, style), "unit": unit, "locator": loc, "extractor": "llm"}


def _one(doc: dict, template: dict, prop: dict) -> dict:
    return check_proposals([prop], doc, template)[0]


def _keyless(doc: dict, template: dict) -> list[dict]:
    return check_proposals(rule_extract(doc, template, "S1")["proposals"], doc, template)


def _codes(rec: dict) -> set[str]:
    return {f["code"] for f in rec["flags"]}


def _clean(rec: dict) -> bool:
    return rec["verify_status"] == "pass" and not any(f["severity"] == "error" for f in rec["flags"])


# ------------------------------------------------------------ unit continuation ---

@pytest.mark.parametrize("unit_text", ["mg · kg−1", "mg (per kg)"])
def test_unit_continued_by_a_dot_or_per_is_not_mg(unit_text):
    doc = _doc(STRONG + f"Subjects were dosed at 5 {unit_text} intravenously.")
    rec = _one(doc, DOSE, _prop("dose", "5", f"dosed at 5 {unit_text}", unit="mg"))
    assert rec["verify_status"] == "pass"
    assert _codes(rec) & {M.FLAG_UNIT_SOURCE_MISMATCH, M.FLAG_UNIT_UNVERIFIED}, rec["flags"]
    for kr in _keyless(doc, DOSE):
        assert kr["unit"] != "mg" or _codes(kr) & {M.FLAG_UNIT_SOURCE_MISMATCH, M.FLAG_UNIT_UNVERIFIED}


@pytest.mark.parametrize("unit_text", ["mg⁄kg", "mg∕kg", "mg／kg", "mg⋅kg−1", "mg∙kg−1", "mg•kg−1"])
def test_unit_continued_by_a_non_ascii_slash_or_dot_fails_verification(unit_text):
    """Was test_unit_continued_by_a_non_ascii_slash_or_dot_is_not_mg (pass + a unit flag, the slash and dot look-alikes
    were folded to '/' and a dot). They are not in the allowed alphabet (the bullet is a LINE character, which may not
    touch the number's neighbour word), so the number next to the unit is unresolved_context: verification fails."""
    doc = _doc(STRONG + f"Subjects were dosed at 5 {unit_text} intravenously.")
    rec = _one(doc, DOSE, _prop("dose", "5", f"dosed at 5 {unit_text}", unit="mg"))
    assert rec["verify_status"] == "fail" and rec["verify_reason"] == M.REASON_UNRESOLVED_CONTEXT, rec
    for kr in _keyless(doc, DOSE):
        assert kr["unit"] != "mg" or kr["verify_status"] == "fail" or _codes(kr) & {M.FLAG_UNIT_SOURCE_MISMATCH,
                                                                                    M.FLAG_UNIT_UNVERIFIED}


def test_plain_mg_dose_stays_decided():
    rec = _one(_doc(STRONG + "Subjects were dosed at 5 mg intravenously."), DOSE,
               _prop("dose", "5", "dosed at 5 mg intravenously", unit="mg"))
    assert _clean(rec) and not _codes(rec) & {M.FLAG_UNIT_SOURCE_MISMATCH, M.FLAG_UNIT_UNVERIFIED}


# ------------------------------------------------------------------ leading zero ---

def test_leading_zero_comma_group_on_a_point_page_is_never_thousands():
    doc = _doc(STRONG + "Mean Cmax was 0,412 ng/mL in the fed arm.")
    rec = _one(doc, NUM, _prop("Cmax", "412", "Mean Cmax was 0,412 ng/mL", unit="ng/mL"))
    assert rec["value_num"] != 412 or rec["verify_status"] == "fail"
    keyless = _keyless(_doc(STRONG + "Table 1", tables=[[["Dose", "Cmax (ng/mL)"], ["150 mg", "0,412"]]]), NUM)
    assert keyless and all(r["value_num"] != 412 for r in keyless)
    assert all(not _clean(r) for r in keyless)                     # a decimal comma on a point page


def test_leading_zero_point_group_on_a_comma_page_is_never_thousands():
    doc = _doc(STRONG_COMMA + "Mean Cmax was 0.412 ng/mL.", style=",")
    rec = _one(doc, NUM, _prop("Cmax", "412", "Mean Cmax was 0.412 ng/mL", unit="ng/mL", style=","))
    assert rec["value_num"] != 412 or rec["verify_status"] == "fail"


# ------------------------------------------------------------- spaced E-notation ---

@pytest.mark.parametrize("cell, value", [("3.4 E-03", 0.0034), ("3.4 E3", 3400.0), ("3.4 E+03", 3400.0)])
def test_spaced_e_notation_is_one_token(cell, value):
    doc = _doc(STRONG + "Table 1", tables=[[["Dose", "Cmax (ng/mL)"], ["150 mg", cell]]])
    for rec in _keyless(doc, NUM):
        assert rec["value_num"] == pytest.approx(value) or rec["verify_status"] == "fail", rec
    mantissa = _one(doc, NUM, _prop("Cmax", "3.4", "3.4", unit="ng/mL", cell=CELL))
    assert mantissa["verify_status"] == "fail"


# -------------------------------------------------------------- grouped fragments ---

@pytest.mark.parametrize("cell, value", [("12 145", "12"), ("12 145", "145"), ("1,23,456", "1.23"),
                                         ("1,23,456", "456")])
def test_a_fragment_of_a_grouped_cell_carries_an_error(cell, value):
    doc = _doc(STRONG + "Table 1", tables=[[["Dose", "Cmax (ng/mL)"], ["150 mg", cell]]])
    snippet = "1,23" if value == "1.23" else value
    rec = _one(doc, NUM, _prop("Cmax", value, snippet, unit="ng/mL", cell=CELL))
    assert not _clean(rec), rec["flags"]


def test_two_numbers_in_a_cell_still_pass_with_a_warning():
    doc = _doc(STRONG + "Table 1", tables=[[["Dose", "Cmax (ng/mL)"], ["150 mg", "3.4 (1.2)"]]])
    rec = _one(doc, NUM, _prop("Cmax", "3.4", "3.4", unit="ng/mL", cell=CELL))
    assert _clean(rec) and M.FLAG_COMPOSITE_CELL in _codes(rec)


# ------------------------------------------------------------------ scaled header ---

def test_multiplier_in_a_second_header_row_is_scaled_header():
    rows = [["", "Day 1", "Day 1"], ["Group", "AUC ×10³ (ng·h/mL)", "Cmax (ng/mL)"], ["150 mg", "3.4", "120"]]
    rec = _one(_doc(STRONG + "Table 1", tables=[rows]), AUC,
               _prop("AUC", "3.4", "3.4", unit="ng·h/mL", cell={"table": 0, "row": 2, "col": 1}))
    assert M.FLAG_SCALED_HEADER in _codes(rec)


def test_upper_case_e_header_suffix_is_scaled_header():
    rows = [["Dose", "AUC (ng·h/mL) E+03"], ["150 mg", "3.4"]]
    rec = _one(_doc(STRONG + "Table 1", tables=[rows]), AUC, _prop("AUC", "3.4", "3.4", unit="ng·h/mL", cell=CELL))
    assert M.FLAG_SCALED_HEADER in _codes(rec)


# ------------------------------------------------------------- ranges and lists ---

@pytest.mark.parametrize("value", ["12", "15"])
def test_a_range_bound_is_not_an_exact_value(value):
    doc = _doc(STRONG + "Cmax ranged 12–15 ng/mL across subjects.")
    rec = _one(doc, NUM, _prop("Cmax", value, "Cmax ranged 12–15 ng/mL", unit="ng/mL"))
    assert not _clean(rec), rec["flags"]


def test_a_named_interval_bound_may_cite_a_range():
    tmpl = {"fields": [{"name": "Cmax_lower", "type": "number"}, {"name": "Cmax_upper", "type": "number"}]}
    doc = _doc(STRONG + "Cmax was 13.1 (12.0–15.2) ng/mL.")
    for field, value in (("Cmax_lower", "12.0"), ("Cmax_upper", "15.2")):
        rec = _one(doc, tmpl, _prop(field, value, "13.1 (12.0–15.2)"))
        assert _clean(rec), rec["flags"]


def test_decimal_separator_against_the_page_style_is_flagged():
    doc = _doc(STRONG + "Patients received oral doses of 1,2 or 4 mg once daily.")
    rec = _one(doc, DOSE, _prop("dose", "1.2", "oral doses of 1,2 or 4 mg", unit="mg"))
    assert M.FLAG_DECIMAL_STYLE_AMBIGUOUS in _codes(rec) and not _clean(rec)
