"""Record checks beyond the literal match: the proposed unit against the unit printed next to the
number (or in the table's header / unit column), a page-level hidden-text flag, conflicting values
in one proposal batch, normalised group keys, and the text-rule / table duplicate."""
from __future__ import annotations

from app.evidence import models as M
from app.evidence.extract_rules import rule_extract, value_number
from app.evidence.extract_tools import check_proposals
from app.evidence.qc_flags import scripted_flags

TEMPLATE = {"fields": [{"name": "dose", "type": "number", "unit": "mg"},
                       {"name": "fraction", "type": "number"},
                       {"name": "CL", "type": "number", "unit": "L/h"},
                       {"name": "cmax", "type": "number", "unit": "ng/mL", "required": True},
                       {"name": "n", "type": "integer"},
                       {"name": "CL_lower", "type": "number"}, {"name": "CL_upper", "type": "number"}]}


def _doc(text: str, tables: list | None = None, hidden: int = 0) -> dict:
    return {"sha256": "x" * 64, "decimal_style": ".",
            "pages": [{"n": 1, "text": text, "small_text": "", "columns": None, "tables": tables or [],
                       "hidden_chars": hidden}]}


def _prop(field, value, snippet, unit=None, group="", kind="text", cell=None, src="S1"):
    loc = M.Locator(page=1, kind=kind, snippet=snippet, cell=cell).to_json()
    return {"source_id": src, "field": field, "group_label": group, "value": value,
            "value_num": value_number(value, "."), "unit": unit, "locator": loc, "extractor": "llm"}


def _codes(p) -> set[str]:
    return {f["code"] for f in p["flags"]}


def _sev(p, code) -> str:
    return next(f["severity"] for f in p["flags"] if f["code"] == code)


def test_unit_printed_next_to_the_number_must_match():
    out = check_proposals([_prop("dose", "100", "dose of 100 mcg", unit="mg")],
                          _doc("A dose of 100 mcg was given."), TEMPLATE)[0]
    assert out["verify_status"] == "pass" and M.FLAG_UNIT_SOURCE_MISMATCH in _codes(out)
    assert _sev(out, M.FLAG_UNIT_SOURCE_MISMATCH) == "error"
    out = check_proposals([_prop("fraction", "45", "absorbed was 45%", unit="fraction")],
                          _doc("fraction absorbed was 45% overall"), TEMPLATE)[0]
    assert M.FLAG_UNIT_SOURCE_MISMATCH in _codes(out)


def test_matching_or_unprinted_units():
    ok = check_proposals([_prop("CL", "2.31", "was 2.31 L/h", unit="L/h")], _doc("CL/F was 2.31 L/h"), TEMPLATE)[0]
    assert not _codes(ok) & {M.FLAG_UNIT_SOURCE_MISMATCH, M.FLAG_UNIT_UNVERIFIED}
    bare = check_proposals([_prop("CL", "2.31", "clearance 2.31 in adults", unit="L/h")],
                           _doc("clearance 2.31 in adults"), TEMPLATE)[0]
    assert M.FLAG_UNIT_UNVERIFIED in _codes(bare) and _sev(bare, M.FLAG_UNIT_UNVERIFIED) == "warn"


def test_table_unit_comes_from_the_header_or_unit_column():
    tables = [{"index": 0, "strategy": "lines",
               "rows": [["Parameter", "Unit", "Value"], ["Cmax", "ug/mL", "3.4"]]}]
    cell = {"table": 0, "row": 1, "col": 2, "header": "Value", "row_label": "Cmax"}
    out = check_proposals([_prop("cmax", "3.4", "3.4", unit="ng/mL", kind="table", cell=cell)],
                          _doc("Parameter Unit Value\nCmax ug/mL 3.4", tables), TEMPLATE)[0]
    assert out["verify_status"] == "pass" and M.FLAG_UNIT_SOURCE_MISMATCH in _codes(out)
    tables = [{"index": 0, "strategy": "lines", "rows": [["Dose", "Cmax (ng/mL)"], ["150 mg", "3,412"]]}]
    cell = {"table": 0, "row": 1, "col": 1, "header": "Cmax (ng/mL)", "row_label": "150 mg"}
    out = check_proposals([_prop("cmax", "3412", "3,412", unit="ng/mL", kind="table", cell=cell, group="150 mg")],
                          _doc("Dose Cmax (ng/mL)\n150 mg 3,412", tables), TEMPLATE)[0]
    assert not _codes(out) & {M.FLAG_UNIT_SOURCE_MISMATCH, M.FLAG_UNIT_UNVERIFIED}


def test_hidden_text_on_the_cited_page_is_flagged():
    out = check_proposals([_prop("n", "12", "(n = 12)")], _doc("Volunteers (n = 12).", hidden=40), TEMPLATE)[0]
    assert M.FLAG_HIDDEN_TEXT in _codes(out)


def test_conflicting_values_in_one_batch_are_flagged():
    recs = [_prop("CL", "12.4", "12.4", group="Adults"), _prop("CL", "99", "99", group="Adults")]
    flags = scripted_flags([{**r, "verify_status": "pass"} for r in recs], TEMPLATE, [])
    assert all(any(f["code"] == M.FLAG_DUPLICATE for f in fl) for fl in flags)
    same = scripted_flags([{**r, "verify_status": "pass"} for r in [recs[0], recs[0]]], TEMPLATE, [])
    assert not any(f["code"] == M.FLAG_DUPLICATE for fl in same for f in fl)


def test_group_keys_are_normalised():
    approved = [{**_prop("CL", "12.4", "12.4", group="Adults"), "status": "approved", "verify_status": "pass"}]
    for g in ("adults", "Adults ", " ADULTS"):
        fl = scripted_flags([{**_prop("CL", "99", "99", group=g), "verify_status": "pass"}], TEMPLATE, approved)[0]
        assert any(f["code"] == M.FLAG_DUPLICATE for f in fl), g
    lo, hi = _prop("CL_lower", "20", "20", group="Adults"), _prop("CL_upper", "10", "10", group="Adults ")
    flags = scripted_flags([{**lo, "verify_status": "pass"}, {**hi, "verify_status": "pass"}], TEMPLATE, [])
    assert all(any(f["code"] == M.FLAG_CI_INVERTED for f in fl) for fl in flags)


def test_required_missing_is_per_source_for_ungrouped_records():
    recs = [{**_prop("n", "12", "n = 12"), "verify_status": "pass"},
            {**_prop("cmax", "3412", "3,412", group="150 mg"), "verify_status": "pass"}]
    flags = scripted_flags(recs, TEMPLATE, [])
    assert not any(f["code"] == M.FLAG_REQUIRED_MISSING for f in flags[0])


def test_text_rule_value_already_in_a_table_is_not_proposed_twice():
    tables = [{"index": 0, "strategy": "lines", "rows": [["Dose", "n", "Cmax (ng/mL)"], ["150 mg", "12", "3,412"]]}]
    doc = _doc("Twelve volunteers (n = 12) received a single 150 mg oral dose.\nDose n Cmax (ng/mL)\n150 mg 12 3,412",
               tables)
    tmpl = {"fields": [{"name": "n", "type": "integer"}, {"name": "dose", "type": "number", "unit": "mg"},
                       {"name": "cmax", "type": "number", "unit": "ng/mL"}]}
    props = rule_extract(doc, tmpl, "S1")["proposals"]
    keys = [(p["field"], p["value_num"]) for p in props]
    assert keys.count(("n", 12.0)) == 1 and keys.count(("dose", 150.0)) == 1


def test_text_rule_attaches_to_the_only_table_group():
    tables = [{"index": 0, "strategy": "lines", "rows": [["Dose", "Cmax (ng/mL)"], ["150 mg", "3,412"]]}]
    doc = _doc("Twelve volunteers (n = 12) received drug.\nDose Cmax (ng/mL)\n150 mg 3,412", tables)
    tmpl = {"fields": [{"name": "n", "type": "integer"}, {"name": "cmax", "type": "number", "unit": "ng/mL"}]}
    n = next(p for p in rule_extract(doc, tmpl, "S1")["proposals"] if p["field"] == "n")
    assert n["group_label"] == "150 mg"
