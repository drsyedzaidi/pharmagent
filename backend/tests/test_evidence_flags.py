"""qc_flags: scripted QC flags per record (origin 'script') and add-only merge of LLM-proposed flags."""
from __future__ import annotations

import pytest

from app.evidence import models as M
from app.evidence.qc_flags import merge_llm_flags, normalise_unit, scripted_flags

TEMPLATE = {"fields": [
    {"name": "Cmax", "type": "number", "unit": "ng/mL", "required": True, "min": 0, "max": 100000, "aliases": []},
    {"name": "n", "type": "integer", "unit": None, "required": True, "aliases": ["N"]},
    {"name": "AUC", "type": "number", "unit": "µg h/mL", "required": False, "aliases": []},
    {"name": "Cmax_se", "type": "number", "unit": "ng/mL", "required": False, "aliases": []},
    {"name": "CL_ci_lower", "type": "number", "unit": "L/h", "required": False, "aliases": []},
    {"name": "CL_ci_upper", "type": "number", "unit": "L/h", "required": False, "aliases": []},
    {"name": "species", "type": "text", "required": False, "aliases": []},
]}


def _rec(field: str, value: str, *, source: str = "src_a", group: str = "", unit: str | None = None,
         num: float | None = "auto", verify: str = "pass", reason: str = "", strategy: str | None = None,
         kind: str = "text") -> dict:
    if num == "auto":
        try:
            num = float(value.replace(",", ""))
        except ValueError:
            num = None
    cell = None if strategy is None else {"table": 0, "row": 1, "col": 1, "header": field, "row_label": "",
                                          "strategy": strategy}
    loc = M.Locator(page=1, kind=kind, snippet=value, cell=cell).to_json()
    return {"source_id": source, "field": field, "group_label": group, "value": value, "value_num": num,
            "unit": unit, "locator": loc, "extractor": "table", "verify_status": verify, "verify_reason": reason,
            "flags": []}


def _codes(flags: list[dict]) -> set[str]:
    return {f["code"] for f in flags}


def _full_group(source: str = "src_a", group: str = "") -> list[dict]:
    return [_rec("Cmax", "3412", source=source, group=group, unit="ng/mL"),
            _rec("n", "12", source=source, group=group)]


def test_every_scripted_flag_has_shape_and_origin_script():
    recs = _full_group() + [_rec("Cmax", "-5", source="src_b", unit="ng/mL")]
    out = scripted_flags(recs, TEMPLATE, [])
    assert len(out) == len(recs)
    for flags in out:
        for f in flags:
            assert set(f) == {"code", "severity", "message", "field", "group", "origin"}
            assert f["origin"] == "script" and f["severity"] in ("warn", "error")
            assert f["code"] in M.FLAG_CODES and f["message"]


def test_clean_group_has_no_flags():
    assert scripted_flags(_full_group(), TEMPLATE, []) == [[], []]


def test_required_missing_per_source_and_group():
    recs = [_rec("Cmax", "3412", unit="ng/mL"),                                     # src_a '' lacks n
            _rec("n", "12", source="src_b", group="arm A"),                          # src_b 'arm A' lacks Cmax
            *_full_group(source="src_b", group="arm B")]                             # complete
    out = scripted_flags(recs, TEMPLATE, [])
    assert [f["code"] for f in out[0]] == [M.FLAG_REQUIRED_MISSING] and out[0][0]["field"] == "n"
    assert out[0][0]["group"] == "" and out[0][0]["severity"] == "warn"
    assert [f["code"] for f in out[1]] == [M.FLAG_REQUIRED_MISSING] and out[1][0]["field"] == "Cmax"
    assert out[1][0]["group"] == "arm A"
    assert out[2] == [] and out[3] == []


def test_required_missing_is_satisfied_by_an_approved_record():
    approved = [_rec("n", "12")]
    out = scripted_flags([_rec("Cmax", "3412", unit="ng/mL")], TEMPLATE, approved)
    assert out == [[]]


def test_out_of_range():
    out = scripted_flags(_full_group() + [_rec("Cmax", "250000", source="src_b", unit="ng/mL"),
                                          _rec("n", "12", source="src_b")], TEMPLATE, [])
    assert _codes(out[2]) == {M.FLAG_OUT_OF_RANGE} and out[2][0]["severity"] == "warn"
    assert "100000" in out[2][0]["message"]
    low = scripted_flags([_rec("Cmax", "-1", unit="ng/mL"), _rec("n", "12")], TEMPLATE, [])
    assert _codes(low[0]) == {M.FLAG_OUT_OF_RANGE}


def test_non_integer_n_is_an_error():
    out = scripted_flags([_rec("Cmax", "3412", unit="ng/mL"), _rec("n", "12.5")], TEMPLATE, [])
    assert [(f["code"], f["severity"]) for f in out[1]] == [(M.FLAG_NON_INTEGER_N, "error")]
    alias = scripted_flags([_rec("Cmax", "3412", unit="ng/mL"), _rec("N", "7.5")], TEMPLATE, [])
    assert M.FLAG_NON_INTEGER_N in _codes(alias[1])


def test_duplicate_against_approved_record_with_different_value():
    approved = [_rec("Cmax", "3412", unit="ng/mL"), _rec("n", "12")]
    out = scripted_flags([_rec("Cmax", "3500", unit="ng/mL")], TEMPLATE, approved)
    assert _codes(out[0]) == {M.FLAG_DUPLICATE} and out[0][0]["severity"] == "warn"
    assert "3412" in out[0][0]["message"]
    same = scripted_flags([_rec("Cmax", "3412", unit="ng/mL")], TEMPLATE, approved)
    assert same == [[]]
    other_group = scripted_flags([_rec("Cmax", "3500", group="arm B", unit="ng/mL"),
                                  _rec("n", "6", group="arm B")], TEMPLATE, approved)
    assert other_group == [[], []]


def test_unit_mismatch_and_micro_equivalence():
    recs = _full_group() + [_rec("AUC", "1.2", unit="ng h/mL")]
    out = scripted_flags(recs, TEMPLATE, [])
    assert _codes(out[2]) == {M.FLAG_UNIT_MISMATCH} and "µg h/mL" in out[2][0]["message"]
    for ok in ("ug h/mL", "µg h/mL", "μg h/mL", "UG H/ML", "ug·h/mL", "ug*h/mL"):
        assert scripted_flags(_full_group() + [_rec("AUC", "1.2", unit=ok)], TEMPLATE, [])[2] == [], ok
    assert normalise_unit("µg/mL") == normalise_unit("ug/mL") == normalise_unit("μg / mL")
    assert normalise_unit("ng/mL") != normalise_unit("µg/mL")


def test_unit_missing_when_template_has_unit():
    out = scripted_flags(_full_group() + [_rec("AUC", "1.2", unit=None)], TEMPLATE, [])
    assert _codes(out[2]) == {M.FLAG_UNIT_MISSING}
    assert scripted_flags(_full_group() + [_rec("species", "rat")], TEMPLATE, [])[2] == []


def test_se_nonpositive():
    out = scripted_flags(_full_group() + [_rec("Cmax_se", "0", unit="ng/mL")], TEMPLATE, [])
    assert _codes(out[2]) == {M.FLAG_SE_NONPOSITIVE}
    neg = scripted_flags(_full_group() + [_rec("Cmax_se", "-3", unit="ng/mL")], TEMPLATE, [])
    assert M.FLAG_SE_NONPOSITIVE in _codes(neg[2])
    assert scripted_flags(_full_group() + [_rec("Cmax_se", "3", unit="ng/mL")], TEMPLATE, [])[2] == []


def test_ci_inverted_flags_both_bounds_in_the_same_group():
    recs = _full_group() + [_rec("CL_ci_lower", "2.74", unit="L/h"), _rec("CL_ci_upper", "1.95", unit="L/h")]
    out = scripted_flags(recs, TEMPLATE, [])
    assert _codes(out[2]) == {M.FLAG_CI_INVERTED} and _codes(out[3]) == {M.FLAG_CI_INVERTED}
    assert out[2][0]["severity"] == "error"
    fine = _full_group() + [_rec("CL_ci_lower", "1.95", unit="L/h"), _rec("CL_ci_upper", "2.74", unit="L/h")]
    assert scripted_flags(fine, TEMPLATE, [])[2:] == [[], []]
    split = _full_group() + [_rec("CL_ci_lower", "2.74", group="x", unit="L/h"),
                             _rec("CL_ci_upper", "1.95", group="y", unit="L/h")]
    assert all(M.FLAG_CI_INVERTED not in _codes(f) for f in scripted_flags(split, TEMPLATE, []))


def test_composite_cell_when_value_holds_more_than_one_number():
    out = scripted_flags([_rec("Cmax", "3,412 (812)", unit="ng/mL", num=3412.0), _rec("n", "12")], TEMPLATE, [])
    assert _codes(out[0]) == {M.FLAG_COMPOSITE_CELL}
    text = scripted_flags(_full_group() + [_rec("species", "rat 12 and 14")], TEMPLATE, [])
    assert text[2] == []


def test_table_low_confidence_from_layout_table_cell():
    out = scripted_flags([_rec("Cmax", "3412", unit="ng/mL", strategy="layout", kind="table"), _rec("n", "12")],
                         TEMPLATE, [])
    assert _codes(out[0]) == {M.FLAG_TABLE_LOW_CONFIDENCE}
    ruled = scripted_flags([_rec("Cmax", "3412", unit="ng/mL", strategy="lines", kind="table"), _rec("n", "12")],
                           TEMPLATE, [])
    assert ruled[0] == []


def test_verify_fail_is_an_error_carrying_the_reason():
    out = scripted_flags([_rec("Cmax", "3412", unit="ng/mL", verify="fail", reason="number_mismatch"),
                          _rec("n", "12")], TEMPLATE, [])
    assert [(f["code"], f["severity"]) for f in out[0]] == [(M.FLAG_VERIFY_FAIL, "error")]
    assert "number_mismatch" in out[0][0]["message"]


def test_non_numeric_value_for_number_field():
    out = scripted_flags([_rec("Cmax", "n.d.", unit="ng/mL", num=None), _rec("n", "12")], TEMPLATE, [])
    assert _codes(out[0]) == {M.FLAG_NON_NUMERIC} and out[0][0]["severity"] == "error"


def test_unknown_field_gets_no_template_flags_but_keeps_verify_flag():
    out = scripted_flags(_full_group() + [_rec("mystery", "1", verify="fail", reason="no_number")], TEMPLATE, [])
    assert _codes(out[2]) == {M.FLAG_VERIFY_FAIL}


def test_scripted_flags_accept_record_dataclass_and_do_not_mutate_input():
    rec = M.Record(id="rec_x", session_id="s", source_id="src_a", template_hash="t", field="Cmax",
                   group_label="", value="3412", value_num=3412.0, unit="ng/mL",
                   locator=M.Locator(page=1, snippet="3412").to_json(), extractor="llm", verify_status="pass",
                   verify_reason="", flags=[], status="proposed", decided_by=None, decided_at=None,
                   decision_reason=None, record_hash="h", created_at="now")
    d = _rec("n", "12")
    before = {k: (list(v) if isinstance(v, list) else v) for k, v in d.items()}
    assert scripted_flags([rec, d], TEMPLATE, []) == [[], []]
    assert d == before and d["flags"] == []


# ------------------------------------------------------------- merge ---

SCRIPTED = [{"code": "out_of_range", "severity": "warn", "message": "above max", "field": "Cmax", "group": "",
             "origin": "script"}]


def test_merge_llm_flags_is_add_only_and_tags_origin_llm():
    proposed = [{"code": "implausible", "severity": "warn", "message": "looks like a typo", "field": "Cmax",
                 "group": "", "origin": "script"},                                   # origin claim ignored
                {"code": "out_of_range", "severity": "error", "message": "override", "field": "Cmax", "group": ""}]
    merged = merge_llm_flags(SCRIPTED, proposed)
    assert merged[0] == SCRIPTED[0] and SCRIPTED == [dict(SCRIPTED[0])]           # untouched
    assert len(merged) == 2 and merged[1]["origin"] == "llm" and merged[1]["code"] == "implausible"
    assert merged is not SCRIPTED


def test_merge_llm_flags_rejects_malformed_and_truncates():
    bad = [{"code": "", "severity": "warn", "message": "x"}, {"code": "a", "severity": "fatal", "message": "x"},
           "not a dict", {"code": "b", "severity": "warn", "message": ""}, None]
    assert merge_llm_flags(SCRIPTED, bad) == SCRIPTED
    long = [{"code": "x" * 100, "severity": "warn", "message": "m" * 1000, "field": "Cmax", "group": ""}]
    out = merge_llm_flags([], long)
    assert len(out[0]["code"]) <= 40 and len(out[0]["message"]) <= 300 and out[0]["origin"] == "llm"


def test_merge_llm_flags_cannot_remove_scripted_flags():
    assert merge_llm_flags(SCRIPTED, []) == SCRIPTED
    assert merge_llm_flags(SCRIPTED, None) == SCRIPTED


@pytest.mark.parametrize("code", ["required_missing", "out_of_range", "non_integer_n", "duplicate", "unit_mismatch",
                                  "unit_missing", "se_nonpositive", "ci_inverted", "composite_cell",
                                  "table_low_confidence", "verify_fail", "non_numeric"])
def test_flag_codes_are_declared(code):
    assert code in M.FLAG_CODES
