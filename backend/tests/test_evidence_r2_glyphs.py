"""Round-2 attack regressions on the parse: the text layer must not name a number the reader does not see.

- a /ToUnicode CMap that renames a drawn digit (reader 231, text 241) and a Type3 font whose glyph
  named 'three' paints a 4 (reader 241, text 231) leave no flag-free passing record;
- a form-field Widget appearance laid over page text is part of the rendered page, so the covered
  value / screening keywords are dropped as hidden;
- an over-struck digit (a second '1' 1.2-3 pt right of the first) cannot give a flag-free '112';
- on a two-column page the keyless 'n =' rule never joins one column's 'n =' with the other's number;
- an 80 %-size raised exponent is not read into the body text ('1.2 x 103') nor lost ('1.2' passes).
"""
from __future__ import annotations

import pytest

from app.evidence import models as M
from app.evidence.config_tools import validate_criteria
from app.evidence.extract_rules import rule_extract, value_number
from app.evidence.extract_tools import check_proposals
from app.evidence.parse import parse_pdf
from app.evidence.screen_rules import keyword_screen
from app.evidence.verify import verify
from tests import evidence_trickpdf as T

TEMPLATE = {"fields": [{"name": "cmax", "type": "number", "unit": "ng/mL", "aliases": ["Cmax"]},
                       {"name": "auc", "type": "number", "aliases": ["AUC"]},
                       {"name": "n", "type": "integer", "aliases": ["N"]}]}


def _doc(tmp_path, data: bytes, name: str = "x.pdf") -> dict:
    path = tmp_path / name
    path.write_bytes(data)
    return parse_pdf(path)


def _keyless(doc: dict) -> list[dict]:
    return check_proposals(rule_extract(doc, TEMPLATE, "src_x")["proposals"], doc, TEMPLATE)


def _clean_pass(rec: dict) -> bool:
    """A record a reviewer would see as verified with nothing to worry about."""
    return rec["verify_status"] == "pass" and not any(f["severity"] == "error" for f in rec["flags"])


def _cmax(records: list[dict], group: str = "150 mg") -> dict:
    rows = [r for r in records if r["field"] == "cmax" and r["group_label"] == group]
    assert len(rows) == 1, rows
    return rows[0]


def _codes(rec: dict) -> set[str]:
    return {f["code"] for f in rec["flags"]}


# ------------------------------------------------------------ glyph identity ---

def test_tounicode_that_renames_a_drawn_digit_is_flagged(tmp_path):
    doc = _doc(tmp_path, T.tounicode_pdf(ruled=True))
    page = doc["pages"][0]
    assert page["ink_check"] == "ok"
    assert "4" in page["glyph_check"].get(M.FLAG_GLYPH_MISMATCH, "")
    rec = _cmax(_keyless(doc))
    assert rec["value"] == "241"                              # the text layer's reading ...
    assert not _clean_pass(rec)                               # ... never a clean verified value
    assert M.FLAG_GLYPH_MISMATCH in _codes(rec)
    auc = [r for r in _keyless(doc) if r["field"] == "auc" and r["group_label"] == "150 mg"][0]
    assert auc["value"] == "1890" and M.FLAG_GLYPH_MISMATCH not in _codes(auc)   # holds no '4'


def test_type3_digits_are_never_a_clean_pass(tmp_path):
    doc = _doc(tmp_path, T.type3_pdf())
    rec = _cmax(_keyless(doc))
    assert rec["value"] == "231"                              # the reader sees 241
    assert not _clean_pass(rec)
    assert M.FLAG_GLYPH_MISMATCH in _codes(rec)


def test_standard_font_raises_no_glyph_flag(tmp_path):
    doc = _doc(tmp_path, T.build(T.table(T.line("F1", 200, 660, "231"), ruled=True).encode()))
    assert doc["pages"][0]["glyph_check"] == {}
    rec = _cmax(_keyless(doc))
    assert rec["value"] == "231" and _clean_pass(rec)


def test_embedded_truetype_fixture_raises_no_glyph_flag(tmp_path):
    from tests import evidence_fixtures as fx
    if not fx.UNICODE_FONT:
        pytest.skip("no Unicode TTF on this machine")
    paths = fx.build_all(tmp_path)
    assert all(p["glyph_check"] == {} for p in parse_pdf(paths["paper_a"])["pages"])


# ---------------------------------------------------------------- widgets ---

def test_widget_appearance_covering_a_value_hides_it(tmp_path):
    page = _doc(tmp_path, T.widget_pdf())["pages"][0]
    assert page["hidden_chars"] >= 3
    cells = [c for t in page["tables"] for row in t["rows"] for c in row]
    assert "241" not in page["text"] and "241" not in cells


CRITERIA = [{"id": "C1", "text": "Human adult pharmacokinetic study", "kind": "inclusion", "required": True,
             "keywords": ["human", "adult", "patients", "volunteers"]},
            {"id": "C2", "text": "Reports Cmax, AUC or clearance", "kind": "inclusion", "required": True,
             "keywords": ["Cmax", "AUC", "clearance"]}]


def test_widget_covered_screening_keywords_do_not_include_a_paper(tmp_path):
    plain = keyword_screen(validate_criteria(CRITERIA), _doc(tmp_path, T.widget_screen_pdf(covered=False),
                                                             "plain.pdf")["pages"])
    covered_doc = _doc(tmp_path, T.widget_screen_pdf(), "covered.pdf")
    covered = keyword_screen(validate_criteria(CRITERIA), covered_doc["pages"])
    assert plain["verdict"] != "include"
    assert covered["verdict"] == plain["verdict"]
    assert covered_doc["pages"][0]["hidden_chars"] > 0
    assert "volunteers" not in covered_doc["pages"][0]["text"]


# ------------------------------------------------------------- over-strike ---

@pytest.mark.parametrize("offset", [1.2, 2.0, 3.0])
def test_overstruck_digit_never_gives_a_clean_pass(tmp_path, offset):
    rec = _cmax(_keyless(_doc(tmp_path, T.overstrike_pdf(offset))))
    if rec["value"] == "112":
        assert not _clean_pass(rec)
        assert M.FLAG_OVERLAPPING_GLYPHS in _codes(rec)


def test_overstrike_within_the_duplicate_window_reads_once(tmp_path):
    rec = _cmax(_keyless(_doc(tmp_path, T.overstrike_pdf(1.0))))
    assert rec["value"] == "12" and _clean_pass(rec)


# ----------------------------------------------------------------- columns ---

def test_two_column_n_rule_never_joins_columns(tmp_path):
    doc = _doc(tmp_path, T.columns_pdf())
    assert doc["pages"][0]["columns"] is not None
    ns = [r for r in _keyless(doc) if r["field"] == "n"]
    assert not any(r["value_num"] == 12 and _clean_pass(r) for r in ns), ns
    crossing = [r for r in ns if r["value_num"] == 12]
    assert all(r["verify_reason"] == M.REASON_CROSSES_COLUMNS for r in crossing), crossing


def test_snippet_found_only_across_the_column_gap_fails(tmp_path):
    doc = _doc(tmp_path, T.columns_pdf())
    loc = M.Locator(page=1, kind="text", snippet="The final cohort size was n = 12 h samples").to_json()
    res = verify({"value": "12", "locator": loc}, doc, field_type="integer")
    assert res.status == "fail" and res.reason == M.REASON_CROSSES_COLUMNS
    one_side = M.Locator(page=1, kind="text", snippet="12 h samples were split for a stability").to_json()
    assert verify({"value": "12", "locator": one_side}, doc, field_type="integer").status == "pass"


# ------------------------------------------------------------ raised exponent ---

@pytest.mark.parametrize("value", ["1.2", "103", "10"])
def test_raised_exponent_at_80_percent_never_passes_as_a_plain_number(tmp_path, value):
    doc = _doc(tmp_path, T.raised_exponent_pdf())
    page = doc["pages"][0]
    assert "103" not in page["text"]
    assert "3" in page["small_text"]
    loc = M.Locator(page=1, kind="text", snippet="Peak count was 1.2 x 10").to_json()
    prop = {"source_id": "s", "field": "peak", "group_label": "", "value": value,
            "value_num": value_number(value, "."), "unit": None, "locator": loc, "extractor": "llm"}
    assert verify(prop, doc).status == "fail"


def test_unruled_row_with_a_blank_cell_keeps_its_columns(tmp_path):
    """A blank (or hidden) cell in an unruled table must not shift the next value into its column."""
    doc = _doc(tmp_path, T.build(T.table("", ruled=False).encode()))
    rows = doc["pages"][0]["tables"][0]["rows"]
    assert rows[1] == ["150 mg", "", "1890"], rows
    cmax = [r for r in _keyless(doc) if r["field"] == "cmax" and r["group_label"] == "150 mg"]
    assert cmax == [] or cmax[0]["value"] != "1890"
