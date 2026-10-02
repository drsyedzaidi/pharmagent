"""Attack round 2 regressions on the parse: the drawn ink of a number differs from its text layer.

- /ActualText on a visible '3' plus a transparent '8' at its origin (reader 231, text 281);
- a '6' drawn with a 180-degree text matrix (reader 239, text 236);
- a decimal point drawn as a filled path between digits (reader 12.4, text 124);
- a white hairline between two digits splitting a ruled cell (reader 281, cell '2');
- an '8' clipped to / covered on its right half (reader 231, text 281);
- a Lancet mid-dot whose advance pdfminer and pdfium disagree on must not drop the rest of the line;
- a glyph-dense page is judged unverified within a bounded time instead of holding a parse slot.
None of these may leave a flag-free passing record of the text layer's number.
"""
from __future__ import annotations

import time

import pytest

from app.evidence import models as M
from app.evidence.extract_rules import rule_extract, value_number
from app.evidence.extract_tools import check_proposals
from app.evidence.parse import parse_pdf
from tests import evidence_trickpdf2 as T2

TEMPLATE = {"fields": [{"name": "cmax", "type": "number", "unit": "ng/mL", "aliases": ["Cmax"]},
                       {"name": "auc", "type": "number", "aliases": ["AUC"]}]}
NUM = {"fields": [{"name": "Cmax", "type": "number", "unit": "ng/mL"}]}


def _doc(tmp_path, data: bytes, name: str = "x.pdf") -> dict:
    path = tmp_path / name
    path.write_bytes(data)
    return parse_pdf(path)


def _clean_pass(rec: dict) -> bool:
    return rec["verify_status"] == "pass" and not any(f["severity"] == "error" for f in rec["flags"])


def _cmax(doc: dict) -> list[dict]:
    recs = check_proposals(rule_extract(doc, TEMPLATE, "src_x")["proposals"], doc, TEMPLATE)
    return [r for r in recs if r["field"] == "cmax" and r["group_label"] == "150 mg"]


def _llm(doc: dict, value: str, snippet: str, *, cell: dict | None = None) -> dict:
    loc = M.Locator(page=1, kind="table" if cell else "text", snippet=snippet, cell=cell, decimal_style=".").to_json()
    prop = {"source_id": "S1", "field": "Cmax", "group_label": "", "value": value,
            "value_num": value_number(value, "."), "unit": "ng/mL", "locator": loc, "extractor": "llm"}
    return check_proposals([prop], doc, NUM)[0]


@pytest.mark.parametrize("build, seen", [
    (T2.actualtext, "231"), (T2.rotated, "239"), (T2.vector_point, "12.4"),
    (T2.clipped_eight, "231"), (T2.covered_half, "231"), (T2.white_ruling, "281"),
])
def test_text_layer_number_the_reader_does_not_see_is_never_a_clean_pass(tmp_path, build, seen):
    doc = _doc(tmp_path, build())
    for rec in _cmax(doc):
        assert rec["value"] == seen or not _clean_pass(rec), (rec["value"], rec["flags"])
    cell = {"table": 0, "row": 1, "col": 1}
    for t in doc["pages"][0]["tables"]:
        wrong = t["rows"][1][1] if len(t["rows"]) > 1 and len(t["rows"][1]) > 1 else ""
        if wrong and wrong != seen:
            assert not _clean_pass(_llm(doc, wrong, wrong, cell=cell))


def test_white_ruling_does_not_split_the_cell(tmp_path):
    rows = _doc(tmp_path, T2.white_ruling())["pages"][0]["tables"][0]["rows"]
    assert rows[1] == ["150 mg", "281", "1890"], rows


def test_cell_fragment_of_a_whole_row_token_fails_partial_number():
    """The verifier itself: a cell '2' whose page-text row reads '281' is part of a longer number."""
    doc = {"sha256": "x" * 64, "decimal_style": ".",
           "pages": [{"n": 1, "text": "Group Cmax AUC\n150 mg 281 1890", "small_text": "", "columns": None,
                      "hidden_chars": 0, "ink_check": "ok", "glyph_check": {},
                      "tables": [{"index": 0, "strategy": "lines",
                                  "rows": [["Group", "Cmax (ng/mL)", "", "AUC"], ["150 mg", "2", "81", "1890"]]}]}]}
    rec = _llm(doc, "2", "2", cell={"table": 0, "row": 1, "col": 1})
    assert rec["verify_status"] == "fail" and rec["verify_reason"] == M.REASON_PARTIAL_NUMBER


def test_plain_cells_stay_clean(tmp_path):
    for text in ("281", "12.4", "3.56"):
        recs = _cmax(_doc(tmp_path, T2.plain_cell(text), f"p{len(text)}.pdf"))
        assert len(recs) == 1 and recs[0]["value"] == text and _clean_pass(recs[0]), recs


def test_middot_drift_keeps_the_rest_of_the_line(tmp_path):
    doc = _doc(tmp_path, T2.middot_line())
    page = doc["pages"][0]
    assert "ng/mL in all subjects." in page["text"], page["text"]
    assert page["hidden_chars"] == 0
    line = next(ln for ln in page["text"].split("\n") if "Mean Cmax" in ln)
    assert _llm(doc, "12", line)["verify_status"] == "fail"


def test_glyph_dense_page_is_unverified_within_a_bounded_time(tmp_path):
    t0 = time.monotonic()
    doc = _doc(tmp_path, T2.dense_page())
    elapsed = time.monotonic() - t0
    page = doc["pages"][0]
    assert page["ink_check"] == "unverified"
    assert elapsed < 20.0, elapsed
