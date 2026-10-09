"""r7 round 4: the table findings of the round-4 audit (each PoC a regression).

* a row of an unruled (layout) table whose cells hold no number of their own and change the reading of the number above
  them (a power of ten printed under each value: '×10^3') continues the cells above it, so the record of the mantissa
  fails instead of passing 1000x low;
* a table locator is the table's POSITION in page['tables'] (the verifier, the layout sidecar and the page-image region
  read it that way), never pdfplumber's numbering, which skips an empty ruled grid;
* a model's table record is bound to the one cell whose whole text (or a run of its whole words) is the snippet,
  preferring the cell of the record's group; several candidates bind none.
"""
from __future__ import annotations

from pathlib import Path

from fpdf import FPDF
from fpdf.fonts import FontFace

from app.evidence.extract_llm import _to_proposal, find_cell
from app.evidence.extract_rules import field_index, rule_extract
from app.evidence.parse import parse_pdf
from app.evidence.parse_tables import continuation
from app.evidence.proposal_check import check_proposals

T = {"fields": [{"name": "Cmax", "type": "number"}]}


def _layout_scale_pdf(tmp_path: Path) -> Path:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=10)
    pdf.cell(0, 7, "Table 1. Pharmacokinetic parameters", new_x="LMARGIN", new_y="NEXT")
    for row in (("Parameter", "Adults", "Children"), ("Cmax (ng/mL)", "12.4", "15.1"), ("", "×10^3", "×10^3"),
                ("Tmax (h)", "2.0", "1.5")):
        y = pdf.get_y()
        for x, c in zip((10, 70, 120), row, strict=True):
            pdf.set_xy(x, y)
            pdf.cell(0, 5, c)
        pdf.set_xy(pdf.l_margin, y + 5)
    path = tmp_path / "scale.pdf"
    pdf.output(str(path))
    return path


def test_a_scale_row_under_the_values_of_an_unruled_table_continues_their_cells(tmp_path):
    doc = parse_pdf(_layout_scale_pdf(tmp_path))
    rows = doc["pages"][0]["tables"][0]["rows"]
    assert rows[1][1:] == ["12.4 ×10^3", "15.1 ×10^3"] and len(rows) == 3, rows
    props = rule_extract(doc, T, "S1")["proposals"]
    out = check_proposals(props, doc, T)
    got = [(o["group_label"], o["verify_status"], o["value_num"]) for o in out]
    assert got == [("Adults", "pass", 12400.0), ("Children", "pass", 15100.0)], got     # the value a reader reads
    mantissa = {**props[0], "value": "12.4", "value_num": 12.4, "locator": {**props[0]["locator"], "snippet": "12.4"}}
    assert check_proposals([mantissa], doc, T)[0]["verify_reason"] == "partial_number"   # never 12.4 alone


def test_the_continuation_production_merges_only_a_row_that_changes_the_number_above():
    assert continuation(["Cmax", "12.4", "15.1"], ["", "×10^3", "×10^3"])
    assert continuation(["Cmax", "12.4", "15.1"], ["", "x10^3", ""])
    assert not continuation(["Cmax", "12.4", "15.1"], ["", "mg", "mg"])             # a unit row reads the same
    assert not continuation(["Cmax", "12.4", "15.1"], ["Tmax", "2.0", "1.5"])       # a row of its own numbers
    assert not continuation(["Cmax", "12.4", "15.1"], ["", "a", "b"])               # footnote letters
    assert not continuation(["Parameter", "Adults", "Children"], ["", "×10^3", ""])  # no number above


def _empty_grid_pdf(tmp_path: Path) -> Path:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=10)
    pdf.rect(20, 20, 80, 30)
    pdf.line(60, 20, 60, 50)
    pdf.line(20, 35, 100, 35)                                  # an EMPTY ruled 2x2 grid (a figure frame, no text)
    pdf.set_xy(10, 60)
    for title, body in (("Table 1. Adults", (("Group", "Cmax"), ("Adults", "12.4"), ("Elderly", "15.1"))),
                        ("Table 2. Children", (("Group", "Cmax"), ("Children", "12.4"), ("Infants", "9.9")))):
        pdf.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
        with pdf.table(headings_style=FontFace(emphasis=None), text_align="LEFT") as tab:
            for row in body:
                r = tab.row()
                for c in row:
                    r.cell(c)
        pdf.ln(6)
    path = tmp_path / "grid.pdf"
    pdf.output(str(path))
    return path


def test_a_table_locator_is_the_position_in_the_page_tables(tmp_path):
    doc = parse_pdf(_empty_grid_pdf(tmp_path))
    tables = doc["pages"][0]["tables"]
    assert [t.get("index") for t in tables] == list(range(len(tables)))
    out = check_proposals(rule_extract(doc, T, "S1")["proposals"], doc, T)
    got = {(o["group_label"], o["value"]): (o["verify_status"], o["locator"]["cell"]["table"]) for o in out}
    assert got == {("Adults", "12.4"): ("pass", 0), ("Elderly", "15.1"): ("pass", 0),
                   ("Children", "12.4"): ("pass", 1), ("Infants", "9.9"): ("pass", 1)}, got


def test_a_cached_parse_with_skipped_table_numbers_is_located_by_position():
    rows_a = [["Group", "Cmax"], ["Adults", "12.4"]]
    rows_b = [["Group", "Cmax"], ["Children", "9.9"]]
    page = {"n": 1, "text": "Group Cmax\nAdults 12.4\nGroup Cmax\nChildren 9.9", "small_text": "", "columns": [],
            "tables": [{"index": 1, "strategy": "lines", "rows": rows_a},
                       {"index": 2, "strategy": "lines", "rows": rows_b}]}
    doc = {"sha256": "x" * 64, "decimal_style": ".", "pages": [page]}
    out = check_proposals(rule_extract(doc, T, "S1")["proposals"], doc, T)
    assert sorted((o["group_label"], o["verify_status"], o["locator"]["cell"]["table"]) for o in out) == [
        ("Adults", "pass", 0), ("Children", "pass", 1)]
    assert find_cell(page, "9.9")["table"] == 1


_ROWS = [["Group", "Cmax (ng/mL)", "Tmax (h)"], ["Adults", "12.4", "2.0"], ["Children", "12.4", "1.5"],
         ["Infants", "2.4", "1.0"]]
_PAGE = {"n": 1, "text": "\n".join(" ".join(r) for r in _ROWS), "small_text": "", "columns": [],
         "tables": [{"index": 0, "strategy": "lines", "rows": _ROWS}]}
_DOC = {"sha256": "x" * 64, "decimal_style": ".", "pages": [_PAGE]}


def _llm(group: str, value: str, snippet: str) -> dict:
    rec = {"field": "Cmax", "group": group, "value": value, "unit": "ng/mL", "page": 1, "kind": "table",
           "snippet": snippet}
    return _to_proposal(rec, field_index(T), _DOC, "S1")


def test_a_model_table_record_binds_the_cell_of_its_group_with_its_whole_text():
    p = _llm("Children", "12.4", "12.4")
    assert (p["locator"]["kind"], p["locator"]["cell"]["row"], p["locator"]["cell"]["row_label"]) == ("table", 2,
                                                                                                         "Children")
    q = _llm("Infants", "2.4", "2.4")
    assert (q["locator"]["cell"]["row"], q["locator"]["cell"]["row_label"]) == (3, "Infants")
    out = check_proposals([p, q], _DOC, T)
    assert [o["verify_status"] for o in out] == ["pass", "pass"], out


def test_a_model_table_record_that_names_no_cell_uniquely_binds_none():
    assert find_cell(_PAGE, "12.4") is None                        # two cells, no group: no guess
    assert find_cell(_PAGE, "2.4")["row"] == 3                     # '2.4' is no whole word of '12.4'
    assert find_cell(_PAGE, "12.4", "Elderly") is None             # the group names no row: still two cells
    p = _llm("", "12.4", "12.4")
    assert p["locator"]["kind"] == "text" and p["locator"]["cell"] is None
