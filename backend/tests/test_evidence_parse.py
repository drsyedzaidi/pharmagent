"""parse_pdf: body-size text filter, small_text, columns, lines/layout tables, metadata, decimal style."""
from __future__ import annotations

import json

import pytest

from app.evidence.parse import PARSER_NAME, infer_decimal_style, parse_pdf
from tests import evidence_fixtures as fx

needs_unicode = pytest.mark.skipif(not fx.UNICODE_FONT, reason="no Unicode TTF found; ASCII variant built")


@pytest.fixture(scope="module")
def papers(tmp_path_factory) -> dict:
    d = tmp_path_factory.mktemp("papers")
    paths = fx.build_all(d)
    paths["no_text"] = fx.no_text_pdf(d)
    return {name: parse_pdf(p) for name, p in paths.items()}


def _page_tables(doc: dict, n: int) -> list[dict]:
    return doc["pages"][n - 1]["tables"]


def test_parse_json_contract_top_level(papers):
    doc = papers["paper_a"]
    assert set(doc) == {"sha256", "parser", "decimal_style", "text_chars", "metadata", "pages"}
    assert doc["parser"] == PARSER_NAME and len(doc["sha256"]) == 64
    assert set(doc["metadata"]) == {"title", "author", "subject", "creator", "producer", "creation_date"}
    assert doc["metadata"]["title"].startswith("Pharmacokinetics of drug X") and doc["metadata"]["author"]
    page = doc["pages"][0]
    assert set(page) == {"n", "text", "small_text", "columns", "tables", "hidden_chars"} and page["n"] == 1
    assert page["hidden_chars"] == 0
    assert doc["text_chars"] == sum(len(p["text"]) for p in doc["pages"]) > 0
    json.dumps(doc)   # must be plain JSON


def test_paper_a_lines_table_with_cmax_header_and_literal_cell(papers):
    tables = _page_tables(papers["paper_a"], 1)
    assert len(tables) == 1
    t = tables[0]
    assert t["strategy"] == "lines" and t["index"] == 0 and t["bbox"] is not None
    assert any("Cmax" in h for h in t["rows"][0])
    assert "3,412 (812)" in t["rows"][1]
    assert t["caption"] is not None and t["caption"].startswith("Table 1")
    assert papers["paper_a"]["decimal_style"] == "."
    assert papers["paper_a"]["pages"][0]["columns"] is None


def test_paper_a_text_keeps_sentence_values(papers):
    text = papers["paper_a"]["pages"][0]["text"]
    assert "3,412 ng/mL" in text and "(n = 12)" in text and "150 mg" in text


@needs_unicode
def test_paper_b_keeps_unicode_minus_and_plus_minus_in_cells(papers):
    t = _page_tables(papers["paper_b"], 1)[0]
    cells = [c for row in t["rows"] for c in row]
    assert "1\u2009234 ± 312" in cells                  # the PDF's thin space survives extraction
    assert any("−" in c for c in cells)


def test_paper_b_small_glyphs_are_split_out_of_body_text(papers):
    page = papers["paper_b"]["pages"][0]
    marker = "†" if fx.UNICODE_FONT else "#"
    assert marker in page["small_text"]
    assert "ng/mL" + marker not in page["text"] and "ng/mL" in page["text"]
    assert "Values are arithmetic mean" in page["text"]      # 9pt footnote stays body text


def test_paper_c_columns_on_page_one(papers):
    page = papers["paper_c"]["pages"][0]
    assert page["columns"] is not None and len(page["columns"]) == 2
    left, right = (" ".join(c.split()) for c in page["columns"])
    assert "Compound Y was incubated with primary rat hepatocytes" in left
    assert "Intrinsic clearance was estimated from the substrate depletion slope" in right
    assert "Methods." not in left and "Introduction." not in right   # no interleaving
    assert papers["paper_c"]["pages"][1]["columns"] is None


def test_paper_c_layout_table_with_merged_continuation_row(papers):
    tables = _page_tables(papers["paper_c"], 2)
    assert len(tables) == 1
    t = tables[0]
    assert t["strategy"] == "layout" and t["bbox"] is None
    assert t["caption"].startswith("Table 1")
    assert t["rows"][0] == ["Parameter", "Rat", "Mouse"]
    unit = "µL" if fx.UNICODE_FONT else "uL"
    assert t["rows"][1][0].startswith("CLint") and unit in t["rows"][1][0]
    assert t["rows"][1][1] == ("18,2 ± 3,4" if fx.UNICODE_FONT else "18,2 +/- 3,4")
    assert t["rows"][2][0] == "Fraction unbound" and len(t["rows"]) == 3
    assert papers["paper_c"]["decimal_style"] == ","


def test_paper_d_trap_value_absent_and_true_value_present(papers):
    text = papers["paper_d"]["pages"][0]["text"]
    assert "2.31" in text and "2.41" not in text
    assert fx.INJECTION in text
    t = _page_tables(papers["paper_d"], 1)[0]
    assert ["CL/F (L/h)", "2.31", "8.4"] in t["rows"]


def test_no_text_pdf_parses_to_empty_pages(papers):
    doc = papers["no_text"]
    assert doc["text_chars"] == 0 and doc["pages"][0]["text"] == "" and doc["pages"][0]["tables"] == []


@pytest.mark.parametrize("text,expected", [
    ("Cmax was 3,412 ng/mL and t1/2 was 11.4 h", "."),
    ("CLint 18,2 ± 3,4 and fu 0,12", ","),
    ("values 1.234,5 and 2.345,6", ","),
    ("no numbers here", "."),
    ("counts 1,200 and 3,400 only", "."),
])
def test_infer_decimal_style(text, expected):
    assert infer_decimal_style(text) == expected


def test_parse_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse_pdf(tmp_path / "nope.pdf")
