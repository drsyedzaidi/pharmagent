"""verify: literal-match verifier over fixture pages (pass paths set span+literal, fail paths carry reason codes)."""
from __future__ import annotations

import pytest

from app.evidence import models as M
from app.evidence.normalize import normalise
from app.evidence.parse import parse_pdf
from app.evidence.verify import MAX_SNIPPET_CHARS, find_number_in_snippet, find_snippet_in_page, verify
from tests import evidence_fixtures as fx

needs_unicode = pytest.mark.skipif(not fx.UNICODE_FONT, reason="no Unicode TTF found; ASCII variant built")


@pytest.fixture(scope="module")
def papers(tmp_path_factory) -> dict:
    d = tmp_path_factory.mktemp("verify_papers")
    return {name: parse_pdf(p) for name, p in fx.build_all(d).items()}


def _doc(*page_texts: str, tables: list | None = None, columns: list | None = None) -> dict:
    pages = [{"n": i + 1, "text": t, "small_text": "", "columns": None, "tables": []}
             for i, t in enumerate(page_texts)]
    if tables is not None:
        pages[0]["tables"] = tables
    if columns is not None:
        pages[0]["columns"] = columns
    return {"sha256": "x" * 64, "decimal_style": ".", "pages": pages}


def _rec(value: str, page: int, snippet: str, *, kind: str = "text", cell: dict | None = None,
         column: int | None = None, decimal_style: str = ".") -> dict:
    loc = M.Locator(page=page, kind=kind, snippet=snippet, cell=cell, column=column,
                    decimal_style=decimal_style).to_json()
    return {"value": value, "locator": loc}


def _page_text(doc: dict, page: int, column: int | None = None) -> str:
    pg = doc["pages"][page - 1]
    return pg["columns"][column] if column is not None else pg["text"]


def _assert_pass(res: M.VerifyResult, doc: dict, snippet: str, literal: str | None) -> None:
    assert res.status == "pass" and res.reason == M.REASON_OK, res
    loc = res.locator
    assert loc["span"] == res.span and loc["literal"] == res.literal == literal
    start, end = res.span
    raw = _page_text(doc, loc["page"], loc.get("column"))
    assert normalise(raw[start:end])[0] == normalise(snippet)[0]
    if literal is not None:
        assert normalise(literal)[0] in normalise(raw[start:end])[0]


# ------------------------------------------------------------------ pass ---

def test_pass_comma_thousands_in_ruled_cell(papers):
    doc = papers["paper_a"]
    cell = {"table": 0, "row": 1, "col": 2, "header": "Cmax (ng/mL)", "row_label": "150 mg"}
    res = verify(_rec("3412", 1, "3,412 (812)", kind="table", cell=cell), doc)
    _assert_pass(res, doc, "3,412 (812)", "3,412")
    assert res.locator["kind"] == "table" and res.locator["cell"]["col"] == 2


@needs_unicode
def test_pass_ascii_hyphen_snippet_against_unicode_minus_page(papers):
    doc = papers["paper_b"]
    res = verify(_rec("-20", 1, "change from baseline was -20 min"), doc)
    _assert_pass(res, doc, "change from baseline was -20 min", "−20")


def test_number_split_by_a_line_break_is_two_numbers():
    """A line break (or any ASCII space) on the page is a gap, not a thousands separator: '1\\n850'
    is the numbers 1 and 850, so '1850' is not on the page."""
    doc = _doc("Cmax was 1\n850 ng/mL in all subjects.")
    res = verify(_rec("1850", 1, "Cmax was 1 850 ng/mL"), doc)
    assert res.status == "fail" and res.reason == M.REASON_PARTIAL_NUMBER


def test_pass_hyphenated_line_break():
    doc = _doc("The apparent clear-\nance was 2.31 L/h.")
    res = verify(_rec("2.31", 1, "clearance was 2.31 L/h"), doc)
    _assert_pass(res, doc, "clearance was 2.31 L/h", "2.31")


def test_pass_plus_minus_cell_with_percent_unit():
    tables = [{"index": 0, "strategy": "lines", "bbox": None, "caption": None,
               "rows": [["Parameter", "CV (%)"], ["CL/F", "23.7 ± 4.2 %"]]}]
    doc = _doc("Parameter CV (%)\nCL/F 23.7 ± 4.2 %", tables=tables)
    cell = {"table": 0, "row": 1, "col": 1, "header": "CV (%)", "row_label": "CL/F"}
    res = verify(_rec("23.7", 1, "23.7 ± 4.2 %", kind="table", cell=cell), doc)
    _assert_pass(res, doc, "23.7 ± 4.2 %", "23.7")


@needs_unicode
def test_pass_comma_decimal_mode(papers):
    doc = papers["paper_c"]
    res = verify(_rec("18,2", 2, "rat hepatocytes was 18,2 µL/min", decimal_style=","), doc)
    _assert_pass(res, doc, "rat hepatocytes was 18,2 µL/min", "18,2")
    res2 = verify(_rec("18.2", 2, "rat hepatocytes was 18,2 µL/min", decimal_style=","), doc)
    assert res2.status == "pass" and res2.literal == "18,2"


def test_two_column_page_fails_at_page_level_then_passes_with_column_locator(papers):
    doc = papers["paper_c"]
    page = doc["pages"][0]
    assert page["columns"] is not None and len(page["columns"]) == 2
    snippet = "rats and incubated at 37 degrees"
    assert find_snippet_in_page(snippet, page["text"]) is None
    assert find_snippet_in_page(snippet, page["columns"][0]) is not None
    res = verify(_rec("37", 1, snippet), doc)            # automatic column fallback
    _assert_pass(res, doc, snippet, "37")
    assert res.locator["column"] == 0
    explicit = verify(_rec("37", 1, snippet, column=0), doc)
    _assert_pass(explicit, doc, snippet, "37")
    wrong_col = verify(_rec("37", 1, snippet, column=1), doc)
    assert wrong_col.status == "fail" and wrong_col.reason == M.REASON_SNIPPET_NOT_ON_PAGE


def test_pass_table_locator_on_the_right_cell(papers):
    doc = papers["paper_d"]
    cell = {"table": 0, "row": 1, "col": 1, "header": "Estimate", "row_label": "CL/F (L/h)"}
    res = verify(_rec("2.31", 1, "2.31", kind="table", cell=cell), doc)
    _assert_pass(res, doc, "2.31", "2.31")


def test_pass_integer_field_and_text_field(papers):
    doc = papers["paper_a"]
    res = verify(_rec("12", 1, "volunteers (n = 12) received"), doc, field_type="integer")
    _assert_pass(res, doc, "volunteers (n = 12) received", "12")
    txt = verify(_rec("healthy adults", 1, "oral dose in healthy adults"), doc, field_type="text")
    _assert_pass(txt, doc, "oral dose in healthy adults", None)


def test_pass_accepts_record_dataclass(papers):
    doc = papers["paper_a"]
    loc = M.Locator(page=1, snippet="t1/2) was 11.4 h").to_json()
    rec = M.Record(id="rec_x", session_id="s", source_id="src_x", template_hash="t", field="t_half",
                   group_label="", value="11.4", value_num=11.4, unit="h", locator=loc, extractor="llm",
                   verify_status="fail", verify_reason="", flags=[], status="proposed", decided_by=None,
                   decided_at=None, decision_reason=None, record_hash="h", created_at="now")
    res = verify(rec, doc)
    _assert_pass(res, doc, "t1/2) was 11.4 h", "11.4")


# ------------------------------------------------------------------ fail ---

def _assert_fail(res: M.VerifyResult, reason: str) -> None:
    assert res.status == "fail" and res.reason == reason, res
    assert reason in M.REASON_CODES and reason != M.REASON_OK
    assert res.span is None and res.literal is None


def test_fail_number_mismatch_trap_value(papers):
    _assert_fail(verify(_rec("2.41", 1, "clearance (CL/F) was 2.31 L/h"), papers["paper_d"]),
                 M.REASON_NUMBER_MISMATCH)


def test_fail_fabricated_snippet_containing_true_number(papers):
    _assert_fail(verify(_rec("2.31", 1, "the clearance of 2.31 L/h was confirmed by all reviewers"),
                        papers["paper_d"]), M.REASON_SNIPPET_NOT_ON_PAGE)


def test_fail_fragment_of_larger_number_is_no_number():
    doc = _doc("AUC was 245.6 ng h/mL")
    _assert_fail(verify(_rec("24", 1, "AUC was 245.6 ng h/mL"), doc), M.REASON_NO_NUMBER)
    _assert_fail(verify(_rec("24", 1, "AUC was high"), doc), M.REASON_NO_NUMBER)


def test_fail_decimal_comma_is_not_read_as_thousands():
    doc = _doc("A dose of 0,5 mg was given.")
    _assert_fail(verify(_rec("5", 1, "dose of 0,5 mg"), doc), M.REASON_NO_NUMBER)   # "5" is a fragment of "0,5"
    assert verify(_rec("0.5", 1, "dose of 0,5 mg"), doc).literal == "0,5"


def test_fail_value_only_on_another_page(papers):
    _assert_fail(verify(_rec("18,2", 1, "rat hepatocytes was 18,2 µL/min", decimal_style=","), papers["paper_c"]),
                 M.REASON_SNIPPET_NOT_ON_PAGE)


def test_fail_page_out_of_range(papers):
    _assert_fail(verify(_rec("2.31", 99, "was 2.31 L/h"), papers["paper_d"]), M.REASON_PAGE_OUT_OF_RANGE)
    _assert_fail(verify(_rec("2.31", 0, "was 2.31 L/h"), papers["paper_d"]), M.REASON_PAGE_OUT_OF_RANGE)


def test_fail_snippet_too_long(papers):
    long = "was 2.31 L/h " * 40
    assert len(long) > MAX_SNIPPET_CHARS == 300
    _assert_fail(verify(_rec("2.31", 1, long), papers["paper_d"]), M.REASON_SNIPPET_TOO_LONG)


def test_fail_non_ascii_digits_checked_before_matching(papers):
    _assert_fail(verify(_rec("2.31", 1, "was ٢.٣١ L/h"), papers["paper_d"]), M.REASON_NON_ASCII_DIGITS)
    _assert_fail(verify(_rec("２.３１", 1, "was 2.31 L/h"), papers["paper_d"]), M.REASON_NON_ASCII_DIGITS)


def test_fail_table_locator_pointing_at_different_cell(papers):
    doc = papers["paper_d"]
    wrong = {"table": 0, "row": 2, "col": 1, "header": "Estimate", "row_label": "V/F (L)"}
    _assert_fail(verify(_rec("2.31", 1, "2.31", kind="table", cell=wrong), doc), M.REASON_TABLE_CELL_MISMATCH)
    missing = {"table": 3, "row": 1, "col": 1, "header": "", "row_label": ""}
    _assert_fail(verify(_rec("2.31", 1, "2.31", kind="table", cell=missing), doc), M.REASON_TABLE_CELL_MISMATCH)
    _assert_fail(verify(_rec("2.31", 1, "2.31", kind="table", cell=None), doc), M.REASON_TABLE_CELL_MISMATCH)


def test_fail_integer_expected(papers):
    _assert_fail(verify(_rec("12.5", 1, "volunteers (n = 12) received"), papers["paper_a"], field_type="integer"),
                 M.REASON_INTEGER_EXPECTED)


def test_fail_empty_snippet(papers):
    _assert_fail(verify(_rec("2.31", 1, ""), papers["paper_d"]), M.REASON_EMPTY_SNIPPET)
    _assert_fail(verify(_rec("2.31", 1, "   \n"), papers["paper_d"]), M.REASON_EMPTY_SNIPPET)


def test_fail_order_empty_before_page_before_digits_before_number(papers):
    doc = papers["paper_d"]
    _assert_fail(verify(_rec("2.41", 99, ""), doc), M.REASON_EMPTY_SNIPPET)
    _assert_fail(verify(_rec("2.41", 99, "١"), doc), M.REASON_PAGE_OUT_OF_RANGE)
    _assert_fail(verify(_rec("2.41", 1, "١ 2.31"), doc), M.REASON_NON_ASCII_DIGITS)
    _assert_fail(verify(_rec("2.41", 1, "nowhere 2.31"), doc), M.REASON_NUMBER_MISMATCH)


def test_fail_no_text_page_and_missing_pages():
    _assert_fail(verify(_rec("2.31", 1, "was 2.31"), {"pages": []}), M.REASON_PAGE_OUT_OF_RANGE)
    _assert_fail(verify(_rec("2.31", 1, "was 2.31"), _doc("")), M.REASON_SNIPPET_NOT_ON_PAGE)


def test_fail_locator_rejects_unknown_kind(papers):
    _assert_fail(verify(_rec("2.31", 1, "was 2.31 L/h", kind="figure"), papers["paper_d"]),
                 M.REASON_SNIPPET_NOT_ON_PAGE)


# --------------------------------------------------------------- helpers ---

def test_find_snippet_in_page_returns_raw_offsets():
    page = "Cmax  was\n3,412 ng/mL"
    span = find_snippet_in_page("cmax was 3,412", page)
    assert span == [0, 15] and page[0:15] == "Cmax  was\n3,412"
    assert find_snippet_in_page("absent", page) is None
    assert find_snippet_in_page("", page) is None


def test_find_number_in_snippet_cases():
    assert find_number_in_snippet("3412", "Cmax 3,412 (812)", ".", False) == "3,412"
    assert find_number_in_snippet("812", "Cmax 3,412 (812)", ".", False) == "812"
    assert find_number_in_snippet("18.2", "18,2 ± 3,4", ",", False) == "18,2"
    assert find_number_in_snippet("-20", "was −20 min", ".", False) == "−20"
    assert find_number_in_snippet("24", "AUC 245.6", ".", False) is None
    assert find_number_in_snippet("12", "n = 12", ".", True) == "12"
    assert find_number_in_snippet("12.5", "n = 12.5", ".", True) is None
    assert find_number_in_snippet("abc", "n = 12", ".", False) is None
