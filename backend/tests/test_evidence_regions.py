"""regions.locate: a record's locator + the parse + the layout sidecar -> where to outline on the page.

Pure (no I/O). Every doubt degrades to precision 'page' with a machine reason: a wrong box is worse
than none."""
from __future__ import annotations

import copy
from pathlib import Path

import pytest

from app.evidence import regions as R
from app.evidence import verify
from app.evidence.parse import parse_pdf_with_layout
from tests import evidence_layout_fixtures as F


@pytest.fixture(scope="module")
def corpus(tmp_path_factory) -> dict[str, Path]:
    return F.corpus(tmp_path_factory.mktemp("regions"))


@pytest.fixture(scope="module")
def docs(corpus) -> dict[str, dict]:
    return {n: parse_pdf_with_layout(corpus[n]) for n in ("paper_a", "paper_c", "raw_plain")}


def _verified(doc: dict, *, value: str, snippet: str, page: int = 1, kind: str = "text", cell=None,
              field_type: str = "number") -> dict:
    loc = {"page": page, "kind": kind, "snippet": snippet, "cell": cell, "column": None}
    res = verify.verify({"value": value, "locator": loc}, {k: v for k, v in doc.items() if k != "layout"},
                        field_type=field_type)
    assert res.status == "pass", res.reason
    return res.locator


def _page(doc: dict, n: int = 1) -> tuple[dict, dict]:
    return doc["pages"][n - 1], doc["layout"]["pages"][n - 1]


def _words(lp: dict) -> list[list]:
    return lp["words"]


def test_text_locator_outlines_the_cited_word_inside_the_snippet(docs):
    doc = docs["raw_plain"]
    loc = _verified(doc, value="3,412", snippet="Cmax was 3,412 ng/mL")
    page, lp = _page(doc)
    reg = R.locate(loc, page, lp)
    assert (reg.precision, reg.resolved, reg.reason) == ("value", True, None)
    start = loc["token"][0]
    word = next(w for w in _words(lp) if w[0] <= start < w[1])
    assert reg.value_boxes == (tuple(word[2:]),)
    sl, sb, sr, st = reg.snippet_box
    vl, vb, vr, vt = reg.value_boxes[0]
    assert sl <= vl and sb <= vb and vr <= sr and vt <= st
    assert sr - sl > vr - vl                                  # the snippet is wider than the value


def test_column_locator_reads_the_column_words(docs):
    doc = docs["paper_c"]
    page, lp = _page(doc)
    loc = {"page": 1, "kind": "text", "snippet": "substrate depletion slope", "column": 1}
    span = [page["columns"][1].index("substrate"), page["columns"][1].index("slope") + 5]
    loc = {**loc, "span": span, "literal": None, "token": None}
    reg = R.locate(loc, page, lp)
    assert reg.precision == "line" and reg.resolved and reg.reason is None
    assert reg.snippet_box is not None and reg.value_boxes


def test_ruled_table_cell_is_outlined_with_header_and_row_label_context(docs):
    doc = docs["paper_a"]
    loc = _verified(doc, value="3,412", snippet="3,412 (812)", kind="table",
                    cell={"table": 0, "row": 1, "col": 2, "header": "Cmax (ng/mL)", "row_label": "150 mg"})
    page, lp = _page(doc)
    reg = R.locate(loc, page, lp)
    assert (reg.precision, reg.resolved) == ("cell", True)
    cells = lp["tables"][0]["cells"]
    assert reg.value_boxes == (tuple(cells["1:2"]),)
    assert set(reg.context_boxes) == {tuple(cells["0:2"]), tuple(cells["1:0"])}


def test_table_cell_without_a_box_degrades_to_its_row_line(docs):
    doc = copy.deepcopy(docs["paper_a"])
    loc = _verified(doc, value="3,412", snippet="3,412 (812)", kind="table",
                    cell={"table": 0, "row": 1, "col": 2, "header": "Cmax (ng/mL)", "row_label": "150 mg"})
    page, lp = _page(doc)
    del lp["tables"][0]["cells"]["1:2"]
    reg = R.locate(loc, page, lp)
    assert loc["span"] is not None
    assert (reg.precision, reg.resolved, reg.reason) == ("line", True, None)
    assert reg.value_boxes


def test_table_cell_with_neither_box_nor_row_is_the_whole_page(docs):
    doc = copy.deepcopy(docs["paper_c"])
    loc = _verified(doc, value="18,2", snippet="18,2 ± 3,4", page=2, kind="table",
                    cell={"table": 0, "row": 1, "col": 1, "header": "Rat", "row_label": "CLint (µL/min/10^6 cells)"})
    page, lp = _page(doc, 2)
    del lp["tables"][0]["cells"]["1:1"]
    loc = {**loc, "span": None}
    reg = R.locate(loc, page, lp)
    assert (reg.precision, reg.resolved, reg.reason) == ("page", False, "cell_unmapped")


@pytest.fixture
def case(docs):
    doc = copy.deepcopy(docs["raw_plain"])
    loc = _verified(doc, value="3,412", snippet="Cmax was 3,412 ng/mL")
    page, lp = _page(doc)
    return loc, page, lp


def _fails(reg, reason):
    assert (reg.precision, reg.resolved, reg.reason) == ("page", False, reason)
    assert not reg.value_boxes and reg.snippet_box is None and not reg.context_boxes


def test_missing_layout_is_layout_unavailable(case):
    loc, page, _ = case
    _fails(R.locate(loc, page, None), "layout_unavailable")


def test_layout_for_other_text_is_stale(case):
    loc, page, lp = case
    _fails(R.locate(loc, {**page, "text": page["text"] + " "}, lp), "layout_stale")


@pytest.mark.parametrize("status,reason", [("no_boxes", "no_boxes"), ("skipped", "no_boxes"),
                                           ("unmapped", "word_unmapped")])
def test_layout_status_without_boxes(case, status, reason):
    loc, page, lp = case
    _fails(R.locate(loc, page, {**lp, "status": status, "words": []}), reason)


def test_failed_verification_has_no_span(case):
    loc, page, lp = case
    _fails(R.locate({**loc, "span": None, "token": None, "literal": None}, page, lp), "no_span")


@pytest.mark.parametrize("bad", [[-1, 5], [5, 3], [0, 10**6], ["a", 2], [True, 3], [1], "xx", None, {}])
def test_garbage_span_never_resolves(case, bad):
    loc, page, lp = case
    reg = R.locate({**loc, "span": bad}, page, lp)
    assert reg.precision == "page" and not reg.resolved


def test_span_that_no_longer_holds_the_snippet_is_stale(case):
    loc, page, lp = case
    _fails(R.locate({**loc, "span": [0, 5]}, page, lp), "span_stale")


def test_token_that_is_not_the_literal_is_stale(case):
    loc, page, lp = case
    _fails(R.locate({**loc, "literal": "3,413"}, page, lp), "span_stale")
    _fails(R.locate({**loc, "token": [0, 5]}, page, lp), "span_stale")


def test_value_word_without_a_box_is_unmapped(case):
    loc, page, lp = case
    words = [w[:2] if w[0] <= loc["token"][0] < w[1] else w for w in lp["words"]]
    _fails(R.locate(loc, page, {**lp, "words": words}), "word_unmapped")


def test_unknown_kind_and_out_of_range_column_do_not_resolve(case):
    loc, page, lp = case
    assert not R.locate({**loc, "kind": "figure"}, page, lp).resolved
    assert not R.locate({**loc, "column": 5}, page, lp).resolved
    assert not R.locate({**loc, "column": True}, page, lp).resolved
    assert not R.locate({**loc, "cell": "x", "kind": "table"}, page, lp).resolved


def test_value_box_lies_inside_snippet_box_for_every_verified_number(docs):
    for name in ("paper_a", "raw_plain"):
        doc = docs[name]
        page, lp = _page(doc)
        for w in _words(lp):
            token = page["text"][w[0]:w[1]]
            if not token.replace(",", "").replace(".", "").isdigit():
                continue
            line_start = page["text"].rfind("\n", 0, w[0]) + 1
            line_end = page["text"].find("\n", w[1])
            snippet = page["text"][line_start:line_end if line_end >= 0 else None].strip()
            loc = {"page": 1, "kind": "text", "snippet": snippet, "column": None,
                   "span": [page["text"].index(snippet), page["text"].index(snippet) + len(snippet)],
                   "literal": token, "token": [w[0], w[1]]}
            reg = R.locate(loc, page, lp)
            if not reg.resolved:
                continue
            vl, vb, vr, vt = reg.value_boxes[0]
            sl, sb, sr, st = reg.snippet_box
            assert sl <= vl and sb <= vb and vr <= sr and vt <= st
