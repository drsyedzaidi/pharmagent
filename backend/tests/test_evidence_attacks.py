"""Literal-match attack suite: every way a proposal can carry a number that is not literally on
the cited page must fail with a reason code from models.REASON_CODES, and a failed record can
never be approved. Proposals go through extract_tools.check_proposals (the path every LLM and
rule proposal takes before insertion) over the parsed fixture papers.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.config import settings
from app.core.pharmstate import PharmState
from app.evidence import extract_tools, ingest, llm_access, screen_tools
from app.evidence import models as M
from app.evidence import store as evstore
from app.evidence.config_tools import validate_criteria, validate_template
from app.evidence.parse import parse_pdf
from app.tools.base import ToolContext
from tests import evidence_fixtures as fx
from tests.test_evidence_extract import INCLUDE_ALL, TEMPLATE, SchemaLLM

needs_unicode = pytest.mark.skipif(not fx.UNICODE_FONT, reason="no Unicode TTF found; ASCII variant built")
CELL_A = {"table": 0, "row": 1, "col": 2, "header": "Cmax (ng/mL)", "row_label": "150 mg"}


@pytest.fixture(scope="module")
def docs(tmp_path_factory) -> dict:
    d = tmp_path_factory.mktemp("attack_papers")
    return {name: parse_pdf(p) for name, p in fx.build_all(d).items()}


def _prop(field: str, value: str, page: int, snippet: str, *, kind: str = "text", cell: dict | None = None,
          style: str = ".", unit: str | None = None) -> dict:
    loc = M.Locator(page=page, kind=kind, snippet=snippet, cell=cell, decimal_style=style).to_json()
    return {"source_id": "src_x", "field": field, "group_label": "", "value": value, "value_num": None,
            "unit": unit, "locator": loc, "extractor": "llm"}


def _check(doc: dict, *props: dict) -> list[dict]:
    out = extract_tools.check_proposals(list(props), doc, TEMPLATE)
    for p in out:
        assert p["verify_reason"] in M.REASON_CODES
        assert (p["verify_status"] == "pass") == (p["verify_reason"] == M.REASON_OK)
    return out


def _reason(doc: dict, prop: dict) -> str:
    return _check(doc, prop)[0]["verify_reason"]


# ---------------------------------------------------------------- digits ---

@pytest.mark.parametrize("snippet", ["Cmax was ٣٬٤١٢ ng/mL", "Cmax was ３４１２ ng/mL", "Cmax was 3,412 ng/mL٣"])
def test_non_ascii_digits_in_snippet_are_refused(docs, snippet):
    assert _reason(docs["paper_a"], _prop("cmax", "3412", 1, snippet)) == M.REASON_NON_ASCII_DIGITS


def test_non_ascii_digits_in_value_are_refused(docs):
    assert _reason(docs["paper_a"], _prop("cmax", "３４１２", 1, "3,412 (812)")) == M.REASON_NON_ASCII_DIGITS


def test_superscript_is_not_folded_into_a_number(docs):
    doc = {**docs["paper_c"], "pages": [{**docs["paper_c"]["pages"][1], "text": "cells per 10³ cells 18,2"}]}
    assert _reason(doc, _prop("fu", "103", 1, "per 10³ cells")) == M.REASON_NON_ASCII_DIGITS
    assert _reason(doc, _prop("fu", "10³", 1, "per 10³ cells")) == M.REASON_NON_ASCII_DIGITS
    # the real paper_c page keeps '10^6' as ASCII text and sup/sub glyphs out of page text
    assert "10^6" in docs["paper_c"]["pages"][1]["text"]


# ------------------------------------------------------------ separators ---

def test_comma_thousands_pass_and_misreadings_fail(docs):
    doc = docs["paper_a"]
    ok = _check(doc, _prop("cmax", "3412", 1, "3,412 (812)", kind="table", cell=CELL_A))[0]
    assert ok["verify_status"] == "pass" and ok["locator"]["literal"] == "3,412"
    assert _reason(doc, _prop("cmax", "3.412", 1, "3,412 (812)", kind="table", cell=CELL_A)) == M.REASON_NUMBER_MISMATCH
    assert _reason(doc, _prop("cmax", "34,12", 1, "3,412 (812)", kind="table", cell=CELL_A)) == M.REASON_NUMBER_MISMATCH
    assert _reason(doc, _prop("cmax", "3412", 1, "3,412 (812)")) == M.REASON_OK      # text kind, same cell text


def test_comma_decimal_page_reads_with_its_own_style(docs):
    doc = docs["paper_c"]
    assert _reason(doc, _prop("clint", "18.2", 2, "CLint 18,2 ± 3,4", style=",")) == M.REASON_OK
    assert _reason(doc, _prop("clint", "18,2", 2, "CLint 18,2 ± 3,4", style=",")) == M.REASON_OK
    assert _reason(doc, _prop("clint", "182", 2, "CLint 18,2 ± 3,4", style=",")) == M.REASON_NUMBER_MISMATCH
    assert _reason(doc, _prop("clint", "1820", 2, "CLint 18,2 ± 3,4", style=",")) == M.REASON_NUMBER_MISMATCH


def test_fragment_of_a_larger_number_is_no_number(docs):
    doc = docs["paper_a"]
    page = {**doc["pages"][0], "text": doc["pages"][0]["text"] + "\nAUC over 0–24 h was 245.6 ng h/mL"}
    doc2 = {**doc, "pages": [page]}
    assert _reason(doc2, _prop("auc", "24", 1, "was 245.6 ng h/mL")) == M.REASON_NO_NUMBER
    assert _reason(doc2, _prop("auc", "24", 1, "AUC over 0–24 h was 245.6")) == M.REASON_OK
    assert _reason(doc2, _prop("auc", "24", 1, "AUC was high")) == M.REASON_NO_NUMBER


# ------------------------------------------------------------- location ---

def test_value_only_on_another_page_is_refused(docs):
    doc = docs["paper_c"]
    assert _reason(doc, _prop("clint", "18,2", 1, "CLint 18,2 ± 3,4", style=",")) == M.REASON_SNIPPET_NOT_ON_PAGE
    assert _reason(doc, _prop("clint", "18,2", 3, "CLint 18,2 ± 3,4", style=",")) == M.REASON_PAGE_OUT_OF_RANGE
    assert _reason(doc, _prop("clint", "18,2", 0, "CLint 18,2 ± 3,4", style=",")) == M.REASON_PAGE_OUT_OF_RANGE


def test_fabricated_sentence_around_a_real_cell_value_is_refused(docs):
    doc = docs["paper_a"]
    assert _reason(doc, _prop("cmax", "3412", 1, "Cmax reached 3,412 ng/mL in every subject")) == \
        M.REASON_SNIPPET_NOT_ON_PAGE
    cited = _check(doc, _prop("cmax", "3412", 1, "3,412 (812)", kind="table", cell=CELL_A))[0]
    assert cited["verify_status"] == "pass" and cited["locator"]["cell"]["strategy"] == "lines"
    wrong_cell = {**CELL_A, "col": 3}
    assert _reason(doc, _prop("cmax", "3412", 1, "3,412 (812)", kind="table", cell=wrong_cell)) == \
        M.REASON_TABLE_CELL_MISMATCH


def test_snippet_edge_cases(docs):
    doc = docs["paper_a"]
    assert _reason(doc, _prop("cmax", "3412", 1, "   ")) == M.REASON_EMPTY_SNIPPET
    assert _reason(doc, _prop("cmax", "3412", 1, "3,412 " * 70)) == M.REASON_SNIPPET_TOO_LONG
    assert _reason(doc, _prop("n", "12.0", 1, "(n = 12)")) == M.REASON_OK
    assert _reason(doc, _prop("n", "12.5", 1, "(n = 12)")) == M.REASON_INTEGER_EXPECTED


def test_text_value_must_sit_inside_its_snippet(docs):
    doc = docs["paper_d"]
    assert _reason(doc, _prop("species", "adult patients", 1, "drug W in adult patients")) == M.REASON_OK
    assert _reason(doc, _prop("species", "rats", 1, "drug W in adult patients")) == M.REASON_TEXT_NOT_IN_SNIPPET
    assert _reason(doc, _prop("species", "rats", 1, "drug W in rats")) == M.REASON_SNIPPET_NOT_ON_PAGE


@needs_unicode
def test_unicode_minus_and_thin_space_map_onto_the_page(docs):
    doc = docs["paper_b"]
    out = _check(doc, _prop("cmax", "1234", 1, "1 234 ± 312"), _prop("cmax_se", "-20", 1, "−20 ± 8"))
    assert [p["verify_status"] for p in out] == ["pass", "pass"]
    # the page keeps the PDF's thin space (U+2009) inside the number; the snippet's plain space matches it
    assert out[0]["locator"]["literal"] == "1\u2009234" and out[1]["locator"]["literal"] == "−20"
    assert _reason(doc, _prop("cmax", "1,234", 1, "1 234 ± 312")) == M.REASON_OK
    assert _reason(doc, _prop("cmax", "1.234", 1, "1 234 ± 312")) == M.REASON_NUMBER_MISMATCH


# ---------------------------------------------------- the planted trap ---

@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    st = evstore.configure(":memory:")
    llm_access.set_llm(None)
    yield SimpleNamespace(store=st, ctx=ToolContext(data_dir=str(tmp_path)), dir=tmp_path)
    llm_access.set_llm(None)


def test_paper_d_planted_value_fails_and_cannot_be_approved(env, tmp_path):
    sid = "sess_trap"
    path = fx.build_pdf(tmp_path / "pdf", "paper_d")
    src = ingest.ingest_pdf(path.read_bytes(), "d.pdf", sid, "alice", data_dir=env.dir, store=env.store)
    env.store.save_config(sid, "criteria", validate_criteria(INCLUDE_ALL), "alice")
    screen_tools.screen_publications(PharmState(session_id=sid), env.ctx, {})
    for s in env.store.list_screens(sid):
        env.store.decide_screen(s.id, "approved", "alice", "ok")
    env.store.save_config(sid, "template", validate_template(TEMPLATE), "alice")
    llm_access.set_llm(SchemaLLM({"records": [
        {"field": "cl", "value": "2.41", "unit": "L/h", "page": 1, "kind": "text", "group": "",
         "snippet": "The mean apparent clearance (CL/F) was 2.31 L/h"},
        {"field": "cl", "value": "2.41", "unit": "L/h", "page": 1, "kind": "table", "group": "",
         "snippet": "2.31"},
        {"field": "cl", "value": "2.41", "unit": "L/h", "page": 1, "kind": "text", "group": "",
         "snippet": "CL/F was 2.41 L/h"}]}))
    extract_tools.extract_evidence(PharmState(session_id=sid), env.ctx, {})
    rows = env.store.list_records(sid, source_id=src.id, field="cl")
    assert len(rows) == 3 and all(r.verify_status == "fail" for r in rows)
    assert {r.verify_reason for r in rows} == {M.REASON_NUMBER_MISMATCH, M.REASON_SNIPPET_NOT_ON_PAGE}
    assert "2.41" not in ingest.load_pages(src)["pages"][0]["text"]
    for r in rows:
        with pytest.raises(ValueError, match="failed literal verification"):
            extract_tools.decide_records(PharmState(session_id=sid), env.ctx,
                                         {"record_id": r.id, "decision": "approve", "reason": "looks fine",
                                          "actor": "alice", "confirm": True})
        assert env.store.get_record(r.id).status == "proposed"
    assert env.store.approved_record_hashes(sid) == []
