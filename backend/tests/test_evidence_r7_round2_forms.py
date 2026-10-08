"""r7 round 2: productions instead of shape detectors (findings of the round-2 review).

* the multi-word forms keep only their ACCEPTING productions (``m x 10^p``, ``n / d``); a mixed number, a detached
  exponent or decimal, a slash or colon between numbers, a ten behind a multiplier and a multiplier without a number are
  refused by the neighbour allow-list (a window with the allow-list switched off reads them);
* a bracket group is read by ONE production (productions.py): an unmistakable citation, or a ``bracket_list`` (``[1,2]``
  after a number); there is no second detector of the shape ``[n,m]``; a spaced list ``[12, 13]`` is a citation too;
* a ratio is dimensionless: a unit or a multiplier after it refuses the ratio production ('150/300 mg', '120/80 mmHg');
* a glued bare capital M, or a scale word after the number ('1.5 million', '1.5 k'), raises ``scaled_header``;
* a thousands-grouped mantissa never carries an exponent ('1.234.567 x 10^3', '1.234.567e3');
* on a two-column page the token is read in its column: a scale or a qualifier on the column's next or previous line is
  never hidden by the merged page text.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from app.evidence import models as M
from app.evidence.numgrammar import tokenizer as TK
from app.evidence.numgrammar import tokens
from app.evidence.proposal_check import check_proposals
from tests.evidence_r5b_support import check
from tests.evidence_r6_round1_support import cell, run

FORMS = Path(__file__).resolve().parents[1] / "app" / "evidence" / "numgrammar" / "forms.py"


# ------------------------------------------------------------- forms accept, the allow-list refuses ---
REFUSED_BY_THE_WINDOW = ["12 3/4", "10 ^ 3", "10 ³", "12 .5", "12 ,5", "Ratio 1 : 2 dilution", "Date 5 / 12 / 2020",
                         "Cmax 12 /15 patients", "Patients took 2 x 10 mg tablets", "2 × daily", "1.2 × l0³",
                         "x 103 cells", "12 x 3/4", "dosed 2x", "12* 10 3"]


@pytest.mark.parametrize("text", REFUSED_BY_THE_WINDOW)
def test_every_refused_multi_word_shape_is_unresolved(text):
    got = tokens(text)
    assert got and all(t.value is None for t in got), (text, got)


@pytest.mark.parametrize("text", REFUSED_BY_THE_WINDOW)
def test_the_refusal_comes_from_the_neighbour_allow_list_not_from_a_form(text, monkeypatch):
    monkeypatch.setattr(TK.Reader, "window_fault", lambda self, fw, lw, dash: None)
    assert any(t.value is not None for t in tokens(text)), text          # no form refuses it on its own


def test_forms_py_holds_only_the_accepting_productions():
    src = FORMS.read_text("utf-8")
    tree = ast.parse(src)
    names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert not names & {"form_fraction", "form_detached", "form_fragment", "ratio_run", "operand", "ten"}, names
    strings = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert not strings & {"mixed_number", "detached_exponent", "detached_decimal", "fraction", "multiplier_operand",
                          "ten_after_multiplier"}, strings


@pytest.mark.parametrize("text, why", [
    ("12 3/4", "mixed_number"), ("10 ^ 3", "detached_exponent"), ("12 .5", "detached_decimal"),
    ("Ratio 1 : 2 dilution", "fraction"), ("Patients took 2 x 10 mg tablets", "ten_after_multiplier"),
    ("2 × daily", "multiplier_operand"), ("1.2 10^3", "detached_exponent"), ("x 103 cells", "ten_after_multiplier")])
def test_the_refused_word_is_named_for_the_reviewer(text, why):
    assert any(t.why == why for t in tokens(text)), (text, tokens(text))


@pytest.mark.parametrize("text, want", [("1.2 x 10^3 cells", [1200.0]), ("1.2 ×10⁻³", [0.0012]),
                                        ("1.2× 10^3", [1200.0]), ("12 / 15 of", [0.8]), ("2 x 3 mg", [2.0, 3.0]),
                                        ("12 - 15 mg", [12.0, 15.0]), ("Responders 12/15 (80%)", [0.8, 80.0])])
def test_the_accepting_productions_still_read(text, want):
    assert [t.value for t in tokens(text)] == want


# ------------------------------------------------------------------------- one bracket production ---
@pytest.mark.parametrize("style", [".", ","])
def test_a_reference_pair_after_a_number_is_a_bracket_list_production(style):
    assert [(t.value, t.why, t.parts) for t in tokens("12 [1,2] here", style)] == [(12.0, "", ()),
                                                                                   (None, "bracket_list", (1.0, 2.0))]
    assert [(t.value, t.why) for t in tokens("[1,5] mg", style)] == [(None, "bracket_list")]
    assert [t.value for t in tokens("[0,5] mg", ",")] == [0.5]                    # head 0: a decimal, no list


def test_the_scan_has_no_second_citation_detector():
    from app.evidence.numgrammar import scan
    assert not hasattr(scan, "_CITATION") and "citation_list" not in Path(scan.__file__).read_text("utf-8")


@pytest.mark.parametrize("page, value, refs", [
    ("P-gp efflux [1, 2, 3] was reported", "3", (1.0, 2.0, 3.0)), ("P-gp mediated efflux [12, 13] was reported", "13",
                                                                   (12.0, 13.0)),
    ("P-gp mediated efflux [12,13 ] was reported", "12.13", (12.0, 13.0)), ("efflux [ 12 ] was", "12", (12.0,)),
    ("efflux [12 - 15] was", "15", (12.0, 15.0))])
def test_a_spaced_reference_list_after_a_word_is_a_citation(page, value, refs):
    cites = [t for t in tokens(page) if t.why == "citation"]
    assert [t.parts for t in cites] == [refs] and all(t.value is None for t in tokens(page)), tokens(page)
    res = check(value, page, page)
    assert res.status == "fail", res
    if "." not in value:                                   # a reference number cited alone: the citation sentence
        assert res.reason == M.REASON_UNRESOLVED_SHAPE and "citation" in res.detail, res


# --------------------------------------------------------------------- a ratio is dimensionless ---
@pytest.mark.parametrize("page, value", [("patients received 150/300 mg daily", "150/300"),
                                         ("blood pressure 120/80 mmHg", "120/80"), ("ratio 12 / 15 x 10^3 cells", "12/15"),
                                         ("ratio 12/15 x 1000", "12/15"), ("dose 3/4 mg/kg", "3/4"), ("BP 140/90 mmHg", "140/90"),
                                         ("dose 1/2 × 10^3", "1/2"), ("ratio 12/15 x", "12/15")])
def test_a_ratio_with_a_unit_or_a_multiplier_after_it_is_refused(page, value):
    res = check(value, page, page)
    assert res.status == "fail", (page, res)
    assert not any(t.shape == "ratio" and t.value is not None for t in tokens(page)), tokens(page)


@pytest.mark.parametrize("page", ["Responders 12/15 (80%) overall", "ratio 12/15 of", "grade 3/4 events", "(12/15)."])
def test_a_ratio_before_a_word_that_is_no_unit_still_reads(page):
    assert [t.shape for t in tokens(page) if t.value is not None][0] == "ratio", page


# ------------------------------------------------------------------- a scale written after the number ---
@pytest.mark.parametrize("page", ["1.5M patients were enrolled", "1.5 million patients were enrolled", "1.5 thousand",
                                  "1.5 k patients", "1.5 bn", "1.5 lakh", "1.5 Mio", "1.5 mn", "1.5 Millions", "(1.5M)",
                                  "1.5 M patients", "1.5 thоusand"])
def test_a_scale_word_or_a_glued_bare_m_after_the_number_flags_scaled_header(page):
    got = run(page, page, "1.5")
    assert got.status == "fail" or M.FLAG_SCALED_HEADER in got.flags, (page, got)


def test_a_scale_word_in_a_cell_flags_and_a_molar_prefix_does_not():
    assert M.FLAG_SCALED_HEADER in cell([["P", "N"], ["Arm", "1.5 million"]], 1, 1, "1.5").flags
    for page in ("5 mM buffer", "12 nM", "1.5 mg daily", "1.5 h later"):
        assert M.FLAG_SCALED_HEADER not in run(page, page, page.split()[0]).flags, page


# --------------------------------------------------------------- a grouped mantissa carries no exponent ---
@pytest.mark.parametrize("page, style", [("value 1.234.567 x 10^3 here", "."), ("value 1.234.567e3 here", "."),
                                         ("value 1.234.567x10^3 here", "."), ("value 1,234,567e3 here", ","),
                                         ("value 1,234,567 x 10^3 here", "."), ("value 1'234 x 10^3", "."),
                                         ("value 1,234.5e3", ".")])
def test_a_thousands_grouped_mantissa_before_an_exponent_is_unresolved(page, style):
    assert all(t.value is None for t in tokens(page, style)), (page, tokens(page, style))


def test_a_one_separator_mantissa_still_reads():
    assert [t.value for t in tokens("1.234 x 10^3")] == [1234.0] and [t.value for t in tokens("1.234e3")] == [1234.0]


# ------------------------------------------------------------------------- the two-column page ---
def _two_columns(left: list[str], right: list[str]) -> dict:
    cols = ["\n".join(left), "\n".join(right)]
    text = "\n".join(f"{a} {b}" for a, b in zip(left, right, strict=True))
    return {"sha256": "x" * 64, "decimal_style": ".", "pages": [{"n": 1, "text": text, "small_text": "",
                                                                  "columns": cols, "tables": []}]}


def _check(doc: dict, value: str, snippet: str, unit: str | None = None) -> dict:
    template = {"fields": [{"name": "x", "type": "number"}]}
    loc = M.Locator(page=1, kind="text", snippet=snippet, decimal_style=".").to_json()
    return check_proposals([{"source_id": "S1", "field": "x", "group_label": "", "value": value, "value_num": None,
                             "unit": unit, "locator": loc, "extractor": "llm"}], doc, template)[0]


PLATELETS = (["Baseline platelet count was 2.5", "x 10^9 cells/L in the cohort and", "no subject was excluded."],
             ["Dosing was adjusted for renal", "function in every participant as", "described in the methods."])


def test_a_column_line_end_number_is_read_in_its_column_never_against_the_merged_line():
    rec = _check(_two_columns(*PLATELETS), "2.5", "Baseline platelet count was 2.5")
    assert rec["verify_status"] == "fail", rec
    assert _check(_two_columns(*PLATELETS), "2.5e9", "Baseline platelet count was 2.5\nx 10^9")["verify_status"] == "pass"


@pytest.mark.parametrize("right, snippet, value", [
    (["Trough levels were less than", "5 ng/mL in all subjects at", "steady state after dosing."], "5 ng/mL in all subjects",
     "5"),
    (["Plasma half-life ranged from 12 to", "15 h across the dose levels", "in every cohort studied."],
     "15 h across the dose levels", "15")])
def test_a_qualifier_or_range_word_on_the_previous_column_line_is_never_hidden(right, snippet, value):
    left = ["The study enrolled 24 healthy", "volunteers who received the drug", "once daily for two weeks."]
    rec = _check(_two_columns(left, right), value, snippet)
    assert rec["verify_status"] == "pass" and rec["locator"]["column"] == 1, rec
    assert M.FLAG_CENSORED_VALUE in [f["code"] for f in rec["flags"]], rec


def test_a_line_that_spans_both_columns_is_still_read_in_the_page_text():
    left, right = ["Left column says 12 mg", "more left text here"], ["Right column says 30 mg", "more right text"]
    doc = _two_columns(left, right)
    doc["pages"][0]["text"] = "Title of the paper 2024 edition\n" + doc["pages"][0]["text"]
    rec = _check(doc, "2024", "Title of the paper 2024 edition")
    assert rec["verify_status"] == "pass" and rec["locator"]["column"] is None, rec
