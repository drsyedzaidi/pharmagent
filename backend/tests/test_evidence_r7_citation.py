"""r7 owner decision 7: an unmistakable bracketed citation is a NON-VALUE token and is ignored.

    citation = "[" ref {("," | "-") ref} "]"      ref = [1-9] digit{0,2}; no digit right after the closing bracket

It is unmistakable when it is glued to a word (after a letter, or after a trailing closer that follows a letter:
``studies[47-49]``, ``al.[1]``, ``(ADC)[4]``, a chain ``[1][2]``) or when the word before it ends with a letter
(``efflux [46].``, ``Unadkat [67] and``, ``[27] [28]``). After a number (``89 [46]``, an IQR ``9[4-9]``), after a
multiplier word, or with no word before it (a table cell ``[62]``) a bracket keeps its old reading.

The citation is ignored by the words around it (it neither taints a number, nor ends a range, nor is an operand), and it is
one unresolved expression of its own (``unresolved_shape:citation``, parts = the reference numbers): a record that cites a
number inside it fails with a sentence that says so.
"""
from __future__ import annotations

import pytest

from app.evidence import models as M
from app.evidence.numgrammar import tokens
from tests.evidence_r6_round1_support import cell, run


def V(text: str, mode: str = "page") -> list[float | None]:
    return [t.value for t in tokens(text, mode=mode)]


@pytest.mark.parametrize("text, literal, parts", [
    ("passive diffusion and P-gp efflux [46]. Population sim", "[46]", (46.0,)),
    ("reducing efficacy [14\u201316]. Information", "[14\u201316]", (14.0, 16.0)),
    ("in pregnant women [1,2] and", "[1,2]", (1.0, 2.0)),
    ("and Unadkat [67] and De Sousa", "[67]", (67.0,)),
    ("exploredinthreestudies[47\u201349]suggestedthat", "[47\u201349]", (47.0, 49.0)),
    ("Sir,\u2013ThestudybyAnsch\u00fctzetal.[1]was", "[1]", (1.0,)),
    ("conjugates(ADC)[4],etc.", "[4]", (4.0,)),
    ("% of those with CH [30]. The", "[30]", (30.0,)),
])
def test_an_unmistakable_citation_is_one_non_value_expression(text, literal, parts):
    (t,) = tokens(text)
    assert (text[t.start:t.end], t.value, t.shape, t.parts) == (literal, None, "unresolved_shape:citation", parts)


def test_a_chain_of_citations_is_read_citation_by_citation():
    got = tokens("the preterm group. [27] [28] Furthermore [3][4]")
    assert [(t.shape, t.parts) for t in got] == [("unresolved_shape:citation", (27.0,)),
                                               ("unresolved_shape:citation", (28.0,)),
                                               ("unresolved_shape:citation", (3.0,)),
                                               ("unresolved_shape:citation", (4.0,))]


@pytest.mark.parametrize("text, want", [
    ("CYP3A4 Vmax 89 [46] and", [89.0, 46.0]),                  # a value then its reference: not unmistakable
    ("Apgar 9[4\u20139] 9[5\u201310] NS", [9.0, 4.0, 9.0, 9.0, 5.0, 10.0]),   # a median and its range
    ("[1] Smith J, Jones K.", [1.0]),                           # no word before it
    ("AUC[0\u201324] was", [0.0, 24.0]),                         # 0 is no reference number: the old label reading
    ("dose 2 x [12] tablets", [2.0, 12.0]),                    # after a multiplier word
])
def test_a_bracket_that_is_not_unmistakably_a_citation_keeps_its_old_reading(text, want):
    assert V(text) == want


def test_a_table_cell_that_is_only_a_bracket_keeps_its_reading_and_a_cell_citation_is_one():
    assert V("[62]", mode="cell") == [62.0]
    (t,) = tokens("Smith et al. [62]", mode="cell")
    assert t.shape == "unresolved_shape:citation"


def test_a_citation_never_makes_a_neighbouring_number_unresolved_or_a_range_end():
    assert V("Cmax was 12.5 mg/L [3]. AUC was 45 mg.h/L [4,5] overall") == [12.5, None, 45.0, None]
    toks = tokens("clearance increased by 12 % in pregnancy [21] -\n[22] and")
    assert [(t.value, t.ranged) for t in toks if t.value is not None] == [(12.0, False)]
    assert V("efflux [46]\u200b and\nok 12 mg") == [None, 12.0]      # the outside character taints its line only


def test_a_record_citing_a_number_inside_a_citation_fails_with_a_clear_reason():
    page = "passive diffusion and P-gp efflux [46]. Population sim"
    got = run(page, "efflux [46].", "46")
    assert (got.status, got.reason) == ("fail", M.REASON_UNRESOLVED_SHAPE)
    assert "citation" in got.detail and "not a value" in got.detail
    ok = run("Cmax was 12.5 mg/L [3]. AUC", "Cmax was 12.5 mg/L [3].", "12.5")
    assert (ok.status, ok.value_num) == ("pass", 12.5) and "censored_value" not in ok.flags
    rows = [["Parameter", "Value", "Ref"], ["Vmax", "89", "Smith et al. [46]"]]
    assert cell(rows, 1, 1, "89").status == "pass"
    ref = cell(rows, 1, 2, "46", snippet="Smith et al. [46]")
    assert (ref.status, ref.reason) == ("fail", M.REASON_UNRESOLVED_SHAPE)
