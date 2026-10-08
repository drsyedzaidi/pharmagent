"""r7 round 3 follow-up (findings 2 and 15 of the round-3 audit, the glued form): a TIMES SIGN or a MIDDLE DOT printed
right before a digit inside a letter-led word is the multiplier production (``MULT number``), never glue of a label
name. Decision 3 reads the digits of a label as letters of its name ('AUC0-24', 'CYP2D6'); read that way 'AUC×1000'
looked like a name with a joined product dot ('AUC×IOOO'), so a header 'AUC×1000' raised no scaled_header and the text
'Count×1000 250' passed 250 with no flag. The label production now refuses a product dot before a printed digit
(glue.label; the neighbour-shape oracle states the same rule independently), so such a word is no inert neighbour and
no scale-free header part.

A product dot BETWEEN LETTERS stays unit glue ('ng·h/mL', 'ng×h'); the star of an allele ('CYP2D6*4', 69 corpus uses)
is no product dot and stays a label. The Latin letter x glued after letters ('AUCx1000') is a letter of the name
(decision 3; 'Vmax1', 'Cmax1', 'Prax2001' in the corpus have the same shape) and is left to the page image: a rule for
it would need a letter-case heuristic, which is a shape detector (docs/evidence-pipeline-limits.md).
"""
from __future__ import annotations

import pytest

from app.evidence import models as M
from app.evidence.numgrammar import glue as G
from app.evidence.numgrammar import scale_shapes, tokens
from tests.evidence_r6_neighbour_rules import label as oracle_label
from tests.evidence_r6_round1_support import cell, run

HEAD_ROW = ["Arm", "Cmax"]


@pytest.mark.parametrize("word", ["AUC×1000", "AUC·1000", "Count×1000", "AUC×10^3", "AUC×1e3", "Dose×2", "q×1",
                                  "(AUC×1000)", "AUC0-24×1000"])
def test_a_product_dot_before_a_digit_is_no_label_glue(word):
    assert G.label(word) is False, word
    assert oracle_label(word) is False, word                     # the independent copy of the production agrees


@pytest.mark.parametrize("word", ["AUC0-24", "CYP2D6*4", "CYP2C19*1/*1", "ng·h/mL", "ng×h", "L·h⁻¹", "t1/2",
                                  "Cmax,ss", "C24h", "AUCx1000"])
def test_a_label_without_a_product_dot_before_a_digit_is_still_a_label(word):
    assert G.label(word) is True, word
    assert oracle_label(word) is True, word


@pytest.mark.parametrize("header", ["AUC×1000", "AUC·1000", "Count×1000", "AUC×1e3 ng·h/mL", "Cmax×10 (ng/mL)",
                                    "Dose×2"])
def test_a_header_with_a_glued_times_sign_before_digits_raises_the_scale_flag(header):
    assert scale_shapes(header) is True, header
    got = cell([HEAD_ROW[:1] + [header], ["A", "12.5"]], 1, 1, "12.5")
    assert got.status == "pass" and M.FLAG_SCALED_HEADER in got.flags, (header, got)


@pytest.mark.parametrize("header", ["AUC0-24 (ng·h/mL)", "CYP2D6*4", "Cmax (ng/mL)", "AUC (ng×h/mL)"])
def test_a_product_dot_between_letters_or_an_allele_star_is_no_scale(header):
    assert scale_shapes(header) is False, header


@pytest.mark.parametrize("page, value", [("Count×1000 250 cells", "250"), ("AUC×1000 12.5 ng", "12.5"),
                                         ("AUC×1000: 12.5", "12.5"), ("Count·1000 250", "250"),
                                         ("Platelets×10 250", "250")])
def test_a_number_beside_a_label_with_a_glued_times_sign_before_digits_is_refused(page, value):
    out = run(page, page, value, field="dose")
    assert out.status == "fail", (page, out)
    at = page.index(value, page.index(" "))
    assert all(t.value is None for t in tokens(page) if t.start <= at < t.end), tokens(page)


@pytest.mark.parametrize("text, want", [("AUC0-24 120 ng", [120.0]), ("CYP2D6*4 carriers 12.5", [12.5]),
                                        ("AUC 12.5 ng·h/mL", [12.5]), ("AUC (ng×h/mL) 12.5", [12.5])])
def test_the_ordinary_labels_and_units_beside_a_number_still_read(text, want):
    assert [t.value for t in tokens(text)] == want, (text, tokens(text))
