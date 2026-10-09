"""r7 round 3: the flag checks around a verified number (findings of the round-3 audit).

* scaled_header is an ALLOW-LIST over the label's words (numgrammar/scale.py): a label is scale-free only when every part
  of every word (split at brackets and slashes) is punctuation, prose, a label whose powers follow unit symbols, a power
  of ten as a unit denominator ('cells/10^6', 'per 10^6'), a sample size ('n = 45'), a percentage, or a plain number that
  is no power of ten; a multiplier, an exponent letter before digits, a power or written-out power of ten, a scale word
  (a look-alike spelling too, digits inside letters included) or a character the reader cannot read raises the flag;
* a scale word right after the number: the measured spellings (mln, mil, bil, bln, hundred, lac, B, MM) and look-alikes;
* hidden qualifiers: digits inside a qualifier word ('bel0w'), a qualifier broken at a line-end hyphen ('approxi-\\nmately'),
  the MEDLINE spelling '< or =', comparatives before 'than', 'or less' / 'and above' after the number, 'between N and M',
  NMT / NLT / around / nearly / roughly / within;
* a range whose left end carries a unit or a percent before the range word ('20 mg - 40 mg', '20 mg to 40 mg').
"""
from __future__ import annotations

import pytest

from app.evidence import models as M
from app.evidence.scale_label import scale_shapes
from app.evidence.word_match import loose_in
from tests.evidence_r6_round1_support import cell, run

SCALED, CENSORED = M.FLAG_SCALED_HEADER, M.FLAG_CENSORED_VALUE


# ----------------------------------------------------------------------------- scaled_header allow-list ---
@pytest.mark.parametrize("label", [
    "AUC (×10−3 ng·h/mL)", "AUC (x100 ng/mL)", "AUC (×1e3 ng·h/mL)", "AUC (x 10000)", "AUC (x10,000)",
    "AUC (hundreds)", "AUC (×1000)", "AUC(x1000)", "AUC(x10^3)", "AUC(10^3 ng.h/mL)", "Cmax (ng/mL)(x10^3)",
    "Cmax (ng/mL)x10^3", "AUC(E3)", "AUC(thousands)", "AUC[x1000]", "AUC (×1OOO)", "AUC (x 1,OOO)", "AUC (th0usands)",
    "AUC (mi11ions)", "Count (10^9/L)", "AUC (1000 ng·h/mL)", "Dose (mg) x 2", "Group 10", "AUC‮abc",
    "AUC (1,000)", "AUC /1000", "AUC (2.5e3)", "AUC (× ·)"])
def test_a_label_part_that_is_no_inert_production_raises_the_scale_flag(label):
    assert scale_shapes(label) is True, label


@pytest.mark.parametrize("label", [
    "AUC0–∞ (ng·h/mL)", "Cmax (μg/L)", "Temperature (°C)", "BSA (m²)", "Dose (mg)",
    "Placebo (n = 45)", "(N = 1208)", "Methadone(n=45)", "95% CI", "t1/2 (h)", "ka (1/min)", "CL/F (L/h/70kg)",
    "Day 1", "Week 12", "CLint (µL/min/10^6 cells)", "per 10^6 cells", "cells / 10^6", "Cmax,ss †", "e", "Visit",
    "AUC0-24", "Time (h) 0.5", "150 mg", "CYP2E1", "(Cmax - Cmin) / Cav", "Mean ± SD", "> 40%", "Obs [59] Pred"])
def test_a_label_of_inert_parts_raises_no_scale_flag(label):
    assert scale_shapes(label) is False, label


def test_a_header_scale_flags_the_cell_end_to_end():
    for hdr in ("AUC (×10−3 ng·h/mL)", "AUC(x1000)", "Cmax (ng/mL)(x10^3)", "AUC (th0usands)", "AUC (×1OOO)"):
        assert SCALED in cell([["Arm", hdr], ["A", "12.5"]], 1, 1, "12.5").flags, hdr
    assert SCALED not in cell([["Arm", "AUC (ng·h/mL)"], ["A", "12.5"]], 1, 1, "12.5").flags


def test_a_data_cell_beside_the_number_is_data_not_a_scale():
    rows = [["Arm", "Cmax", "AUC"], ["A", "12.5", "1.2 × 10^3"], ["B", "12.5", "10^3"], ["C", "12.5", "(thousands)"]]
    assert SCALED not in cell(rows, 1, 1, "12.5").flags
    assert cell(rows, 2, 1, "12.5").status == "fail"                   # a bare power beside it: a detached scale
    assert SCALED in cell(rows, 3, 1, "12.5").flags                     # a scale word beside it raises the flag
    from app.evidence.scale_label import scale_shapes as S
    assert S("1.2 × 10^3", data=True) is False and S("10^3", data=True) is True and S("x1000", data=True) is True


@pytest.mark.parametrize("page, value", [("Exposure was 1.5 mi11ion person-years", "1.5"),
                                         ("an estimated 1.5 th0usand", "1.5"), ("an estimated 1.5 mln patients", "1.5"),
                                         ("an estimated 1.5 mil patients", "1.5"), ("sales of 1.5 B in 2020", "1.5"),
                                         ("sales of 1.5 MM in 2020", "1.5"), ("an estimated 1.5 hundred patients", "1.5"),
                                         ("an estimated 1.5 lacs patients", "1.5"), ("an estimated 1.5 bil", "1.5"),
                                         ("an estimated 1.5 bln", "1.5")])
def test_a_scale_word_after_the_number_flags(page, value):
    snip = page[page.index(value):]
    out = run(page, snip, value, field="dose")
    assert out.status == "pass" and SCALED in out.flags, (page, out)


# --------------------------------------------------------------------------------- hidden qualifiers ---
@pytest.mark.parametrize("page, value", [
    ("Cmax was bel0w 0.5 ng/mL", "0.5"), ("concentrations appr0ximately 0.5 ng/mL", "0.5"), ("values BL0Q 0.5 ng/mL", "0.5"),
    ("LLOQ was ab0ve 0.5 ng/mL", "0.5"), ("Cmax was approxi-\nmately 12.5 ng/mL", "12.5"),
    ("Cmax was approxi‐\nmately 12.5 ng/mL", "12.5"), ("Cmax was approxi­\nmately 12.5 ng/mL", "12.5"),
    ("Cmax was be-\nlow 12.5 ng/mL", "12.5"), ("LLOQ was < or = 0.5 ng/mL", "0.5"), ("age > or = 18 years", "18"),
    ("patients aged > or = 18 years", "18"), ("concentrations 0.5 or less", "0.5"), ("patients 18 or older", "18"),
    ("values 12 and above", "12"), ("lower than 0.5 ng/mL", "0.5"), ("fewer than 10 patients", "10"),
    ("around 12 h", "12"), ("nearly 12 h", "12"), ("roughly 12 h", "12"), ("NMT 12 h", "12"), ("assay NLT 90 %", "90"),
    ("between 12 and 24 h", "12"), ("between 12 and 24 h", "24"), ("within 24 h", "24"), ("12 or more", "12")])
def test_a_hidden_qualifier_raises_censored_value(page, value):
    snip = page[page.index(value):]
    out = run(page, snip, value, field="dose")
    assert out.status == "pass" and CENSORED in out.flags, (page, out)


@pytest.mark.parametrize("page, value", [("Cmax was 12.5 ng/mL", "12.5"), ("C1 12.5 mg/L", "12.5"),
                                         ("the 12-week study of 40 patients", "40"), ("12 and 24 h", "12"),
                                         ("12 or 13 patients", "12")])
def test_no_qualifier_no_flag(page, value):
    out = run(page, page[page.index(value):], value, field="dose")
    assert out.status == "pass" and CENSORED not in out.flags, (page, out)


def test_digits_inside_a_letter_word_stand_for_a_letter_in_a_word_list():
    assert loose_in("bel0w", frozenset({"below"})) and loose_in("mi11ions", frozenset({"millions"}))
    # r7 round 4: a digit run that ENDS a letter word is read through the OCR skeleton (0 for o, 1 for l): 'up t0' is
    # 'up to'; a digit with no letter look-alike ('t2') and a digit that reads as another letter ('c1' is 'cl') stay
    # apart (was: not loose_in("t0", {"to"}))
    assert not loose_in("c1", frozenset({"ca"})) and loose_in("t0", frozenset({"to"}))
    assert not loose_in("t2", frozenset({"to"})) and not loose_in("c2", frozenset({"ca"}))


# ----------------------------------------------------------------------------- a range with a unit ---
@pytest.mark.parametrize("page, value", [
    ("doses of 20 mg – 40 mg daily", "20"), ("doses of 20 mg – 40 mg daily", "40"),
    ("doses of 20 mg–40 mg daily", "20"), ("doses of 20 mg to 40 mg daily", "20"), ("doses of 20 mg to 40 mg daily", "40"),
    ("Tmax 12 h - 15", "12"), ("Tmax 12 h ~ 15 h", "12"), ("response 45 % – 55 %", "45"), ("doses of 20 mg -40 mg", "20")])
def test_both_ends_of_a_range_with_a_unit_between_are_flagged(page, value):
    out = run(page, page[page.index(value):], value, field="dose")
    assert out.status == "pass" and CENSORED in out.flags, (page, out)


def test_a_range_cell_with_units_flags_its_left_end():
    rows = [["Parameter", "Dose", "Unit"], ["Drug", "20 mg – 40 mg", "mg"]]
    assert CENSORED in cell(rows, 1, 1, "20", snippet="20 mg – 40 mg", unit="mg").flags
