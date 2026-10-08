"""Attack round 3 regressions on number reading (verifier level, no model needed).

A character a reader takes as part of the number, but the tokenizer took as a boundary, must never
let a FRAGMENT of the printed number pass clean:

- a letter glued to the digits ('12З' drawn like 123, '1O5', '23l');
- a multiplication-sign look-alike in scientific notation ('1.2 ⋅ 10^3');
- a decimal / grouping separator look-alike between digits ('12․4', '24‘560', backtick);
- a minus / plus look-alike in sign position ('˗20', '1.2E˗03');
- a vulgar fraction glued after a number ('12½');
- an ISO thin-space decimal group ('0.123 4').

Units: an SI prefix case ('MIU' vs 'mIU') and a per-mille sign are different units.
"""
from __future__ import annotations

import pytest

from app.evidence import models as M
from app.evidence.extract_rules import rule_extract, value_number
from app.evidence.extract_tools import check_proposals
from app.evidence.numgrammar import tokens

STRONG = "Clearance 12.4 L/h, volume 45.6 L, half-life 3.5 h, ka 1.25 /h. "
CMAX = {"fields": [{"name": "Cmax", "type": "number", "unit": "ng/mL"}]}
CHG = {"fields": [{"name": "change", "type": "number", "unit": "%"}]}
DOSE = {"fields": [{"name": "dose", "type": "number", "unit": "mg"}, {"name": "tmax", "type": "number", "unit": "h"}]}
K = {"fields": [{"name": "k", "type": "number"}]}
CELL = {"table": 0, "row": 1, "col": 1}


def _doc(text: str, *, tables: list | None = None, style: str = ".") -> dict:
    return {"sha256": "x" * 64, "decimal_style": style,
            "pages": [{"n": 1, "text": text, "small_text": "", "columns": None, "hidden_chars": 0, "ink_check": "ok",
                       "glyph_check": {},
                       "tables": [{"index": i, "strategy": "lines", "rows": r} for i, r in enumerate(tables or [])]}]}


def _prop(field: str, value: str, snippet: str, *, unit: str | None = None, cell: dict | None = None) -> dict:
    loc = M.Locator(page=1, kind="table" if cell else "text", snippet=snippet, cell=cell, decimal_style=".").to_json()
    return {"source_id": "S1", "field": field, "group_label": "", "value": value,
            "value_num": value_number(value, "."), "unit": unit, "locator": loc, "extractor": "llm"}


def _one(doc: dict, template: dict, prop: dict) -> dict:
    return check_proposals([prop], doc, template)[0]


def _clean(rec: dict) -> bool:
    return rec["verify_status"] == "pass" and not any(f["severity"] == "error" for f in rec["flags"])


def _codes(rec: dict) -> set[str]:
    return {f["code"] for f in rec["flags"]}


# ----------------------------------------------------------- glued letters ---

@pytest.mark.parametrize("text", ["23l", "23I", "4б", "12З", "1O5", "б12"])
def test_letter_glued_to_digits_makes_the_token_unreadable(text):
    toks = tokens(text, ".", mode="page")
    assert toks and all(t.value is None for t in toks), toks


@pytest.mark.parametrize("text,want", [
    ("150mg", (150.0, "int")), ("24h", (24.0, "int")), ("1.73m2", (1.73, "decimal")), ("100µg", (100.0, "int")),
    ("10mL", (10.0, "int")), ("3MIU", (None, "unresolved_shape:glued_letters")),
    ("1OmL", (None, "unresolved_shape:glued_letters")), ("23l", (None, "unresolved_shape:glued_letters")),
    ("150mgtid", (None, "unresolved_shape:glued_letters")), ("10m L", (None, "unresolved_shape:split_unit")),
    ("2x daily", (None, "unresolved_shape:multiplier_operand"))])
def test_a_glued_unit_reads_through_the_unit_production_and_a_broken_one_is_unresolved(text, want):
    """Was test_a_glued_unit_is_unresolved (r6: every glued letter rejected the word). r7, owner decision 2: a unit
    glued to the digits is read by the grammar's unit production (an explicit prefix and base set, units.py) and the
    number is the literal alone; a glued word that is not a full production ('3MIU', '1OmL', '23l', '150mgtid') or a
    unit the next word continues ('10m L') stays unresolved. '2x daily' is the multiplier without a number."""
    assert [(t.value, t.shape) for t in tokens(text, ".", mode="page")] == [want]


@pytest.mark.parametrize("cell,value", [("12З", "12"), ("1O5", "1"), ("23l", "23")])
def test_table_cell_with_glued_lookalike_letter_never_passes_clean(cell, value):
    doc = _doc(STRONG + f"Table 1\nDose Cmax (ng/mL)\n150 mg {cell}",
               tables=[[["Dose", "Cmax (ng/mL)"], ["150 mg", cell]]])
    assert not _clean(_one(doc, CMAX, _prop("Cmax", value, cell, unit="ng/mL", cell=CELL)))
    for rec in check_proposals(rule_extract(doc, CMAX, "S1")["proposals"], doc, CMAX):
        assert not _clean(rec) or rec["value_num"] != float(value), rec


# ------------------------------------------------- multiplier look-alikes ---

MULTS = ["⋅", "∙", "⨯", "✕", "✖", "•", "х", "Х"]


@pytest.mark.parametrize("mul", MULTS)
def test_lookalike_multiplier_is_scientific_notation(mul):
    page = STRONG + f"Mean Cmax was 1.2 {mul} 10^3 ng/mL in fasted subjects."
    assert not _clean(_one(_doc(page), CMAX, _prop("Cmax", "1.2", f"Mean Cmax was 1.2 {mul} 10^3", unit="ng/mL")))
    full = _one(_doc(page), CMAX, _prop("Cmax", f"1.2 {mul} 10^3", f"Mean Cmax was 1.2 {mul} 10^3", unit="ng/mL"))
    assert full["verify_status"] != "pass" or full["value_num"] == 1200.0, full


def test_lookalike_multiplier_in_keyless_table_cell():
    cell = "1.2 ⋅ 10^3"
    doc = _doc(STRONG + f"Table 1\nDose Cmax (ng/mL)\n150 mg {cell}",
               tables=[[["Dose", "Cmax (ng/mL)"], ["150 mg", cell]]])
    for rec in check_proposals(rule_extract(doc, CMAX, "S1")["proposals"], doc, CMAX):
        assert not (_clean(rec) and rec["value_num"] == 1.2), rec


# ------------------------------------------------- separator look-alikes ---

SEPS = ["․", "．", "﹒", "‚", "，", "﹐", "،", "‘", "´", "ʹ",
        "٬", "`"]


@pytest.mark.parametrize("sep", SEPS)
def test_separator_lookalike_between_digits_never_passes_a_piece(sep):
    page = STRONG + f"Mean Cmax was 24{sep}560 ng/mL at steady state."
    for piece in ("24", "560"):
        rec = _one(_doc(page), CMAX, _prop("Cmax", piece, f"Mean Cmax was 24{sep}560 ng/mL", unit="ng/mL"))
        assert not _clean(rec), (sep, piece, rec["verify_reason"])


def test_separator_lookalike_in_keyless_table_cell():
    cell = "24‘560"
    doc = _doc(STRONG + f"Table 1\nDose Cmax (ng/mL)\n150 mg {cell}",
               tables=[[["Dose", "Cmax (ng/mL)"], ["150 mg", cell]]])
    for rec in check_proposals(rule_extract(doc, CMAX, "S1")["proposals"], doc, CMAX):
        assert not _clean(rec), rec


# ------------------------------------------------------ sign look-alikes ---

@pytest.mark.parametrize("sign", ["\u02d7", "\u2796", "\u2043", "\u2e3a", "\u2e3b", "\u00ad", "\u207b"])
def test_minus_lookalike_before_a_number_is_never_dropped(sign):
    page = STRONG + f"The mean change from baseline was {sign}20% at week 12."
    rec = _one(_doc(page), CHG, _prop("change", "20", f"change from baseline was {sign}20%", unit="%"))
    assert not _clean(rec), (hex(ord(sign)), rec["verify_reason"], rec["flags"])


@pytest.mark.parametrize("sign", ["\ufe62", "\uff0b"])
def test_plus_lookalike_reads_as_plus(sign):
    page = STRONG + f"The mean change from baseline was {sign}20% at week 12."
    rec = _one(_doc(page), CHG, _prop("change", "-20", f"change from baseline was {sign}20%", unit="%"))
    assert rec["verify_status"] == "fail", rec
    rec = _one(_doc(page), CHG, _prop("change", "20", f"change from baseline was {sign}20%", unit="%"))
    assert rec["verify_status"] != "pass" or rec["value_num"] == 20.0, rec


@pytest.mark.parametrize("lit", ["1.2E˗03", "1.2E＋03", "1.2E﹢03"])
def test_exponent_with_lookalike_sign_is_not_its_mantissa(lit):
    page = STRONG + f"Cmax was {lit} ng/mL in fasted subjects."
    assert not _clean(_one(_doc(page), CMAX, _prop("Cmax", "1.2", f"Cmax was {lit} ng/mL", unit="ng/mL")))


# ------------------------------------------------- fractions, ISO groups ---

@pytest.mark.parametrize("page,prop", [
    ("Patients received 12½ mg twice daily.", _prop("dose", "12", "Patients received 12", unit="mg")),
    ("Median tmax was 2¼ h after dosing.", _prop("tmax", "2", "Median tmax was 2", unit="h")),
])
def test_vulgar_fraction_glued_after_a_number(page, prop):
    assert not _clean(_one(_doc(STRONG + page), DOSE, prop))


def test_iso_thin_space_decimal_group_is_not_cut():
    page = STRONG + "The rate constant was 0.123 4 per hour."
    assert not _clean(_one(_doc(page), K, _prop("k", "0.123", "The rate constant was 0.123 4")))


def test_thin_space_thousands_group_still_reads():
    page = STRONG + "The AUC was 18 900 ng·h/mL."
    rec = _one(_doc(page), K, _prop("k", "18900", "The AUC was 18 900"))
    assert rec["verify_status"] == "pass" and rec["value_num"] == 18900.0, rec


# ------------------------------------------------------------------ units ---

@pytest.mark.parametrize("printed,proposed", [("MIU", "mIU"), ("Mg", "mg"), ("mIU", "MIU")])
def test_si_prefix_case_is_a_different_unit(printed, proposed):
    template = {"fields": [{"name": "dose", "type": "number", "unit": proposed}]}
    page = STRONG + f"Interferon alfa was given at 3 {printed} three times weekly."
    rec = _one(_doc(page), template, _prop("dose", "3", f"given at 3 {printed}", unit=proposed))
    assert M.FLAG_UNIT_SOURCE_MISMATCH in _codes(rec), rec["flags"]


def test_same_prefix_case_still_matches():
    template = {"fields": [{"name": "dose", "type": "number", "unit": "MIU"}]}
    page = STRONG + "Interferon alfa was given at 3 MIU three times weekly."
    rec = _one(_doc(page), template, _prop("dose", "3", "given at 3 MIU", unit="MIU"))
    assert _clean(rec) and M.FLAG_UNIT_SOURCE_MISMATCH not in _codes(rec), rec["flags"]


def test_all_caps_page_unit_is_not_a_clean_milli_match():
    template = {"fields": [{"name": "dose", "type": "number", "unit": "mg"}]}
    page = STRONG + "PATIENTS RECEIVED 150 MG ONCE DAILY."
    rec = _one(_doc(page), template, _prop("dose", "150", "RECEIVED 150 MG", unit="mg"))
    assert _codes(rec) & {M.FLAG_UNIT_UNVERIFIED, M.FLAG_UNIT_SOURCE_MISMATCH}, rec["flags"]


def test_per_mille_is_not_percent():
    template = {"fields": [{"name": "fe", "type": "number", "unit": "%"}]}
    page = STRONG + "Urinary recovery was 4.5\u2030 of the dose."
    rec = _one(_doc(page), template, _prop("fe", "4.5", "Urinary recovery was 4.5\u2030", unit="%"))
    assert M.FLAG_UNIT_SOURCE_MISMATCH in _codes(rec), rec["flags"]


def test_a_per_ten_thousand_sign_is_outside_the_alphabet_and_never_a_percent():
    """Was test_per_mille_is_not_percent[U+2031] (passed + unit_source_mismatch): U+2031 is not in the allowed
    alphabet, so the cited number touches an unreadable character: verification fails closed."""
    template = {"fields": [{"name": "fe", "type": "number", "unit": "%"}]}
    page = STRONG + "Urinary recovery was 4.5\u2031 of the dose."
    rec = _one(_doc(page), template, _prop("fe", "4.5", "Urinary recovery was 4.5\u2031", unit="%"))
    assert rec["verify_status"] == "fail" and rec["verify_reason"] == M.REASON_UNRESOLVED_CONTEXT, rec
