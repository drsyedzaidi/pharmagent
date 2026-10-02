"""Round-2 attack regressions on number reading: a printed number must be read whole, with its
qualifier, scale and unit, or carry a flag.

- Lancet mid-dot decimals ('12·4') are one token: neither '12' nor '4' passes; 12.4 does.
- Swiss apostrophe grouping with U+2019 ('24’560') is one token.
- A non-ASCII digit glued to a number ('１2.5') makes the token unreadable.
- Censored / qualified values ('<0.5', '≤1.0', '>4', '~2') carry an error flag.
- A multiplier in the column header ('AUC ×10³ (ng·h/mL)') carries an error flag.
- The unit is read after the VERIFIED token, never after the first substring hit ('5' in '15 mg/kg').
- Bracketed citations '[1,2]' are not decimal-comma evidence; a style-dependent token on a page with
  weak style evidence carries an error flag.
- A text field never stores a number; CI / dispersion flags work for common field names.
"""
from __future__ import annotations

import pytest

from app.evidence import models as M
from app.evidence.extract_rules import rule_extract, value_number
from app.evidence.extract_tools import check_proposals
from app.evidence.normalize import decimal_evidence, infer_decimal_style


def _doc(text: str, *, tables: list[list[list[str]]] | None = None, style: str = ".",
         evidence: tuple[int, int] | None = None) -> dict:
    doc = {"sha256": "x" * 64, "decimal_style": style,
           "pages": [{"n": 1, "text": text, "small_text": "", "columns": None, "hidden_chars": 0, "ink_check": "ok",
                      "tables": [{"index": i, "strategy": "lines", "rows": rows} for i, rows in enumerate(tables or [])]}]}
    assert evidence is None or evidence == decimal_evidence(text)    # counted from the page text
    return doc


def _prop(field: str, value: str, snippet: str, *, unit: str | None = None, cell: dict | None = None,
          style: str = ".") -> dict:
    loc = M.Locator(page=1, kind="table" if cell else "text", snippet=snippet, cell=cell,
                    decimal_style=style).to_json()
    return {"source_id": "S1", "field": field, "group_label": "", "value": value,
            "value_num": value_number(value, style), "unit": unit, "locator": loc, "extractor": "llm"}


def _check(doc: dict, template: dict, *props: dict) -> list[dict]:
    return check_proposals(list(props), doc, template)


def _codes(rec: dict) -> set[str]:
    return {f["code"] for f in rec["flags"]}


def _errors(rec: dict) -> set[str]:
    return {f["code"] for f in rec["flags"] if f["severity"] == "error"}


NUM = {"fields": [{"name": "Cmax", "type": "number", "unit": "ng/mL"}]}


# ------------------------------------------------------------- mid-dot ---

MIDDOT = "Results. Mean Cmax was 12·4 ng/mL in all subjects."


@pytest.mark.parametrize("value", ["4", "12"])
def test_middot_decimal_fragments_fail(value):
    rec = _check(_doc(MIDDOT), NUM, _prop("Cmax", value, "Mean Cmax was 12·4 ng/mL"))[0]
    assert rec["verify_status"] == "fail"


@pytest.mark.parametrize("value", ["12.4", "12·4"])
def test_middot_decimal_reads_whole(value):
    rec = _check(_doc(MIDDOT), NUM, _prop("Cmax", value, "Mean Cmax was 12·4 ng/mL", unit="ng/mL"))[0]
    assert rec["verify_status"] == "pass" and rec["value_num"] == pytest.approx(12.4)


def test_keyless_middot_cell_is_never_its_integer_part():
    doc = _doc("Table 1", tables=[[["Dose", "Cmax (ng/mL)"], ["150 mg", "12·4"]]])
    recs = check_proposals(rule_extract(doc, NUM, "S1")["proposals"], doc, NUM)
    assert [r["value_num"] for r in recs] == [pytest.approx(12.4)]


def test_middot_still_multiplies_a_power_of_ten():
    rec = _check(_doc("Titre 2.5·10^4 copies"), NUM, _prop("Cmax", "25000", "Titre 2.5·10^4 copies"))[0]
    assert rec["verify_status"] == "pass" and rec["value_num"] == pytest.approx(25000)


# ------------------------------------------------------- apostrophe / fullwidth ---

APOS = "Mean AUC was 24’560 ng·h/mL in the fasted arm."


@pytest.mark.parametrize("value", ["560", "24"])
def test_typographic_apostrophe_group_fragments_fail(value):
    rec = _check(_doc(APOS), NUM, _prop("Cmax", value, "Mean AUC was 24’560 ng·h/mL"))[0]
    assert rec["verify_status"] == "fail"


def test_typographic_apostrophe_group_reads_whole():
    rec = _check(_doc(APOS), NUM, _prop("Cmax", "24560", "Mean AUC was 24’560 ng·h/mL"))[0]
    assert rec["verify_status"] == "pass" and rec["value_num"] == 24560


def test_non_ascii_digit_glued_to_a_number_fails():
    doc = _doc("Terminal half-life was １2.5 h in all subjects.")
    rec = _check(doc, NUM, _prop("Cmax", "2.5", "2.5 h in all subjects"))[0]
    assert rec["verify_status"] == "fail"


# ---------------------------------------------------------------- censored ---

CENSORED = _doc("Trough concentrations were <0.5 ng/mL in all subjects. Tmax 1.5 h when fed.",
                tables=[[["Group", "Tmax", "Cmin"], ["Fasted", "≤1.0", "<0.5"], ["Fed", ">4", "~2"]]])
TMAX_CMIN = {"fields": [{"name": "Tmax", "type": "number"}, {"name": "Cmin", "type": "number"}]}


def test_keyless_censored_cells_carry_an_error_flag():
    recs = check_proposals(rule_extract(CENSORED, TMAX_CMIN, "S1")["proposals"], CENSORED, TMAX_CMIN)
    assert len(recs) == 4
    for rec in recs:
        assert M.FLAG_CENSORED_VALUE in _errors(rec), rec["value"]


def test_llm_censored_text_value_carries_an_error_flag_and_a_plain_one_does_not():
    censored, plain = _check(CENSORED, TMAX_CMIN,
                             _prop("Cmin", "0.5", "Trough concentrations were <0.5 ng/mL"),
                             _prop("Tmax", "1.5", "Tmax 1.5 h when fed"))
    assert censored["verify_status"] == "pass" and M.FLAG_CENSORED_VALUE in _errors(censored)
    assert M.FLAG_CENSORED_VALUE not in _codes(plain)


# ------------------------------------------------------------ header scale ---

SCALED = _doc("Table 2. Exposure.", tables=[[["Dose", "AUC ×10³ (ng·h/mL)"], ["150 mg", "3.4"]]])


def test_llm_value_under_a_scaled_header_is_flagged():
    t = {"fields": [{"name": "AUC", "type": "number", "unit": "ng·h/mL"}]}
    rec = _check(SCALED, t, _prop("AUC", "3.4", "3.4", unit="ng·h/mL", cell={"table": 0, "row": 1, "col": 1}))[0]
    assert rec["verify_status"] == "pass" and M.FLAG_SCALED_HEADER in _errors(rec)


def test_keyless_value_under_a_scaled_header_is_flagged():
    t = {"fields": [{"name": "AUC", "type": "number", "unit": "ng·h/mL", "aliases": ["AUC ×10³"]}]}
    recs = check_proposals(rule_extract(SCALED, t, "S1")["proposals"], SCALED, t)
    assert len(recs) == 1 and M.FLAG_SCALED_HEADER in _errors(recs[0])


def test_plain_header_is_not_flagged_as_scaled():
    doc = _doc("Table 2.", tables=[[["Dose", "AUC (ng·h/mL)"], ["150 mg", "3400"]]])
    t = {"fields": [{"name": "AUC", "type": "number", "unit": "ng·h/mL"}]}
    rec = _check(doc, t, _prop("AUC", "3400", "3400", unit="ng·h/mL", cell={"table": 0, "row": 1, "col": 1}))[0]
    assert M.FLAG_SCALED_HEADER not in _codes(rec)


# ------------------------------------------------------------ unit position ---

UNIT_PAGE = _doc("Rats got 15 mg/kg; the human arm got 5 mg once daily.")
DOSE = {"fields": [{"name": "dose", "type": "number"}]}


def test_unit_is_read_after_the_verified_token_not_a_substring_hit():
    rec = _check(UNIT_PAGE, DOSE, _prop("dose", "5", "Rats got 15 mg/kg; the human arm got 5 mg once daily.",
                                        unit="mg/kg"))[0]
    assert rec["verify_status"] == "pass"
    assert M.FLAG_UNIT_SOURCE_MISMATCH in _errors(rec)


def test_unit_after_the_verified_token_still_vouches_for_the_right_unit():
    rec = _check(UNIT_PAGE, DOSE, _prop("dose", "15", "Rats got 15 mg/kg; the human arm", unit="mg/kg"))[0]
    assert rec["verify_status"] == "pass" and not _codes(rec) & {M.FLAG_UNIT_SOURCE_MISMATCH,
                                                                 M.FLAG_UNIT_UNVERIFIED}


# ---------------------------------------------------------- decimal style ---

CITES = ("Introduction. Drug X is a CYP2B6 substrate [1,2] with nonlinear kinetics [3,4] in adults [5,6].\n"
         "Methods. Twelve volunteers received 150 mg [7,8]. Half-life was 11.4 h.")


def test_bracketed_citations_are_not_decimal_comma_evidence():
    assert infer_decimal_style(CITES) == "."
    assert decimal_evidence(CITES) == (0, 1)


def test_style_dependent_token_on_weak_evidence_is_flagged():
    doc = _doc(CITES, tables=[[["Dose", "Cmax (ng/mL)"], ["150 mg", "3,412"]]], evidence=decimal_evidence(CITES))
    recs = check_proposals(rule_extract(doc, NUM, "S1")["proposals"], doc, NUM)
    assert len(recs) == 1 and recs[0]["value_num"] == 3412
    assert M.FLAG_DECIMAL_STYLE_AMBIGUOUS in _errors(recs[0])
    true = _check(doc, NUM, _prop("Cmax", "3412", "3,412", unit="ng/mL", cell={"table": 0, "row": 1, "col": 1}))[0]
    assert true["verify_status"] == "pass"


def test_decisive_style_evidence_raises_no_ambiguity_flag():
    text = "Half-life 11.4 h, CL 2.31 L/h, V 45.2 L and ka 1.05 per h."
    doc = _doc(text, tables=[[["Dose", "Cmax (ng/mL)"], ["150 mg", "3,412"]]], evidence=decimal_evidence(text))
    recs = check_proposals(rule_extract(doc, NUM, "S1")["proposals"], doc, NUM)
    assert M.FLAG_DECIMAL_STYLE_AMBIGUOUS not in _codes(recs[0])


# ------------------------------------------------------------- text field ---

def test_text_field_never_stores_a_number():
    t = {"fields": [{"name": "regimen", "type": "text"}]}
    doc = _doc("Pharmacokinetics of drug X. Dose regimen 12.4 mg/kg once daily.")
    rec = _check(doc, t, _prop("regimen", "2.4", "Dose regimen 12.4 mg/kg once daily"))[0]
    assert rec["value_num"] is None


# ------------------------------------------------------------ CI / dispersion ---

CI_TEXT = "CL was 3.0 L/h (95% CI 4.1 to 2.2); between-subject variability %CV -12.5 and SD(CL) -0.4."


def test_ci_and_dispersion_flags_follow_common_field_names():
    t = {"fields": [{"name": n, "type": "number"} for n in ("CL", "Lower 95% CI", "Upper 95% CI", "%CV", "SD(CL)")]}
    snip = "CL was 3.0 L/h (95% CI 4.1 to 2.2)"
    _, lo, hi, cv, sd = _check(_doc(CI_TEXT), t, _prop("CL", "3.0", snip), _prop("Lower 95% CI", "4.1", snip),
                               _prop("Upper 95% CI", "2.2", snip), _prop("%CV", "-12.5", "%CV -12.5"),
                               _prop("SD(CL)", "-0.4", "SD(CL) -0.4"))
    assert M.FLAG_CI_INVERTED in _codes(lo) and M.FLAG_CI_INVERTED in _codes(hi)
    assert M.FLAG_SE_NONPOSITIVE in _codes(cv) and M.FLAG_SE_NONPOSITIVE in _codes(sd)


def test_estimate_outside_its_own_ci_is_flagged():
    t = {"fields": [{"name": n, "type": "number"} for n in ("cl", "cl_lower", "cl_upper")]}
    doc = _doc("CL 5.0 L/h (1.0 to 2.0)")
    est, lo, hi = _check(doc, t, _prop("cl", "5.0", "CL 5.0 L/h"), _prop("cl_lower", "1.0", "(1.0 to 2.0)"),
                         _prop("cl_upper", "2.0", "(1.0 to 2.0)"))
    assert M.FLAG_ESTIMATE_OUTSIDE_CI in _codes(est)
    inside = _check(_doc("CL 1.5 L/h (1.0 to 2.0)"), t, _prop("cl", "1.5", "CL 1.5 L/h"),
                    _prop("cl_lower", "1.0", "(1.0 to 2.0)"), _prop("cl_upper", "2.0", "(1.0 to 2.0)"))[0]
    assert M.FLAG_ESTIMATE_OUTSIDE_CI not in _codes(inside)


def test_logk_like_names_are_not_read_as_lower_bounds():
    t = {"fields": [{"name": n, "type": "number"} for n in ("logauc", "gauc")]}
    recs = _check(_doc("logAUC 9.0 and gAUC 1.0"), t, _prop("logauc", "9.0", "logAUC 9.0"),
                  _prop("gauc", "1.0", "gAUC 1.0"))
    assert all(M.FLAG_CI_INVERTED not in _codes(r) for r in recs)
