"""r6 migration: every caller reads numbers through numgrammar and nothing else.

verify() (value, snippet, page, cell), the reviewer sentences and reason codes, value_checks (qualifier, scale, style,
group), record_checks (the unit next to the number), extract_rules (keyless extraction), the old fail-opens the grammar
closes, and the deletion of the exception-list code.
"""
from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

from app.evidence import models as M
from app.evidence.extract_rules import leading_number, text_extract, value_number
from app.evidence.extract_tools import check_proposals
from app.evidence.numgrammar import find_n_lead, tokens, unreadable, words_before
from app.evidence.record_checks import source_unit
from app.evidence.value_checks import (
    cell_numbers,
    censored_flags,
    decimal_flags,
    grouped_flags,
    scaled_flags,
)
from app.evidence.verify_messages import context_sentence, named, shape_sentence, unreadable_sentence
from tests.evidence_r5b_support import check, doc

EVIDENCE = Path(__file__).resolve().parents[1] / "app" / "evidence"


def _flags(res: M.VerifyResult, page: str, fn=censored_flags) -> list[dict]:
    p = {"field": "Cmax", "locator": res.locator, "verify_status": "pass", "unit": None}
    return fn(p, doc(page)["pages"][0])


# ------------------------------------------------------- the owner's cases ---

def test_a_private_use_character_on_the_numbers_line_fails_with_a_named_reviewer_sentence():
    res = check("12.5", "Cmax was 12.5 ng/mL", "Cmax was 12.5 ng/mL ")
    assert (res.status, res.reason) == ("fail", M.REASON_UNRESOLVED_CONTEXT)
    assert "'12.5'" in res.detail and "U+E000" in res.detail and "elsewhere on the same line" in res.detail
    assert res.locator["span"] is None and res.locator["literal"] is None and res.locator["token"] is None


def test_a_mixed_number_is_unresolved_and_its_whole_is_never_the_integer():
    page = "n = 12 3/4 patients"
    assert (check("12", "n = 12", page).status, check("12", "n = 12", page).reason) == ("fail", M.REASON_PARTIAL_NUMBER)
    whole = check("12 3/4", "12 3/4", page)
    assert (whole.reason, "a mixed number" in whole.detail) == (M.REASON_UNRESOLVED_SHAPE, True)


def test_a_braille_blank_between_a_qualifier_and_the_number_is_outside_the_alphabet():
    res = check("12", "<⠀12", "AUC <⠀12 mg/L")
    assert (res.status, res.reason) == ("fail", M.REASON_UNRESOLVED_CONTEXT)
    assert "U+2800 BRAILLE PATTERN BLANK" in res.detail and "snippet" in res.detail


@pytest.mark.parametrize("value, page, reason, why", [
    ("1.2 × l0³", "Peak 1.2 × l0³ cells", M.REASON_UNRESOLVED_SHAPE, "a multiplier without a number"),
    ("10 ^ 3", "Peak 10 ^ 3 cells", M.REASON_UNRESOLVED_SHAPE, "an exponent detached from its number"),
    ("2 × daily", "dosed 2 × daily", M.REASON_UNRESOLVED_SHAPE, "a multiplier without a number"),
    ("(cid:3)85", "(cid:3)85% of subjects", M.REASON_UNRESOLVED_CONTEXT, "unmapped glyph marker (cid:3)"),
    ("18", "Patients $18 years old", M.REASON_UNRESOLVED_CONTEXT, "U+0024 DOLLAR SIGN"),
    ("37", "stored at 37¥C overnight", M.REASON_UNRESOLVED_CONTEXT, "U+00A5 YEN SIGN"),
])
def test_the_judges_cases_fail_with_the_right_code_and_reason(value, page, reason, why):
    res = check(value, page, page)
    assert (res.status, res.reason) == ("fail", reason), res
    assert why in res.detail, res.detail


def test_the_en_dash_minus_table_of_a_real_paper_reads_its_negative_values():
    page = "SystolicBP –11.8* –14.2*** –19.9*** –3.1 0.0001"
    assert [t.value for t in tokens(page, ".")] == [-11.8, -14.2, -19.9, -3.1, 0.0001]
    res = check("-11.8", "–11.8*", page)
    assert res.status == "pass" and res.number == -11.8


def test_an_approximately_tilde_is_a_qualifier_and_a_degree_bullet_is_a_degree_sign():
    page = "response in ∼45% of subjects"
    res = check("45", "∼45%", page)
    assert res.status == "pass" and [f["code"] for f in _flags(res, page)] == [M.FLAG_CENSORED_VALUE]
    res = check("37", "stored at 37◦C", "stored at 37◦C overnight")
    assert res.status == "pass" and res.number == 37.0


# --------------------------------------------- the old fail-opens, pinned ---

def test_a_range_starting_with_ten_is_two_numbers_not_ten_to_the_minus_thirty():
    """Old reader: '10-30%' read 10^-30 (a fail-open). Now: 10 and 30, both flagged as range bounds."""
    page = "In urine, 10–30% of the dose"
    assert [t.value for t in tokens(page, ".")] == [10.0, 30.0]
    for value in ("10", "30"):
        res = check(value, "10–30%", page)
        assert res.status == "pass" and _flags(res, page)[0]["code"] == M.FLAG_CENSORED_VALUE


def test_a_percent_range_never_gives_the_upper_bound_a_minus_sign():
    """Old reader: '1%-10%' read 1 and -10."""
    assert [t.value for t in tokens("1%–10% of subjects", ".")] == [1.0, 10.0]
    assert check("-10", "1%–10%", "1%–10% of subjects").status == "fail"


def test_a_comma_list_is_not_numbers_and_an_ambiguous_tail_is_unresolved():
    assert tokens("c(0,1,2,3,4)", ".") == []                         # old reader: 0.1 and 2.3
    assert [(t.value, t.shape) for t in tokens("(21741,229)", ".")] == [(None, "unresolved_shape:ambiguous_tail")]


def test_an_unmapped_glyph_marker_is_never_read_around():
    """Old reader passed '10 (cid:1)M' and '(cid:3)85%' clean (the marker is the text layer's U+FFFD)."""
    for page in ("10 (cid:1)M", "(cid:3)85%", "7.29 ×10(cid:2879)(cid:2871)"):
        assert all(t.value is None for t in tokens(page, ".")), page
    assert unreadable("at 10 (cid:1)M") == (6, "(cid:1)") and unreadable("clean 12.5") is None


def test_a_claim_groups_a_space_only_after_a_head_that_starts_a_number():
    """Found by the end-to-end corpus run: a snippet is read in claim mode (an ASCII space between a 1-3 digit head and
    a 3-digit group is a thousands group). The head must itself START a number: '229.64 233.64' are two decimals and
    'CYP2B6 125.0' a label and a number, not '64 233' and '6 125' groups (which made ordinary table rows unresolvable)."""
    def vals(text: str) -> list[float | None]:
        return [t.value for t in tokens(text, ".", mode="claim")]
    assert vals("229.64 233.64") == [229.64, 233.64] and vals("177.6 464.6") == [177.6, 464.6]
    assert vals("CYP2B6 125.0 6.6 0.89") == [125.0, 6.6, 0.89] and vals("12.5 145") == [12.5, 145.0]
    assert vals("18 900") == [18900.0] and vals("Cmax 18 900 ng") == [18900.0] and vals("(12 145)") == [12145.0]
    assert vals("n=12 145") == [12145.0] and vals("-12 145") == [-12145.0] and vals("12 1500") == [12.0, 1500.0]
    page = "Maximum 232.0 236.0 225.0 197.0"
    res = check("236.0", "Maximum 232.0 236.0 225.0", page)
    assert res.status == "pass" and res.number == 236.0


def test_find_n_lead_gives_the_phrase_and_the_integer_spans():
    text = "Subjects (n = 12) and N=40"
    assert [(text[a:e], text[s:e], n) for a, s, e, n in find_n_lead(text)] == [("n = 12", "12", 12), ("N=40", "40", 40)]


# -------------------------------------------- reason codes and reviewer text ---

def test_the_two_new_codes_join_the_tuple_and_the_legacy_code_is_gone_from_the_code():
    assert M.REASON_UNRESOLVED_CONTEXT in M.REASON_CODES and M.REASON_UNRESOLVED_SHAPE in M.REASON_CODES
    assert not hasattr(M, "REASON_NON_ASCII_DIGITS") and "non_ascii_digits" not in M.REASON_CODES   # round 2: a stored row
    assert len(set(M.REASON_CODES)) == len(M.REASON_CODES) == 15      # keeps the string (test_evidence_r6_round2_hygiene)


def test_the_sentences_are_fixed_templates_naming_the_code_point():
    assert named("U+200B") == "U+200B ZERO WIDTH SPACE" and named("(cid:3)") == "an unmapped glyph marker (cid:3)"
    assert named("U+E000") == "U+E000 (no assigned name)"
    assert context_sentence("12.5", "U+200B", "one word before") == (
        "the cited number '12.5' touches U+200B ZERO WIDTH SPACE (one word before), a character outside the allowed "
        "number alphabet; the text layer cannot be read as a number here. Compare the value with the page image.")
    assert shape_sentence("12 3/4", "mixed_number").endswith("(a mixed number); read it on the page image.")
    assert unreadable_sentence("value", "U+2800").startswith("the value holds U+2800 BRAILLE PATTERN BLANK")


def test_the_verify_fail_flag_carries_the_sentence_and_the_reason():
    template = {"fields": [{"name": "Cmax", "type": "number", "unit": "ng/mL"}]}
    page = {"sha256": "x" * 64, "decimal_style": ".", "pages": [
        {"n": 1, "text": "Cmax was 12.5 ng/mL ​", "small_text": "", "columns": None, "tables": []}]}
    loc = M.Locator(page=1, kind="text", snippet="Cmax was 12.5 ng/mL", decimal_style=".").to_json()
    rec = check_proposals([{"source_id": "S1", "field": "Cmax", "group_label": "", "value": "12.5", "value_num": 12.5,
                            "unit": "ng/mL", "locator": loc, "extractor": "llm"}], page, template)[0]
    assert rec["verify_status"] == "fail" and rec["verify_reason"] == M.REASON_UNRESOLVED_CONTEXT
    msg = next(f for f in rec["flags"] if f["code"] == M.FLAG_VERIFY_FAIL)["message"]
    assert msg.startswith("Cmax: verification failed (unresolved_context): the cited number '12.5' touches U+200B ZERO")
    assert msg.endswith("Compare the value with the page image.")


def test_a_table_cell_with_a_glued_footnote_mark_fails_in_its_own_cell():
    cell = "12.5†"
    res = check("12.5", cell, cell, cell=True)
    assert (res.status, res.reason) == ("fail", M.REASON_UNRESOLVED_CONTEXT)    # the glued dagger taints its word
    assert check("12.5", "12.5", "12.5", cell=True).status == "pass"


# ------------------------------------------------------------ value checks ---

@pytest.mark.parametrize("page, flagged", [
    ("AUC was below 12.4 ng/mL", True), ("BLQ 12.4", True), ("approx. 12.4", True), ("about 12.4", True),
    ("less than 12.4", True), ("up to 12.4", True), ("at least 12.4", True), ("Cmax ≤ 12.4", True),
    ("p<12.4", True), ("12.4-15.0 mg", True), ("from 12.4 to 15.0 mg", True), ("12.4 -15.0", True),
    ("increased to 12.4", False), ("AUC was 12.4", False), ("the LLOQ of 12.4", False), ("AUC 12.4 ng/mL", False),
    ("over 12.4 days of", True),
])
def test_censored_flags_read_the_word_before_and_after(page, flagged):
    res = check("12.4", "12.4", page)
    assert res.status == "pass", page
    assert bool(_flags(res, page)) is flagged, page


def test_an_interval_field_may_cite_one_end_of_a_range_but_never_a_qualified_value():
    page = "range 12.4 to 15.0 mg"
    res = check("12.4", "12.4", page)
    p = {"field": "cl_lower", "locator": res.locator, "verify_status": "pass"}
    assert censored_flags(p, doc(page)["pages"][0]) == []
    page = "CL < 12.4"
    res = check("12.4", "12.4", page)
    assert censored_flags({"field": "cl_lower", "locator": res.locator}, doc(page)["pages"][0])


def test_scaled_header_reads_labels_through_the_grammar():
    def flags(header: str, unit: str | None = None):
        page = {"tables": [{"rows": [["Group", header], ["A", "12"]]}]}
        p = {"field": "AUC", "unit": unit, "locator": {"kind": "table", "cell": {"table": 0, "row": 1, "col": 1}}}
        return scaled_flags(p, page)
    assert flags("AUC ×10^3 (ng·h/mL)") and flags("x1000") and flags("Cmax", "(thousands)")
    assert not flags("AUC0–∞ (ng·h/mL)") and not flags("CLint (µL/min/10^6 cells)")


def test_decimal_style_flags_come_from_the_grammar_shape():
    comma_doc = {"decimal_evidence": (0, 9)}
    page = "Cmax 0,412 ng/mL"
    res = check("0.412", "0,412", page)
    assert res.status == "pass"                                      # foreign decimal on a point page: pass + flag
    p = {"field": "Cmax", "locator": res.locator, "verify_status": "pass"}
    assert [f["code"] for f in decimal_flags(p, doc(page)["pages"][0], comma_doc, None)] == [
        M.FLAG_DECIMAL_STYLE_AMBIGUOUS]
    res = check("12.4", "12·4", "Lancet 12·4 units")
    p = {"field": "Cmax", "locator": res.locator, "verify_status": "pass"}
    assert decimal_flags(p, doc("Lancet 12·4 units")["pages"][0], {"decimal_evidence": (0, 2)}, None) == []
    assert len(decimal_flags(p, doc("Lancet 12·4 units")["pages"][0], {"decimal_evidence": (0, 3)}, None)) == 1


def test_a_bare_digit_group_beside_the_token_is_flagged_composite():
    page = "AUC 12 145 ng/mL"
    res = check("12", "12", page)
    assert res.status == "pass" and [f["code"] for f in _flags(res, page, grouped_flags)] == [M.FLAG_COMPOSITE_CELL]
    page = "AUC 12.4 145 ng/mL"
    assert _flags(check("12.4", "12.4", page), page, grouped_flags) == []


def test_cell_numbers_counts_resolved_numbers_in_the_cell():
    page = {"pages": [{"n": 1, "tables": [{"rows": [["A", "3,412 (812)"]]}], "text": ""}]}
    p = {"locator": {"page": 1, "kind": "table", "token": [0, 5], "cell": {"table": 0, "row": 0, "col": 1}}}
    assert cell_numbers(p, page) == 2


def test_words_before_lists_the_words_in_reading_order():
    assert words_before("AUC was less than 12.4", 18, 2) == ["less", "than"]
    assert words_before("12.4", 0, 2) == [] and words_before("a\n\n b 12", 6, 5) == ["a", "b"]


# ------------------------------------------------------------ record checks ---

@pytest.mark.parametrize("page, unit, decided", [
    ("conc 45% of the dose", "%", True), ("37°C overnight", None, True), ("dose 5 mg intravenously", "mg", True),
    ("dose 5 mg", "mg", True), ("dose 5 mg kg-1 given", None, None), ("dose 5 mg/kg given", "mg/kg", True),
])
def test_the_printed_unit_is_read_from_the_glued_rest_and_the_next_words(page, unit, decided):
    anchor = "45" if "45" in page else ("37" if "37" in page else "5")
    res = check(anchor, anchor, page)
    assert res.status == "pass", page
    got, dec, _ = source_unit({"locator": res.locator}, doc(page)["pages"][0])
    if unit is None and decided is None:
        assert got == "mg" and dec is False                          # 'mg kg-1' may continue: undecided
    elif unit is None:
        assert got is None
    else:
        assert (got, dec) == (unit, decided)


# ----------------------------------------------------------- extract rules ---

def test_keyless_text_rules_read_n_and_a_dose_through_the_grammar():
    d = {"decimal_style": ".", "pages": [{"n": 1, "text": "Subjects (n = 12) received a dose of 150 mg daily.\n"
                                          "Others took 150mg once.", "columns": [], "tables": []}]}
    template = {"fields": [{"name": "n", "type": "integer"}, {"name": "dose", "type": "number", "unit": "mg"}]}
    out = {p["field"]: p for p in text_extract(d, template, "S1")}
    assert out["n"]["value"] == "12" and out["n"]["value_num"] == 12.0
    assert out["dose"]["value"] == "150" and out["dose"]["unit"] == "mg" and out["dose"]["value_num"] == 150.0
    glued = {"decimal_style": ".", "pages": [{"n": 1, "text": "Others took 150mg dose once.", "columns": [],
                                              "tables": []}]}
    assert text_extract(glued, template, "S1") == []                 # a glued unit is unresolved: no candidate


def test_leading_number_proposes_what_verify_will_read_and_never_a_later_number():
    assert leading_number("12 145", ".")[:2] == ("12 145", 12145.0)  # the claim reading (verify then fails it)
    assert leading_number("3,412 (812)", ".")[:2] == ("3,412", 3412.0)
    assert leading_number("4.5 x 10 (1.2)", ".") is None and leading_number("n.d.", ".") is None
    assert value_number("12.4 ng/mL", ".") == 12.4 and value_number("18,2", ",") == 18.2


# ------------------------------------------------------------- the deletion ---

@pytest.mark.parametrize("name", ["invisible", "number_glue", "sci_tail", "visible_window"])
def test_the_exception_list_modules_are_gone(name):
    assert not (EVIDENCE / f"{name}.py").exists()
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(f"app.evidence.{name}")


def test_the_old_reader_names_are_gone_from_the_normalizer():
    N = importlib.import_module("app.evidence.normalize")
    for gone in ("number_tokens", "parse_number", "NumToken", "fold_chars", "has_non_ascii_digits", "NUM_RE", "MIDDOTS",
                 "digits_only"):
        assert not hasattr(N, gone), gone
    assert hasattr(N, "number_spans") and hasattr(N, "numbers_in") and hasattr(N, "normalise")
    V = importlib.import_module("app.evidence.verify")
    assert not hasattr(V, "bare_ten_unresolved") and not hasattr(V, "has_non_ascii_digits")


READER = ("verify.py", "value_checks.py", "record_checks.py", "extract_rules.py", "normalize.py", "unit_norm.py",
          "parse_ink_match.py",
          "numgrammar/alphabet.py", "numgrammar/api.py", "numgrammar/body.py", "numgrammar/equivalents.py",
          "numgrammar/scan.py", "numgrammar/taint.py", "numgrammar/tok.py", "numgrammar/tokenizer.py",
          "numgrammar/words.py")
BANNED_ATTRS = {"category", "name", "numeric", "decimal", "digit", "east_asian_width", "bidirectional"}
BANNED_STR_METHODS = {"isdigit", "isalpha", "isnumeric", "isdecimal", "isalnum"}


def _calls(path: Path) -> list[tuple[str, int]]:
    tree = ast.parse(path.read_text("utf-8"))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == "unicodedata" and node.attr in BANNED_ATTRS:
                out.append((f"unicodedata.{node.attr}", node.lineno))
            if node.attr in BANNED_STR_METHODS:
                out.append((f"str.{node.attr}", node.lineno))
    return out


def test_no_category_or_unicode_property_test_is_left_in_the_reader():
    """The only list in the reader is the alphabet, the context sets, the gap set and the equivalents table: no
    unicodedata.category / name / numeric / decimal, no str.isdigit / isalpha / isnumeric in any reader module. Round 2:
    the locator (normalize.py) has no exception any more (it was NFKC and the matching-only str.isalpha / isspace),
    and the unit comparison, the header key and the ink pairing are listed with it."""
    bad = {f: _calls(EVIDENCE / f) for f in READER}
    for f, hits in bad.items():
        assert hits == [], f


def test_the_reader_keeps_no_codepoint_blocklist_by_name():
    text = "\n".join((EVIDENCE / f).read_text("utf-8") for f in READER)
    for token in ("Default_Ignorable", "NONCHAR", "PRIVATE_USE", "SOFT =", "HARD =", "strip_invisible", "is_invisible",
                  "UNIT_FOLD", "CONTINUATION_CHARS", "QUALIFIER_CHARS", "_SCALE_RE", "_QUALIFIER_RE", "_N_RE =",
                  "_DOSE_RE ="):
        assert token not in text, token


def test_the_digit_helpers_of_the_callers_are_the_alphabets_not_private_copies():
    import app.evidence.value_checks as VC
    import app.evidence.verify as V
    for module, names in ((VC, ("_DIGITS", "_DIGITS_SET", "_bare_group")), (V, ("_NON_DIGITS", "digits_only", "_close"))):
        for name in names:
            assert not hasattr(module, name), (module.__name__, name)
    assert hasattr(V, "digits_of") and hasattr(VC, "digit_group")


def test_the_message_module_uses_unicodedata_only_to_name_a_code_point():
    assert {a for a, _ in _calls(EVIDENCE / "verify_messages.py")} == {"unicodedata.name"}
