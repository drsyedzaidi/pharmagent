"""r7 writer 3: the neighbour-word ALLOW-LIST (numgrammar/neighbours.py, glue.py) and its regressions.

The final recheck of r7 reproduced wrong values the character-level rule cannot see (every character involved is allowed):
'1.2 e-3.' cited as 1.2, '5.2 ( x 10^3 )' cited as 5.2, '-(12)' cited as +12. The hand-written shape detectors behind them
(``form_exponent``, ``form_bracket_mult``, the in-word sign skip) are gone: a resolved number keeps its value only when every
word of its window, and the glue of its own word, is an accepted production. The two P2 items of the recheck (a range dash
glued to the end of a neighbouring cell; the NFKC group key) are regressions here too, with the crash the new sweep found.
"""
from __future__ import annotations

import random

import pytest

from app.evidence import models as M
from app.evidence.numgrammar import tokenizer as TK
from app.evidence.numgrammar import tokens
from app.evidence.numgrammar.glue import glue, label, prefix, prose
from app.evidence.qc_flags import group_key, label_key, scripted_flags
from tests import evidence_r6_neighbour_spec as N
from tests.evidence_r6_round1_support import cell, run


def V(text: str) -> list[float | None]:
    return [t.value for t in tokens(text)]


# ------------------------------------------------------------------ the three reproduced P1 findings ---
EXPONENT_PAGES = [("Cmax was 1.2 e-3.", "1.2"), ("Cmax was 1.2 e-3. The next", "1.2"), ("Ki (1.2 E-3)", "1.2"),
                  ("Cmax 12 e3,", "12"), ("Cmax 12 E+3;", "12"), ("Cmax 12 e3/mL", "12"), ("Cmax 12 e-3*", "12"),
                  ("Cmax 1.2 e 3", "1.2"), ("Cmax 1.2 e 3 ng", "1.2"), ("Cmax 1.2\ne-3.", "1.2")]
BRACKET_PAGES = ["Cmax 5.2 ( x 10^3 )", "Cmax 5.2 ( × 10^3 )", "Cmax 5.2 ( x10^3 )", "Cmax 5.2 [ x 10^3 ]",
                 "Cmax 5.2 / x 10^3", "Cmax 5.2 (x 1000)", "Cmax 5.2 (x10^3 cells/mL)", "Cmax 5.2 ( X 10³ )"]
SIGN_PAGES = [("Cmax was -(12) ng", "12"), ("Cmax was −[12] ng", "12"), ("Cmax was -( 12 ) ng", "12"),
              ("Cmax was - ( 12 ) ng", "12"), ("Cmax was -=12 ng", "12"), ("Cmax was -(12.5) ng", "12.5"),
              ("Cmax was +(12) ng", "12"), ("Cmax was − [ 12 ] ng", "12")]


@pytest.mark.parametrize("page, value", EXPONENT_PAGES)
def test_a_spaced_exponent_with_any_trailing_punctuation_never_passes_its_mantissa(page, value):
    got = run(page, value, value)
    assert got.status == "fail", (page, got)
    assert all(t.value is None for t in tokens(page) if page[t.start:t.end].startswith(value)), page


@pytest.mark.parametrize("text", ["1.2 e-3.", "1.2 e-3)", "1.2 E-3,", "12 e3/mL", "1.2 e 3"])
def test_the_same_spaced_exponent_in_a_table_cell_fails(text):
    value = text.split()[0]
    got = cell([["P", "V"], ["x", text]], 1, 1, value)
    assert got.status == "fail", (text, got)


@pytest.mark.parametrize("page", BRACKET_PAGES)
def test_a_scale_in_a_bracket_whose_opener_is_its_own_word_never_passes_the_mantissa(page):
    got = run(page, "5.2", "5.2")
    assert got.status == "fail", (page, got)
    hit = tokens(page)[0]
    assert hit.value is None and 5.2 in hit.parts, (page, hit)
    assert cell([["P", "V"], ["x", page[5:]]], 1, 1, "5.2").status == "fail", page


@pytest.mark.parametrize("page, value", SIGN_PAGES)
def test_a_sign_before_an_opener_run_never_drops_and_verifies_as_a_positive_number(page, value):
    got = run(page, value, value)
    assert got.status == "fail", (page, got)
    assert not any(t.value is not None and abs(t.value) == float(value) for t in tokens(page)), page


@pytest.mark.parametrize("text", ["-(12)", "−(12.5)", "-[12]", "+(12)", "-=12"])
def test_a_sign_before_an_opener_in_a_cell_fails(text):
    value = text.strip("-+=()[]−")
    assert cell([["P", "V"], ["x", text]], 1, 1, value).status == "fail", text


def test_the_refused_word_is_named_for_the_reviewer():
    got = run("Cmax was - ( 12 ) ng", "12", "- ( 12")
    assert got.status == "fail" and "a word beside the number that belongs to no accepted form" in got.detail, got


# ------------------------------------------------------------------ what the allow-list accepts ---
@pytest.mark.parametrize("text, want", [
    ("AUC 12.5 ng·h/mL", [12.5]), ("ka 0.12 h-1", [0.12]), ("BSA 1.73 m²", [1.73]), ("a 10-fold rise", [10.0]),
    ("AUC0-24 120 ng", [120.0]), ("t1/2 4.5 h", [4.5]), ("CL 5.3 ± 0.4 L/h", [5.3, 0.4]), ("12 (25%)", [12.0, 25.0]),
    ("p < 0.05", [0.05]), ("n = 12", [12.0]), ("45 %", [45.0]), ("(16.9 %, 13.1 %)", [16.9, 13.1]),
    ("(E = 0.9)", [0.9]), ("x = 12", [12.0]), ("E0 = 0 and Emax = 8", [None, 8.0]),   # r7 round 3: E0 is no variable
    ("12 - 15 mg", [12.0, 15.0]),
    ("20 mg – 40 mg", [20.0, 40.0]), ("2 x 3 mg", [2.0, 3.0]), ("ratio 1.25 > 1.0", [1.25, 1.0]),
    ("after a 10- fold reduction", [10.0]), ("150- and 300-mg doses", [150.0, 300.0]), ("efflux [46] 12 mg", [None, 12.0]),
    ("30–50% [26,27] 0.66–3.5 [3]", [30.0, 50.0, None, 0.66, 3.5, 3.0]), ("Cmax* 12.5 ng", [12.5]),
    ("(e.g. 12 mg)", [12.0]), ("24.5% , 17.8%", [24.5, 17.8]), ("1.2 x 10^3 cells", [1200.0]), ("0.5-\n1.5 mg", [0.5, 1.5]),
])
def test_a_number_beside_accepted_productions_keeps_its_value(text, want):
    assert V(text) == want


@pytest.mark.parametrize("text, refused", [
    ("12 5th", "12"), ("60mgOD 466 ng", "466"), ("Table 3 - Results", "3"), ("- 90% CI of GMR", "90"),
    ("GMR + 90% CI", "90"), ("12 . 5 ng", "12"), ("E2 12 pg/mL", "12"), ("x 12", "12"), ("12 x3 daily", "12"),
    ("R = 1 + (I/K)", "1"), ("12 <", "12"), ("vitamin E 400 IU", "400"), ("0.693/8 = 0.0866 1/h", "0.0866"),
    ("Cmax 12 e", "12"), ("the 3-", "3"), ("12- -", "12"), ("Cmax 12.5 (*", "12.5"),
])
def test_a_number_beside_a_word_that_is_no_accepted_production_is_unresolved(text, refused):
    hit = next(t for t in tokens(text) if text[t.start:t.end].find(refused) >= 0 or float(refused) in t.parts)
    assert hit.value is None, (text, hit)


@pytest.mark.parametrize("word, ok", [
    ("x(12)", False), ("Cmax-(12)", False), ("12)x(15", False), ("12-x", False), ("12(e)", False), ("3ml,E", False),
    ("12-fold", True), ("(n=12)", True), ("Cmax=12", True), ("p<0.05", True), ("x=12", True), ("12/day", True),
    ("**p<0.05", True), ("±(1.2)", True),
])
def test_the_glue_of_a_number_inside_its_own_word_is_an_accepted_production(word, ok):
    got = [t.value is not None for t in tokens(word)]
    assert (bool(got) and all(got)) is ok, (word, tokens(word))


@pytest.mark.parametrize("text, want", [
    ("Cmax 5.0-", [None]), ("Cmax 45%-", [None]), ("37°C-", [37.0]), ("(Table 1).Threatenedcomplica-", [1.0]),
    ("n=97)metaboliser-", [97.0]), ("5.0- 12.4 mg", [5.0, 12.4]), ("45%- 55%", [45.0, 55.0]),
    ("Hypertension.2011;58(1):63-", [1.0]), ("ActaPaediatr.2006;95(10):1224-", [10.0]),
])
def test_a_word_final_dash_is_a_range_dash_only_right_after_the_number(text, want):
    """A dash that ends the word right after the number or its percent is a range dash and needs its right end; a dash
    after letters is the hyphen of a word the line broke ('complica-'), which the number's own glue already judged."""
    assert V(text) == want


def test_the_character_productions_read_the_owner_examples():
    assert prose("ng·h/mL") and prose("once-daily") and prose("-fold") and prose("e.g.") and prose("%CV")
    assert not prose("e") and not prose("(x") and not prose("-") and not prose("*") and not prose("^") and not prose("<")
    assert label("AUC0-24") and label("CYP2D6*4") and label("h-1") and not label("E-3") and not label("x10^3")
    assert not label("e3,") and not label("(x10^3") and not label("E2") and not label("AUC0-24^")
    assert label("h^-1") and label("(\u03bcL/min/10^6") and not prose("^3") and not label("m^")   # a unit exponent
    assert prefix("(") and prefix("<=") and prefix("(p<") and prefix("x=") and not prefix("-(") and not prefix("-=")
    assert glue(")") and glue("-fold") and glue(",44.83") and not glue(",E") and not glue("(x)")


# -------------------------------------------------------------------------------- the recheck's P2 items ---
def test_a_range_dash_glued_to_the_end_of_the_previous_cell_flags_the_next_cell_and_fails_its_own():
    rows = [["Parameter", "Low", "High"], ["Cmax", "5.0-", "12.4"]]
    right = cell(rows, 1, 2, "12.4")
    assert right.status == "pass" and M.FLAG_CENSORED_VALUE in right.flags, right
    assert cell(rows, 1, 1, "5.0").status == "fail"                       # a dash with no right end: never 5.0 exactly
    assert M.FLAG_CENSORED_VALUE not in cell([["P", "L", "H"], ["Cmax", "5.0", "12.4"]], 1, 2, "12.4").flags


def _flags(g1: str, g2: str) -> list[list[str]]:
    tpl = {"fields": [{"name": "Cmax", "type": "number", "required": True},
                      {"name": "AUC", "type": "number", "required": True}]}

    def rec(f: str, g: str, v: float) -> dict:
        return {"source_id": "S1", "field": f, "group_label": g, "value": str(v), "value_num": v, "unit": None,
                "locator": {}, "verify_status": "pass"}
    return [[f["code"] for f in fl] for fl in scripted_flags([rec("Cmax", g1, 1.0), rec("AUC", g2, 2.0)], tpl, [])]


@pytest.mark.parametrize("g1, g2", [("Arm 1²", "Arm 12"), ("Arm ②", "Arm 2"), ("Arm １２", "Arm 12"),
                                    ("Arm 1​2", "Arm 12"), ("Arm 12­", "Arm 12")])
def test_distinct_group_labels_never_silence_a_required_field(g1, g2):
    assert label_key(g1) != label_key(g2)
    assert _flags(g1, g2) == [[M.FLAG_REQUIRED_MISSING], [M.FLAG_REQUIRED_MISSING]]


def test_the_merging_key_has_no_nfkc_and_still_catches_an_invisible_duplicate():
    assert group_key("Arm 1²") != group_key("Arm 12") and group_key("Arm ②") != group_key("Arm 2")
    assert group_key("Healthy​") == group_key(" healthy ") == "healthy"
    tpl = {"fields": [{"name": "Cmax", "type": "number"}]}
    recs = [{"source_id": "S1", "field": "Cmax", "group_label": g, "value": v, "value_num": float(v), "unit": None,
             "locator": {}, "verify_status": "pass"} for g, v in (("Healthy", "1"), ("Healthy​", "2"))]
    assert all(M.FLAG_DUPLICATE in [f["code"] for f in fl] for fl in scripted_flags(recs, tpl, []))


# ------------------------------------------------------------------------- the crash the sweep found ---
@pytest.mark.parametrize("text", ["3/4 x 10^3 cells", "ratio 12/15 ×10³", "12/15 × 10^3", "ratio 12/15 x10^3"])
def test_a_ratio_is_never_the_mantissa_of_a_power_and_never_crashes_the_reader(text):
    got = tokens(text)                                     # 3ca2b0a raised ValueError('12/15e3') here
    assert got and not any(t.value is not None and t.shape == "sci" for t in got), got


# ----------------------------------------------------------------------- the neighbour-shape sweep ---
def _sample() -> list[str]:
    rnd = random.Random(7)
    longer = [w for w in N.vocabulary(3) if len(w) > 1]
    return sorted(set(N.ALPHABET) | set(N.SAMPLES) | set(rnd.sample(longer, 400)))


def test_the_neighbour_sweep_holds_on_a_sample_of_its_vocabulary():
    errors, strict = N.sweep(_sample())
    assert errors == []
    words = N.vocabulary()                                            # the full vocabulary the exhaustive tier runs
    extra = [w for w in N.SAMPLES if len(w) > 3 or not set(w) <= set(N.ALPHABET)]   # samples beyond the generated words
    assert len(words) == len(set(words)) == 31 + 31 ** 2 + 31 ** 3 + len(extra) and len(N.ALPHABET) == 31
    assert all("+1" in s or "-1" in s or "-2" in s for s in strict)    # a form took T: fail closed, never an error


def test_the_sweep_catches_a_reader_without_the_window_rule(monkeypatch):
    monkeypatch.setattr(TK.Reader, "window_fault", lambda self, fw, lw, dash: None)
    errors, _ = N.sweep(["e-3", "e", "(x", "-", "+", "5th", "E2"])
    assert len(errors) >= 7


def test_the_sweep_catches_a_prose_rule_that_accepts_an_operator_letter(monkeypatch):
    from app.evidence.numgrammar import glue as G
    monkeypatch.setattr(G, "operator_letter", lambda w, i, *bounds: False)        # r7 round 2: (w, i, core bounds)
    errors, _ = N.sweep(["e", "x", "(x", "E,"])
    assert errors
