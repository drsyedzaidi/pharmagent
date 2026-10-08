"""r7 round 3: the PRODUCT production replaces the ten carve-out; a power word belongs to its scientific token; a power
of ten never stands as a value beside another number of its own word (findings of the round-3 audit).

product = count MULT amount, spaced (a lone multiplier word) or glued (x X * ×; a middle dot glued between digits is the
separator production): both operands plain numbers (int or decimal: never grouped, scientific, a power or a ratio); the
amount is no raised ten ('10', '103': digits '10' then a digit 1-9 or nothing), and when it is a power of ten written
out ('100', '1000', '10000') it stands after an integer count of one or two digits and before a unit production
('2 x 100 mg', '3 x 1000 mg'). Every other `number MULT number` is one unresolved expression.
"""
from __future__ import annotations

import pytest

from app.evidence import models as M
from app.evidence.numgrammar import tokens
from tests.evidence_r6_round1_support import cell, run


def values(text: str) -> list[float | None]:
    return [t.value for t in tokens(text)]


# --------------------------------------------------------------------------------- the audit's PoCs ---
@pytest.mark.parametrize("page, value", [
    ("viral load 1.2 × 10000 copies/mL", "1.2"), ("count 2.5 × 1e6 cells", "2.5"),
    ("count 1.2 × 1,000,000 cells", "1.2"), ("count 3.1 · 1000 cells", "3.1"), ("count 1.2 × 10 copies", "1.2"),
    ("viral load 1.2×10000 copies/mL", "1.2"), ("count 2.5×1e6 cells", "2.5"), ("count 2 × 10000 cells", "2"),
    ("count 2 x 1e6 cells", "2"), ("count 2 x 10^6x cells", "2"), ("count 5 × 100000", "5")])
def test_a_scale_after_a_multiplier_never_leaves_the_mantissa_as_the_value(page, value):
    out = run(page, page, value, field="dose")
    assert out.status == "fail" and out.reason in (M.REASON_PARTIAL_NUMBER, M.REASON_UNRESOLVED_SHAPE), (page, out)
    hit = [t for t in tokens(page) if t.start <= page.index(value) < t.end]
    assert hit and all(t.value is None for t in hit), (page, hit)


@pytest.mark.parametrize("text, want", [
    ("2 x 100 mg", [2.0, 100.0]), ("3 × 250", [3.0, 250.0]), ("3 x 1000 mg", [3.0, 1000.0]),
    ("4.6×150 mm", [4.6, 150.0]), ("2 × 1003", [2.0, 1003.0]), ("2×5 and 3×250", [2.0, 5.0, 3.0, 250.0]),
    ("2 x 3 mg", [2.0, 3.0]), ("2 x -3", [2.0, -3.0]), ("took 2 x 100 mg", [2.0, 100.0]), ("2 x 100mg", [2.0, 100.0]),
    ("2x100mg", [2.0, 100.0]), ("1.06 × 12 h", [1.06, 12.0]), ("0.0015 × 8", [0.0015, 8.0]),
    ("150 × 4.6 mm", [150.0, 4.6]), ("12* 15", [12.0, 15.0])])
def test_a_product_of_plain_numbers_reads_both_operands(text, want):
    assert values(text) == want, (text, tokens(text))


@pytest.mark.parametrize("text, why", [
    ("2 x 100", "ten_after_multiplier"), ("3 x 1000", "ten_after_multiplier"), ("2.0 × 100 mm", "ten_after_multiplier"),
    ("1.2 × 10000 copies", "ten_after_multiplier"), ("2.5 × 1e6", "multiplier_operand"),
    ("1.2 × 1,000,000", "ten_after_multiplier"), ("2×100", "ten_after_multiplier"), ("0·100", "ten_after_multiplier"),
    ("12·105", "ten_after_multiplier"), ("1.2×103", "ten_after_multiplier"), ("12* 10", "ten_after_multiplier"),
    ("2 × 103", "ten_after_multiplier"), ("1,234 × 5", "multiplier_operand")])
def test_every_other_number_mult_number_is_one_unresolved_expression_named_for_the_reviewer(text, why):
    toks = tokens(text)
    assert toks and all(t.value is None for t in toks), (text, toks)
    assert any(t.why == why for t in toks), (text, [(t.shape, text[t.start:t.end]) for t in toks])


def test_the_power_of_ten_amount_needs_the_unit_right_after_it():
    assert values("2 x 100 mg") == [2.0, 100.0] and values("2 x 100 (mg)") == [None]
    assert values("2 x 100 tablets") == [None] and values("(2 x 100) mg") == [2.0, 100.0]


def test_a_table_cell_product_follows_the_same_production():
    rows = [["Arm", "Count"], ["A", "1.2 × 10000"], ["B", "2 x 100 mg"]]
    assert cell(rows, 1, 1, "1.2", snippet="1.2 × 10000").status == "fail"
    assert cell(rows, 2, 1, "100", snippet="2 x 100 mg").status == "pass"


# ------------------------------------------------------------------- a power word belongs to its token ---
@pytest.mark.parametrize("page", ["WBC 4.5 x 10^9/L 3.2 x 10^9/L", "WBC 4.5 x 10^9/L; 3.2 x 10^9/L",
                                  "WBC 4.5 × 10^9/L 3.2 × 10^9/L", "WBC 4.5x10^9/L 3.2 x 10^9/L"])
def test_the_second_of_two_spaced_scientific_values_reads(page):
    toks = tokens(page)
    assert [t.value for t in toks] == [4.5e9, 3.2e9] and {t.shape for t in toks} == {"sci"}, toks
    snip = page[page.rindex("3.2"):]
    assert run(page, snip, "3.2e9", field="dose").status == "pass"
    assert run(page, snip, "3.2", field="dose").reason == M.REASON_PARTIAL_NUMBER


def test_an_unconsumed_bare_power_beside_a_number_is_still_a_detached_exponent():
    assert [t.why for t in tokens("count 1.2 10^3 cells")] == ["detached_exponent"]
    assert [t.why for t in tokens("count 1.2\n10^9 cells")] == ["detached_exponent"]


# --------------------------------------------------------------- a power of ten glued after a number ---
@pytest.mark.parametrize("page, value", [
    ("count 1.2(10^3) cells", "1.2"), ("count 1.2(10^3) cells", "10^3"), ("count 1.2(10^3) cells", "1000"),
    ("count 1.2[10^3] cells", "1.2"), ("count (1.2)(10^3) cells", "1.2"), ("WBC 12(10^9/L)", "12"),
    ("conc 5(10^-6)M", "5"), ("count 1.2(1e3) cells", "1.2"), ("count 1.2(10³) cells", "1.2")])
def test_a_power_of_ten_glued_after_a_number_is_a_detached_scale(page, value):
    snip = page.split(" ", 1)[1]
    out = run(page, snip, value, field="dose")
    assert out.status == "fail", (page, value, out)
    hit = [t for t in tokens(page) if t.value is None and t.why in ("detached_exponent", "bare_superscript_power")]
    assert hit, (page, tokens(page))


def test_a_glued_power_keeps_both_parts_and_a_cell_fails_too():
    tok = tokens("count 1.2(10^3) cells")
    assert len(tok) == 1 and tok[0].why == "detached_exponent" and set(tok[0].parts) >= {1.2, 1000.0}, tok
    rows = [["Parameter", "Count (cells/L)"], ["WBC", "1.2(10^9)"]]
    assert cell(rows, 1, 1, "1.2", snippet="1.2(10^9)").status == "fail"


@pytest.mark.parametrize("text", ["10^3-10^6 cells", "(10^3)(5)", "10^3,5"])
def test_a_power_of_ten_is_never_a_value_beside_another_number_of_its_own_word(text):
    assert all(t.value is None for t in tokens(text)), (text, tokens(text))


def test_a_power_of_ten_standing_alone_still_reads():
    assert values("count (10^3) cells") == [1000.0] and values("10^6/L") == [1e6] and values("10^-3,") == [1e-3]
