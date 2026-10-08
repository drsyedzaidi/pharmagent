"""r7 owner decision 2: a unit glued to a number is read through the UNIT PRODUCTION (numgrammar/units.py).

    unit     = atom {("/" | "per") atom}         directly after the number
    atom     = ([prefix] base | prefix prefixed_base) [exponent]
    exponent = "2" | "3" | "^" [sign] digit{1,2} | [superscript sign] superscript digit{1,2}

The terminals (prefix, base and prefixed-base sets) are grammar data of the CONTEXT allow-list, each admitted by its corpus
count (alphabet.py). The production must end at a boundary (word end, a trailing closer not before a digit, an opener).
'1OmL', '10m L' in odd spacing and a glued word that is not a full production stay unresolved. The unit is context: the
token's literal is the number alone and its value never changes.
"""
from __future__ import annotations

import time

import pytest

from app.evidence import models as M
from app.evidence.numgrammar import alphabet as A
from app.evidence.numgrammar import tokens
from app.evidence.numgrammar.units import splits, unit_end
from app.evidence.record_checks import source_unit, unit_flags
from tests import evidence_r6_pins as P
from tests.evidence_r5b_support import check, doc


def R(text: str, mode: str = "page") -> list[tuple[str, float | None, str]]:
    return [(text[t.start:t.end], t.value, t.shape) for t in tokens(text, ".", mode=mode)]


@pytest.mark.parametrize("text, literal, value", [
    ("150mg", "150", 150.0), ("24h", "24", 24.0), ("10mL", "10", 10.0), ("1.73m2", "1.73", 1.73),
    ("100µg", "100", 100.0), ("(20mg)", "20", 20.0), ("2.5mg/kg/h", "2.5", 2.5), ("0.6ng/ml,", "0.6", 0.6),
    ("5h^-1", "5", 5.0), ("5h⁻¹", "5", 5.0), ("1.73m²", "1.73", 1.73), ("140mmHg", "140", 140.0),
    ("40Hz),", "40", 40.0), ("4000IU).", "4000", 4000.0), ("2000g", "2000", 2000.0), ("26w", "26", 26.0),
    ("0.66y", "0.66", 0.66), ("59.6nM", "59.6", 59.6), ("8.5cm", "8.5", 8.5), ("0.18mm,", "0.18", 0.18),
    ("3ml", "3", 3.0), ("10μl", "10", 10.0), ("5mgperkg", "5", 5.0), ("12mg/m2", "12", 12.0),
    ("15min.", "15", 15.0), ("700mL/min,", "700", 700.0), ("20mg(immediate", "20", 20.0), ("2h.Then", "2", 2.0),
])
def test_a_number_with_a_glued_unit_production_reads_the_number_alone(text, literal, value):
    assert R(text) == [(literal, value, "int" if "." not in literal else "decimal")]


def test_a_glued_unit_reads_after_every_token_shape_and_inside_a_label_after_a_starter():
    assert R("1.2e-3mg") == [("1.2e-3", 0.0012, "sci")]
    assert R("BMI≥35kg/m") == [("35", 35.0, "int")] and R("Cmax=12.5ng/mL") == [("12.5", 12.5, "decimal")]
    assert R("12(5M)") == [("12", 12.0, "int"), ("5", 5.0, "int")]
    assert R("2×5mg") == [("2", 2.0, "int"), ("5", 5.0, "int")]
    assert [t.value for t in tokens("(25–100ng/mL)")] == [25.0, 100.0]
    assert [t.ranged for t in tokens("(25–100ng/mL)")] == [True, True]


@pytest.mark.parametrize("text", [
    "1OmL", "1Og", "3MIU", "150mgtid", "2nd", "3rd", "5th", "1st", "1970s", "10s", "2d", "23l", "5mg/l", "24hours",
    "0.5μ", "5h-1", "9.9mg*", "5mg,10mg", "5mg/", "5mg/tablet", "1.73m23", "12mgx", "75–150mg/day",
    "200-2400mg/d", "12/15mg", "140/90mmHg", "10mg†", "2.5kgs", "12e", "12mgL",
])
def test_a_glued_word_that_is_not_a_full_unit_production_stays_unresolved(text):
    toks = tokens(text)
    assert toks and all(t.value is None for t in toks), (text, R(text))
    assert {t.why for t in toks} <= {"glued_letters", "glued_symbol"}, (text, R(text))


@pytest.mark.parametrize("text, expr, parts", [
    ("took 10m L daily", "10m L", (10.0,)), ("C =2 81ng/m L,AU C", "81ng/m L,AU", (81.0,)),
    ("15m in at72 rpm", "15m in", (15.0,)), ("10mg /kg", "10mg /kg", (10.0,)), ("10m\nL", "10m\nL", (10.0,)),
    ("0.5m g of", "0.5m g", (0.5,)), ("5m Hz", "5m Hz", (5.0,)), ("5-10m L", "5-10m L", (5.0, 10.0)),
])
def test_a_unit_that_the_next_word_continues_is_a_split_unit(text, expr, parts):
    hit = [t for t in tokens(text) if t.value is None]
    assert [(text[t.start:t.end], t.shape, t.parts) for t in hit] == [(expr, "unresolved_shape:split_unit", parts)]
    lo = text.index(expr)
    assert all(t.value is None for t in tokens(text) if t.start < lo + len(expr) and t.end > lo)


@pytest.mark.parametrize("text, want", [
    ("10mg kg", [10.0]), ("24h )", [24.0]), ("10m mL", [10.0]), ("20mg daily", [20.0]), ("10mg, L", [10.0]),
    ("2h min", [2.0]), ("10 mL", [10.0]), ("10m (n=3)", [10.0, 3.0]),
])
def test_a_next_word_that_does_not_continue_the_unit_leaves_it_read(text, want):
    assert [t.value for t in tokens(text)] == want


def test_unit_end_and_splits_are_the_production_itself():
    assert unit_end("150mg", 3) == 5 and unit_end("150mgx", 3) == -1 and unit_end("12(5M)", 4) == 5
    assert unit_end("1.73m2", 4) == 6 and unit_end("1.73m23", 4) == -1 and unit_end("5h^-1", 1) == 5
    assert splits("m", "L") and splits("m", "in") and splits("mg", "/kg") and splits("ng/m", "L,AU")
    assert not splits("mg", "kg") and not splits("h", ")") and not splits("m", "mL") and not splits("g", "daily")


def test_the_unit_terminals_are_pinned_and_made_of_context_letters():
    assert A.UNIT_PREFIXES == P.UNIT_PREFIX_PIN and A.UNIT_BASES == P.UNIT_BASE_PIN
    assert A.UNIT_PREFIXED_BASES == P.UNIT_PREFIXED_BASE_PIN
    assert (len(A.UNIT_PREFIXES), len(A.UNIT_BASES), len(A.UNIT_PREFIXED_BASES)) == (5, 11, 1)
    for term in A.UNIT_PREFIXES | A.UNIT_BASES | A.UNIT_PREFIXED_BASES:
        assert all(c in A.CONTEXT_LETTERS or c in A.GREEK for c in term), term
    assert not (A.UNIT_BASES & A.UNIT_PREFIXED_BASES) and "d" not in A.UNIT_BASES and "s" not in A.UNIT_BASES


def test_the_production_is_linear_in_a_long_glued_word():
    t0 = time.perf_counter()
    bad = tokens("1" + "mg/" * 40_000 + "x")
    good = tokens("1" + "mg/" * 40_000 + "kg")
    many = tokens(" ".join(["10m"] * 20_000 + ["L"]))
    assert time.perf_counter() - t0 < 5.0
    assert [t.value for t in bad] == [None] and [t.value for t in good] == [1.0]
    assert [t.value for t in many[:-1]] == [10.0] * (len(many) - 1) and many[-1].why == "split_unit"


# ------------------------------------------------------------------ the verify() path ---
def test_a_record_citing_the_number_of_a_glued_unit_passes_with_the_number_alone():
    res = check("150", "took 150mg daily", "Patients took 150mg daily for 7 days.")
    assert (res.status, res.number, res.literal) == ("pass", 150.0, "150")
    cell = check("20", "(20mg)", "(20mg)", cell=True)
    assert (cell.status, cell.number) == ("pass", 20.0)


def test_a_record_citing_a_number_of_a_broken_unit_fails():
    for value, snippet, page in (("10", "10m L", "took 10m L daily"), ("1", "1OmL", "took 1OmL daily"),
                                 ("23", "23l", "volume 23l total")):
        res = check(value, snippet, page)
        assert res.status == "fail" and res.reason in (M.REASON_PARTIAL_NUMBER, M.REASON_UNRESOLVED_SHAPE), value


def test_the_printed_unit_of_a_glued_number_is_the_glued_unit():
    page = "Patients took 150mg daily for 7 days."
    res = check("150", "took 150mg daily", page)
    unit, decided, printed = source_unit({"locator": res.locator}, doc(page)["pages"][0])
    assert (unit, decided, printed) == ("mg", True, "mg")
    p = {"field": "dose", "unit": "mg", "value_num": 150.0, "verify_status": "pass", "locator": res.locator}
    assert unit_flags(p, doc(page)["pages"][0]) == []
    assert unit_flags({**p, "unit": "mg/kg"}, doc(page)["pages"][0])[0]["code"] == M.FLAG_UNIT_SOURCE_MISMATCH
