"""normalize: the index-mapped locator, the decimal-style inference, and the number reader's surface (parse_claim,
numbers_in) and its alphabet classifier (r6: numgrammar)."""
from __future__ import annotations

import pytest

from app.evidence.normalize import (
    dehyphenate_with_map,
    infer_decimal_style,
    normalise,
    numbers_in,
    strip_ws_with_map,
)
from app.evidence.numgrammar import cls, parse_claim, tokens
from app.evidence.parse import parse_pdf
from tests import evidence_fixtures as fx
from tests import evidence_r6_pins as pins

# ------------------------------------------------------------ parse_claim ---


@pytest.mark.parametrize("tok, style, expected", [
    ("3,412", ".", 3412.0),
    ("1 234", ".", 1234.0),
    ("18,2", ",", 18.2),
    ("1,250", ".", 1250.0),
    ("1.250", ",", 1250.0),
    ("−0.45", ".", -0.45),
    ("-0.45", ".", -0.45),
    ("23.7%", ".", 23.7),
    ("1'234", ".", 1234.0),
    ("0,12", ",", 0.12),
    ("12", ",", 12.0),
    ("48.0", ".", 48.0),
    ("1.250.000", ".", 1250000.0),
    ("+7", ".", 7.0),
    ("0,5", ".", 0.5),            # a 1-digit comma group is decimal whatever the style
    ("1,234,567.89", ".", 1234567.89),
    ("1.234.567,89", ",", 1234567.89),
    ("18,200", ",", 18.2),
])
def test_parse_claim(tok, style, expected):
    assert parse_claim(tok, style) == pytest.approx(expected)


@pytest.mark.parametrize("tok", ["", "abc", "n/a", "−", "ng/mL"])
def test_parse_claim_returns_none_without_a_number(tok):
    assert parse_claim(tok, ".") is None


# ------------------------------------------------------------- numbers_in ---

def _values(text: str, style: str = ".") -> list[float]:
    return [v for _, v in numbers_in(text, style)]


def test_numbers_in_ci_triplet():
    assert _values("2.31 (1.95–2.74)") == pytest.approx([2.31, 1.95, 2.74])


def test_numbers_in_range_yields_endpoints_only():
    assert _values("0-24") == [0.0, 24.0]
    assert _values("AUC0-24 was 0-24 h") == [0.0, 24.0]


def test_numbers_in_never_yields_fragment_of_larger_number():
    assert _values("245.6") == pytest.approx([245.6])
    assert 24.0 not in _values("245.6")
    assert _values("3,412 (812)") == [3412.0, 812.0]
    assert _values("1 234 ± 312") == [1234.0, 312.0]
    assert _values("12 1500") == [12.0, 1500.0]      # no space-thousands across two numbers


def test_numbers_in_returns_literals_as_written():
    lits = [lit for lit, _ in numbers_in("Cmax was 3,412 ng/mL and −20 min")]
    assert lits == ["3,412", "−20"]


def test_numbers_in_comma_mode():
    assert _values("18,2 ± 3,4", ",") == pytest.approx([18.2, 3.4])


# ------------------------------------------------------- the alphabet ---

@pytest.mark.parametrize("text", ["١٢", "１２", "10²", "①", "Ⅻ", "¾", "𝟏𝟐", "٣٬٤١٢"])
def test_a_digit_that_is_not_ascii_is_outside_the_alphabet_and_never_a_value(text):
    """Was test_has_non_ascii_digits / test_num_re_is_ascii_only: the digit blocklist is gone; every non-ASCII digit
    is simply not in the allowed alphabet, so a number it touches is unresolved (the superscript digits are the one
    INSIDE exception, readable only after a multiplier)."""
    if text == "10²":
        assert [(t.value, t.shape) for t in tokens(text)] == [(None, "unresolved_shape:bare_superscript_power")]
        return
    assert all(cls(c) == "outside" for c in text)
    assert [t.value for t in tokens("12 " + text)] == [None]


def test_the_classifier_agrees_with_the_second_copy_of_the_sets_on_every_listed_character():
    for cp in (*pins.INSIDE_CPS, *pins.GAP_CPS, *pins.CONTEXT_CPS):                # r7: the LINE list is deleted
        assert cls(chr(cp)) == pins.expected_class(cp)
    assert cls("x") == "inside" and cls("\u2212") == "inside" and cls("\u200b") == "outside"


def test_a_numeric_token_is_ascii_only():
    assert tokens("١٢٣") == [] and [t.value for t in tokens("１２")] == []
    assert [t.value for t in tokens("q 12 y")] == [12.0]
    assert [t.value for t in tokens("x 12 y")] == [None]      # r7 writer 3: a lone x is a multiplier beside the number


# -------------------------------------------------------------- normalise ---

def test_normalise_maps_dashes_spaces_micro_and_case():
    assert normalise("−20")[0] == normalise("-20")[0] == normalise("–20")[0]
    assert normalise("A B")[0] == "a b"
    assert normalise("µL")[0] == normalise("μL")[0]
    assert normalise("Cmax")[0] == "cmax"
    assert normalise("1 234")[0] == "1 234"


def test_normalise_index_map_round_trips_after_dehyphenation_and_ws():
    raw = "clear-\nance  was\n2.31 L/h"
    norm, m = normalise(raw)
    assert norm == "clearance was 2.31 l/h"
    assert len(m) == len(norm)
    i = norm.index("clearance")
    assert raw[m[i]:m[i + len("clearance") - 1] + 1] == "clear-\nance"
    j = norm.index("ance was")
    assert raw[m[j]:m[j + len("ance was") - 1] + 1] == "ance  was"
    k = norm.index("2.31")
    assert raw[m[k]:m[k + 3] + 1] == "2.31"


def test_dehyphenate_with_map():
    text, m = dehyphenate_with_map("clear-\nance")
    assert text == "clearance" and m == [0, 1, 2, 3, 4, 7, 8, 9, 10]
    text, m = dehyphenate_with_map("0-24\n")      # digit-hyphen is a range, not a wrap
    assert text == "0-24\n"
    text, m = dehyphenate_with_map("a - \nb")     # spaced dash is not a wrap
    assert text == "a - \nb"


def test_strip_ws_with_map():
    text, m = strip_ws_with_map("  a  b\n\nc ")
    assert text == "a b c" and m == [2, 3, 5, 6, 8]
    assert strip_ws_with_map("") == ("", [])


def test_normalise_composes_maps_through_a_ligature_expansion():
    raw = "ﬁne 1"
    norm, m = normalise(raw)
    assert norm == "fine 1" and m == [0, 0, 1, 2, 3, 4]


# ---------------------------------------------------- infer_decimal_style ---

def test_infer_decimal_style_on_paper_c(tmp_path):
    doc = parse_pdf(fx.build_pdf(tmp_path, "paper_c"))
    text = "\n".join(p["text"] for p in doc["pages"])
    assert infer_decimal_style(text) == ","
    assert doc["decimal_style"] == ","
    assert infer_decimal_style("Cmax was 3,412 ng/mL and 11.4 h") == "."
