"""normalize: index-mapped normaliser, ASCII-only NUM_RE, parse_number, numbers_in, non-ASCII digit guard."""
from __future__ import annotations

import pytest

from app.evidence.normalize import (
    NUM_RE,
    dehyphenate_with_map,
    has_non_ascii_digits,
    infer_decimal_style,
    normalise,
    numbers_in,
    parse_number,
    strip_ws_with_map,
)
from app.evidence.parse import parse_pdf
from tests import evidence_fixtures as fx

# ----------------------------------------------------------- parse_number ---


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
def test_parse_number(tok, style, expected):
    assert parse_number(tok, style) == pytest.approx(expected)


@pytest.mark.parametrize("tok", ["", "abc", "n/a", "−", "ng/mL"])
def test_parse_number_returns_none_without_a_number(tok):
    assert parse_number(tok, ".") is None


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


# ------------------------------------------------------ non-ASCII digits ---

@pytest.mark.parametrize("text, expected", [
    ("Cmax 3,412", False),
    ("١٢", True),            # Arabic-Indic
    ("１２", True),                     # fullwidth
    ("10²", True),                      # superscript (would NFKC-fold into '102')
    ("", False),
])
def test_has_non_ascii_digits(text, expected):
    assert has_non_ascii_digits(text) is expected


def test_num_re_is_ascii_only():
    assert NUM_RE.search("١٢٣") is None
    assert NUM_RE.search("１２") is None
    assert NUM_RE.search("x 12 y").group(0) == "12"


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


def test_normalise_composes_maps_through_nfkc_expansion():
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
