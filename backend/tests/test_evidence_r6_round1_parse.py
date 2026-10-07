"""r6 round-1 regressions at the parse layer: the thin-space placeholder, the space characters the parse keeps, and the
one list of number characters the glyph-identity checks share with the reader."""
from __future__ import annotations

import pdfplumber
import pytest

from app.evidence import parse as P
from app.evidence import parse_fonts as PF
from app.evidence import parse_guard as G
from app.evidence import parse_ink_match as PIM
from app.evidence import parse_marks as PM
from app.evidence.numgrammar import NUMBER_CHARS, tokens
from app.evidence.numgrammar.alphabet import INSIDE
from app.evidence.numgrammar.equivalents import EQUIV, THIN
from tests import evidence_trickpdf as T


@pytest.fixture(scope="module")
def pdf_path(tmp_path_factory):
    content = T.line("F1", 50, 700, "AUC 1.234 ng.h/mL here") + T.line("F1", 50, 650, "plain line 5 mg")
    path = tmp_path_factory.mktemp("r6round1") / "digits.pdf"
    path.write_bytes(T.build(content.encode()))
    return path


def parse_with(path, monkeypatch, replace: dict[int, str]):
    """parse_pdf of *path* with the text of the chars at the given indexes replaced right before the thin spaces are
    protected: what a PDF whose text layer holds those characters would give the parser."""
    original = G.protect_thin_spaces

    def patched(page):
        for k, ch in replace.items():
            page.chars[k]["text"] = ch
        return original(page)

    monkeypatch.setattr(P, "protect_thin_spaces", patched)
    return P.parse_pdf(path)


@pytest.fixture(scope="module")
def dot_index(pdf_path):
    with pdfplumber.open(str(pdf_path)) as pdf:
        return "".join(c["text"] for c in pdf.pages[0].chars).index("1.234") + 1


def first_line(doc) -> str:
    return doc["pages"][0]["text"].split("\n")[0]


# ------------------------------------------------- a real private-use character is never a thin space ---

def test_a_real_private_use_character_inside_a_number_stays_one_and_the_number_is_unresolved(pdf_path, dot_index, monkeypatch):
    doc = parse_with(pdf_path, monkeypatch, {dot_index: ""})
    line = first_line(doc)
    assert line == "AUC 1234 ng.h/mL here"                         # not U+2009: the old restore() rewrote it
    got = tokens(line, ".")
    assert got[0].value is None and got[0].reason == "unresolved_context" and "U+E000" in got[0].detail


def test_a_real_private_use_character_and_a_thin_space_on_one_page_are_told_apart(pdf_path, dot_index, monkeypatch):
    doc = parse_with(pdf_path, monkeypatch, {dot_index: " ", dot_index + 6: ""})
    assert first_line(doc) == "AUC 1 234 n.h/mL here"


@pytest.mark.parametrize("space", [" ", " ", " "])
def test_exactly_the_approved_spaces_keep_digits_together(pdf_path, dot_index, monkeypatch, space):
    doc = parse_with(pdf_path, monkeypatch, {dot_index: space})
    assert first_line(doc) == "AUC 1 234 ng.h/mL here"
    assert [(t.value, t.shape) for t in tokens(first_line(doc), ".")][0] == (1234.0, "grouped")


@pytest.mark.parametrize("space", [" ", " ", " ", " ", "　"])
def test_every_other_space_character_is_a_word_break_and_never_joins_two_digit_groups(pdf_path, dot_index, monkeypatch, space):
    doc = parse_with(pdf_path, monkeypatch, {dot_index: space})
    got = [t.value for t in tokens(first_line(doc), ".")]
    assert got[:2] == [1.0, 234.0] and 1234.0 not in got


def test_the_thin_space_set_is_built_from_the_equivalents_table():
    assert G.THIN_SPACES == frozenset({THIN} | {k for k, v in EQUIV.items() if v == THIN})
    assert G.THIN_SPACES == {" ", " ", " "}
    for other in (" ", " ", " "):
        assert other not in G.THIN_SPACES


def test_the_layout_sidecar_still_maps_a_page_with_protected_spaces(pdf_path, dot_index, monkeypatch):
    original = G.protect_thin_spaces

    def patched(page):
        page.chars[dot_index]["text"] = " "
        return original(page)

    monkeypatch.setattr(P, "protect_thin_spaces", patched)
    doc = P.parse_pdf_with_layout(pdf_path)
    page = doc["layout"]["pages"][0]
    assert page["status"] == "ok" and any(len(w) == 6 for w in page["words"])


# ------------------------------------------------------------------------ the placeholder itself ---

class FakePage:
    page_number = 1

    def __init__(self, texts):
        self.chars = [{"text": t} for t in texts]


def test_the_placeholder_is_the_first_private_use_character_no_glyph_of_the_page_is():
    thin = G.protect_thin_spaces(FakePage(["1", " ", "2", "", ""]))
    assert thin.mark == ""
    page = FakePage(["1", " ", "2"])
    assert G.protect_thin_spaces(page).mark == "" and page.chars[1]["text"] == ""


def test_a_page_without_thin_spaces_is_untouched_and_restore_is_the_identity():
    page = FakePage(["1", "", "2"])
    thin = G.protect_thin_spaces(page)
    assert thin.mark == "" and thin.restore("ab") == "ab" and [c["text"] for c in page.chars] == ["1", "", "2"]


def test_restore_walks_strings_lists_and_dicts_and_only_touches_its_own_mark():
    thin = G.ThinSpaces("")
    value = {"text": "123 ", "rows": [["ab", ""], ["c"]], "n": 3}
    assert thin.restore(value) == {"text": "1 23 ", "rows": [["a b", ""], ["c"]], "n": 3}


def test_a_page_that_draws_every_private_use_character_is_refused():
    every = [chr(cp) for cp in G._MARKS]
    with pytest.raises(G.ParseLimitError):
        G.protect_thin_spaces(FakePage([" ", *every]))


def test_a_ligature_or_multi_character_glyph_text_counts_as_used():
    thin = G.protect_thin_spaces(FakePage([" ", "ab"]))
    assert thin.mark == ""


# ---------------------------------------- one list of number characters for every glyph-identity check ---

def test_the_glyph_checks_share_the_readers_list_of_number_characters():
    assert PM.CRITICAL is NUMBER_CHARS and PIM.OVERLAP_CRITICAL is NUMBER_CHARS
    assert not hasattr(PF, "CRITICAL") and not hasattr(PF, "_BENIGN") and not hasattr(PF, "_DASHES")
    assert not hasattr(PIM, "_DASHES") and not hasattr(PIM, "_fold")
    assert P.RAISABLE >= frozenset("0123456789+-−⁺⁻") and "–" in P.RAISABLE and "x" not in P.RAISABLE


@pytest.mark.parametrize("ch", sorted(INSIDE))
def test_every_inside_character_is_critical_for_the_font_check(ch):
    assert PF._critical(ch) == {ch} or ch == " "           # a thin space on its own folds to a gap


@pytest.mark.parametrize("src, dst", sorted(EQUIV.items()))
def test_an_equivalent_is_critical_through_its_target(src, dst):
    assert src in NUMBER_CHARS
    assert PF._critical(src) == PF._critical(dst)


@pytest.mark.parametrize("a, b", [("<", ">"), ("%", "‰"), ("±", "+"), ("≤", "<"), ("e", "E"), ("x", "×"),
                                  ("‒", "-"), ("—", "-"), ("\u00ad", "-"), ("²", "2"), ("２", "2")])
def test_a_glyph_whose_text_layer_names_another_number_character_is_in_doubt(a, b):
    """The glyph-identity comparison folds through the equivalents table only: a figure dash, an em dash, a soft
    hyphen, a superscript or a fullwidth digit is never the same character as the ASCII one it looks like."""
    same = PF.fold(a) == PF.fold(b)
    assert same is False


def test_a_digit_drawn_but_reported_as_a_soft_hyphen_is_a_hidden_digit():
    assert PF._hides("\u00ad", {"3"}) is True                       # the old _BENIGN exemption said False
    assert PF._hides("", {"-"}) is True and PF._hides("-", {"-"}) is False


class _Stub:
    """A simple font as ``parse_fonts._simple`` reads it: a ToUnicode map and the encoding's own glyph names."""

    def __init__(self, to_unicode: dict[int, str], encoding: dict[int, str]) -> None:
        self.unicode_map = type("U", (), {"cid2unichr": to_unicode})()
        self.cid2unicode = encoding


@pytest.mark.parametrize("ch", sorted(INSIDE - {"\u2009"}))
def test_a_remapped_glyph_of_every_inside_class_is_a_glyph_text_mismatch(ch):
    """A ToUnicode that reports any character the reader can read for a glyph the encoding names another letter (a font
    remap): the old CRITICAL list missed `<` for `>`, `%`, `\u00b1`, `x` and every superscript."""
    bad, unknown = PF._simple(_Stub({7: ch}, {7: "q"}), {"Encoding": 1})
    assert ch in bad and unknown == set()


@pytest.mark.parametrize("a, b", [("<", ">"), ("%", "\u2030"), ("\u00b1", "+"), ("\u00b2", "2"), ("x", "\u00d7"),
                                  ("-", "\u2212")])
def test_a_remap_between_two_number_characters_is_flagged_unless_the_table_makes_them_one(a, b):
    bad, _ = PF._simple(_Stub({7: a}, {7: b}), {"Encoding": 1})
    assert bool(bad) is (PF.fold(a) != PF.fold(b))
