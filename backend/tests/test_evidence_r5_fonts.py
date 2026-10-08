"""Glyph identity (parse_fonts): the comparison of /ToUnicode text with the glyph a font draws.

Each case is a real PDF with an embedded font program built by tests/evidence_fontpdf.py, judged by a
fresh GlyphChecker over the pdfplumber page, so what is asserted is the verdict a reader's page gets:
no entry when the text is the glyph, ``glyph_text_mismatch`` when a number glyph is reported as another
character (or hidden as a space), ``glyph_unverified`` when the font gives no identity to compare.
"""
from __future__ import annotations

import io
from types import SimpleNamespace

import pdfplumber
import pytest
from pdfminer.pdftypes import PDFStream

from app.evidence import parse_fonts as PF
from app.evidence.numgrammar.alphabet import NUMBER_CHARS
from tests import evidence_fontpdf as F
from tests import evidence_trickpdf as T

DIGITS = "0123456789"
NUMBER_CHARS_TEXT = "".join(sorted(NUMBER_CHARS))
HELV_TOUNI = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding /ToUnicode 7 0 R >>"


def _verdict(data: bytes) -> dict[str, str]:
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        return PF.GlyphChecker().page(pdf.pages[0])


def _tou(overrides: dict[int, str] | None = None) -> dict[int, bytes]:
    return {7: T.stream(F.cmap(F.digit_cmap(overrides)))}


def _cid(*, program: bytes | None = None, mapping: dict[int, str] | None = None, hex_codes: str = "0003 0004 0002",
         subtype: str = "CIDFontType2", gid_map: bytes | None = None) -> dict[str, str]:
    prog = F.ttf_program() if program is None else program
    objs = F.cid_objects(prog, mapping=mapping, subtype=subtype, gid_map=gid_map)
    return _verdict(F.font_pdf(F.type0_font(), objs, hex_codes=hex_codes))


# ------------------------------------------------- Type0 / embedded TrueType ---

def test_cid_truetype_whose_tounicode_names_the_glyph_is_clean():
    assert _cid() == {}


def test_cid_tounicode_naming_another_digit_is_a_mismatch():
    assert PF.MISMATCH in _cid(mapping={4: "4"})                       # glyph 'three' reported as '4'
    assert "4" in _cid(mapping={4: "4"})[PF.MISMATCH]


def test_cid_tounicode_hiding_a_digit_as_space_puts_every_digit_in_doubt():
    assert _cid(mapping={4: " "}) == {PF.MISMATCH: DIGITS}


def test_cid_with_an_unreadable_font_program_is_unverified_not_clean():
    assert _cid(program=b"not a font at all") == {PF.UNVERIFIED: DIGITS}


def test_cid_without_an_embedded_program_is_unverified():
    objs = F.cid_objects(None)
    verdict = _verdict(F.font_pdf(F.type0_font(), objs, hex_codes="0003 0004 0002"))
    assert verdict == {PF.UNVERIFIED: DIGITS}


def test_cid_cff_descendant_has_no_glyph_identity_so_it_is_unverified():
    assert _cid(subtype="CIDFontType0", program=F.cff_program(None)) == {PF.UNVERIFIED: DIGITS}


def _gids(*pairs: tuple[int, int], size: int = 11) -> bytes:
    """A CIDToGIDMap stream: identity for CIDs 0..size-1 except the (cid, gid) *pairs*."""
    table = {cid: cid for cid in range(size)} | dict(pairs)
    return b"".join(table[cid].to_bytes(2, "big") for cid in range(size))


def test_cid_to_gid_stream_remaps_the_glyph_that_is_compared():
    """CID 1 draws glyph 4 ('three'): ToUnicode '3' is right for it, '1' is not."""
    gid_map = _gids((1, 4), (4, 1))
    assert _cid(gid_map=gid_map, hex_codes="0001", mapping={1: "3", 4: "0"}) == {}
    assert _cid(gid_map=gid_map, hex_codes="0001", mapping={1: "1", 4: "0"}) == {PF.MISMATCH: "1"}


def test_cid_past_the_end_of_the_gid_map_has_no_identity():
    verdict = _cid(gid_map=_gids(), hex_codes="0001", mapping={50: "5"})   # CID 50 is not in the 11-entry map
    assert verdict == {PF.UNVERIFIED: "5"}


def test_cid_mapped_to_a_glyph_the_font_does_not_have_has_no_identity():
    verdict = _cid(gid_map=_gids((1, 999)), hex_codes="0001", mapping={1: "7"})
    assert verdict == {PF.UNVERIFIED: "7"}


def test_cid_glyph_found_by_name_when_the_cmap_does_not_list_it():
    """The digit glyphs are not in the cmap: their Adobe glyph names ('seven') still give the character."""
    prog = F.ttf_program(cmap_digits=False)
    assert _cid(program=prog, hex_codes="0008") == {}
    assert _cid(program=prog, mapping={8: "6"}, hex_codes="0008") == {PF.MISMATCH: "6"}


def test_cid_glyph_with_an_unparsable_name_gives_no_identity():
    prog = F.ttf_program(extra_glyphs=("q7zz",), cmap_digits=False)
    verdict = _cid(program=prog, mapping={11: "5"}, hex_codes="0001")      # glyph id 11 is 'q7zz'
    assert verdict == {PF.UNVERIFIED: "5"}


# ------------------------------------------------------------ simple fonts ---

def _simple(font: bytes, extra: dict[int, bytes], overrides: dict[int, str] | None = None) -> dict[str, str]:
    return _verdict(F.font_pdf(font, _tou(overrides) | extra))


def _type1_font(descriptor_entry: bytes) -> tuple[bytes, dict[int, bytes]]:
    font = (b"<< /Type /Font /Subtype /Type1 /BaseFont /TType1 /FontDescriptor 8 0 R /FirstChar 48 /LastChar 57 "
            b"/Widths [500 500 500 500 500 500 500 500 500 500] /ToUnicode 7 0 R >>")
    desc = (b"<< /Type /FontDescriptor /FontName /TType1 /Flags 4 /FontBBox [0 0 500 700] /ItalicAngle 0 /Ascent 800 "
            b"/Descent -200 /CapHeight 700 /StemV 80 " + descriptor_entry + b" >>")
    return font, {8: desc}


def _digit_names() -> dict[int, str]:
    return {0x30 + d: F.DIGIT_NAMES[d] for d in range(10)}


def test_embedded_type1_uses_its_builtin_encoding_for_the_comparison():
    header = F.type1_header(_digit_names())
    font, extra = _type1_font(b"/FontFile 9 0 R")
    extra[9] = T.stream(header, f" /Length1 {len(header)}")
    assert _simple(font, extra) == {}
    assert _simple(font, extra, {0x33: "4"}) == {PF.MISMATCH: "4"}


def test_embedded_cff_uses_its_own_encoding_array_and_skips_names_it_cannot_convert():
    enc = _digit_names() | {0x41: "xq9"}                                   # 'xq9' is no Adobe glyph name
    font, extra = _type1_font(b"/FontFile3 9 0 R")
    extra[9] = T.stream(F.cff_program(enc, extra_glyphs=("xq9",)), " /Subtype /Type1C")
    assert _simple(font, extra) == {}
    assert _simple(font, extra, {0x33: "4"}) == {PF.MISMATCH: "4"}


@pytest.mark.parametrize("program,subtype", [
    (F.cff_program(None), "Type1C"),                                        # StandardEncoding is a name, not a table
    (b"junk bytes, no CFF here", "Type1C"),                                 # unreadable program
    (F.cff_program(_digit_names()), "OpenType"),                            # not a bare CFF stream
])
def test_cff_without_a_readable_encoding_leaves_the_digits_unverified(program, subtype):
    font, extra = _type1_font(b"/FontFile3 9 0 R")
    extra[9] = T.stream(program, f" /Subtype /{subtype}")
    assert _simple(font, extra) == {PF.UNVERIFIED: DIGITS}


def test_truetype_without_encoding_names_no_glyphs_so_it_is_unverified():
    font = b"<< /Type /Font /Subtype /TrueType /BaseFont /Arial /ToUnicode 7 0 R >>"
    assert _simple(font, {}) == {PF.UNVERIFIED: DIGITS}


def test_standard_font_encoding_disagreement_is_a_mismatch_and_agreement_is_clean():
    assert _simple(HELV_TOUNI, {}) == {}
    assert _simple(HELV_TOUNI, {}, {0x33: "4"}) == {PF.MISMATCH: "4"}


def test_a_soft_hyphen_for_a_hyphen_glyph_is_not_a_mismatch():
    assert _simple(HELV_TOUNI, {}, {0x2D: "­"}) == {}


@pytest.mark.parametrize("text,drawn,expected", [
    ("­", {"-"}, False),                       # benign soft hyphen
    (" ", {"3"}, True),                             # digit reported as a space
    ("4", {"3"}, False),                            # another digit: a plain mismatch, nothing hidden
    ("x", {"3"}, True),                             # digit reported as a letter
    ("", {"."}, True),                              # decimal mark reported as nothing
    (".", {"."}, False),
    ("a", set(), False),                            # nothing number-like drawn
])
def test_hides_only_when_a_number_glyph_is_not_reported_as_one(text, drawn, expected):
    assert PF._hides(text, drawn) is expected


# --------------------------------------------------- which fonts are judged ---

def test_a_font_the_page_does_not_draw_with_is_not_judged():
    """A tampered font that is only listed (named Courier here, the page draws in Helvetica) is ignored."""
    font = HELV_TOUNI.replace(b"Helvetica", b"Courier")
    data = T.build(T.table(T.line("F1", 200, 660, "231"), ruled=True).encode(), fonts="/F1 3 0 R /F2 6 0 R",
                   extra={6: font, **_tou({0x33: "4"})})
    assert _verdict(data) == {}


def test_fonts_inside_a_form_xobject_are_judged():
    content = b"BT /F2 11 Tf 1 0 0 1 200 660 Tm (231) Tj ET"
    assert _verdict(F.font_pdf(HELV_TOUNI, _tou({0x33: "4"}), xobject=content)) == {PF.MISMATCH: "4"}
    assert _verdict(F.font_pdf(HELV_TOUNI, _tou(), xobject=content)) == {}


def test_an_unreadable_font_dict_in_a_form_the_page_lists_leaves_every_digit_unverified():
    bad = b"<< /Type /Font /Subtype /Type0 /BaseFont /X >>"                # no DescendantFonts
    data = F.font_pdf(bad, {}, xobject=b"", draw_form=False)
    # r7 round 2: every number character the reader can resolve (NUMBER_CHARS: digits, signs, separators, multipliers),
    # the digits included: an unjudgeable font marks the whole number, never only its digits
    assert _verdict(data) == {PF.UNVERIFIED: NUMBER_CHARS_TEXT} and set(DIGITS) <= set(NUMBER_CHARS_TEXT)


def test_unreadable_page_resources_leave_every_digit_unverified():
    page = SimpleNamespace(chars=[], page_obj=None)                       # page_obj.resources raises
    assert PF.GlyphChecker().page(page) == {PF.UNVERIFIED: NUMBER_CHARS_TEXT}         # r7 round 2: digits and more


def test_a_font_is_judged_once_per_document():
    checker = PF.GlyphChecker()
    calls = []
    real = checker.rsrc.get_font
    checker.rsrc.get_font = lambda *a, **k: (calls.append(1), real(*a, **k))[1]
    data = F.font_pdf(HELV_TOUNI, _tou())
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        checker.page(pdf.pages[0])
        n = len(calls)
        checker.page(pdf.pages[0])
    assert n >= 1 and len(calls) == n


# ------------------------------------------------------ resource discovery ---

def _form(resources: dict) -> PDFStream:
    return PDFStream({"Subtype": SimpleNamespace(name="Form"), "Resources": resources}, b"")


def _font(name: str) -> dict:
    return {"Subtype": SimpleNamespace(name="Type1"), "BaseFont": name}


def test_font_specs_ignore_non_dict_resources():
    assert PF._font_specs(None) == [] and PF._font_specs("x") == []


def test_font_specs_stop_below_the_form_nesting_limit():
    inner = {"Font": {"Deep": _font("Deep")}}
    for _ in range(PF.MAX_FORM_DEPTH + 1):                                # one level past the limit
        inner = {"XObject": {"X": _form(inner)}}
    names = [spec["BaseFont"] for _, spec in PF._font_specs({"Font": {"Top": _font("Top")}, **inner})]
    assert names == ["Top"]                                               # 'Deep' sits beyond MAX_FORM_DEPTH


def test_font_specs_reach_a_font_exactly_at_the_nesting_limit():
    inner = {"Font": {"Deep": _font("Deep")}}
    for _ in range(PF.MAX_FORM_DEPTH):
        inner = {"XObject": {"X": _form(inner)}}
    assert [s["BaseFont"] for _, s in PF._font_specs(inner)] == ["Deep"]


def test_font_specs_visit_a_form_drawn_twice_once_and_skip_non_forms():
    shared = _form({"Font": {"F": _font("Shared")}})
    image = PDFStream({"Subtype": SimpleNamespace(name="Image")}, b"")
    out = PF._font_specs({"XObject": {"A": shared, "B": shared, "C": image, "D": "not a stream"}})
    assert [s["BaseFont"] for _, s in out] == ["Shared"]


def test_font_specs_cap_the_number_of_fonts():
    fonts = {f"F{i}": _font(f"F{i}") for i in range(PF.MAX_FONTS + 25)}
    assert len(PF._font_specs({"Font": fonts})) == PF.MAX_FONTS
