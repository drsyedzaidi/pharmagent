"""Hand-built PDFs with embedded font programs for the glyph-identity tests (parse_fonts).

Every program is generated with fontTools, so no system font is needed: a TrueType whose glyph order is
.notdef, zero .. nine (glyph id g draws digit g - 1, cmap U+0030 + d -> glyph d + 1), a bare CFF (Type1C)
program with a chosen built-in encoding, and a clear-text Type1 header with a chosen /Encoding array.
``font_pdf`` puts one font object (6) beside Helvetica (F1) and draws '231' with it as F2.
"""
from __future__ import annotations

import io

from fontTools.fontBuilder import FontBuilder
from fontTools.pens.t2CharStringPen import T2CharStringPen
from fontTools.pens.ttGlyphPen import TTGlyphPen

from tests import evidence_trickpdf as T

DIGIT_NAMES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
GLYPHS = [".notdef", *DIGIT_NAMES]


def _setup_common(fb: FontBuilder, names: list[str]) -> None:
    fb.setupHorizontalMetrics({n: (500, 50) for n in names})
    fb.setupHorizontalHeader(ascent=800, descent=-200)
    fb.setupNameTable({"familyName": "T", "styleName": "R"})
    fb.setupOS2()
    fb.setupPost()


def ttf_program(*, extra_glyphs: tuple[str, ...] = (), cmap_digits: bool = True) -> bytes:
    """TrueType with glyph ids per GLYPHS (+ *extra_glyphs* appended, never in the cmap). *cmap_digits*
    False leaves the digit glyphs out of the cmap too (only .notdef is listed): their names identify them."""
    names = [*GLYPHS, *extra_glyphs]
    fb = FontBuilder(1000, isTTF=True)
    fb.setupGlyphOrder(names)
    fb.setupCharacterMap({0x30 + d: DIGIT_NAMES[d] for d in range(10)} if cmap_digits else {0x41: ".notdef"})
    glyphs = {}
    for name in names:
        pen = TTGlyphPen(None)
        pen.moveTo((50, 0)), pen.lineTo((50, 700)), pen.lineTo((450, 700)), pen.lineTo((450, 0)), pen.closePath()
        glyphs[name] = pen.glyph()
    fb.setupGlyf(glyphs)
    _setup_common(fb, names)
    out = io.BytesIO()
    fb.save(out)
    return out.getvalue()


def cff_program(encoding: dict[int, str] | None, *, extra_glyphs: tuple[str, ...] = ()) -> bytes:
    """A bare CFF table (what /FontFile3 /Type1C holds). *encoding* None keeps StandardEncoding (a name,
    not an array); otherwise code -> glyph name is written as the program's own Encoding array."""
    names = [*GLYPHS, *extra_glyphs]
    fb = FontBuilder(1000, isTTF=False)
    fb.setupGlyphOrder(names)
    fb.setupCharacterMap({0x30 + d: DIGIT_NAMES[d] for d in range(10)})
    strings = {}
    for name in names:
        pen = T2CharStringPen(500, None)
        pen.moveTo((50, 0)), pen.lineTo((50, 700)), pen.lineTo((450, 700)), pen.closePath()
        strings[name] = pen.getCharString()
    fb.setupCFF("TCFF", {"FullName": "TCFF"}, strings, {})
    if encoding is not None:
        table = [".notdef"] * 256
        for code, glyph in encoding.items():
            table[code] = glyph
        fb.font["CFF "].cff.topDictIndex[0].Encoding = table
    _setup_common(fb, names)
    return fb.font["CFF "].compile(fb.font)


def type1_header(encoding: dict[int, str]) -> bytes:
    """The clear-text part of a Type1 program: pdfminer reads its /Encoding array (``dup n /name put``)."""
    puts = "".join(f"dup {code} /{name} put\n" for code, name in encoding.items())
    return ("%!PS-AdobeFont-1.0: TType1 1.0\n/FontName /TType1 def\n/Encoding 256 array\n"
            f"0 1 255 {{1 index exch /.notdef put}} for\n{puts}readonly def\ncurrentdict end\n").encode()


def cmap(mapping: dict[int, str], *, width: int = 1) -> bytes:
    """A /ToUnicode CMap over *width*-byte codes; codes not in *mapping* are absent."""
    pairs = [f"<{code:0{2 * width}X}> <{''.join(f'{ord(ch):04X}' for ch in text)}>"
             for code, text in sorted(mapping.items())]
    return ("/CIDInit /ProcSet findresource begin 12 dict begin begincmap\n/CMapName /X def /CMapType 2 def\n"
            f"1 begincodespacerange <{'00' * width}> <{'FF' * width}> endcodespacerange\n"
            f"{len(pairs)} beginbfchar\n" + "\n".join(pairs)
            + "\nendbfchar endcmap CMapName currentdict /CMap defineresource pop end end\n").encode()


def digit_cmap(overrides: dict[int, str] | None = None, *, base: int = 0x30) -> dict[int, str]:
    """code -> text for the ten digits ('0' at code *base*), with *overrides* applied on top."""
    return {base + d: str(d) for d in range(10)} | (overrides or {})


def font_pdf(font: bytes, extra: dict[int, bytes], *, text: str = "231", hex_codes: str | None = None,
             xobject: bytes | None = None, draw_form: bool = True) -> bytes:
    """The ruled Table-1 page with *text* (or the hex string *hex_codes*) drawn in font F2 (object 6).

    *xobject*: put the F2 font into a Form XObject (the object after the last of *extra*; T.build's xref needs contiguous numbers) with that content stream; the page draws the
    form (*draw_form*) or only lists it in its resources (pdfminer then never loads the form's fonts)."""
    shown = f"<{hex_codes}>" if hex_codes else f"({text})"
    if xobject is None:
        cmax = f"BT /F2 11 Tf 1 0 0 1 200 660 Tm {shown} Tj ET\n"
        return T.build(T.table(cmax, ruled=True).encode(), fonts="/F1 3 0 R /F2 6 0 R", extra={6: font, **extra})
    cmax = "q /X Do Q\n" if draw_form else T.line("F1", 200, 660, text)
    form_id = max([6, *extra]) + 1
    form = T.stream(xobject, " /Type /XObject /Subtype /Form /BBox [0 0 612 792] /Resources << /Font << /F2 6 0 R >> >>")
    # '>> /XObject <<' closes the page's /Font dictionary early so the page also gets an /XObject entry
    return T.build(T.table(cmax, ruled=True).encode(), fonts=f"/F1 3 0 R >> /XObject << /X {form_id} 0 R",
                   extra={6: font, form_id: form, **extra})


def type0_font(*, to_unicode: int = 7, descendant: int = 8) -> bytes:
    return (f"<< /Type /Font /Subtype /Type0 /BaseFont /TTT /Encoding /Identity-H /DescendantFonts [{descendant} 0 R] "
            f"/ToUnicode {to_unicode} 0 R >>").encode()


def cid_objects(program: bytes | None, *, subtype: str = "CIDFontType2", gid_map: bytes | None = None,
                mapping: dict[int, str] | None = None) -> dict[int, bytes]:
    """Objects 7 (ToUnicode over 2-byte codes), 8 (descendant), 9 (descriptor), 10 (font program),
    11 (CIDToGIDMap stream, when *gid_map*). CID c is glyph c with /Identity, else gid_map[2c:2c+2].
    *mapping* overrides the ToUnicode text per CID (CID 1..10 -> '0'..'9' by default)."""
    mapping = digit_cmap(mapping, base=1)
    descriptor = b"<< /Type /FontDescriptor /FontName /TTT /Flags 4 /FontBBox [0 0 500 700] /ItalicAngle 0 " \
                 b"/Ascent 800 /Descent -200 /CapHeight 700 /StemV 80" \
                 + (b" /FontFile2 10 0 R" if program is not None and subtype == "CIDFontType2" else b"") \
                 + (b" /FontFile3 10 0 R" if program is not None and subtype != "CIDFontType2" else b"") + b" >>"
    to_gid = b"/CIDToGIDMap 11 0 R" if gid_map is not None else b"/CIDToGIDMap /Identity"
    objs = {7: T.stream(cmap(mapping, width=2)),
            8: (f"<< /Type /Font /Subtype /{subtype} /BaseFont /TTT /CIDSystemInfo << /Registry (Adobe) "
                f"/Ordering (Identity) /Supplement 0 >> /FontDescriptor 9 0 R /DW 500 ").encode() + to_gid + b" >>",
            9: descriptor}
    if program is not None:
        objs[10] = T.stream(program, f" /Length1 {len(program)}")
    if gid_map is not None:
        objs[11] = T.stream(gid_map)
    return objs
