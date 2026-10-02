"""One corpus of PDFs for the layout / page-image tests (built at test time, nothing binary in the repo).

``corpus(dir)`` returns {name: path}: the four fixture papers, hand-built pages with Rotate / CropBox /
MediaBox origin variants, and the adversarial PoCs whose text layer disagrees with what is drawn."""
from __future__ import annotations

from pathlib import Path

from tests import evidence_fixtures as fx
from tests import evidence_rawpdf as raw
from tests import evidence_trickpdf as T
from tests import evidence_trickpdf2 as T2

LINE = "Cmax was 3,412 ng/mL"
BOXES = {"plain": ([0, 0, 612, 792], {}),
         "mbo": ([50, 60, 662, 852], {}),
         "crop": ([0, 0, 612, 792], {"CropBox": [100, 300, 500, 600]}),
         "rot90": ([0, 0, 612, 792], {"Rotate": 90}),
         "rot180": ([0, 0, 612, 792], {"Rotate": 180}),
         "rot270": ([0, 0, 612, 792], {"Rotate": 270}),
         "rot90_crop": ([0, 0, 612, 792], {"Rotate": 90, "CropBox": [100, 300, 500, 600]}),
         "rot180_crop": ([0, 0, 612, 792], {"Rotate": 180, "CropBox": [100, 300, 500, 600]}),
         "rot270_crop": ([0, 0, 612, 792], {"Rotate": 270, "CropBox": [100, 300, 500, 600]}),
         "mbo_rot90_crop": ([50, 60, 662, 852], {"Rotate": 90, "CropBox": [150, 360, 550, 660]}),
         "mbo_rot270_crop": ([50, 60, 662, 852], {"Rotate": 270, "CropBox": [150, 360, 550, 660]})}


def raw_page(name: str) -> bytes:
    """Three lines of 'Cmax was 3,412 ng/mL' at x=170, y=500.. on a page with the named box variant."""
    box, extras = BOXES[name]
    return raw.build([(box, raw.text([LINE] * 3, x=170, y0=500), extras)])


def poc1_space() -> bytes:
    """Reader sees Cmax 231; F2 = Helvetica + ToUnicode mapping code '1' -> U+0020 (r2 PoC 1)."""
    font = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding /ToUnicode 7 0 R >>"
    return T.build(T.table(T.line("F2", 200, 660, "231"), ruled=True).encode(), fonts="/F1 3 0 R /F2 6 0 R",
                   extra={6: font, 7: T.stream(T._cmap({0x31: " "}))})


def adversarial() -> dict[str, bytes]:
    return {"tounicode": T.tounicode_pdf(), "poc1_space": poc1_space(), "type3": T.type3_pdf(),
            "widget": T.widget_pdf(), "overstrike": T.overstrike_pdf(0.0), "columns_pdf": T.columns_pdf(),
            "raised_exponent": T.raised_exponent_pdf(), "actualtext": T2.actualtext(),
            "rotated_glyph": T2.rotated(), "vector_point": T2.vector_point(), "white_ruling": T2.white_ruling(),
            "clipped_eight": T2.clipped_eight(), "covered_half": T2.covered_half(), "plain_cell": T2.plain_cell()}


def _one_path(x: float, y: float, s: float = 11.0) -> str:
    k = s / 1000
    pts = [(259, 0), (359, 0), (359, 703), (300, 703), (300, 640), (140, 590), (140, 520), (259, 560)]
    p = " ".join(f"{x + a * k:.3f} {y + b * k:.3f} {'m' if i == 0 else 'l'}" for i, (a, b) in enumerate(pts))
    return f"0 g {p} h f\n"


def poc_extras() -> dict[str, bytes]:
    """Round-2 attack PoCs 2, 3, 4, 6 and 7 (a Type3 glyph named /space that paints a '1', a '1' drawn as a
    filled path, a stroked bar over a number, an optional-content group that is off, a zero-width space)."""
    proc = b"556 0 d0 BT /H 1000 Tf 0 0 Td (1) Tj ET"
    type3 = (b"<< /Type /Font /Subtype /Type3 /FontBBox [0 0 600 1000] /FontMatrix [0.001 0 0 0.001 0 0] "
             b"/CharProcs << /space 7 0 R >> /Encoding << /Type /Encoding /Differences [32 /space] >> "
             b"/FirstChar 32 /LastChar 32 /Widths [556] /Resources << /Font << /H 3 0 R >> >> >>")
    poc2 = T.build(T.table("BT /F1 11 Tf 1 0 0 1 200 660 Tm (23) Tj /F3 11 Tf ( ) Tj ET\n", ruled=True).encode(),
                   fonts="/F1 3 0 R /F3 6 0 R", extra={6: type3, 7: T.stream(proc)})
    c3 = T.table("BT /F1 11 Tf 1 0 0 1 200 660 Tm (23) Tj ET\n" + _one_path(200 + 2 * 6.116, 660), ruled=True)
    c3 = c3.replace("1 0 0 1 200 646 Tm (455)", "1 0 0 1 206.116 646 Tm (455)") + _one_path(200, 646)
    poc3 = T.build(c3.encode())
    bar = "".join(f"q BT /F1 11 Tf 2 Tr 16 w 0 j 0 0 0 RG 0 g 1 0 0 1 {x} 665.5 Tm (-) Tj ET Q\n" for x in (203, 210))
    poc4 = T.build(T.table(T.line("F1", 200, 660, "231") + bar, ruled=True).encode())
    ocg = b"<< /Type /OCG /Name (hidden) >>"
    poc6 = T.build(T.table("/OC /L1 BDC " + T.line("F1", 200, 660, "241") + "EMC\n", ruled=True).encode(),
                   extra={6: ocg}, catalog_extra=" /OCProperties << /OCGs [6 0 R] /D << /OFF [6 0 R] /Order [6 0 R] >> >>"
                   ).replace(b"/Resources << /Font << /F1 3 0 R >> >>",
                             b"/Resources << /Font << /F1 3 0 R >> /Properties << /L1 6 0 R >> >>")
    poc7 = T.build(T.table("BT /F1 11 Tf -3.058 Tw 1 0 0 1 200 660 Tm (2 31) Tj 0 Tw ET\n", ruled=True).encode())
    return {"poc2_type3_space": poc2, "poc3_path_digit": poc3, "poc4_stroke_cover": poc4, "poc6_ocg": poc6,
            "poc7_zero_width": poc7}


def corpus(directory: Path) -> dict[str, Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    out = dict(fx.build_all(directory))
    for name in BOXES:
        out[f"raw_{name}"] = _write(directory / f"raw_{name}.pdf", raw_page(name))
    for name, data in adversarial().items():
        out[name] = _write(directory / f"{name}.pdf", data)
    return out


def _write(path: Path, data: bytes) -> Path:
    path.write_bytes(data)
    return path
