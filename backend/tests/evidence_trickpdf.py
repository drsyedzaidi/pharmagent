"""Hand-built PDFs whose text layer disagrees with what a reader sees (round-2 attack regressions):
a /ToUnicode CMap that renames a drawn digit, a Type3 font whose glyph named 'three' paints a 4, a
form-field Widget appearance laid over page text, an over-struck digit, a two-column page whose
merged line joins 'n =' with the other column's number, a raised 80 %-size exponent, and a tiny
upload whose catalog sits in a huge FlateDecode object stream. F1 is Helvetica (standard 14)."""
from __future__ import annotations

import io
import zlib

HELV = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
BODY = ["Results. Healthy adult volunteers received a single oral dose of bupropion.",
        "Plasma concentrations were measured by LC-MS/MS and analysed by NCA.",
        "Pharmacokinetic parameters are summarised in Table 1 below."]


def stream(body: bytes, extra: str = "") -> bytes:
    return f"<< /Length {len(body)}{extra} >>\nstream\n".encode() + body + b"\nendstream"


def build(content: bytes, *, fonts: str = "/F1 3 0 R", extra: dict[int, bytes] | None = None,
          page_extra: str = "", catalog_extra: str = "") -> bytes:
    """One 612 x 792 page; objects 1-5 are fixed (catalog, pages, Helvetica, page, contents)."""
    objs = {1: f"<< /Type /Catalog /Pages 2 0 R{catalog_extra} >>".encode(),
            2: b"<< /Type /Pages /Kids [4 0 R] /Count 1 >>", 3: HELV,
            4: (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 5 0 R "
                f"/Resources << /Font << {fonts} >> >>{page_extra} >>").encode(),
            5: stream(content), **(extra or {})}
    out, offs = bytearray(b"%PDF-1.5\n"), {}
    for k in sorted(objs):
        offs[k] = len(out)
        out += f"{k} 0 obj\n".encode() + objs[k] + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for k in sorted(objs):
        out += f"{offs[k]:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def line(font: str, x: float, y: float, s: str, size: int = 11) -> str:
    esc = s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    return f"BT /{font} {size} Tf 1 0 0 1 {x} {y} Tm ({esc}) Tj ET\n"


def table(cmax: str, *, ruled: bool = False) -> str:
    """Body lines plus 'Table 1' with Cmax drawn by the raw operators *cmax* in the 150 mg row."""
    grid = ("0.5 w\n" + "".join(f"45 {y} m 445 {y} l S\n" for y in (686, 670, 656, 642))
            + "".join(f"{x} 642 m {x} 686 l S\n" for x in (45, 195, 345, 445))) if ruled else ""
    c = grid + "".join(line("F1", 50, 750 - 14 * i, t) for i, t in enumerate(BODY))
    c += line("F1", 50, 694 if ruled else 690, "Table 1 Pharmacokinetic parameters")
    c += line("F1", 50, 674, "Group") + line("F1", 200, 674, "Cmax (ng/mL)") + line("F1", 350, 674, "AUC")
    c += line("F1", 50, 660, "150 mg") + cmax + line("F1", 350, 660, "1890")
    c += line("F1", 50, 646, "300 mg") + line("F1", 200, 646, "455") + line("F1", 350, 646, "3720")
    return c


def _cmap(mapping: dict[int, str]) -> bytes:
    pairs = [f"<{c:02X}> <{''.join(f'{ord(ch):04X}' for ch in mapping.get(c, chr(c)))}>" for c in range(0x20, 0x7F)]
    return ("/CIDInit /ProcSet findresource begin 12 dict begin begincmap\n/CMapName /X def /CMapType 2 def\n"
            f"1 begincodespacerange <00> <FF> endcodespacerange\n{len(pairs)} beginbfchar\n" + "\n".join(pairs)
            + "\nendbfchar endcmap CMapName currentdict /CMap defineresource pop end end\n").encode()


def tounicode_pdf(*, ruled: bool = True) -> bytes:
    """A reader sees Cmax 231; font F2 (Helvetica + ToUnicode code '3' -> '4') makes the text 241."""
    font = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding /ToUnicode 7 0 R >>"
    return build(table(line("F2", 200, 660, "231"), ruled=ruled).encode(), fonts="/F1 3 0 R /F2 6 0 R",
                 extra={6: font, 7: stream(_cmap({0x33: "4"}))})


_SEG = {"a": (100, 900, 400, 80), "b": (420, 520, 80, 380), "c": (420, 100, 80, 380), "d": (100, 20, 400, 80),
        "e": (20, 100, 80, 380), "f": (20, 520, 80, 380), "g": (100, 460, 400, 80)}
_DIGITS = {"0": "abcdef", "1": "bc", "2": "abged", "3": "abgcd", "4": "fgbc", "5": "afgcd", "6": "afgedc",
           "7": "abc", "8": "abcdefg", "9": "abcdfg"}
_NAMES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


def type3_pdf() -> bytes:
    """Seven-segment Type3 digits, no ToUnicode; the glyph NAMED 'three' paints a 4 (reader: 241)."""
    draw = dict(_DIGITS, **{"3": _DIGITS["4"]})
    extra, procs = {}, []
    for i, (d, name) in enumerate(zip("0123456789", _NAMES, strict=True)):
        ops = ["600 0 0 0 520 1000 d1"] + [f"{x} {y} {w} {h} re f" for x, y, w, h in (_SEG[s] for s in draw[d])]
        extra[7 + i] = stream("\n".join(ops).encode())
        procs.append(f"/{name} {7 + i} 0 R")
    extra[6] = (f"<< /Type /Font /Subtype /Type3 /FontBBox [0 0 600 1000] /FontMatrix [0.001 0 0 0.001 0 0] "
                f"/CharProcs << {' '.join(procs)} >> /Encoding << /Type /Encoding /Differences "
                f"[48 {' '.join('/' + n for n in _NAMES)}] >> /FirstChar 48 /LastChar 57 "
                f"/Widths [{' '.join(['600'] * 10)}] /Resources << >> >>").encode()
    c = table(line("F3", 200, 660, "231"), ruled=True)
    return build(c.encode(), fonts="/F1 3 0 R /F3 6 0 R", extra=extra)


def _widget(rect: str, appearance: bytes, bbox: str) -> dict[int, bytes]:
    return {6: (f"<< /Type /Annot /Subtype /Widget /FT /Tx /T (c) /Ff 1 /F 4 /Rect [{rect}] "
                f"/AP << /N 7 0 R >> /DA (/F1 11 Tf 0 g) >>").encode(),
            7: stream(appearance, f" /Type /XObject /Subtype /Form /BBox [{bbox}] /Resources << /Font << /F1 3 0 R >> >>")}


def widget_pdf() -> bytes:
    """Page text draws 241; a read-only text-field Widget's appearance (white box + '231') covers it."""
    ap = b"1 1 1 rg 0 0 60 14 re f BT 0 0 0 rg /F1 11 Tf 2 3 Td (231) Tj ET"
    return build(table(line("F1", 200, 660, "241"), ruled=True).encode(),
                 extra=_widget("197 656 257 670", ap, "0 0 60 14"), page_extra=" /Annots [6 0 R]",
                 catalog_extra=" /AcroForm << /Fields [6 0 R] /DA (/F1 11 Tf 0 g) >>")


SCREEN_LINES = ["Results. Cmax and AUC were estimated by NCA after a single oral dose.",
                "Plasma was assayed by LC-MS/MS; sampling ran to 72 h after dosing.",
                "Parameters are reported as geometric means."]


def widget_screen_pdf(*, covered: bool = True) -> bytes:
    """A PK paper with no population words; 'human adult volunteers patients' lies under a white Widget."""
    c = "".join(line("F1", 50, 750 - 14 * i, t) for i, t in enumerate(SCREEN_LINES))
    if not covered:
        return build(c.encode())
    c += line("F1", 50, 700, "human adult volunteers patients")
    return build(c.encode(), extra=_widget("45 694 300 714", b"1 1 1 rg 0 0 255 20 re f", "0 0 255 20"),
                 page_extra=" /Annots [6 0 R]", catalog_extra=" /AcroForm << /Fields [6 0 R] >>")


def overstrike_pdf(offset_pt: float) -> bytes:
    """Cmax '12' whose '1' is struck twice, the copy *offset_pt* right of the first (reads as a bold 1)."""
    adjust = (6.116 - offset_pt) / 11 * 1000                 # Helvetica '1' advance is 0.556 em
    cm = f"BT /F1 11 Tf 1 0 0 1 200 660 Tm [(1) {adjust:.1f} (12)] TJ ET\n"
    return build(table(cm, ruled=True).encode())


LEFT = ["Methods. Healthy adult volunteers were", "screened at a single clinical site and",
        "those meeting all criteria were dosed.", "The final cohort size was n =",
        "40 subjects, all of whom completed the", "study. Blood was drawn at fixed times",
        "after each oral dose of bupropion and", "plasma was stored frozen until assay.",
        "Concentrations were measured by a", "validated LC-MS/MS method."]
RIGHT = ["Sampling. Samples were taken before", "the dose and at 1, 2, 4, 6 and 8 h, then",
         "at 10 and 12 h and once daily after.", "12 h samples were split for a stability",
         "check. Results were analysed by NCA", "with linear-up log-down integration.",
         "Values below the limit of quantification", "were set to missing before analysis.",
         "No protocol deviations were recorded", "during the sampling period."]


def columns_pdf() -> bytes:
    """Two columns; the left reads 'n = 40', the merged text line reads 'n = 12 h samples'."""
    c = "".join(line("F1", 40, 740 - 13 * i, lt, size=10) + line("F1", 330, 740 - 13 * i, rt, size=10)
                for i, (lt, rt) in enumerate(zip(LEFT, RIGHT, strict=True)))
    return build(c.encode())


def raised_exponent_pdf() -> bytes:
    """'Peak count was 1.2 x 10' at 10 pt with a raised '3' at 8 pt (80 %: above the small-text cut)."""
    c = line("F1", 50, 750, "Viral load fell in all patients over the first week of dosing.", size=10)
    c += line("F1", 50, 735, "Clearance of the antibody was reported in the second table below.", size=10)
    c += "BT /F1 10 Tf 1 0 0 1 50 700 Tm (Peak count was 1.2 x 10) Tj ET\n"
    c += "BT /F1 8 Tf 1 0 0 1 159.2 704 Tm (3) Tj ET\n"
    c += "BT /F1 10 Tf 1 0 0 1 163.7 700 Tm ( copies/mL at day 3.) Tj ET\n"
    return build(c.encode())


def objstm_bomb_pdf(pad: int) -> bytes:
    """Catalog, Pages and the page live in ONE FlateDecode object stream padded with *pad* NUL bytes
    (inflates ~1000x), indexed by a cross-reference stream: counting pages inflates all of it."""
    inner = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
             b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>"]
    bodies, offsets = b"", []
    for i, obj in enumerate(inner, 1):
        offsets.append(f"{i} {len(bodies)}")
        bodies += obj + b" "
    header = " ".join(offsets).encode() + b" "
    comp = zlib.compress(header + bodies + b"\x00" * pad, 9)
    out = io.BytesIO()
    out.write(b"%PDF-1.5\n%\xe2\xe3\xcf\xd3\n")
    objstm_at = out.tell()
    out.write(f"4 0 obj\n<< /Type /ObjStm /N 3 /First {len(header)} /Filter /FlateDecode /Length {len(comp)} >>"
              f"\nstream\n".encode() + comp + b"\nendstream\nendobj\n")
    xref_at = out.tell()

    def entry(t: int, a: int, b: int) -> bytes:
        return bytes([t]) + a.to_bytes(4, "big") + b.to_bytes(2, "big")
    rows = (entry(0, 0, 65535) + entry(2, 4, 0) + entry(2, 4, 1) + entry(2, 4, 2) + entry(1, objstm_at, 0)
            + entry(1, xref_at, 0))
    xcomp = zlib.compress(rows, 9)
    out.write(f"5 0 obj\n<< /Type /XRef /Size 6 /Root 1 0 R /W [1 4 2] /Filter /FlateDecode /Length {len(xcomp)} >>"
              f"\nstream\n".encode() + xcomp + b"\nendstream\nendobj\n")
    out.write(f"startxref\n{xref_at}\n%%EOF".encode())
    return out.getvalue()
