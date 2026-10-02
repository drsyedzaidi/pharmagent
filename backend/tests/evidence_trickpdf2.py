"""Round-2b hand-built PDFs (attack round 2): the drawn ink of a number differs from its text layer.

Each puts Cmax '281' (or similar) into the ruled 'Table 1' of evidence_trickpdf.table(); W is the
Helvetica digit advance at 11 pt. A reader sees what the docstring says, the text layer says otherwise.
"""
from __future__ import annotations

import zlib

from tests import evidence_trickpdf as T

W = 6.116


def _tj(x: float, s: str, matrix: str = "1 0 0 1", y: float = 660) -> str:
    return f"BT /F1 11 Tf {matrix} {x} {y} Tm ({s}) Tj ET\n"


def actualtext() -> bytes:
    """Reader 231: '3' is wrapped in /ActualText (8); a fully transparent '8' sits at its origin."""
    cm = (_tj(200, "2") + "/Span << /ActualText (8) >> BDC\n" + _tj(200 + W, "3") + "EMC\n"
          + "q /G0 gs\n" + _tj(200 + W, "8") + "Q\n" + _tj(200 + 2 * W, "1"))
    return T.build(T.table(cm, ruled=True).encode(), fonts="/F1 3 0 R >> /ExtGState << /G0 6 0 R",
                   extra={6: b"<< /Type /ExtGState /ca 0 >>"})


def rotated() -> bytes:
    """Reader 239: the last glyph is a '6' rotated 180 degrees about its centre (a 9)."""
    cx, cy = 200 + 2 * W + W / 2, 660 + 11 * 0.703 / 2
    return T.build(T.table(_tj(200, "23") + _tj(cx + W / 2, "6", "-1 0 0 -1", round(2 * cy - 660, 3)),
                           ruled=True).encode())


def vector_point(gap: float = 2.4) -> bytes:
    """Reader 12.4: the decimal point is a filled path in a *gap* pt gap, not a glyph."""
    return T.build(T.table(_tj(200, "12") + f"{200 + 2 * W + 0.6:.3f} 660 1.2 1.3 re f\n"
                           + _tj(200 + 2 * W + gap, "4"), ruled=True).encode())


def white_ruling() -> bytes:
    """Reader 281 in one cell; a white 0.1 pt stroke between '2' and '8' could split the cell."""
    return T.build(T.table(_tj(200, "281") + f"1 G 0.1 w {200 + W:.3f} 657 m {200 + W:.3f} 669 l S 0 G 0.5 w\n",
                           ruled=True).encode())


def clipped_eight() -> bytes:
    """Reader 231: the '8' of 281 is clipped to its right half (it reads as 3)."""
    x8 = 200 + W
    return T.build(T.table(_tj(200, "2") + f"q {x8 + 2.6:.3f} 650 20 25 re W n\n" + _tj(x8, "8") + "Q\n"
                           + _tj(200 + 2 * W, "1"), ruled=True).encode())


def covered_half() -> bytes:
    """Reader 231: a white bar covers the left strokes of the '8' of 281."""
    return T.build(T.table(_tj(200, "281") + f"1 g {200 + W + 0.2:.3f} 659 2.30 10 re f 0 g\n",
                           ruled=True).encode())


def plain_cell(text: str = "281") -> bytes:
    return T.build(T.table(_tj(200, text), ruled=True).encode())


def middot_line(code: str = "\\267") -> bytes:
    """WinAnsi Helvetica without /Widths: 'Mean Cmax was 12<periodcentered>4 ng/mL ...' (pdfminer and
    pdfium disagree on the dot's advance by 0.6 pt, so every later origin drifts)."""
    win = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"
    body = "".join(T.line("F1", 50, 750 - 14 * i, t) for i, t in enumerate(T.BODY))
    body += f"BT /F2 11 Tf 1 0 0 1 50 690 Tm (Mean Cmax was 12{code}4 ng/mL in all subjects.) Tj ET\n"
    return T.build(body.encode("latin-1"), fonts="/F1 3 0 R /F2 6 0 R", extra={6: win})


def dense_page(n: int = 64_000) -> bytes:
    """One page of *n* one-glyph text objects on a 2 pt grid (3 pt digits): every glyph visible, none a
    duplicate, all number characters -- the visibility passes' worst case."""
    ops = [f"BT /F1 3 Tf 1 0 0 1 {5 + (k % 300) * 2.0:.1f} {787 - (k // 300) * 2.0:.1f} Tm ({k % 10}) Tj ET"
           for k in range(n)]
    comp = zlib.compress("\n".join(ops).encode(), 9)
    return T.build(b"", extra={5: T.stream(comp, " /Filter /FlateDecode")})
