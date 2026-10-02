"""Hand-built minimal PDFs for parser-guard tests (hidden text, huge page boxes, inflate bombs, ruled
grids). Each page is (mediabox, content stream bytes); Helvetica is the only font."""
from __future__ import annotations

import zlib


def build(pages: list[tuple[list[float], bytes]], *, compress: bool = False) -> bytes:
    objs: dict[int, bytes] = {}
    kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(len(pages)))
    objs[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objs[2] = f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode()
    objs[3] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    for i, (box, content) in enumerate(pages):
        p, c = 4 + 2 * i, 5 + 2 * i
        objs[p] = (f"<< /Type /Page /Parent 2 0 R /MediaBox [{' '.join(str(x) for x in box)}] /Contents {c} 0 R "
                   "/Resources << /Font << /F1 3 0 R >> >> >>").encode()
        body = zlib.compress(content, 9) if compress else content
        filt = " /Filter /FlateDecode" if compress else ""
        objs[c] = f"<< /Length {len(body)}{filt} >>\nstream\n".encode() + body + b"\nendstream"
    out = bytearray(b"%PDF-1.4\n")
    offsets = {}
    for k in sorted(objs):
        offsets[k] = len(out)
        out += f"{k} 0 obj\n".encode() + objs[k] + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for k in sorted(objs):
        out += f"{offsets[k]:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def text(lines: list[str], *, size: int = 11, x: float = 50, y0: float = 750, step: float = 14,
         prefix: str = "") -> bytes:
    """One BT block; *prefix* is raw operators (e.g. '3 Tr' or '1 1 1 rg') applied to every line."""
    s = f"BT /F1 {size} Tf {prefix}\n".encode()
    y = y0
    for ln in lines:
        esc = ln.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        s += f"1 0 0 1 {x} {y} Tm ({esc}) Tj\n".encode("latin-1")
        y -= step
    return s + b"ET\n"


def grid(n: int, *, x0: float = 50, y0: float = 50, size: float = 500) -> bytes:
    """n horizontal and n vertical ruling lines (a dense table grid)."""
    step = size / max(1, n - 1)
    ops = [b"0.5 w"]
    for i in range(n):
        ops.append(f"{x0} {y0 + i * step:.2f} m {x0 + size} {y0 + i * step:.2f} l S".encode())
        ops.append(f"{x0 + i * step:.2f} {y0} m {x0 + i * step:.2f} {y0 + size} l S".encode())
    return b"\n".join(ops) + b"\n"


BODY = ["Results. In vitro rat hepatocytes were incubated at 37 degrees (n = 12).",
        "Intrinsic clearance was estimated from the substrate depletion slope."]
