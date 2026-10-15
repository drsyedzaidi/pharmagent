"""Sandboxed PDF worker. Runs as ``python -I worker.py '<json request>'``.

SELF-CONTAINED ON PURPOSE: ``-I`` removes the cwd/script dir from sys.path, so
this file may import only the stdlib and ``pypdfium2``. It is the only code that
ever parses an untrusted PDF. Output: PNG bytes (render) or one JSON line (info)
on stdout; a coarse exit code tells the parent what went wrong. It never prints
paths or exception text.

Defence in depth: the parent already validated the request; the worker
re-checks it, applies rlimits, and enforces the pixel/byte caps again.
"""
from __future__ import annotations

import json
import math
import struct
import sys
import zlib

EXIT_OK = 0
EXIT_BAD_REQUEST = 2
EXIT_ENCRYPTED = 3
EXIT_MALFORMED = 4
EXIT_NO_PAGE = 5
EXIT_TOO_LARGE = 6       # requested pixels / page count exceed the caps
EXIT_EMPTY = 7           # crop collapses to < 1 pixel
EXIT_OUTPUT_TOO_BIG = 8  # encoded PNG exceeds the byte cap


def apply_limits(cpu_seconds: int, address_space: int) -> None:
    """Best-effort rlimits. RLIMIT_AS is not enforced on macOS (setrlimit may
    raise or be a no-op there); the parent's wall-clock kill is the backstop."""
    try:
        import resource
    except ImportError:  # Windows
        return
    for name, soft in (
        ("RLIMIT_CPU", cpu_seconds),
        ("RLIMIT_AS", address_space),
        ("RLIMIT_FSIZE", 0),      # the worker never needs to write a file
        ("RLIMIT_CORE", 0),
        ("RLIMIT_NOFILE", 64),
    ):
        rl = getattr(resource, name, None)
        if rl is None:
            continue
        try:
            _, hard = resource.getrlimit(rl)
            if hard != resource.RLIM_INFINITY:
                soft = min(soft, hard)
            resource.setrlimit(rl, (soft, hard))
        except (ValueError, OSError):
            pass


def _chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def encode_png(width: int, height: int, rgb, stride: int, max_bytes: int) -> bytes | None:
    """Encode 8-bit RGB rows as a PNG. Returns None when the encoded size
    would exceed ``max_bytes`` (stops early; no unbounded allocation)."""
    row_len = width * 3
    view = memoryview(rgb).cast("B")
    comp = zlib.compressobj(6)
    parts: list[bytes] = []
    total = 0
    for y in range(height):
        start = y * stride
        data = comp.compress(b"\x00" + bytes(view[start:start + row_len]))
        if data:
            parts.append(data)
            total += len(data)
            if total > max_bytes:
                return None
    parts.append(comp.flush())
    idat = b"".join(parts)
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit truecolour
    png = b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")
    return png if len(png) <= max_bytes else None


def plan_output_size(page_w: float, page_h: float, scale: float, bbox,
                     max_dim: int, max_pixels: int):
    """Pixel size (w, h) of the cropped render, or an EXIT_* code if refused."""
    fx = (bbox[2] - bbox[0]) if bbox else 1.0
    fy = (bbox[3] - bbox[1]) if bbox else 1.0
    w = math.ceil(page_w * fx * scale)
    h = math.ceil(page_h * fy * scale)
    if w < 1 or h < 1:
        return EXIT_EMPTY
    # +1 tolerates the library's own rounding of the crop box.
    if w + 1 > max_dim or h + 1 > max_dim or (w + 1) * (h + 1) > max_pixels:
        return EXIT_TOO_LARGE
    return (w, h)


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _num(v, lo, hi) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and lo <= v <= hi


def _parse(raw: str) -> dict:
    req = json.loads(raw)
    if not isinstance(req, dict) or req.get("op") not in ("info", "render"):
        raise ValueError
    if not isinstance(req.get("path"), str) or not isinstance(req.get("limits"), dict):
        raise ValueError
    lim = req["limits"]
    for k in ("cpu_seconds", "address_space", "max_dim", "max_pixels", "max_png_bytes", "max_pages"):
        if not _is_int(lim.get(k)) or lim[k] < 0:
            raise ValueError
    if req["op"] == "render":
        if not _is_int(req.get("page")) or req["page"] < 0:
            raise ValueError
        if not _num(req.get("scale"), 0.01, 10):
            raise ValueError
        bb = req.get("bbox")
        if bb is not None:
            ok = (isinstance(bb, list) and len(bb) == 4 and all(_num(v, 0, 1) for v in bb)
                  and bb[0] < bb[2] and bb[1] < bb[3])
            if not ok:
                raise ValueError
    return req


def main(argv: list[str]) -> int:
    try:
        if len(argv) != 2:
            raise ValueError
        req = _parse(argv[1])
    except Exception:
        return EXIT_BAD_REQUEST
    lim = req["limits"]
    apply_limits(lim["cpu_seconds"], lim["address_space"])
    try:
        import pypdfium2 as pdfium
    except Exception:
        return EXIT_MALFORMED
    try:
        doc = pdfium.PdfDocument(req["path"])
    except pdfium.PdfiumError as e:
        return EXIT_ENCRYPTED if getattr(e, "err_code", None) == pdfium.raw.FPDF_ERR_PASSWORD else EXIT_MALFORMED
    except Exception:
        return EXIT_MALFORMED
    try:
        n_pages = len(doc)
        if req["op"] == "info":
            if n_pages > lim["max_pages"]:
                return EXIT_TOO_LARGE
            sys.stdout.write(json.dumps({"pages": n_pages}) + "\n")
            sys.stdout.flush()
            return EXIT_OK
        if req["page"] >= n_pages:
            return EXIT_NO_PAGE
        page = doc[req["page"]]
        page_w, page_h = page.get_size()
        if not (math.isfinite(page_w) and math.isfinite(page_h) and page_w > 0 and page_h > 0):
            return EXIT_MALFORMED
        bbox = req.get("bbox")
        planned = plan_output_size(page_w, page_h, req["scale"], bbox, lim["max_dim"], lim["max_pixels"])
        if isinstance(planned, int):
            return planned
        crop = (0.0, 0.0, 0.0, 0.0)
        if bbox:
            x0, y0, x1, y1 = bbox
            crop = (x0 * page_w, (1 - y1) * page_h, (1 - x1) * page_w, y0 * page_h)  # left, bottom, right, top
        bitmap = page.render(scale=req["scale"], crop=crop, rev_byteorder=True, may_draw_forms=False)
        if bitmap.n_channels != 3:
            return EXIT_MALFORMED
        if (bitmap.width > lim["max_dim"] or bitmap.height > lim["max_dim"]
                or bitmap.width * bitmap.height > lim["max_pixels"]):
            return EXIT_TOO_LARGE
        png = encode_png(bitmap.width, bitmap.height, bitmap.buffer, bitmap.stride, lim["max_png_bytes"])
        if png is None:
            return EXIT_OUTPUT_TOO_BIG
        sys.stdout.buffer.write(png)
        sys.stdout.buffer.flush()
        return EXIT_OK
    except pdfium.PdfiumError:
        return EXIT_MALFORMED
    except MemoryError:
        return EXIT_TOO_LARGE
    except Exception:
        return EXIT_MALFORMED


if __name__ == "__main__":
    sys.exit(main(sys.argv))
