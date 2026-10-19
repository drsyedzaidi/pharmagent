"""Hand-built PDFs and a tiny PNG decoder for the page-image tests.

No PDF/imaging library is used to *build* fixtures, so the tests exercise the
real renderer against independent input.
"""
from __future__ import annotations

import hashlib
import struct
import zlib


def _rc4(key: bytes, data: bytes) -> bytes:
    s = list(range(256))
    j = 0
    for i in range(256):
        j = (j + s[i] + key[i % len(key)]) % 256
        s[i], s[j] = s[j], s[i]
    out = bytearray()
    i = j = 0
    for b in data:
        i = (i + 1) % 256
        j = (j + s[i]) % 256
        s[i], s[j] = s[j], s[i]
        out.append(b ^ s[(s[i] + s[j]) % 256])
    return bytes(out)


_PAD = bytes.fromhex("28BF4E5E4E758A4164004E56FFFA01082E2E00B6D0683E802F0CA9FE6453697A")


def _assemble(objects: list[bytes], trailer_extra: bytes = b"") -> bytes:
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root 1 0 R %s>>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1, trailer_extra, xref)
    return bytes(out)


def make_pdf(pages: int = 1, width: int = 200, height: int = 100,
             rect: tuple[float, float, float, float] | None = None,
             encrypt_user_password: str | None = None) -> bytes:
    """A valid PDF with ``pages`` blank pages (white) of width x height points.
    ``rect`` = (x, y, w, h) in PDF user space (origin bottom-left) is filled black
    on every page. ``encrypt_user_password`` makes it require that password."""
    objs: list[bytes] = [b"<< /Type /Catalog /Pages 2 0 R >>", b""]
    kids = []
    content = (b"0 0 0 rg %g %g %g %g re f" % rect) if rect else b"0.9 g 0 0 0 0 re f"
    for i in range(pages):
        page_id = 3 + 2 * i
        kids.append(b"%d 0 R" % page_id)
        objs.append(b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %d %d] /Contents %d 0 R >>"
                    % (width, height, page_id + 1))
        objs.append(b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream")
    objs[1] = b"<< /Type /Pages /Kids [%s] /Count %d >>" % (b" ".join(kids), pages)
    extra = b""
    if encrypt_user_password is not None:
        doc_id = b"0123456789abcdef"
        pw = (encrypt_user_password.encode() + _PAD)[:32]
        o_key = hashlib.md5((b"owner" + _PAD)[:32]).digest()[:5]
        o = _rc4(o_key, pw)
        p = -4
        key = hashlib.md5(pw + o + struct.pack("<i", p) + doc_id).digest()[:5]
        u = _rc4(key, _PAD)
        objs.append(b"<< /Filter /Standard /V 1 /R 2 /O <%s> /U <%s> /P %d >>"
                    % (o.hex().encode(), u.hex().encode(), p))
        extra = b"/Encrypt %d 0 R /ID [<%s> <%s>] " % (len(objs), doc_id.hex().encode(), doc_id.hex().encode())
    return _assemble(objs, extra)


def make_zero_page_pdf() -> bytes:
    return _assemble([b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [] /Count 0 >>"])


class DecodedPng:
    def __init__(self, width: int, height: int, rows: list[bytes]) -> None:
        self.width, self.height, self.rows = width, height, rows

    def pixel(self, x: int, y: int) -> tuple[int, int, int]:
        r = self.rows[y]
        return (r[3 * x], r[3 * x + 1], r[3 * x + 2])


def decode_png(data: bytes) -> DecodedPng:
    """Strict decoder for 8-bit RGB, non-interlaced, filter-0 PNGs (what we emit)."""
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos, idat, ihdr = 8, b"", None
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        tag = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + length]
        (crc,) = struct.unpack(">I", data[pos + 8 + length:pos + 12 + length])
        assert crc == zlib.crc32(tag + body) & 0xFFFFFFFF, "bad chunk CRC"
        if tag == b"IHDR":
            ihdr = struct.unpack(">IIBBBBB", body)
        elif tag == b"IDAT":
            idat += body
        pos += 12 + length
    assert ihdr is not None
    w, h, depth, ctype, comp, filt, interlace = ihdr
    assert (depth, ctype, comp, filt, interlace) == (8, 2, 0, 0, 0)
    raw = zlib.decompress(idat)
    stride = 1 + 3 * w
    assert len(raw) == stride * h
    rows = []
    for y in range(h):
        assert raw[y * stride] == 0, "only filter 0 expected"
        rows.append(raw[y * stride + 1:(y + 1) * stride])
    return DecodedPng(w, h, rows)


def unenforced_limits_off_linux(monkeypatch, tmp_path):
    """Where the OS cannot enforce RLIMIT_AS (macOS) the renderer refuses to run unless an operator sets the UNSAFE
    override. Tests that need a REAL render there set it; on Linux the full limits are always enforced. Also keeps the
    render-slot lock files (under data_dir) out of the working directory."""
    import sys

    from app.config import settings
    monkeypatch.setattr(settings, "pageimage_unsafe_allow_unenforced_limits", sys.platform != "linux")
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")


def valid_report(**over) -> dict:
    """The attestation a trustworthy worker writes to stderr once every limit is set and confirmed (protocol v1)."""
    from app.pageimage import runner
    entries = [{"name": n, "requested": req, "soft": req, "hard": req + slack, "state": "enforced"}
               for n, (req, slack) in runner.expected_limits().items()]
    body = {"v": runner.REPORT_VERSION, "ok": True, "limits": entries}
    body.update(over)
    return {"pageimage_limits": body}


def report_snippet(report: dict | None = None) -> str:
    """Python source for a fake worker that writes ``report`` (default: a valid one) as one stderr line."""
    import json
    return f"import sys as _s; _s.stderr.write({json.dumps(report or valid_report())!r} + chr(10)); "


REPORT_OK = report_snippet()
