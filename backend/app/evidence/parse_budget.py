"""Glyph budget: a pre-layout upper bound on the glyphs each page asks the interpreter to lay out.

pdfminer's layout cost (time and memory) is linear in glyphs, and a few KB of Flate content can
inflate to hundreds of thousands of glyphs (review PoC: 600k glyphs in a 16.7 KB file took 19 s and
1.36 GB). Before any page is interpreted, every page's decompressed content streams are scanned at C
/ numpy speed and the page is refused (ParseLimitError) above MAX_GLYPHS_PER_PAGE, the document above
MAX_GLYPHS_PER_DOC.

The count is an UPPER bound, so it can only refuse too much, never too little:
- inline image data (``ID ... EI``) is scanned like the rest: skipping it by pattern could be faked
  inside strings, so its stray parentheses can only over-count (inline images are small by spec);
- every byte inside a literal string at any nesting depth (escapes included) and every two hex digits
  of a hex string count as one glyph; 2-byte CID codes therefore count twice;
- every text-show operator (Tj, TJ, ', ") counts one more, so millions of empty ``() Tj`` still cost;
- string depth is tracked with a floor at zero (Lindley recursion), so a stray ')' in a comment
  cannot drive the depth negative and hide later strings; a stray '(' only over-counts;
- each ``/Name Do`` of a Form XObject (comments and '#xx' name escapes allowed) adds that form's
  own count (recursively, memoised, nesting depth capped), so one small form drawn thousands of
  times is charged for every draw.
A stream that cannot be decoded is charged its raw length.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any

import numpy as np
from pdfminer.pdftypes import PDFStream, resolve1

from app.evidence.parse_guard import ParseLimitError

MAX_GLYPHS_PER_PAGE = 200_000
MAX_GLYPHS_PER_DOC = 2_000_000
MAX_FORM_DEPTH = 8
_CHUNK = 1 << 22                     # 4 MB per numpy pass (bounded scratch memory)
_ESCAPE = re.compile(rb"\\.", re.S)
_HEX = re.compile(rb"<[0-9A-Fa-f\s]*>")
_DO = re.compile(rb"/([^\s/\[\]()<>{}%]+)(?:\s|%[^\r\n]*)*Do\b")   # comments may separate them
_NAME_ESCAPE = re.compile(rb"#([0-9A-Fa-f]{2})")
_SHOW_OPS = (b"Tj", b"TJ", b"'", b'"')


def _string_bytes(data: bytes) -> int:
    """Bytes at literal-string depth >= 1 (an upper bound: nested and closing parentheses count too)."""
    total, depth = 0, 0
    for start in range(0, len(data), _CHUNK):
        arr = np.frombuffer(data, dtype=np.uint8, count=min(_CHUNK, len(data) - start), offset=start)
        step = (arr == 0x28).astype(np.int32) - (arr == 0x29).astype(np.int32)
        run = np.cumsum(step)
        # Lindley: depth_i = max(0, depth_{i-1} + step_i) = run_i - min(-depth0, min_{j<=i} run_j)
        level = run - np.minimum(np.minimum.accumulate(run), -depth)
        before = np.concatenate(([depth], level[:-1]))
        total += int(np.count_nonzero(before > 0))     # nested parens and the closing one included
        depth = int(level[-1])
    return total


def content_glyphs(data: bytes) -> int:
    """Upper bound on the glyphs one decoded content stream shows (Form XObject draws not included)."""
    flat = _ESCAPE.sub(b"x", data)                     # an escape is one glyph and never a delimiter
    hex_digits = sum(len(m) - 2 for m in _HEX.findall(flat))
    shows = sum(flat.count(op) for op in _SHOW_OPS)
    return _string_bytes(flat) + (hex_digits + 1) // 2 + shows


def _decoded(stream: Any) -> bytes:
    try:
        return stream.get_data() or b""
    except Exception:                                  # undecodable: charge the raw size (fail closed)
        return b"x" * len(getattr(stream, "rawdata", b"") or b"")


def _name(raw: bytes) -> str:
    """A content-stream name as pdfminer resolves it ('#xx' escapes decoded)."""
    return _NAME_ESCAPE.sub(lambda m: bytes([int(m.group(1), 16)]), raw).decode("latin-1")


def _forms(resources: Any) -> dict[str, Any]:
    res = resolve1(resources) if resources is not None else None
    xobjs = resolve1(res.get("XObject")) if isinstance(res, dict) else None
    return xobjs if isinstance(xobjs, dict) else {}


class _Counter:
    def __init__(self) -> None:
        self.memo: dict[tuple[int, int], int] = {}

    def streams(self, streams: list[Any], resources: Any, depth: int) -> int:
        # one buffer, as the interpreter sees it: a string may open in one stream and close in the next
        data = b"\n".join(_decoded(x) for x in map(resolve1, streams) if isinstance(x, PDFStream))
        n = content_glyphs(data)
        draws = Counter(_name(m) for m in _DO.findall(data))
        forms = _forms(resources)
        for name, times in draws.items():
            form = resolve1(forms.get(name))
            if isinstance(form, PDFStream) and getattr(resolve1(form.attrs.get("Subtype")), "name", "") == "Form":
                n += times * self.form(form, resources, depth + 1)
                if n > MAX_GLYPHS_PER_DOC:             # no need to finish an already-refused sum
                    return n
        return n

    def form(self, form: PDFStream, parent_resources: Any, depth: int) -> int:
        if depth > MAX_FORM_DEPTH:
            raise ParseLimitError(f"form XObjects nest deeper than {MAX_FORM_DEPTH} levels")
        resources = form.attrs.get("Resources") or parent_resources
        key = (id(form), id(resources))
        if key not in self.memo:
            self.memo[key] = self.streams([form], resources, depth)
        return self.memo[key]


def check_glyph_budget(pages: list[Any]) -> None:
    """Refuse the document before layout when any page, or all pages together, exceed the glyph caps."""
    counter, total = _Counter(), 0
    for page in pages:
        obj = page.page_obj
        n = counter.streams(list(obj.contents or []), obj.resources, 0)
        if n > MAX_GLYPHS_PER_PAGE:
            raise ParseLimitError(f"page {page.page_number} shows about {n} glyphs "
                                  f"(limit {MAX_GLYPHS_PER_PAGE} per page)")
        total += n
        if total > MAX_GLYPHS_PER_DOC:
            raise ParseLimitError(f"document shows more than {MAX_GLYPHS_PER_DOC} glyphs (limit per document)")
