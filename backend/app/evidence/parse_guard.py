"""Parser guards: resource caps that run before / during parse, and visibility filtering.

Resource caps (upload DoS): page boxes larger than MAX_PAGE_PT, Flate streams that inflate past
MAX_STREAM_BYTES (or MAX_TOTAL_INFLATED for the whole file), a wall-clock budget across pages, and
a ruling-line budget per page before table detection (cell finding grows ~n^4 in ruled lines).
Every breach raises ParseLimitError, which ingest records as parse_status 'failed'.

Visibility: the verifier proves 'this literal is in the page text', so the page text must be what
a reader SEES. These cheap colour / geometry rules are the first pass: characters drawn with text
render mode 3 / 7 (invisible / clip only), white glyphs not on a filled non-white box, glyphs
squashed to (near) zero width or height, and glyphs outside the page box or the CropBox are dropped
before extraction and counted per page (hidden_chars). The second pass (parse_ink.py) judges what is
left from rendered ink, so hidden text can drive neither a verdict nor a value.
Thin / no-break spaces inside numbers ('18 900') survive extraction as U+2009 (pdfplumber would
turn them into ASCII spaces, which the verifier treats as a word gap).
"""
from __future__ import annotations

import time
import unicodedata
import zlib
from typing import Any

from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdftypes import PDFStream, resolve1
from pdfplumber.page import PDFPageAggregatorWithMarkedContent

MAX_PAGE_PT = 20_000.0                 # 7 m: no real page; huge boxes drive per-x allocations
MAX_STREAM_BYTES = 64 * 1024 * 1024    # one inflated stream
MAX_TOTAL_INFLATED = 256 * 1024 * 1024
PARSE_BUDGET_S = 120.0                 # whole document, checked between pages
MAX_RULING_PRODUCT = 10_000            # horizontal x vertical ruling edges before find_tables
INVISIBLE_RENDER_MODES = (3, 7)
MIN_WIDTH_EM = 0.04                    # narrowest real glyph ('i' at 50 % Tz) is ~0.1 em wide
MIN_GLYPH_PT = 1.0                     # a glyph box under 1 pt tall cannot be read
THIN_SPACES = frozenset("\u2009\u202f\u00a0\u2007\u2008\u200a")
THIN = "\u2009"
_PLACEHOLDER = "\ue000"                # private use, not whitespace: survives word grouping


class ParseLimitError(ValueError):
    """The PDF exceeds a parser resource cap; nothing about its content is trusted."""


# ------------------------------------------------------------- resource caps ---

def _filters(stream: PDFStream) -> list[str]:
    raw = resolve1(stream.attrs.get("Filter") or stream.attrs.get("F"))
    items = raw if isinstance(raw, list) else [raw] if raw is not None else []
    return [str(getattr(resolve1(f), "name", resolve1(f))) for f in items]


def check_streams(doc: Any) -> None:
    """Inflate every Flate stream with a hard output cap (bounded memory) before any page is laid out."""
    total = 0
    for xref in doc.xrefs:
        for objid in xref.get_objids():
            try:
                obj = doc.getobj(objid)
            except Exception:                       # unreadable object: the parser will report it
                continue
            if not isinstance(obj, PDFStream) or "FlateDecode" not in _filters(obj):
                continue
            inflater = zlib.decompressobj()
            try:
                out = inflater.decompress(obj.rawdata or b"", MAX_STREAM_BYTES)
            except zlib.error:
                continue
            if inflater.unconsumed_tail:
                raise ParseLimitError(f"compressed stream inflates past {MAX_STREAM_BYTES // 2**20} MB")
            total += len(out)
            if total > MAX_TOTAL_INFLATED:
                raise ParseLimitError(f"compressed content inflates past {MAX_TOTAL_INFLATED // 2**20} MB")


def check_page_box(page: Any) -> None:
    if float(page.width) > MAX_PAGE_PT or float(page.height) > MAX_PAGE_PT:
        raise ParseLimitError(f"page {page.page_number} box {page.width:.0f} x {page.height:.0f} pt exceeds "
                              f"{MAX_PAGE_PT:.0f} pt")


class Deadline:
    def __init__(self, budget_s: float | None = None) -> None:
        self.budget = PARSE_BUDGET_S if budget_s is None else budget_s
        self.end = time.monotonic() + self.budget

    def check(self, page_number: int) -> None:
        if time.monotonic() > self.end:
            raise ParseLimitError(f"parse exceeded {self.budget:.0f} s at page {page_number}")


def tables_affordable(page: Any) -> bool:
    """False when the ruling grid is large enough to make cell detection explode."""
    edges = page.edges
    h = sum(1 for e in edges if e.get("orientation") == "h")
    return h * (len(edges) - h) <= MAX_RULING_PRODUCT


# ---------------------------------------------------------------- visibility ---

class _Sink:
    """Stands in for the layout container while one invisible glyph is rendered (a fresh one per glyph,
    so nothing accumulates); pdfplumber's marked-content hook reads ``_objs[-1]``."""

    def __init__(self) -> None:
        self._objs: list[Any] = []

    def add(self, item: Any) -> None:
        self._objs.append(item)


class _VisibleAggregator(PDFPageAggregatorWithMarkedContent):
    """Lays out only glyphs that are painted: render modes 3 / 7 advance the pen but add nothing."""

    hidden = 0
    _invisible = False

    def render_string(self, textstate, seq, ncs, graphicstate):  # noqa: ANN001 - pdfminer signature
        self._invisible = getattr(textstate, "render", 0) in INVISIBLE_RENDER_MODES
        return super().render_string(textstate, seq, ncs, graphicstate)

    def render_char(self, *args, **kwargs):  # noqa: ANN002, ANN003 - pdfminer signature
        if not self._invisible:
            return super().render_char(*args, **kwargs)
        self.hidden += 1
        real, self.cur_item = self.cur_item, _Sink()
        try:
            return super().render_char(*args, **kwargs)
        finally:
            self.cur_item = real


def lay_out_visible(page: Any) -> int:
    """Lay the page out without invisible-mode glyphs (pdfplumber 0.11 reads ``page._layout``);
    returns how many glyphs were dropped."""
    device = _VisibleAggregator(page.pdf.rsrcmgr, pageno=page.page_number, laparams=page.pdf.laparams)
    PDFPageInterpreter(page.pdf.rsrcmgr, device).process_page(page.page_obj)
    page._layout = device.get_result()
    return device.hidden


def _is_white(color: Any) -> bool:
    vals = color if isinstance(color, (list, tuple)) else (color,)
    try:
        nums = [float(v) for v in vals]
    except (TypeError, ValueError):
        return False
    if len(nums) == 4:                                  # CMYK: no ink
        return all(v <= 0.01 for v in nums)
    return bool(nums) and all(v >= 0.99 for v in nums)  # gray / RGB


def _on_dark_box(c: dict[str, Any], boxes: list[dict[str, Any]]) -> bool:
    return any(b["x0"] <= c["x0"] and c["x1"] <= b["x1"] and b["top"] <= c["top"] and c["bottom"] <= b["bottom"]
               for b in boxes)


def _squashed(c: dict[str, Any]) -> bool:
    """Drawn with (near) zero width or height: '1 Tz', a squashing Tm, or a sub-point font."""
    text = str(c.get("text", ""))
    if not text.strip() or unicodedata.category(text[:1]) in ("Mn", "Me"):   # spaces, combining marks
        return False
    width, height = float(c["x1"]) - float(c["x0"]), float(c["bottom"]) - float(c["top"])
    return height < MIN_GLYPH_PT or width < MIN_WIDTH_EM * max(height, float(c.get("size") or 0.0))


def visibility_filter(page: Any):
    """(predicate for page.filter keeping visible chars, count of chars it drops). The visible area is
    the page box cut to the CropBox (what a viewer shows)."""
    x0, top, x1, bottom = page.bbox
    cx0, ctop, cx1, cbottom = page.cropbox
    x0, top, x1, bottom = max(x0, cx0), max(top, ctop), min(x1, cx1), min(bottom, cbottom)
    boxes = [r for r in page.rects if r.get("fill") and not _is_white(r.get("non_stroking_color"))]

    def visible(c: dict[str, Any]) -> bool:
        if c["x1"] <= x0 or c["x0"] >= x1 or c["bottom"] <= top or c["top"] >= bottom or _squashed(c):
            return False
        return not (_is_white(c.get("non_stroking_color")) and not _on_dark_box(c, boxes))

    dropped = sum(1 for c in page.chars if not visible(c))
    return visible, dropped


def protect_thin_spaces(page: Any) -> None:
    """Mark thin / no-break spaces so extraction keeps them inside the number (restore() undoes it).
    Edits pdfplumber's own char dicts for this page only; the page object is discarded after parse."""
    for c in page.chars:
        if c.get("text") in THIN_SPACES:
            c["text"] = _PLACEHOLDER


def restore(value: Any) -> Any:
    """The parse output with every placeholder back as U+2009 (str, list or nested rows)."""
    if isinstance(value, str):
        return value.replace(_PLACEHOLDER, THIN)
    if isinstance(value, list):
        return [restore(v) for v in value]
    if isinstance(value, dict):
        return {k: restore(v) for k, v in value.items()}
    return value
