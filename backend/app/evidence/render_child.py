"""Child-process target of the page-image renderer: one page of one PDF -> a cropped, un-annotated PNG.

``render_region(path, page=, scale=, value_boxes=, snippet_box=, context_boxes=, whole_page=)`` runs ONLY
inside the parse sandbox's killable child (render.py starts it); pdfium never parses a hostile PDF
in the server process. It renders the page exactly like the ink check does (the document's forms
initialised first, ``may_draw_forms`` and ``draw_annots`` on: the raster the visibility verdicts
were computed from), maps the PDF-user-space boxes to pixels with the same affine the ink check fits
(rotation, CropBox and MediaBox origin are pdfium's business), applies the crop policy and encodes
the pixels it rendered itself: nothing from the PDF is echoed. The PNG carries NO highlight: rects
come back beside it, so the reviewer sees untouched pixels.

Crop policy (display axes, in points): the snippet's words (or the value's when the snippet is over
CROP_MAX_FRACTION of the page) united with the context boxes, padded PAD_X x PAD_Y, at least MIN_W x
MIN_H centred on the value and moved back inside the page; a crop over CROP_MAX_FRACTION of the page
is the whole page. Output is kept to MAX_OUT_PIXELS / MAX_SIDE by lowering the scale in 0.75 steps; a
page that cannot be shown at 72 dpi (or whose PNG stays over PNG_MAX_BYTES after two such steps) is
refused with an error code, never degraded silently.
"""
from __future__ import annotations

import base64
import io
import math
from typing import Any

import numpy as np

from app.evidence.parse_ink import MAX_RENDER_PIXELS, _device_affine, _open_document

PAD_X, PAD_Y = 60.0, 48.0
MIN_W, MIN_H = 240.0, 120.0
CROP_MAX_FRACTION = 0.55
MAX_OUT_PIXELS = 4_000_000
MAX_SIDE = 3200
MIN_SCALE = 1.0                        # 72 dpi
SCALE_STEP = 0.75
PNG_MAX_BYTES = 2 * 2**20
PNG_RETRIES = 2
PNG_COMPRESS_LEVEL = 6
MAX_BOXES = 40
Rect = tuple[float, float, float, float]


def _to_px(boxes: list[list[float]], affine: np.ndarray) -> list[Rect]:
    """User-space [l, b, r, t] boxes -> display pixel rects (x0, y0, x1, y1) of the render."""
    out: list[Rect] = []
    for l, b, r, t in boxes:
        pts = np.array([[l, b], [r, t], [l, t], [r, b]], float) @ affine[:, :2].T + affine[:, 2]
        out.append((float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())))
    return out


def _union(rects: list[Rect]) -> Rect:
    return (min(r[0] for r in rects), min(r[1] for r in rects), max(r[2] for r in rects),
            max(r[3] for r in rects))


def _area(r: Rect) -> float:
    return max(r[2] - r[0], 0.0) * max(r[3] - r[1], 0.0)


def _fit(lo: float, hi: float, size: float, limit: float) -> tuple[float, float]:
    """[lo, hi] widened to at least *size* around its centre, then shifted back inside [0, limit]."""
    if hi - lo < size:
        mid = (lo + hi) / 2
        lo, hi = mid - size / 2, mid + size / 2
    if hi - lo >= limit:
        return 0.0, limit
    shift = max(-lo, 0.0) - max(hi - limit, 0.0)
    return lo + shift, hi + shift


def crop_rect(page: tuple[float, float], scale: float, value: list[Rect], snippet: Rect | None,
              context: list[Rect], whole_page: bool) -> Rect:
    """The crop in pixels of a render at *scale* of a page of *page* pixels (module docstring)."""
    full = (0.0, 0.0, page[0], page[1])
    if whole_page or not value:
        return full
    base = snippet if snippet is not None and _area(snippet) <= CROP_MAX_FRACTION * _area(full) else _union(value)
    base = _union([base, *context])
    v = _union(value)
    x0, x1 = _fit(base[0] - PAD_X * scale, base[2] + PAD_X * scale, MIN_W * scale, page[0])
    y0, y1 = _fit(base[1] - PAD_Y * scale, base[3] + PAD_Y * scale, MIN_H * scale, page[1])
    if (v[0] + v[2]) / 2 < x0 or (v[0] + v[2]) / 2 > x1:        # the value must stay inside the crop
        return full
    crop = (x0, y0, x1, y1)
    return full if _area(crop) > CROP_MAX_FRACTION * _area(full) else crop


def _encode(rgb: np.ndarray) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb)).save(buf, "PNG", compress_level=PNG_COMPRESS_LEVEL)
    return buf.getvalue()


def _attempt(pdf_page: Any, scale: float, value_boxes: list[list[float]], snippet_box: list[float] | None,
             context_boxes: list[list[float]], whole_page: bool) -> dict[str, Any] | None:
    """One render at *scale*; None when the crop is over the pixel caps (the caller lowers the scale)."""
    bitmap = pdf_page.render(scale=scale, may_draw_forms=True, draw_annots=True)
    arr = bitmap.to_numpy()
    h, w = arr.shape[:2]
    affine = _device_affine(bitmap.get_posconv(pdf_page))
    value = _to_px(value_boxes, affine)
    snippet = _to_px([snippet_box], affine)[0] if snippet_box else None
    crop = crop_rect((float(w), float(h)), scale, value, snippet, _to_px(context_boxes, affine), whole_page)
    x0, y0 = max(int(math.floor(crop[0])), 0), max(int(math.floor(crop[1])), 0)
    x1, y1 = min(int(math.ceil(crop[2])), w), min(int(math.ceil(crop[3])), h)
    cw, ch = x1 - x0, y1 - y0
    if cw <= 0 or ch <= 0 or cw * ch > MAX_OUT_PIXELS or max(cw, ch) > MAX_SIDE:
        return None
    rgb = arr[y0:y1, x0:x1, :3][:, :, ::-1]                      # pdfium paints BGR; alpha is dropped
    inv = np.linalg.inv(np.vstack([affine, [0.0, 0.0, 1.0]]))
    corners = np.array([[x0, y0, 1.0], [x1, y1, 1.0], [x0, y1, 1.0], [x1, y0, 1.0]]) @ inv.T
    rects = [[round(r[0] - x0, 1), round(r[1] - y0, 1), round(r[2] - r[0], 1), round(r[3] - r[1], 1)] for r in value]
    snip = None if snippet is None else [round(snippet[0] - x0, 1), round(snippet[1] - y0, 1),
                                         round(snippet[2] - snippet[0], 1), round(snippet[3] - snippet[1], 1)]
    return {"rgb": rgb, "width": cw, "height": ch, "scale": scale, "crop_px": [x0, y0, x1, y1],
            "crop_pt": [round(float(corners[:, 0].min()), 1), round(float(corners[:, 1].min()), 1),
                        round(float(corners[:, 0].max()), 1), round(float(corners[:, 1].max()), 1)],
            "rects_px": rects, "snippet_px": snip, "page_px": [w, h]}


def render_region(path: str, *, page: int, scale: float, value_boxes: list[list[float]] | None = None,
                  snippet_box: list[float] | None = None, context_boxes: list[list[float]] | None = None,
                  whole_page: bool = False) -> dict[str, Any]:
    """The JSON reply of the render child: {png_b64, width, height, scale, dpi, crop_px, crop_pt,
    rects_px, snippet_px, page_px} or {error: 'page_too_large' | 'png_too_large'}."""
    value_boxes, context_boxes = list(value_boxes or []), list(context_boxes or [])
    if len(value_boxes) + len(context_boxes) > MAX_BOXES:
        return {"error": "too_many_boxes"}
    doc = _open_document(path)
    try:
        pdf_page = doc[page - 1]
        try:
            width, height = pdf_page.get_size()
            scale = min(float(scale), math.sqrt(MAX_RENDER_PIXELS / max(width * height, 1.0)))
            png_tries = 0
            while scale >= MIN_SCALE:
                got = _attempt(pdf_page, scale, value_boxes, snippet_box, context_boxes, whole_page)
                if got is None:
                    scale *= SCALE_STEP
                    continue
                png = _encode(got.pop("rgb"))
                if len(png) > PNG_MAX_BYTES:
                    if png_tries >= PNG_RETRIES or scale * SCALE_STEP < MIN_SCALE:
                        return {"error": "png_too_large"}
                    png_tries, scale = png_tries + 1, scale * SCALE_STEP
                    continue
                return {**got, "dpi": round(got["scale"] * 72, 1), "png_b64": base64.b64encode(png).decode("ascii")}
            return {"error": "page_too_large"}
        finally:
            pdf_page.close()
    finally:
        doc.close()
