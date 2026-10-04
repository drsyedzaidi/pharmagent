"""Page image of a record's cited value: ``GET .../records/{id}/image/meta`` and ``GET .../records/{id}/image``.

The reviewer compares the extracted value with the PIXELS of the cited page. The meta call does the work
(authorise, re-hash the PDF, resolve where the value sits on its page, render in the sandbox or hit
the cache) and describes the result: outline rectangles in image pixels, the precision of the match
and server-written notices. The image call serves the cached PNG the meta described; its ``sha``
parameter must equal that image's sha256 (409 otherwise), so an outline computed for one raster is never
drawn on another. Nothing a client sends influences geometry: zoom is 1 or 2 and every other
query parameter is ignored. The PNG is un-annotated pixels; the highlight is the client's overlay.

Both handlers are async and run their blocking part under the route's own limiter
(render_limits.run_blocking), never the shared threadpool. Reading is not a ledger event: no audit
entry, no mutation.
"""
from __future__ import annotations

import json
import threading
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse, Response

from app.evidence import ingest, regions, relayout, render, render_cache, render_limits
from app.evidence.models import (
    FLAG_GLYPH_INK,
    FLAG_GLYPH_MISMATCH,
    FLAG_GLYPH_UNVERIFIED,
    FLAG_HIDDEN_TEXT,
    FLAG_OVERLAPPING_GLYPHS,
    Record,
    Source,
)
from app.evidence.path_guard import OutsideDataDirError
from app.evidence.routes import http_error, owned_record_id
from app.evidence.store import get_store

URGENT_FLAGS = (FLAG_GLYPH_MISMATCH, FLAG_GLYPH_INK, FLAG_GLYPH_UNVERIFIED, FLAG_OVERLAPPING_GLYPHS,
                FLAG_HIDDEN_TEXT)
MAX_HIGHLIGHTS = 12
LAYOUT_LRU = 8
NO_STORE = "private, no-store"
NOTICE_DISAGREE = "Text layer and page may disagree: compare the image with the extracted value before approving."
NOTICE_UNCHECKED = ("This page could not be checked for hidden text. Compare the image with the extracted "
                    "value before approving.")
NOTICE_NOT_LOCATED = ("Could not locate the cited value on the page. Showing the whole page at lower "
                      "resolution: find it yourself.")
NOTICE_LINE_ONLY = "Only the cited row is outlined; the exact value was not located."
REASON_TEXT = {
    "layout_unavailable": "This source was parsed before page positions were stored.",
    "layout_stale": "The stored page positions do not match the page text.",
    "span_stale": "The record's cited text could not be matched to the page again.",
    "no_span": "The record's cited text was not found on the page when it was checked.",
    "no_boxes": "The glyph positions of this page could not be trusted.",
    "word_unmapped": "The cited words could not be mapped to page positions.",
    "cell_unmapped": "The cited table cell could not be mapped to page positions.",
}
_layouts: OrderedDict[tuple[str, int, int], dict[str, Any] | None] = OrderedDict()
_layouts_lock = threading.Lock()


class _Context:
    """Everything one request resolved before it rendered or looked in the cache."""

    def __init__(self, rec: Record, src: Source, doc: dict[str, Any], n: int, region: regions.Region,
                 layout: dict[str, Any] | None, zoom: int, data_dir: str, pdf_path: Path) -> None:
        self.rec, self.src, self.doc, self.n, self.region = rec, src, doc, n, region
        self.layout, self.zoom, self.data_dir = layout, zoom, data_dir
        self.page_doc = doc["pages"][n - 1]
        self.args = render.render_args(n, zoom, region)
        self.key = render_cache.cache_key(src.sha256, self.args, render.render_version())
        self.pdf_path = pdf_path                  # resolved and confined by _resolve before anything was read


def _zoom(raw: str) -> int:
    if raw not in ("1", "2"):
        raise HTTPException(400, "zoom must be 1 or 2")
    return int(raw)


def _layout_doc(src: Source, parser: Any) -> dict[str, Any] | None:
    """The source's layout sidecar when it belongs to this PDF and parse (LRU by path, mtime, size)."""
    path = Path(src.parse_path).with_name(f"{src.sha256}.layout.json")
    try:
        st = path.stat()
    except OSError:
        return None
    key = (str(path), st.st_mtime_ns, st.st_size)
    with _layouts_lock:
        if key in _layouts:
            _layouts.move_to_end(key)
            doc = _layouts[key]
        else:
            doc = None
            try:
                loaded = json.loads(path.read_text("utf-8"))
                doc = loaded if isinstance(loaded, dict) else None
            except (OSError, ValueError):
                doc = None
            _layouts[key] = doc
            while len(_layouts) > LAYOUT_LRU:
                _layouts.popitem(last=False)
    return doc if relayout.layout_binds(doc, src.sha256, parser) else None


def _layout_page(layout: dict[str, Any] | None, n: int) -> dict[str, Any] | None:
    pages = layout.get("pages") if layout else None
    if not isinstance(pages, list) or n > len(pages) or not isinstance(pages[n - 1], dict):
        return None
    page = pages[n - 1]
    return page if page.get("n") == n else None


def _resolve(sid: str, record_id: str, zoom: int, data_dir: str) -> _Context:
    owned_record_id(sid, record_id)
    store = get_store()
    rec = store.get_record(record_id)
    src = store.get_source(rec.source_id) if rec is not None else None
    if rec is None or src is None or src.session_id != sid:
        raise HTTPException(404, f"unknown record: {record_id}")
    try:
        pdf_path, _ = ingest.stored_paths(src, data_dir)       # confinement first: nothing is hashed or read yet
        doc = ingest.load_pages(src, data_dir)
    except OutsideDataDirError as exc:
        raise HTTPException(409, str(exc)) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(409, "source file missing or changed on disk") from exc
    pages = doc.get("pages") or []
    n = rec.locator.get("page")
    if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= len(pages):
        raise HTTPException(422, "record cites a page that does not exist")
    layout = _layout_doc(src, doc.get("parser"))
    region = regions.locate(rec.locator, pages[n - 1], _layout_page(layout, n))
    return _Context(rec, src, doc, n, region, layout, zoom, data_dir, pdf_path)


def _entry(ctx: _Context, sid: str) -> tuple[bytes, dict[str, Any]]:
    """(png, meta) from the cache, or rendered once (single flight) in the sandbox and cached."""
    hit = render_cache.get(ctx.key, ctx.data_dir)
    if hit is not None:
        return hit
    with render_cache.single_flight(ctx.key):
        hit = render_cache.get(ctx.key, ctx.data_dir)
        if hit is not None:
            return hit
        render_limits.admit(sid)
        with render_limits.in_flight():
            out = render.render_page(ctx.pdf_path, ctx.n, ctx.zoom, ctx.region)
        render_cache.put(ctx.key, out.png, out.meta, ctx.data_dir)
        return out.png, out.meta


def _notices(ctx: _Context, urgent: list[str]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    if urgent:
        out.append({"code": "text_layer_may_disagree", "severity": "error", "text": NOTICE_DISAGREE})
    if ctx.page_doc.get("ink_check") == "unverified":
        out.append({"code": "page_unchecked", "severity": "warn", "text": NOTICE_UNCHECKED})
    reg = ctx.region
    if not reg.resolved:
        text = f"{NOTICE_NOT_LOCATED} {REASON_TEXT.get(reg.reason or '', '')}".strip()
        out.append({"code": "value_not_located", "severity": "warn", "text": text})
    elif reg.precision == "line" and ctx.rec.locator.get("kind") == "table":
        out.append({"code": "row_only", "severity": "warn", "text": NOTICE_LINE_ONLY})
    return out


def _rect(r: list[float] | None) -> dict[str, float] | None:
    return None if r is None else {"x": r[0], "y": r[1], "w": r[2], "h": r[3]}


def _meta(ctx: _Context, entry: dict[str, Any]) -> dict[str, Any]:
    rec, reg = ctx.rec, ctx.region
    urgent = [f.get("code") for f in rec.flags if isinstance(f, dict) and f.get("code") in URGENT_FLAGS]
    urgent = list(dict.fromkeys(urgent))
    page_checks = {"ink_check": ctx.page_doc.get("ink_check"), "hidden_chars": ctx.page_doc.get("hidden_chars"),
                   "glyph_check": sorted((ctx.page_doc.get("glyph_check") or {}).keys())}
    unverified = ctx.page_doc.get("ink_check") == "unverified"
    located = reg.resolved
    layout = relayout.layout_info(ctx.src)
    return {
        "record_id": rec.id, "source_id": ctx.src.id, "page": ctx.n, "pages": len(ctx.doc.get("pages") or []),
        "zoom": ctx.zoom, "dpi": entry["dpi"], "resolved": located, "precision": reg.precision,
        "reason": reg.reason, "width": entry["width"], "height": entry["height"],
        "highlight": [_rect(r) for r in (entry.get("rects_px") or [])[:MAX_HIGHLIGHTS]] if located else [],
        "snippet_box": _rect(entry.get("snippet_px")) if located else None,
        "crop_pt": entry["crop_pt"], "image_sha256": entry["png_sha256"], "png_bytes": entry["png_bytes"],
        "pdf_sha256": ctx.src.sha256,
        "layout_status": layout["layout_status"],
        "layout_rebuild_available": reg.reason == "layout_unavailable" and layout["layout_status"] == "missing",
        "layout_version": (ctx.layout or {}).get("layout_version") if located else None,
        "extracted": {"value": rec.value, "unit": rec.unit, "literal": rec.locator.get("literal"),
                      "kind": rec.locator.get("kind") or "text"},
        "page_checks": page_checks, "flags": urgent,
        "compare_urgent": bool(urgent) or unverified or not located,
        "notices": _notices(ctx, urgent),
    }


def _guarded(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Map any failure to the evidence error map (HTTPException passes through unchanged)."""
    def run(*args: Any) -> Any:
        try:
            return fn(*args)
        except HTTPException:
            raise
        except Exception as exc:
            raise http_error(exc) from exc
    return run


@_guarded
def _meta_job(sid: str, record_id: str, zoom: int, data_dir: str) -> dict[str, Any]:
    ctx = _resolve(sid, record_id, zoom, data_dir)
    _, entry = _entry(ctx, sid)
    return _meta(ctx, entry)


@_guarded
def _image_job(sid: str, record_id: str, zoom: int, sha: str | None, data_dir: str) -> tuple[bytes, str]:
    ctx = _resolve(sid, record_id, zoom, data_dir)
    png, entry = _entry(ctx, sid)
    if not sha or sha != entry["png_sha256"]:
        raise HTTPException(409, "page image changed; reload")
    return png, entry["png_sha256"]


def add_image_routes(router: APIRouter, owned_session: Callable[..., Any]) -> None:
    """Mount the two page-image GET routes on the evidence router (same prefix and auth)."""

    @router.get("/records/{record_id}/image/meta")
    async def image_meta(sid: str, record_id: str, zoom: str = Query("1"), sess=Depends(owned_session)) -> Response:
        body = await render_limits.run_blocking(_meta_job, sid, record_id, _zoom(zoom), str(sess.ctx.data_dir))
        return JSONResponse(body, headers={"Cache-Control": NO_STORE, "Vary": "Authorization"})

    @router.get("/records/{record_id}/image")
    async def image(sid: str, record_id: str, zoom: str = Query("1"), sha: str | None = Query(None),
                    sess=Depends(owned_session)) -> Response:
        png, digest = await render_limits.run_blocking(_image_job, sid, record_id, _zoom(zoom), sha,
                                                       str(sess.ctx.data_dir))
        return Response(png, media_type="image/png", headers={
            "Content-Length": str(len(png)), "X-Content-Type-Options": "nosniff", "Cache-Control": NO_STORE,
            "Vary": "Authorization", "Content-Disposition": 'inline; filename="record.png"',
            "Content-Security-Policy": "default-src 'none'; sandbox", "Cross-Origin-Resource-Policy": "same-origin",
            "X-Image-SHA256": digest})
