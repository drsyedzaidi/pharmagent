"""Page-image rendering, server side: validated arguments in, a PNG and its geometry out.

The render itself (render_child.render_region) runs in the parse sandbox's killable child process, never
in the server: a hostile PDF is parsed by pdfium only there, under a wall-clock kill (RENDER_WALL_S), an RSS cap
(RENDER_MAX_RSS) and a MINIMAL environment (no API keys or tokens). Renders take their own slot pool
(RENDER_SLOTS, waiting at most RENDER_QUEUE_WAIT_S -> RenderBusyError, HTTP 503): a render storm cannot
refuse uploads and a long parse cannot starve the reviewer. The parent never decodes the PNG: it checks
the signature, the IHDR size against the reply and the byte cap, and hashes it.

The only inputs are a page number, a zoom level from a fixed table and boxes that come from the layout
sidecar (finite numbers, at most MAX_BOXES): a client cannot ask for a dpi, a crop or a path.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import math
import os
import struct
import threading
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

from app.evidence import render_child
from app.evidence.models import IngestError
from app.evidence.parse_sandbox import ParseBusyError, run_child
from app.evidence.regions import Region

RENDER_TARGET = "app.evidence.render_child:render_region"
RENDER_WALL_S = 20.0                    # measured typical 0.22 s including the interpreter spawn
RENDER_MAX_RSS = 384 * 2**20            # measured ~55 MB
RENDER_QUEUE_WAIT_S = 8.0
RENDER_RETRY_AFTER_S = 3
RENDER_SLOTS = max(1, int(os.environ.get("EVIDENCE_RENDER_SLOTS", "2") or 2))
ZOOMS = (1, 2)
ZOOM_SCALE = {1: 2.0, 2: 3.0}           # 144 dpi (the ink check's own scale) and 216 dpi
WHOLE_PAGE_SCALE = {1: 1.25, 2: 2.0}    # 90 dpi / 144 dpi when the value could not be located
POLICY_VERSION = "crop1"
MAX_COORD = 1e7
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_render_slots = threading.BoundedSemaphore(RENDER_SLOTS)


class RenderFailedError(IngestError):
    """The page could not be rendered (child killed or failed, page too large, PNG too large)."""

    def __init__(self, message: str = "the page could not be rendered", code: str = "render_failed") -> None:
        super().__init__(message)
        self.code = code


class RenderBusyError(ParseBusyError):
    """No render slot within the bounded wait (HTTP 503 + Retry-After); nothing was rendered."""


@dataclass(frozen=True)
class RenderResult:
    png: bytes
    meta: dict[str, Any]


def render_version() -> str:
    """Everything that changes the pixels or the crop: the crop policy, render flags, encoder settings
    and the pdfium / Pillow versions (part of every cache key)."""
    def ver(name: str) -> str:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return "?"
    return (f"{POLICY_VERSION}|forms+annots|png{render_child.PNG_COMPRESS_LEVEL}|pdfium{ver('pypdfium2')}"
            f"|pillow{ver('pillow')}|{render_child.PAD_X}x{render_child.PAD_Y}|{render_child.MIN_W}x{render_child.MIN_H}"
            f"|{render_child.CROP_MAX_FRACTION}|{render_child.MAX_OUT_PIXELS}|{render_child.MAX_SIDE}")


def _finite_box(box: Any) -> list[float]:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise RenderFailedError()
    vals = [float(v) for v in box]
    if not all(math.isfinite(v) and abs(v) <= MAX_COORD for v in vals):
        raise RenderFailedError()
    return vals


def render_args(page: int, zoom: int, region: Region) -> dict[str, Any]:
    """The child's keyword arguments for *region* (validated; raises RenderFailedError on a bad box)."""
    if zoom not in ZOOMS:
        raise ValueError("zoom must be 1 or 2")
    whole = region.precision == "page" or not region.value_boxes
    return {"page": int(page), "scale": (WHOLE_PAGE_SCALE if whole else ZOOM_SCALE)[zoom],
            "value_boxes": [] if whole else [_finite_box(b) for b in region.value_boxes],
            "snippet_box": None if whole or region.snippet_box is None else _finite_box(region.snippet_box),
            "context_boxes": [] if whole else [_finite_box(b) for b in region.context_boxes],
            "whole_page": whole}


def _png_size(png: bytes) -> tuple[int, int]:
    if len(png) < 33 or not png.startswith(PNG_MAGIC) or png[12:16] != b"IHDR":
        raise RenderFailedError()
    return struct.unpack(">II", png[16:24])


def _checked(reply: Any) -> RenderResult:
    """The child's reply as a RenderResult; anything off (error code, bad PNG, size disagreement) fails."""
    if not isinstance(reply, dict):
        raise RenderFailedError()
    if reply.get("error") == "page_too_large":
        raise RenderFailedError("the page is too large to render at a readable resolution", "page_too_large")
    if "error" in reply or not isinstance(reply.get("png_b64"), str):
        raise RenderFailedError(code=str(reply.get("error") or "render_failed")[:40])
    try:
        png = base64.b64decode(reply["png_b64"], validate=True)
    except (binascii.Error, ValueError):
        raise RenderFailedError() from None
    width, height = _png_size(png)
    if len(png) > render_child.PNG_MAX_BYTES or (width, height) != (reply.get("width"), reply.get("height")):
        raise RenderFailedError()
    meta = {k: reply[k] for k in ("width", "height", "scale", "dpi", "crop_px", "crop_pt", "rects_px",
                                  "snippet_px", "page_px")}
    return RenderResult(png, {**meta, "png_sha256": hashlib.sha256(png).hexdigest(), "png_bytes": len(png)})


def render_page(pdf_path: str | Path, page: int, zoom: int, region: Region) -> RenderResult:
    """Render *page* of *pdf_path* around *region* in the sandboxed child (module docstring)."""
    args = render_args(page, zoom, region)
    if not _render_slots.acquire(timeout=RENDER_QUEUE_WAIT_S):
        raise RenderBusyError("the page renderer is busy; try again shortly", RENDER_RETRY_AFTER_S)
    try:
        reply = run_child(pdf_path, target=RENDER_TARGET, wall_s=RENDER_WALL_S, max_rss=RENDER_MAX_RSS,
                          args=args, scrub_env=True)
    except IngestError as exc:                      # killed at a limit, child died, child raised
        raise RenderFailedError() from exc
    except (OSError, ValueError) as exc:
        raise RenderFailedError() from exc
    finally:
        _render_slots.release()
    return _checked(reply)
