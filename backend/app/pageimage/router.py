"""HTTP routes for the page-image viewer.

Built by a factory so the existing ownership dependency (``owned_session`` in
``app.main``) is injected unchanged rather than imported (avoids a cycle and
leaves auth code untouched). Rendering is read-only and is NOT audited.
"""
from __future__ import annotations

import logging
import re
import sys
from collections.abc import Callable

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import Response
from fastapi.routing import APIRoute
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.routing import Match

from app.config import settings
from app.pageimage import documents, limits
from app.pageimage.runner import RenderError, probe_pdf, render_page
from app.pageimage.types import NormBBox, RenderParams

log = logging.getLogger("pharmagent.pageimage")

_PAGE_FILE_RE = re.compile(r"^(\d{1,4})\.png$")
_PDF_MAGIC = b"%PDF-"

# RenderError.kind -> (HTTP status, fixed public message). Nothing here may
# contain a path or library error text.
_UPLOAD_ERRORS = {
    "encrypted": (400, "password-protected PDFs are not supported"),
    "malformed": (400, "file is not a readable PDF"),
    "failed": (400, "file is not a readable PDF"),
    "bad_request": (400, "file is not a readable PDF"),
    "too_large": (400, f"PDF has too many pages (max {limits.MAX_PAGES})"),
    "timeout": (400, "PDF took too long to process"),
    "output_too_big": (400, "file is not a readable PDF"),
    "busy": (429, "too many documents being processed; retry shortly"),
    "unavailable": (503, "PDF processing is unavailable on this server"),
    "limit_unavailable": (503, "PDF processing is unavailable on this server"),
    "memory_limit": (400, "PDF is too complex to process within the resource limits"),
    "cpu_limit": (400, "PDF took too long to process"),
}
_RENDER_ERRORS = {
    "no_page": (404, "page not found"),
    "too_large": (400, "requested image is too large; lower the scale or crop a region"),
    "empty": (400, "requested region is empty"),
    "output_too_big": (413, "rendered image exceeds the size limit; lower the scale or crop a region"),
    "timeout": (504, "rendering timed out"),
    "busy": (429, "renderer busy; retry shortly"),
    "unavailable": (503, "PDF rendering is unavailable on this server"),
    "limit_unavailable": (503, "PDF rendering is unavailable on this server"),
    "memory_limit": (413, "page is too complex to render within the resource limits"),
    "cpu_limit": (504, "rendering timed out"),
    "encrypted": (422, "page could not be rendered"),
    "malformed": (422, "page could not be rendered"),
    "failed": (422, "page could not be rendered"),
    "bad_request": (400, "invalid render request"),
}


def _http(table: dict, kind: str) -> HTTPException:
    status, msg = table.get(kind, (422, "page could not be rendered"))
    return HTTPException(status, msg)


def _enabled() -> bool:
    return bool(settings.pageimage_enabled) and not getattr(sys, "frozen", False)


class BodyTooLarge(HTTPException):
    def __init__(self) -> None:
        super().__init__(413, f"request body exceeds {limits.MAX_PDF_BYTES // (1024 * 1024)} MB limit")


def _capped(request: Request) -> Request:
    """The same request with a receive channel that raises once more than MAX_BODY_BYTES have been read. This is what
    stops a client that sends no Content-Length, or understates it, from being buffered without bound."""
    seen = 0
    original = request.receive

    async def receive():
        nonlocal seen
        msg = await original()
        if msg["type"] == "http.request":
            seen += len(msg.get("body", b""))
            if seen > limits.MAX_BODY_BYTES:
                raise BodyTooLarge()
        return msg

    return Request(request.scope, receive, request._send)


class GatedRoute(APIRoute):
    """A route that does not exist unless the feature is on. ``matches`` runs before FastAPI reads the body, so a
    disabled (or frozen) server never parses multipart input or spools it to disk, and the OpenAPI schema omits the
    routes (``include_in_schema`` follows the same switch)."""

    def matches(self, scope):
        if not _enabled():
            return Match.NONE, {}
        return super().matches(scope)

    def get_route_handler(self):
        inner = super().get_route_handler()

        async def handler(request: Request):
            # Before FastAPI reads the body: parsing spools the upload to disk and runs ahead of every dependency
            # (auth and ownership included), so the size gate has to be here, not in the endpoint.
            if request.method in ("POST", "PUT", "PATCH"):
                declared = request.headers.get("content-length")
                if declared is not None:
                    try:
                        n = int(declared)
                    except ValueError:
                        raise HTTPException(400, "invalid Content-Length") from None
                    if n < 0:
                        raise HTTPException(400, "invalid Content-Length")
                    if n > limits.MAX_BODY_BYTES:
                        raise BodyTooLarge()
                request = _capped(request)
            return await inner(request)

        return handler

    @property
    def include_in_schema(self) -> bool:
        return _enabled()          # FastAPI re-reads this when it builds /openapi.json (and passes it on at include)

    @include_in_schema.setter
    def include_in_schema(self, value: bool) -> None:
        pass                       # the switch alone decides; the declared value is irrelevant


def build_documents_router(owned_session: Callable) -> APIRouter:
    router = APIRouter(tags=["documents"], route_class=GatedRoute)

    @router.post("/api/sessions/{sid}/documents", status_code=201)
    async def upload_document(file: UploadFile = File(...), sess=Depends(owned_session)) -> dict:
        try:
            doc_id, part = documents.new_slot(sess.id)
        except documents.DocumentLimitError:
            raise HTTPException(409, f"document limit reached ({limits.MAX_DOCS_PER_SESSION} per session)") from None
        except ValueError:
            raise HTTPException(400, "invalid session") from None
        try:
            size = 0
            with part.open("wb") as out:
                while chunk := await file.read(1 << 20):
                    if size == 0 and not chunk.startswith(_PDF_MAGIC):
                        raise HTTPException(400, "file is not a PDF")
                    size += len(chunk)
                    if size > limits.MAX_PDF_BYTES:
                        raise HTTPException(413, f"file exceeds {limits.MAX_PDF_BYTES // (1024 * 1024)} MB limit")
                    out.write(chunk)
            if size == 0:
                raise HTTPException(400, "file is not a PDF")
            try:
                pages = await run_in_threadpool(probe_pdf, part)
            except RenderError as e:
                raise _http(_UPLOAD_ERRORS, e.kind) from None
            if pages < 1:
                raise HTTPException(400, "PDF has no pages")
            if pages > limits.MAX_PAGES:
                raise _http(_UPLOAD_ERRORS, "too_large")
            meta = documents.commit(sess.id, doc_id, part, filename=file.filename, size=size, pages=pages)
        except BaseException:
            documents.discard(part)
            raise
        return meta.to_dict()

    @router.get("/api/sessions/{sid}/documents")
    def list_documents(sess=Depends(owned_session)) -> dict:
        return {"documents": [m.to_dict() for m in documents.list_documents(sess.id)]}

    @router.get("/api/sessions/{sid}/documents/{doc_id}")
    def get_document(doc_id: str, sess=Depends(owned_session)) -> dict:
        meta = documents.get(sess.id, doc_id)
        if meta is None:
            raise HTTPException(404, "document not found")
        return meta.to_dict()

    @router.get("/api/sessions/{sid}/documents/{doc_id}/pages/{page_file}")
    def render_document_page(
        doc_id: str, page_file: str,
        scale: float = Query(limits.DEFAULT_SCALE, ge=limits.MIN_SCALE, le=limits.MAX_SCALE),
        x0: float | None = Query(None), y0: float | None = Query(None),
        x1: float | None = Query(None), y1: float | None = Query(None),
        sess=Depends(owned_session),
    ) -> Response:
        m = _PAGE_FILE_RE.match(page_file)
        meta = documents.get(sess.id, doc_id)
        if m is None or meta is None or int(m.group(1)) >= meta.pages:
            raise HTTPException(404, "page not found")
        page = int(m.group(1))
        corners = (x0, y0, x1, y1)
        try:
            bbox = None
            if any(c is not None for c in corners):
                if any(c is None for c in corners):
                    raise HTTPException(400, "bbox needs all of x0, y0, x1, y1")
                bbox = NormBBox(x0=x0, y0=y0, x1=x1, y1=y1)
            params = RenderParams(scale=scale, bbox=bbox)
        except ValidationError:
            raise HTTPException(400, "invalid bbox: need 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1") from None
        path = documents.pdf_path(sess.id, doc_id)
        if path is None:
            raise HTTPException(404, "document not found")
        try:
            png = render_page(path, page, params)
        except RenderError as e:
            raise _http(_RENDER_ERRORS, e.kind) from None
        return Response(content=png, media_type="image/png", headers={
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "private, no-store",
            "Content-Disposition": "inline",
        })

    return router
