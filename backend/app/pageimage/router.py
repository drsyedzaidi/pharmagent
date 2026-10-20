"""HTTP routes for the page-image viewer.

Built by a factory so the existing ownership dependency (``owned_session`` in
``app.main``) is injected unchanged rather than imported (avoids a cycle and
leaves auth code untouched). Rendering is read-only and is NOT audited.
"""
from __future__ import annotations

import contextlib
import logging
import re
import sys
import time
from collections.abc import Callable
from pathlib import Path

import anyio
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import Response
from fastapi.routing import APIRoute
from pydantic import ValidationError
from python_multipart.exceptions import FormParserError
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import FormData, UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser
from starlette.requests import ClientDisconnect
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


_MALFORMED = "upload must be a well-formed multipart/form-data body with one 'file' part"
_INTERRUPTED = "upload was interrupted"
# Client-caused parser failures: bad framing or headers, too many parts, an unknown or broken charset.
_CLIENT_PARSE_ERRORS = (MultiPartException, FormParserError, UnicodeError, LookupError)


async def _read_form(request: Request) -> FormData:
    """Parse the upload's multipart body. Done by hand (not ``request.form()``) because the route reads the body only
    after authentication and the quota reservation, so FastAPI's own parse-error handling does not apply.

    Limits: exactly the media type multipart/form-data, at most one file part, at most 4 other fields of at most
    64 KiB each. Every parser complaint (no boundary, bad framing or part headers, too many parts, an unknown charset)
    is the client's mistake: a fixed 400 in the standard envelope, not a 500 and not an ERROR traceback; a client that
    disconnects mid-body gets the same quiet treatment. Every temp file the parser spooled is closed here, on failure
    (including the body cap firing mid-parse) and also on success for a file part that never reached the form (an
    unterminated body), instead of being left to the garbage collector."""
    media_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if media_type != "multipart/form-data":
        raise HTTPException(400, _MALFORMED)
    parser = MultiPartParser(request.headers, request.stream(), max_files=1, max_fields=4, max_part_size=64 * 1024)
    spooled = getattr(parser, "_files_to_close_on_error", [])          # Starlette's own list of opened spools
    try:
        form = await parser.parse()
    except BaseException as exc:
        _close_all(spooled)
        if isinstance(exc, _CLIENT_PARSE_ERRORS):
            raise HTTPException(400, _MALFORMED) from None
        if isinstance(exc, ClientDisconnect):
            raise HTTPException(400, _INTERRUPTED) from None
        raise
    in_form = {id(v.file) for v in form.values() if isinstance(v, UploadFile)}
    _close_all([f for f in spooled if id(f) not in in_form])
    return form


def _close_all(files) -> None:
    for f in files:
        try:
            f.close()
        except Exception:                                               # noqa: BLE001 - closing must not mask exc
            pass


async def _discard_quietly(part) -> None:
    """Release a failed upload's pieces and reservation without blocking the event loop and without ever replacing the
    error that caused the failure. ``documents.discard`` waits for the storage lock, so it runs in a worker thread,
    shielded from cancellation so a cancelled request still cleans up. If it cannot run at all, nothing is deleted:
    the reservation keeps the upload charged and locked stale cleanup reclaims it later. A cancellation is re-raised;
    any other failure of the cleanup is only logged."""
    try:
        with anyio.CancelScope(shield=True):
            await run_in_threadpool(documents.discard, part)
    except BaseException as exc:
        if not isinstance(exc, Exception):
            raise
        log.warning("upload cleanup did not run; the reservation stays for stale cleanup")


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


def _heartbeating(request: Request, part: Path, clock: Callable[[], float] = time.monotonic) -> Request:
    """The same request with a receive channel that keeps the upload's reservation alive while its body is still
    arriving: a body message received at least HEARTBEAT_SECONDS after the reservation or the last heartbeat refreshes
    it through the locked ``documents.touch``, in a worker thread (it may wait for the storage lock). Paced by time,
    not bytes, and driven by what arrives: a body that sends nothing gets no heartbeat and goes stale. If the
    reservation was already reclaimed, the heartbeat raises ReservationLostError and the rest of the body is never
    read."""
    last = clock()
    original = request.receive

    async def receive():
        nonlocal last
        msg = await original()
        if msg["type"] == "http.request" and clock() - last >= limits.HEARTBEAT_SECONDS:
            await run_in_threadpool(documents.touch, part)
            last = clock()
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
    async def upload_document(request: Request, sess=Depends(owned_session)) -> dict:
        # No ``UploadFile = File(...)`` parameter on purpose: FastAPI would parse (and spool to disk) the multipart body
        # BEFORE running ``owned_session`` or any quota check. Here the order is: authenticate/own the session, reserve
        # quota, and only then read the form.
        # Worst case, always: the client chooses Content-Length, so it must never size the reservation (a declared 0
        # followed by 50 MiB would otherwise bypass both byte budgets). The header only matters to the route gate's cap.
        reserve = limits.MAX_PDF_BYTES
        try:
            doc_id, part = await run_in_threadpool(documents.new_slot, sess.id, reserve)
        except documents.DocumentLimitError:
            raise HTTPException(409, f"document limit reached ({limits.MAX_DOCS_PER_SESSION} per session)") from None
        except documents.QuotaExceeded as e:
            if e.scope == "session":
                raise HTTPException(409, "storage quota for this session reached") from None
            raise HTTPException(503, "document storage is full on this server; retry later") from None
        except documents.QuotaUnavailable:
            log.warning("document storage accounting unavailable; upload refused")
            raise HTTPException(503, "document storage is unavailable; retry later") from None
        except ValueError:
            raise HTTPException(400, "invalid session") from None
        form = None
        try:
            form = await _read_form(_heartbeating(request, part))
            file = form.get("file")
            if not isinstance(file, UploadFile):
                raise HTTPException(400, _MALFORMED)
            size = 0
            with contextlib.ExitStack() as stack:
                out = None
                while chunk := await file.read(1 << 20):
                    if size == 0 and not chunk.startswith(_PDF_MAGIC):
                        raise HTTPException(400, "file is not a PDF")
                    size += len(chunk)
                    if size > limits.MAX_PDF_BYTES:
                        raise HTTPException(413, f"file exceeds {limits.MAX_PDF_BYTES // (1024 * 1024)} MB limit")
                    # Before every write, the first included: refresh the reservation and confirm it still exists, so
                    # an upload whose reservation was reclaimed never creates or writes its .part (409 instead).
                    await run_in_threadpool(documents.touch, part)
                    if out is None:
                        out = stack.enter_context(part.open("wb"))
                    out.write(chunk)
            if size == 0:
                raise HTTPException(400, "file is not a PDF")
            try:
                pages = await run_in_threadpool(lambda: probe_pdf(part, session=sess.id))
            except RenderError as e:
                raise _http(_UPLOAD_ERRORS, e.kind) from None
            if pages < 1:
                raise HTTPException(400, "PDF has no pages")
            if pages > limits.MAX_PAGES:
                raise _http(_UPLOAD_ERRORS, "too_large")
            meta = await run_in_threadpool(documents.commit, sess.id, doc_id, part,
                                           filename=file.filename, size=size, pages=pages)
        except BaseException as exc:
            await _discard_quietly(part)      # deletes the pieces, then releases the reservation (if all went)
            if isinstance(exc, documents.ReservationLostError):
                raise HTTPException(409, "upload expired; please retry") from None
            if isinstance(exc, documents.QuotaUnavailable):
                raise HTTPException(503, "document storage is unavailable; retry later") from None
            raise
        else:
            return meta.to_dict()
        finally:
            if form is not None:
                await form.close()

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
            png = render_page(path, page, params, session=sess.id)
        except RenderError as e:
            raise _http(_RENDER_ERRORS, e.kind) from None
        return Response(content=png, media_type="image/png", headers={
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "private, no-store",
            "Content-Disposition": "inline",
        })

    return router
