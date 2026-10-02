"""Evidence endpoints: upload, criteria, template, snapshot, page text, PDF download.

``build_evidence_router`` is given main.py's auth dependencies and a getter for the live
Orchestrator (tests reassign ``main.orch``), so main.py only imports and includes it.
Every mutation goes through ``Orchestrator.run_tool`` (audit + write-access); the
routes themselves only validate, stage files and map errors:
ValueError -> 400, EvidenceStateError -> 409, IngestError -> 422.
"""
from __future__ import annotations

import logging
import os
import uuid
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.core.provenance import file_sha256
from app.evidence import ingest
from app.evidence.config_tools import source_row
from app.evidence.models import EvidenceStateError, IngestError, Source
from app.evidence.store import get_store, source_id

log = logging.getLogger(__name__)

AGENT = "evidence"
CHUNK = 1 << 20


def _unwrap(body: Any, key: str) -> Any:
    """{key: ...} envelope or the bare value; the tool validates the shape itself."""
    return body.get(key) if isinstance(body, dict) and key in body else body


def http_error(exc: Exception) -> HTTPException:
    """The one error map for evidence routes (IngestError before its ValueError base)."""
    if isinstance(exc, IngestError):
        return HTTPException(422, str(exc))
    if isinstance(exc, EvidenceStateError):
        return HTTPException(409, str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(400, str(exc))
    if isinstance(exc, HTTPException):
        return exc
    log.exception("evidence route failed")
    return HTTPException(500, "evidence operation failed")


def run_tool(orch: Any, sid: str, tool: str, args: dict[str, Any], actor: str) -> dict[str, Any]:
    try:
        return orch.run_tool(sid, tool, AGENT, args, actor=actor)
    except Exception as exc:          # mapped, never swallowed
        raise http_error(exc) from exc


def owned_source(sid: str, src_id: str) -> Source:
    src = get_store().get_source(src_id)
    if src is None or src.session_id != sid:
        raise HTTPException(404, f"unknown evidence source: {src_id}")
    return src


async def _stage_upload(file: UploadFile, data_dir: str) -> Path:
    """Stream the upload to evidence/incoming/<uuid>.pdf under the size cap; nothing else is written."""
    incoming = ingest.evidence_dir(data_dir) / "incoming"
    incoming.mkdir(parents=True, exist_ok=True)
    tmp = incoming / f"{uuid.uuid4().hex}.pdf"
    size = 0
    try:
        with tmp.open("wb") as fh:
            while chunk := await file.read(CHUNK):
                size += len(chunk)
                if size > ingest.MAX_BYTES:
                    raise HTTPException(413, f"file exceeds {ingest.MAX_BYTES // (1024 * 1024)} MB limit")
                fh.write(chunk)
        with tmp.open("rb") as fh:
            head = fh.read(len(ingest.PDF_MAGIC))
        if head != ingest.PDF_MAGIC:
            raise HTTPException(400, "not a PDF (missing %PDF- header)")
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return tmp


def _config_response(sid: str, kind: str, out: dict[str, Any]) -> dict[str, Any]:
    cfg = get_store().get_config(sid, kind) or {}
    return {**cfg, "kind": kind, "summary": out["summary"]}


def build_evidence_router(owned_session: Callable[..., Any], actor_id: Callable[..., str],
                          get_orch: Callable[[], Any]) -> APIRouter:
    router = APIRouter(prefix="/api/sessions/{sid}/evidence", tags=["evidence"])

    @router.post("/upload")
    async def upload_evidence(sid: str, file: UploadFile = File(...), licence_note: str | None = Form(None),
                              sess=Depends(owned_session), actor: str = Depends(actor_id)) -> dict:
        """Stage -> magic/size -> sha256 -> dedupe -> move to evidence/<sha>.pdf -> ingest tool."""
        name = file.filename or "upload.pdf"
        if Path(name).suffix.lower() != ".pdf":
            raise HTTPException(400, f"unsupported file type: {Path(name).suffix or '(none)'}; PDF only")
        tmp = await _stage_upload(file, sess.ctx.data_dir)
        try:
            sha = file_sha256(tmp)
            existing = get_store().get_source(source_id(sid, sha))
            if existing is not None:
                return {"source": source_row(existing), "duplicate": True,
                        "summary": f"{existing.filename} was already uploaded to this session ({existing.id})."}
            try:
                ingest.validate_pdf_file(tmp)
            except IngestError as exc:
                raise http_error(exc) from exc
            dest = ingest.evidence_dir(sess.ctx.data_dir) / f"{sha}.pdf"
            if not dest.exists():
                os.replace(tmp, dest)
        finally:
            tmp.unlink(missing_ok=True)
        args = {"path": str(dest), "sha256": sha, "filename": ingest.sanitise_filename(name),
                "licence_note": (licence_note or "").strip() or None, "actor": actor}
        out = run_tool(get_orch(), sid, "ingest_evidence_source", args, actor)
        return {"source": out["result"]["source"], "duplicate": False, "summary": out["summary"]}

    @router.post("/criteria")
    def save_criteria(sid: str, body: Any = Body(...), sess=Depends(owned_session),
                      actor: str = Depends(actor_id)) -> dict:
        out = run_tool(get_orch(), sid, "save_evidence_config",
                       {"kind": "criteria", "body": _unwrap(body, "criteria"), "actor": actor}, actor)
        return _config_response(sid, "criteria", out)

    @router.post("/template")
    def save_template(sid: str, body: Any = Body(...), sess=Depends(owned_session),
                      actor: str = Depends(actor_id)) -> dict:
        out = run_tool(get_orch(), sid, "save_evidence_config",
                       {"kind": "template", "body": body, "actor": actor}, actor)
        return _config_response(sid, "template", out)

    @router.get("")
    def snapshot(sid: str, sess=Depends(owned_session)) -> dict:
        store = get_store()
        return {"session_id": sid,
                "sources": [source_row(s) for s in store.list_sources(sid)],
                "criteria": store.get_config(sid, "criteria"),
                "template": store.get_config(sid, "template"),
                "screens": [asdict(s) for s in store.list_screens(sid)],
                "records": [asdict(r) for r in store.list_records(sid)],
                "datasets": [asdict(d) for d in store.list_datasets(sid)],
                "counts": store.counts(sid),
                "summary": sess.state.evidence_summary}

    @router.get("/sources/{src_id}/pages/{n}")
    def source_page(sid: str, src_id: str, n: int, sess=Depends(owned_session)) -> dict:
        src = owned_source(sid, src_id)
        try:
            doc = ingest.load_pages(src)
        except (OSError, ValueError) as exc:
            raise http_error(exc) from exc
        pages = doc.get("pages") or []
        if n < 1 or n > len(pages):
            raise HTTPException(404, f"page {n} out of range (1..{len(pages)})")
        page = pages[n - 1]
        return {"source_id": src.id, "page": n, "pages": len(pages), "text": page.get("text", ""),
                "small_text": page.get("small_text", ""), "columns": page.get("columns"),
                "tables": page.get("tables") or [], "decimal_style": doc.get("decimal_style", ".")}

    @router.get("/sources/{src_id}/pdf")
    def source_pdf(sid: str, src_id: str, sess=Depends(owned_session)) -> FileResponse:
        src = owned_source(sid, src_id)
        path = Path(src.pdf_path)
        if not path.is_file() or file_sha256(path) != src.sha256:
            raise HTTPException(409, "source file missing or changed on disk")
        return FileResponse(str(path), media_type="application/pdf", filename=src.filename,
                            content_disposition_type="inline")

    return router
