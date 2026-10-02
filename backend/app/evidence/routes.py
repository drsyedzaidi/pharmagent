"""Evidence endpoints: upload, criteria, template, snapshot, page text, PDF download, screening,
extraction, record verification and record decisions.

``build_evidence_router`` is given main.py's auth dependencies and a getter for the live
Orchestrator (tests reassign ``main.orch``), so main.py only imports and includes it.
Every mutation goes through ``Orchestrator.run_tool`` (audit + write-access); the
routes themselves only validate, stage files and map errors:
ValueError -> 400, EvidenceStateError -> 409, IngestError -> 422, ParseBusyError -> 503.
POST /screen and POST /extract run as JobManager jobs when a real LLM is configured (model
calls per source/page), synchronously on the keyless path; POST /screen/decide and
POST /records/decide loop over the reviewer's decisions, one audited tool call each.
POST /export writes the approved records as CSV; GET /dataset/{filename} serves only a file
this session exported (basename match against the state mirror or the session's ledger rows,
resolved under data_dir), like the /report download.
"""
from __future__ import annotations

import logging
import os
import uuid
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from app.core.provenance import file_sha256
from app.evidence import ingest, llm_access, upload_limits
from app.evidence.config_tools import source_row
from app.evidence.models import DECISION_STATUSES, VERIFY_STATUSES, EvidenceStateError, IngestError, Source
from app.evidence.parse_sandbox import ParseBusyError
from app.evidence.route_headers import HeaderKeepingRoute
from app.evidence.store import get_store, source_id

log = logging.getLogger(__name__)

AGENT = "evidence"
CHUNK = 1 << 20
JOB_KINDS = {"screen_publications": "evidence_screen", "extract_evidence": "evidence_extract"}


def _unwrap(body: Any, key: str) -> Any:
    """{key: ...} envelope or the bare value; the tool validates the shape itself."""
    return body.get(key) if isinstance(body, dict) and key in body else body


def http_error(exc: Exception) -> HTTPException:
    """The one error map for evidence routes (IngestError before its ValueError base)."""
    if isinstance(exc, ParseBusyError):
        return HTTPException(503, str(exc), headers={"Retry-After": str(exc.retry_after)})
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


def owned_screen_id(sid: str, screen_id: Any) -> str:
    if not isinstance(screen_id, str) or not any(s.id == screen_id for s in get_store().list_screens(sid)):
        raise HTTPException(404, f"unknown screening verdict: {screen_id}")
    return screen_id


def owned_record_id(sid: str, record_id: Any) -> str:
    rec = get_store().get_record(record_id) if isinstance(record_id, str) and record_id else None
    if rec is None or rec.session_id != sid:
        raise HTTPException(404, f"unknown record: {record_id}")
    return record_id


def _decisions(body: Any, id_key: str) -> list[dict[str, Any]]:
    """{decisions:[{<id_key>, decision, reason}]} or one bare decision object; never empty."""
    items = body.get("decisions") if isinstance(body, dict) and "decisions" in body else [body]
    if not isinstance(items, list) or not items or not all(isinstance(d, dict) for d in items):
        raise HTTPException(400, "body must be a decision object or {decisions: [..]} (non-empty)")
    keys = (id_key, "decision", "reason") + (("allow_conflict",) if id_key == "record_id" else ())
    return [{k: d.get(k) for k in keys if k in d or k != "allow_conflict"} for d in items]


def _decide_loop(orch: Any, sid: str, actor: str, body: Any, *, tool: str, id_key: str,
                 owned: Callable[[str, Any], str], result_key: str) -> dict[str, Any]:
    """One audited decide_* tool call per decision (confirm=True is the HTTP call itself); a failure
    mid-list stops the loop, earlier decisions stand, and the 400 names what was applied."""
    decisions = _decisions(body, id_key)
    decided: list[dict[str, Any]] = []
    last: dict[str, Any] = {}
    for d in decisions:
        try:
            owned(sid, d[id_key])
            last = orch.run_tool(sid, tool, AGENT, {**d, "actor": actor, "confirm": True}, actor=actor)
        except Exception as exc:
            err = http_error(exc)
            if not decided:
                raise err from exc
            applied = ", ".join(x[result_key]["id"] for x in decided)
            raise HTTPException(400, f"{len(decided)} of {len(decisions)} decisions applied ({applied}) "
                                     f"before {d[id_key]} failed: {err.detail}") from exc
        decided.append(last["result"])
    return {"decided": decided, "summary": last["summary"], "state": last["state"]}


def _run_or_job(get_orch: Callable[[], Any], get_jobs: Callable[[], Any] | None, sid: str, tool: str,
                body: Any, actor: str) -> dict[str, Any]:
    """Propose via *tool*: a job when a real LLM is configured and a JobManager exists, else inline."""
    body = body or {}
    if not isinstance(body, dict):
        raise HTTPException(400, "body must be an object ({source_ids?: [..]})")
    args = {"actor": actor, **({"source_ids": body["source_ids"]} if "source_ids" in body else {})}
    if llm_access.llm_available() and get_jobs is not None:
        orch = get_orch()
        job_id = get_jobs().submit(session_id=sid, kind=JOB_KINDS[tool],
                                   fn=lambda: orch.run_tool(sid, tool, AGENT, args, actor=actor))
        return {"job_id": job_id, "status": "running", "kind": JOB_KINDS[tool], "tool": tool}
    return run_tool(get_orch(), sid, tool, args, actor)


def _record_filter(status: str | None, verify: str | None, flagged: bool | None) -> Callable[[Any], bool]:
    if status is not None and status not in DECISION_STATUSES:
        raise HTTPException(400, f"status must be one of {list(DECISION_STATUSES)}")
    if verify is not None and verify not in VERIFY_STATUSES:
        raise HTTPException(400, f"verify must be one of {list(VERIFY_STATUSES)}")
    return lambda r: ((status is None or r.status == status) and (verify is None or r.verify_status == verify)
                      and (flagged is None or bool(r.flags) == flagged))


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


def dataset_file(sid: str, state: Any, filename: str, data_dir: str) -> Path:
    """The exported CSV *filename* names, or 404: must equal the basename of the session's last export
    (state mirror) or of a dataset row in this session's ledger, resolve under data_dir and be a file."""
    summary = state.evidence_summary or {}
    known = {state.evidence_dataset_path, summary.get("wide_path")}
    for ds in get_store().list_datasets(sid):
        known.update((ds.long_path, ds.wide_path))
    root = Path(data_dir).resolve()
    for candidate in known:
        if candidate and Path(candidate).name == filename:
            path = Path(candidate).resolve()
            if path.is_relative_to(root) and path.is_file():
                return path
    raise HTTPException(404, "evidence dataset not found")


def _finish_upload(get_orch: Callable[[], Any], sid: str, data_dir: str, tmp: Path, name: str,
                   licence_note: str | None, actor: str) -> dict[str, Any]:
    """sha256 -> dedupe -> validate -> move to evidence/<sha>.pdf -> ingest tool (parse). Blocking."""
    try:
        upload_limits.admit_size(sid, tmp.stat().st_size)
        sha = file_sha256(tmp)
        existing = get_store().get_source(source_id(sid, sha))
        if existing is not None:
            return {"source": source_row(existing), "duplicate": True,
                    "summary": f"{existing.filename} was already uploaded to this session ({existing.id})."}
        try:
            ingest.validate_pdf_file(tmp, sid)
        except IngestError as exc:
            raise http_error(exc) from exc
        dest = ingest.evidence_dir(data_dir) / f"{sha}.pdf"
        if not dest.exists():
            os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)
    args = {"path": str(dest), "sha256": sha, "filename": ingest.sanitise_filename(name),
            "licence_note": (licence_note or "").strip() or None, "actor": actor}
    out = run_tool(get_orch(), sid, "ingest_evidence_source", args, actor)
    return {"source": out["result"]["source"], "duplicate": False, "summary": out["summary"]}


def _config_response(sid: str, kind: str, out: dict[str, Any]) -> dict[str, Any]:
    cfg = get_store().get_config(sid, kind) or {}
    return {**cfg, "kind": kind, "summary": out["summary"]}


def build_evidence_router(owned_session: Callable[..., Any], actor_id: Callable[..., str],
                          get_orch: Callable[[], Any],
                          get_jobs: Callable[[], Any] | None = None) -> APIRouter:
    """``get_jobs`` returns main.py's JobManager; without it the LLM screening path runs inline."""
    router = APIRouter(prefix="/api/sessions/{sid}/evidence", tags=["evidence"],
                       route_class=HeaderKeepingRoute)

    @router.post("/upload")
    async def upload_evidence(sid: str, file: UploadFile = File(...), licence_note: str | None = Form(None),
                              sess=Depends(owned_session), actor: str = Depends(actor_id)) -> dict:
        """Admit (caps, rate, server-wide slot) -> stage -> magic/size -> [upload worker] sha256 -> dedupe
        -> move -> ingest tool."""
        name = file.filename or "upload.pdf"
        if Path(name).suffix.lower() != ".pdf":
            raise HTTPException(400, f"unsupported file type: {Path(name).suffix or '(none)'}; PDF only")
        upload_limits.admit(sid)
        async with upload_limits.upload_slot():           # server-wide cap: 503 when the queue is full
            tmp = await _stage_upload(file, sess.ctx.data_dir)
            # hashing, validation and the parse are blocking: off the event loop AND off the shared
            # threadpool, so queued uploads never starve the synchronous endpoints
            return await upload_limits.run_blocking(_finish_upload, get_orch, sid, sess.ctx.data_dir, tmp,
                                                    name, licence_note, actor)

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
    def snapshot(sid: str, status: str | None = Query(None), verify: str | None = Query(None),
                 flagged: bool | None = Query(None), sess=Depends(owned_session)) -> dict:
        """Ledger snapshot; status / verify / flagged filter the records list only (counts never)."""
        keep = _record_filter(status, verify, flagged)
        store = get_store()
        return {"session_id": sid,
                "sources": [source_row(s) for s in store.list_sources(sid)],
                "criteria": store.get_config(sid, "criteria"),
                "template": store.get_config(sid, "template"),
                "screens": [asdict(s) for s in store.list_screens(sid)],
                "records": [asdict(r) for r in store.list_records(sid) if keep(r)],
                "datasets": [asdict(d) for d in store.list_datasets(sid)],
                "counts": store.counts(sid),
                "summary": sess.state.evidence_summary}

    @router.post("/screen")
    def screen(sid: str, body: Any = Body(None), sess=Depends(owned_session),
               actor: str = Depends(actor_id)) -> dict:
        """Propose screening verdicts: a job when a real LLM is configured, else synchronous."""
        return _run_or_job(get_orch, get_jobs, sid, "screen_publications", body, actor)

    @router.post("/screen/decide")
    def decide_screen(sid: str, body: Any = Body(...), sess=Depends(owned_session),
                      actor: str = Depends(actor_id)) -> dict:
        out = _decide_loop(get_orch(), sid, actor, body, tool="decide_screening", id_key="screen_id",
                           owned=owned_screen_id, result_key="screen")
        return {**out, "decided": [d["screen"] for d in out["decided"]],
                "screening_results": out["state"]["screening_results"]}

    @router.post("/extract")
    def extract(sid: str, body: Any = Body(None), sess=Depends(owned_session),
                actor: str = Depends(actor_id)) -> dict:
        """Propose extracted records from the included publications: a job with a real LLM, else synchronous."""
        return _run_or_job(get_orch, get_jobs, sid, "extract_evidence", body, actor)

    @router.post("/records/verify")
    def verify_records(sid: str, body: Any = Body(None), sess=Depends(owned_session),
                       actor: str = Depends(actor_id)) -> dict:
        body = body or {}
        if not isinstance(body, dict):
            raise HTTPException(400, "body must be an object ({record_ids?: [..]})")
        args = {"actor": actor, **({"record_ids": body["record_ids"]} if "record_ids" in body else {})}
        return run_tool(get_orch(), sid, "verify_records", args, actor)

    @router.post("/records/decide")
    def decide_records(sid: str, body: Any = Body(...), sess=Depends(owned_session),
                       actor: str = Depends(actor_id)) -> dict:
        out = _decide_loop(get_orch(), sid, actor, body, tool="decide_records", id_key="record_id",
                           owned=owned_record_id, result_key="record")
        return {**out, "extraction_results": out["state"]["extraction_results"]}

    @router.post("/export")
    def export_dataset(sid: str, body: Any = Body(None), sess=Depends(owned_session),
                       actor: str = Depends(actor_id)) -> dict:
        """Write the approved records as long + wide CSV (deterministic; >= 1 approved record)."""
        return run_tool(get_orch(), sid, "export_evidence_dataset", {"actor": actor}, actor)

    @router.get("/dataset/{filename}")
    def download_dataset(sid: str, filename: str, sess=Depends(owned_session)) -> FileResponse:
        path = dataset_file(sid, sess.state, filename, sess.ctx.data_dir)
        return FileResponse(str(path), media_type="text/csv", filename=path.name)

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
