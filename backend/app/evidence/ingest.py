"""Upload validation, content-addressed storage, parse cache and fail-closed reload of sources.

Nothing is written before validation passes. Files live under
data_dir/evidence/<sha256>.pdf and <sha256>.pages.json; load_pages() re-hashes
the PDF on every read and refuses to serve a parse whose source changed.
"""
from __future__ import annotations

import io
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import pdfplumber
from pdfminer.pdfdocument import PDFPasswordIncorrect

from app.evidence import parse_sandbox
from app.evidence.models import IngestError, Source
from app.evidence.parse import PARSER_NAME, file_sha256
from app.evidence.parse_sandbox import ParseBusyError
from app.evidence.store import EvidenceStore, get_store

log = logging.getLogger(__name__)

MAX_BYTES = 25 * 1024 * 1024
MAX_PAGES = 300
MAX_FILENAME = 200
MAX_TITLE = 200
PDF_MAGIC = b"%PDF-"
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


def evidence_dir(data_dir: str | Path | None = None) -> Path:
    if data_dir is None:
        from app.config import settings
        data_dir = settings.data_dir
    d = Path(data_dir) / "evidence"
    d.mkdir(parents=True, exist_ok=True)
    return d


def sanitise_filename(name: str) -> str:
    """Basename only, [A-Za-z0-9._-] only, <= 200 chars, 'upload.pdf' when nothing is left."""
    base = Path((name or "").replace("\\", "/")).name
    clean = _UNSAFE.sub("_", base)
    if not re.search(r"[A-Za-z0-9]", clean):
        return "upload.pdf"
    if len(clean) > MAX_FILENAME:
        stem, ext = os.path.splitext(clean)
        ext = ext[:10]
        clean = stem[: MAX_FILENAME - len(ext)] + ext
    return clean


def _is_password_error(exc: BaseException) -> bool:
    seen, cur = set(), exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, PDFPasswordIncorrect):
            return True
        inner = next((a for a in getattr(cur, "args", ()) if isinstance(a, BaseException)), None)
        cur = cur.__cause__ or cur.__context__ or inner
    return False


def validate_pdf_bytes(data: bytes) -> int:
    """Raise IngestError unless *data* is an openable, unencrypted PDF within limits. Returns page count."""
    if not data or not data.startswith(PDF_MAGIC):
        raise IngestError("not a PDF (missing %PDF- header)")
    if len(data) > MAX_BYTES:
        raise IngestError(f"too large ({len(data)} bytes > {MAX_BYTES})")
    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            n = len(pdf.pages)
    except Exception as exc:
        if _is_password_error(exc):
            raise IngestError("encrypted PDF (password required)") from exc
        raise IngestError(f"unreadable PDF: {type(exc).__name__}") from exc
    if n == 0:
        raise IngestError("unreadable PDF: no pages")
    if n > MAX_PAGES:
        raise IngestError(f"page cap exceeded ({n} pages > {MAX_PAGES})")
    return n


def validate_pdf_file(path: str | Path) -> int:
    path = Path(path)
    if path.stat().st_size > MAX_BYTES:
        raise IngestError(f"too large ({path.stat().st_size} bytes > {MAX_BYTES})")
    return validate_pdf_bytes(path.read_bytes())


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def write_by_sha256(data: bytes, data_dir: str | Path | None = None) -> tuple[str, Path]:
    """Store bytes as <sha256>.pdf; an existing file with that name is left untouched."""
    import hashlib
    sha = hashlib.sha256(data).hexdigest()
    path = evidence_dir(data_dir) / f"{sha}.pdf"
    if not path.exists():
        _atomic_write(path, data)
    return sha, path


def parse_and_cache(pdf_path: str | Path, data_dir: str | Path | None = None) -> tuple[dict[str, Any], Path]:
    """Parse once per (sha256, parser) and cache as <sha256>.pages.json. The parse runs in a killable
    child process under a process-wide concurrency cap (parse_sandbox)."""
    pdf_path = Path(pdf_path)
    sha = file_sha256(pdf_path)
    parse_path = evidence_dir(data_dir) / f"{sha}.pages.json"
    if parse_path.exists():
        try:
            cached = json.loads(parse_path.read_text("utf-8"))
            if cached.get("sha256") == sha and cached.get("parser") == PARSER_NAME and "error" not in cached:
                return cached, parse_path
        except (OSError, ValueError):
            log.warning("evidence: unreadable parse cache %s; re-parsing", parse_path)
    doc = parse_sandbox.parse_isolated(pdf_path)
    _atomic_write(parse_path, json.dumps(doc, ensure_ascii=False).encode("utf-8"))
    return doc, parse_path


def load_pages(source: Source) -> dict[str, Any]:
    """Parse JSON for a source; fails closed if the PDF or the cache no longer match the ledger."""
    pdf_path = Path(source.pdf_path)
    if not pdf_path.is_file():
        raise ValueError("source file missing")
    if file_sha256(pdf_path) != source.sha256:
        raise ValueError("source file changed")
    doc = json.loads(Path(source.parse_path).read_text("utf-8"))
    if doc.get("sha256") != source.sha256:
        raise ValueError("parse cache mismatch")
    return doc


def derive_title(doc: dict[str, Any]) -> str | None:
    """PDF metadata Title, else the first text line of page 1 (<= 200 chars). Untrusted text."""
    title = " ".join(str((doc.get("metadata") or {}).get("title") or "").split())
    if not title and doc.get("pages"):
        first = next((ln.strip() for ln in doc["pages"][0].get("text", "").splitlines() if ln.strip()), "")
        title = " ".join(first.split())
    return title[:MAX_TITLE] or None


def ingest_pdf(data: bytes, filename: str, session_id: str, uploaded_by: str, *,
               data_dir: str | Path | None = None, store: EvidenceStore | None = None,
               licence_note: str | None = None) -> Source:
    """Validate -> store by sha256 -> parse+cache -> ledger row. Idempotent per (session, sha256)."""
    pages = validate_pdf_bytes(data)
    sha, pdf_path = write_by_sha256(data, data_dir)
    try:
        doc, parse_path = parse_and_cache(pdf_path, data_dir)
        status = "parsed" if doc["text_chars"] > 0 else "no_text"
    except ParseBusyError:     # no slot: nothing parsed, nothing recorded; the upload can be retried
        raise
    except Exception as exc:   # parser failure (limits, kill, crash) is recorded, never raised past the upload
        log.warning("evidence: parse failed for %s: %s", sha, exc)
        doc = {"sha256": sha, "parser": PARSER_NAME, "decimal_style": ".", "text_chars": 0,
               "metadata": {}, "pages": [], "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
        parse_path = evidence_dir(data_dir) / f"{sha}.pages.json"
        _atomic_write(parse_path, json.dumps(doc).encode("utf-8"))
        status = "failed"
    return (store or get_store()).add_source(
        session_id=session_id, sha256=sha, filename=sanitise_filename(filename), pdf_path=str(pdf_path),
        parse_path=str(parse_path), pages=pages, title=derive_title(doc), metadata=doc.get("metadata") or {},
        text_chars=int(doc.get("text_chars") or 0), parse_status=status,
        decimal_style=doc.get("decimal_style") or ".", licence_note=licence_note, uploaded_by=uploaded_by)
