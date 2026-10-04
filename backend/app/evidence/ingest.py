"""Upload validation, content-addressed storage, parse cache and fail-closed reload of sources.

Nothing is written before validation passes; validation opens the PDF only inside the parse
sandbox's killable child (a tiny upload can hide an object stream that inflates to gigabytes). Files
live under
data_dir/evidence/<sha256>.pdf and <sha256>.pages.json; load_pages() re-hashes
the PDF on every read and refuses to serve a parse whose source changed.
"""
from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import pdfplumber
from pdfminer.pdfdocument import PDFPasswordIncorrect

from app.evidence import parse_sandbox
from app.evidence.models import IngestError, Source
from app.evidence.parse import PARSER_NAME, file_sha256
from app.evidence.parse_sandbox import ParseBusyError
from app.evidence.path_guard import OutsideDataDirError, confine
from app.evidence.store import EvidenceStore, get_store

log = logging.getLogger(__name__)

MAX_BYTES = 25 * 1024 * 1024
MAX_PAGES = 300
MAX_FILENAME = 200
MAX_TITLE = 200
PDF_MAGIC = b"%PDF-"
VALIDATE_WALL_S = 30.0                 # counting pages is quick; an object-stream bomb is not
VALIDATE_MAX_RSS = 512 * 2**20
VALIDATE_TARGET = "app.evidence.ingest:page_count_report"
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


def page_count_report(path: str) -> dict[str, Any]:
    """Child side of validation (parse_sandbox target): {'pages': n}, {'encrypted': True} or
    {'unreadable': exception type}. Opening a hostile PDF can inflate huge object streams, so this
    never runs in the server process."""
    try:
        with pdfplumber.open(path) as pdf:
            return {"pages": len(pdf.pages)}
    except Exception as exc:
        if _is_password_error(exc):
            return {"encrypted": True}
        return {"unreadable": type(exc).__name__}


def _isolated_page_count(path: Path, session_id: str | None = None) -> int:
    """Page count from a killable child (wall clock + RSS caps, the parse slots); IngestError otherwise."""
    try:
        report = parse_sandbox.run_isolated(path, target=VALIDATE_TARGET, wall_s=VALIDATE_WALL_S,
                                            max_rss=VALIDATE_MAX_RSS, session_id=session_id)
    except ParseBusyError:
        raise
    except IngestError as exc:                   # killed at a limit, or the child died
        raise IngestError(f"unreadable PDF: validation stopped ({exc})") from exc
    except Exception as exc:
        raise IngestError(f"unreadable PDF: {type(exc).__name__}") from exc
    if report.get("encrypted"):
        raise IngestError("encrypted PDF (password required)")
    if "pages" not in report:
        raise IngestError(f"unreadable PDF: {report.get('unreadable') or 'unknown error'}")
    n = int(report["pages"])
    if n == 0:
        raise IngestError("unreadable PDF: no pages")
    if n > MAX_PAGES:
        raise IngestError(f"page cap exceeded ({n} pages > {MAX_PAGES})")
    return n


def _check_head(head: bytes, size: int) -> None:
    if not head.startswith(PDF_MAGIC):
        raise IngestError("not a PDF (missing %PDF- header)")
    if size > MAX_BYTES:
        raise IngestError(f"too large ({size} bytes > {MAX_BYTES})")


def validate_pdf_bytes(data: bytes, session_id: str | None = None) -> int:
    """Raise IngestError unless *data* is an openable, unencrypted PDF within limits. Returns page count.
    The open itself runs in the sandbox child (validate_pdf_file)."""
    _check_head(data or b"", len(data or b""))
    fd, name = tempfile.mkstemp(prefix="evvalidate-", suffix=".pdf")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        return _isolated_page_count(Path(name), session_id)
    finally:
        Path(name).unlink(missing_ok=True)


def validate_pdf_file(path: str | Path, session_id: str | None = None) -> int:
    """validate_pdf_bytes for a file on disk, without reading it into the server process."""
    path = Path(path)
    with path.open("rb") as fh:
        head = fh.read(len(PDF_MAGIC))
    _check_head(head, path.stat().st_size)
    return _isolated_page_count(path, session_id)


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


def layout_path(sha: str, data_dir: str | Path | None = None) -> Path:
    return evidence_dir(data_dir) / f"{sha}.layout.json"


def _layout_current(path: Path, sha: str) -> bool:
    """True when the layout sidecar at *path* belongs to this PDF and this parser version."""
    try:
        side = json.loads(path.read_text("utf-8"))
        return side.get("sha256") == sha and side.get("parser") == PARSER_NAME
    except (OSError, ValueError):
        return False


def parse_and_cache(pdf_path: str | Path, data_dir: str | Path | None = None,
                    session_id: str | None = None) -> tuple[dict[str, Any], Path]:
    """Parse once per (sha256, parser) and cache as <sha256>.pages.json plus its layout sidecar
    <sha256>.layout.json (word boxes: parse_layout). The parse runs in a killable child process under
    a process-wide concurrency cap (parse_sandbox). A cache hit needs both files, written sidecar
    first: a pages file without its sidecar (an interrupted write) is parsed again."""
    pdf_path = Path(pdf_path)
    sha = file_sha256(pdf_path)
    parse_path = evidence_dir(data_dir) / f"{sha}.pages.json"
    side_path = layout_path(sha, data_dir)
    if parse_path.exists():
        try:
            cached = json.loads(parse_path.read_text("utf-8"))
            if (cached.get("sha256") == sha and cached.get("parser") == PARSER_NAME and "error" not in cached
                    and _layout_current(side_path, sha)):
                return cached, parse_path
        except (OSError, ValueError):
            log.warning("evidence: unreadable parse cache %s; re-parsing", parse_path)
    doc = parse_sandbox.parse_isolated(pdf_path, session_id=session_id)
    layout = doc.pop("layout", None)
    if layout is not None:
        _atomic_write(side_path, json.dumps(layout, ensure_ascii=False).encode("utf-8"))
    _atomic_write(parse_path, json.dumps(doc, ensure_ascii=False).encode("utf-8"))
    return doc, parse_path


def stored_paths(source: Source, data_dir: str | Path) -> tuple[Path, Path]:
    """(PDF path, parse-cache path) of *source*, resolved and confined to the evidence directory of *data_dir*
    BEFORE any of them is opened (path_guard); OutsideDataDirError when either leaves it."""
    root = evidence_dir(data_dir)
    return confine(source.pdf_path, root), confine(source.parse_path, root)


def source_file_intact(source: Source, data_dir: str | Path | None = None) -> bool:
    """True when the stored PDF still hashes to the ledger's sha256 (fail closed on any error). With a
    *data_dir* the stored path must also lie inside its evidence directory (checked before the file is read)."""
    try:
        pdf_path = Path(source.pdf_path) if data_dir is None else stored_paths(source, data_dir)[0]
        return pdf_path.is_file() and file_sha256(pdf_path) == source.sha256
    except (OSError, OutsideDataDirError):
        return False


def load_pages(source: Source, data_dir: str | Path | None = None) -> dict[str, Any]:
    """Parse JSON for a source; fails closed if the PDF or the cache no longer match the ledger. With a
    *data_dir* both stored paths are confined to its evidence directory first (OutsideDataDirError, nothing
    read); callers without one (agent tools) read the stored paths as they are."""
    pdf_path, parse_path = ((Path(source.pdf_path), Path(source.parse_path)) if data_dir is None
                            else stored_paths(source, data_dir))
    if not pdf_path.is_file():
        raise ValueError("source file missing")
    if file_sha256(pdf_path) != source.sha256:
        raise ValueError("source file changed")
    doc = json.loads(parse_path.read_text("utf-8"))
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
    pages = validate_pdf_bytes(data, session_id)
    sha, pdf_path = write_by_sha256(data, data_dir)
    try:
        doc, parse_path = parse_and_cache(pdf_path, data_dir, session_id)
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
