"""Session-scoped store for uploaded source PDFs.

Layout: ``<data_dir>/documents/<session id>/<doc_id>.pdf`` plus a ``<doc_id>.json``
metadata sidecar. ``doc_id`` is a server-issued uuid4 hex; the client filename is
kept only as sanitized display text and never takes part in any path. This is
deliberately separate from the session/owner/audit stores: ownership is enforced
by the route's ``owned_session`` dependency, and a document is only ever looked up
inside its own session directory.
"""
from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from app.config import settings
from app.pageimage import limits

_SID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_DOC_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_CTRL_RE = re.compile(r"[\x00-\x1f\x7f]")


class DocumentLimitError(Exception):
    """Per-session document quota reached."""


@dataclass(frozen=True)
class DocumentMeta:
    id: str
    filename: str
    size_bytes: int
    pages: int
    created_at: str

    def to_dict(self) -> dict:
        return asdict(self)


def sanitize_filename(raw: str | None) -> str:
    """Display-only name: basename, no control chars, bounded length."""
    name = (raw or "").replace("\\", "/").rsplit("/", 1)[-1]
    name = _CTRL_RE.sub("", name).strip().strip(".")
    return (name[:120] or "document.pdf")


def valid_doc_id(doc_id: str) -> bool:
    return bool(_DOC_ID_RE.match(doc_id))


def _root() -> Path:
    return Path(settings.data_dir) / "documents"


def _session_dir(sid: str) -> Path | None:
    """The session's document directory, or None if ``sid`` is not a plain id or
    would resolve outside the documents root."""
    if not _SID_RE.match(sid):
        return None
    root = _root().resolve()
    d = (root / sid).resolve()
    return d if d.is_relative_to(root) and d != root else None


def new_slot(sid: str) -> tuple[str, Path]:
    """Reserve a server-generated id and a ``.part`` path for a streaming upload."""
    d = _session_dir(sid)
    if d is None:
        raise ValueError("invalid session id")
    d.mkdir(parents=True, exist_ok=True)
    if len(list(d.glob("*.json"))) >= limits.MAX_DOCS_PER_SESSION:
        raise DocumentLimitError
    doc_id = uuid.uuid4().hex
    return doc_id, d / f"{doc_id}.part"


def commit(sid: str, doc_id: str, part: Path, *, filename: str | None, size: int, pages: int) -> DocumentMeta:
    """Promote a validated ``.part`` file to its final name and write metadata."""
    d = _session_dir(sid)
    if d is None or not valid_doc_id(doc_id) or part != d / f"{doc_id}.part":
        raise ValueError("invalid document slot")
    meta = DocumentMeta(id=doc_id, filename=sanitize_filename(filename), size_bytes=size, pages=pages,
                        created_at=datetime.now(UTC).isoformat(timespec="seconds"))
    os.replace(part, d / f"{doc_id}.pdf")
    tmp = d / f"{doc_id}.json.tmp"
    tmp.write_text(json.dumps(meta.to_dict()), encoding="utf-8")
    os.replace(tmp, d / f"{doc_id}.json")
    return meta


def discard(part: Path) -> None:
    part.unlink(missing_ok=True)


def _read_meta(p: Path) -> DocumentMeta | None:
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        return DocumentMeta(id=str(raw["id"]), filename=str(raw["filename"]), size_bytes=int(raw["size_bytes"]),
                            pages=int(raw["pages"]), created_at=str(raw["created_at"]))
    except (OSError, ValueError, KeyError, TypeError):
        return None


def get(sid: str, doc_id: str) -> DocumentMeta | None:
    d = _session_dir(sid)
    if d is None or not valid_doc_id(doc_id):
        return None
    meta = _read_meta(d / f"{doc_id}.json")
    if meta is None or meta.id != doc_id or not (d / f"{doc_id}.pdf").is_file():
        return None
    return meta


def pdf_path(sid: str, doc_id: str) -> Path | None:
    """Server-chosen path of a stored PDF; None unless it belongs to ``sid``."""
    if get(sid, doc_id) is None:
        return None
    d = _session_dir(sid)
    assert d is not None
    p = (d / f"{doc_id}.pdf").resolve()
    return p if p.is_relative_to(d) else None


def list_documents(sid: str) -> list[DocumentMeta]:
    d = _session_dir(sid)
    if d is None or not d.is_dir():
        return []
    metas = [m for p in d.glob("*.json") if (m := _read_meta(p)) is not None and get(sid, m.id) is not None]
    return sorted(metas, key=lambda m: (m.created_at, m.id))
