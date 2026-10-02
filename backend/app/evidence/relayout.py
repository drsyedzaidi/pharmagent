"""Layout rebuild for sources parsed before evidence-parse 7 (no ``<sha256>.layout.json``).

Re-uploading the same PDF dedupes by sha256 and never re-parses, so such a source keeps the whole-page
image (``layout_unavailable``) for ever. ``rebuild_layout`` re-parses the stored PDF in the sandboxed
child (the fair parse queue, the session id) and writes ONLY the sidecar, under one safety rule:

- every text-bearing field of the re-parse (page ``text``, ``small_text``, ``columns``, ``tables``,
  ``hidden_chars``, ``ink_check``, ``glyph_check``, plus sha256, decimal style, text length) must equal the
  stored ``pages.json`` byte for byte, and every page's ``text_sha256`` / ``col_sha256`` in the new
  layout must equal the hash of the stored text. Then the sidecar is written atomically (temp file in the
  same directory, fsync, rename: a kill leaves the old state, never a truncated file) and ``pages.json``,
  the ledger and every record stay untouched (nothing here writes to the evidence store).
- otherwise NOTHING is written to the cache: a marker file records ``text_changed`` (or ``unreadable``) with
  a reason, and the reviewer compares each value with the PDF itself.

``layout_info`` is the per-source status the ledger exposes: ``ready``, ``missing``, ``text_changed``,
``unreadable``. Idempotent: a ``ready`` source is not parsed again. The marker is content-level (keyed by
sha256 and parser) like the cache it describes.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

from app.evidence import ingest, parse_sandbox
from app.evidence.models import Source, now_iso, sha256_text
from app.evidence.parse import PARSER_NAME
from app.evidence.parse_sandbox import ParseBusyError

STATUSES = ("ready", "missing", "text_changed", "unreadable")
OUTCOMES = ("rebuilt", "unchanged", "text_changed", "unreadable")
MAX_BULK = 10                         # sources rebuilt per bulk call (each is a full parse)
STATUS_LRU = 64
REASON_MISSING = "Page positions were not stored when this source was parsed."
REASON_TEXT_CHANGED = ("The page text of this PDF now parses differently ({where}), so stored page positions "
                       "cannot be tied to the text the records cite. Upload the PDF to a new session, run "
                       "extraction there, and compare every value with the PDF itself.")
REASON_FAILED_PARSE = "The original parse failed, so there is no page text to attach positions to."
_ok_cache: OrderedDict[tuple[Any, ...], bool] = OrderedDict()
_ok_lock = threading.Lock()
_sha_locks: dict[str, threading.Lock] = {}
_sha_locks_guard = threading.Lock()


# ----------------------------------------------------------------- files ---

def layout_file(src: Source) -> Path:
    return Path(src.parse_path).with_name(f"{src.sha256}.layout.json")


def marker_file(src: Source) -> Path:
    return Path(src.parse_path).with_name(f"{src.sha256}.layout.status.json")


def write_atomic(path: Path, data: bytes) -> None:
    """Temp file beside *path*, flushed to disk, then renamed over it: readers see the old file or the
    new one, never a part of it."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        return None


def _stat_key(path: Path) -> tuple[str, int, int]:
    st = path.stat()
    return str(path), st.st_mtime_ns, st.st_size


# --------------------------------------------------------------- binding ---

def layout_binds(side: Any, src_sha: str, pages_parser: Any) -> bool:
    """True when sidecar *side* belongs to this PDF and to the current parser, and its parse is the stored
    cache's own (same parser) or was rebuilt for the stored cache's parser. Per page text hashes are
    checked separately (``regions.locate``: layout_stale)."""
    if not isinstance(side, dict) or side.get("sha256") != src_sha or side.get("parser") != PARSER_NAME:
        return False
    return pages_parser == side["parser"] or pages_parser == side.get("rebuilt_for")


def _page_binds(layout_page: Any, n: int, page: dict[str, Any]) -> bool:
    if not isinstance(layout_page, dict) or layout_page.get("n") != n:
        return False
    cols = page.get("columns")
    want_cols = [sha256_text(str(c)) for c in cols] if cols else None
    return (layout_page.get("text_sha256") == sha256_text(str(page.get("text") or ""))
            and layout_page.get("col_sha256") == want_cols)


def _pages_of(doc: Any) -> list[dict[str, Any]]:
    pages = doc.get("pages") if isinstance(doc, dict) else None
    return pages if isinstance(pages, list) and all(isinstance(p, dict) for p in pages) else []


def _all_pages_bind(side: dict[str, Any], doc: dict[str, Any]) -> bool:
    lpages, pages = side.get("pages"), _pages_of(doc)
    return (isinstance(lpages, list) and len(lpages) == len(pages)
            and all(_page_binds(lp, i + 1, p) for i, (lp, p) in enumerate(zip(lpages, pages, strict=True))))


def _sidecar_ok(src: Source) -> bool:
    """A sidecar that parses, is bound to this PDF, the current parser and every page text (cached by file
    stat, so a ledger snapshot does not re-read megabytes per call)."""
    side_path, pages_path = layout_file(src), Path(src.parse_path)
    try:
        key = (_stat_key(side_path), _stat_key(pages_path), src.sha256)
    except OSError:
        return False
    with _ok_lock:
        if key in _ok_cache:
            _ok_cache.move_to_end(key)
            return _ok_cache[key]
    side, doc = _read_json(side_path), _read_json(pages_path)
    ok = (isinstance(doc, dict) and doc.get("sha256") == src.sha256 and "error" not in doc
          and layout_binds(side, src.sha256, doc.get("parser")) and _all_pages_bind(side, doc))
    with _ok_lock:
        _ok_cache[key] = ok
        while len(_ok_cache) > STATUS_LRU:
            _ok_cache.popitem(last=False)
    return ok


# ---------------------------------------------------------------- status ---

def _marker(src: Source) -> dict[str, Any] | None:
    m = _read_json(marker_file(src))
    if (isinstance(m, dict) and m.get("sha256") == src.sha256 and m.get("parser") == PARSER_NAME
            and m.get("status") in ("text_changed", "unreadable")):
        return m
    return None


def layout_info(src: Source) -> dict[str, Any]:
    """{'layout_status': ready|missing|text_changed|unreadable, 'layout_reason': why, or None when ready}."""
    if src.parse_status == "failed":
        return {"layout_status": "unreadable", "layout_reason": REASON_FAILED_PARSE}
    if _sidecar_ok(src):
        return {"layout_status": "ready", "layout_reason": None}
    mark = _marker(src)
    if mark is not None:
        return {"layout_status": mark["status"], "layout_reason": str(mark.get("reason") or "")}
    return {"layout_status": "missing", "layout_reason": REASON_MISSING}


def rebuildable(src: Source) -> bool:
    """A rebuild would help: no usable sidecar and no recorded reason it cannot be made."""
    return layout_info(src)["layout_status"] == "missing"


# --------------------------------------------------------------- rebuild ---

def first_difference(stored: dict[str, Any], new: dict[str, Any]) -> str | None:
    """Where the re-parse's text-bearing fields stop equalling the stored cache's, or None when identical."""
    for key in ("sha256", "decimal_style", "text_chars"):
        if stored.get(key) != new.get(key):
            return key
    old_pages, new_pages = _pages_of(stored), _pages_of(new)
    if len(old_pages) != len(new_pages):
        return f"page count {len(old_pages)} -> {len(new_pages)}"
    for i, (a, b) in enumerate(zip(old_pages, new_pages, strict=True), start=1):
        if a != b:
            field = next((k for k in sorted({*a, *b}) if a.get(k) != b.get(k)), "page")
            return f"page {i} {field}"
    return None


def _sha_lock(sha: str) -> threading.Lock:
    with _sha_locks_guard:
        return _sha_locks.setdefault(sha, threading.Lock())


def _record(src: Source, status: str, reason: str) -> None:
    body = {"sha256": src.sha256, "parser": PARSER_NAME, "status": status, "reason": reason, "at": now_iso()}
    write_atomic(marker_file(src), json.dumps(body).encode("utf-8"))


def _page_hashes(doc: dict[str, Any] | None) -> list[str]:
    return [sha256_text(str(p.get("text") or "")) for p in _pages_of(doc)]


def _outcome(src: Source, outcome: str, reason: str | None, doc: dict[str, Any] | None) -> dict[str, Any]:
    side = layout_file(src)
    side_sha = ingest.file_sha256(side) if outcome in ("rebuilt", "unchanged") and side.is_file() else None
    hashes = _page_hashes(doc)
    return {"source_id": src.id, "outcome": outcome, "reason": reason, "sidecar_sha256": side_sha,
            "n_pages": len(hashes), "page_text_sha256": hashes,
            "pages_sha256": sha256_text("\n".join(hashes)) if hashes else None,
            **layout_info(src)}


def _refuse(src: Source, status: str, reason: str, doc: dict[str, Any] | None) -> dict[str, Any]:
    _record(src, status, reason)
    return _outcome(src, status, reason, doc)


def rebuild_layout(src: Source, session_id: str) -> dict[str, Any]:
    """Rebuild *src*'s layout sidecar (module docstring). Returns the outcome dict; ParseBusyError (no
    parse slot) propagates and nothing is recorded. Writes nothing but the sidecar and its status marker."""
    with _sha_lock(src.sha256):
        if src.parse_status != "failed" and _sidecar_ok(src):
            return _outcome(src, "unchanged", None, _read_json(Path(src.parse_path)))
        try:
            stored = ingest.load_pages(src)
        except (OSError, ValueError) as exc:
            return _refuse(src, "unreadable", f"The stored PDF or its parse cache cannot be read: {exc}", None)
        if src.parse_status == "failed" or "error" in stored:
            return _refuse(src, "unreadable", REASON_FAILED_PARSE, stored)
        try:
            doc = parse_sandbox.parse_isolated(Path(src.pdf_path), session_id=session_id)
        except ParseBusyError:
            raise
        except Exception as exc:                  # encrypted, corrupt, over a limit, killed: not rebuildable
            return _refuse(src, "unreadable", f"The PDF could not be parsed again ({type(exc).__name__}: "
                           f"{str(exc)[:200]}).", stored)
        layout = doc.pop("layout", None)
        where = first_difference(stored, doc)
        if where is not None:
            return _refuse(src, "text_changed", REASON_TEXT_CHANGED.format(where=where), stored)
        if not isinstance(layout, dict) or not _all_pages_bind(layout, stored):
            return _refuse(src, "unreadable", "The re-parsed page positions do not match the stored page text.",
                           stored)
        out = {**layout, "rebuilt_for": stored.get("parser")}
        write_atomic(layout_file(src), json.dumps(out, ensure_ascii=False).encode("utf-8"))
        marker_file(src).unlink(missing_ok=True)
        return _outcome(src, "rebuilt", None, stored)


def pending_sources(sources: list[Source]) -> list[Source]:
    """Sources a bulk rebuild should try: no sidecar and no recorded reason it cannot be made."""
    return [s for s in sources if rebuildable(s)]
