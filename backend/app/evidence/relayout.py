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
``unreadable``. Idempotent: a ``ready`` source is not parsed again, and a refused one is not parsed again
until the stored parse cache changes (the marker carries the cache's stat key), so repeated calls cost one
parse. The marker is content-level (keyed by sha256 and parser) like the cache it describes. A reason is
always a fixed sentence: exception text and server paths go to the log, never to the marker or the listing.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

from app.evidence import ingest, parse_sandbox
from app.evidence.models import Source, now_iso, sha256_text
from app.evidence.parse import PARSER_NAME
from app.evidence.parse_guard import ParseLimitError
from app.evidence.parse_sandbox import ParseAbortedError, ParseBusyError
from app.evidence.path_guard import OutsideDataDirError

MAX_BULK = 10                         # sources rebuilt per bulk call (each is a full parse)
STATUS_LRU = 4096                     # >= a session's 200-source cap for many sessions (tuples of a few bytes)
REASON_MISSING = "Page positions were not stored when this source was parsed."
REASON_TEXT_CHANGED = ("The page text of this PDF now parses differently ({where}), so stored page positions "
                       "cannot be tied to the text the records cite. Upload the PDF to a new session, run "
                       "extraction there, and compare every value with the PDF itself.")
REASON_FAILED_PARSE = "The original parse failed, so there is no page text to attach positions to."
REASON_UNREADABLE_STORE = "The stored PDF or its parse cache cannot be read."
REASON_OUTSIDE = "The stored files of this source lie outside the evidence directory, so they were not read."
REASON_REPARSE = "The PDF could not be parsed again ({why})."
REASON_NO_MATCH = "The re-parsed page positions do not match the stored page text."
log = logging.getLogger(__name__)
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


def _pages_stat(src: Source) -> list[int] | None:
    """[mtime_ns, size] of the stored parse cache: what a recorded refusal was made against."""
    try:
        return list(_stat_key(Path(src.parse_path))[1:])
    except OSError:
        return None


def _record(src: Source, status: str, reason: str) -> None:
    body = {"sha256": src.sha256, "parser": PARSER_NAME, "status": status, "reason": reason, "at": now_iso(),
            "pages_stat": _pages_stat(src)}
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


def _why_not_parsed(exc: Exception) -> str:
    """A fixed phrase for a failed re-parse; the exception itself (it may carry paths) goes to the log."""
    log.info("layout rebuild: re-parse failed (%s): %s", type(exc).__name__, str(exc)[:300])
    if isinstance(exc, ParseAbortedError):
        return "the parse was stopped at its time or memory limit"
    if isinstance(exc, ParseLimitError):
        return "the PDF is over a parse limit"
    if isinstance(exc, FileNotFoundError):
        return "the stored PDF file is missing"
    return "the parser could not read it"


def _remembered(src: Source) -> dict[str, Any] | None:
    """The recorded refusal of this PDF and parser while the stored parse cache is the one it was made
    against (an older marker without the stat key, or a rewritten cache, is not trusted)."""
    mark = _marker(src)
    stat = _pages_stat(src)
    if mark is None or stat is None or mark.get("pages_stat") != stat:
        return None
    return mark


def _outside(src: Source) -> dict[str, Any]:
    """The refusal for a source whose stored paths leave the evidence directory: it touches no file (not even
    the status marker, which would be written beside the forged path), so nothing is recorded."""
    return {"source_id": src.id, "outcome": "unreadable", "reason": REASON_OUTSIDE, "sidecar_sha256": None,
            "n_pages": 0, "page_text_sha256": [], "pages_sha256": None,
            "layout_status": "unreadable", "layout_reason": REASON_OUTSIDE}


def outside_outcome(src: Source, data_dir: str | Path | None) -> dict[str, Any] | None:
    """The 'unreadable' outcome when *src*'s stored paths leave the evidence directory of *data_dir*, else None
    (also None without a data_dir). Touches no file."""
    if data_dir is None:
        return None
    try:
        ingest.stored_paths(src, data_dir)
    except OutsideDataDirError:
        return _outside(src)
    return None


def rebuild_layout(src: Source, session_id: str, data_dir: str | Path | None = None) -> dict[str, Any]:
    """Rebuild *src*'s layout sidecar (module docstring). Returns the outcome dict; ParseBusyError (no
    parse slot) propagates and nothing is recorded. Writes nothing but the sidecar and its status marker.
    With a *data_dir* the stored PDF and parse paths must lie inside its evidence directory, checked before
    anything is read, parsed or written; a source that does not is 'unreadable' and nothing is recorded."""
    if (refused := outside_outcome(src, data_dir)) is not None:
        return refused
    with _sha_lock(src.sha256):
        if src.parse_status != "failed" and _sidecar_ok(src):
            return _outcome(src, "unchanged", None, _read_json(Path(src.parse_path)))
        if src.parse_status != "failed" and (mark := _remembered(src)) is not None:
            return _outcome(src, mark["status"], str(mark.get("reason") or ""), _read_json(Path(src.parse_path)))
        try:
            stored = ingest.load_pages(src, data_dir)
        except (OSError, ValueError) as exc:
            log.info("layout rebuild: stored parse unreadable: %s", exc)
            return _refuse(src, "unreadable", REASON_UNREADABLE_STORE, None)
        if src.parse_status == "failed" or "error" in stored:
            return _refuse(src, "unreadable", REASON_FAILED_PARSE, stored)
        try:
            pdf_path = Path(src.pdf_path) if data_dir is None else ingest.stored_paths(src, data_dir)[0]
            doc = parse_sandbox.parse_isolated(pdf_path, session_id=session_id)
        except ParseBusyError:
            raise
        except Exception as exc:                  # encrypted, corrupt, over a limit, killed: not rebuildable
            return _refuse(src, "unreadable", REASON_REPARSE.format(why=_why_not_parsed(exc)), stored)
        layout = doc.pop("layout", None)
        where = first_difference(stored, doc)
        if where is not None:
            return _refuse(src, "text_changed", REASON_TEXT_CHANGED.format(where=where), stored)
        if not isinstance(layout, dict) or not _all_pages_bind(layout, stored):
            return _refuse(src, "unreadable", REASON_NO_MATCH, stored)
        out = {**layout, "rebuilt_for": stored.get("parser")}
        write_atomic(layout_file(src), json.dumps(out, ensure_ascii=False).encode("utf-8"))
        marker_file(src).unlink(missing_ok=True)
        return _outcome(src, "rebuilt", None, stored)


def pending_sources(sources: list[Source]) -> list[Source]:
    """Sources a bulk rebuild should try: no sidecar and no recorded reason it cannot be made."""
    return [s for s in sources if rebuildable(s)]


# ------------------------------------------------------------- hand-over ---
# The HTTP endpoint runs the (slow) rebuild first, outside the session lock, and then lets the audited
# ``rebuild_source_layout`` tool call (which takes that lock) pick the finished outcome up instead of parsing
# under the lock. Keyed by session and source; consumed once; the tool parses itself when nothing is waiting
# (a call from chat).
_handover: dict[tuple[str, str], dict[str, Any]] = {}
_handover_lock = threading.Lock()


def hand_over(session_id: str, source_id: str, outcome: dict[str, Any]) -> None:
    with _handover_lock:
        _handover[(session_id, source_id)] = outcome


def take_handover(session_id: str, source_id: str) -> dict[str, Any] | None:
    with _handover_lock:
        return _handover.pop((session_id, source_id), None)
