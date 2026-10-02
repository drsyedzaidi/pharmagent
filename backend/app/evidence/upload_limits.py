"""Per-session upload admission: a source-count cap, a stored-bytes cap and a sliding-window rate
limit, checked before anything is staged (and the byte cap again once the size is known), so one
session cannot queue unbounded parse work or disk use. Refusals are HTTP 429 (rate) / 413 (caps)."""
from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path

from fastapi import HTTPException

from app.evidence.store import get_store

MAX_SOURCES_PER_SESSION = 200
MAX_SESSION_BYTES = 1024 * 1024 * 1024
UPLOADS_PER_WINDOW, WINDOW_S = 30, 60.0

_lock = threading.Lock()
_recent: dict[str, deque[float]] = {}


def _stored_bytes(sid: str) -> int:
    total = 0
    for src in get_store().list_sources(sid):
        try:
            total += Path(src.pdf_path).stat().st_size
        except OSError:
            continue
    return total


def admit(sid: str) -> None:
    """Refuse the upload before staging when the session is over a cap or over the rate."""
    if len(get_store().list_sources(sid)) >= MAX_SOURCES_PER_SESSION:
        raise HTTPException(413, f"this session already holds {MAX_SOURCES_PER_SESSION} publications (limit)")
    if _stored_bytes(sid) >= MAX_SESSION_BYTES:
        raise HTTPException(413, f"this session's publications exceed {MAX_SESSION_BYTES // 2**20} MB (limit)")
    now = time.monotonic()
    with _lock:
        window = _recent.setdefault(sid, deque())
        while window and now - window[0] > WINDOW_S:
            window.popleft()
        if len(window) >= UPLOADS_PER_WINDOW:
            raise HTTPException(429, f"too many uploads: at most {UPLOADS_PER_WINDOW} per "
                                     f"{WINDOW_S:.0f} s per session; try again shortly")
        window.append(now)


def admit_size(sid: str, new_bytes: int) -> None:
    """Once the staged size is known: refuse when it would take the session past the byte cap."""
    if _stored_bytes(sid) + new_bytes > MAX_SESSION_BYTES:
        raise HTTPException(413, f"this upload would take the session past {MAX_SESSION_BYTES // 2**20} MB (limit)")


def reset() -> None:
    """Forget the rate windows (tests)."""
    with _lock:
        _recent.clear()
