"""Upload admission. Per session: a source-count cap, a stored-bytes cap and a sliding-window rate
limit, checked before anything is staged (and the byte cap again once the size is known), so one
session cannot queue unbounded parse work or disk use. Refusals are HTTP 429 (rate) / 413 (caps).

Server-wide: at most MAX_UPLOADS_IN_FLIGHT uploads at once (503 at once beyond that, whatever the
session), and their blocking part (hash, validation, the wait for a parse slot, the parse) runs on
its own UPLOAD_WORKERS-token limiter, never on the shared threadpool every synchronous endpoint uses:
a flood of uploads from many sessions waits without holding a thread and cannot starve the API."""
from __future__ import annotations

import asyncio
import threading
import time
import weakref
from collections import deque
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import anyio
import anyio.to_thread
from fastapi import HTTPException

from app.evidence.store import get_store

MAX_SOURCES_PER_SESSION = 200
MAX_SESSION_BYTES = 1024 * 1024 * 1024
UPLOADS_PER_WINDOW, WINDOW_S = 30, 60.0
UPLOAD_WORKERS = 4                 # threads finishing uploads (2 parse slots + hashing / validation)
MAX_UPLOADS_IN_FLIGHT = 32         # all sessions together; the rest are refused with 503 at once

_lock = threading.Lock()
_recent: dict[str, deque[float]] = {}
_in_flight = [0]
_limiters: weakref.WeakKeyDictionary[Any, anyio.CapacityLimiter] = weakref.WeakKeyDictionary()


@asynccontextmanager
async def upload_slot() -> AsyncIterator[None]:
    """Hold one of the server-wide upload slots for the body of the block; 503 when none is free."""
    with _lock:
        if _in_flight[0] >= MAX_UPLOADS_IN_FLIGHT:
            raise HTTPException(503, f"the server is processing {MAX_UPLOADS_IN_FLIGHT} uploads; "
                                     "try again shortly")
        _in_flight[0] += 1
    try:
        yield
    finally:
        with _lock:
            _in_flight[0] -= 1


async def run_blocking(fn: Callable[..., Any], *args: Any) -> Any:
    """fn(*args) in a worker thread under the upload limiter (one per event loop), not the shared pool."""
    loop = asyncio.get_running_loop()
    with _lock:
        limiter = _limiters.get(loop)
        if limiter is None:
            limiter = _limiters[loop] = anyio.CapacityLimiter(UPLOAD_WORKERS)
    return await anyio.to_thread.run_sync(fn, *args, limiter=limiter)


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
