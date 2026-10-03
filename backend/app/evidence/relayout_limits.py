"""Admission for the layout rebuild endpoint (``POST .../sources/{id}/relayout``).

A rebuild is a full sandboxed parse (up to 150 s, one parse slot), so the endpoint is limited like an upload,
but with its own numbers and its own worker limiter:

- per session: one rebuild at a time and a sliding-window rate (429 + Retry-After beyond either);
- server-wide: at most MAX_IN_FLIGHT rebuilds at once (503 + Retry-After at once beyond that);
- the blocking part runs on a RELAYOUT_WORKERS-token limiter that is neither the shared threadpool nor the
  upload limiter, so a relayout flood cannot delay another session's upload.
"""
from __future__ import annotations

import asyncio
import threading
import time
import weakref
from collections import deque
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

import anyio
import anyio.to_thread
from fastapi import HTTPException

RELAYOUTS_PER_WINDOW, WINDOW_S = 12, 60.0     # rebuild requests per session per window
MAX_IN_FLIGHT = 4                              # all sessions together
RELAYOUT_WORKERS = 2                           # threads running rebuilds (the parse itself has 2 slots)
RETRY_AFTER_S = 5

_lock = threading.Lock()
_recent: dict[str, deque[float]] = {}
_busy: set[str] = set()
_in_flight = [0]
_limiters: weakref.WeakKeyDictionary[Any, anyio.CapacityLimiter] = weakref.WeakKeyDictionary()


def _refuse(status: int, message: str, retry_after: int = RETRY_AFTER_S) -> HTTPException:
    return HTTPException(status, message, headers={"Retry-After": str(retry_after)})


@asynccontextmanager
async def relayout_slot(sid: str) -> AsyncIterator[None]:
    """Admit one rebuild of session *sid* for the body of the block, or raise 429 / 503 before any work."""
    now = time.monotonic()
    with _lock:
        if sid in _busy:
            raise _refuse(429, "a layout rebuild of this session is already running; wait for it to finish")
        window = _recent.setdefault(sid, deque())
        while window and now - window[0] > WINDOW_S:
            window.popleft()
        if len(window) >= RELAYOUTS_PER_WINDOW:
            wait = int(WINDOW_S - (now - window[0])) + 1
            raise _refuse(429, f"too many layout rebuilds: at most {RELAYOUTS_PER_WINDOW} per "
                               f"{WINDOW_S:.0f} s per session; try again shortly", wait)
        if _in_flight[0] >= MAX_IN_FLIGHT:
            raise _refuse(503, f"the server is rebuilding {MAX_IN_FLIGHT} layouts; try again shortly")
        window.append(now)
        _busy.add(sid)
        _in_flight[0] += 1
    try:
        yield
    finally:
        with _lock:
            _busy.discard(sid)
            _in_flight[0] -= 1


async def run_blocking(fn: Callable[..., Any], *args: Any) -> Any:
    """fn(*args) in a worker thread under the relayout limiter (one per event loop)."""
    loop = asyncio.get_running_loop()
    with _lock:
        limiter = _limiters.get(loop)
        if limiter is None:
            limiter = _limiters[loop] = anyio.CapacityLimiter(RELAYOUT_WORKERS)
    return await anyio.to_thread.run_sync(fn, *args, limiter=limiter)


def reset() -> None:
    """Forget the windows and counters (tests)."""
    with _lock:
        _recent.clear()
        _busy.clear()
        _in_flight[0] = 0
