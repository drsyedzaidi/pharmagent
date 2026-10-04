"""Admission for the page-image route: a per-session sliding window over UNCACHED renders (cache hits are
free) and a server-wide cap on renders in flight, so one session cannot queue unbounded render work
and a flood waits at most RENDER_QUEUE_WAIT_S (render.py) or is refused at once (503)."""
from __future__ import annotations

import asyncio
import threading
import time
import weakref
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import anyio
import anyio.to_thread

from app.evidence.busy import refuse, seconds_until_window_opens

RENDERS_PER_WINDOW, WINDOW_S = 60, 60.0
MAX_RENDERS_IN_FLIGHT = 8          # running + waiting for a slot, all sessions together
RENDER_WORKERS = 4                 # threads of the route's own limiter (not the shared pool)
RETRY_AFTER_S = 3

_lock = threading.Lock()
_recent: dict[str, deque[float]] = {}
_in_flight = [0]
_limiters: weakref.WeakKeyDictionary[Any, anyio.CapacityLimiter] = weakref.WeakKeyDictionary()


def admit(sid: str) -> None:
    """Count one uncached render against *sid*; 429 + Retry-After when the window is full."""
    now = time.monotonic()
    with _lock:
        window = _recent.setdefault(sid, deque())
        while window and now - window[0] > WINDOW_S:
            window.popleft()
        if len(window) >= RENDERS_PER_WINDOW:
            raise refuse(429, f"too many page renders: at most {RENDERS_PER_WINDOW} per "
                              f"{WINDOW_S:.0f} s per session; try again shortly",
                         seconds_until_window_opens(WINDOW_S, now, window[0]))
        window.append(now)


@contextmanager
def in_flight() -> Iterator[None]:
    """Hold one of the MAX_RENDERS_IN_FLIGHT places; 503 at once when none is free."""
    with _lock:
        if _in_flight[0] >= MAX_RENDERS_IN_FLIGHT:
            raise refuse(503, "the server is rendering many pages; try again shortly", RETRY_AFTER_S)
        _in_flight[0] += 1
    try:
        yield
    finally:
        with _lock:
            _in_flight[0] -= 1


async def run_blocking(fn: Callable[..., Any], *args: Any) -> Any:
    """fn(*args) in a worker thread under the route's own limiter (one per event loop)."""
    loop = asyncio.get_running_loop()
    with _lock:
        limiter = _limiters.get(loop)
        if limiter is None:
            limiter = _limiters[loop] = anyio.CapacityLimiter(RENDER_WORKERS)
    return await anyio.to_thread.run_sync(fn, *args, limiter=limiter)


def reset() -> None:
    """Forget the windows and the in-flight count (tests)."""
    with _lock:
        _recent.clear()
        _in_flight[0] = 0
