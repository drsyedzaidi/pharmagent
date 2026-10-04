"""The one way evidence admission limits refuse a request: 429 / 503 always carry ``Retry-After``.

The global HTTPException handler forwards ``exc.headers`` (app/response_headers.py), so a refusal built here
tells the client when to come back, like the parse queue (ParseBusyError) does.
"""
from __future__ import annotations

import math

from fastapi import HTTPException

__all__ = ["refuse", "seconds_until_window_opens"]


def refuse(status: int, message: str, retry_after: int) -> HTTPException:
    """HTTPException(*status*) with ``Retry-After: retry_after`` (whole seconds, at least 1)."""
    return HTTPException(status, message, headers={"Retry-After": str(max(1, int(retry_after)))})


def seconds_until_window_opens(window_s: float, now: float, oldest: float) -> int:
    """Whole seconds until the oldest request of a sliding window falls out of it (at least 1)."""
    return max(1, math.ceil(window_s - (now - oldest)))
