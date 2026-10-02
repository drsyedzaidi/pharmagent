"""Stand-in parse targets for the sandbox tests (each runs inside the child process)."""
from __future__ import annotations

import os
import time
from typing import Any


def report_pid(_path: str) -> dict[str, Any]:
    return {"pid": os.getpid()}


def sleep_forever(_path: str) -> dict[str, Any]:
    while True:
        time.sleep(0.05)


def hog_memory(_path: str) -> dict[str, Any]:
    blocks = []
    while True:                                   # ~50 MB per step, written so it is resident
        blocks.append(bytearray(b"x" * (50 * 2**20)))
        time.sleep(0.02)


def sleep_with_args(_path: str, **_kw: Any) -> dict[str, Any]:
    while True:
        time.sleep(0.05)


def hog_memory_with_args(_path: str, **_kw: Any) -> dict[str, Any]:
    return hog_memory(_path)


def report_env(_path: str, **kw: Any) -> dict[str, Any]:
    return {"env": sorted(os.environ), "kwargs": kw}
