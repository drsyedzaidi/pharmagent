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


def raise_runtime(_path: str) -> dict[str, Any]:
    raise RuntimeError("boom " + "x" * 800)            # longer than the 500 characters a report keeps


def raise_missing(path: str) -> dict[str, Any]:
    raise FileNotFoundError(f"no such file: {path}")


def raise_limit(_path: str) -> dict[str, Any]:
    from app.evidence.parse_guard import ParseLimitError
    raise ParseLimitError("glyph budget exceeded")


def raise_system_exit(_path: str) -> dict[str, Any]:
    raise SystemExit("asked to stop")                  # a BaseException: reported, not lost


def die_without_answer(_path: str) -> dict[str, Any]:
    os._exit(3)                                        # no result file is ever written


def echo_args(path: str, **kw: Any) -> dict[str, Any]:
    return {"path": path, "kwargs": kw}
