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
