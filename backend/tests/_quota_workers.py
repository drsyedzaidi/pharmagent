"""Worker for the process-level quota tests. Kept free of ``app.main`` so a spawned interpreter starts fast."""
from __future__ import annotations

import contextlib
import time
from pathlib import Path


def run(data_dir: str, sid: str, nbytes: int, overrides: dict, bypass_lock: bool, widen: float, barrier, queue) -> None:
    """Wait at the barrier, then make ONE reservation and report its outcome.

    ``widen`` sleeps between reading usage and creating the reservation, so without the lock every process reads the same
    usage before any of them writes: the check-then-act race becomes deterministic instead of a matter of timing.
    ``bypass_lock`` replaces the storage lock by a no-op, which is the canary that proves the test can see the race."""
    from app.config import settings
    from app.pageimage import documents, limits, quota

    settings.data_dir = Path(data_dir)
    for name, value in overrides.items():
        setattr(limits, name, value)
    if bypass_lock:
        quota.locked = lambda root: contextlib.nullcontext()
    if widen:
        real_usage = quota.usage

        def slow_usage(root):
            result = real_usage(root)
            time.sleep(widen)
            return result
        quota.usage = slow_usage
    barrier.wait(timeout=60)
    try:
        documents.new_slot(sid, nbytes)
        outcome = "ok"
    except documents.DocumentLimitError:
        outcome = "count"
    except documents.QuotaExceeded as exc:
        outcome = exc.scope
    except documents.QuotaUnavailable:
        outcome = "unavailable"
    queue.put(outcome)
