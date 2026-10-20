"""Spawn-safe workers for the render-slot process tests (no app.main import)."""
from __future__ import annotations

import time
from pathlib import Path


def _setup(data_dir, overrides):
    from app.config import settings
    from app.pageimage import limits
    settings.data_dir = Path(data_dir)
    limits.QUEUE_WAIT_SECONDS = 0.2
    for k, v in overrides.items():
        setattr(limits, k, v)


def hold_forever(data_dir, session, ready):
    _setup(data_dir, {})
    from app.pageimage import slots
    with slots.render_slot(session, wait=5):
        ready.set()
        time.sleep(600)


def try_and_hold(data_dir, session, bypass, overrides, barrier, queue):
    """Try for a slot with a tiny wait; whoever gets one HOLDS it until everyone has tried, so the outcomes are exact."""
    _setup(data_dir, overrides)
    from app.pageimage import slots
    if bypass:
        slots.fcntl = type("NoLock", (), {"LOCK_EX": 0, "LOCK_NB": 0, "LOCK_UN": 0,
                                          "flock": staticmethod(lambda *a, **k: None)})
    barrier_ok = False
    try:
        with slots.render_slot(session, wait=0.2):
            queue.put("ok")
            barrier.wait(timeout=60)
            barrier_ok = True
    except slots.SlotBusy:
        queue.put("busy")
        barrier.wait(timeout=60)
        barrier_ok = True
    assert barrier_ok
