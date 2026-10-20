"""Quotas across real processes, barrier-synchronised.

Six interpreters are released from a barrier at the same instant and each makes one reservation. ``widen`` holds every
process between "read usage" and "create reservation", so WITHOUT the storage lock they all read the same usage and all
succeed. Each scenario therefore runs twice: with the lock the budget holds exactly, and with the lock bypassed the same
harness overshoots it (the canary), which is what proves the test fails when locking is missing."""
from __future__ import annotations

import multiprocessing

import pytest

from app.config import settings
from app.pageimage import quota

WORKERS = 6
WIDEN = 0.3


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    return tmp_path / "data" / "documents"


def _race(tmp_path, sids, nbytes, overrides, bypass):
    ctx = multiprocessing.get_context("spawn")
    barrier, queue = ctx.Barrier(WORKERS), ctx.Queue()
    from tests import _quota_workers
    procs = [ctx.Process(target=_quota_workers.run,
                         args=(str(tmp_path / "data"), sids[i], nbytes, overrides, bypass, WIDEN, barrier, queue))
             for i in range(WORKERS)]
    for p in procs:
        p.start()
    results = [queue.get(timeout=120) for _ in procs]
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0
    return results


SCENARIOS = {
    "document count": (["s1"] * WORKERS, 10, {"MAX_DOCS_PER_SESSION": 3}, "count", 3),
    "session bytes": (["s1"] * WORKERS, 30, {"MAX_SESSION_BYTES": 100}, "session", 3),
    "global bytes across sessions": ([f"s{i}" for i in range(WORKERS)], 30, {"MAX_GLOBAL_BYTES": 100}, "global", 3),
}


@pytest.mark.parametrize("name", SCENARIOS)
def test_the_lock_keeps_every_budget_exact_across_processes(name, root, tmp_path):
    sids, nbytes, overrides, refusal, allowed = SCENARIOS[name]
    results = _race(tmp_path, sids, nbytes, overrides, bypass=False)
    assert sorted(results) == sorted(["ok"] * allowed + [refusal] * (WORKERS - allowed))
    per, total = quota.usage(root)
    assert sum(n for n, _ in per.values()) == allowed and total == allowed * nbytes


@pytest.mark.parametrize("name", SCENARIOS)
def test_canary_without_the_lock_the_same_race_overshoots(name, root, tmp_path):
    sids, nbytes, overrides, _, allowed = SCENARIOS[name]
    results = _race(tmp_path, sids, nbytes, overrides, bypass=True)
    assert results.count("ok") > allowed, "the harness can no longer see a missing lock"
