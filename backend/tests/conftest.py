"""Test isolation for the app's shared background-job pool.

``app.main.jobs`` is one module-level JobManager shared by every HTTP test. A test that submits a job and returns
without waiting leaves a worker thread running into later tests; a real fit then calls whatever ``population_fit``
the next test has monkeypatched, which made tests/test_simest_chat.py fail on slow CI runners. This guard fails the
test that leaves a job running, names it, and waits for the job so later tests start from an idle pool."""
from __future__ import annotations

import sys
import time

import pytest

_DRAIN_SECONDS = 600


def _running(jobs) -> list[dict]:
    with jobs._lock:
        return [dict(j) for j in jobs._jobs.values() if j["status"] == "running"]


@pytest.fixture(autouse=True)
def _no_job_outlives_its_test(request):
    yield
    main = sys.modules.get("app.main")              # only tests that imported the app can have started a job
    if main is None:
        return
    left = _running(main.jobs)
    if not left:
        return
    deadline = time.monotonic() + _DRAIN_SECONDS
    while _running(main.jobs) and time.monotonic() < deadline:
        time.sleep(0.05)
    kinds = sorted(j["kind"] for j in left)
    pytest.fail(f"{request.node.nodeid} left {len(left)} background job(s) running {kinds}; wait for them "
                "(poll the job to completion) so they cannot run into later tests", pytrace=False)
