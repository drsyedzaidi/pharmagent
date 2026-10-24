"""Suite-wide test isolation.

1. Background jobs. ``app.main.jobs`` is one module-level JobManager shared by every HTTP test. A test that
   submits a job and returns without waiting leaves a worker thread running into later tests; a real fit then
   calls whatever ``population_fit`` the next test has monkeypatched, which made tests/test_simest_chat.py fail
   on slow CI runners. The guard below fails the test that leaves a job running, names it, and waits for the job
   so later tests start from an idle pool.
2. Mikebrain rule catalog. ``adversarial_review`` reads the repository's rule catalog. Once the owner approves a
   real rule, any test that runs a review on a state that happens to satisfy it would start producing an extra
   finding. So every test sees an EMPTY catalog unless it points ``catalog_io.DEFAULT_CATALOG`` somewhere itself
   (``test_knowledge_rules.py`` does, and reads the shipped file through ``SHIPPED_CATALOG``).
"""
from __future__ import annotations

import json
import sys
import time

import pytest

from app.knowledge.rules import catalog_io

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


@pytest.fixture(autouse=True)
def _empty_rule_catalog(tmp_path_factory, monkeypatch):
    path = tmp_path_factory.mktemp("rule_catalog") / "catalog.json"
    path.write_text(json.dumps({"catalog_version": "0.0.0", "rules": []}), encoding="utf-8")
    monkeypatch.setattr(catalog_io, "DEFAULT_CATALOG", path)
    catalog_io.cached_rule_catalog.cache_clear()
    yield
    catalog_io.cached_rule_catalog.cache_clear()
