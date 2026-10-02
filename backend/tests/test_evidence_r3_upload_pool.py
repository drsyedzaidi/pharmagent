"""Round-3 attack regression: an upload flood across many sessions must not starve the shared
threadpool that every synchronous endpoint runs on (Starlette's default 40 tokens).

The blocking part of an upload (hash, validate, wait for a parse slot, parse) is replaced by a
function that blocks until released, standing in for uploads queued behind the 2 parse slots.
"""
from __future__ import annotations

import itertools
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.config import settings
from app.core.llm import MockLLM
from app.core.orchestrator import Orchestrator
from app.core.store import SessionStore
from app.evidence import routes, upload_limits
from app.evidence import store as evstore

EV = "/api/sessions/{sid}/evidence"
PDF = b"%PDF-1.4\n% tiny\n"
N_SESSIONS, PER_SESSION = 3, 15                # 45 uploads: more than the 40 shared tokens


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    evstore.configure(":memory:")
    upload_limits.reset()
    main.orch = Orchestrator(llm=MockLLM(), clock=lambda c=itertools.count(): f"t{next(c)}",
                             store=SessionStore(":memory:"))
    with TestClient(main.app) as c:              # one event loop (and one shared threadpool) for all requests
        yield c
    upload_limits.reset()


def test_upload_flood_does_not_starve_sync_endpoints(client, monkeypatch):
    release, running, peak, lock = threading.Event(), [0], [0], threading.Lock()

    def blocked_finish(*_args, **_kw):
        with lock:
            running[0] += 1
            peak[0] = max(peak[0], running[0])
        try:
            release.wait(20)
            return {"source": None, "duplicate": True, "summary": "stub"}
        finally:
            with lock:
                running[0] -= 1

    monkeypatch.setattr(routes, "_finish_upload", blocked_finish)
    sids = [client.post("/api/sessions").json()["id"] for _ in range(N_SESSIONS)]

    def upload(sid: str) -> int:
        return client.post(EV.format(sid=sid) + "/upload",
                           files={"file": ("p.pdf", PDF, "application/pdf")}).status_code

    with ThreadPoolExecutor(max_workers=N_SESSIONS * PER_SESSION) as pool:
        futures = [pool.submit(upload, sid) for sid in sids for _ in range(PER_SESSION)]
        try:
            time.sleep(2.0)                      # every upload is in flight, queued or refused
            start = time.monotonic()
            health = client.get("/api/health")
            elapsed = time.monotonic() - start
        finally:
            release.set()
        codes = [f.result(timeout=30) for f in futures]
    assert health.status_code == 200
    assert elapsed < 0.5, f"/api/health took {elapsed:.2f} s during the upload flood"
    assert peak[0] <= upload_limits.UPLOAD_WORKERS, peak[0]
    assert set(codes) <= {200, 503}, codes
    assert codes.count(200) >= upload_limits.UPLOAD_WORKERS
