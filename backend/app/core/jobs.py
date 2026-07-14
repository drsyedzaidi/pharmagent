"""Background job execution for long-running tools (NLME, SCM).

These tools run for minutes; submitting them as background jobs lets the HTTP
request return immediately with a ``job_id`` the client polls, instead of
holding one connection open for the whole computation and freezing the UI.

Jobs run in a small ``ThreadPoolExecutor``. The heavy compute is pure-Python
numpy/scipy, so the GIL limits *true* CPU parallelism — the win here is a
responsive request/UI and the ability to serve I/O-bound endpoints (health,
state, polling) while a fit proceeds. For genuine multi-core parallelism a
process pool would be needed, but that cannot share the in-process session
state these jobs mutate, so it is deliberately out of scope.

The registry is in-memory: jobs do not survive a restart (their *results* are
written to the persisted session state by the tool itself, so the analysis
output is not lost — only the transient job record is).
"""
from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

# Keep at most this many finished jobs; prune oldest-finished beyond it.
_MAX_JOBS = 200


class JobRejected(Exception):
    """Admission control declined a job (too many in flight). Maps to HTTP 429."""


_MAX_INFLIGHT = 32       # global cap on concurrently live (running/queued) jobs
_MAX_PER_SESSION = 2     # per-session cap — blocks a client flooding one session


class JobManager:
    """Submit callables to a thread pool and poll their status/result."""

    def __init__(self, max_workers: int = 4,
                 clock: Callable[[], str] | None = None) -> None:
        self._pool = ThreadPoolExecutor(max_workers=max_workers,
                                        thread_name_prefix="pharmjob")
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._clock = clock or (lambda: "")
        self._seq = 0

    def submit(self, *, session_id: str, kind: str,
               fn: Callable[[], Any]) -> str:
        """Schedule ``fn`` to run in the pool; return a job id to poll."""
        job_id = f"job_{uuid.uuid4().hex[:10]}"
        with self._lock:
            # Admission control: bound live jobs globally and per session so a
            # client cannot flood the unbounded executor queue (memory/CPU DoS).
            live = [j for j in self._jobs.values() if j["status"] == "running"]
            if len(live) >= _MAX_INFLIGHT:
                raise JobRejected(
                    f"too many jobs in flight ({_MAX_INFLIGHT}); retry shortly")
            if sum(1 for j in live if j["session_id"] == session_id) >= _MAX_PER_SESSION:
                raise JobRejected(
                    f"session already has {_MAX_PER_SESSION} jobs running; "
                    "wait for one to finish")
            self._seq += 1
            self._jobs[job_id] = {
                "job_id": job_id, "session_id": session_id, "kind": kind,
                "status": "running", "result": None, "error": None,
                "created_at": self._clock(), "finished_at": None,
                "_seq": self._seq,
            }
            self._prune_locked()

        def runner() -> None:
            try:
                res = fn()
                self._finish(job_id, status="done", result=res)
            except Exception as exc:  # surface as a failed job, never crash the pool
                self._finish(job_id, status="error", error=str(exc))

        self._pool.submit(runner)
        return job_id

    def _finish(self, job_id: str, *, status: str, result: Any = None,
                error: str | None = None) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.update(status=status, result=result, error=error,
                           finished_at=self._clock())

    def get(self, job_id: str) -> dict[str, Any] | None:
        """Return a copy of the job record (without internal fields), or None."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            return {k: v for k, v in job.items() if not k.startswith("_")}

    def _prune_locked(self) -> None:
        """Drop oldest *finished* jobs once the registry exceeds the cap."""
        if len(self._jobs) <= _MAX_JOBS:
            return
        finished = sorted(
            (j for j in self._jobs.values() if j["status"] != "running"),
            key=lambda j: j["_seq"])
        for job in finished[: len(self._jobs) - _MAX_JOBS]:
            self._jobs.pop(job["job_id"], None)
