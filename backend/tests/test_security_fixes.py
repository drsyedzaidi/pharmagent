"""Regression tests for the confirmed Codex security findings (2026-07-13).

Each test pins one fix so the vulnerability cannot silently return.
"""
from __future__ import annotations

import threading

import pytest
from fastapi import HTTPException

from app.compute.compartmental import _accum
from app.config import settings
from app.core.jobs import _MAX_PER_SESSION, JobManager, JobRejected
from app.main import current_owner


def test_current_owner_never_returns_the_raw_token():
    """P1-owner: the persisted owner / audit actor must be a non-secret principal,
    never the bearer token itself (a DB or audit export would else leak the token)."""
    orig = settings.api_token
    try:
        settings.api_token = "super-secret-token"
        owner = current_owner(authorization="Bearer super-secret-token")
        assert owner is not None
        assert owner != "super-secret-token"
        assert "super-secret-token" not in owner       # not embedded anywhere
        assert owner.startswith("token:")
        # deterministic — same token yields the same principal, so ownership works
        assert current_owner(authorization="Bearer super-secret-token") == owner
        with pytest.raises(HTTPException):             # wrong token still 401
            current_owner(authorization="Bearer wrong")
    finally:
        settings.api_token = orig


def test_job_admission_caps_per_session():
    """P1-jobs: a session cannot flood the executor — the per-session cap rejects
    the (cap+1)-th live job with JobRejected (mapped to HTTP 429)."""
    jm = JobManager(max_workers=8)
    gate = threading.Event()

    def blocker() -> None:
        gate.wait(timeout=5)     # stay "running" until released

    try:
        for _ in range(_MAX_PER_SESSION):
            jm.submit(session_id="s1", kind="test", fn=blocker)
        with pytest.raises(JobRejected):
            jm.submit(session_id="s1", kind="test", fn=blocker)
        # a different session is unaffected by s1's cap
        jm.submit(session_id="s2", kind="test", fn=blocker)
    finally:
        gate.set()               # release the blockers so the pool drains


def test_accum_rejects_nonpositive_tau():
    """P2-tau: a non-positive steady-state dosing interval is a pole, not a
    removable singularity — reject it instead of clamping to a huge factor."""
    with pytest.raises(ValueError):
        _accum(0.1, 0.0)
    with pytest.raises(ValueError):
        _accum(0.1, -12.0)
    assert _accum(0.1, 12.0) > 1.0     # a valid interval accumulates (> 1)
