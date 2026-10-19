"""Render isolation through the HTTP API: a session whose render slot is busy is told to retry (429) while another
session's page still renders, and a session can never take both global slots."""
from __future__ import annotations

import itertools
import threading

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.config import settings
from app.core.llm import MockLLM
from app.core.orchestrator import Orchestrator
from app.core.store import SessionStore
from app.pageimage import limits, slots
from tests.pdf_fixtures import make_pdf, unenforced_limits_off_linux

_weak = pytest.fixture(autouse=True)(unenforced_limits_off_linux)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "pageimage_enabled", True)
    monkeypatch.setattr(limits, "QUEUE_WAIT_SECONDS", 0.2)
    main.orch = Orchestrator(llm=MockLLM(), clock=lambda c=itertools.count(): f"t{next(c)}",
                             store=SessionStore(":memory:"))
    token = settings.api_token
    yield TestClient(main.app)
    settings.api_token = token


_RELEASES: list[threading.Event] = []


@pytest.fixture(autouse=True)
def _release_every_holder():
    _RELEASES.clear()
    yield
    for e in _RELEASES:
        e.set()


def _doc(client):
    sid = client.post("/api/sessions").json()["id"]
    r = client.post(f"/api/sessions/{sid}/documents", files={"file": ("a.pdf", make_pdf(), "application/pdf")})
    assert r.status_code == 201
    return sid, r.json()["id"]


def _hold(session):
    started, release = threading.Event(), threading.Event()
    _RELEASES.append(release)

    def run():
        with slots.render_slot(session, wait=5):
            started.set()
            release.wait(30)
    t = threading.Thread(daemon=True, target=run)
    t.start()
    assert started.wait(10)
    return t, release


def test_a_busy_session_gets_429_while_another_session_still_renders(client):
    (sa, da), (sb, db) = _doc(client), _doc(client)
    t, release = _hold(sa)                                       # session A's one render slot is in use
    try:
        busy = client.get(f"/api/sessions/{sa}/documents/{da}/pages/0.png")
        assert busy.status_code == 429
        ok = client.get(f"/api/sessions/{sb}/documents/{db}/pages/0.png")
        assert ok.status_code == 200 and ok.content.startswith(b"\x89PNG")
    finally:
        release.set()
        t.join()
    assert client.get(f"/api/sessions/{sa}/documents/{da}/pages/0.png").status_code == 200   # and A recovers


def test_one_session_cannot_take_both_global_slots(client):
    (sa, da), (sb, db) = _doc(client), _doc(client)
    t, release = _hold(sa)
    try:
        # A is running; a second A render is refused although a global slot is free, and B uses that free slot
        assert client.get(f"/api/sessions/{sa}/documents/{da}/pages/0.png").status_code == 429
        assert client.get(f"/api/sessions/{sb}/documents/{db}/pages/0.png").status_code == 200
    finally:
        release.set()
        t.join()
