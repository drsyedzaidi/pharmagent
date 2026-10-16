"""The page-image routes are OFF by default (no UI reaches them yet) and never exist in a frozen desktop build."""
from __future__ import annotations

import itertools
import sys

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.config import settings
from app.core.llm import MockLLM
from app.core.orchestrator import Orchestrator
from app.core.store import SessionStore
from tests.pdf_fixtures import make_pdf


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    main.orch = Orchestrator(llm=MockLLM(), clock=lambda c=itertools.count(): f"t{next(c)}",
                             store=SessionStore(":memory:"))
    token = settings.api_token
    yield TestClient(main.app)
    settings.api_token = token


def _requests(client, sid):
    pdf = ("a.pdf", make_pdf(), "application/pdf")
    return [
        client.post(f"/api/sessions/{sid}/documents", files={"file": pdf}),
        client.get(f"/api/sessions/{sid}/documents"),
        client.get(f"/api/sessions/{sid}/documents/{'a' * 32}"),
        client.get(f"/api/sessions/{sid}/documents/{'a' * 32}/pages/0.png"),
    ]


def test_the_default_is_disabled():
    assert settings.pageimage_enabled is False
    assert settings.pageimage_allow_weak_limits is False


def test_every_route_is_404_when_disabled_and_nothing_is_stored(client, tmp_path):
    sid = client.post("/api/sessions").json()["id"]
    assert [r.status_code for r in _requests(client, sid)] == [404, 404, 404, 404]
    assert not (tmp_path / "data" / "documents").exists()


def test_routes_exist_once_enabled(client, monkeypatch):
    monkeypatch.setattr(settings, "pageimage_enabled", True)
    sid = client.post("/api/sessions").json()["id"]
    assert client.get(f"/api/sessions/{sid}/documents").status_code == 200


def test_a_frozen_build_stays_disabled_even_when_enabled(client, monkeypatch):
    monkeypatch.setattr(settings, "pageimage_enabled", True)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    sid = client.post("/api/sessions").json()["id"]
    assert [r.status_code for r in _requests(client, sid)] == [404, 404, 404, 404]
