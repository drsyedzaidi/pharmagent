"""The page-image routes are OFF by default (no UI reaches them yet) and never exist in a frozen desktop build.

Disabled means ABSENT: not matched by the router, so FastAPI never parses the multipart body (no form parsing, no
disk spooling) and the OpenAPI schema does not list them."""
from __future__ import annotations

import itertools
import sys

import pytest
from fastapi.testclient import TestClient
from starlette import formparsers, requests

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
    main.app.openapi_schema = None
    token = settings.api_token
    yield TestClient(main.app)
    settings.api_token = token
    main.app.openapi_schema = None


def _requests(client, sid):
    pdf = ("a.pdf", make_pdf(), "application/pdf")
    return [
        client.post(f"/api/sessions/{sid}/documents", files={"file": pdf}),
        client.get(f"/api/sessions/{sid}/documents"),
        client.get(f"/api/sessions/{sid}/documents/{'a' * 32}"),
        client.get(f"/api/sessions/{sid}/documents/{'a' * 32}/pages/0.png"),
    ]


def _absent(r) -> bool:
    """Not the feature's own answer: 404/405, or the SPA shell; never JSON metadata or an image."""
    return r.status_code in (404, 405) or r.headers.get("content-type", "").startswith("text/html")


def _documents_paths(client) -> list[str]:
    return [p for p in client.get("/openapi.json").json()["paths"] if "/documents" in p]


def test_the_defaults_are_disabled_and_safe():
    assert settings.pageimage_enabled is False
    assert settings.pageimage_unsafe_allow_unenforced_limits is False


def test_every_route_is_absent_when_disabled_and_nothing_is_stored(client, tmp_path):
    sid = client.post("/api/sessions").json()["id"]
    assert all(_absent(r) for r in _requests(client, sid))
    assert not (tmp_path / "data" / "documents").exists()


def test_disabled_routes_are_not_in_the_openapi_schema(client):
    assert _documents_paths(client) == []


@pytest.mark.parametrize("body, ctype", [
    (b"not multipart at all", "multipart/form-data; boundary=x"),
    (b"--x\r\nContent-Disposition: form-data; name=file; filename=a.pdf\r\n\r\n" + b"A" * 3_000_000,
     "multipart/form-data; boundary=x"),
    (b"", "multipart/form-data"),
])
def test_disabled_upload_never_reaches_form_parsing_or_the_disk(client, tmp_path, monkeypatch, body, ctype):
    def boom(*a, **k):
        raise AssertionError("the body was parsed although the feature is disabled")

    monkeypatch.setattr(requests.Request, "_get_form", boom)
    monkeypatch.setattr(formparsers.MultiPartParser, "parse", boom)
    sid = client.post("/api/sessions").json()["id"]
    r = client.post(f"/api/sessions/{sid}/documents", content=body, headers={"content-type": ctype})
    assert _absent(r)
    assert not (tmp_path / "data" / "documents").exists()


def test_routes_exist_once_enabled(client, monkeypatch):
    monkeypatch.setattr(settings, "pageimage_enabled", True)
    sid = client.post("/api/sessions").json()["id"]
    assert client.get(f"/api/sessions/{sid}/documents").status_code == 200
    assert any(p.endswith("/documents") for p in _documents_paths(client))


def test_a_frozen_build_stays_absent_even_when_enabled(client, monkeypatch):
    monkeypatch.setattr(settings, "pageimage_enabled", True)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    sid = client.post("/api/sessions").json()["id"]
    assert all(_absent(r) for r in _requests(client, sid))
    assert _documents_paths(client) == []
