"""Re-review MEDIUM 1 and LOW 4 of the upload path: no stored file survives any failure after it is created; an
unstorable filename is a client error; Starlette's own form errors use the app's error envelope."""
from __future__ import annotations

import itertools
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.config import settings
from app.core.llm import MockLLM
from app.core.orchestrator import Orchestrator
from app.core.store import SessionStore

SAMPLE = Path(__file__).parent.parent / "sample_data" / "oral_pk.csv"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    main.orch = Orchestrator(llm=MockLLM(), clock=lambda c=itertools.count(): f"t{next(c)}",
                             store=SessionStore(":memory:"))
    return TestClient(main.app, raise_server_exceptions=False)


def _csvs(d: Path) -> list[Path]:
    return [p for p in d.iterdir() if p.suffix == ".csv"]


def test_a_failure_after_the_csv_is_stored_removes_it(client, tmp_path, monkeypatch):
    import app.core.schema_extractor as se

    def boom(*a, **k):
        raise RuntimeError("schema failed")

    monkeypatch.setattr(se, "extract_schema", boom)
    sid = client.post("/api/sessions").json()["id"]
    r = client.post(f"/api/sessions/{sid}/upload", files={"file": ("a.csv", SAMPLE.read_bytes(), "text/csv")})
    assert r.status_code >= 400 and _csvs(tmp_path) == []


def test_a_write_failure_midway_leaves_no_partial_file(client, tmp_path, monkeypatch):
    from starlette.datastructures import UploadFile

    calls = {"n": 0}
    original = UploadFile.read

    async def flaky(self, size=-1):
        calls["n"] += 1
        if calls["n"] > 1:
            raise OSError("disk full")
        return await original(self, size)

    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 10 * 1024 * 1024)
    monkeypatch.setattr(UploadFile, "read", flaky)
    sid = client.post("/api/sessions").json()["id"]
    r = client.post(f"/api/sessions/{sid}/upload", files={"file": ("a.csv", SAMPLE.read_bytes(), "text/csv")})
    assert r.status_code >= 400 and _csvs(tmp_path) == []


def test_an_unstorable_filename_is_a_client_error(client, tmp_path):
    sid = client.post("/api/sessions").json()["id"]
    r = client.post(f"/api/sessions/{sid}/upload", files={"file": ("x" * 300 + ".csv", b"a,b\n1,2\n", "text/csv")})
    assert r.status_code == 400 and _csvs(tmp_path) == []


@pytest.mark.parametrize("headers,content", [
    ({"Content-Type": "multipart/form-data"}, b"no boundary"),                 # missing boundary
])
def test_starlette_form_errors_use_the_app_error_envelope(client, headers, content):
    sid = client.post("/api/sessions").json()["id"]
    r = client.post(f"/api/sessions/{sid}/upload", content=content, headers=headers)
    assert r.status_code == 400 and r.json().get("success") is False and "error" in r.json()


def test_too_many_fields_use_the_app_error_envelope(client):
    sid = client.post("/api/sessions").json()["id"]
    r = client.post(f"/api/sessions/{sid}/upload", data={f"f{i}": "x" for i in range(10)},
                    files={"file": ("a.csv", b"a\n1\n", "text/csv")})
    assert r.status_code == 400 and r.json().get("success") is False
