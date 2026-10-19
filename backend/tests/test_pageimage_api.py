"""HTTP layer of the page-image viewer: PDF upload / document store / PNG render,
plus a regression check that the CSV upload is unchanged."""
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
from tests.pdf_fixtures import REPORT_OK, decode_png, make_pdf, unenforced_limits_off_linux

SAMPLE = Path(__file__).parent.parent / "sample_data" / "oral_pk.csv"


_weak = pytest.fixture(autouse=True)(unenforced_limits_off_linux)   # macOS: the explicit UNSAFE operator override

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(settings, "pageimage_enabled", True)
    main.orch = Orchestrator(llm=MockLLM(), clock=lambda c=itertools.count(): f"t{next(c)}",
                             store=SessionStore(":memory:"))
    token = settings.api_token
    yield TestClient(main.app)
    settings.api_token = token


def _sid(client) -> str:
    return client.post("/api/sessions").json()["id"]


def _upload(client, sid, data=None, name="paper.pdf", ctype="application/pdf", headers=None):
    return client.post(f"/api/sessions/{sid}/documents",
                       files={"file": (name, data if data is not None else make_pdf(pages=2), ctype)},
                       headers=headers or {})


# ── CSV regression ───────────────────────────────────────────────────────────
def test_csv_upload_still_works(client):
    sid = _sid(client)
    r = client.post(f"/api/sessions/{sid}/upload",
                    files={"file": ("oral_pk.csv", SAMPLE.read_bytes(), "text/csv")})
    assert r.status_code == 200
    body = r.json()
    assert body["dataset_id"].startswith("oral_pk_")
    assert body["metadata"]["dataset_path"].endswith(".csv")


def test_old_upload_route_still_rejects_pdf_and_txt(client):
    sid = _sid(client)
    for name, data in (("a.pdf", make_pdf()), ("x.txt", b"hello")):
        r = client.post(f"/api/sessions/{sid}/upload", files={"file": (name, data, "application/octet-stream")})
        assert r.status_code == 400


# ── document store ───────────────────────────────────────────────────────────
def test_upload_list_get_roundtrip(client, tmp_path):
    sid = _sid(client)
    r = _upload(client, sid)
    assert r.status_code == 201
    meta = r.json()
    assert set(meta) == {"id", "filename", "size_bytes", "pages", "created_at"}
    assert len(meta["id"]) == 32 and meta["pages"] == 2 and meta["filename"] == "paper.pdf"
    assert str(tmp_path) not in r.text  # no server paths in the response
    listed = client.get(f"/api/sessions/{sid}/documents").json()["documents"]
    assert [d["id"] for d in listed] == [meta["id"]]
    assert client.get(f"/api/sessions/{sid}/documents/{meta['id']}").json() == meta


def test_stored_under_server_generated_name_in_session_dir(client, tmp_path):
    sid = _sid(client)
    meta = _upload(client, sid, name="../../evil name.pdf").json()
    root = tmp_path / "data" / "documents" / sid
    assert (root / f"{meta['id']}.pdf").is_file()
    import hashlib
    assert sorted(p.name for p in root.iterdir()) == sorted([
        f"{meta['id']}.json", f"{meta['id']}.pdf",
        f".render-{hashlib.sha256(sid.encode()).hexdigest()}-0"])          # document, metadata, this session's lock
    assert meta["filename"] == "evil name.pdf"  # display-only, basename'd


def test_list_empty_and_unknown_document(client):
    sid = _sid(client)
    assert client.get(f"/api/sessions/{sid}/documents").json() == {"documents": []}
    assert client.get(f"/api/sessions/{sid}/documents/{'a' * 32}").status_code == 404


def test_unknown_session_404(client):
    assert client.get("/api/sessions/sess_nope/documents").status_code == 404


# ── render endpoint ──────────────────────────────────────────────────────────
def _page_url(sid, doc_id, page=0):
    return f"/api/sessions/{sid}/documents/{doc_id}/pages/{page}.png"


def test_render_returns_png_with_safe_headers(client):
    sid = _sid(client)
    doc = _upload(client, sid, make_pdf(width=200, height=100)).json()
    r = client.get(_page_url(sid, doc["id"]), params={"scale": 1.0})
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.headers["x-content-type-options"] == "nosniff"
    assert "no-store" in r.headers["cache-control"]
    img = decode_png(r.content)
    assert (img.width, img.height) == (200, 100)


def test_render_default_scale_and_second_page(client):
    sid = _sid(client)
    doc = _upload(client, sid, make_pdf(pages=2, width=200, height=100)).json()
    r = client.get(_page_url(sid, doc["id"], 1))
    assert r.status_code == 200
    img = decode_png(r.content)
    assert (img.width, img.height) == (300, 150)  # default scale 1.5


def test_render_bbox_crop(client):
    sid = _sid(client)
    doc = _upload(client, sid, make_pdf(width=200, height=100, rect=(0, 0, 50, 50))).json()
    r = client.get(_page_url(sid, doc["id"]),
                   params={"scale": 1.0, "x0": 0, "y0": 0.5, "x1": 0.25, "y1": 1})
    assert r.status_code == 200
    img = decode_png(r.content)
    assert abs(img.width - 50) <= 1 and abs(img.height - 50) <= 1
    assert img.pixel(img.width // 2, img.height // 2) == (0, 0, 0)


@pytest.mark.parametrize("params,status", [
    ({"scale": 0.1}, 422),
    ({"scale": 9}, 422),
    ({"scale": "nan"}, 422),
    ({"scale": "abc"}, 422),
    ({"x0": 0.1, "y0": 0.1, "x1": 0.5}, 400),                          # partial bbox
    ({"x0": 0.6, "y0": 0.1, "x1": 0.5, "y1": 0.9}, 400),               # reversed
    ({"x0": -1, "y0": 0.1, "x1": 0.5, "y1": 0.9}, 400),                # out of range
    ({"x0": "nan", "y0": 0.1, "x1": 0.5, "y1": 0.9}, 400),
])
def test_render_param_validation(client, params, status):
    sid = _sid(client)
    doc = _upload(client, sid).json()
    r = client.get(_page_url(sid, doc["id"]), params=params)
    assert r.status_code == status


@pytest.mark.parametrize("page_file", ["2.png", "99999.png", "-1.png", "1.jpg", "x.png", "1", "01.png.exe", ".png"])
def test_render_bad_page_is_404(client, page_file):
    sid = _sid(client)
    doc = _upload(client, sid, make_pdf(pages=2)).json()
    r = client.get(f"/api/sessions/{sid}/documents/{doc['id']}/pages/{page_file}")
    assert r.status_code == 404


def test_render_unknown_document_404(client):
    sid = _sid(client)
    assert client.get(_page_url(sid, "b" * 32)).status_code == 404


# ── ownership ────────────────────────────────────────────────────────────────
def test_auth_required_and_cross_owner_blocked(client):
    settings.api_token = "tok-a"
    h = {"Authorization": "Bearer tok-a"}
    sid = client.post("/api/sessions", headers=h).json()["id"]
    doc = _upload(client, sid, headers=h).json()
    # no token
    assert client.get(f"/api/sessions/{sid}/documents").status_code == 401
    assert client.get(_page_url(sid, doc["id"])).status_code == 401
    assert _upload(client, sid).status_code == 401
    # a session owned by someone else is 403 for every documents route
    other = main.orch.create_session(owner="token:someone-else")
    assert client.get(f"/api/sessions/{other.id}/documents", headers=h).status_code == 403
    assert client.get(_page_url(other.id, doc["id"]), headers=h).status_code == 403
    assert _upload(client, other.id, headers=h).status_code == 403
    # my own session cannot reach a document that lives in another session
    mine2 = client.post("/api/sessions", headers=h).json()["id"]
    assert client.get(_page_url(mine2, doc["id"]), headers=h).status_code == 404
    assert client.get(f"/api/sessions/{mine2}/documents/{doc['id']}", headers=h).status_code == 404
    # legit access still works
    assert client.get(_page_url(sid, doc["id"]), headers=h).status_code == 200


# ── resource-limit outcomes reach the client as the right status, with no detail ─────────────────────────────
def _fake_worker_file(tmp_path, monkeypatch, source: str):
    from app.pageimage import runner
    w = tmp_path / "fake_api_worker.py"
    w.write_text(source)
    monkeypatch.setattr(runner, "WORKER_PATH", w)


@pytest.mark.parametrize("source, status, text", [
    ("import sys; sys.exit(9)", 503, "unavailable"),                                          # limit_unavailable
    (REPORT_OK + "import sys; sys.exit(10)", 413, "resource limits"),                         # memory_limit
    (REPORT_OK + "import os, signal; os.kill(os.getpid(), signal.SIGXCPU)", 504, "timed out"),   # cpu_limit
    (REPORT_OK + "import time; time.sleep(60)", 504, "timed out"),                            # wall-clock timeout
])
def test_render_limit_outcomes_map_to_distinct_statuses(client, tmp_path, monkeypatch, source, status, text):
    from app.pageimage import limits as pi_limits
    sid = _sid(client)
    doc = _upload(client, sid).json()                                                         # uploaded with the real worker
    monkeypatch.setattr(pi_limits, "WALL_SECONDS", 3.0)
    _fake_worker_file(tmp_path, monkeypatch, source)
    r = client.get(f"/api/sessions/{sid}/documents/{doc['id']}/pages/0.png")
    assert r.status_code == status
    assert text in r.text                       # the fixed public message, whatever the error envelope
    assert str(tmp_path) not in r.text and "Traceback" not in r.text


@pytest.mark.parametrize("source, status", [
    ("import sys; sys.exit(9)", 503),                                                         # limit_unavailable
    (REPORT_OK + "import sys; sys.exit(10)", 400),                                            # memory_limit
])
def test_upload_limit_outcomes_map_to_distinct_statuses(client, tmp_path, monkeypatch, source, status):
    sid = _sid(client)
    _fake_worker_file(tmp_path, monkeypatch, source)
    r = _upload(client, sid)
    assert r.status_code == status
    assert client.get(f"/api/sessions/{sid}/documents").json() == {"documents": []}          # nothing was stored
