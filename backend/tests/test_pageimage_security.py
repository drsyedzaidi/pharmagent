"""Adversarial tests for the page-image viewer: hostile uploads, path
traversal, resource abuse, cross-user access, and information leakage."""
from __future__ import annotations

import itertools
import textwrap

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.config import settings
from app.core.llm import MockLLM
from app.core.orchestrator import Orchestrator
from app.core.store import SessionStore
from app.pageimage import limits, runner
from tests.pdf_fixtures import make_pdf, make_zero_page_pdf


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    main.orch = Orchestrator(llm=MockLLM(), clock=lambda c=itertools.count(): f"t{next(c)}",
                             store=SessionStore(":memory:"))
    token = settings.api_token
    yield TestClient(main.app)
    settings.api_token = token


def _sid(c):
    return c.post("/api/sessions").json()["id"]


def _up(c, sid, data, name="x.pdf", ctype="application/pdf", headers=None):
    return c.post(f"/api/sessions/{sid}/documents", files={"file": (name, data, ctype)}, headers=headers or {})


def _no_leak(resp, tmp_path):
    t = resp.text
    for needle in (str(tmp_path), "Traceback", "pypdfium", ".pdf\"", "site-packages"):
        assert needle not in t, needle


# ── hostile uploads ──────────────────────────────────────────────────────────
@pytest.mark.parametrize("data", [
    b"MZ\x90\x00 this is an exe",
    b"<html><script>alert(1)</script></html>",
    b"id,time,conc\n1,0,0\n",
    b"",
    b"%PD",                        # truncated magic
    b"\n%PDF-1.4\n",              # junk before the header
])
def test_non_pdf_with_pdf_suffix_and_pdf_content_type_rejected(client, tmp_path, data):
    sid = _sid(client)
    r = _up(client, sid, data, name="report.pdf", ctype="application/pdf")
    assert r.status_code == 400
    _no_leak(r, tmp_path)
    assert client.get(f"/api/sessions/{sid}/documents").json() == {"documents": []}
    # nothing is left on disk either
    root = tmp_path / "data" / "documents" / sid
    assert not root.exists() or list(root.iterdir()) == []


def test_pdf_magic_is_authoritative_not_suffix_or_content_type(client):
    sid = _sid(client)
    r = _up(client, sid, make_pdf(), name="innocent.txt", ctype="text/plain")
    assert r.status_code == 201  # real PDF bytes are accepted regardless of the claimed name/type


def test_valid_magic_but_garbage_body_rejected_and_cleaned(client, tmp_path):
    sid = _sid(client)
    r = _up(client, sid, b"%PDF-1.7\n" + b"\x00\xff" * 4000)
    assert r.status_code == 400
    _no_leak(r, tmp_path)
    root = tmp_path / "data" / "documents" / sid
    assert list(root.iterdir()) == []


def test_truncated_pdf_rejected(client):
    sid = _sid(client)
    assert _up(client, sid, make_pdf()[:150]).status_code == 400


def test_encrypted_pdf_rejected(client, tmp_path):
    sid = _sid(client)
    r = _up(client, sid, make_pdf(encrypt_user_password="secret"))
    assert r.status_code == 400
    assert "password" in r.json()["error"]["message"]
    _no_leak(r, tmp_path)


def test_zero_page_pdf_rejected(client, tmp_path):
    sid = _sid(client)
    r = _up(client, sid, make_zero_page_pdf())
    assert r.status_code == 400
    _no_leak(r, tmp_path)


def test_oversized_upload_rejected_with_413_and_cleaned(client, tmp_path, monkeypatch):
    monkeypatch.setattr(limits, "MAX_PDF_BYTES", 2048)
    sid = _sid(client)
    r = _up(client, sid, b"%PDF-1.4\n" + b"A" * 5000)
    assert r.status_code == 413
    root = tmp_path / "data" / "documents" / sid
    assert list(root.iterdir()) == []


def test_page_count_cap(client, monkeypatch):
    monkeypatch.setattr(limits, "MAX_PAGES", 3)
    sid = _sid(client)
    assert _up(client, sid, make_pdf(pages=3)).status_code == 201
    r = _up(client, sid, make_pdf(pages=4))
    assert r.status_code == 400 and "pages" in r.json()["error"]["message"]


def test_documents_per_session_cap(client, monkeypatch):
    monkeypatch.setattr(limits, "MAX_DOCS_PER_SESSION", 2)
    sid = _sid(client)
    assert _up(client, sid, make_pdf()).status_code == 201
    assert _up(client, sid, make_pdf()).status_code == 201
    assert _up(client, sid, make_pdf()).status_code == 409


# ── path traversal ───────────────────────────────────────────────────────────
def _assert_not_served(resp):
    """The client (httpx) may normalise some hostile paths onto unrelated routes
    (and a built frontend/dist adds an SPA fallback that answers 200 text/html).
    Whatever answers, it must never be document metadata or an image."""
    ctype = resp.headers.get("content-type", "")
    assert not ctype.startswith("image/")
    assert resp.status_code in (404, 405, 422) or (resp.status_code == 200 and ctype.startswith("text/html"))
    assert "size_bytes" not in resp.text and "\x89PNG" not in resp.text


@pytest.mark.parametrize("name", [
    "../../../../etc/passwd.pdf", "..\\..\\windows\\system32\\x.pdf", "/abs/path.pdf",
    "a/b/c.pdf", "x\x00.pdf", "%2e%2e%2fx.pdf", "....//x.pdf",
])
def test_hostile_client_filenames_never_reach_the_filesystem(client, tmp_path, name):
    sid = _sid(client)
    r = _up(client, sid, make_pdf(), name=name)
    assert r.status_code == 201
    meta = r.json()
    assert "/" not in meta["filename"] and "\\" not in meta["filename"] and "\x00" not in meta["filename"]
    created = {p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file()}
    assert created == {f"data/documents/{sid}/{meta['id']}.pdf", f"data/documents/{sid}/{meta['id']}.json"}


@pytest.mark.parametrize("doc_id", [
    "..", "../..", "%2e%2e", "..%2f..%2fetc%2fpasswd", "....", "a" * 31, "a" * 33, "A" * 32,
    "g" * 32, "0" * 32, "x/../" + "a" * 27, "%00",
])
def test_hostile_document_ids_are_404(client, doc_id):
    sid = _sid(client)
    _assert_not_served(client.get(f"/api/sessions/{sid}/documents/{doc_id}"))
    _assert_not_served(client.get(f"/api/sessions/{sid}/documents/{doc_id}/pages/0.png"))


def test_client_cannot_name_a_file_via_page_segment(client, tmp_path):
    sid = _sid(client)
    doc = _up(client, sid, make_pdf()).json()
    for seg in ("../0.png", "..%2f0.png", "0.png%00"):
        _assert_not_served(client.get(f"/api/sessions/{sid}/documents/{doc['id']}/pages/{seg}"))


def test_session_dir_confinement_for_odd_session_ids():
    from app.pageimage import documents
    for bad in ("..", "../x", "a/b", "", "x" * 65, ".", "sess 1"):
        assert documents.list_documents(bad) == []
        assert documents.get(bad, "a" * 32) is None
        with pytest.raises(ValueError):
            documents.new_slot(bad)


# ── resource abuse on render ─────────────────────────────────────────────────
def test_oversized_dimensions_are_400_not_a_crash(client, tmp_path):
    sid = _sid(client)
    doc = _up(client, sid, make_pdf(width=14400, height=14400)).json()
    r = client.get(f"/api/sessions/{sid}/documents/{doc['id']}/pages/0.png", params={"scale": 3})
    assert r.status_code == 400
    _no_leak(r, tmp_path)
    # a cropped corner of the same page renders fine
    ok = client.get(f"/api/sessions/{sid}/documents/{doc['id']}/pages/0.png",
                    params={"scale": 1, "x0": 0, "y0": 0, "x1": 0.05, "y1": 0.05})
    assert ok.status_code == 200


def test_render_timeout_kills_worker_and_returns_504(client, tmp_path, monkeypatch):
    pidfile = tmp_path / "pid.txt"
    sid = _sid(client)
    doc = _up(client, sid, make_pdf()).json()  # upload with the REAL worker first
    fake = tmp_path / "hang.py"
    fake.write_text(textwrap.dedent(f"""
        import os, time
        open({str(pidfile)!r}, "w").write(str(os.getpid()))
        time.sleep(120)
    """))
    monkeypatch.setattr(runner, "WORKER_PATH", fake)
    monkeypatch.setattr(limits, "WALL_SECONDS", 1.0)
    r = client.get(f"/api/sessions/{sid}/documents/{doc['id']}/pages/0.png")
    assert r.status_code == 504
    _no_leak(r, tmp_path)
    import os
    import time
    time.sleep(0.2)
    with pytest.raises(ProcessLookupError):
        os.kill(int(pidfile.read_text()), 0)


def test_crashing_worker_is_422_without_details(client, tmp_path, monkeypatch):
    sid = _sid(client)
    doc = _up(client, sid, make_pdf()).json()
    fake = tmp_path / "crash.py"
    fake.write_text("import sys; sys.stderr.write('/secret/path/boom'); import os, signal; os.kill(os.getpid(), signal.SIGSEGV)")
    monkeypatch.setattr(runner, "WORKER_PATH", fake)
    r = client.get(f"/api/sessions/{sid}/documents/{doc['id']}/pages/0.png")
    assert r.status_code == 422
    assert "secret" not in r.text
    _no_leak(r, tmp_path)


def test_worker_emitting_html_is_never_served_as_image(client, tmp_path, monkeypatch):
    sid = _sid(client)
    doc = _up(client, sid, make_pdf()).json()
    fake = tmp_path / "html.py"
    fake.write_text("import sys; sys.stdout.write('<script>alert(1)</script>')")
    monkeypatch.setattr(runner, "WORKER_PATH", fake)
    r = client.get(f"/api/sessions/{sid}/documents/{doc['id']}/pages/0.png")
    assert r.status_code == 422
    assert r.headers["content-type"].startswith("application/json")


def test_oversized_png_output_is_413(client, tmp_path, monkeypatch):
    sid = _sid(client)
    doc = _up(client, sid, make_pdf()).json()
    monkeypatch.setattr(limits, "MAX_PNG_BYTES", 64)  # smaller than any real PNG
    r = client.get(f"/api/sessions/{sid}/documents/{doc['id']}/pages/0.png")
    assert r.status_code == 413


# ── cross-user access ────────────────────────────────────────────────────────
def test_other_users_document_is_unreachable(client):
    settings.api_token = "tok-a"
    h = {"Authorization": "Bearer tok-a"}
    mine = client.post("/api/sessions", headers=h).json()["id"]
    theirs = main.orch.create_session(owner="token:someone-else")
    # plant a document in the other owner's session directly
    from app.pageimage import documents
    doc_id, part = documents.new_slot(theirs.id)
    part.write_bytes(make_pdf())
    documents.commit(theirs.id, doc_id, part, filename="secret.pdf", size=part.stat().st_size, pages=1)
    # by their session id: owner check refuses
    assert client.get(f"/api/sessions/{theirs.id}/documents", headers=h).status_code == 403
    assert client.get(f"/api/sessions/{theirs.id}/documents/{doc_id}", headers=h).status_code == 403
    assert client.get(f"/api/sessions/{theirs.id}/documents/{doc_id}/pages/0.png", headers=h).status_code == 403
    # by my session id with their doc id: not found (never resolved across sessions)
    assert client.get(f"/api/sessions/{mine}/documents/{doc_id}", headers=h).status_code == 404
    assert client.get(f"/api/sessions/{mine}/documents/{doc_id}/pages/0.png", headers=h).status_code == 404


def test_error_bodies_use_the_standard_envelope_without_paths(client, tmp_path):
    sid = _sid(client)
    r = client.get(f"/api/sessions/{sid}/documents/{'c' * 32}/pages/0.png")
    assert r.status_code == 404
    body = r.json()
    assert body["success"] is False and set(body["error"]) == {"code", "message", "request_id"}
    _no_leak(r, tmp_path)
