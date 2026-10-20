"""Malformed multipart bodies on the (enabled) upload route.

PR #2 parses the form by hand, after authentication and the quota reservation, so FastAPI's own body-parse wrapper no
longer turns parser errors into 400. Every malformed body must still get a stable 400 in the standard error envelope,
log nothing at ERROR level, release its reservation, leave no upload pieces behind, and close every spooled temp file."""
from __future__ import annotations

import asyncio
import itertools
import logging

import pytest
from fastapi.testclient import TestClient
from starlette import formparsers

import app.main as main
from app.config import settings
from app.core.llm import MockLLM
from app.core.orchestrator import Orchestrator
from app.core.store import SessionStore
from app.pageimage import limits, quota
from tests.pdf_fixtures import make_pdf, unenforced_limits_off_linux

_weak = pytest.fixture(autouse=True)(unenforced_limits_off_linux)

MALFORMED = "upload must be a well-formed multipart/form-data body with one 'file' part"
B = "multipart/form-data; boundary=x"
FILE_HEAD = (b'--x\r\nContent-Disposition: form-data; name="file"; filename="a.pdf"\r\n'
             b"Content-Type: application/pdf\r\n\r\n")


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "pageimage_enabled", True)
    main.orch = Orchestrator(llm=MockLLM(), clock=lambda c=itertools.count(): f"t{next(c)}",
                             store=SessionStore(":memory:"))
    token = settings.api_token
    yield TestClient(main.app, raise_server_exceptions=False)
    settings.api_token = token


@pytest.fixture
def spools(monkeypatch):
    """Every temp file the multipart parser spools to, so a test can check each one was closed."""
    made = []
    real = formparsers.SpooledTemporaryFile

    def tracking(*a, **k):
        f = real(*a, **k)
        made.append(f)
        return f
    monkeypatch.setattr(formparsers, "SpooledTemporaryFile", tracking)
    return made


def _root():
    return settings.data_dir / "documents"


def _pieces(sid):
    d = _root() / sid
    return sorted(p.name for p in d.iterdir() if not p.name.startswith(".render-")) if d.exists() else []


CASES = {
    "garbage instead of a boundary": (b"A" * 1000, B),
    "no boundary parameter": (b"--x\r\n", "multipart/form-data"),
    "a file part, then an invalid part header": (FILE_HEAD + b"%PDF-1.7\r\n--x\r\nBad Header: v\r\n\r\nzz\r\n--x--\r\n", B),
    "two file parts": (FILE_HEAD + b"%PDF-1.7\r\n" + FILE_HEAD + b"%PDF-1.7\r\n--x--\r\n", B),
    "an unknown charset": (FILE_HEAD + b"%PDF-1.7\r\n--x--\r\n", B + "; charset=undefined"),
    "more text fields than allowed": (b"".join(b'--x\r\nContent-Disposition: form-data; name="f%d"\r\n\r\nv\r\n' % i
                                               for i in range(5)) + FILE_HEAD + b"%PDF-1.7\r\n--x--\r\n", B),
    "a media type that only starts like multipart": (FILE_HEAD + b"%PDF-1.7\r\n--x--\r\n", "multipart/form-datax; boundary=x"),
    "no content type at all": (b"%PDF-1.7", ""),
    "a urlencoded body": (b"file=%25PDF-1.7", "application/x-www-form-urlencoded"),
    "no 'file' part": (b'--x\r\nContent-Disposition: form-data; name="other"\r\n\r\nv\r\n--x--\r\n', B),
}


@pytest.mark.parametrize("name", CASES)
def test_a_malformed_body_is_a_stable_400_and_leaves_nothing(client, caplog, spools, name):
    body, ctype = CASES[name]
    sid = client.post("/api/sessions").json()["id"]
    with caplog.at_level(logging.ERROR):
        r = client.post(f"/api/sessions/{sid}/documents", content=body,
                        headers={"content-type": ctype} if ctype else {})
    assert r.status_code == 400
    assert r.json()["success"] is False and r.json()["error"]["message"] == MALFORMED      # the standard envelope
    assert not [x for x in caplog.records if x.levelno >= logging.ERROR], "no ERROR traceback for a client mistake"
    assert _pieces(sid) == []                                                             # reservation released
    assert quota.usage(_root())[1] == 0
    assert all(f.closed for f in spools), "every spooled temp file is closed"


def test_a_good_upload_still_works_and_closes_its_spool(client, spools):
    sid = client.post("/api/sessions").json()["id"]
    r = client.post(f"/api/sessions/{sid}/documents", files={"file": ("a.pdf", make_pdf(), "application/pdf")})
    assert r.status_code == 201
    assert all(f.closed for f in spools)


def test_a_body_cut_off_by_the_cap_mid_form_closes_its_spool(client, monkeypatch, spools):
    """The counted body cap fires INSIDE the form parser (no Content-Length). The 413 must not leave the file part's
    spooled temp file open for the garbage collector."""
    monkeypatch.setattr(limits, "MAX_BODY_BYTES", 2 * 1024 * 1024)
    sid = client.post("/api/sessions").json()["id"]
    chunk = b"x" * (1 << 20)
    state = {"reads": 0, "status": None}

    async def receive():
        state["reads"] += 1
        if state["reads"] > 6:
            return {"type": "http.request", "body": b"", "more_body": False}
        return {"type": "http.request", "body": FILE_HEAD + b"%PDF-" if state["reads"] == 1 else chunk, "more_body": True}

    async def send(msg):
        if msg["type"] == "http.response.start":
            state["status"] = msg["status"]
    path = f"/api/sessions/{sid}/documents"
    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST", "scheme": "http",
             "path": path, "raw_path": path.encode(), "query_string": b"", "server": ("t", 80), "client": ("c", 1),
             "root_path": "", "headers": [(b"content-type", B.encode())]}
    asyncio.run(main.app(scope, receive, send))
    assert state["status"] == 413
    assert spools and all(f.closed for f in spools)
    assert _pieces(sid) == []


def _drive(sid, chunks, *, end):
    """Send ``chunks`` to the upload route over ASGI, then either finish the body or disconnect. Returns the status."""
    state = {"i": 0, "status": None}

    async def receive():
        if state["i"] < len(chunks):
            state["i"] += 1
            return {"type": "http.request", "body": chunks[state["i"] - 1], "more_body": True}
        if end == "disconnect":
            return {"type": "http.disconnect"}
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(msg):
        if msg["type"] == "http.response.start":
            state["status"] = msg["status"]
    path = f"/api/sessions/{sid}/documents"
    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST", "scheme": "http",
             "path": path, "raw_path": path.encode(), "query_string": b"", "server": ("t", 80), "client": ("c", 1),
             "root_path": "", "headers": [(b"content-type", B.encode())]}
    asyncio.run(main.app(scope, receive, send))
    return state["status"]


def test_an_unterminated_body_closes_its_orphaned_spool(client, caplog, spools):
    """A file part with no closing boundary parses 'successfully' but never reaches the form, so form.close() cannot
    close its spool: the reader must."""
    sid = client.post("/api/sessions").json()["id"]
    with caplog.at_level(logging.ERROR):
        status = _drive(sid, [FILE_HEAD + b"%PDF-1.7 partial"], end="finish")
    assert status == 400
    assert spools and all(f.closed for f in spools)
    assert not [x for x in caplog.records if x.levelno >= logging.ERROR]
    assert _pieces(sid) == []


def test_a_client_that_disconnects_mid_body_is_not_a_server_error(client, caplog, spools):
    sid = client.post("/api/sessions").json()["id"]
    with caplog.at_level(logging.ERROR):
        _drive(sid, [FILE_HEAD + b"%PDF-1.7 partial"], end="disconnect")
    assert not [x for x in caplog.records if x.levelno >= logging.ERROR], "a dropped client is not a server error"
    assert all(f.closed for f in spools)
    assert _pieces(sid) == [] and quota.usage(_root())[1] == 0


def test_a_cancelled_upload_still_releases_its_reservation(client):
    """The request task is cancelled while its body is still arriving (server shutdown, a cancelled task group). The
    reservation and pieces must still be released, not left to hold quota for an hour."""
    sid = client.post("/api/sessions").json()["id"]

    async def go():
        stalled = asyncio.Event()
        reads = {"n": 0}

        async def receive():
            reads["n"] += 1
            if reads["n"] == 1:
                return {"type": "http.request", "body": FILE_HEAD + b"%PDF-1.7", "more_body": True}
            stalled.set()
            await asyncio.Event().wait()                                 # never arrives

        async def send(msg):
            pass
        path = f"/api/sessions/{sid}/documents"
        scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST",
                 "scheme": "http", "path": path, "raw_path": path.encode(), "query_string": b"", "server": ("t", 80),
                 "client": ("c", 1), "root_path": "", "headers": [(b"content-type", B.encode())]}
        task = asyncio.create_task(main.app(scope, receive, send))
        await asyncio.wait_for(stalled.wait(), 20)
        assert any(p.suffix == ".resv" for p in (_root() / sid).iterdir())
        task.cancel()
        try:
            await asyncio.wait_for(task, 20)
        except (asyncio.CancelledError, Exception):                     # noqa: BLE001 - only the files matter here
            pass
    asyncio.run(go())
    assert _pieces(sid) == [] and quota.usage(_root())[1] == 0
