"""Round-4 review fixes for the layout rebuild endpoint: a refused rebuild is remembered (one parse, not one
per click), the endpoint is admission-controlled and runs on its own limiter outside the session lock, the
status cache covers a full session, and a stored reason never carries a server path."""
from __future__ import annotations

import dataclasses
import json
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from app.config import settings
from app.evidence import parse_sandbox, relayout, relayout_limits, upload_limits
from tests.evidence_image_support import EV, build_papers, make_client, session_with
from tests.test_evidence_r4_relayout import _pages_file, _src, make_old, post, status_of


@pytest.fixture(scope="module")
def papers(tmp_path_factory):
    return build_papers(tmp_path_factory)


@pytest.fixture
def client(tmp_path, monkeypatch):
    relayout_limits.reset()
    upload_limits.reset()
    for c in make_client(tmp_path, monkeypatch):
        with c:                      # one event loop for every request: the limiters are per loop
            yield c
    relayout_limits.reset()


def _doctor(doc):
    doc["pages"][0]["text"] = doc["pages"][0]["text"].replace("3,412", "3,421")


def _count_parses(monkeypatch):
    calls = []
    real = relayout.parse_sandbox.parse_isolated

    def counting(*a, **k):
        calls.append(1)
        return real(*a, **k)
    monkeypatch.setattr(relayout.parse_sandbox, "parse_isolated", counting)
    return calls


# ------------------------------------------------- a refusal is remembered ---

def test_text_changed_source_parses_once_for_repeated_posts(client, papers, monkeypatch):
    sid = session_with(client, {"paper_a": papers["paper_a"].read_bytes()}, extract=False)
    src = _src(client, sid, "paper_a")
    make_old(src, edit=_doctor)
    calls = _count_parses(monkeypatch)
    bodies = [post(client, sid, src.id).json() for _ in range(3)]
    assert [b["outcome"] for b in bodies] == ["text_changed"] * 3
    assert len(calls) == 1
    assert bodies[0]["reason"] == bodies[2]["reason"] and "page 1 text" in bodies[2]["reason"]


def test_refusal_is_retried_when_the_stored_parse_changed(client, papers, monkeypatch):
    sid = session_with(client, {"paper_a": papers["paper_a"].read_bytes()}, extract=False)
    src = _src(client, sid, "paper_a")
    make_old(src, edit=_doctor)
    calls = _count_parses(monkeypatch)
    assert post(client, sid, src.id).json()["outcome"] == "text_changed"
    marker = relayout.marker_file(src).read_bytes()
    make_old(src)                                  # the cache is rewritten (e.g. by a new-session parse) ...
    relayout.marker_file(src).write_bytes(marker)  # ... while the old refusal is still on disk
    doc = json.loads(_pages_file(src).read_text("utf-8"))
    doc["pages"][0]["text"] = doc["pages"][0]["text"].replace("3,421", "3,412")
    _pages_file(src).write_text(json.dumps(doc, ensure_ascii=False), "utf-8")
    assert post(client, sid, src.id).json()["outcome"] == "rebuilt"
    assert len(calls) == 2


def test_marker_without_a_stat_key_is_not_trusted(client, papers, monkeypatch):
    sid = session_with(client, {"paper_a": papers["paper_a"].read_bytes()}, extract=False)
    src = _src(client, sid, "paper_a")
    make_old(src)
    relayout.marker_file(src).write_text(json.dumps({"sha256": src.sha256, "parser": relayout.PARSER_NAME,
                                                     "status": "text_changed", "reason": "old"}), "utf-8")
    calls = _count_parses(monkeypatch)
    assert post(client, sid, src.id).json()["outcome"] == "rebuilt" and len(calls) == 1


def test_unreadable_source_parses_once(client, papers, monkeypatch):
    sid = session_with(client, {"paper_a": papers["paper_a"].read_bytes()}, extract=False)
    src = _src(client, sid, "paper_a")
    make_old(src)
    calls = []

    def corrupt(*a, **k):
        calls.append(1)
        raise parse_sandbox.ParseFailedError("PdfminerException: broken")
    monkeypatch.setattr(relayout.parse_sandbox, "parse_isolated", corrupt)
    assert [post(client, sid, src.id).json()["outcome"] for _ in range(3)] == ["unreadable"] * 3
    assert len(calls) == 1


# ------------------------------------------------------ no path in a reason ---

def test_reasons_never_carry_a_server_path(client, papers):
    sid = session_with(client, {"paper_a": papers["paper_a"].read_bytes()}, extract=False)
    src = _src(client, sid, "paper_a")
    make_old(src)
    _pages_file(src).unlink()                                    # the parse cache is gone
    body = post(client, sid, src.id).json()
    assert body["outcome"] == "unreadable"
    listing = status_of(client, sid, "paper_a")
    for text in (body["reason"], listing["layout_reason"], relayout.marker_file(src).read_text("utf-8")):
        assert str(settings.data_dir) not in text and "Errno" not in text and "pages.json" not in text, text


def test_parse_failure_reason_is_a_fixed_sentence(client, papers, monkeypatch):
    sid = session_with(client, {"paper_a": papers["paper_a"].read_bytes()}, extract=False)
    src = _src(client, sid, "paper_a")
    make_old(src)

    def boom(*a, **k):
        raise RuntimeError(f"secret detail at {settings.data_dir}/x.pdf")
    monkeypatch.setattr(relayout.parse_sandbox, "parse_isolated", boom)
    body = post(client, sid, src.id).json()
    assert body["outcome"] == "unreadable" and "could not be parsed again" in body["reason"]
    assert "secret" not in body["reason"] and str(settings.data_dir) not in body["reason"]


# ------------------------------------------------------------ status cache ---

def test_status_cache_covers_a_full_session(tmp_path, client, papers):
    sid = session_with(client, {"paper_a": papers["paper_a"].read_bytes()}, extract=False)
    base = _src(client, sid, "paper_a")
    srcs = []
    n = upload_limits.MAX_SOURCES_PER_SESSION
    for i in range(n):
        d = tmp_path / f"copy{i}"
        d.mkdir()
        for suffix in (".pages.json", ".layout.json"):
            shutil.copy(Path(base.parse_path).with_name(base.sha256 + suffix), d / (base.sha256 + suffix))
        srcs.append(dataclasses.replace(base, parse_path=str(d / f"{base.sha256}.pages.json")))
    assert all(relayout.layout_info(s)["layout_status"] == "ready" for s in srcs)
    reads = []
    real = relayout._read_json
    relayout._read_json = lambda p: (reads.append(p), real(p))[1]
    try:
        assert all(relayout.layout_info(s)["layout_status"] == "ready" for s in srcs)
    finally:
        relayout._read_json = real
    assert reads == []                                           # second listing: no file was read


# ------------------------------------------------------------- admission ---

def _hold_rebuilds(monkeypatch):
    """rebuild_layout blocks until released; returns (release, entered counter)."""
    release, entered = threading.Event(), []

    def blocked(src, session_id):
        entered.append(session_id)
        release.wait(30)
        return relayout._outcome(src, "unchanged", None, None)
    monkeypatch.setattr(relayout, "rebuild_layout", blocked)
    return release, entered


def test_one_session_cannot_flood_the_endpoint(client, papers, monkeypatch):
    sid = session_with(client, {"paper_a": papers["paper_a"].read_bytes()}, extract=False)
    src = _src(client, sid, "paper_a")
    release, entered = _hold_rebuilds(monkeypatch)
    try:
        with ThreadPoolExecutor(20) as pool:
            futs = [pool.submit(post, client, sid, src.id) for _ in range(20)]
            time.sleep(1.5)
            release.set()
            codes = sorted(f.result(timeout=30).status_code for f in futs)
    finally:
        release.set()
    assert codes.count(200) >= 1 and codes.count(429) >= 15, codes
    assert len(entered) == codes.count(200)


def test_429_carries_retry_after(client, papers, monkeypatch):
    sid = session_with(client, {"paper_a": papers["paper_a"].read_bytes()}, extract=False)
    src = _src(client, sid, "paper_a")
    release, _ = _hold_rebuilds(monkeypatch)
    try:
        t = threading.Thread(target=post, args=(client, sid, src.id))
        t.start()
        time.sleep(0.5)
        r = post(client, sid, src.id)
        assert r.status_code == 429 and int(r.headers["Retry-After"]) >= 1
    finally:
        release.set()
        t.join(30)


def test_rate_window_refuses_after_the_cap(client, papers, monkeypatch):
    monkeypatch.setattr(relayout_limits, "RELAYOUTS_PER_WINDOW", 3)
    sid = session_with(client, {"paper_a": papers["paper_a"].read_bytes()}, extract=False)
    src = _src(client, sid, "paper_a")
    assert [post(client, sid, src.id).status_code for _ in range(5)] == [200, 200, 200, 429, 429]


def test_server_wide_cap_is_a_503(client, papers, monkeypatch):
    monkeypatch.setattr(relayout_limits, "MAX_IN_FLIGHT", 2)
    sids = [session_with(client, {"paper_a": papers["paper_a"].read_bytes()}, extract=False) for _ in range(3)]
    srcs = [_src(client, s, "paper_a") for s in sids]
    release, _ = _hold_rebuilds(monkeypatch)
    try:
        with ThreadPoolExecutor(3) as pool:
            futs = [pool.submit(post, client, s, x.id) for s, x in zip(sids, srcs, strict=True)]
            time.sleep(1.5)
            codes = sorted(f.result(timeout=30).status_code if f.done() else 0 for f in futs)
            release.set()
            [f.result(timeout=30) for f in futs]
    finally:
        release.set()
    assert codes == [0, 0, 503], codes


def test_relayout_flood_does_not_starve_another_sessions_upload(client, papers, monkeypatch):
    """Four sessions each hold a rebuild (the upload limiter has four tokens): an upload from a fifth session
    still finishes, because rebuilds run on their own limiter."""
    sids = [session_with(client, {"paper_a": papers["paper_a"].read_bytes()}, extract=False) for _ in range(4)]
    srcs = [_src(client, s, "paper_a") for s in sids]
    other = client.post("/api/sessions").json()["id"]
    release, entered = _hold_rebuilds(monkeypatch)
    try:
        with ThreadPoolExecutor(4) as pool:
            futs = [pool.submit(post, client, s, x.id) for s, x in zip(sids, srcs, strict=True)]
            time.sleep(1.0)
            t0 = time.monotonic()
            r = client.post(EV.format(sid=other) + "/upload",
                            files={"file": ("p.pdf", papers["paper_b"].read_bytes(), "application/pdf")})
            took = time.monotonic() - t0
            release.set()
            [f.result(timeout=30) for f in futs]
    finally:
        release.set()
    assert r.status_code == 200 and took < 15, (r.status_code, took)


def test_audit_is_readable_while_a_parse_runs(client, papers, monkeypatch):
    """The parse is not run under the session lock: the audit trail of the session answers meanwhile."""
    sid = session_with(client, {"paper_a": papers["paper_a"].read_bytes()}, extract=False)
    src = _src(client, sid, "paper_a")
    make_old(src)
    release, started = threading.Event(), threading.Event()
    real = relayout.parse_sandbox.parse_isolated

    def slow(*a, **k):
        started.set()
        release.wait(30)
        return real(*a, **k)
    monkeypatch.setattr(relayout.parse_sandbox, "parse_isolated", slow)
    t = threading.Thread(target=lambda: post(client, sid, src.id))
    t.start()
    try:
        assert started.wait(30)
        t0 = time.monotonic()
        assert client.get(f"/api/sessions/{sid}/audit").status_code == 200
        assert time.monotonic() - t0 < 2.0
    finally:
        release.set()
        t.join(60)
    entries = [e for e in client.get(f"/api/sessions/{sid}/audit").json()["entries"]
               if e["tool"] == "rebuild_source_layout"]
    assert len(entries) == 1 and "outcome=rebuilt" in entries[0]["action"]       # the real outcome is audited
