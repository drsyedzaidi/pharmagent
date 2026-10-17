"""Step-5 hardening, item 2: atomic quotas and stale-file cleanup in the document store.

Quotas (document count, per-session bytes, global bytes) are taken under one interprocess file lock, so concurrent
uploads, and concurrent server workers, cannot exceed them. Reservations and ``.part`` files count. Cleanup removes only
stale temporary/reservation files, never an active upload and never a committed PDF. Anything uncertain refuses the
upload; nothing is deleted to make room."""
from __future__ import annotations

import fcntl
import itertools
import multiprocessing
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.config import settings
from app.core.llm import MockLLM
from app.core.orchestrator import Orchestrator
from app.core.store import SessionStore
from app.pageimage import documents, limits, quota
from tests.pdf_fixtures import make_pdf, unenforced_limits_off_linux

_weak = pytest.fixture(autouse=True)(unenforced_limits_off_linux)

HOUR = limits.STALE_SECONDS


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    return tmp_path / "data" / "documents"


def _age(path: Path, seconds: float) -> None:
    t = time.time() - seconds
    os.utime(path, (t, t))


def _files(root: Path, sid: str) -> set[str]:
    d = root / sid
    return {p.name for p in d.iterdir()} if d.exists() else set()


def _try(sid: str, nbytes: int) -> str:
    try:
        documents.new_slot(sid, nbytes)
        return "ok"
    except documents.DocumentLimitError:
        return "count"
    except documents.QuotaExceeded as e:
        return e.scope
    except documents.QuotaUnavailable:
        return "unavailable"


# ───────────────────────────── atomic reservations ─────────────────────────────

def test_simultaneous_uploads_never_exceed_the_document_limit(store):
    with ThreadPoolExecutor(max_workers=24) as ex:
        out = list(ex.map(lambda _: _try("s1", 10), range(80)))
    assert out.count("ok") == limits.MAX_DOCS_PER_SESSION
    assert out.count("count") == 80 - limits.MAX_DOCS_PER_SESSION
    assert len([n for n in _files(store, "s1") if n.endswith(".resv")]) == limits.MAX_DOCS_PER_SESSION


def _worker(args):
    root, sid, n = args
    settings.data_dir = Path(root)
    return [_try(sid, 10) for _ in range(n)]


def test_several_processes_share_one_limit(store, tmp_path):
    ctx = multiprocessing.get_context("spawn")             # fresh interpreters: no state shared except the directory
    with ctx.Pool(6) as pool:
        results = pool.map(_worker, [(str(tmp_path / "data"), "s1", 10)] * 6)
    flat = [r for rs in results for r in rs]
    assert flat.count("ok") == limits.MAX_DOCS_PER_SESSION and set(flat) <= {"ok", "count"}


def test_simultaneous_reservations_never_exceed_the_session_byte_budget(store, monkeypatch):
    monkeypatch.setattr(limits, "MAX_SESSION_BYTES", 100)
    with ThreadPoolExecutor(max_workers=16) as ex:
        out = list(ex.map(lambda _: _try("s1", 30), range(16)))
    assert out.count("ok") == 3 and set(out) <= {"ok", "session"}
    _, total = quota.usage(store)
    assert total == 90


def test_global_byte_budget_holds_across_different_sessions(store, monkeypatch):
    monkeypatch.setattr(limits, "MAX_GLOBAL_BYTES", 100)
    with ThreadPoolExecutor(max_workers=16) as ex:
        out = list(ex.map(lambda i: _try(f"sess{i}", 30), range(16)))
    assert out.count("ok") == 3 and set(out) <= {"ok", "global"}
    assert quota.usage(store)[1] == 90


def test_part_files_and_reservations_both_count(store, monkeypatch):
    monkeypatch.setattr(limits, "MAX_SESSION_BYTES", 100)
    _, part = documents.new_slot("s1", 10)
    assert quota.usage(store)[1] == 10                        # the reservation alone
    part.write_bytes(b"x" * 70)
    assert quota.usage(store)[1] == 70                        # the growing part exceeds its reservation: it counts
    assert _try("s1", 40) == "session"                        # 70 + 40 > 100
    assert _try("s1", 30) == "ok"


def test_a_committed_document_counts_once(store, monkeypatch):
    doc_id, part = documents.new_slot("s1", 5000)
    pdf = make_pdf()
    part.write_bytes(pdf)
    documents.commit("s1", doc_id, part, filename="a.pdf", size=len(pdf), pages=1)
    per, total = quota.usage(store)
    assert per["s1"] == (1, len(pdf)) and total == len(pdf)    # not reservation + pdf
    assert not (store / "s1" / f"{doc_id}.resv").exists()


def test_failed_upload_releases_its_reservation(store):
    _, part = documents.new_slot("s1", 50)
    part.write_bytes(b"x")
    documents.discard(part)
    assert _files(store, "s1") == set() and quota.usage(store) == ({"s1": (0, 0)}, 0)


# ───────────────────────────── crashes and cleanup ─────────────────────────────

def test_a_crash_leaves_a_reservation_that_counts_then_is_recovered(store, monkeypatch):
    monkeypatch.setattr(limits, "MAX_DOCS_PER_SESSION", 2)
    slots = [documents.new_slot("s1", 10) for _ in range(2)]          # two uploads start, then the process dies
    slots[0][1].write_bytes(b"%PDF-partial")
    assert _try("s1", 10) == "count"                                  # still counted: the quota is not leaked open
    for doc_id, _ in slots:
        _age(store / "s1" / f"{doc_id}.resv", HOUR + 60)              # time passes beyond the stale age
    assert _try("s1", 10) == "ok"                                     # reclaimed on the next reservation
    left = _files(store, "s1")
    assert not any(n.startswith(slots[0][0]) or n.startswith(slots[1][0]) for n in left)


def test_cleanup_spares_active_and_committed_and_only_removes_stale(store):
    done_id, done_part = documents.new_slot("s1", 10)                 # committed long ago
    pdf = make_pdf()
    done_part.write_bytes(pdf)
    documents.commit("s1", done_id, done_part, filename="a.pdf", size=len(pdf), pages=1)
    for ext in ("pdf", "json"):
        _age(store / "s1" / f"{done_id}.{ext}", HOUR * 24 * 365)

    fresh_id, fresh_part = documents.new_slot("s1", 10)               # active
    fresh_part.write_bytes(b"%PDF-live")
    slow_id, slow_part = documents.new_slot("s1", 10)                 # active but slow: old data, fresh heartbeat
    slow_part.write_bytes(b"%PDF-slow")
    _age(slow_part, HOUR * 5)
    _age(store / "s1" / f"{slow_id}.resv", HOUR * 5)
    documents.touch(slow_part)                                        # heartbeat arrives
    dead_id, dead_part = documents.new_slot("s1", 10)                 # stale
    dead_part.write_bytes(b"%PDF-dead")
    _age(dead_part, HOUR * 2)
    _age(store / "s1" / f"{dead_id}.resv", HOUR * 2)

    removed = quota.cleanup(store)

    assert sorted(removed) == sorted([f"s1/{dead_id}.resv", f"s1/{dead_id}.part"])
    left = _files(store, "s1")
    assert {f"{fresh_id}.resv", f"{fresh_id}.part", f"{slow_id}.resv", f"{slow_id}.part",
            f"{done_id}.pdf", f"{done_id}.json"} <= left
    assert not any(n.startswith(dead_id) for n in left)


def test_abandoned_temporaries_are_removed_after_the_age_but_orphan_committed_files_never(store):
    d = store / "s1"
    d.mkdir(parents=True)
    ids = [f"{i:032x}" for i in range(1, 7)]
    old_part, new_part, old_tmp, old_pdf, old_json, new_tmp = ids
    for name in (f"{old_part}.part", f"{old_tmp}.json.tmp", f"{old_pdf}.pdf", f"{old_json}.json"):
        (d / name).write_bytes(b"x")
        _age(d / name, HOUR * 3)
    (d / f"{new_part}.part").write_bytes(b"x")
    (d / f"{new_tmp}.json.tmp").write_bytes(b"x")
    removed = quota.cleanup(store)
    assert sorted(removed) == sorted([f"s1/{old_part}.part", f"s1/{old_tmp}.json.tmp"])
    assert _files(store, "s1") == {f"{old_pdf}.pdf", f"{old_json}.json", f"{new_part}.part", f"{new_tmp}.json.tmp"}


def test_state_classification(store):
    now = time.time()
    active, stale = quota._Doc(resv={"bytes": 1}, resv_mtime=now), quota._Doc(resv={"bytes": 1}, resv_mtime=now - HOUR - 1)
    assert quota.classify(active, now) == "active" and quota.classify(stale, now) == "stale"
    assert quota.classify(quota._Doc(pdf=5, has_json=True), now) == "committed"
    assert quota.classify(quota._Doc(part=5), now) == "abandoned"
    assert quota.classify(quota._Doc(resv_corrupt=True, resv_mtime=now), now) == "active"   # unreadable: counted as live


def test_commit_after_the_reservation_was_reclaimed_is_refused(store):
    doc_id, part = documents.new_slot("s1", 10)
    part.write_bytes(make_pdf())
    _age(store / "s1" / f"{doc_id}.resv", HOUR + 60)
    quota.cleanup(store)
    with pytest.raises(documents.ReservationLostError):
        documents.commit("s1", doc_id, part, filename="a.pdf", size=10, pages=1)
    assert not (store / "s1" / f"{doc_id}.pdf").exists()


# ───────────────────────────── fail closed ─────────────────────────────

def test_unknown_usage_refuses_and_deletes_nothing(store, monkeypatch):
    doc_id, part = documents.new_slot("s1", 10)
    part.write_bytes(make_pdf())
    documents.commit("s1", doc_id, part, filename="a.pdf", size=10, pages=1)
    before = _files(store, "s1")

    def boom(_):
        raise quota.QuotaUnavailable("disk error")
    monkeypatch.setattr(quota, "_scan", boom)
    assert _try("s1", 10) == "unavailable"
    monkeypatch.undo()
    assert _files(store, "s1") == before


def test_a_busy_lock_refuses_rather_than_proceeding_unlocked(store, monkeypatch):
    monkeypatch.setattr(limits, "LOCK_WAIT_SECONDS", 0.05)
    store.mkdir(parents=True)
    fd = os.open(store / quota.LOCK_NAME, os.O_CREAT | os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        assert _try("s1", 10) == "unavailable"
    finally:
        os.close(fd)
    assert _try("s1", 10) == "ok"


def test_a_corrupt_reservation_is_counted_at_the_worst_case_not_ignored(store, monkeypatch):
    monkeypatch.setattr(limits, "MAX_SESSION_BYTES", limits.MAX_PDF_BYTES + 10)
    d = store / "s1"
    d.mkdir(parents=True)
    (d / f"{'a' * 32}.resv").write_text("{not json", encoding="utf-8")
    assert quota.usage(store)[1] == limits.MAX_PDF_BYTES
    assert _try("s1", 100) == "session"


# ───────────────────────────── path confinement ─────────────────────────────

def test_cleanup_never_follows_links_or_touches_foreign_names(store, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    victim = outside / "victim.part"
    victim.write_bytes(b"keep me")
    _age(victim, HOUR * 9)
    hostile_id = "b" * 32
    d = store / "s1"
    d.mkdir(parents=True)
    (d / f"{hostile_id}.part").symlink_to(victim)                      # a stale-looking link to an outside file
    (store / "linked").symlink_to(outside, target_is_directory=True)   # a session "directory" that is a link
    (outside / f"{'c' * 32}.part").write_bytes(b"also keep")
    _age(outside / f"{'c' * 32}.part", HOUR * 9)
    for name in ("notanid.part", f"{'D' * 32}.part", f"{'d' * 31}.part", f"{'e' * 32}.part.bak", "..part"):
        (d / name).write_bytes(b"x")
        _age(d / name, HOUR * 9)
    (d / f"{'f' * 32}.part").mkdir()                                   # a directory wearing a part name
    (store / f"{'9' * 32}.part").write_bytes(b"x")                     # a loose file in the root, not in a session
    _age(store / f"{'9' * 32}.part", HOUR * 9)
    (store / "bad name!").mkdir()

    assert quota.cleanup(store) == []
    assert victim.read_bytes() == b"keep me" and (outside / f"{'c' * 32}.part").exists()
    assert (d / f"{hostile_id}.part").is_symlink() and (store / f"{'9' * 32}.part").exists()
    assert {p.name for p in d.iterdir()} >= {"notanid.part", f"{'D' * 32}.part", "..part"}


# ───────────────────────────── through the HTTP API ─────────────────────────────

@pytest.fixture
def client(store, monkeypatch):
    monkeypatch.setattr(settings, "pageimage_enabled", True)
    main.orch = Orchestrator(llm=MockLLM(), clock=lambda c=itertools.count(): f"t{next(c)}",
                             store=SessionStore(":memory:"))
    token = settings.api_token
    yield TestClient(main.app)
    settings.api_token = token


def _up(client, sid, data=None):
    return client.post(f"/api/sessions/{sid}/documents",
                       files={"file": ("a.pdf", data if data is not None else make_pdf(), "application/pdf")})


def test_failed_uploads_do_not_consume_the_quota(client, store):
    sid = client.post("/api/sessions").json()["id"]
    for _ in range(limits.MAX_DOCS_PER_SESSION + 5):
        assert _up(client, sid, b"not a pdf").status_code == 400
    assert not [n for n in _files(store, sid) if n.endswith((".resv", ".part"))]
    assert all(_up(client, sid).status_code == 201 for _ in range(limits.MAX_DOCS_PER_SESSION))
    assert _up(client, sid).status_code == 409


def test_api_reports_byte_quotas_distinctly(client, monkeypatch):
    sid, other = (client.post("/api/sessions").json()["id"] for _ in range(2))
    size = len(make_pdf())
    monkeypatch.setattr(limits, "MAX_SESSION_BYTES", size * 2 + 400)
    assert _up(client, sid).status_code == 201
    assert _up(client, sid).status_code == 201
    assert _up(client, sid).status_code == 409                         # session byte budget: a client-side limit
    monkeypatch.setattr(limits, "MAX_SESSION_BYTES", 10 * limits.MAX_PDF_BYTES)
    monkeypatch.setattr(limits, "MAX_GLOBAL_BYTES", size * 2 + 400)
    r = _up(client, other)
    assert r.status_code == 503


def test_api_refuses_when_accounting_is_unavailable(client, monkeypatch):
    sid = client.post("/api/sessions").json()["id"]

    def boom(*a, **k):
        raise quota.QuotaUnavailable("x")
    monkeypatch.setattr(quota, "usage", boom)
    assert _up(client, sid).status_code == 503
