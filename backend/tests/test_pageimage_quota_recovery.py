"""Quota store: serialised heartbeat, half-finished commits, link rejection, failed reservations."""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import pytest

from app.config import settings
from app.pageimage import documents, limits, quota
from tests.pdf_fixtures import make_pdf

HOUR = limits.STALE_SECONDS


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    return tmp_path / "data" / "documents"


def _age(path: Path, seconds: float) -> None:
    t = time.time() - seconds
    os.utime(path, (t, t))


def _files(root: Path, sid: str = "s1") -> set[str]:
    d = root / sid
    return {p.name for p in d.iterdir()} if d.exists() else set()


# ───────────────────── heartbeat vs cleanup ─────────────────────

def test_cleanup_cannot_delete_an_upload_after_a_concurrent_heartbeat(store, monkeypatch):
    """Cleanup has classified the upload stale and is about to delete it; a heartbeat arrives. The heartbeat must wait for
    the lock (it cannot slip in between classify and delete), and afterwards must NOT resurrect the reclaimed file."""
    doc_id, part = documents.new_slot("s1", 10)
    part.write_bytes(b"%PDF-")
    resv = store / "s1" / f"{doc_id}.resv"
    _age(resv, HOUR + 60)
    scanned, go, hb_done = threading.Event(), threading.Event(), threading.Event()
    real_scan = quota._scan

    def paused(d):
        result = real_scan(d)
        if not scanned.is_set():
            scanned.set()
            assert go.wait(20)
        return result
    monkeypatch.setattr(quota, "_scan", paused)
    removed, hb_error = [], []
    cleaner = threading.Thread(target=lambda: removed.extend(quota.cleanup(store)))
    cleaner.start()
    assert scanned.wait(20)                       # cleanup holds the lock, stale classification already made

    def beat():
        try:
            documents.touch(part)
        except Exception as exc:                  # noqa: BLE001 - the test inspects it
            hb_error.append(exc)
        hb_done.set()
    beater = threading.Thread(target=beat)
    beater.start()
    assert not hb_done.wait(0.5), "the heartbeat ran inside cleanup's critical section"
    go.set()
    cleaner.join(20)
    beater.join(20)
    assert f"s1/{doc_id}.resv" in removed and not resv.exists()
    assert len(hb_error) == 1 and isinstance(hb_error[0], documents.ReservationLostError)


def test_a_heartbeat_that_lands_first_keeps_the_upload_alive(store):
    doc_id, part = documents.new_slot("s1", 10)
    part.write_bytes(b"%PDF-")
    _age(store / "s1" / f"{doc_id}.resv", HOUR + 60)
    documents.touch(part)
    assert quota.cleanup(store) == []
    assert {f"{doc_id}.resv", f"{doc_id}.part"} <= _files(store)


# ───────────────────── half-finished commits ─────────────────────

class Boom(OSError):
    pass


def _fail_when(m, where: str) -> None:
    real_replace, real_write = os.replace, Path.write_text

    def replace(src, dst, *a, **k):
        if str(dst).endswith(where):
            raise Boom(where)
        return real_replace(src, dst, *a, **k)

    def write_text(self, *a, **k):
        if where == ".json.tmp" and self.name.endswith(".json.tmp"):
            raise Boom(where)
        return real_write(self, *a, **k)
    m.setattr(os, "replace", replace)
    m.setattr(Path, "write_text", write_text)


def _slot_with_pdf(store):
    doc_id, part = documents.new_slot("s1", 5000)
    pdf = make_pdf()
    part.write_bytes(pdf)
    return doc_id, part, len(pdf)


@pytest.mark.parametrize("where", [".json.tmp", ".json", ".pdf"])
def test_a_failed_commit_leaves_nothing_behind_once_the_caller_discards(store, monkeypatch, where):
    doc_id, part, n = _slot_with_pdf(store)
    with monkeypatch.context() as m:
        _fail_when(m, where)
        with pytest.raises(Boom):
            documents.commit("s1", doc_id, part, filename="a.pdf", size=n, pages=1)
    assert documents.get("s1", doc_id) is None and documents.list_documents("s1") == []
    documents.discard(part)                                   # what the upload endpoint does on every failure
    assert _files(store) == set() and quota.usage(store)[1] == 0


@pytest.mark.parametrize("where", [".json.tmp", ".json", ".pdf"])
def test_a_crash_mid_commit_is_never_a_permanent_inaccessible_file(store, monkeypatch, where):
    """The process dies (no rollback, no discard). The pieces must still count toward quota, must not be reachable as a
    document, and must be fully reclaimed once the reservation goes stale."""
    doc_id, part, n = _slot_with_pdf(store)
    with monkeypatch.context() as m:
        _fail_when(m, where)
        m.setattr(documents, "_rollback", lambda *a: None)               # a dead process runs no cleanup code
        with pytest.raises(Boom):
            documents.commit("s1", doc_id, part, filename="a.pdf", size=n, pages=1)
    assert documents.get("s1", doc_id) is None and documents.list_documents("s1") == []
    assert quota.usage(store)[1] > 0                                     # still counted: not a free leak
    assert quota.cleanup(store) == []                                    # fresh reservation: left alone
    for p in (store / "s1").iterdir():
        _age(p, HOUR + 60)
    quota.cleanup(store)
    assert _files(store) == set() and quota.usage(store) == ({"s1": (0, 0)}, 0)


def test_a_failure_to_release_the_reservation_does_not_undo_a_committed_document(store, monkeypatch):
    doc_id, part, n = _slot_with_pdf(store)

    def boom(_):
        raise Boom("release")
    with monkeypatch.context() as m:
        m.setattr(quota, "_release_locked", boom)                  # commit releases with the lock already held
        meta = documents.commit("s1", doc_id, part, filename="a.pdf", size=n, pages=1)
    assert documents.get("s1", doc_id) == meta
    assert quota.usage(store)[1] == n                        # a complete document counts its PDF, not the reservation
    _age(store / "s1" / f"{doc_id}.resv", HOUR + 60)
    assert quota.cleanup(store) == [f"s1/{doc_id}.resv"]
    assert documents.get("s1", doc_id) == meta


def test_no_filesystem_change_when_the_lock_cannot_be_taken_for_a_commit(store, monkeypatch):
    doc_id, part, n = _slot_with_pdf(store)
    before = _files(store)

    def busy(root):
        raise quota.QuotaUnavailable("busy")
    monkeypatch.setattr(quota, "locked", busy)
    with pytest.raises(quota.QuotaUnavailable):
        documents.commit("s1", doc_id, part, filename="a.pdf", size=n, pages=1)
    assert _files(store) == before


# ───────────────────── symlinked session directories ─────────────────────

def test_a_session_directory_that_is_a_link_is_rejected_even_inside_the_root(store):
    doc_id, part = documents.new_slot("s1", 10)
    part.write_bytes(make_pdf())
    documents.commit("s1", doc_id, part, filename="a.pdf", size=10, pages=1)
    (store / "alias").symlink_to(store / "s1", target_is_directory=True)     # points INSIDE the documents root
    assert documents.get("alias", doc_id) is None
    assert documents.list_documents("alias") == []
    assert documents.pdf_path("alias", doc_id) is None
    with pytest.raises(ValueError):
        documents.new_slot("alias", 10)
    with pytest.raises(ValueError):
        quota.reserve(store, "alias", "a" * 32, 10)
    assert _files(store, "s1") == {f"{doc_id}.pdf", f"{doc_id}.json"}         # the real directory is untouched
    assert quota.usage(store)[0].keys() == {"s1"}                             # the link is not a session either


def test_cleanup_does_not_walk_an_inside_the_root_link(store):
    d = store / "s1"
    d.mkdir(parents=True)
    stale = d / f"{'a' * 32}.part"
    stale.write_bytes(b"x")
    _age(stale, HOUR * 5)
    (store / "alias").symlink_to(d, target_is_directory=True)
    assert quota.cleanup(store) == [f"s1/{'a' * 32}.part"]                   # reclaimed once, under its real name


# ───────────────────── reservation creation failures ─────────────────────

@pytest.mark.parametrize("target", ["utime", "dump", "fdopen"])
def test_a_failed_reservation_leaves_no_partial_file_and_no_quota(store, monkeypatch, target):
    import json as jsonmod

    def boom(*a, **k):
        raise OSError(target)
    with monkeypatch.context() as m:
        m.setattr(*{"utime": (os, "utime"), "dump": (jsonmod, "dump"), "fdopen": (os, "fdopen")}[target], boom)
        with pytest.raises(quota.QuotaUnavailable):
            documents.new_slot("s1", 10)
    assert _files(store) == set()
    assert quota.usage(store) == ({"s1": (0, 0)}, 0)
    documents.new_slot("s1", 10)                                              # and the slot is genuinely free
