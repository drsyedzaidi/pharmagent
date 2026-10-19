"""The quota scan versus concurrent deletions (B2 of the PR #2 review).

Every deletion of upload pieces (release, discard) now runs under the storage lock, so a scan, which also holds it,
cannot see a file disappear half-way. And if something outside the app removes a file mid-scan anyway, the scan must
never report the whole session as empty: it retries from a fresh listing and fails closed if the directory will not hold
still."""
from __future__ import annotations

import fcntl
import os
import threading
import time
from pathlib import Path

import pytest

from app.config import settings
from app.pageimage import documents, limits, quota

MIB = 1024 * 1024


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    return tmp_path / "data" / "documents"


def _reserve(sid, n):
    return [documents.new_slot(sid, limits.MAX_PDF_BYTES) for _ in range(n)]


class _Listing:
    """An os.scandir stand-in: lists the directory for real, then runs ``after_listing`` before the caller stats."""

    def __init__(self, path, after_listing):
        self.entries = list(_REAL_SCANDIR(path))
        after_listing()

    def __enter__(self):
        return iter(self.entries)

    def __exit__(self, *exc):
        return False


_REAL_SCANDIR = os.scandir


def test_one_vanished_entry_never_empties_a_session(store, monkeypatch):
    _reserve("s1", 3)
    victim = sorted((store / "s1").glob("*.resv"))[0]
    calls = {"n": 0}

    def scandir(path):
        if os.path.basename(path) != "s1":
            return _REAL_SCANDIR(path)
        calls["n"] += 1
        return _Listing(path, (lambda: victim.unlink(missing_ok=True)) if calls["n"] == 1 else (lambda: None))
    monkeypatch.setattr(quota.os, "scandir", scandir)
    per, total = quota.usage(store)
    assert per["s1"] == (2, 2 * limits.MAX_PDF_BYTES) and total == 2 * limits.MAX_PDF_BYTES    # a fresh, true snapshot
    assert calls["n"] >= 2                                                                      # it re-listed


def test_a_directory_that_will_not_hold_still_fails_closed(store, monkeypatch):
    _reserve("s1", 1)
    ghost = store / "s1" / f"{'e' * 32}.part"

    def scandir(path):
        if os.path.basename(path) != "s1":
            return _REAL_SCANDIR(path)
        ghost.write_bytes(b"x")                                  # appears in every listing...
        return _Listing(path, lambda: ghost.unlink())            # ...and is gone before it can be stat-ed
    monkeypatch.setattr(quota.os, "scandir", scandir)
    with pytest.raises(quota.QuotaUnavailable):
        quota.usage(store)
    with pytest.raises(quota.QuotaUnavailable):                  # and admission refuses rather than guess
        documents.new_slot("s2", 10)


def test_a_discard_waits_for_a_scan_that_holds_the_lock(store, monkeypatch):
    """A scan is paused after listing s1, holding the storage lock. A concurrent discard() of one of s1's uploads must
    block until the scan finishes, so the scan sees a consistent directory."""
    (doc_id, part), _ = _reserve("s1", 2)
    part.write_bytes(b"%PDF-partial")
    listed, go, deleted = threading.Event(), threading.Event(), threading.Event()

    def scandir(path):
        if os.path.basename(path) != "s1" or listed.is_set():
            return _REAL_SCANDIR(path)

        def pause():
            listed.set()
            assert go.wait(20)
        return _Listing(path, pause)
    monkeypatch.setattr(quota.os, "scandir", scandir)
    seen = {}

    def scan():
        with quota.locked(store):
            seen["usage"] = quota.usage(store)

    monkeypatch.setattr(limits, "DISCARD_LOCK_WAIT_SECONDS", 20.0)       # wait out the paused scan, no fallback

    def delete():
        documents.discard(part)
        deleted.set()
    scanner = threading.Thread(target=scan, daemon=True)
    scanner.start()
    assert listed.wait(20)
    remover = threading.Thread(target=delete, daemon=True)
    remover.start()
    try:
        assert not deleted.wait(0.5), "the deletion ran inside the scan's critical section"
    finally:
        go.set()
        scanner.join(20)
        remover.join(20)
    assert not scanner.is_alive() and not remover.is_alive()
    assert seen["usage"][0]["s1"] == (2, 2 * limits.MAX_PDF_BYTES)          # the scan saw both, consistently
    assert deleted.is_set() and not part.with_suffix(".resv").exists()


def test_a_discard_that_cannot_get_the_lock_still_releases_without_raising(store, monkeypatch):
    """discard() runs on the upload's failure path: it must not raise, and it must not leave a reservation holding a
    document slot and 50 MiB for an hour just because the lock is busy. After a short wait it deletes the pieces
    without the lock; the scan tolerates that (it re-lists when an entry vanishes)."""
    ((doc_id, part),) = _reserve("s1", 1)
    part.write_bytes(b"%PDF-partial")
    fd = os.open(store / quota.LOCK_NAME, os.O_CREAT | os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        t0 = time.monotonic()
        documents.discard(part)                                          # returns, does not raise
        assert time.monotonic() - t0 < limits.DISCARD_LOCK_WAIT_SECONDS + 1.0
    finally:
        os.close(fd)
    assert not part.exists() and not part.with_suffix(".resv").exists()
    assert quota.usage(store)[1] == 0


def test_a_discard_whose_deletes_fail_never_raises(store, monkeypatch):
    ((doc_id, part),) = _reserve("s1", 1)
    part.write_bytes(b"%PDF-partial")
    real = Path.unlink

    def deny(self, *a, **k):
        if self.suffix in (".part", ".resv"):
            raise PermissionError("denied")
        return real(self, *a, **k)
    monkeypatch.setattr(Path, "unlink", deny)
    documents.discard(part)                                              # logs, does not raise


def test_a_reservation_that_is_not_utf8_counts_at_the_worst_case_and_blocks_no_one(store):
    """Corruption must not take uploads down: an undecodable reservation is counted at 50 MiB, every other session can
    still reserve, and stale cleanup reclaims it."""
    d = store / "s1"
    d.mkdir(parents=True)
    bad = d / f"{'a' * 32}.resv"
    bad.write_bytes(b"\xff\xfe\x00garbage")
    assert quota.usage(store) == ({"s1": (1, limits.MAX_PDF_BYTES)}, limits.MAX_PDF_BYTES)
    documents.new_slot("s2", 10)                                         # other sessions are unaffected
    old = time.time() - limits.STALE_SECONDS - 60
    os.utime(bad, (old, old))
    assert f"s1/{bad.name}" in quota.cleanup(store)


def test_a_reservation_that_vanishes_between_stat_and_read_is_relisted(store, monkeypatch):
    _reserve("s1", 2)
    victim = sorted((store / "s1").glob("*.resv"))[0]
    real = Path.read_text
    calls = {"n": 0}

    def flaky(self, *a, **k):
        if self == victim and calls["n"] == 0:
            calls["n"] += 1
            self.unlink()                                                # gone after its stat, before its read
        return real(self, *a, **k)
    monkeypatch.setattr(Path, "read_text", flaky)
    assert quota.usage(store) == ({"s1": (1, limits.MAX_PDF_BYTES)}, limits.MAX_PDF_BYTES)   # re-listed, not charged
