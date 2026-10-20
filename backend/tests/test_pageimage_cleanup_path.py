"""The quota cleanup path: a reservation is released only after every piece of its upload is confirmed gone, always
under the storage lock. A delete that fails leaves the reservation (and its full worst-case charge) in place for locked
stale cleanup to recover later, so a failed cleanup can never be used to slip past a quota."""
from __future__ import annotations

import fcntl
import os
import threading
import time
from pathlib import Path

import pytest

from app.config import settings
from app.pageimage import documents, limits, quota

MAX = limits.MAX_PDF_BYTES


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    return tmp_path / "data" / "documents"


def _failed_upload(sid, *, part=b"%PD", tmp=None):
    """A reserved upload with a small (3-byte) .part and optionally a .json.tmp, as a failure path finds it."""
    doc_id, part_path = documents.new_slot(sid, MAX)
    part_path.write_bytes(part)
    if tmp is not None:
        part_path.with_suffix(".json.tmp").write_bytes(tmp)
    return part_path


def _deny_unlink(monkeypatch, *suffixes):
    real = Path.unlink

    def deny(self, *a, **k):
        if self.name.endswith(suffixes):
            raise PermissionError("denied")
        return real(self, *a, **k)
    monkeypatch.setattr(Path, "unlink", deny)


def _age(path, seconds):
    t = time.time() - seconds
    os.utime(path, (t, t))


# ───────────────────── discard: keep the reservation unless every piece is gone ─────────────────────

@pytest.mark.parametrize("stuck", [".part", ".json.tmp"])
def test_a_piece_that_cannot_be_deleted_keeps_the_reservation_and_its_full_charge(store, monkeypatch, stuck):
    part = _failed_upload("s1", tmp=b"{}")
    with monkeypatch.context() as m:
        _deny_unlink(m, stuck)
        documents.discard(part)                                           # never raises on the failure path
    assert part.with_suffix(".resv").exists(), "the reservation must survive a failed piece deletion"
    assert quota.usage(store)[0]["s1"] == (1, MAX)                         # still charged at the worst case


def test_a_failed_cleanup_cannot_be_used_to_exceed_a_quota(store, monkeypatch):
    """With the reservation released early, the 3-byte .part would be all that is charged and a new worst-case upload
    would fit under budgets set just above one upload. With the reservation kept, it does not."""
    monkeypatch.setattr(limits, "MAX_SESSION_BYTES", MAX + 5)
    monkeypatch.setattr(limits, "MAX_GLOBAL_BYTES", MAX + 5)
    part = _failed_upload("s1")
    with monkeypatch.context() as m:
        _deny_unlink(m, ".part")
        documents.discard(part)
    with pytest.raises(documents.QuotaExceeded) as same_session:
        documents.new_slot("s1", MAX)
    assert same_session.value.scope == "session"
    with pytest.raises(documents.QuotaExceeded) as other_session:
        documents.new_slot("s2", MAX)
    assert other_session.value.scope == "global"


def test_with_the_lock_busy_a_failed_upload_stays_charged_until_locked_stale_cleanup(store, monkeypatch):
    part = _failed_upload("s1")
    monkeypatch.setattr(limits, "LOCK_WAIT_SECONDS", 0.05)
    fd = os.open(store / quota.LOCK_NAME, os.O_CREAT | os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        documents.discard(part)                                           # returns; deletes nothing without the lock
    finally:
        os.close(fd)
    assert part.exists() and part.with_suffix(".resv").exists()
    assert quota.usage(store)[1] == MAX
    _age(part.with_suffix(".resv"), limits.STALE_SECONDS + 60)
    removed = quota.cleanup(store)                                        # takes the lock
    assert {f"s1/{part.name}", f"s1/{part.with_suffix('.resv').name}"} <= set(removed)
    assert quota.usage(store) == ({"s1": (0, 0)}, 0)


# ───────────────────── stale cleanup: pieces first, the reservation last ─────────────────────

def test_cleanup_keeps_a_stale_reservation_whose_piece_cannot_be_deleted(store, monkeypatch):
    part = _failed_upload("s1")
    resv = part.with_suffix(".resv")
    _age(resv, limits.STALE_SECONDS + 60)
    with monkeypatch.context() as m:
        _deny_unlink(m, ".part")
        removed = quota.cleanup(store)                                    # does not fail closed for everyone
        assert f"s1/{resv.name}" not in removed and resv.exists()
        assert quota.usage(store)[0]["s1"] == (1, MAX)                    # still charged at the worst case
        documents.new_slot("s2", 10)                                      # other sessions can still upload meanwhile
        assert resv.exists()
    assert {f"s1/{part.name}", f"s1/{resv.name}"} <= set(quota.cleanup(store))   # recovered once deletion works


def test_cleanup_and_a_concurrent_scan_are_serialised(store, monkeypatch):
    """Cleanup is paused after deleting the stale upload's .part, holding the lock. A concurrent admission (which
    scans) must wait, and at that moment the reservation must still exist: pieces go first, the reservation last."""
    part = _failed_upload("s1")
    resv = part.with_suffix(".resv")
    _age(resv, limits.STALE_SECONDS + 60)
    paused, go, admitted = threading.Event(), threading.Event(), threading.Event()
    real = Path.unlink
    seen = {}

    def unlink(self, *a, **k):
        result = real(self, *a, **k)
        if not paused.is_set() and self.parent.name == "s1":
            seen["resv_at_pause"] = resv.exists()
            paused.set()
            assert go.wait(20)
        return result
    monkeypatch.setattr(Path, "unlink", unlink)
    cleaner = threading.Thread(target=lambda: quota.cleanup(store), daemon=True)
    cleaner.start()
    assert paused.wait(20)

    def admit():
        documents.new_slot("s2", 10)
        admitted.set()
    admitter = threading.Thread(target=admit, daemon=True)
    admitter.start()
    try:
        assert not admitted.wait(0.5), "admission scanned while cleanup held the lock"
    finally:
        go.set()
        cleaner.join(20)
        admitter.join(20)
    assert not cleaner.is_alive() and not admitter.is_alive() and admitted.is_set()
    assert seen["resv_at_pause"] is True, "the reservation was deleted before its pieces"
    assert not part.exists() and not resv.exists()


# ───────────────────── pins added after the independent recheck ─────────────────────

def test_a_failed_rollback_json_keeps_the_reservation_until_it_is_gone(store, monkeypatch):
    """Double fault: commit fails at the PDF rename AND its rollback cannot delete the metadata it already wrote. The
    lone .json is a piece of an upload that never completed, so the reservation must not go while it remains."""
    doc_id, part = documents.new_slot("s1", MAX)
    part.write_bytes(b"%PDF-1.7")
    js = part.with_suffix(".json")
    real_replace, real_unlink = os.replace, Path.unlink
    with monkeypatch.context() as m:
        m.setattr(os, "replace", lambda s, d, *a, **k: (_ for _ in ()).throw(OSError("rename")) if str(d).endswith(".pdf")
                  else real_replace(s, d, *a, **k))
        m.setattr(Path, "unlink", lambda self, *a, **k: (_ for _ in ()).throw(PermissionError("x")) if self.suffix == ".json"
                  else real_unlink(self, *a, **k))
        with pytest.raises(OSError):
            documents.commit("s1", doc_id, part, filename="a.pdf", size=8, pages=1)
        documents.discard(part)
        assert js.exists() and part.with_suffix(".resv").exists()           # json stuck -> reservation kept
        assert quota.usage(store)[0]["s1"] == (1, MAX)
    documents.discard(part)                                                 # once it can be deleted, all goes
    assert not js.exists() and not part.with_suffix(".resv").exists()


def test_a_discard_never_touches_a_committed_document(store):
    """A cancellation can arrive after commit finished: discard must leave the document and its metadata alone."""
    doc_id, part = documents.new_slot("s1", MAX)
    pdf_bytes = b"%PDF-1.7"
    part.write_bytes(pdf_bytes)
    meta = documents.commit("s1", doc_id, part, filename="a.pdf", size=len(pdf_bytes), pages=1)
    documents.discard(part)
    assert documents.get("s1", doc_id) == meta


def test_only_the_reservation_delete_failing_is_not_fatal(store, monkeypatch):
    part = _failed_upload("s1")
    resv = part.with_suffix(".resv")
    _age(resv, limits.STALE_SECONDS + 60)
    with monkeypatch.context() as m:
        _deny_unlink(m, ".resv")
        quota.cleanup(store)                                                # does not raise
        assert resv.exists() and not part.exists()
        documents.new_slot("s2", 10)                                        # admissions keep working
    assert quota.usage(store)[0]["s1"] == (1, MAX)                          # still charged until it can go


def test_an_undeterminable_piece_counts_as_present(store, monkeypatch):
    part = _failed_upload("s1")
    real = os.lstat

    def lstat(p, *a, **k):
        if str(p) == str(part):
            raise PermissionError("cannot stat")
        return real(p, *a, **k)
    with monkeypatch.context() as m:
        m.setattr(quota.os, "lstat", lstat)
        documents.discard(part)
    assert part.with_suffix(".resv").exists()                               # not released on an unknown


@pytest.mark.parametrize("kind", ["symlink", "directory"])
def test_a_non_file_in_a_pieces_place_is_left_alone_and_does_not_pin_the_reservation(store, tmp_path, kind):
    """The scan ignores links and directories, so they are not pieces: they are not deleted (a link is not followed)
    and they do not keep the reservation."""
    part = _failed_upload("s1")
    part.unlink()
    target = tmp_path / "outside.bin"
    target.write_bytes(b"keep")
    if kind == "symlink":
        part.symlink_to(target)
    else:
        part.mkdir()
    documents.discard(part)
    assert not part.with_suffix(".resv").exists()
    assert target.read_bytes() == b"keep" and (part.is_symlink() or part.is_dir())
