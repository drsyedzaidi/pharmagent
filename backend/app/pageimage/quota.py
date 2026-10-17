"""Interprocess-safe storage accounting for the document store.

Everything that decides "may this upload start" or "is this file abandoned" runs under ONE advisory file lock
(``<documents root>/.quota.lock``, ``flock``), so several server workers sharing a data directory cannot jointly
exceed a quota. A ``threading.Lock`` would only cover one process.

Per document id the directory may hold:
  <id>.resv      reservation: JSON {v, id, sid, bytes, pid, created}; its mtime is the upload's heartbeat
  <id>.part      the streaming upload
  <id>.pdf/.json the committed document and its metadata
  <id>.json.tmp  metadata being written
States (see ``classify``): active = fresh reservation; stale = reservation older than STALE_SECONDS;
committed = pdf+json;
abandoned = .part/.json.tmp with no fresh reservation, or a reservation without an upload that has gone stale.
Accounting counts every id once: its bytes are max(reserved, part + pdf), so a reservation, its growing .part and the
committed pdf are never double counted and never under counted. Anything unreadable raises QuotaUnavailable and the
upload is refused: the store fails closed and never deletes data to make room.
Cleanup removes ONLY stale .resv/.part/.json.tmp. Committed .pdf/.json are never touched (no retention policy yet).
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from app.pageimage import limits

try:                                   # POSIX. Elsewhere the store refuses uploads rather than run unlocked.
    import fcntl
except ImportError:                    # pragma: no cover
    fcntl = None  # type: ignore[assignment]

SID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_NAME_RE = re.compile(r"^([0-9a-f]{32})\.(pdf|json|part|resv|json\.tmp)$")
LOCK_NAME = ".quota.lock"


class DocumentLimitError(Exception):
    """Per-session document count reached (committed + in-flight)."""


class QuotaExceeded(Exception):
    """A byte budget would be exceeded. ``scope`` is "session" or "global"."""

    def __init__(self, scope: str) -> None:
        super().__init__(scope)
        self.scope = scope


class ReservationLostError(Exception):
    """The upload sat idle past the stale age and its reservation was reclaimed before it could commit."""


class QuotaUnavailable(Exception):
    """Usage could not be determined (lock, scan or I/O failure): refuse, never guess."""


@dataclass
class _Doc:
    pdf: int = 0
    part: int = 0
    has_json: bool = False
    tmp: bool = False
    resv: dict | None = None
    resv_mtime: float = 0.0
    part_mtime: float = 0.0
    tmp_mtime: float = 0.0
    resv_corrupt: bool = False

    def bytes(self) -> int:
        reserved = 0
        if self.resv_corrupt:
            reserved = limits.MAX_PDF_BYTES                     # unreadable reservation: assume the worst
        elif self.resv is not None:
            reserved = int(self.resv.get("bytes", limits.MAX_PDF_BYTES))
        return max(reserved, self.part + self.pdf)


@contextlib.contextmanager
def locked(root: Path) -> Iterator[None]:
    if fcntl is None:
        raise QuotaUnavailable("no interprocess lock on this platform")
    try:
        root.mkdir(parents=True, exist_ok=True)
        fd = os.open(root / LOCK_NAME, os.O_CREAT | os.O_RDWR, 0o600)
    except OSError as exc:
        raise QuotaUnavailable(str(exc)) from exc
    try:
        deadline = time.monotonic() + limits.LOCK_WAIT_SECONDS
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise QuotaUnavailable("storage lock busy") from None
                time.sleep(0.005)
        try:
            yield
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _session_dirs(root: Path) -> list[Path]:
    """Real (non-symlink) directories with a plain session-id name, directly under ``root``."""
    out = []
    try:
        with os.scandir(root) as it:
            for e in it:
                if SID_RE.match(e.name) and e.is_dir(follow_symlinks=False):
                    out.append(Path(e.path))
    except FileNotFoundError:
        return []
    except OSError as exc:
        raise QuotaUnavailable(str(exc)) from exc
    return out


def _scan(d: Path) -> dict[str, _Doc]:
    docs: dict[str, _Doc] = {}
    try:
        with os.scandir(d) as it:
            for e in it:
                m = _NAME_RE.match(e.name)
                if m is None or e.is_symlink() or not e.is_file(follow_symlinks=False):
                    continue                                   # foreign names, links and directories are ignored
                st = e.stat(follow_symlinks=False)
                doc = docs.setdefault(m.group(1), _Doc())
                kind = m.group(2)
                if kind == "pdf":
                    doc.pdf = st.st_size
                elif kind == "json":
                    doc.has_json = True
                elif kind == "part":
                    doc.part, doc.part_mtime = st.st_size, st.st_mtime
                elif kind == "json.tmp":
                    doc.tmp, doc.tmp_mtime = True, st.st_mtime
                else:
                    doc.resv_mtime = st.st_mtime
                    try:
                        raw = json.loads(Path(e.path).read_text(encoding="utf-8"))
                        if not isinstance(raw, dict) or int(raw["bytes"]) < 0:
                            raise ValueError
                        doc.resv = raw
                    except (OSError, ValueError, KeyError, TypeError):
                        doc.resv_corrupt = True
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise QuotaUnavailable(str(exc)) from exc
    return docs


def classify(doc: _Doc, now: float, stale_after: float | None = None) -> str:
    """active | stale | committed | abandoned (see module docstring)."""
    stale_after = limits.STALE_SECONDS if stale_after is None else stale_after
    if doc.resv is not None or doc.resv_corrupt:
        return "active" if now - doc.resv_mtime <= stale_after else "stale"
    if doc.pdf and doc.has_json:
        return "committed"
    return "abandoned"


def usage(root: Path) -> tuple[dict[str, tuple[int, int]], int]:
    """({sid: (documents, bytes)}, total bytes). Caller holds the lock."""
    per: dict[str, tuple[int, int]] = {}
    total = 0
    for d in _session_dirs(root):
        docs = _scan(d)
        n, b = len(docs), sum(x.bytes() for x in docs.values())
        per[d.name] = (n, b)
        total += b
    return per, total


def reserve(root: Path, sid: str, doc_id: str, nbytes: int, now: float | None = None) -> Path:
    """Atomically take one document slot and ``nbytes`` of both budgets, or raise. Returns the reservation path."""
    now = time.time() if now is None else now
    if not SID_RE.match(sid) or not _NAME_RE.match(f"{doc_id}.resv"):
        raise ValueError("invalid reservation")
    nbytes = max(0, int(nbytes))
    with locked(root):
        _cleanup_locked(root, now)
        per, total = usage(root)
        n, b = per.get(sid, (0, 0))
        if n >= limits.MAX_DOCS_PER_SESSION:
            raise DocumentLimitError
        if b + nbytes > limits.MAX_SESSION_BYTES:
            raise QuotaExceeded("session")
        if total + nbytes > limits.MAX_GLOBAL_BYTES:
            raise QuotaExceeded("global")
        d = root / sid
        path = d / f"{doc_id}.resv"
        try:
            d.mkdir(parents=True, exist_ok=True)
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump({"v": 1, "id": doc_id, "sid": sid, "bytes": nbytes, "pid": os.getpid(),
                           "created": now}, fh)
            os.utime(path, (now, now))
        except OSError as exc:
            raise QuotaUnavailable(str(exc)) from exc
        return path


def release(resv: Path) -> None:
    """Drop a reservation (upload failed or finished). Idempotent; lowers usage, so no lock is needed."""
    resv.unlink(missing_ok=True)


def heartbeat(resv: Path) -> None:
    """Mark the upload alive so cleanup never treats it as abandoned."""
    with contextlib.suppress(OSError):
        os.utime(resv, None)


def _cleanup_locked(root: Path, now: float) -> list[str]:
    removed: list[str] = []
    cutoff = now - limits.STALE_SECONDS
    for d in _session_dirs(root):
        for doc_id, doc in _scan(d).items():
            state = classify(doc, now)
            victims: list[str] = []
            if state == "stale":
                victims = [f"{doc_id}.resv"]
                if not (doc.pdf and doc.has_json):             # never the committed pair
                    victims.append(f"{doc_id}.part")
            elif state == "abandoned":
                if doc.part and doc.part_mtime <= cutoff:
                    victims.append(f"{doc_id}.part")
                if doc.tmp and doc.tmp_mtime <= cutoff:
                    victims.append(f"{doc_id}.json.tmp")
            for name in victims:
                p = d / name
                try:
                    if p.is_symlink() or not p.is_file():
                        continue
                    p.unlink()
                    removed.append(f"{d.name}/{name}")
                except FileNotFoundError:
                    continue
                except OSError as exc:
                    raise QuotaUnavailable(str(exc)) from exc
    return removed


def cleanup(root: Path, now: float | None = None) -> list[str]:
    """Remove stale temporary/reservation files only. Returns what was removed ("sid/name")."""
    with locked(root):
        return _cleanup_locked(root, time.time() if now is None else now)
