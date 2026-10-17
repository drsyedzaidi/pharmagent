"""Interprocess-safe storage accounting for the document store.

Everything that decides "may this upload start" or "is this file abandoned" runs under ONE advisory file lock
(``<documents root>/.quota.lock``, ``flock``), so server workers sharing a data directory cannot jointly exceed
a quota. A ``threading.Lock`` would only cover one process.

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
Cleanup removes only the pieces of an upload that never became a complete document: a stale reservation and whatever
half-finished .part/.json.tmp/.pdf/.json it left, and such pieces with no reservation once they are older than the
stale age. A complete document (.pdf AND .json) is never touched (no retention policy yet).
Commit writes the metadata first and the PDF last, so a half-committed upload is never an inaccessible PDF holding
quota: it is either complete or made only of pieces cleanup reclaims.
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
    has_pdf: bool = False
    part: int = 0
    has_part: bool = False
    has_json: bool = False
    tmp: bool = False
    resv: dict | None = None
    resv_mtime: float = 0.0
    part_mtime: float = 0.0
    tmp_mtime: float = 0.0
    pdf_mtime: float = 0.0
    json_mtime: float = 0.0
    resv_corrupt: bool = False

    @property
    def committed(self) -> bool:
        return self.has_pdf and self.has_json

    def bytes(self) -> int:
        if self.committed:
            return self.pdf                                    # a leftover reservation no longer holds anything
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
                    doc.pdf, doc.has_pdf, doc.pdf_mtime = st.st_size, True, st.st_mtime
                elif kind == "json":
                    doc.has_json, doc.json_mtime = True, st.st_mtime
                elif kind == "part":
                    doc.part, doc.has_part, doc.part_mtime = st.st_size, True, st.st_mtime
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
    if doc.committed:
        return "committed"
    if doc.resv is not None or doc.resv_corrupt:
        return "active" if now - doc.resv_mtime <= stale_after else "stale"
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
        if d.is_symlink():
            raise ValueError("session directory is a link")
        path = d / f"{doc_id}.resv"
        created = False
        try:
            d.mkdir(parents=True, exist_ok=True)
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            created = True
            try:
                fh = os.fdopen(fd, "w", encoding="utf-8")
            except BaseException:
                os.close(fd)
                raise
            with fh:
                json.dump({"v": 1, "id": doc_id, "sid": sid, "bytes": nbytes, "pid": os.getpid(),
                           "created": now}, fh)
            os.utime(path, (now, now))
        except OSError as exc:
            if created:                                    # never leave a half-made reservation holding quota
                with contextlib.suppress(OSError):
                    path.unlink()
            raise QuotaUnavailable(str(exc)) from exc
        return path


def release(resv: Path) -> None:
    """Drop a reservation (upload failed or finished). Idempotent; lowers usage, so no lock is needed."""
    resv.unlink(missing_ok=True)


def heartbeat(root: Path, resv: Path) -> None:
    """Mark the upload alive. Takes the SAME lock as cleanup, so cleanup can never classify an upload stale and delete
    it after this refresh; if the reservation was already reclaimed it says so instead of resurrecting a timestamp."""
    with locked(root):
        if not resv.is_file():
            raise ReservationLostError
        try:
            os.utime(resv, None)
        except OSError as exc:
            raise QuotaUnavailable(str(exc)) from exc


def _cleanup_locked(root: Path, now: float) -> list[str]:
    removed: list[str] = []
    cutoff = now - limits.STALE_SECONDS
    for d in _session_dirs(root):
        for doc_id, doc in _scan(d).items():
            files = {"resv": (doc.resv is not None or doc.resv_corrupt, doc.resv_mtime),
                     "part": (doc.has_part, doc.part_mtime),
                     "json.tmp": (doc.tmp, doc.tmp_mtime),
                     "pdf": (doc.has_pdf, doc.pdf_mtime),
                     "json": (doc.has_json, doc.json_mtime)}
            victims: list[str] = []
            if files["resv"][0]:
                if doc.resv_mtime <= cutoff:                               # stale reservation
                    victims.append("resv")
                    if not doc.committed:                                  # its upload, in any half-finished form
                        victims += [k for k in ("part", "json.tmp", "pdf", "json") if files[k][0]]
            elif not doc.committed:                                        # no reservation, not a complete document
                victims += [k for k in ("part", "json.tmp", "pdf", "json") if files[k][0] and files[k][1] <= cutoff]
            for ext in victims:
                name = f"{doc_id}.{ext}"
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
