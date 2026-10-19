"""Render slots: who may run a sandboxed render right now.

Two kinds of ``flock`` slot file, so the limits hold across every server worker sharing the data directory and are
released by the OS if a process dies:
  <data_dir>/.pageimage-locks/global-<i>
        MAX_CONCURRENT_RENDERS in total.
  <data_dir>/documents/<session id>/.render-<sha256 of the session id>-<j>
        MAX_RENDERS_PER_SESSION per session. The lock sits beside that session's document storage, so it adds no
        namespace of its own, and its name carries the full digest of the exact id: two different sessions never
        share a lock, not even by hash collision and not on a case-insensitive filesystem. (The previous scheme hashed
        ids into 256 groups, so an attacker could pick an id that shared a victim's group and keep it at 429.)
A render takes its session slot FIRST and a global slot second, always in that order (no deadlock), so a session that is
waiting behind its own earlier render holds no global slot and cannot starve other sessions.
Waiting happens on a request thread, so the number of waiters is bounded per process (MAX_WAITING_RENDERS, and
MAX_WAITING_PER_SESSION of those for one session): past that a request is refused at once instead of tying up a thread.
Anything unclear (no ``fcntl``, an unusable lock directory, a session id that is not a plain id, a session directory
that is a link) refuses the render: ``SlotUnavailable``.
This isolates sessions from each other; it does not rank callers (the wait is a poll, not a FIFO queue), and a client
that can open unlimited sessions in open mode can still occupy every global slot.
"""
from __future__ import annotations

import contextlib
import hashlib
import os
import threading
import time
from collections.abc import Iterator
from pathlib import Path

from app.config import settings
from app.pageimage import limits
from app.pageimage.quota import SID_RE

try:                                   # POSIX; elsewhere renders are refused rather than run unmetered
    import fcntl
except ImportError:                    # pragma: no cover
    fcntl = None  # type: ignore[assignment]


class SlotBusy(Exception):
    """No slot became free in time, or too many requests are already waiting."""


class SlotUnavailable(Exception):
    """The slot mechanism itself cannot be used: refuse the render."""


_mutex = threading.Lock()
_waiting_total = 0
_waiting_by_session: dict[str, int] = {}


def _lock_dir() -> Path:
    d = Path(settings.data_dir) / ".pageimage-locks"
    try:
        d.mkdir(parents=True, exist_ok=True, mode=0o700)
    except OSError as exc:
        raise SlotUnavailable(str(exc)) from exc
    return d


def _session_locks(session: str) -> list[Path]:
    """The lock files of exactly this session, inside its (real, non-link) document directory."""
    if not SID_RE.match(session):
        raise SlotUnavailable("invalid session id")
    root = Path(settings.data_dir) / "documents"
    d = root / session
    try:
        if d.is_symlink():
            raise SlotUnavailable("session directory is a link")
        d.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not d.resolve().is_relative_to(root.resolve()):
            raise SlotUnavailable("session directory escapes the documents root")
    except OSError as exc:
        raise SlotUnavailable(str(exc)) from exc
    digest = hashlib.sha256(session.encode("utf-8")).hexdigest()
    return [d / f".render-{digest}-{j}" for j in range(limits.MAX_RENDERS_PER_SESSION)]


def waiting_counts() -> tuple[int, dict[str, int]]:
    """(threads waiting for a slot in this process, the same per session): observable by tests and diagnostics."""
    with _mutex:
        return _waiting_total, dict(_waiting_by_session)


def _try_lock(path: Path) -> int | None:
    try:
        fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    except OSError as exc:
        raise SlotUnavailable(str(exc)) from exc
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fd
    except BlockingIOError:
        os.close(fd)
        return None
    except OSError as exc:
        os.close(fd)
        raise SlotUnavailable(str(exc)) from exc


def _take_one(paths: list[Path], deadline: float) -> int:
    while True:
        for p in paths:
            fd = _try_lock(p)
            if fd is not None:
                return fd
        if time.monotonic() >= deadline:
            raise SlotBusy
        time.sleep(limits.SLOT_POLL_SECONDS)


@contextlib.contextmanager
def render_slot(session: str | None, *, wait: float | None = None) -> Iterator[None]:
    """Hold one render slot for the ``with`` body. ``session`` None means an unattributed caller (global slot only)."""
    global _waiting_total
    if fcntl is None:
        raise SlotUnavailable("no interprocess lock on this platform")
    wait = limits.QUEUE_WAIT_SECONDS if wait is None else wait
    with _mutex:
        if _waiting_total >= limits.MAX_WAITING_RENDERS or (
                session is not None and _waiting_by_session.get(session, 0) >= limits.MAX_WAITING_PER_SESSION):
            raise SlotBusy
        _waiting_total += 1
        if session is not None:
            _waiting_by_session[session] = _waiting_by_session.get(session, 0) + 1
    fds: list[int] = []
    try:
        try:
            deadline = time.monotonic() + wait
            d = _lock_dir()
            if session is not None:
                fds.append(_take_one(_session_locks(session), deadline))
            fds.append(_take_one([d / f"global-{i}" for i in range(limits.MAX_CONCURRENT_RENDERS)], deadline))
        finally:                                        # no longer waiting: holding, or refused
            with _mutex:
                _waiting_total -= 1
                if session is not None:
                    left = _waiting_by_session.get(session, 1) - 1
                    if left > 0:
                        _waiting_by_session[session] = left
                    else:
                        _waiting_by_session.pop(session, None)
        yield
    finally:
        for fd in fds:
            os.close(fd)                                # closing releases the flock
