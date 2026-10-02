"""Fair scheduling of the parse slots.

A plain semaphore lets two sessions that keep both slots busy return 503 to every other session.
FairSlots instead:

- lets a session hold at most one slot at a time (a second parse of the same session queues behind
  its first, even while another slot is free);
- queues waiting parses and grants a free slot to the next session in rotation, so a session with many
  queued uploads gets one grant per round and cannot starve the others (first come, first served
  between sessions, first in, first out inside one);
- bounds the wait (``wait_s``, about 20 s in parse_sandbox) and the queue (in total and per session);
  past either, ParseBusyError carries ``retry_after`` and the route answers 503 with a Retry-After header.

The key is the session id the caller passes. When every client shares one anonymous owner, fairness is
per session id only: one client can open many sessions. A multi-user deployment needs auth-level quotas
(per user, not per session id) in front of this scheduler. A call without a session id gets a key of its
own, so unattributed callers neither share a slot nor starve each other.
"""
from __future__ import annotations

import threading
import uuid
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager

from app.evidence.models import IngestError

RETRY_AFTER_S = 10


class ParseBusyError(IngestError):
    """No parse slot within the bounded wait, or the wait queue is full; nothing was parsed or
    recorded and the upload can be retried after ``retry_after`` seconds (HTTP 503 + Retry-After)."""

    def __init__(self, message: str, retry_after: int = RETRY_AFTER_S) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class _Waiter:
    __slots__ = ("event", "granted")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.granted = False


class FairSlots:
    def __init__(self, capacity: int, max_queue: int, max_queued_per_session: int) -> None:
        self._capacity = capacity
        self._max_queue = max_queue
        self._max_per_session = max_queued_per_session
        self._lock = threading.Lock()
        self._held: set[str] = set()
        self._queues: dict[str, deque[_Waiter]] = {}
        self._order: deque[str] = deque()          # sessions with waiters, next to be served first
        self._waiting = 0

    @property
    def queued(self) -> int:
        with self._lock:
            return self._waiting

    def _busy(self, why: str) -> ParseBusyError:
        return ParseBusyError(f"the PDF parser is busy ({why}); try the upload again in "
                              f"{RETRY_AFTER_S} s", RETRY_AFTER_S)

    def _dispatch(self) -> None:
        """Grant free slots to the next eligible sessions in rotation (caller holds the lock)."""
        while len(self._held) < self._capacity and self._order:
            for _ in range(len(self._order)):
                session = self._order[0]
                self._order.rotate(-1)             # served or skipped: to the back of the rotation
                if session not in self._held:
                    break
            else:
                return                             # every queued session already holds its one slot
            waiter = self._queues[session].popleft()
            self._waiting -= 1
            self._held.add(session)
            waiter.granted = True
            waiter.event.set()
            if not self._queues[session]:
                self._drop_session(session)

    def _drop_session(self, session: str) -> None:
        del self._queues[session]
        self._order.remove(session)

    def acquire(self, session: str, wait_s: float) -> None:
        """Block until *session* holds a slot; ParseBusyError when the queue is full or the wait ends."""
        waiter = _Waiter()
        with self._lock:
            queue = self._queues.get(session)
            if queue is None:
                queue = self._queues[session] = deque()
                self._order.append(session)
            queue.append(waiter)
            self._waiting += 1
            self._dispatch()
            if waiter.granted:
                return
            if self._waiting > self._max_queue or len(queue) > self._max_per_session:
                self._withdraw(session, waiter)
                raise self._busy("too many uploads are waiting")
        if waiter.event.wait(wait_s):
            return
        with self._lock:
            if waiter.granted:                     # granted between the timeout and the lock
                return
            self._withdraw(session, waiter)
        raise self._busy(f"no slot within {wait_s:.0f} s")

    def _withdraw(self, session: str, waiter: _Waiter) -> None:
        queue = self._queues[session]
        queue.remove(waiter)
        self._waiting -= 1
        if not queue:
            self._drop_session(session)

    def release(self, session: str) -> None:
        with self._lock:
            self._held.discard(session)
            self._dispatch()

    @contextmanager
    def slot(self, session: str | None, wait_s: float) -> Iterator[None]:
        key = session or f"anonymous-{uuid.uuid4().hex}"
        self.acquire(key, wait_s)
        try:
            yield
        finally:
            self.release(key)
