"""A fake child process for parse_sandbox tests."""
from __future__ import annotations

import time


class FakeProc:
    """The multiprocessing.Process surface _watch() uses: alive for *lifetime* seconds, or until killed."""

    def __init__(self, pid: int = 4242, lifetime: float = 60.0, exitcode: int | None = 0) -> None:
        self.pid, self.killed, self.joins = pid, False, []
        self._end, self._exit = time.monotonic() + lifetime, exitcode

    def is_alive(self) -> bool:
        return not self.killed and time.monotonic() < self._end

    def join(self, timeout: float | None = None) -> None:
        self.joins.append(timeout)
        time.sleep(min(timeout or 0.01, 0.01))

    def kill(self) -> None:
        self.killed = True

    @property
    def exitcode(self) -> int | None:
        return -9 if self.killed else self._exit
