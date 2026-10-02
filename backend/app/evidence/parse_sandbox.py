"""Run the PDF parse in a child process with a hard wall-clock kill, an RSS watchdog and a
process-wide concurrency cap.

The in-process guards (stream inflate caps, the glyph budget, the per-page deadline) bound the cost of
the content they know about; this module bounds everything else. One parse runs in a fresh interpreter
(``python -c`` with the backend on PYTHONPATH, so no state is shared with the server and the caller's
main module is never re-imported; a frozen desktop build uses a multiprocessing 'spawn' child, which
its entry point's freeze_support() serves), killable at any instruction:

- PARSE_WALL_S: the parent kills the child once it has run this long. parse_guard's 120 s budget is
  only checked between pages; this limit is not.
- MAX_PARSE_RSS: the parent polls the child's resident set (psutil when installed, /proc on Linux,
  ``ps`` otherwise, every RSS_POLL_S) and kills it above the cap. RLIMIT_AS is not enforced on macOS,
  so the child also runs its own watchdog thread on ru_maxrss and exits with _MEMORY_EXIT first.
- MAX_CONCURRENT_PARSES: one semaphore for the whole server process, shared by every session, so
  opening new sessions does not multiply concurrent parses. A caller that waits QUEUE_WAIT_S without a
  slot gets ParseBusyError (HTTP 503, nothing recorded; the upload can be retried).

A killed child raises ParseAbortedError and a child exception comes back with its type (ParseLimitError
stays ParseLimitError); ingest records either as parse_status 'failed' with the message.
"""
from __future__ import annotations

import importlib
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from app.evidence.models import IngestError
from app.evidence.parse_guard import ParseLimitError

PARSE_WALL_S = 150.0
MAX_PARSE_RSS = 1024 * 2**20      # measured legit peak ~0.5 GB (20-page paper with ink check)
MAX_CONCURRENT_PARSES = 2
QUEUE_WAIT_S = 30.0
POLL_S = 0.05
RSS_POLL_S = 0.25
DEFAULT_TARGET = "app.evidence.parse:parse_pdf"
_MEMORY_EXIT = 86
_slots = threading.BoundedSemaphore(MAX_CONCURRENT_PARSES)


class ParseAbortedError(IngestError):
    """The parse child was killed (time or memory limit) or died without an answer."""


class ParseBusyError(IngestError):
    """Every parse slot stayed taken for QUEUE_WAIT_S; nothing was parsed or recorded."""


class ParseFailedError(IngestError):
    """The parser raised an unexpected exception inside the child."""


# ------------------------------------------------------------------- child ---

def _own_peak_rss() -> int:
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def _self_watchdog(max_rss: int) -> None:
    while True:
        if _own_peak_rss() > max_rss:
            os._exit(_MEMORY_EXIT)
        time.sleep(POLL_S)


def _child(target: str, path: str, out_path: str, max_rss: int) -> None:
    threading.Thread(target=_self_watchdog, args=(max_rss,), daemon=True).start()
    try:
        module, func = target.split(":")
        result: dict[str, Any] = {"ok": getattr(importlib.import_module(module), func)(path)}
    except BaseException as exc:                     # reported to the parent, never lost
        result = {"error": {"type": type(exc).__name__, "limit": isinstance(exc, ParseLimitError),
                            "missing": isinstance(exc, FileNotFoundError), "message": str(exc)[:500]}}
    Path(out_path).write_text(json.dumps(result, ensure_ascii=False), "utf-8")


def _child_main() -> None:
    """Entry of the ``python -c`` child: argv = target, pdf path, result path, RSS cap in bytes."""
    target, path, out_path, max_rss = sys.argv[1:5]
    _child(target, path, out_path, int(max_rss))


# ------------------------------------------------------------------ parent ---

class _Interpreter:
    """A ``python -c`` child with the multiprocessing.Process methods _watch() uses."""

    def __init__(self, args: tuple[str, str, str, int]) -> None:
        root = str(Path(__file__).resolve().parents[2])          # the directory that holds 'app'
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, (root, os.environ.get("PYTHONPATH"))))}
        code = "from app.evidence.parse_sandbox import _child_main; _child_main()"
        self.proc = subprocess.Popen([sys.executable, "-c", code, *map(str, args)], env=env,
                                     stdin=subprocess.DEVNULL)
        self.pid = self.proc.pid

    def is_alive(self) -> bool:
        return self.proc.poll() is None

    def join(self, timeout: float | None = None) -> None:
        try:
            self.proc.wait(timeout)
        except subprocess.TimeoutExpired:
            pass

    def kill(self) -> None:
        self.proc.kill()

    @property
    def exitcode(self) -> int | None:
        return self.proc.poll()


def _start(args: tuple[str, str, str, int]) -> Any:
    if getattr(sys, "frozen", False):                            # PyInstaller bundle: no python -c
        proc = multiprocessing.get_context("spawn").Process(target=_child, args=args, daemon=True)
        proc.start()
        return proc
    return _Interpreter(args)


def _rss_bytes(pid: int) -> int | None:
    """Resident set of *pid* in bytes, or None when it cannot be read (the wall clock still applies)."""
    try:
        import psutil  # type: ignore[import-not-found]
        return int(psutil.Process(pid).memory_info().rss)
    except ImportError:
        pass
    except Exception:
        return None
    statm = Path(f"/proc/{pid}/statm")
    try:
        if statm.exists():
            return int(statm.read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, timeout=2)
        return int(out.stdout.strip()) * 1024 if out.stdout.strip() else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _kill(proc: Any) -> None:
    proc.kill()
    proc.join(5)


def _watch(proc: Any, wall_s: float, max_rss: int) -> None:
    """Return when the child exits; kill it and raise ParseAbortedError at either limit."""
    deadline, next_rss = time.monotonic() + wall_s, 0.0
    while proc.is_alive():
        now = time.monotonic()
        if now > deadline:
            _kill(proc)
            raise ParseAbortedError(f"parse killed: exceeded the {wall_s:.0f} s time limit")
        if now >= next_rss:
            rss = _rss_bytes(proc.pid)
            if rss is not None and rss > max_rss:
                _kill(proc)
                raise ParseAbortedError(f"parse killed: memory above {max_rss // 2**20} MB")
            next_rss = now + RSS_POLL_S
        proc.join(POLL_S)
    if proc.exitcode == _MEMORY_EXIT:
        raise ParseAbortedError(f"parse killed: memory above {max_rss // 2**20} MB")


def _answer(out: Path, exitcode: int | None) -> dict[str, Any]:
    try:
        result = json.loads(out.read_text("utf-8"))
    except (OSError, ValueError):
        raise ParseAbortedError(f"parse process died without an answer (exit code {exitcode})") from None
    if "ok" in result:
        return result["ok"]
    err = result.get("error") or {}
    message = str(err.get("message") or "")
    if err.get("limit"):
        raise ParseLimitError(message)
    if err.get("missing"):
        raise FileNotFoundError(message)
    raise ParseFailedError(f"{err.get('type', 'Error')}: {message}")


def parse_isolated(path: str | Path, *, wall_s: float | None = None, max_rss: int | None = None,
                   target: str = DEFAULT_TARGET) -> dict[str, Any]:
    """parse_pdf(path) in a killable child process (see module docstring)."""
    wall_s = PARSE_WALL_S if wall_s is None else wall_s
    max_rss = MAX_PARSE_RSS if max_rss is None else max_rss
    slots = _slots
    if not slots.acquire(timeout=QUEUE_WAIT_S):
        raise ParseBusyError(f"the PDF parser is busy ({MAX_CONCURRENT_PARSES} parses running); "
                             "try the upload again shortly")
    scratch = Path(tempfile.mkdtemp(prefix="evparse-"))
    try:
        out = scratch / "result.json"
        proc = _start((target, str(path), str(out), max_rss))
        try:
            _watch(proc, wall_s, max_rss)
        finally:
            if proc.is_alive():
                _kill(proc)
        return _answer(out, proc.exitcode)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
        slots.release()
