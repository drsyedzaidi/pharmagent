"""Parent side of the sandbox: spawn ``worker.py`` with a cleared environment,
an empty temp cwd, a wall-clock kill and a hard cap on bytes read from stdout.

Only typed, pre-validated values reach the worker (one JSON argv element, no
shell). Failures surface as ``RenderError`` carrying a coarse ``kind``; the
worker's stderr is discarded and no path or exception text ever reaches a caller.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

from app.config import settings
from app.pageimage import limits
from app.pageimage.types import RenderParams

log = logging.getLogger("pharmagent.pageimage")

WORKER_PATH = Path(__file__).resolve().with_name("worker.py")

# Worker exit codes (mirrors worker.py; duplicated because the worker is
# deliberately import-free).
_EXIT_KINDS = {
    2: "bad_request",
    3: "encrypted",
    4: "malformed",
    5: "no_page",
    6: "too_large",
    7: "empty",
    8: "output_too_big",
    9: "limit_unavailable",
    10: "memory_limit",
}
_SIGXCPU = getattr(signal, "SIGXCPU", None)

_MAX_REPORT_BYTES = 8192
REPORT_VERSION = 1
CPU_HARD_SLACK = 2        # mirrors worker.CPU_HARD_SLACK (a test pins the pair)


def expected_limits() -> dict[str, tuple[int, int]]:
    """name -> (requested ceiling, allowed extra on the hard limit): the ONLY limits a report may hold."""
    return {
        "RLIMIT_CPU": (limits.CPU_SECONDS, CPU_HARD_SLACK),
        "RLIMIT_AS": (limits.ADDRESS_SPACE_BYTES, 0),
        "RLIMIT_FSIZE": (0, 0),
        "RLIMIT_CORE": (0, 0),
        "RLIMIT_NOFILE": (limits.NOFILE_LIMIT, 0),
    }


def _is_int(v: object) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _attest(stderr: bytes) -> bool:
    """Validate the worker's attestation (protocol v1) and return True when every limit is ENFORCED.

    Fail-closed: exactly one report line; version 1; ``ok`` true; exactly the expected limits, once each, with the
    requested value we asked for; an enforced limit needs a confirmed soft <= request and soft <= hard <= request
    (CPU: + slack). Anything else raises ``limit_unavailable``: silence, an empty or partial list, a duplicate, a
    wrong value. Unenforced limits are accepted only under the UNSAFE operator override (and logged loudly)."""
    reports = []
    for line in stderr.decode("utf-8", "replace").splitlines():
        try:
            got = json.loads(line)
        except ValueError:
            continue
        if isinstance(got, dict) and "pageimage_limits" in got:
            reports.append(got["pageimage_limits"])
    if len(reports) != 1 or not isinstance(reports[0], dict):
        raise RenderError("limit_unavailable")
    rep = reports[0]
    if rep.get("v") != REPORT_VERSION or rep.get("ok") is not True or not isinstance(rep.get("limits"), list):
        raise RenderError("limit_unavailable")
    want = expected_limits()
    seen: set[str] = set()
    all_enforced = True
    for e in rep["limits"]:
        if not isinstance(e, dict) or set(e) != {"name", "requested", "soft", "hard", "state"}:
            raise RenderError("limit_unavailable")
        name = e["name"]
        if name not in want or name in seen or e["requested"] != want[name][0]:
            raise RenderError("limit_unavailable")
        seen.add(name)
        request, slack = want[name]
        if e["state"] == "enforced":
            soft, hard = e["soft"], e["hard"]
            if not (_is_int(soft) and _is_int(hard) and 0 <= soft <= request and soft <= hard <= request + slack):
                raise RenderError("limit_unavailable")
        elif e["state"] == "unenforced":
            all_enforced = False
            if any(v is not None and not _is_int(v) for v in (e["soft"], e["hard"])):
                raise RenderError("limit_unavailable")
        else:
            raise RenderError("limit_unavailable")
    if seen != set(want):
        raise RenderError("limit_unavailable")
    if not all_enforced:
        if not _allow_unenforced():
            raise RenderError("limit_unavailable")
        log.warning("pageimage_unsafe_unenforced_limits: rendering WITHOUT every resource limit enforced "
                    "(UNSAFE operator override; never enable on a server exposed to untrusted users)")
    return all_enforced


_slots = threading.BoundedSemaphore(limits.MAX_CONCURRENT_RENDERS)


class RenderError(Exception):
    """``kind`` is one of: bad_request encrypted malformed no_page too_large
    empty output_too_big timeout cpu_limit memory_limit limit_unavailable
    failed unavailable busy."""

    def __init__(self, kind: str) -> None:
        super().__init__(kind)
        self.kind = kind


def _allow_unenforced() -> bool:
    """The UNSAFE operator override for hosts that cannot enforce every limit; never honoured in a frozen build."""
    return bool(settings.pageimage_unsafe_allow_unenforced_limits) and not getattr(sys, "frozen", False)


def _worker_limits() -> dict:
    return {
        "allow_unenforced": _allow_unenforced(),
        "cpu_seconds": limits.CPU_SECONDS,
        "address_space": limits.ADDRESS_SPACE_BYTES,
        "nofile": limits.NOFILE_LIMIT,
        "fsize": 0,
        "core": 0,
        "max_dim": limits.MAX_DIM_PX,
        "max_pixels": limits.MAX_PIXELS,
        "max_png_bytes": limits.MAX_PNG_BYTES,
        "max_pages": limits.MAX_PAGES,
    }


def _clean_env() -> dict[str, str]:
    env: dict[str, str] = {}
    if os.name == "nt":  # DLL loading needs it
        env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", "")
    return env


def _kill(proc: subprocess.Popen) -> None:
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except (ProcessLookupError, PermissionError, OSError):
        pass


def _spawn_and_collect(request: dict, *, wall_seconds: float, max_output: int) -> bytes:
    if getattr(sys, "frozen", False):  # PyInstaller bundle: sys.executable is not a Python
        raise RenderError("unavailable")
    if not _slots.acquire(timeout=limits.QUEUE_WAIT_SECONDS):
        raise RenderError("busy")
    try:
        with tempfile.TemporaryDirectory(prefix="pageimage_") as cwd:
            try:
                proc = subprocess.Popen(
                    [sys.executable, "-I", str(WORKER_PATH), json.dumps(request)],
                    stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    cwd=cwd, env=_clean_env(), close_fds=True,
                    start_new_session=(os.name == "posix"),
                )
            except OSError:
                log.exception("pageimage_spawn_failed")
                raise RenderError("unavailable") from None
            chunks: list[bytes] = []
            state = {"overflow": False}

            def _reader() -> None:
                total = 0
                assert proc.stdout is not None
                while True:
                    data = proc.stdout.read(65536)
                    if not data:
                        return
                    total += len(data)
                    if total > max_output:
                        state["overflow"] = True
                        _kill(proc)
                        return
                    chunks.append(data)

            err_chunks: list[bytes] = []

            def _err_reader() -> None:
                assert proc.stderr is not None
                total = 0
                while data := proc.stderr.read(4096):
                    total += len(data)
                    if total <= _MAX_REPORT_BYTES:     # keep reading (never block the worker), keep little
                        err_chunks.append(data)

            t = threading.Thread(target=_reader, daemon=True)
            te = threading.Thread(target=_err_reader, daemon=True)
            t.start()
            te.start()
            try:
                proc.wait(timeout=wall_seconds)
            except subprocess.TimeoutExpired:
                _kill(proc)
                proc.wait()
                t.join(timeout=2)
                te.join(timeout=2)
                _attest(b"".join(err_chunks))          # a kill after parsing began still needs the attestation
                raise RenderError("timeout") from None
            t.join(timeout=5)
            te.join(timeout=5)
            rc = proc.returncode
            if rc == 2:                                 # before any limit or parse: nothing to attest
                raise RenderError("bad_request")
            if rc == 9:                                 # setup failure, reported as such
                raise RenderError("limit_unavailable")
            _attest(b"".join(err_chunks))              # EVERY other outcome happened after limits + parsing began
            if state["overflow"]:
                raise RenderError("output_too_big")
            if rc == 0:
                return b"".join(chunks)
            if rc in _EXIT_KINDS:
                raise RenderError(_EXIT_KINDS[rc])
            if _SIGXCPU is not None and rc == -_SIGXCPU:
                raise RenderError("cpu_limit")
            log.error("pageimage_worker_failed", extra={"returncode": rc})
            raise RenderError("failed")
    finally:
        _slots.release()


def probe_pdf(path: Path, *, wall_seconds: float | None = None) -> int:
    """Page count of an untrusted PDF, measured inside the sandbox."""
    out = _spawn_and_collect(
        {"op": "info", "path": str(path), "limits": _worker_limits()},
        wall_seconds=wall_seconds or limits.WALL_SECONDS, max_output=4096)
    try:
        pages = int(json.loads(out.decode("ascii"))["pages"])
    except (ValueError, KeyError, TypeError):
        raise RenderError("failed") from None
    return pages


def render_page(path: Path, page: int, params: RenderParams, *,
                wall_seconds: float | None = None) -> bytes:
    """PNG bytes of one page (optionally cropped), rendered inside the sandbox."""
    bbox = None
    if params.bbox is not None:
        b = params.bbox
        bbox = [b.x0, b.y0, b.x1, b.y1]
    png = _spawn_and_collect(
        {"op": "render", "path": str(path), "page": page, "scale": params.scale,
         "bbox": bbox, "limits": _worker_limits()},
        wall_seconds=wall_seconds or limits.WALL_SECONDS, max_output=limits.MAX_PNG_BYTES)
    if not png.startswith(b"\x89PNG\r\n\x1a\n"):
        raise RenderError("failed")
    return png
