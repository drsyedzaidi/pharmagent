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
}
_SIGXCPU = getattr(signal, "SIGXCPU", None)

_slots = threading.BoundedSemaphore(limits.MAX_CONCURRENT_RENDERS)


class RenderError(Exception):
    """``kind`` is one of: bad_request encrypted malformed no_page too_large
    empty output_too_big timeout failed unavailable busy."""

    def __init__(self, kind: str) -> None:
        super().__init__(kind)
        self.kind = kind


def _worker_limits() -> dict:
    return {
        "cpu_seconds": limits.CPU_SECONDS,
        "address_space": limits.ADDRESS_SPACE_BYTES,
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
                    stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
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

            t = threading.Thread(target=_reader, daemon=True)
            t.start()
            try:
                proc.wait(timeout=wall_seconds)
            except subprocess.TimeoutExpired:
                _kill(proc)
                proc.wait()
                t.join(timeout=2)
                raise RenderError("timeout") from None
            t.join(timeout=5)
            if state["overflow"]:
                raise RenderError("output_too_big")
            rc = proc.returncode
            if rc == 0:
                return b"".join(chunks)
            if rc in _EXIT_KINDS:
                raise RenderError(_EXIT_KINDS[rc])
            if _SIGXCPU is not None and rc == -_SIGXCPU:
                raise RenderError("timeout")
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
