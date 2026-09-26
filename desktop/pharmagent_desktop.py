"""PharmAgent desktop launcher — one process, one window.

Starts the FastAPI backend (which also serves the built React frontend from
the same origin, see ``app.main`` "static frontend") on a loopback-only port
in a background thread, then opens a native WebKit/WebView window on it.
Closing the window stops the server.

Runs two ways:

* dev:     ``backend/.venv/bin/python desktop/pharmagent_desktop.py``
           (needs ``frontend/dist`` — run ``npm --prefix frontend run build``)
* bundled: ``desktop/dist/PharmAgent.app`` built by ``desktop/build.sh``
           (PyInstaller; ``frontend/dist`` and ``backend/sample_data`` are
           shipped inside the bundle and located via ``sys._MEIPASS``).

Environment is configured BEFORE ``app.main`` is imported because
``app.config.settings`` is evaluated at import time (and creates data_dir).
User data lives in the OS per-user application directory, never next to the
bundle (which is read-only once installed).

Overrides (all optional): ``PHARMAGENT_DESKTOP_PORT`` (default: a free port),
``PHARMAGENT_DESKTOP_NO_WINDOW=1`` (serve only; for smoke tests / CI),
``PHARMAGENT_DESKTOP_DATA_DIR`` (default: per-user app-support dir).
"""
from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

APP_NAME = "PharmAgent"
HOST = "127.0.0.1"
HEALTH_TIMEOUT_S = 40.0
WINDOW_SIZE = (1320, 880)


def resource_root() -> Path:
    """Where bundled read-only resources live: the PyInstaller temp/onedir root
    when frozen, else the repository root."""
    frozen = getattr(sys, "_MEIPASS", None)
    return Path(frozen) if frozen else Path(__file__).resolve().parents[1]


def frontend_dist(root: Path) -> Path | None:
    """The built frontend: ``frontend_dist/`` inside a bundle, ``frontend/dist``
    in the repo. None when it has not been built."""
    for cand in (root / "frontend_dist", root / "frontend" / "dist"):
        if (cand / "index.html").is_file():
            return cand
    return None


def user_data_dir() -> Path:
    """Per-user writable directory for the SQLite DB, uploaded datasets and
    reports. Honours PHARMAGENT_DESKTOP_DATA_DIR."""
    override = os.environ.get("PHARMAGENT_DESKTOP_DATA_DIR")
    if override:
        base = Path(override)
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support" / APP_NAME
    elif os.name == "nt":
        base = Path(os.environ.get("APPDATA", str(Path.home()))) / APP_NAME
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))) / APP_NAME
    base.mkdir(parents=True, exist_ok=True)
    return base


def free_port() -> int:
    """An ephemeral loopback port the OS just handed out (tiny race window,
    acceptable for a single-user desktop launcher)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, 0))
        return int(s.getsockname()[1])


def configure_env(*, root: Path, data_dir: Path) -> dict[str, str]:
    """Point the backend at bundled resources and per-user writable storage.
    ``setdefault`` so an operator's explicit PHARMAGENT_* env still wins.
    Returns the values in effect (for logging / tests)."""
    dist = frontend_dist(root)
    if dist is None:
        raise RuntimeError(
            "frontend build not found — run `npm --prefix frontend run build` "
            f"(looked under {root})")
    values = {
        "PHARMAGENT_FRONTEND_DIST": str(dist),
        "PHARMAGENT_DATA_DIR": str(data_dir / "data"),
        "PHARMAGENT_DB_PATH": str(data_dir / "pharmagent.db"),
    }
    for k, v in values.items():
        os.environ.setdefault(k, v)
    return {k: os.environ[k] for k in values}


def wait_for_health(url: str, timeout_s: float = HEALTH_TIMEOUT_S) -> dict:
    """Poll ``/api/health`` until the server answers; raise on timeout."""
    deadline = time.monotonic() + timeout_s
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url + "/api/health", timeout=2) as r:  # noqa: S310 loopback
                return json.loads(r.read().decode())
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
            last_err = e
            time.sleep(0.15)
    raise RuntimeError(f"backend did not come up at {url} within {timeout_s:.0f}s: {last_err}")


class BackendServer:
    """uvicorn in a daemon thread; ``stop()`` asks it to exit cleanly."""

    def __init__(self, port: int) -> None:
        import uvicorn

        # dev layout: the backend package lives in <repo>/backend and is not
        # installed; the bundle puts it at the resource root (spec pathex).
        backend_src = resource_root() / "backend"
        if backend_src.is_dir() and str(backend_src) not in sys.path:
            sys.path.insert(0, str(backend_src))
        from app.main import app  # imported AFTER configure_env()

        self.port = port
        self.url = f"http://{HOST}:{port}"
        self._server = uvicorn.Server(uvicorn.Config(
            app, host=HOST, port=port, log_level="warning", access_log=False))
        self._thread = threading.Thread(target=self._server.run, name="pharmagent-backend",
                                        daemon=True)

    def start(self) -> None:
        self._thread.start()
        wait_for_health(self.url)

    def stop(self, timeout_s: float = 5.0) -> None:
        self._server.should_exit = True
        self._thread.join(timeout_s)


def main() -> int:
    root = resource_root()
    data_dir = user_data_dir()
    env = configure_env(root=root, data_dir=data_dir)
    port = int(os.environ.get("PHARMAGENT_DESKTOP_PORT") or free_port())
    server = BackendServer(port)
    server.start()
    print(f"{APP_NAME} backend at {server.url} · data in {data_dir}", flush=True)
    for k, v in env.items():
        print(f"  {k}={v}", flush=True)

    if os.environ.get("PHARMAGENT_DESKTOP_NO_WINDOW") == "1":
        # headless smoke mode: serve until interrupted (used by build verification)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        server.stop()
        return 0

    import webview  # pywebview; imported late so headless mode needs no GUI libs

    webview.create_window(APP_NAME, server.url, width=WINDOW_SIZE[0], height=WINDOW_SIZE[1],
                          min_size=(960, 640))
    try:
        webview.start()
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
