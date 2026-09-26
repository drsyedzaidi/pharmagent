"""Desktop launcher (desktop/pharmagent_desktop.py): resource resolution,
per-user storage, env wiring and the in-process backend server. No window is
opened; the launcher's GUI import is deferred to main() so none of this needs
pywebview installed.
"""
from __future__ import annotations

import importlib.util
import socket
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
LAUNCHER = REPO / "desktop" / "pharmagent_desktop.py"


@pytest.fixture(scope="module")
def launcher():
    spec = importlib.util.spec_from_file_location("pharmagent_desktop", LAUNCHER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@pytest.fixture
def clean_env():
    """Snapshot/restore os.environ. configure_env() writes with setdefault, and
    monkeypatch.delenv on an ABSENT key records nothing to undo, so without
    this the PHARMAGENT_* values would leak into later tests (and into the
    first import of app.main, which reads PHARMAGENT_FRONTEND_DIST once)."""
    import os
    saved = dict(os.environ)
    for k in ("PHARMAGENT_FRONTEND_DIST", "PHARMAGENT_DATA_DIR", "PHARMAGENT_DB_PATH"):
        os.environ.pop(k, None)
    yield
    os.environ.clear()
    os.environ.update(saved)


def _fake_dist(root: Path, name: str) -> Path:
    d = root / name
    (d / "assets").mkdir(parents=True)
    (d / "index.html").write_text('<div id="root"></div>')
    return d


def test_resource_root_is_repo_root_when_not_frozen(launcher, monkeypatch):
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)
    assert launcher.resource_root() == REPO


def test_resource_root_is_meipass_when_frozen(launcher, monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    assert launcher.resource_root() == tmp_path


def test_frontend_dist_prefers_bundle_layout_then_repo_layout(launcher, tmp_path):
    assert launcher.frontend_dist(tmp_path) is None
    repo_dist = _fake_dist(tmp_path / "frontend", "dist")
    assert launcher.frontend_dist(tmp_path) == repo_dist
    bundle_dist = _fake_dist(tmp_path, "frontend_dist")
    assert launcher.frontend_dist(tmp_path) == bundle_dist


def test_user_data_dir_honours_override_and_creates_it(launcher, monkeypatch, tmp_path):
    monkeypatch.setenv("PHARMAGENT_DESKTOP_DATA_DIR", str(tmp_path / "custom"))
    d = launcher.user_data_dir()
    assert d == tmp_path / "custom" and d.is_dir()


def test_user_data_dir_is_per_user_not_next_to_the_bundle(launcher, monkeypatch, tmp_path):
    monkeypatch.delenv("PHARMAGENT_DESKTOP_DATA_DIR", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    d = launcher.user_data_dir()
    assert d.is_relative_to(tmp_path) and d.name == "PharmAgent" and d.is_dir()


def test_free_port_is_bindable_loopback_port(launcher):
    port = launcher.free_port()
    assert 1024 < port < 65536
    with socket.socket() as s:
        s.bind(("127.0.0.1", port))  # still free right after


def test_configure_env_points_backend_at_bundle_and_user_storage(launcher, clean_env, tmp_path):
    dist = _fake_dist(tmp_path, "frontend_dist")
    data = tmp_path / "userdata"
    env = launcher.configure_env(root=tmp_path, data_dir=data)
    assert env["PHARMAGENT_FRONTEND_DIST"] == str(dist)
    assert env["PHARMAGENT_DATA_DIR"] == str(data / "data")
    assert env["PHARMAGENT_DB_PATH"] == str(data / "pharmagent.db")


def test_configure_env_does_not_override_operator_env(launcher, clean_env, tmp_path):
    import os
    _fake_dist(tmp_path, "frontend_dist")
    os.environ["PHARMAGENT_DB_PATH"] = "/elsewhere/x.db"
    env = launcher.configure_env(root=tmp_path, data_dir=tmp_path)
    assert env["PHARMAGENT_DB_PATH"] == "/elsewhere/x.db"


def test_configure_env_fails_clearly_without_a_frontend_build(launcher, clean_env, tmp_path):
    with pytest.raises(RuntimeError, match="frontend build not found"):
        launcher.configure_env(root=tmp_path, data_dir=tmp_path)


def test_wait_for_health_times_out_when_nothing_listens(launcher):
    port = launcher.free_port()
    with pytest.raises(RuntimeError, match="did not come up"):
        launcher.wait_for_health(f"http://127.0.0.1:{port}", timeout_s=0.6)


def test_backend_server_serves_api_and_spa_from_one_origin(launcher, monkeypatch, tmp_path):
    """The real app.main on a loopback port: /api/health answers and / returns
    the built index.html (single origin, no Vite). Uses the repo's real
    frontend/dist when built, else a stub dist, and an isolated data dir."""
    import urllib.request

    import app.main as main_mod
    from app.core.llm import MockLLM
    from app.core.orchestrator import Orchestrator
    from app.core.store import SessionStore

    dist = launcher.frontend_dist(REPO) or _fake_dist(tmp_path, "frontend_dist")
    # app.main was imported by other tests already; make its SPA routes point at
    # this dist by mounting on a fresh orchestrator + the module's _DIST guard.
    if getattr(main_mod, "_DIST", None) is None:
        pytest.skip("app.main imported without a frontend dist; build the frontend to cover the SPA route")
    main_mod.orch = Orchestrator(llm=MockLLM(), store=SessionStore(":memory:"))
    port = launcher.free_port()
    server = launcher.BackendServer(port)
    server.start()
    try:
        with urllib.request.urlopen(f"{server.url}/api/health", timeout=3) as r:
            assert b'"status":"ok"' in r.read().replace(b" ", b"")
        with urllib.request.urlopen(f"{server.url}/", timeout=3) as r:
            body = r.read()
        assert b'<div id="root">' in body
        assert (dist / "index.html").read_bytes()[:64] in body
    finally:
        server.stop()
    assert not server._thread.is_alive()
