"""Resource-limit controls of the page renderer: fail-closed setup, observable enforcement, paired controls.

Each test runs the REAL worker routines in a child process (setting rlimits in the pytest process would cripple it).
Three false-green cases are closed on purpose: limits silently ignored, a worker that always fails, and invalid
output mistaken for enforcement. So every 'over the limit' test has an 'under the limit' twin that must succeed."""
from __future__ import annotations

import sys
import textwrap

import pytest

from app.config import settings
from app.pageimage import limits, runner
from app.pageimage.runner import RenderError, probe_pdf, render_page
from app.pageimage.types import RenderParams
from tests.pdf_fixtures import decode_png, make_pdf, weak_limits_off_linux

_weak = pytest.fixture(autouse=True)(weak_limits_off_linux)
LINUX = sys.platform.startswith("linux")
POSIX = sys.platform != "win32"


@pytest.fixture
def pdf(tmp_path):
    p = tmp_path / "doc.pdf"
    p.write_bytes(make_pdf())
    return p


def _worker(tmp_path, monkeypatch, body: str):
    """A child that loads the REAL worker module as ``mod`` and then runs ``body``."""
    real = str(runner.WORKER_PATH)
    w = tmp_path / "fake_worker.py"
    w.write_text(textwrap.dedent(f"""
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location("real_worker", {real!r})
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    """) + textwrap.dedent(body))
    monkeypatch.setattr(runner, "WORKER_PATH", w)


# ── limit setup is fail-closed and observable ────────────────────────────────
@pytest.mark.skipif(not POSIX, reason="needs the resource module")
def test_a_rejected_setrlimit_is_limit_unavailable_not_failed(tmp_path, monkeypatch, pdf):
    _worker(tmp_path, monkeypatch, """
        import resource
        def refuse(*a): raise OSError("refused")
        resource.setrlimit = refuse
        sys.exit(mod.main(sys.argv))
    """)
    with pytest.raises(RenderError) as e:
        render_page(pdf, 0, RenderParams())
    assert e.value.kind == "limit_unavailable"


@pytest.mark.skipif(not POSIX, reason="needs the resource module")
def test_a_ceiling_that_getrlimit_does_not_confirm_is_limit_unavailable(tmp_path, monkeypatch, pdf):
    _worker(tmp_path, monkeypatch, """
        import resource
        real_get = resource.getrlimit
        resource.getrlimit = lambda r: (resource.RLIM_INFINITY, resource.RLIM_INFINITY)   # never confirms
        sys.exit(mod.main(sys.argv))
    """)
    with pytest.raises(RenderError) as e:
        render_page(pdf, 0, RenderParams())
    assert e.value.kind == "limit_unavailable"


def test_a_worker_that_reports_no_limits_is_not_trusted(tmp_path, monkeypatch, pdf):
    _worker(tmp_path, monkeypatch, """
        sys.stdout.buffer.write(b"\\x89PNG\\r\\n\\x1a\\n")      # valid-looking output, no verified limits
    """)
    with pytest.raises(RenderError) as e:
        render_page(pdf, 0, RenderParams())
    assert e.value.kind == "limit_unavailable"


def test_a_report_that_says_not_enforced_is_refused_unless_weak_mode(tmp_path, monkeypatch, pdf):
    _worker(tmp_path, monkeypatch, """
        import json
        sys.stderr.write(json.dumps({"pageimage_limits": {"ok": True, "enforced": False, "limits": []}}) + "\\n")
        sys.stdout.buffer.write(b"\\x89PNG\\r\\n\\x1a\\n")
    """)
    monkeypatch.setattr(settings, "pageimage_allow_weak_limits", False)
    with pytest.raises(RenderError) as e:
        render_page(pdf, 0, RenderParams())
    assert e.value.kind == "limit_unavailable"
    monkeypatch.setattr(settings, "pageimage_allow_weak_limits", True)
    assert render_page(pdf, 0, RenderParams()).startswith(b"\x89PNG")


@pytest.mark.skipif(LINUX or not POSIX, reason="macOS has no RLIMIT_AS: the real worker must refuse by default")
def test_without_weak_mode_the_real_worker_refuses_where_rlimit_as_is_not_enforced(pdf, monkeypatch):
    monkeypatch.setattr(settings, "pageimage_allow_weak_limits", False)
    with pytest.raises(RenderError) as e:
        render_page(pdf, 0, RenderParams())
    assert e.value.kind == "limit_unavailable"


def test_weak_mode_renders_and_warns_loudly(pdf, monkeypatch, caplog):
    monkeypatch.setattr(settings, "pageimage_allow_weak_limits", True)
    with caplog.at_level("WARNING", logger="pharmagent.pageimage"):
        png = render_page(pdf, 0, RenderParams())
    assert png.startswith(b"\x89PNG")
    assert LINUX or any("weak" in r.getMessage().lower() for r in caplog.records)


# ── paired controls: enforcement is distinguishable from always-failing ──────
def _alloc_in_render(mb: int) -> str:
    """Run the real main(), but allocate ``mb`` MiB inside its guarded section (via plan_output_size)."""
    return f"""
        real_plan = mod.plan_output_size
        def plan(*a, **k):
            blob = bytearray({mb} * 1024 * 1024)     # real allocation under the real rlimits
            blob[-1] = 1
            return real_plan(*a, **k)
        mod.plan_output_size = plan
        sys.exit(mod.main(sys.argv))
    """


def test_under_the_limit_the_real_render_completes_and_emits_a_valid_png(tmp_path, monkeypatch, pdf):
    _worker(tmp_path, monkeypatch, _alloc_in_render(32))
    img = decode_png(render_page(pdf, 0, RenderParams()))
    assert img.width > 0 and img.height > 0


@pytest.mark.skipif(not LINUX, reason="RLIMIT_AS is enforced only on Linux")
def test_over_the_limit_is_memory_limit_not_generic_failed(tmp_path, monkeypatch, pdf):
    monkeypatch.setattr(limits, "ADDRESS_SPACE_BYTES", 256 * 1024 * 1024)
    _worker(tmp_path, monkeypatch, _alloc_in_render(2048))
    with pytest.raises(RenderError) as e:
        render_page(pdf, 0, RenderParams())
    assert e.value.kind == "memory_limit"


@pytest.mark.skipif(not LINUX, reason="RLIMIT_AS is enforced only on Linux")
def test_a_normal_page_renders_with_headroom_under_the_production_limits(pdf):
    import resource
    assert render_page(pdf, 0, RenderParams()).startswith(b"\x89PNG")
    peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024     # KiB on Linux
    print(f"pageimage peak RSS {peak / 1e6:.0f} MB of {limits.ADDRESS_SPACE_BYTES / 1e6:.0f} MB limit")
    assert peak < limits.ADDRESS_SPACE_BYTES / 2, "the address-space ceiling leaves too little headroom"


@pytest.mark.skipif(not POSIX, reason="RLIMIT_CPU / SIGXCPU")
def test_cpu_limit_is_cpu_limit_not_timeout_or_failed(tmp_path, monkeypatch, pdf):
    _worker(tmp_path, monkeypatch, """
        mod.apply_limits(1, 1024 * 1024 * 1024, allow_weak=True)
        while True:
            pass
    """)
    with pytest.raises(RenderError) as e:
        render_page(pdf, 0, RenderParams(), wall_seconds=30)
    assert e.value.kind == "cpu_limit"


def test_wall_clock_timeout_is_still_its_own_kind(tmp_path, monkeypatch, pdf):
    _worker(tmp_path, monkeypatch, """
        import time
        mod.apply_limits(10, 1024 * 1024 * 1024, allow_weak=True)
        time.sleep(60)
    """)
    with pytest.raises(RenderError) as e:
        render_page(pdf, 0, RenderParams(), wall_seconds=1.0)
    assert e.value.kind == "timeout"


def test_probe_uses_the_same_fail_closed_path(tmp_path, monkeypatch, pdf):
    _worker(tmp_path, monkeypatch, """
        sys.stdout.write('{"pages": 1}\\n')            # plausible output, no verified limits
    """)
    with pytest.raises(RenderError) as e:
        probe_pdf(pdf)
    assert e.value.kind == "limit_unavailable"
