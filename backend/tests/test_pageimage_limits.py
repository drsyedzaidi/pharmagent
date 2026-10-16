"""Resource-limit controls of the page renderer: one versioned attestation protocol, fail-closed setup, inherited
limits kept, paired controls under the SAME ceiling.

Each test that sets rlimits runs the REAL worker routines in a child process (setting them in the pytest process
would cripple it). Three false-green cases are closed on purpose: limits silently ignored, a worker that always
fails, and invalid output mistaken for enforcement. So every 'over the limit' test has an 'under the limit' twin
that must succeed under the same ceiling."""
from __future__ import annotations

import sys
import textwrap

import pytest

from app.config import settings
from app.pageimage import limits, runner
from app.pageimage.runner import RenderError, probe_pdf, render_page
from app.pageimage.types import RenderParams
from tests.pdf_fixtures import (
    REPORT_OK,
    decode_png,
    make_pdf,
    report_snippet,
    unenforced_limits_off_linux,
    valid_report,
)

_weak = pytest.fixture(autouse=True)(unenforced_limits_off_linux)
LINUX = sys.platform.startswith("linux")
POSIX = sys.platform != "win32"
PNG_MAGIC = '"\\x89PNG\\r\\n\\x1a\\n"'


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


def _fake(tmp_path, monkeypatch, source: str):
    w = tmp_path / "fake_worker.py"
    w.write_text(source)
    monkeypatch.setattr(runner, "WORKER_PATH", w)


def _kind(pdf, **kw) -> str:
    with pytest.raises(RenderError) as e:
        render_page(pdf, 0, RenderParams(), **kw)
    return e.value.kind


# ── setup is fail-closed ─────────────────────────────────────────────────────
@pytest.mark.skipif(not POSIX, reason="needs the resource module")
def test_a_rejected_setrlimit_is_limit_unavailable_not_failed(tmp_path, monkeypatch, pdf):
    _worker(tmp_path, monkeypatch, """
        import resource
        def refuse(*a): raise OSError("refused")
        resource.setrlimit = refuse
        sys.exit(mod.main(sys.argv))
    """)
    assert _kind(pdf) == "limit_unavailable"


@pytest.mark.skipif(not POSIX, reason="needs the resource module")
def test_a_ceiling_that_getrlimit_does_not_confirm_is_limit_unavailable(tmp_path, monkeypatch, pdf):
    _worker(tmp_path, monkeypatch, """
        import resource
        resource.getrlimit = lambda r: (resource.RLIM_INFINITY, resource.RLIM_INFINITY)   # never confirms
        sys.exit(mod.main(sys.argv))
    """)
    assert _kind(pdf) == "limit_unavailable"


@pytest.mark.skipif(LINUX or not POSIX, reason="macOS has no RLIMIT_AS: the real worker must refuse by default")
def test_without_the_unsafe_override_the_real_worker_refuses_where_rlimit_as_is_not_enforced(pdf, monkeypatch):
    monkeypatch.setattr(settings, "pageimage_unsafe_allow_unenforced_limits", False)
    assert _kind(pdf) == "limit_unavailable"


def test_the_unsafe_override_renders_and_logs_loudly(pdf, monkeypatch, caplog):
    monkeypatch.setattr(settings, "pageimage_unsafe_allow_unenforced_limits", True)
    with caplog.at_level("WARNING", logger="pharmagent.pageimage"):
        png = render_page(pdf, 0, RenderParams())
    assert png.startswith(b"\x89PNG")
    assert LINUX or any("UNSAFE" in r.getMessage() for r in caplog.records)


def test_the_cpu_slack_is_the_same_in_the_worker_and_the_parent():
    from app.pageimage import worker
    assert worker.CPU_HARD_SLACK == runner.CPU_HARD_SLACK and worker.REPORT_VERSION == runner.REPORT_VERSION


def test_the_unsafe_override_is_ignored_in_a_frozen_build(pdf, monkeypatch):
    monkeypatch.setattr(settings, "pageimage_unsafe_allow_unenforced_limits", True)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert runner._allow_unenforced() is False
    assert _kind(pdf) == "unavailable"


# ── the attestation protocol is exact ────────────────────────────────────────
def _report_with(mutate) -> dict:
    rep = valid_report()
    mutate(rep["pageimage_limits"])
    return rep


BAD_REPORTS = {
    "no report": None,
    "empty limits": _report_with(lambda r: r.update(limits=[])),
    "one limit missing": _report_with(lambda r: r["limits"].pop()),
    "unknown extra limit": _report_with(lambda r: r["limits"].append(
        {"name": "RLIMIT_STACK", "requested": 1, "soft": 1, "hard": 1, "state": "enforced"})),
    "duplicate limit": _report_with(lambda r: r["limits"].append(dict(r["limits"][0]))),
    "wrong version": _report_with(lambda r: r.update(v=99)),
    "ok false": _report_with(lambda r: r.update(ok=False)),
    "wrong requested value": _report_with(lambda r: r["limits"][1].update(requested=1 << 40)),
    "soft above the request": _report_with(lambda r: r["limits"][1].update(soft=1 << 40)),
    "hard above the request": _report_with(lambda r: r["limits"][1].update(hard=1 << 40)),
    "hard missing": _report_with(lambda r: r["limits"][2].pop("hard")),
    "soft is a string": _report_with(lambda r: r["limits"][2].update(soft="0")),
    "unknown state": _report_with(lambda r: r["limits"][0].update(state="sure")),
}


@pytest.mark.parametrize("name", list(BAD_REPORTS))
def test_an_incomplete_or_wrong_report_never_passes(tmp_path, monkeypatch, pdf, name):
    report = BAD_REPORTS[name]
    prefix = report_snippet(report) if report is not None else ""
    _fake(tmp_path, monkeypatch, prefix + f"import sys; sys.stdout.buffer.write(b{PNG_MAGIC})")
    assert _kind(pdf) == "limit_unavailable"


def test_two_reports_are_not_one_report(tmp_path, monkeypatch, pdf):
    _fake(tmp_path, monkeypatch, REPORT_OK + REPORT_OK + f"sys.stdout.buffer.write(b{PNG_MAGIC})")
    assert _kind(pdf) == "limit_unavailable"


def test_a_complete_valid_report_passes(tmp_path, monkeypatch, pdf):
    _fake(tmp_path, monkeypatch, REPORT_OK + f"import sys; sys.stdout.buffer.write(b{PNG_MAGIC})")
    assert render_page(pdf, 0, RenderParams()).startswith(b"\x89PNG")


def test_an_unenforced_state_is_refused_unless_the_unsafe_override(tmp_path, monkeypatch, pdf):
    rep = _report_with(lambda r: r["limits"][1].update(state="unenforced", soft=None, hard=None))
    _fake(tmp_path, monkeypatch, report_snippet(rep) + f"import sys; sys.stdout.buffer.write(b{PNG_MAGIC})")
    monkeypatch.setattr(settings, "pageimage_unsafe_allow_unenforced_limits", False)
    assert _kind(pdf) == "limit_unavailable"
    monkeypatch.setattr(settings, "pageimage_unsafe_allow_unenforced_limits", True)
    assert render_page(pdf, 0, RenderParams()).startswith(b"\x89PNG")


def test_probe_uses_the_same_fail_closed_path(tmp_path, monkeypatch, pdf):
    _fake(tmp_path, monkeypatch, "import sys; sys.stdout.write('{\"pages\": 1}\\n')")     # plausible output, no report
    with pytest.raises(RenderError) as e:
        probe_pdf(pdf)
    assert e.value.kind == "limit_unavailable"


# ── attestation is required for EVERY outcome after limits were applied ──────
AFTER_PARSE = [
    ("import sys; sys.exit(3)", "encrypted"),
    ("import sys; sys.exit(4)", "malformed"),
    ("import sys; sys.exit(5)", "no_page"),
    ("import sys; sys.exit(6)", "too_large"),
    ("import sys; sys.exit(7)", "empty"),
    ("import sys; sys.exit(8)", "output_too_big"),
    ("import sys; sys.exit(10)", "memory_limit"),
    ("import sys; sys.exit(77)", "failed"),
    ("import os, signal; os.kill(os.getpid(), signal.SIGSEGV)", "failed"),
]


@pytest.mark.parametrize("body, kind", AFTER_PARSE)
def test_an_outcome_after_parsing_needs_an_attestation(tmp_path, monkeypatch, pdf, body, kind):
    _fake(tmp_path, monkeypatch, body)
    assert _kind(pdf) == "limit_unavailable"                   # same exit, no valid report: not trusted
    _fake(tmp_path, monkeypatch, REPORT_OK + body)
    assert _kind(pdf) == kind                                  # with the report the real kind comes through


@pytest.mark.skipif(not POSIX, reason="SIGXCPU")
def test_a_cpu_limit_signal_needs_an_attestation(tmp_path, monkeypatch, pdf):
    body = "import os, signal; os.kill(os.getpid(), signal.SIGXCPU)"
    _fake(tmp_path, monkeypatch, body)
    assert _kind(pdf) == "limit_unavailable"
    _fake(tmp_path, monkeypatch, REPORT_OK + body)
    assert _kind(pdf) == "cpu_limit"


def test_a_wall_clock_kill_needs_an_attestation(tmp_path, monkeypatch, pdf):
    _fake(tmp_path, monkeypatch, "import time; time.sleep(60)")
    assert _kind(pdf, wall_seconds=1.0) == "limit_unavailable"
    _fake(tmp_path, monkeypatch, REPORT_OK + "import time; time.sleep(60)")
    assert _kind(pdf, wall_seconds=1.0) == "timeout"


def test_an_output_overflow_needs_an_attestation(tmp_path, monkeypatch, pdf):
    flood = ("import sys\nb = b'x' * 65536\nwhile True:\n    sys.stdout.buffer.write(b); sys.stdout.buffer.flush()\n")
    _fake(tmp_path, monkeypatch, flood)
    assert _kind(pdf, wall_seconds=20) == "limit_unavailable"
    _fake(tmp_path, monkeypatch, REPORT_OK + "\n" + flood)
    assert _kind(pdf, wall_seconds=20) == "output_too_big"


def test_setup_failure_and_pre_parse_bad_request_stay_distinct(tmp_path, monkeypatch, pdf):
    _fake(tmp_path, monkeypatch, "import sys; sys.exit(2)")                         # before any limit: bad_request
    assert _kind(pdf) == "bad_request"
    _fake(tmp_path, monkeypatch, "import sys; sys.exit(9)")                         # setup failure
    assert _kind(pdf) == "limit_unavailable"


# ── inherited limits are kept; hard limits are lowered ───────────────────────
@pytest.mark.skipif(not POSIX, reason="needs the resource module")
def test_inherited_tighter_soft_limits_are_never_raised_and_hard_limits_are_lowered(tmp_path, monkeypatch, pdf):
    # RLIMIT_FSIZE is 0 in the worker, so the result travels on stdout behind the PNG magic, not through a file
    _worker(tmp_path, monkeypatch, f"""
        import resource, json
        resource.setrlimit(resource.RLIMIT_NOFILE, (32, 4096))       # tighter soft than the request (64), loose hard
        rep = mod.apply_limits(10, {limits.ADDRESS_SPACE_BYTES}, True)
        nofile = [r for r in rep if r["name"] == "RLIMIT_NOFILE"][0]
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (4096, 4096)); restored = True
        except (ValueError, OSError):
            restored = False
        {REPORT_OK}
        sys.stdout.buffer.write(b{PNG_MAGIC} + json.dumps([nofile, resource.getrlimit(resource.RLIMIT_NOFILE), restored]).encode())
    """)
    import json
    nofile, now, restored = json.loads(render_page(pdf, 0, RenderParams())[8:])
    assert nofile["soft"] == 32 and nofile["hard"] == 64            # soft kept at 32, hard lowered from 4096 to 64
    assert now == [32, 64] and restored is False                    # worker code cannot restore the ceiling


@pytest.mark.skipif(not POSIX, reason="SIGXCPU")
def test_cpu_limit_is_cpu_limit_and_its_hard_limit_does_not_pre_empt_sigxcpu(tmp_path, monkeypatch, pdf):
    _worker(tmp_path, monkeypatch, f"""
        mod.apply_limits(1, {limits.ADDRESS_SPACE_BYTES}, True)
        {REPORT_OK}
        while True:
            pass
    """)
    assert _kind(pdf, wall_seconds=30) == "cpu_limit"                # SIGXCPU at the soft limit, SIGKILL only later


# ── paired controls under the SAME ceiling ───────────────────────────────────
def _alloc_in_render(mb: int) -> str:
    """Run the real main(), allocating ``mb`` MiB inside its guarded section (via plan_output_size)."""
    return f"""
        real_plan = mod.plan_output_size
        def plan(*a, **k):
            blob = bytearray({mb} * 1024 * 1024)     # real allocation under the real rlimits
            blob[-1] = 1
            return real_plan(*a, **k)
        mod.plan_output_size = plan
        sys.exit(mod.main(sys.argv))
    """


CEILING = limits.ADDRESS_SPACE_BYTES


def test_under_the_ceiling_the_real_render_completes_and_emits_a_valid_png(tmp_path, monkeypatch, pdf):
    _worker(tmp_path, monkeypatch, _alloc_in_render(64))
    img = decode_png(render_page(pdf, 0, RenderParams()))
    assert img.width > 0 and img.height > 0


@pytest.mark.skipif(not LINUX, reason="RLIMIT_AS is enforced only on Linux")
def test_over_the_same_ceiling_is_memory_limit_not_generic_failed(tmp_path, monkeypatch, pdf):
    _worker(tmp_path, monkeypatch, _alloc_in_render(CEILING // (1024 * 1024) + 256))
    assert _kind(pdf) == "memory_limit"


@pytest.mark.skipif(not LINUX, reason="RLIMIT_AS is enforced only on Linux")
def test_the_unmodified_real_render_succeeds_under_that_same_ceiling(pdf):
    assert render_page(pdf, 0, RenderParams()).startswith(b"\x89PNG")
