"""Sandboxed renderer: types, worker/runner behaviour, caps, kill paths."""
from __future__ import annotations

import os
import sys
import textwrap
import time

import pytest
from pydantic import ValidationError

from app.pageimage import limits, runner
from app.pageimage.runner import RenderError, probe_pdf, render_page
from app.pageimage.types import NormBBox, PageLocator, RenderParams
from tests.pdf_fixtures import REPORT_OK, decode_png, make_pdf, make_zero_page_pdf, unenforced_limits_off_linux

_weak = pytest.fixture(autouse=True)(unenforced_limits_off_linux)   # macOS: the explicit UNSAFE operator override

@pytest.fixture
def pdf(tmp_path):
    def _write(data: bytes, name: str = "doc.pdf"):
        p = tmp_path / name
        p.write_bytes(data)
        return p
    return _write


# ── types ────────────────────────────────────────────────────────────────────
def test_bbox_valid_and_frozen():
    b = NormBBox(x0=0.1, y0=0.2, x1=0.5, y1=0.9)
    assert (b.x0, b.y1) == (0.1, 0.9)
    with pytest.raises(ValidationError):
        b.x0 = 0.3  # frozen


@pytest.mark.parametrize("kw", [
    dict(x0=0.5, y0=0.1, x1=0.5, y1=0.9),        # zero width
    dict(x0=0.6, y0=0.1, x1=0.5, y1=0.9),        # reversed
    dict(x0=-0.1, y0=0.1, x1=0.5, y1=0.9),       # out of range
    dict(x0=0.1, y0=0.1, x1=1.5, y1=0.9),
    dict(x0=float("nan"), y0=0.1, x1=0.5, y1=0.9),
    dict(x0=0.1, y0=0.1, x1=float("inf"), y1=0.9),
])
def test_bbox_rejects_bad_values(kw):
    with pytest.raises(ValidationError):
        NormBBox(**kw)


def test_locator_validates_id_and_page():
    PageLocator(document_id="a" * 32, page=0)
    for bad in ({"document_id": "../etc/passwd", "page": 0}, {"document_id": "A" * 32, "page": 0},
                {"document_id": "a" * 32, "page": -1}, {"document_id": "a" * 32, "page": limits.MAX_PAGES}):
        with pytest.raises(ValidationError):
            PageLocator(**bad)


def test_render_params_scale_range():
    assert RenderParams().scale == limits.DEFAULT_SCALE
    for bad in (0.49, 3.01, float("nan"), float("inf")):
        with pytest.raises(ValidationError):
            RenderParams(scale=bad)


# ── happy path ───────────────────────────────────────────────────────────────
def test_probe_counts_pages(pdf):
    assert probe_pdf(pdf(make_pdf(pages=3))) == 3


def test_render_full_page_dimensions_and_content(pdf):
    # 200x100pt page, black 50x50 square at the bottom-left corner.
    p = pdf(make_pdf(width=200, height=100, rect=(0, 0, 50, 50)))
    img = decode_png(render_page(p, 0, RenderParams(scale=1.0)))
    assert (img.width, img.height) == (200, 100)
    assert img.pixel(10, 90) == (0, 0, 0)          # inside the square (bottom-left)
    assert img.pixel(150, 10) == (255, 255, 255)   # far top-right is white


def test_render_scale_changes_size(pdf):
    p = pdf(make_pdf(width=200, height=100))
    img = decode_png(render_page(p, 0, RenderParams(scale=2.0)))
    assert (img.width, img.height) == (400, 200)


def test_bbox_crop_returns_region_only(pdf):
    p = pdf(make_pdf(width=200, height=100, rect=(0, 0, 50, 50)))
    # Crop the bottom-left quarter: x 0..0.25 (=50pt), y 0.5..1 (=50pt, top-left origin).
    bb = NormBBox(x0=0.0, y0=0.5, x1=0.25, y1=1.0)
    img = decode_png(render_page(p, 0, RenderParams(scale=1.0, bbox=bb)))
    assert abs(img.width - 50) <= 1 and abs(img.height - 50) <= 1
    assert img.pixel(img.width // 2, img.height // 2) == (0, 0, 0)  # entirely the black square
    # Opposite quarter is white.
    bb2 = NormBBox(x0=0.75, y0=0.0, x1=1.0, y1=0.5)
    img2 = decode_png(render_page(p, 0, RenderParams(scale=1.0, bbox=bb2)))
    assert img2.pixel(img2.width // 2, img2.height // 2) == (255, 255, 255)


# ── refusals (exit-code mapping) ─────────────────────────────────────────────
def test_page_out_of_range(pdf):
    p = pdf(make_pdf(pages=2))
    with pytest.raises(RenderError) as e:
        render_page(p, 2, RenderParams())
    assert e.value.kind == "no_page"


def test_oversized_dimensions_refused(pdf):
    p = pdf(make_pdf(width=14400, height=14400))
    with pytest.raises(RenderError) as e:
        render_page(p, 0, RenderParams(scale=3.0))
    assert e.value.kind == "too_large"
    # ...but a small crop of the same giant page is fine and cheap.
    bb = NormBBox(x0=0.0, y0=0.0, x1=0.05, y1=0.05)
    img = decode_png(render_page(p, 0, RenderParams(scale=1.0, bbox=bb)))
    assert img.width <= limits.MAX_DIM_PX and img.height <= limits.MAX_DIM_PX


def test_pixel_area_cap_refused(pdf):
    # 3000x3000 px each side < 4096 but 9M px; push area over 16M with 4000x4000.
    p = pdf(make_pdf(width=4000, height=4000))
    with pytest.raises(RenderError) as e:
        render_page(p, 0, RenderParams(scale=1.0))
    assert e.value.kind == "too_large"


def test_tiny_crop_is_empty_or_minimal(pdf):
    p = pdf(make_pdf(width=200, height=100))
    bb = NormBBox(x0=0.5, y0=0.5, x1=0.500001, y1=0.500001)
    try:
        img = decode_png(render_page(p, 0, RenderParams(scale=0.5, bbox=bb)))
        assert img.width >= 1 and img.height >= 1
    except RenderError as e:
        assert e.kind in {"empty", "failed", "malformed"}


@pytest.mark.parametrize("data,kind", [
    (b"this is not a pdf at all", "malformed"),
    (b"%PDF-1.4\n" + b"\x00garbage" * 50, "malformed"),
    (make_pdf()[:120], "malformed"),                       # truncated
    (make_pdf(encrypt_user_password="secret"), "encrypted"),
])
def test_bad_pdfs_are_classified(pdf, data, kind):
    p = pdf(data)
    with pytest.raises(RenderError) as e:
        probe_pdf(p)
    assert e.value.kind == kind


def test_zero_page_pdf(pdf):
    p = pdf(make_zero_page_pdf())
    try:
        assert probe_pdf(p) == 0
    except RenderError as e:  # PDFium may refuse to load it at all
        assert e.kind == "malformed"


def test_missing_file_is_malformed(tmp_path):
    with pytest.raises(RenderError) as e:
        probe_pdf(tmp_path / "nope.pdf")
    assert e.value.kind == "malformed"


# ── sandbox behaviour (fake workers) ─────────────────────────────────────────
def _fake_worker(tmp_path, monkeypatch, body: str):
    w = tmp_path / "fake_worker.py"
    w.write_text(REPORT_OK + "\n" + textwrap.dedent(body))     # a trustworthy worker attests its limits first
    monkeypatch.setattr(runner, "WORKER_PATH", w)


def test_wall_clock_timeout_kills_worker(tmp_path, monkeypatch, pdf):
    pidfile = tmp_path / "pid.txt"
    _fake_worker(tmp_path, monkeypatch, f"""
        import os, time
        open({str(pidfile)!r}, "w").write(str(os.getpid()))
        time.sleep(60)
    """)
    p = pdf(make_pdf())
    t0 = time.monotonic()
    with pytest.raises(RenderError) as e:
        render_page(p, 0, RenderParams(), wall_seconds=1.0)
    assert e.value.kind == "timeout"
    assert time.monotonic() - t0 < 10
    pid = int(pidfile.read_text())
    time.sleep(0.2)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)  # the worker is really gone


def test_output_byte_cap_kills_worker(tmp_path, monkeypatch, pdf):
    _fake_worker(tmp_path, monkeypatch, """
        import sys
        blob = b"x" * 65536
        while True:
            sys.stdout.buffer.write(blob); sys.stdout.buffer.flush()
    """)
    with pytest.raises(RenderError) as e:
        render_page(pdf(make_pdf()), 0, RenderParams(), wall_seconds=20)
    assert e.value.kind == "output_too_big"


def test_non_png_output_rejected(tmp_path, monkeypatch, pdf):
    _fake_worker(tmp_path, monkeypatch, "import sys; sys.stdout.write('<html>')")
    with pytest.raises(RenderError) as e:
        render_page(pdf(make_pdf()), 0, RenderParams())
    assert e.value.kind == "failed"


def test_worker_crash_is_failed_not_leaky(tmp_path, monkeypatch, pdf):
    _fake_worker(tmp_path, monkeypatch, "import os, signal; os.kill(os.getpid(), signal.SIGSEGV)")
    with pytest.raises(RenderError) as e:
        render_page(pdf(make_pdf()), 0, RenderParams())
    assert e.value.kind == "failed"
    assert str(tmp_path) not in str(e.value)


def test_worker_sees_clean_env_and_empty_cwd(tmp_path, monkeypatch, pdf):
    out = tmp_path / "probe.txt"
    monkeypatch.setenv("PHARMAGENT_API_TOKEN", "super-secret")
    _fake_worker(tmp_path, monkeypatch, f"""
        import os
        open({str(out)!r}, "w").write(repr((sorted(os.environ), os.listdir("."))))
        import sys; sys.stdout.buffer.write(b"\\x89PNG\\r\\n\\x1a\\n")
    """)
    render_page(pdf(make_pdf()), 0, RenderParams())
    env_keys, cwd_files = eval(out.read_text())  # noqa: S307 - test-local repr of lists
    assert "PHARMAGENT_API_TOKEN" not in env_keys
    assert cwd_files == []


def test_slot_exhaustion_returns_busy(monkeypatch, pdf):
    monkeypatch.setattr(limits, "QUEUE_WAIT_SECONDS", 0.05)
    acquired = []
    try:
        for _ in range(limits.MAX_CONCURRENT_RENDERS):
            assert runner._slots.acquire(timeout=1)
            acquired.append(1)
        with pytest.raises(RenderError) as e:
            render_page(pdf(make_pdf()), 0, RenderParams())
        assert e.value.kind == "busy"
    finally:
        for _ in acquired:
            runner._slots.release()


def test_frozen_build_reports_unavailable(monkeypatch, pdf):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    with pytest.raises(RenderError) as e:
        render_page(pdf(make_pdf()), 0, RenderParams())
    assert e.value.kind == "unavailable"
