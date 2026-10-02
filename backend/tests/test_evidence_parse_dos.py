"""Parse DoS: glyph floods are refused from a pre-layout count of the decompressed content streams
(per page and per document, Form XObject draws multiplied in), and every ingest parse runs in a child
process with a hard wall-clock kill, an RSS watchdog and a process-wide concurrency cap."""
from __future__ import annotations

import itertools
import resource
import sys
import threading
import time

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.config import settings
from app.core.llm import MockLLM
from app.core.orchestrator import Orchestrator
from app.core.store import SessionStore
from app.evidence import ingest, parse_budget, parse_sandbox, upload_limits
from app.evidence import store as evstore
from app.evidence.models import IngestError
from app.evidence.parse import parse_pdf
from app.evidence.parse_guard import ParseLimitError
from tests import evidence_rawpdf as raw

BOX = [0, 0, 612, 792]
DIGITS = "1234567890" * 10


def flood(n_glyphs: int) -> bytes:
    """The review PoC: one page, n glyphs in 100-glyph Tj lines, Flate-compressed to a few KB."""
    body = b"".join(f"1 0 0 1 10 {10 + (i % 700)} Tm ({DIGITS}) Tj\n".encode() for i in range(n_glyphs // 100))
    return raw.build([(BOX, b"BT /F1 1 Tf\n" + body + b"ET\n")], compress=True)


def _write(tmp_path, data: bytes, name: str = "x.pdf"):
    path = tmp_path / name
    path.write_bytes(data)
    return path


def test_glyph_flood_is_refused_before_layout(tmp_path):
    path = _write(tmp_path, flood(600_000))
    assert path.stat().st_size < 20_000
    t0 = time.monotonic()
    with pytest.raises(ParseLimitError, match="glyph"):
        parse_pdf(path)
    assert time.monotonic() - t0 < 5.0


def test_per_document_glyph_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(parse_budget, "MAX_GLYPHS_PER_DOC", 2_500)
    page = raw.text([DIGITS] * 10)                    # 1 000 glyphs + 10 show operators
    with pytest.raises(ParseLimitError, match="document"):
        parse_pdf(_write(tmp_path, raw.build([(BOX, page)] * 3)))
    assert parse_pdf(_write(tmp_path, raw.build([(BOX, page)] * 2), "ok.pdf"))["text_chars"] > 0


def test_form_xobject_draws_are_multiplied_in(tmp_path):
    form = raw.text([DIGITS] * 10)                    # ~1 000 glyphs, drawn 300 times
    page = b"q /Fm1 Do Q\n" * 300
    with pytest.raises(ParseLimitError, match="glyph"):
        parse_pdf(_write(tmp_path, raw.build([(BOX, page, {"XObject": {"Fm1": form}})], compress=True)))


def test_empty_show_operators_count():
    assert parse_budget.content_glyphs(b"BT () Tj ET\n" * 1000) >= 1000


def test_nested_strings_escapes_and_hex_are_counted():
    data = b"BT (a(b)c\\)d) Tj <41424344> Tj [(xy) -20 (z)] TJ ET"
    assert parse_budget.content_glyphs(data) >= 6 + 2 + 3 + 3


def test_stray_close_paren_in_a_comment_does_not_hide_later_strings():
    data = b"% ))))))))\nBT (" + b"9" * 5000 + b") Tj ET"
    assert parse_budget.content_glyphs(data) >= 5000


def test_normal_fixture_counts_are_far_below_the_caps(tmp_path):
    from tests import evidence_fixtures as fx
    for name, path in fx.build_all(tmp_path).items():
        assert parse_pdf(path)["text_chars"] > 0, name


# ------------------------------------------------------------------- sandbox --

def _peak_child_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return peak / 2**20 if sys.platform == "darwin" else peak / 2**10


def test_ingest_parse_of_the_flood_is_refused_fast_and_small(tmp_path):
    evstore.configure(":memory:")
    t0 = time.monotonic()
    src = ingest.ingest_pdf(flood(600_000), "flood.pdf", "s1", "alice", data_dir=tmp_path)
    elapsed = time.monotonic() - t0
    assert src.parse_status == "failed" and elapsed < 5.0
    assert _peak_child_rss_mb() < 500
    cached = (tmp_path / "evidence" / f"{src.sha256}.pages.json").read_text()
    assert "glyph" in cached


def test_parse_runs_in_a_child_process(tmp_path):
    import os
    path = _write(tmp_path, raw.build([(BOX, raw.text(raw.BODY))]))
    out = parse_sandbox.parse_isolated(path, target="tests.evidence_sandbox_targets:report_pid")
    assert out["pid"] != os.getpid()
    assert "rat hepatocytes" in parse_sandbox.parse_isolated(path)["pages"][0]["text"]


def test_wall_clock_kill(tmp_path):
    t0 = time.monotonic()
    with pytest.raises(parse_sandbox.ParseAbortedError, match="time"):
        parse_sandbox.parse_isolated(_write(tmp_path, b"%PDF-1.4"), wall_s=1.0,
                                     target="tests.evidence_sandbox_targets:sleep_forever")
    assert time.monotonic() - t0 < 6.0


def test_memory_watchdog_kill(tmp_path):
    t0 = time.monotonic()
    with pytest.raises(parse_sandbox.ParseAbortedError, match="memory"):
        parse_sandbox.parse_isolated(_write(tmp_path, b"%PDF-1.4"), max_rss=300 * 2**20, wall_s=30.0,
                                     target="tests.evidence_sandbox_targets:hog_memory")
    assert time.monotonic() - t0 < 20.0


def test_child_exception_is_reported_with_its_type(tmp_path):
    with pytest.raises(ParseLimitError, match="glyph"):
        parse_sandbox.parse_isolated(_write(tmp_path, flood(600_000)))


def test_abort_is_an_ingest_error_and_is_recorded_as_failed(tmp_path, monkeypatch):
    evstore.configure(":memory:")
    assert issubclass(parse_sandbox.ParseAbortedError, IngestError)

    def aborted(_path):
        raise parse_sandbox.ParseAbortedError("parse killed: exceeded 1 s wall clock")

    monkeypatch.setattr(parse_sandbox, "parse_isolated", aborted)
    src = ingest.ingest_pdf(raw.build([(BOX, raw.text(raw.BODY))]), "a.pdf", "s1", "alice", data_dir=tmp_path)
    assert src.parse_status == "failed"


def test_concurrency_cap_is_process_wide(tmp_path, monkeypatch):
    monkeypatch.setattr(parse_sandbox, "_slots", threading.BoundedSemaphore(1))
    monkeypatch.setattr(parse_sandbox, "QUEUE_WAIT_S", 0.2)
    path = _write(tmp_path, raw.build([(BOX, raw.text(raw.BODY))]))
    assert parse_sandbox._slots.acquire(timeout=1)    # another session's parse holds the only slot
    try:
        with pytest.raises(parse_sandbox.ParseBusyError):
            parse_sandbox.parse_isolated(path)
    finally:
        parse_sandbox._slots.release()
    assert parse_sandbox.parse_isolated(path)["text_chars"] > 0


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    evstore.configure(":memory:")
    upload_limits.reset()
    main.orch = Orchestrator(llm=MockLLM(), clock=lambda c=itertools.count(): f"t{next(c)}",
                             store=SessionStore(":memory:"))
    yield TestClient(main.app)
    upload_limits.reset()


def test_busy_parser_is_503_and_records_nothing(client, monkeypatch):
    def busy(_path):
        raise parse_sandbox.ParseBusyError("parser busy")

    monkeypatch.setattr(parse_sandbox, "parse_isolated", busy)
    sid = client.post("/api/sessions").json()["id"]
    pdf = raw.build([(BOX, raw.text(raw.BODY))])
    res = client.post(f"/api/sessions/{sid}/evidence/upload", files={"file": ("a.pdf", pdf, "application/pdf")})
    assert res.status_code == 503
    assert evstore.get_store().list_sources(sid) == []
