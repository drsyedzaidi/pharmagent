"""Parser guards: invisible / white / off-page text never reaches the page text (so it can drive neither
a verdict nor a verified value); huge page boxes, inflate bombs, dense ruling grids and slow documents
fail fast; uploads run off the event loop and are capped per session; keyword screening tokenises
each page once."""
from __future__ import annotations

import asyncio
import itertools
import time

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.config import settings
from app.core.llm import MockLLM
from app.core.orchestrator import Orchestrator
from app.core.store import SessionStore
from app.evidence import ingest, parse_guard, upload_limits
from app.evidence import store as evstore
from app.evidence.parse import parse_pdf
from app.evidence.screen_rules import keyword_screen
from tests import evidence_rawpdf as raw

EV = "/api/sessions/{sid}/evidence"
BOX = [0, 0, 612, 792]
HIDDEN = "We enrolled n = 9999 subjects in a randomized trial."


def _parse(tmp_path, pages, name="x.pdf", **kw):
    path = tmp_path / name
    path.write_bytes(raw.build(pages, **kw))
    return parse_pdf(path)


@pytest.mark.parametrize("prefix", ["3 Tr", "7 Tr", "1 1 1 rg", "1 g", "0 0 0 0 k"])
def test_invisible_or_white_text_is_dropped_and_counted(tmp_path, prefix):
    content = raw.text(raw.BODY) + raw.text([HIDDEN], y0=600, prefix=prefix)
    page = _parse(tmp_path, [(BOX, content)])["pages"][0]
    assert "9999" not in page["text"] and "randomized" not in page["text"]
    assert "rat hepatocytes" in page["text"] and page["hidden_chars"] >= len(HIDDEN.replace(" ", ""))


def test_off_page_text_is_dropped(tmp_path):
    content = raw.text(raw.BODY) + raw.text([HIDDEN], x=900, y0=600)
    page = _parse(tmp_path, [(BOX, content)])["pages"][0]
    assert "9999" not in page["text"] and page["hidden_chars"] > 0


def test_white_text_on_a_dark_box_is_kept(tmp_path):
    box = b"0 0 0.4 rg 40 590 560 30 re f\n"
    content = raw.text(raw.BODY) + box + raw.text(["Header on dark band n = 7"], y0=600, prefix="1 1 1 rg")
    page = _parse(tmp_path, [(BOX, content)])["pages"][0]
    assert "dark band n = 7" in page["text"] and page["hidden_chars"] == 0


def test_hidden_text_cannot_flip_a_keyword_verdict(tmp_path):
    content = raw.text(raw.BODY) + raw.text([HIDDEN], y0=600, prefix="3 Tr")
    pages = _parse(tmp_path, [(BOX, content)])["pages"]
    crit = [{"id": "C1", "text": "randomized trial", "kind": "inclusion", "required": True, "keywords": ["randomized"]}]
    assert keyword_screen(crit, pages)["criteria"][0]["met"] is False


def test_huge_page_box_fails_fast(tmp_path):
    t0 = time.monotonic()
    with pytest.raises(parse_guard.ParseLimitError, match="box"):
        _parse(tmp_path, [([0, 0, 3e7, 792], raw.text(raw.BODY))])
    assert time.monotonic() - t0 < 2.0


def test_inflate_bomb_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(parse_guard, "MAX_STREAM_BYTES", 1 << 20)
    bomb = raw.text(raw.BODY) + b"% " + b"0" * (4 << 20) + b"\n"
    with pytest.raises(parse_guard.ParseLimitError, match="inflates"):
        _parse(tmp_path, [(BOX, bomb)], compress=True)


def test_dense_ruling_grid_skips_table_detection(tmp_path):
    t0 = time.monotonic()
    page = _parse(tmp_path, [(BOX, raw.grid(160) + raw.text(raw.BODY))])["pages"][0]
    assert page["tables"] == [] and time.monotonic() - t0 < 5.0


def test_parse_budget_is_enforced_between_pages(tmp_path, monkeypatch):
    monkeypatch.setattr(parse_guard, "PARSE_BUDGET_S", -1.0)
    with pytest.raises(parse_guard.ParseLimitError, match="exceeded"):
        _parse(tmp_path, [(BOX, raw.text(raw.BODY))] * 2)


def test_parse_limit_becomes_parse_status_failed(tmp_path):
    evstore.configure(":memory:")
    data = raw.build([([0, 0, 3e7, 792], raw.text(raw.BODY))])
    src = ingest.ingest_pdf(data, "wide.pdf", "s1", "alice", data_dir=tmp_path)
    assert src.parse_status == "failed"


def test_keyword_screen_tokenises_each_page_once():
    text = " ".join(f"word{i} plasma concentration" for i in range(400))
    pages = [{"n": i + 1, "text": text, "columns": None} for i in range(100)]
    crit = [{"id": f"C{i}", "text": "x", "kind": "inclusion", "required": False,
             "keywords": [f"absent{i}{j}" for j in range(20)]} for i in range(2)]
    t0 = time.monotonic()
    keyword_screen(crit, pages)
    assert time.monotonic() - t0 < 4.0


# ---------------------------------------------------------------- upload route --

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    evstore.configure(":memory:")
    upload_limits.reset()
    main.orch = Orchestrator(llm=MockLLM(), clock=lambda c=itertools.count(): f"t{next(c)}",
                             store=SessionStore(":memory:"))
    yield TestClient(main.app)
    upload_limits.reset()


def _up(client, sid, data, name):
    return client.post(EV.format(sid=sid) + "/upload", files={"file": (name, data, "application/pdf")})


def _pdf(i: int) -> bytes:
    return raw.build([(BOX, raw.text([f"Paper {i}. Human adult study (n = {i + 3})."]))])


def test_upload_parses_off_the_event_loop(client, monkeypatch):
    seen = []
    real = ingest.validate_pdf_file

    def probe(path, *args, **kw):
        try:
            asyncio.get_running_loop()
            seen.append("event-loop")
        except RuntimeError:
            seen.append("worker")
        return real(path, *args, **kw)

    monkeypatch.setattr(ingest, "validate_pdf_file", probe)
    sid = client.post("/api/sessions").json()["id"]
    assert _up(client, sid, _pdf(1), "a.pdf").status_code == 200
    assert seen == ["worker"]


def test_per_session_source_cap_and_rate_limit(client, monkeypatch):
    sid = client.post("/api/sessions").json()["id"]
    monkeypatch.setattr(upload_limits, "MAX_SOURCES_PER_SESSION", 2)
    assert [_up(client, sid, _pdf(i), f"p{i}.pdf").status_code for i in range(3)] == [200, 200, 413]
    monkeypatch.setattr(upload_limits, "MAX_SOURCES_PER_SESSION", 200)
    monkeypatch.setattr(upload_limits, "UPLOADS_PER_WINDOW", 4)
    codes = [_up(client, sid, _pdf(i), f"q{i}.pdf").status_code for i in range(10, 13)]
    assert codes[-1] == 429 and 200 in codes


def test_per_session_byte_cap(client, monkeypatch):
    sid = client.post("/api/sessions").json()["id"]
    assert _up(client, sid, _pdf(1), "a.pdf").status_code == 200
    monkeypatch.setattr(upload_limits, "MAX_SESSION_BYTES", 600)
    assert _up(client, sid, _pdf(2), "b.pdf").status_code == 413


def test_parse_cache_from_before_the_guards_is_reparsed(tmp_path):
    """A cache written by the earlier parser (no visibility filter, thin spaces lost) is not reused."""
    import json

    import pdfplumber

    data = raw.build([(BOX, raw.text(raw.BODY) + raw.text([HIDDEN], y0=600, prefix="3 Tr"))])
    sha, pdf = ingest.write_by_sha256(data, tmp_path)
    stale = {"sha256": sha, "parser": f"pdfplumber {pdfplumber.__version__}", "decimal_style": ".",
             "text_chars": 1, "metadata": {}, "pages": [{"n": 1, "text": HIDDEN, "small_text": "",
                                                          "columns": None, "tables": []}]}
    (ingest.evidence_dir(tmp_path) / f"{sha}.pages.json").write_text(json.dumps(stale))
    doc, _ = ingest.parse_and_cache(pdf, tmp_path)
    assert "9999" not in doc["pages"][0]["text"] and doc["pages"][0]["hidden_chars"] > 0
