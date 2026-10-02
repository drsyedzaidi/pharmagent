"""Layout rebuild of sources parsed before evidence-parse 7: the sidecar is derived from a re-parse of the
stored PDF, written atomically, and ONLY when every page text equals the stored parse. Pages file, ledger
and records are never touched; a source that cannot be rebuilt says why (text_changed / unreadable)."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

import app.main as main
from app.config import settings
from app.core.audit import hash_payload
from app.evidence import ingest, relayout
from app.evidence import store as evstore
from tests import evidence_fixtures as fx
from tests.evidence_image_support import (
    EV,
    build_papers,
    make_client,
    meta,
    record_of,
    records,
    session_with,
    source_id,
)

OLD_PARSER = "pdfplumber 0.11.10 / evidence-parse 6"


@pytest.fixture(scope="module")
def papers(tmp_path_factory):
    return build_papers(tmp_path_factory)


@pytest.fixture
def client(tmp_path, monkeypatch):
    yield from make_client(tmp_path, monkeypatch)


def _src(client, sid, name):
    return evstore.get_store().get_source(source_id(client, sid, name))


def _pages_file(src) -> Path:
    return Path(src.parse_path)


def make_old(src, *, edit=None) -> None:
    """Simulate a source parsed before layout sidecars: no sidecar, cache stamped with the previous parser
    (optionally with *edit(doc)* applied to the cached parse first)."""
    relayout.layout_file(src).unlink(missing_ok=True)
    relayout.marker_file(src).unlink(missing_ok=True)
    doc = json.loads(_pages_file(src).read_text("utf-8"))
    doc["parser"] = OLD_PARSER
    if edit:
        edit(doc)
    _pages_file(src).write_text(json.dumps(doc, ensure_ascii=False), "utf-8")


def post(client, sid, src_id=None):
    url = EV.format(sid=sid) + (f"/sources/{src_id}/relayout" if src_id else "/relayout")
    return client.post(url)


def wait_job(client, sid, job_id, timeout=240):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        job = client.get(f"/api/sessions/{sid}/jobs/{job_id}").json()
        if job["status"] != "running":
            return job
        time.sleep(0.2)
    raise AssertionError("job did not finish")


def status_of(client, sid, name):
    rows = client.get(EV.format(sid=sid)).json()["sources"]
    return next(s for s in rows if s["filename"] == f"{name}.pdf")


def audit_entries(client, sid):
    return [e for e in client.get(f"/api/sessions/{sid}/audit").json()["entries"]
            if e["tool"] == "rebuild_source_layout"]


def _ledger(sid):
    store = evstore.get_store()
    return ([r.__dict__ for r in store.list_records(sid)], [s.__dict__ for s in store.list_sources(sid)],
            store.counts(sid))


# ---------------------------------------------------------------- rebuild ---

def test_rebuild_writes_only_the_sidecar_and_the_image_resolves(client, papers):
    sid = session_with(client, {"paper_a": papers["paper_a"].read_bytes()})
    rec = record_of(client, sid, "paper_a", "cmax", "3,412")
    src = _src(client, sid, "paper_a")
    assert client.post(EV.format(sid=sid) + "/records/decide", json={"decisions": [
        {"record_id": rec["id"], "decision": "approve", "reason": "checked"}]}).status_code == 200
    make_old(src)
    pages_before, ledger_before = _pages_file(src).read_bytes(), _ledger(sid)

    m = meta(client, sid, rec["id"]).json()
    assert (m["reason"], m["layout_status"], m["layout_rebuild_available"]) == ("layout_unavailable", "missing", True)
    assert status_of(client, sid, "paper_a")["layout_status"] == "missing"

    r = post(client, sid, src.id)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["outcome"] == "rebuilt" and body["layout_status"] == "ready" and body["reason"] is None
    side = relayout.layout_file(src)
    assert side.is_file() and body["sidecar_sha256"] == ingest.file_sha256(side)
    assert json.loads(side.read_text("utf-8"))["rebuilt_for"] == OLD_PARSER
    assert _pages_file(src).read_bytes() == pages_before          # parse cache byte-identical
    assert _ledger(sid) == ledger_before                          # records (approved one too), sources, counts
    assert not list(side.parent.glob(".*.tmp"))                   # no temp files left behind

    m = meta(client, sid, rec["id"]).json()
    assert m["resolved"] and m["precision"] in ("value", "cell") and m["reason"] is None
    assert m["layout_status"] == "ready" and m["layout_rebuild_available"] is False
    assert status_of(client, sid, "paper_a")["layout_status"] == "ready"
    assert records(client, sid)[0]["status"] in ("approved", "proposed")


def test_second_call_is_a_no_op_and_does_not_parse(client, papers, monkeypatch):
    sid = session_with(client, {"paper_a": papers["paper_a"].read_bytes()}, extract=False)
    src = _src(client, sid, "paper_a")
    make_old(src)
    assert post(client, sid, src.id).json()["outcome"] == "rebuilt"
    side = relayout.layout_file(src)
    before = (side.read_bytes(), side.stat().st_mtime_ns, _pages_file(src).read_bytes())
    monkeypatch.setattr(relayout.parse_sandbox, "parse_isolated",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("parsed again")))
    again = post(client, sid, src.id)
    assert again.status_code == 200 and again.json()["outcome"] == "unchanged"
    assert (side.read_bytes(), side.stat().st_mtime_ns, _pages_file(src).read_bytes()) == before


def test_changed_page_text_writes_nothing_and_marks_text_changed(client, papers):
    sid = session_with(client, {"paper_a": papers["paper_a"].read_bytes()})
    rec = record_of(client, sid, "paper_a", "cmax", "3,412")
    src = _src(client, sid, "paper_a")

    def doctor(doc):
        doc["pages"][0]["text"] = doc["pages"][0]["text"].replace("3,412", "3,421")
    make_old(src, edit=doctor)
    pages_before, ledger_before = _pages_file(src).read_bytes(), _ledger(sid)

    body = post(client, sid, src.id).json()
    assert body["outcome"] == "text_changed" and body["layout_status"] == "text_changed"
    assert "page 1 text" in body["reason"] and "new session" in body["reason"]
    assert body["sidecar_sha256"] is None
    assert not relayout.layout_file(src).exists()
    assert _pages_file(src).read_bytes() == pages_before and _ledger(sid) == ledger_before
    row = status_of(client, sid, "paper_a")
    assert row["layout_status"] == "text_changed" and "page 1 text" in row["layout_reason"]
    m = meta(client, sid, rec["id"]).json()
    assert m["resolved"] is False and m["layout_rebuild_available"] is False
    assert m["compare_urgent"] is True and m["layout_status"] == "text_changed"


@pytest.mark.parametrize("make_blob", [
    lambda d: b"%PDF-1.4\n1 0 obj <<\n garbage that never closes",
    lambda d: fx.encrypted_pdf(d).read_bytes()], ids=["corrupt", "encrypted"])
def test_unparseable_pdf_is_unreadable(client, papers, tmp_path, make_blob):
    sid = client.post("/api/sessions").json()["id"]
    sha, pdf = ingest.write_by_sha256(make_blob(tmp_path), settings.data_dir)
    parse_path = ingest.evidence_dir(settings.data_dir) / f"{sha}.pages.json"
    page = {"n": 1, "text": "x", "small_text": "", "columns": None, "tables": [], "hidden_chars": 0,
            "ink_check": "ok", "glyph_check": {}}
    parse_path.write_text(json.dumps({"sha256": sha, "parser": OLD_PARSER, "decimal_style": ".", "text_chars": 1,
                                      "metadata": {}, "pages": [page]}), "utf-8")
    src = evstore.get_store().add_source(session_id=sid, sha256=sha, filename="bad.pdf", pdf_path=str(pdf),
                                         parse_path=str(parse_path), pages=1, title=None, metadata={},
                                         text_chars=1, parse_status="parsed", uploaded_by="t")
    assert relayout.layout_info(src)["layout_status"] == "missing"
    before = parse_path.read_bytes()
    body = post(client, sid, src.id).json()
    assert body["outcome"] == "unreadable" and body["layout_status"] == "unreadable"
    assert "could not be parsed again" in body["reason"]
    assert not relayout.layout_file(src).exists() and parse_path.read_bytes() == before
    assert status_of(client, sid, "bad")["layout_status"] == "unreadable"
    assert post(client, sid).json()["selected"] == 0              # bulk skips what cannot be rebuilt


def test_source_of_another_session_is_404(client, papers):
    s1 = session_with(client, {"paper_a": papers["paper_a"].read_bytes()}, extract=False)
    s2 = session_with(client, {"paper_b": papers["paper_b"].read_bytes()}, extract=False)
    foreign = _src(client, s1, "paper_a")
    make_old(foreign)
    assert post(client, s2, foreign.id).status_code == 404
    assert post(client, s2, "nope").status_code == 404
    assert not relayout.layout_file(foreign).exists() and audit_entries(client, s2) == []
    with pytest.raises(ValueError, match="unknown evidence source"):      # the tool itself, e.g. from chat
        main.orch.run_tool(s2, "rebuild_source_layout", "evidence", {"source_id": foreign.id}, actor="t")
    assert not relayout.layout_file(foreign).exists()


def test_rebuild_is_audited_with_source_sidecar_and_page_hashes(client, papers):
    sid = session_with(client, {"paper_a": papers["paper_a"].read_bytes()}, extract=False)
    src = _src(client, sid, "paper_a")
    make_old(src)
    first_body = post(client, sid, src.id).json()
    post(client, sid, src.id)                                      # the no-op is audited too
    entries = audit_entries(client, sid)
    assert len(entries) == 2
    first, second = entries
    sidecar_sha = ingest.file_sha256(relayout.layout_file(src))
    hashes = [relayout.sha256_text(p["text"]) for p in json.loads(_pages_file(src).read_text("utf-8"))["pages"]]
    for token in (src.id, "outcome=rebuilt", f"sidecar_sha256={sidecar_sha}", f"pages={len(hashes)}",
                  f"page_text_sha256={relayout.sha256_text(chr(10).join(hashes))}"):
        assert token in first["action"], token
    assert "outcome=unchanged" in second["action"]
    assert first["actor"] and first["agent"] == "evidence"
    result = {k: v for k, v in first_body.items() if k not in ("source", "summary")}
    assert result["page_text_sha256"] == hashes
    assert first["outputs_hash"] == hash_payload(result)           # the full per-page list is bound


# ------------------------------------------------------------------- bulk ---

def test_bulk_respects_its_cap_and_picks_up_the_rest_next_call(client, papers, monkeypatch):
    monkeypatch.setattr(relayout, "MAX_BULK", 2)
    sid = session_with(client, {n: papers[n].read_bytes() for n in ("paper_a", "paper_b", "paper_c")}, extract=False)
    srcs = [_src(client, sid, n) for n in ("paper_a", "paper_b", "paper_c")]
    for s in srcs:
        make_old(s)
    first = post(client, sid)
    assert first.status_code == 200
    body = first.json()
    assert (body["selected"], body["remaining"], body["cap"]) == (2, 1, 2) and body["job_id"]
    job = wait_job(client, sid, body["job_id"])
    assert job["status"] == "done", job
    assert job["result"]["counts"] == {"rebuilt": 2}
    assert sorted(relayout.layout_info(s)["layout_status"] for s in srcs) == ["missing", "ready", "ready"]
    second = post(client, sid).json()
    assert (second["selected"], second["remaining"]) == (1, 0)
    assert wait_job(client, sid, second["job_id"])["result"]["counts"] == {"rebuilt": 1}
    third = post(client, sid).json()
    assert third["selected"] == 0 and third["job_id"] is None
    assert len(audit_entries(client, sid)) == 3                   # one audited call per rebuilt source


def test_bulk_of_another_session_does_not_touch_foreign_sources(client, papers):
    s1 = session_with(client, {"paper_a": papers["paper_a"].read_bytes()}, extract=False)
    s2 = client.post("/api/sessions").json()["id"]
    foreign = _src(client, s1, "paper_a")
    make_old(foreign)
    assert post(client, s2).json()["selected"] == 0
    assert not relayout.layout_file(foreign).exists()


# ----------------------------------------------------------- atomic write ---

KILL_SCRIPT = """
import os, signal, sys
from pathlib import Path
from app.evidence import relayout
os.replace = lambda *a, **k: os.kill(os.getpid(), signal.SIGKILL)     # killed after the temp is written
relayout.write_atomic(Path(sys.argv[1]), b'{"complete": true}' * 1000)
"""


def test_sigkill_during_the_write_never_leaves_a_truncated_sidecar(client, papers, tmp_path):
    sid = session_with(client, {"paper_a": papers["paper_a"].read_bytes()}, extract=False)
    src = _src(client, sid, "paper_a")
    make_old(src)
    side = relayout.layout_file(src)
    backend = Path(__file__).resolve().parents[1]
    proc = subprocess.run([sys.executable, "-c", KILL_SCRIPT, str(side)], cwd=backend, timeout=60)
    assert proc.returncode != 0                                    # SIGKILL
    assert not side.exists()                                       # the final name was never half-written
    assert relayout.layout_info(src)["layout_status"] == "missing"
    assert post(client, sid, src.id).json()["outcome"] == "rebuilt"   # a stray temp file is harmless
    assert relayout.layout_info(src)["layout_status"] == "ready"


def test_a_truncated_sidecar_is_not_ready_and_is_replaced(client, papers):
    sid = session_with(client, {"paper_a": papers["paper_a"].read_bytes()}, extract=False)
    src = _src(client, sid, "paper_a")
    side = relayout.layout_file(src)
    side.write_bytes(side.read_bytes()[:200])                      # a legacy partial write
    assert relayout.layout_info(src)["layout_status"] == "missing"
    assert post(client, sid, src.id).json()["outcome"] == "rebuilt"
    assert json.loads(side.read_text("utf-8"))["sha256"] == src.sha256


def test_failed_rename_leaves_no_file_and_no_temp(tmp_path, monkeypatch):
    target = tmp_path / "x.layout.json"
    monkeypatch.setattr(relayout.os, "replace", lambda *a: (_ for _ in ()).throw(OSError("disk")))
    with pytest.raises(OSError):
        relayout.write_atomic(target, b"data")
    assert list(tmp_path.iterdir()) == []
