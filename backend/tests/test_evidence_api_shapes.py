"""JSON shapes the frontend (S8/S9 types.ts) reads from the evidence endpoints.

Key SETS are pinned, not just presence, so a renamed or dropped field fails here before
the TypeScript types silently diverge from the API.
"""
from __future__ import annotations

import itertools

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.config import settings
from app.core.llm import MockLLM
from app.core.orchestrator import Orchestrator
from app.core.pharmstate import AGENT_WRITE_FIELDS, PharmState
from app.core.store import SessionStore
from app.evidence import store as evstore
from tests import evidence_fixtures as fx

EV = "/api/sessions/{sid}/evidence"

SOURCE_KEYS = {"id", "session_id", "sha256", "filename", "pages", "title", "metadata", "text_chars",
               "parse_status", "decimal_style", "licence_note", "uploaded_by", "uploaded_at"}
METADATA_KEYS = {"title", "author", "subject", "creator", "producer", "creation_date"}
UPLOAD_KEYS = {"source", "duplicate", "summary"}
CONFIG_KEYS = {"kind", "body", "body_hash", "saved_by", "saved_at", "summary"}
SNAPSHOT_KEYS = {"session_id", "sources", "criteria", "template", "screens", "records", "datasets",
                 "counts", "summary"}
COUNTS_KEYS = {"sources", "screens", "records", "datasets"}
SUMMARY_KEYS = {"sources", "parsed", "no_text", "failed", "screens", "records", "datasets",
                "criteria_hash", "template_hash", "wide_path", "updated_at"}
PAGE_KEYS = {"source_id", "page", "pages", "text", "small_text", "columns", "tables", "decimal_style"}
TABLE_KEYS = {"index", "strategy", "bbox", "caption", "rows"}
STATE_FIELDS = {"evidence_summary", "screening_results", "extraction_results", "evidence_dataset_path"}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    evstore.configure(":memory:")
    main.orch = Orchestrator(llm=MockLLM(), clock=lambda c=itertools.count(): f"t{next(c)}",
                             store=SessionStore(":memory:"))
    yield TestClient(main.app)


@pytest.fixture
def loaded(client, tmp_path):
    paths = fx.build_all(tmp_path / "papers")
    sid = client.post("/api/sessions").json()["id"]
    up = client.post(EV.format(sid=sid) + "/upload",
                     files={"file": ("paper_c.pdf", paths["paper_c"].read_bytes(), "application/pdf")})
    assert up.status_code == 200, up.text
    crit = client.post(EV.format(sid=sid) + "/criteria", json={"criteria": [
        {"id": "c1", "text": "human PK", "kind": "inclusion", "required": True, "keywords": ["pharmacokinetic"]}]})
    tmpl = client.post(EV.format(sid=sid) + "/template", json={"fields": [
        {"name": "cmax", "type": "number", "unit": "ng/mL"}]})
    assert crit.status_code == 200 and tmpl.status_code == 200
    return sid, up.json(), crit.json(), tmpl.json()


def test_upload_and_source_row_shape(loaded):
    _, up, _, _ = loaded
    assert set(up) == UPLOAD_KEYS
    assert set(up["source"]) == SOURCE_KEYS
    assert set(up["source"]["metadata"]) == METADATA_KEYS
    assert isinstance(up["duplicate"], bool) and isinstance(up["summary"], str)


def test_config_response_shape(loaded):
    _, _, crit, tmpl = loaded
    assert set(crit) == CONFIG_KEYS and set(tmpl) == CONFIG_KEYS
    assert set(crit["body"][0]) == {"id", "text", "kind", "required", "keywords"}
    assert set(tmpl["body"]) == {"fields"}
    assert set(tmpl["body"]["fields"][0]) == {"name", "type", "unit", "description", "required", "min", "max",
                                              "aliases", "decimal"}


def test_snapshot_shape(client, loaded):
    sid, up, crit, tmpl = loaded
    snap = client.get(EV.format(sid=sid)).json()
    assert set(snap) == SNAPSHOT_KEYS and snap["session_id"] == sid
    assert [s["id"] for s in snap["sources"]] == [up["source"]["id"]]
    assert set(snap["sources"][0]) == SOURCE_KEYS
    assert set(snap["criteria"]) == CONFIG_KEYS - {"summary"} and set(snap["template"]) == CONFIG_KEYS - {"summary"}
    assert snap["screens"] == [] and snap["records"] == [] and snap["datasets"] == []
    assert set(snap["counts"]) == COUNTS_KEYS
    assert set(snap["counts"]["screens"]) == {"proposed", "approved", "rejected"}
    assert set(snap["summary"]) == SUMMARY_KEYS
    assert snap["summary"]["criteria_hash"] == crit["body_hash"]
    assert snap["summary"]["template_hash"] == tmpl["body_hash"]


def test_empty_snapshot_shape(client):
    sid = client.post("/api/sessions").json()["id"]
    snap = client.get(EV.format(sid=sid)).json()
    assert set(snap) == SNAPSHOT_KEYS
    assert snap["sources"] == [] and snap["criteria"] is None and snap["template"] is None
    assert snap["summary"] is None and snap["counts"]["sources"] == 0


def test_page_response_shape(client, loaded):
    sid, up, _, _ = loaded
    page = client.get(EV.format(sid=sid) + f"/sources/{up['source']['id']}/pages/1").json()
    assert set(page) == PAGE_KEYS
    assert page["columns"] is None or (isinstance(page["columns"], list) and len(page["columns"]) == 2)
    assert all(set(t) == TABLE_KEYS for t in page["tables"])


def test_pharmstate_mirrors_and_whitelist():
    state = PharmState()
    assert STATE_FIELDS <= set(PharmState.model_fields)
    assert all(getattr(state, f) is None for f in STATE_FIELDS)
    assert AGENT_WRITE_FIELDS["evidence"] == STATE_FIELDS
    assert not any(STATE_FIELDS & fields for agent, fields in AGENT_WRITE_FIELDS.items() if agent != "evidence")


def test_evidence_routes_are_registered():
    paths = sorted({r.path for r in main.app.routes if "evidence" in r.path})
    base = "/api/sessions/{sid}/evidence"
    assert {base, base + "/upload", base + "/criteria", base + "/template",
            base + "/sources/{src_id}/pages/{n}", base + "/sources/{src_id}/pdf"} <= set(paths)
    assert len(paths) >= 6


# ------------------------------------------------- S8: pipeline response shapes --
# frontend/src/evidence/types.ts mirrors every set below (ToolEnvelope, ScreenRow, RecordRow, ...).

TOOL_ENVELOPE_KEYS = {"agent", "tool", "summary", "state", "result", "audit_ok"}
SCREEN_ROW_KEYS = {"id", "session_id", "source_id", "criteria_hash", "verdict", "score", "reasoning", "criteria",
                   "method", "status", "decided_by", "decided_at", "decision_reason", "created_at"}
SCREEN_RESULT_KEYS = {"screens", "skipped", "criteria_hash", "n_new", "n_screened", "method"}
SCREENING_RESULTS_KEYS = {"criteria_hash", "n_screened", "verdicts", "proposed", "approved", "rejected", "method",
                          "updated_at"}
SCREEN_DECIDE_KEYS = {"decided", "summary", "state", "screening_results"}
RECORD_ROW_KEYS = {"id", "session_id", "source_id", "template_hash", "field", "group_label", "value", "value_num",
                   "unit", "locator", "extractor", "verify_status", "verify_reason", "flags", "status",
                   "decided_by", "decided_at", "decision_reason", "record_hash", "created_at"}
LOCATOR_KEYS = {"page", "kind", "snippet", "span", "literal", "cell", "column", "decimal_style"}
EXTRACT_RESULT_KEYS = {"records", "sources", "skipped", "template_hash", "method", "n_candidates", "pass", "fail",
                       "flagged", "n_new"}
EXTRACT_SOURCE_KEYS = {"source_id", "method", "note", "n_candidates", "pass", "fail", "flagged", "n_new",
                       "unmatched_headers", "dropped"}
EXTRACTION_RESULTS_KEYS = {"template_hash", "n_records", "proposed", "approved", "rejected", "verify_fail",
                           "flagged", "method", "updated_at"}
VERIFY_RESULT_KEYS = {"n_checked", "pass", "fail", "changed", "records"}
RECORD_DECIDE_KEYS = {"decided", "summary", "state", "extraction_results"}
RECORD_DECIDED_KEYS = {"record", "record_hash", "source_sha256", "locator", "value", "unit", "verify_status",
                       "decision", "reason"}
EXPORT_RESULT_KEYS = {"dataset", "long_path", "wide_path", "long_sha256", "wide_sha256", "record_hashes",
                      "n_records", "n_sources", "template_hash", "long_columns", "wide_columns"}
DATASET_ROW_KEYS = {"id", "session_id", "template_hash", "long_path", "wide_path", "long_sha256", "wide_sha256",
                    "record_hashes", "n_records", "n_sources", "exported_by", "exported_at"}


def _post(client, sid: str, path: str, body: dict) -> dict:
    r = client.post(EV.format(sid=sid) + path, json=body)
    assert r.status_code == 200, f"{path}: {r.status_code} {r.text}"
    return r.json()


def test_pipeline_response_shapes(client, tmp_path):
    paths = fx.build_all(tmp_path / "papers")
    sid = client.post("/api/sessions").json()["id"]
    client.post(EV.format(sid=sid) + "/upload",
                files={"file": ("paper_a.pdf", paths["paper_a"].read_bytes(), "application/pdf")})
    _post(client, sid, "/criteria", {"criteria": [
        {"id": "c1", "text": "human PK", "kind": "inclusion", "required": True, "keywords": ["pharmacokinetic"]}]})

    screen = _post(client, sid, "/screen", {})
    assert set(screen) == TOOL_ENVELOPE_KEYS and screen["tool"] == "screen_publications"
    assert set(screen["result"]) == SCREEN_RESULT_KEYS
    row = screen["result"]["screens"][0]
    assert set(row) == SCREEN_ROW_KEYS and row["verdict"] == "include"
    assert set(screen["state"]["screening_results"]) == SCREENING_RESULTS_KEYS
    assert set(screen["state"]["screening_results"]["verdicts"]) == {"include", "exclude", "unsure"}

    dec = _post(client, sid, "/screen/decide", {"decisions": [
        {"screen_id": row["id"], "decision": "approve", "reason": "human PK"}]})
    assert set(dec) == SCREEN_DECIDE_KEYS and set(dec["decided"][0]) == SCREEN_ROW_KEYS
    assert dec["decided"][0]["status"] == "approved"

    _post(client, sid, "/template", {"fields": [
        {"name": "cmax", "type": "number", "unit": "ng/mL", "aliases": ["Cmax"]}]})
    ext = _post(client, sid, "/extract", {})
    assert set(ext) == TOOL_ENVELOPE_KEYS and set(ext["result"]) == EXTRACT_RESULT_KEYS
    assert set(ext["result"]["sources"][0]) == EXTRACT_SOURCE_KEYS
    rec = ext["result"]["records"][0]
    assert set(rec) == RECORD_ROW_KEYS and set(rec["locator"]) == LOCATOR_KEYS
    assert set(ext["state"]["extraction_results"]) == EXTRACTION_RESULTS_KEYS

    ver = _post(client, sid, "/records/verify", {})
    assert set(ver) == TOOL_ENVELOPE_KEYS and set(ver["result"]) == VERIFY_RESULT_KEYS

    passing = next(r for r in client.get(EV.format(sid=sid)).json()["records"] if r["verify_status"] == "pass")
    rdec = _post(client, sid, "/records/decide", {"decisions": [
        {"record_id": passing["id"], "decision": "approve", "reason": "on page"}]})
    assert set(rdec) == RECORD_DECIDE_KEYS and set(rdec["decided"][0]) == RECORD_DECIDED_KEYS
    assert set(rdec["decided"][0]["record"]) == RECORD_ROW_KEYS

    exp = _post(client, sid, "/export", {})
    assert set(exp) == TOOL_ENVELOPE_KEYS and set(exp["result"]) == EXPORT_RESULT_KEYS
    assert set(exp["result"]["dataset"]) == DATASET_ROW_KEYS
    snap = client.get(EV.format(sid=sid)).json()
    assert set(snap["datasets"][0]) == DATASET_ROW_KEYS and set(snap["screens"][0]) == SCREEN_ROW_KEYS
    assert set(snap["records"][0]) == RECORD_ROW_KEYS
    assert snap["summary"]["wide_path"] == exp["result"]["wide_path"]


def _context_text(page: dict, loc: dict) -> str:
    """The text RecordContext (frontend S9) highlights: a text locator with a column set
    refers to that column's text, every other locator to the page text."""
    if loc["kind"] == "text" and loc["column"] is not None:
        return page["columns"][loc["column"]]
    return page["text"]


def test_record_locators_round_trip_through_the_pages_endpoint(client, tmp_path):
    """RecordContext marks locator.span on the GET .../pages/{n} text with raw (code point)
    offsets and marks table cells by {table, row, col}; pin both against the server."""
    from app.evidence.normalize import normalise
    from tests.test_evidence_extract import INCLUDE_ALL, TEMPLATE

    paths = fx.build_all(tmp_path / "papers")
    sid = client.post("/api/sessions").json()["id"]
    for name in ("paper_a", "paper_b", "paper_c", "paper_d"):
        r = client.post(EV.format(sid=sid) + "/upload",
                        files={"file": (f"{name}.pdf", paths[name].read_bytes(), "application/pdf")})
        assert r.status_code == 200, r.text
    _post(client, sid, "/criteria", {"criteria": INCLUDE_ALL})
    screens = _post(client, sid, "/screen", {})["result"]["screens"]
    _post(client, sid, "/screen/decide", {"decisions": [
        {"screen_id": s["id"], "decision": "approve", "reason": "ok"} for s in screens]})
    _post(client, sid, "/template", TEMPLATE)
    _post(client, sid, "/extract", {})

    records = client.get(EV.format(sid=sid)).json()["records"]
    assert records, "extraction proposed no records"
    kinds, spans, cells = set(), 0, 0
    pages: dict[tuple[str, int], dict] = {}
    for rec in records:
        loc = rec["locator"]
        assert set(rec) == RECORD_ROW_KEYS and set(loc) == LOCATOR_KEYS
        assert isinstance(loc["page"], int) and loc["page"] >= 1 and loc["kind"] in {"text", "table"}
        assert loc["decimal_style"] in {".", ","}
        assert loc["span"] is None or (len(loc["span"]) == 2 and all(isinstance(x, int) for x in loc["span"]))
        key = (rec["source_id"], loc["page"])
        if key not in pages:
            r = client.get(EV.format(sid=sid) + f"/sources/{rec['source_id']}/pages/{loc['page']}")
            assert r.status_code == 200, r.text
            pages[key] = r.json()
        page = pages[key]
        assert set(page) == PAGE_KEYS and page["page"] == loc["page"]
        kinds.add(loc["kind"])
        if rec["verify_status"] == "fail":
            assert loc["span"] is None and loc["literal"] is None
            continue
        if loc["span"] is not None:
            spans += 1
            start, end = loc["span"]
            text = _context_text(page, loc)
            assert 0 <= start < end <= len(text)
            assert normalise(text[start:end])[0] == normalise(loc["snippet"])[0], rec["id"]
            if loc["literal"] is not None:
                assert loc["literal"] in text[start:end], rec["id"]
        if loc["kind"] == "table":
            cells += 1
            cell = loc["cell"]
            assert {"table", "row", "col", "header", "row_label"} <= set(cell)
            table = next(t for t in page["tables"] if t["index"] == cell["table"])
            raw = table["rows"][cell["row"]][cell["col"]]
            assert normalise(loc["snippet"])[0] in normalise(raw)[0], rec["id"]
            if loc["literal"] is not None:
                assert normalise(loc["literal"])[0] in normalise(raw)[0], rec["id"]
    assert kinds == {"text", "table"} and spans > 0 and cells > 0
