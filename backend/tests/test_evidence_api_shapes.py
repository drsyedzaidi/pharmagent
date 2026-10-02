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
                "criteria_hash", "template_hash", "updated_at"}
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
