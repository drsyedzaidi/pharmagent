"""Commit 4 of the provenance corrections: verification status on every access, and quarantine on mismatch.

Owner decisions: a mismatched session opens read-only (quarantined): tools, state writes and trusted exports are
blocked; audit inspection and raw state stay readable, with run_provenance_status exposed. Legacy sessions open
normally, labelled legacy_unverified. The status is always recomputed, never trusted from storage."""
from __future__ import annotations

import itertools
import json
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.core.llm import MockLLM
from app.core.orchestrator import Orchestrator
from app.core.provenance import ProvenanceQuarantined, collect_provenance
from app.core.store import SessionStore

SAMPLE = str(Path(__file__).parent.parent / "sample_data" / "oral_pk.csv")
FORGED = tuple({**collect_provenance(), "conventions_catalog_canonical_json_sha256": "0" * 64}.items())


@pytest.fixture
def orch(tmp_path):
    return Orchestrator(llm=MockLLM(), store=SessionStore(str(tmp_path / "s.db")))


def _forge(orch, sid):
    sess = orch.sessions[sid]
    sess.state = sess.state.model_copy(update={"run_provenance": FORGED})
    return orch.get_session(sid)


# ── status ─────────────────────────────────────────────────────────────────────────────────────
def test_a_new_session_is_verified(orch):
    assert orch.create_session().state.run_provenance_status == "verified"


def test_the_status_is_recomputed_on_access_not_trusted(orch):
    sid = orch.create_session().id
    assert _forge(orch, sid).state.run_provenance_status == "mismatch"


def test_a_stored_status_is_never_trusted_on_reload(tmp_path):
    db = str(tmp_path / "s.db")
    sid = Orchestrator(llm=MockLLM(), store=SessionStore(db)).create_session().id
    con = sqlite3.connect(db)
    (sj,) = con.execute("SELECT state_json FROM sessions WHERE id = ?", (sid,)).fetchone()
    st = json.loads(sj)
    st["run_provenance"] = [list(p) for p in FORGED]
    st["run_provenance_status"] = "verified"                       # a lie written into storage
    con.execute("UPDATE sessions SET state_json = ? WHERE id = ?", (json.dumps(st), sid))
    con.commit()
    con.close()
    sess = Orchestrator(llm=MockLLM(), store=SessionStore(db)).get_session(sid)
    assert sess.state.run_provenance_status == "mismatch"


def test_audit_status_reports_the_provenance_status(orch):
    sid = orch.create_session().id
    assert orch.audit_status(sid)["run_provenance_status"] == "verified"
    _forge(orch, sid)
    assert orch.audit_status(sid)["run_provenance_status"] == "mismatch"


# ── quarantine at the orchestrator and the registry ────────────────────────────────────────────
def test_a_mismatched_session_cannot_run_tools_and_its_audit_is_untouched(orch):
    sid = orch.create_session().id
    sess = _forge(orch, sid)
    n = len(sess.audit.entries)
    with pytest.raises(ProvenanceQuarantined):
        orch.run_tool(sid, "load_dataset", "data_manager", {"path": SAMPLE})
    with pytest.raises(ProvenanceQuarantined):
        orch.chat(sid, "run NCA")
    with pytest.raises(ProvenanceQuarantined):
        orch.set_roles(sid, {"TIME": "TIME"})
    with pytest.raises(ProvenanceQuarantined):
        orch.start_workflow(sid, "nca_full", {"path": SAMPLE})
    assert len(orch.get_session(sid).audit.entries) == n


def test_the_registry_refuses_a_quarantined_state_whoever_calls_it(orch):
    sid = orch.create_session().id
    sess = _forge(orch, sid)
    with pytest.raises(ProvenanceQuarantined):
        orch.registry.execute("load_dataset", state=sess.state, ctx=sess.ctx, args={"path": SAMPLE},
                              audit=sess.audit, timestamp="t", allow_expensive=True)


def test_state_writes_are_refused_except_system_integrity_records(orch):
    sid = orch.create_session().id
    sess = _forge(orch, sid)
    with pytest.raises(ProvenanceQuarantined):
        orch._persist(sess)
    orch._persist(sess, system_record=True)                          # forensic system records still persist


def test_a_legacy_session_opens_normally_labelled_legacy_unverified(tmp_path):
    from tests.test_run_provenance import _make_legacy

    db = str(tmp_path / "s.db")
    sid = Orchestrator(llm=MockLLM(), store=SessionStore(db)).create_session().id
    _make_legacy(db, sid)
    orch = Orchestrator(llm=MockLLM(), store=SessionStore(db))
    assert orch.get_session(sid).state.run_provenance_status == "legacy_unverified"
    assert orch.audit_status(sid)["run_provenance_status"] == "legacy_unverified"
    orch.run_tool(sid, "load_dataset", "data_manager", {"path": SAMPLE})   # not blocked


# ── HTTP: forensic access stays open, trusted outputs are blocked ──────────────────────────────
@pytest.fixture
def client():
    main.orch = Orchestrator(llm=MockLLM(), clock=lambda c=itertools.count(): f"t{next(c)}",
                             store=SessionStore(":memory:"))
    return TestClient(main.app)


def test_api_exposes_the_mismatch_and_keeps_audit_and_state_readable(client):
    sid = client.post("/api/sessions").json()["id"]
    _forge(main.orch, sid)
    audit = client.get(f"/api/sessions/{sid}/audit")
    assert audit.status_code == 200 and audit.json()["run_provenance_status"] == "mismatch"
    assert audit.json()["integrity"]["run_provenance_status"] == "mismatch" and audit.json()["entries"]
    state = client.get(f"/api/sessions/{sid}/state")
    assert state.status_code == 200 and state.json()["run_provenance_status"] == "mismatch"


@pytest.mark.parametrize("method,path", [
    ("get", "/export/nca"), ("get", "/export/control/nonmem"), ("get", "/cdisc"),
    ("get", "/report/x.docx"), ("get", "/report/272/x.docx"), ("get", "/memo/x.docx"),
    ("post", "/chat"), ("post", "/roles"), ("post", "/workflow/start"),
])
def test_api_blocks_trusted_exports_and_writes_for_a_quarantined_session(client, method, path):
    sid = client.post("/api/sessions").json()["id"]
    _forge(main.orch, sid)
    body = {"message": "x", "overrides": {}, "workflow": "nca_full", "params": {}}
    r = getattr(client, method)(f"/api/sessions/{sid}{path}", **({"json": body} if method == "post" else {}))
    assert r.status_code == 423, (path, r.status_code, r.text[:200])
    assert "mismatch" in r.text
