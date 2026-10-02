"""POST /extract, /records/verify, /records/decide and the record filters on GET over HTTP
(keyless MockLLM), audit accounting, the immutability trigger, and the LLM job path."""
from __future__ import annotations

import itertools
import sqlite3
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.config import settings
from app.core.llm import MockLLM
from app.core.orchestrator import Orchestrator
from app.core.store import SessionStore
from app.evidence import llm_access
from app.evidence import models as M
from app.evidence import store as evstore
from app.evidence.normalize import normalise
from tests import evidence_fixtures as fx
from tests.test_evidence_extract import INCLUDE_ALL, TEMPLATE, SchemaLLM

EV = "/api/sessions/{sid}/evidence"


@pytest.fixture(scope="module")
def papers(tmp_path_factory) -> dict[str, Path]:
    return fx.build_all(tmp_path_factory.mktemp("papers"))


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    evstore.configure(":memory:")
    main.orch = Orchestrator(llm=MockLLM(), clock=lambda c=itertools.count(): f"t{next(c)}",
                             store=SessionStore(":memory:"))
    token = settings.api_token
    yield TestClient(main.app)
    settings.api_token = token
    llm_access.set_llm(None)


def _err(r) -> str:
    return r.json()["error"]["message"]


def _audit(client, sid, h=None) -> list[dict]:
    return client.get(f"/api/sessions/{sid}/audit", headers=h or {}).json()["entries"]


def _setup(client, papers, h=None, names=("paper_a", "paper_b", "paper_c", "paper_d")) -> str:
    """Upload, criteria, screen, approve every verdict, template -> ready to extract."""
    sid = client.post("/api/sessions", headers=h or {}).json()["id"]
    for n in names:
        r = client.post(EV.format(sid=sid) + "/upload", headers=h or {},
                        files={"file": (f"{n}.pdf", papers[n].read_bytes(), "application/pdf")})
        assert r.status_code == 200, r.text
    assert client.post(EV.format(sid=sid) + "/criteria", json={"criteria": INCLUDE_ALL}, headers=h or {}).status_code == 200
    r = client.post(EV.format(sid=sid) + "/screen", json={}, headers=h or {})
    assert r.status_code == 200, r.text
    decisions = [{"screen_id": s["id"], "decision": "approve", "reason": "ok"} for s in r.json()["result"]["screens"]]
    assert client.post(EV.format(sid=sid) + "/screen/decide", json={"decisions": decisions}, headers=h or {}).status_code == 200
    assert client.post(EV.format(sid=sid) + "/template", json=TEMPLATE, headers=h or {}).status_code == 200
    return sid


def _records(client, sid, h=None, **params) -> list[dict]:
    return client.get(EV.format(sid=sid), headers=h or {}, params=params).json()["records"]


def _decide(client, sid, decisions, h=None):
    return client.post(EV.format(sid=sid) + "/records/decide", json={"decisions": decisions}, headers=h or {})


# ----------------------------------------------------------------- extract ---

def test_extract_then_decide_with_audit_accounting(client, papers):
    sid = _setup(client, papers)
    r = client.post(EV.format(sid=sid) + "/extract", json={})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["tool"] == "extract_evidence" and body["audit_ok"] is True
    out = body["result"]
    assert out["n_new"] >= 12 and out["pass"] == out["n_candidates"] and out["fail"] == 0
    assert len(out["sources"]) == 4 and out["method"] == "table"
    rows = _records(client, sid)
    assert len(rows) == out["n_new"] and all(r["status"] == "proposed" for r in rows)
    entries = [e for e in _audit(client, sid) if e["tool"] == "extract_evidence"]
    assert len(entries) == 1 and entries[0]["action"].startswith("extracted ")
    state = client.get(f"/api/sessions/{sid}/state").json()
    assert state["extraction_results"]["proposed"] == len(rows)
    assert state["evidence_summary"]["records"]["proposed"] == len(rows)

    cmax = next(r for r in rows if r["field"] == "cmax" and r["value"] == "3,412")
    auc = next(r for r in rows if r["field"] == "auc" and r["value"] == "24,560")
    prior = len(_audit(client, sid))
    r = _decide(client, sid, [{"record_id": cmax["id"], "decision": "approve", "reason": "matches table 1"},
                              {"record_id": auc["id"], "decision": "reject", "reason": "unit differs"}])
    assert r.status_code == 200, r.text
    decided = r.json()["decided"]
    assert [d["decision"] for d in decided] == ["approved", "rejected"]
    assert decided[0]["record_hash"] == cmax["record_hash"] and decided[0]["locator"]["literal"] == "3,412"
    after = _audit(client, sid)
    assert len(after) == prior + 2
    assert all(e["tool"] == "decide_records" and e["actor"] == "anonymous" for e in after[prior:])
    assert client.get(f"/api/sessions/{sid}/audit").json()["integrity"]["chain_ok"] is True
    db = {r["id"]: r for r in _records(client, sid)}
    assert db[cmax["id"]]["status"] == "approved" and db[cmax["id"]]["decided_by"] == "anonymous"
    assert db[auc["id"]]["status"] == "rejected" and db[auc["id"]]["decision_reason"] == "unit differs"
    er = r.json()["extraction_results"]
    assert (er["approved"], er["rejected"]) == (1, 1) and er == client.get(f"/api/sessions/{sid}/state").json()["extraction_results"]
    assert all(r["verify_status"] == "pass" for r in _records(client, sid, status="approved"))


def test_decide_errors_and_immutability(client, papers):
    sid = _setup(client, papers, names=("paper_a",))
    client.post(EV.format(sid=sid) + "/extract", json={})
    a, b = [r["id"] for r in _records(client, sid)[:2]]
    assert _decide(client, sid, [{"record_id": a, "decision": "approve", "reason": ""}]).status_code == 400
    assert _decide(client, sid, [{"record_id": "rec_nope", "decision": "approve", "reason": "x"}]).status_code == 404
    assert _decide(client, sid, [{"record_id": a, "decision": "maybe", "reason": "x"}]).status_code == 400
    assert _decide(client, sid, []).status_code == 400
    assert _decide(client, sid, [{"record_id": a, "decision": "approve", "reason": "x"}]).status_code == 200
    r = _decide(client, sid, [{"record_id": a, "decision": "reject", "reason": "changed my mind"}])
    assert r.status_code == 409 and "already decided" in _err(r)
    prior = len(_audit(client, sid))
    r = _decide(client, sid, [{"record_id": b, "decision": "approve", "reason": "x"},
                              {"record_id": a, "decision": "approve", "reason": "x"}])
    assert r.status_code == 400 and "1 of 2" in _err(r) and b in _err(r) and len(_audit(client, sid)) == prior + 1
    conn = evstore.get_store()._conn
    with pytest.raises(sqlite3.IntegrityError, match="approved record is immutable"):
        conn.execute("UPDATE evidence_record SET value='9' WHERE id=?", (a,))
    with pytest.raises(sqlite3.IntegrityError, match="approved record is immutable"):
        conn.execute("DELETE FROM evidence_record WHERE id=?", (a,))
    assert {r["id"]: r["status"] for r in _records(client, sid)}[a] == "approved"
    assert evstore.get_store().get_record(a).value != "9"


def test_extract_errors(client, papers):
    sid = client.post("/api/sessions").json()["id"]
    r = client.post(EV.format(sid=sid) + "/extract", json={})
    assert r.status_code == 400 and "template" in _err(r)
    client.post(EV.format(sid=sid) + "/template", json=TEMPLATE)
    r = client.post(EV.format(sid=sid) + "/extract", json={})
    assert r.status_code == 400 and "include" in _err(r)
    assert client.post(EV.format(sid=sid) + "/extract", json={"source_ids": "x"}).status_code == 400
    assert client.post(EV.format(sid=sid) + "/extract", json=[1]).status_code == 400


# ----------------------------------------------------------------- filters ---

def test_snapshot_filters(client, papers):
    sid = _setup(client, papers, names=("paper_a", "paper_d"))
    client.post(EV.format(sid=sid) + "/extract", json={})
    rows = _records(client, sid)
    flagged = _records(client, sid, flagged="true")
    assert flagged and all(r["flags"] for r in flagged) and len(flagged) < len(rows)
    clean = _records(client, sid, flagged="false")
    assert clean and not any(r["flags"] for r in clean) and len(clean) + len(flagged) == len(rows)
    assert _records(client, sid, verify="fail") == [] and len(_records(client, sid, verify="pass")) == len(rows)
    rid = rows[0]["id"]
    _decide(client, sid, [{"record_id": rid, "decision": "reject", "reason": "no"}])
    assert [r["id"] for r in _records(client, sid, status="rejected")] == [rid]
    assert len(_records(client, sid, status="proposed")) == len(rows) - 1
    assert client.get(EV.format(sid=sid), params={"status": "bogus"}).status_code == 400
    snap = client.get(EV.format(sid=sid), params={"status": "rejected"}).json()
    assert snap["counts"]["records"]["proposed"] == len(rows) - 1          # counts are never filtered


# ------------------------------------------------------------------ verify ---

def test_verify_endpoint_is_idempotent_and_page_span_round_trips(client, papers):
    sid = _setup(client, papers, names=("paper_a", "paper_b"))
    client.post(EV.format(sid=sid) + "/extract", json={})
    before = {r["id"]: (r["record_hash"], r["locator"], r["flags"]) for r in _records(client, sid)}
    r = client.post(EV.format(sid=sid) + "/records/verify", json={})
    assert r.status_code == 200, r.text
    out = r.json()["result"]
    assert out["n_checked"] == len(before) and out["fail"] == 0 and out["changed"] == 0
    assert {r["id"]: (r["record_hash"], r["locator"], r["flags"]) for r in _records(client, sid)} == before
    assert [e for e in _audit(client, sid) if e["tool"] == "verify_records"]
    for rec in _records(client, sid):
        loc = rec["locator"]
        page = client.get(EV.format(sid=sid) + f"/sources/{rec['source_id']}/pages/{loc['page']}").json()
        raw = page["columns"][loc["column"]] if loc.get("column") is not None else page["text"]
        s, e = loc["span"]
        assert normalise(raw[s:e])[0] == normalise(loc["snippet"])[0]
    one = next(iter(before))
    r = client.post(EV.format(sid=sid) + "/records/verify", json={"record_ids": [one]})
    assert r.status_code == 200 and r.json()["result"]["n_checked"] == 1
    assert client.post(EV.format(sid=sid) + "/records/verify", json={"record_ids": ["rec_nope"]}).status_code == 400


def test_changed_pdf_fails_closed_through_verify_and_blocks_approval(client, papers):
    sid = _setup(client, papers, names=("paper_a",))
    client.post(EV.format(sid=sid) + "/extract", json={})
    src = evstore.get_store().list_sources(sid)[0]
    Path(src.pdf_path).write_bytes(papers["paper_d"].read_bytes())
    r = client.post(EV.format(sid=sid) + "/records/verify", json={})
    assert r.status_code == 200 and r.json()["result"]["fail"] == r.json()["result"]["n_checked"] > 0
    rows = _records(client, sid, verify="fail")
    assert rows and all(r["verify_reason"] == M.REASON_SOURCE_CHANGED for r in rows)
    r = _decide(client, sid, [{"record_id": rows[0]["id"], "decision": "approve", "reason": "fine"}])
    assert r.status_code == 400 and "failed literal verification" in _err(r)
    assert _decide(client, sid, [{"record_id": rows[0]["id"], "decision": "reject", "reason": "stale"}]).status_code == 200
    assert client.get(f"/api/sessions/{sid}/state").json()["extraction_results"]["verify_fail"] == len(rows) - 1
    assert _records(client, sid, status="approved") == []


# ---------------------------------------------------------------- job path ---

def test_extract_runs_as_a_job_with_a_real_llm_and_the_trap_value_fails(client, papers):
    sid = _setup(client, papers, names=("paper_d",))
    llm_access.set_llm(SchemaLLM({"records": [{"field": "cl", "value": "2.41", "unit": "L/h", "page": 1,
                                               "kind": "text", "group": "",
                                               "snippet": "clearance (CL/F) was 2.31 L/h"}]}))
    r = client.post(EV.format(sid=sid) + "/extract", json={})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "running" and body["kind"] == "evidence_extract" and body["job_id"]
    for _ in range(200):
        job = client.get(f"/api/sessions/{sid}/jobs/{body['job_id']}").json()
        if job["status"] != "running":
            break
        time.sleep(0.05)
    assert job["status"] == "done", job
    rows = _records(client, sid)
    assert len(rows) == 1 and rows[0]["extractor"] == "llm" and rows[0]["verify_status"] == "fail"
    assert rows[0]["verify_reason"] == M.REASON_NUMBER_MISMATCH
    r = _decide(client, sid, [{"record_id": rows[0]["id"], "decision": "approve", "reason": "trust me"}])
    assert r.status_code == 400 and "failed literal verification" in _err(r)
    assert _records(client, sid, status="approved") == []


# ------------------------------------------------------- ownership + chat ---

def test_other_session_and_token_principal(client, papers):
    settings.api_token = "secret-token"
    h = {"Authorization": "Bearer secret-token"}
    s1 = _setup(client, papers, h=h, names=("paper_a",))
    s2 = client.post("/api/sessions", headers=h).json()["id"]
    assert client.post(EV.format(sid=s1) + "/extract", json={}).status_code == 401
    assert client.post(EV.format(sid=s1) + "/extract", json={}, headers=h).status_code == 200
    rid = _records(client, s1, h)[0]["id"]
    assert _decide(client, s2, [{"record_id": rid, "decision": "approve", "reason": "x"}], h=h).status_code == 404
    r = _decide(client, s1, [{"record_id": rid, "decision": "approve", "reason": "x"}], h=h)
    assert r.status_code == 200 and r.json()["decided"][0]["record"]["decided_by"].startswith("token:")
    assert _audit(client, s1, h)[-1]["actor"].startswith("token:")


def test_keyless_chat_extracts_the_evidence(client, papers):
    sid = _setup(client, papers, names=("paper_a", "paper_d"))
    out = client.post(f"/api/sessions/{sid}/chat", json={"message": "extract the evidence"}).json()
    assert out["agent"] == "evidence" and [c["tool"] for c in out["tool_calls"]][:1] == ["extract_evidence"]
    assert out["state"]["extraction_results"]["proposed"] == len(_records(client, sid)) > 0
    again = client.post(f"/api/sessions/{sid}/chat", json={"message": "extract the evidence"}).json()
    assert again["tool_calls"] == [] and "await your decision" in again["messages"][0]
