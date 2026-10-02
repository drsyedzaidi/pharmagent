"""POST /screen and /screen/decide over HTTP (keyless MockLLM), audit accounting, the job path
when a real LLM is configured, and the keyless chat route to screen_publications."""
from __future__ import annotations

import itertools
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
from app.evidence import store as evstore
from tests import evidence_fixtures as fx
from tests.test_evidence_llm import FakeLLM
from tests.test_evidence_screen import CRITERIA

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
    sid = client.post("/api/sessions", headers=h or {}).json()["id"]
    for n in names:
        r = client.post(EV.format(sid=sid) + "/upload", headers=h or {},
                        files={"file": (f"{n}.pdf", papers[n].read_bytes(), "application/pdf")})
        assert r.status_code == 200, r.text
    r = client.post(EV.format(sid=sid) + "/criteria", json={"criteria": CRITERIA}, headers=h or {})
    assert r.status_code == 200, r.text
    return sid


def _screens(client, sid, h=None) -> list[dict]:
    return client.get(EV.format(sid=sid), headers=h or {}).json()["screens"]


def _decide(client, sid, decisions, h=None):
    return client.post(EV.format(sid=sid) + "/screen/decide", json={"decisions": decisions}, headers=h or {})


# ------------------------------------------------------------------ screen ---

def test_screen_then_decide_with_audit_accounting(client, papers):
    sid = _setup(client, papers)
    r = client.post(EV.format(sid=sid) + "/screen", json={})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["tool"] == "screen_publications" and body["audit_ok"] is True
    assert len(body["result"]["screens"]) == 4 and body["result"]["n_new"] == 4
    rows = _screens(client, sid)
    assert len(rows) == 4 and all(s["status"] == "proposed" and s["method"] == "keyword" for s in rows)
    entries = _audit(client, sid)
    screened = [e for e in entries if e["tool"] == "screen_publications"]
    assert len(screened) == 1 and screened[0]["action"].startswith("screened 4 sources")
    state = client.get(f"/api/sessions/{sid}/state").json()
    assert state["screening_results"]["proposed"] == 4 and state["evidence_summary"]["screens"]["proposed"] == 4

    verdict = {s["id"]: s["verdict"] for s in rows}
    decisions = [{"screen_id": i, "decision": "approve" if v == "include" else "reject",
                  "reason": f"agree with {v}"} for i, v in verdict.items()]
    prior = len(_audit(client, sid))
    r = _decide(client, sid, decisions)
    assert r.status_code == 200, r.text
    out = r.json()
    assert [d["status"] for d in out["decided"]] == ["approved" if v == "include" else "rejected"
                                                      for v in verdict.values()]
    assert all(d["decided_by"] == "anonymous" for d in out["decided"])
    after = _audit(client, sid)
    assert len(after) == prior + 4
    assert all(e["actor"] == "anonymous" and e["tool"] == "decide_screening" for e in after[prior:])
    assert client.get(f"/api/sessions/{sid}/audit").json()["integrity"]["chain_ok"] is True
    db = _screens(client, sid)
    assert {s["status"] for s in db} == {"approved", "rejected"}
    assert all(s["decided_by"] == "anonymous" and s["decided_at"] for s in db)
    sr = client.get(f"/api/sessions/{sid}/state").json()["screening_results"]
    assert (sr["proposed"], sr["approved"], sr["rejected"]) == (0, 3, 1)
    assert out["screening_results"] == sr


def test_single_decision_body_is_accepted(client, papers):
    sid = _setup(client, papers, names=("paper_a",))
    client.post(EV.format(sid=sid) + "/screen", json={})
    sid_scr = _screens(client, sid)[0]["id"]
    r = client.post(EV.format(sid=sid) + "/screen/decide",
                    json={"screen_id": sid_scr, "decision": "approve", "reason": "fine"})
    assert r.status_code == 200 and r.json()["decided"][0]["status"] == "approved"


def test_decide_errors(client, papers):
    sid = _setup(client, papers, names=("paper_a", "paper_b"))
    client.post(EV.format(sid=sid) + "/screen", json={})
    a, b = [s["id"] for s in _screens(client, sid)]
    r = _decide(client, sid, [{"screen_id": a, "decision": "approve", "reason": ""}])
    assert r.status_code == 400 and "reason" in _err(r)
    r = _decide(client, sid, [{"screen_id": "scr_nope", "decision": "approve", "reason": "x"}])
    assert r.status_code == 404
    r = _decide(client, sid, [{"screen_id": a, "decision": "approve", "reason": "x"}])
    assert r.status_code == 200
    r = _decide(client, sid, [{"screen_id": a, "decision": "reject", "reason": "changed my mind"}])
    assert r.status_code == 409 and "already decided" in _err(r)
    r = _decide(client, sid, [{"screen_id": b, "decision": "maybe", "reason": "x"}])
    assert r.status_code == 400
    r = _decide(client, sid, [])
    assert r.status_code == 400
    r = client.post(EV.format(sid=sid) + "/screen/decide", json={"decisions": "nope"})
    assert r.status_code == 400
    assert {s["id"]: s["status"] for s in _screens(client, sid)} == {a: "approved", b: "proposed"}


def test_mid_list_failure_keeps_earlier_decisions(client, papers):
    sid = _setup(client, papers, names=("paper_a", "paper_b", "paper_d"))
    client.post(EV.format(sid=sid) + "/screen", json={})
    a, b, d = [s["id"] for s in _screens(client, sid)]
    prior = len(_audit(client, sid))
    r = _decide(client, sid, [{"screen_id": a, "decision": "approve", "reason": "x"},
                              {"screen_id": "scr_nope", "decision": "approve", "reason": "x"},
                              {"screen_id": b, "decision": "approve", "reason": "x"}])
    assert r.status_code == 400
    msg = _err(r)
    assert "1 of 3" in msg and a in msg and "scr_nope" in msg
    assert {s["id"]: s["status"] for s in _screens(client, sid)} == {a: "approved", b: "proposed",
                                                                      d: "proposed"}
    assert len(_audit(client, sid)) == prior + 1


def test_rescreen_same_criteria_adds_nothing_and_new_criteria_adds_rows(client, papers):
    sid = _setup(client, papers)
    h1 = client.get(EV.format(sid=sid)).json()["criteria"]["body_hash"]
    client.post(EV.format(sid=sid) + "/screen", json={})
    r = client.post(EV.format(sid=sid) + "/screen", json={})
    assert r.json()["result"]["n_new"] == 0 and len(_screens(client, sid)) == 4
    r = client.post(EV.format(sid=sid) + "/criteria", json={"criteria": CRITERIA[:2]})
    h2 = r.json()["body_hash"]
    assert h2 != h1
    r = client.post(EV.format(sid=sid) + "/screen", json={})
    assert r.json()["result"]["n_new"] == 4
    rows = _screens(client, sid)
    assert len(rows) == 8 and {s["criteria_hash"] for s in rows} == {h1, h2}
    state = client.get(f"/api/sessions/{sid}/state").json()
    assert state["evidence_summary"]["criteria_hash"] == h2
    assert state["screening_results"]["criteria_hash"] == h2 and state["screening_results"]["n_screened"] == 4


def test_screen_subset_and_errors(client, papers):
    sid = client.post("/api/sessions").json()["id"]
    r = client.post(EV.format(sid=sid) + "/screen", json={})
    assert r.status_code == 400 and "criteria" in _err(r)
    client.post(EV.format(sid=sid) + "/criteria", json={"criteria": CRITERIA})
    r = client.post(EV.format(sid=sid) + "/screen", json={})
    assert r.status_code == 400 and "publication" in _err(r)
    src = client.post(EV.format(sid=sid) + "/upload",
                      files={"file": ("a.pdf", papers["paper_a"].read_bytes(), "application/pdf")}).json()
    client.post(EV.format(sid=sid) + "/upload",
                files={"file": ("b.pdf", papers["paper_b"].read_bytes(), "application/pdf")})
    r = client.post(EV.format(sid=sid) + "/screen", json={"source_ids": [src["source"]["id"]]})
    assert r.status_code == 200 and len(r.json()["result"]["screens"]) == 1
    r = client.post(EV.format(sid=sid) + "/screen", json={"source_ids": ["src_nope"]})
    assert r.status_code == 400
    r = client.post(EV.format(sid=sid) + "/screen", json={"source_ids": "src_x"})
    assert r.status_code == 400


def test_token_principal_is_recorded(client, papers):
    settings.api_token = "secret-token"
    h = {"Authorization": "Bearer secret-token"}
    sid = _setup(client, papers, h=h, names=("paper_a",))
    assert client.post(EV.format(sid=sid) + "/screen", json={}).status_code == 401
    r = client.post(EV.format(sid=sid) + "/screen", json={}, headers=h)
    assert r.status_code == 200
    scr = _screens(client, sid, h)[0]["id"]
    r = _decide(client, sid, [{"screen_id": scr, "decision": "approve", "reason": "ok"}], h=h)
    assert r.status_code == 200 and r.json()["decided"][0]["decided_by"].startswith("token:")
    assert _audit(client, sid, h)[-1]["actor"].startswith("token:")


def test_other_sessions_screen_is_not_decidable(client, papers):
    s1 = _setup(client, papers, names=("paper_a",))
    s2 = _setup(client, papers, names=("paper_a",))
    client.post(EV.format(sid=s1) + "/screen", json={})
    scr = _screens(client, s1)[0]["id"]
    r = _decide(client, s2, [{"screen_id": scr, "decision": "approve", "reason": "x"}])
    assert r.status_code == 404 and _screens(client, s1)[0]["status"] == "proposed"


# --------------------------------------------------------------- job path ---

def test_screen_runs_as_a_job_when_a_real_llm_is_configured(client, papers):
    sid = _setup(client, papers, names=("paper_a", "paper_d"))
    llm_access.set_llm(FakeLLM(None))                  # keyed-looking client; invalid answer -> keyword
    r = client.post(EV.format(sid=sid) + "/screen", json={})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "running" and body["kind"] == "evidence_screen" and body["job_id"]
    for _ in range(200):
        job = client.get(f"/api/sessions/{sid}/jobs/{body['job_id']}").json()
        if job["status"] != "running":
            break
        time.sleep(0.05)
    assert job["status"] == "done", job
    screens = job["result"]["result"]["screens"]
    assert len(screens) == 2 and all(s["method"] == "keyword" for s in screens)
    assert all(c["note"] == "llm_invalid" for s in screens for c in s["criteria"])
    assert [e for e in _audit(client, sid) if e["tool"] == "screen_publications"]


# ------------------------------------------------------------ keyless chat ---

def test_keyless_chat_screens_the_publications(client, papers):
    sid = _setup(client, papers)
    r = client.post(f"/api/sessions/{sid}/chat", json={"message": "screen the publications"})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["agent"] == "evidence"
    assert [c["tool"] for c in out["tool_calls"]][:1] == ["screen_publications"]
    assert out["state"]["screening_results"]["proposed"] == 4
    assert len(_screens(client, sid)) == 4
    again = client.post(f"/api/sessions/{sid}/chat", json={"message": "screen the publications"}).json()
    assert again["tool_calls"] == [] and "await your decision" in again["messages"][0]
