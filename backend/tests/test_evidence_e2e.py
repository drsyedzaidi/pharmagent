"""Keyless end-to-end run of the evidence pipeline over HTTP, in the task order:
upload 4 -> criteria -> screen -> decide (approve a, b, d; reject c) -> template -> extract
-> verify -> decide records (approve every pass, reject every fail) -> export -> download both.
Asserts the counts, the audit tool sequence + chain, the download bytes against the ledger's
sha256, the record hashes against the store, that paper_c contributes nothing and that paper_d's
planted 2.41 never reaches the dataset."""
from __future__ import annotations

import csv
import hashlib
import io
import itertools
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
from app.evidence.export_tools import LONG_COLUMNS, WIDE_FIXED
from tests import evidence_fixtures as fx
from tests.test_evidence_screen import CRITERIA

EV = "/api/sessions/{sid}/evidence"
PAPERS = ("paper_a", "paper_b", "paper_c", "paper_d")
TEMPLATE = {"fields": [
    {"name": "cmax", "type": "number", "unit": "ng/mL", "required": True, "aliases": ["Cmax"]},
    {"name": "auc", "type": "number", "aliases": ["AUC0-inf", "AUC"]},
    {"name": "n", "type": "integer", "required": True, "aliases": ["N"]},
    {"name": "dose", "type": "number", "unit": "mg", "aliases": ["Dose"]},
    {"name": "half_life", "type": "number", "unit": "h", "min": 0.5, "max": 200, "aliases": ["t1/2"]},
    {"name": "notes", "type": "text"},
]}


@pytest.fixture(scope="module")
def papers(tmp_path_factory) -> dict[str, Path]:
    return fx.build_all(tmp_path_factory.mktemp("papers"))


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    evstore.configure(":memory:")
    main.orch = Orchestrator(llm=MockLLM(), clock=lambda c=itertools.count(): f"t{next(c)}",
                             store=SessionStore(":memory:"))
    yield TestClient(main.app)
    llm_access.set_llm(None)


# ------------------------------------------------------------------ helpers --

def _ok(r):
    assert r.status_code == 200, f"{r.request.method} {r.request.url}: {r.status_code} {r.text}"
    return r.json()


def _upload(client, sid: str, path: Path, name: str) -> dict:
    return _ok(client.post(EV.format(sid=sid) + "/upload",
                           files={"file": (name, path.read_bytes(), "application/pdf")}))["source"]


def _snapshot(client, sid: str, **params) -> dict:
    return _ok(client.get(EV.format(sid=sid), params=params))


def _decide_screens(client, sid: str, decisions: list[dict]) -> dict:
    return _ok(client.post(EV.format(sid=sid) + "/screen/decide", json={"decisions": decisions}))


def _decide_records(client, sid: str, decisions: list[dict]) -> dict:
    return _ok(client.post(EV.format(sid=sid) + "/records/decide", json={"decisions": decisions}))


def _download(client, sid: str, path: str) -> bytes:
    r = client.get(EV.format(sid=sid) + f"/dataset/{Path(path).name}")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv"), r.text
    return r.content


def _rows(data: bytes) -> list[dict]:
    return list(csv.DictReader(io.StringIO(data.decode("utf-8"))))


# ---------------------------------------------------------------------- run --

def test_keyless_pipeline_end_to_end(client, papers):
    sid = _ok(client.post("/api/sessions"))["id"]
    sources = {n: _upload(client, sid, papers[n], f"{n}.pdf") for n in PAPERS}
    assert all(s["parse_status"] == "parsed" for s in sources.values())
    by_src = {s["id"]: n for n, s in sources.items()}

    crit = _ok(client.post(EV.format(sid=sid) + "/criteria", json={"criteria": CRITERIA}))
    assert crit["body_hash"] and isinstance(crit["summary"], str)

    screen = _ok(client.post(EV.format(sid=sid) + "/screen", json={}))
    assert screen["tool"] == "screen_publications" and screen["audit_ok"] is True
    verdicts = {by_src[s["source_id"]]: s for s in screen["result"]["screens"]}
    assert {n: v["verdict"] for n, v in verdicts.items()} == {
        "paper_a": "include", "paper_b": "include", "paper_c": "exclude", "paper_d": "include"}
    assert all(v["status"] == "proposed" and v["method"] == "keyword" for v in verdicts.values())
    decided = _decide_screens(client, sid, [
        {"screen_id": verdicts[n]["id"], "decision": "approve" if n != "paper_c" else "reject",
         "reason": f"reviewed {n}"} for n in PAPERS])
    assert [d["status"] for d in decided["decided"]] == ["approved", "approved", "rejected", "approved"]
    assert decided["screening_results"]["proposed"] == 0 and decided["screening_results"]["approved"] == 3

    tmpl = _ok(client.post(EV.format(sid=sid) + "/template", json=TEMPLATE))
    assert [f["name"] for f in tmpl["body"]["fields"]] == [f["name"] for f in TEMPLATE["fields"]]

    extract = _ok(client.post(EV.format(sid=sid) + "/extract", json={}))
    out = extract["result"]
    assert out["method"] == "table" and out["n_new"] == out["n_candidates"] >= 8
    assert {r["source_id"] for r in out["sources"]} == {sources[n]["id"] for n in ("paper_a", "paper_b", "paper_d")}
    assert [s["source_id"] for s in out["skipped"]] == [sources["paper_c"]["id"]]
    assert "rejected" in out["skipped"][0]["reason"]
    records = _snapshot(client, sid)["records"]
    assert len(records) == out["n_new"] and all(r["status"] == "proposed" for r in records)
    assert not any(r["source_id"] == sources["paper_c"]["id"] for r in records)
    assert not any(r["value"] == "2.41" for r in records)

    verify = _ok(client.post(EV.format(sid=sid) + "/records/verify", json={}))
    assert verify["result"]["n_checked"] == len(records) and verify["result"]["changed"] == 0
    records = _snapshot(client, sid)["records"]
    n_pass = sum(1 for r in records if r["verify_status"] == "pass")
    n_fail = len(records) - n_pass
    rec_decided = _decide_records(client, sid, [
        {"record_id": r["id"], "decision": "approve" if r["verify_status"] == "pass" else "reject",
         "reason": "literal match confirmed" if r["verify_status"] == "pass" else "not on page"}
        for r in records])
    er = rec_decided["extraction_results"]
    assert (er["approved"], er["rejected"], er["proposed"]) == (n_pass, n_fail, 0)
    assert all(r["status"] == "approved" for r in _snapshot(client, sid, status="approved")["records"])
    assert all(r["verify_status"] == "pass" for r in _snapshot(client, sid, status="approved")["records"])

    export = _ok(client.post(EV.format(sid=sid) + "/export", json={}))
    assert export["tool"] == "export_evidence_dataset" and export["audit_ok"] is True
    ds = export["result"]
    assert ds["n_records"] == n_pass and ds["n_sources"] == 3
    long_bytes, wide_bytes = _download(client, sid, ds["long_path"]), _download(client, sid, ds["wide_path"])
    assert hashlib.sha256(long_bytes).hexdigest() == ds["long_sha256"]
    assert hashlib.sha256(wide_bytes).hexdigest() == ds["wide_sha256"]
    assert ds["record_hashes"] == evstore.get_store().approved_record_hashes(sid)
    assert sorted(ds["record_hashes"]) == ds["record_hashes"] and len(ds["record_hashes"]) == n_pass

    long_rows, wide_rows = _rows(long_bytes), _rows(wide_bytes)
    assert list(long_rows[0]) == list(LONG_COLUMNS) and len(long_rows) == n_pass
    assert list(wide_rows[0])[: len(WIDE_FIXED)] == list(WIDE_FIXED)
    assert {r["filename"] for r in long_rows} == {"paper_a.pdf", "paper_b.pdf", "paper_d.pdf"}
    assert "paper_c.pdf" not in {r["filename"] for r in wide_rows}
    assert "2.41" not in long_bytes.decode("utf-8") and "2.41" not in wide_bytes.decode("utf-8")
    assert all(r["verify_status"] == "pass" and r["decided_by"] == "anonymous" for r in long_rows)
    a = next(r for r in long_rows if r["filename"] == "paper_a.pdf" and r["field"] == "cmax")
    assert a["value"] == "3,412" and a["literal"] == "3,412" and a["unit"] == "ng/mL"
    assert {r["record_hash"] for r in long_rows} == set(ds["record_hashes"])

    # the state mirrors and the snapshot agree with the ledger
    state = _ok(client.get(f"/api/sessions/{sid}/state"))
    assert Path(state["evidence_dataset_path"]).name == Path(ds["long_path"]).name
    assert state["evidence_summary"]["datasets"] == 1 and state["evidence_summary"]["records"]["approved"] == n_pass
    snap = _snapshot(client, sid)
    assert snap["counts"] == {"sources": 4, "screens": {"proposed": 0, "approved": 3, "rejected": 1},
                              "records": {"proposed": 0, "approved": n_pass, "rejected": n_fail}, "datasets": 1}
    assert snap["datasets"][0]["long_sha256"] == ds["long_sha256"]

    # audit: one entry per tool call, in order, chain intact
    audit = _ok(client.get(f"/api/sessions/{sid}/audit"))
    assert audit["integrity"]["chain_ok"] is True and audit["count"] == len(audit["entries"])
    tools = [e["tool"] for e in audit["entries"]]
    assert tools[0] == "session"                                  # the session-created entry
    tools = tools[1:]
    expected = (["ingest_evidence_source"] * 4 + ["save_evidence_config", "screen_publications"]
                + ["decide_screening"] * 4 + ["save_evidence_config", "extract_evidence", "verify_records"]
                + ["decide_records"] * len(records) + ["export_evidence_dataset"])
    assert tools == expected
    assert all(e["agent"] == "evidence" for e in audit["entries"][1:])
    assert audit["entries"][-1]["action"].startswith(f"exported evidence dataset {ds['dataset']['id']}")

    # the keyless chat now has nothing left to run
    chat = _ok(client.post(f"/api/sessions/{sid}/chat", json={"message": "export the evidence dataset"}))
    assert chat["agent"] == "evidence" and chat["tool_calls"] == [] and "last export" in chat["messages"][0]
