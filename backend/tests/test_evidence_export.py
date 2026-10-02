"""export_evidence_dataset: long/wide CSV shapes, csv-injection guard, sha256 + ledger row,
idempotent record_hashes, and the download endpoint's path discipline."""
from __future__ import annotations

import csv
import hashlib
import io
import itertools
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.config import settings
from app.core.llm import MockLLM
from app.core.orchestrator import Orchestrator
from app.core.pharmstate import PharmState, apply_writes
from app.core.provenance import file_sha256
from app.core.store import SessionStore
from app.evidence import export_tools, extract_tools, llm_access
from app.evidence import store as evstore
from app.evidence.export_tools import LONG_COLUMNS, WIDE_FIXED, build_long, build_wide, csv_cell
from app.tools.base import ToolContext
from app.tools.builtins import default_registry
from tests import evidence_fixtures as fx
from tests.test_evidence_extract import _include, _template

SID = "sess_export"
EV = "/api/sessions/{sid}/evidence"


@pytest.fixture(scope="module")
def papers(tmp_path_factory) -> dict[str, Path]:
    return fx.build_all(tmp_path_factory.mktemp("papers"))


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    st = evstore.configure(":memory:")
    llm_access.set_llm(None)
    yield SimpleNamespace(store=st, ctx=ToolContext(data_dir=str(tmp_path)), sid=SID, dir=tmp_path)
    llm_access.set_llm(None)


def _approve_all(env, papers, names=("paper_a", "paper_b", "paper_d")) -> list:
    """Ingest + include + extract (rules) + approve every passing record."""
    _include(env, papers, names=names, sid=env.sid)
    _template(env, sid=env.sid)
    extract_tools.extract_evidence(PharmState(session_id=env.sid), env.ctx, {})
    for r in env.store.list_records(env.sid, status="proposed"):
        if r.verify_status == "pass":
            env.store.decide_record(r.id, "approved", "alice", "ok")
    return env.store.list_records(env.sid, status="approved")


def _export(env, state: PharmState | None = None):
    state = state or PharmState(session_id=env.sid)
    # the ledger names the authenticated caller (ctx.actor); an 'actor' argument is ignored
    res = export_tools.export_evidence_dataset(state, replace(env.ctx, actor="alice"), {"actor": "mallory"})
    return res, apply_writes(state, "evidence", res.writes)


def _read(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ------------------------------------------------------------------ builders --

def test_csv_cell_guard_escapes_formula_leaders():
    assert csv_cell("=cmd()") == "'=cmd()"
    assert csv_cell("+1") == "'+1" and csv_cell("-20") == "'-20" and csv_cell("@x") == "'@x"
    assert csv_cell("3,412") == "3,412" and csv_cell("") == "" and csv_cell(None) == ""
    assert csv_cell(12) == "12" and csv_cell(2.5) == "2.5"
    assert csv_cell("\t=1+1") == "'\t=1+1" and csv_cell("\r=1") == "'\r=1"
    assert csv_cell(" =1") == " =1"                # only the first character counts


def test_long_rows_only_approved_in_deterministic_order(env, papers):
    approved = _approve_all(env, papers)
    assert approved and any(r.status == "rejected" for r in env.store.list_records(env.sid)) is False
    # add a rejected row: must never appear
    pending = [r for r in env.store.list_records(env.sid, status="proposed")]
    if pending:
        env.store.decide_record(pending[0].id, "rejected", "alice", "no")
    sources = {s.id: s for s in env.store.list_sources(env.sid)}
    rows = build_long(env.store.list_records(env.sid), sources)
    assert [list(r.keys()) for r in rows] == [list(LONG_COLUMNS)] * len(rows)
    assert len(rows) == len(approved)
    keys = [(r["filename"], r["source_id"], r["group"], r["field"], r["record_hash"]) for r in rows]
    assert keys == sorted(keys)
    cmax = next(r for r in rows if r["field"] == "cmax" and r["value"] == "3,412")
    assert cmax["unit"] == "ng/mL" and cmax["page"] == "1" and cmax["kind"] == "table"
    assert cmax["literal"] == "3,412" and cmax["verify_status"] == "pass" and cmax["decided_by"] == "alice"
    assert cmax["source_sha256"] == sources[cmax["source_id"]].sha256 and cmax["filename"] == "paper_a.pdf"
    assert "composite_cell" in cmax["flags"]


def test_wide_one_row_per_source_group_with_every_field_and_unit(env, papers):
    approved = _approve_all(env, papers)
    sources = {s.id: s for s in env.store.list_sources(env.sid)}
    template = env.store.get_config(env.sid, "template")["body"]
    rows, columns = build_wide(approved, sources, template)
    names = [f["name"] for f in template["fields"]]
    assert columns == list(WIDE_FIXED) + [c for n in names for c in (n, f"{n}_literal", f"{n}_unit")]
    pairs = {(r.source_id, r.group_label) for r in approved}
    assert len(rows) == len(pairs)
    keys = [(r["filename"], r["source_id"], r["group"]) for r in rows]
    assert keys == sorted(keys)
    a = next(r for r in rows if r["filename"] == "paper_a.pdf" and r["group"] == "150 mg")
    assert a["cmax"] == "3412" and a["cmax_literal"] == "3,412" and a["cmax_unit"] == "ng/mL" and a["n"] == "12"
    assert all(set(r) == set(columns) for r in rows)
    # a field with no approved record is an empty cell, never missing
    assert all(r["species"] == "" for r in rows)


def test_wide_keeps_fields_outside_the_current_template(env, papers):
    approved = _approve_all(env, papers)
    sources = {s.id: s for s in env.store.list_sources(env.sid)}
    rows, columns = build_wide(approved, sources, {"fields": [{"name": "cmax", "type": "number"}]})
    extra = sorted({r.field for r in approved} - {"cmax"})
    assert columns == list(WIDE_FIXED) + ["cmax", "cmax_literal", "cmax_unit"] + [
        c for n in extra for c in (n, f"{n}_literal", f"{n}_unit")]
    assert any(r["cmax"] for r in rows)


# ---------------------------------------------------------------------- tool --

def test_export_requires_an_approved_record(env, papers):
    _include(env, papers, names=("paper_a",), sid=env.sid)
    _template(env, sid=env.sid)
    extract_tools.extract_evidence(PharmState(session_id=env.sid), env.ctx, {})
    with pytest.raises(ValueError, match="approved record"):
        _export(env)
    assert env.store.list_datasets(env.sid) == []


def test_export_writes_files_hashes_ledger_and_state(env, papers):
    approved = _approve_all(env, papers)
    res, state = _export(env)
    out = res.result
    long_path, wide_path = Path(out["long_path"]), Path(out["wide_path"])
    exports = (env.dir / "evidence" / "exports").resolve()
    assert long_path.parent.resolve() == exports and wide_path.parent.resolve() == exports
    assert long_path.name.startswith(f"evidence_{env.sid}_") and long_path.suffix == ".csv"
    assert wide_path.name == long_path.stem + "_wide.csv"
    assert hashlib.sha256(long_path.read_bytes()).hexdigest() == out["long_sha256"]
    assert hashlib.sha256(wide_path.read_bytes()).hexdigest() == out["wide_sha256"]
    assert out["record_hashes"] == sorted(r.record_hash for r in approved) == env.store.approved_record_hashes(env.sid)
    assert out["n_records"] == len(approved) and out["n_sources"] == len({r.source_id for r in approved})
    ds = env.store.list_datasets(env.sid)
    assert len(ds) == 1 and ds[0].id == out["dataset"]["id"] == "ds_" + out["long_sha256"][:12]
    assert ds[0].record_hashes == out["record_hashes"] and ds[0].exported_by == "alice"
    assert state.evidence_dataset_path == str(long_path)
    assert state.evidence_summary["datasets"] == 1 and state.evidence_summary["wide_path"] == str(wide_path)
    assert res.action.startswith("exported evidence dataset ds_")
    long_rows = _read(str(long_path))
    assert [list(r) for r in long_rows][0] == list(LONG_COLUMNS) and len(long_rows) == len(approved)
    assert all(r["record_hash"] in out["record_hashes"] for r in long_rows)
    assert "2.41" not in long_path.read_text(encoding="utf-8")


def test_two_exports_of_the_same_approved_set_are_identical(env, papers):
    _approve_all(env, papers)
    r1, _ = _export(env)
    r2, _ = _export(env)
    assert r1.result["record_hashes"] == r2.result["record_hashes"]
    assert r1.result["long_sha256"] == r2.result["long_sha256"] and r1.result["dataset"]["id"] == r2.result["dataset"]["id"]
    assert len(env.store.list_datasets(env.sid)) == 1
    assert Path(r2.result["long_path"]).is_file() and Path(r2.result["wide_path"]).is_file()


def test_formula_leading_value_is_escaped_in_the_files(env, papers, monkeypatch):
    approved = _approve_all(env, papers)
    rec = approved[0]
    evil = {**rec.__dict__, "value": "=cmd()|'/C calc'!A0", "unit": "+x", "group_label": "@grp"}
    monkeypatch.setattr(env.store, "list_records",
                        lambda sid, status=None, **kw: [type(rec)(**evil)] if status == "approved" else [])
    res, _ = _export(env)
    text = Path(res.result["long_path"]).read_text(encoding="utf-8")
    assert "'=cmd()" in text and "\n=cmd()" not in text and ",=cmd()" not in text
    assert ",'+x" in text and ",'@grp" in text
    wide = Path(res.result["wide_path"]).read_text(encoding="utf-8")
    assert "'=cmd()" in wide and ",=cmd()" not in wide


def test_tool_is_registered_for_the_evidence_agent():
    reg = default_registry()
    assert "export_evidence_dataset" in {t.name for t in reg.for_agent("evidence")}


# ------------------------------------------------------------------ download --

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


def _ready(client, papers, h=None) -> str:
    from tests.test_evidence_records_api import _setup
    sid = _setup(client, papers, h, names=("paper_a", "paper_d"))
    assert client.post(EV.format(sid=sid) + "/extract", json={}, headers=h or {}).status_code == 200
    recs = client.get(EV.format(sid=sid), headers=h or {}).json()["records"]
    decisions = [{"record_id": r["id"], "decision": "approve", "reason": "ok"} for r in recs
                 if r["verify_status"] == "pass"]
    assert client.post(EV.format(sid=sid) + "/records/decide", json={"decisions": decisions},
                       headers=h or {}).status_code == 200
    return sid


def test_export_endpoint_then_download_both_files(client, papers):
    sid = _ready(client, papers)
    r = client.post(EV.format(sid=sid) + "/export", json={})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["tool"] == "export_evidence_dataset" and body["audit_ok"] is True
    out = body["result"]
    assert set(out) >= {"dataset", "long_path", "wide_path", "long_sha256", "wide_sha256", "record_hashes",
                        "n_records", "n_sources"}
    for key, sha in (("long_path", out["long_sha256"]), ("wide_path", out["wide_sha256"])):
        name = Path(out[key]).name
        d = client.get(EV.format(sid=sid) + f"/dataset/{name}")
        assert d.status_code == 200, d.text
        assert d.headers["content-type"].startswith("text/csv")
        assert hashlib.sha256(d.content).hexdigest() == sha
        assert list(csv.reader(io.StringIO(d.text)))[0][0] in ("record_id", "source_id")
    state = client.get(f"/api/sessions/{sid}/state").json()
    assert Path(state["evidence_dataset_path"]).name == Path(out["long_path"]).name
    snap = client.get(EV.format(sid=sid)).json()
    assert len(snap["datasets"]) == 1 and snap["datasets"][0]["id"] == out["dataset"]["id"]
    assert snap["counts"]["datasets"] == 1


def test_export_endpoint_with_nothing_approved_is_400(client, papers):
    from tests.test_evidence_records_api import _setup
    sid = _setup(client, papers, names=("paper_a",))
    r = client.post(EV.format(sid=sid) + "/export", json={})
    assert r.status_code == 400 and "approved record" in r.json()["error"]["message"]


def assert_download_refused(r, name: str) -> None:
    """404 from the dataset route. With frontend/dist built, main.py's SPA fallback answers a URL whose
    dot-segments the client normalised away from /dataset/ with the index.html shell (200, text/html):
    accepted only as that shell, never a file's bytes."""
    if r.status_code == 200:
        assert r.headers["content-type"].startswith("text/html"), (name, r.headers["content-type"])
        assert r.content.lstrip().lower().startswith(b"<!doctype html"), name
        return
    assert r.status_code == 404, (name, r.status_code)


@pytest.mark.parametrize("name", ["../pharmagent.db", "%2e%2e%2fpharmagent.db", "..%2F..%2Fpharmagent.db",
                                  "/etc/passwd", "pharmagent.db", "evidence_other.csv"])
def test_download_refuses_other_names(client, papers, name):
    sid = _ready(client, papers)
    client.post(EV.format(sid=sid) + "/export", json={})
    assert_download_refused(client.get(EV.format(sid=sid) + f"/dataset/{name}"), name)


def test_download_before_any_export_is_404(client, papers):
    sid = _ready(client, papers)
    assert client.get(EV.format(sid=sid) + f"/dataset/evidence_{sid}_20260101T000000Z.csv").status_code == 404


def test_download_of_another_sessions_file_is_404(client, papers):
    s1 = _ready(client, papers)
    out = client.post(EV.format(sid=s1) + "/export", json={}).json()["result"]
    s2 = client.post("/api/sessions").json()["id"]
    name = Path(out["long_path"]).name
    assert client.get(EV.format(sid=s2) + f"/dataset/{name}").status_code == 404
    assert client.get(EV.format(sid=s1) + f"/dataset/{name}").status_code == 200


def test_download_404_when_the_file_was_removed(client, papers):
    sid = _ready(client, papers)
    out = client.post(EV.format(sid=sid) + "/export", json={}).json()["result"]
    Path(out["wide_path"]).unlink()
    assert client.get(EV.format(sid=sid) + f"/dataset/{Path(out['wide_path']).name}").status_code == 404
    assert client.get(EV.format(sid=sid) + f"/dataset/{Path(out['long_path']).name}").status_code == 200


def test_keyless_chat_exports_the_dataset(client, papers):
    sid = _ready(client, papers)
    r = client.post(f"/api/sessions/{sid}/chat", json={"message": "export the evidence dataset"})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["agent"] == "evidence" and [c["tool"] for c in out["tool_calls"]][:1] == ["export_evidence_dataset"]
    assert out["state"]["evidence_dataset_path"]
    again = client.post(f"/api/sessions/{sid}/chat", json={"message": "export the evidence dataset"}).json()
    assert again["tool_calls"] == [] and "last export" in again["messages"][0]


def test_re_export_restores_the_ledger_files_when_they_were_removed(env, papers, monkeypatch):
    """Same approved set, earlier files gone: the ledger row's paths are filled again (same bytes)."""
    _approve_all(env, papers)
    r1, _ = _export(env)
    Path(r1.result["long_path"]).unlink()
    Path(r1.result["wide_path"]).unlink()
    monkeypatch.setattr(export_tools, "_stamp", lambda: "20990101T000000Z")
    r2, state = _export(env)
    assert r2.result["long_path"] == r1.result["long_path"] and r2.result["dataset"]["id"] == r1.result["dataset"]["id"]
    for key, sha in (("long_path", "long_sha256"), ("wide_path", "wide_sha256")):
        p = Path(r2.result[key])
        assert p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == r1.result[sha]
    assert state.evidence_dataset_path == r1.result["long_path"]
    exports = env.dir / "evidence" / "exports"
    assert sorted(p.name for p in exports.iterdir()) == sorted(
        Path(r1.result[k]).name for k in ("long_path", "wide_path"))


def test_re_export_never_restores_a_wide_file_whose_bytes_differ_from_the_ledger(env, papers, monkeypatch):
    """Same approved set under a changed template: long sha matches the ledger, the wide sha does not.
    The old ledger row and its (missing) file are never refilled with other bytes; the new wide file
    gets its own ledger row (round-3 fix: the result used to name the old template's wide file)."""
    _approve_all(env, papers)
    r1, _ = _export(env)
    Path(r1.result["wide_path"]).unlink()
    tmpl = env.store.get_config(env.sid, "template")["body"]
    env.store.save_config(env.sid, "template", {"fields": list(reversed(tmpl["fields"]))}, "alice")
    monkeypatch.setattr(export_tools, "_stamp", lambda: "20990101T000000Z")
    r2, _ = _export(env)
    rows = {d.id: d for d in env.store.list_datasets(env.sid)}
    assert rows[r1.result["dataset"]["id"]].wide_sha256 == r1.result["wide_sha256"]   # old row unchanged
    assert not Path(r1.result["wide_path"]).exists()                      # never filled with other bytes
    assert r2.result["dataset"]["id"] != r1.result["dataset"]["id"]
    assert r2.result["wide_sha256"] != r1.result["wide_sha256"]
    assert file_sha256(r2.result["wide_path"]) == r2.result["wide_sha256"] == rows[r2.result["dataset"]["id"]].wide_sha256
