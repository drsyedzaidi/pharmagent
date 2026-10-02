"""Attack round 2 regressions on approval, audit and export:

- a group label differing only by an invisible format character (U+200B) is the SAME slot: the
  second value needs allow_conflict;
- after a template turns an old field name into an alias, a rival value for the canonical slot
  needs allow_conflict (the gate agrees with the duplicate flag);
- the reviewer's reason is in the audit entry of decide_records / decide_screening;
- the ledger refuses any change to a decided (rejected) record or screen, and an approved row whose
  verify_status is not 'pass'; the export leaves such a row out.
"""
from __future__ import annotations

import csv
import sqlite3
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.config import settings
from app.core.pharmstate import PharmState
from app.evidence import export_tools, extract_tools, llm_access
from app.evidence import store as evstore
from app.evidence.models import EvidenceStateError
from app.tools.base import ToolContext
from tests import evidence_fixtures as fx
from tests.test_evidence_extract import SID, SchemaLLM, _extract, _include, _records, _template
from tests.test_evidence_records_api import (
    EV,
    _audit,
    _decide,
    _setup,
    client,  # noqa: F401 - fixture
)
from tests.test_evidence_records_api import _records as _api_records


@pytest.fixture(scope="module")
def papers(tmp_path_factory) -> dict[str, Path]:
    return fx.build_all(tmp_path_factory.mktemp("papers"))


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    st = evstore.configure(str(tmp_path / "ledger.db"))
    llm_access.set_llm(None)
    yield SimpleNamespace(store=st, ctx=ToolContext(data_dir=str(tmp_path), actor="alice"), sid=SID, dir=tmp_path)
    llm_access.set_llm(None)


def _approve(env, rec_id: str, **kw):
    return extract_tools.decide_records(PharmState(session_id=SID), env.ctx, {
        "record_id": rec_id, "decision": "approve", "reason": "checked", "confirm": True, **kw})


def _propose(env, records: list[dict]) -> None:
    llm_access.set_llm(SchemaLLM({"records": records}))
    _extract(env)
    llm_access.set_llm(None)


def _cl(value: str, group: str, snippet: str) -> dict:
    return {"field": "cl", "value": value, "page": 1, "kind": "text", "group": group, "snippet": snippet}


def test_zero_width_space_in_a_group_label_is_the_same_slot(env, papers):
    src = _include(env, papers, names=("paper_d",))["paper_d"]
    _template(env, {"fields": [{"name": "cl", "type": "number"}]})
    _propose(env, [_cl("2.31", "Arm A", "was 2.31 L/h"), _cl("48.0", "Arm​ A", "volume of distribution was 48.0")])
    rows = {r.value: r for r in _records(env, src, "cl")}
    assert all("duplicate" in {f["code"] for f in r.flags} for r in rows.values())
    _approve(env, rows["2.31"].id)
    with pytest.raises(ValueError, match="allow_conflict"):
        _approve(env, rows["48.0"].id)


def test_alias_renamed_field_still_meets_the_conflict_gate(env, papers):
    src = _include(env, papers, names=("paper_d",))["paper_d"]
    _template(env, {"fields": [{"name": "cl_obs", "type": "number"}]})
    _propose(env, [{**_cl("2.31", "", "was 2.31 L/h"), "field": "cl_obs"}])
    _approve(env, _records(env, src, "cl_obs")[0].id)
    _template(env, {"fields": [{"name": "cl", "type": "number", "aliases": ["cl_obs"]}]})
    _propose(env, [_cl("48.0", "", "volume of distribution was 48.0")])
    rival = _records(env, src, "cl")[0]
    with pytest.raises(ValueError, match="allow_conflict"):
        _approve(env, rival.id)
    assert _approve(env, rival.id, allow_conflict=True).result["record"]["status"] == "approved"


def test_reviewer_reason_is_in_the_audit_entry(client, papers):  # noqa: F811 - fixture
    sid = _setup(client, papers, names=("paper_a",))
    client.post(EV.format(sid=sid) + "/extract", json={})
    rec = _api_records(client, sid)[0]
    assert _decide(client, sid, [{"record_id": rec["id"], "decision": "approve",
                                  "reason": "REVIEWER-REASON-XYZ matches table 1"}]).status_code == 200
    entries = {e["tool"]: e for e in _audit(client, sid)}
    assert entries["decide_records"]["reason"] == "REVIEWER-REASON-XYZ matches table 1"
    assert entries["decide_screening"]["reason"] == "ok"


def _decided(env, papers):
    src = _include(env, papers, names=("paper_d",))["paper_d"]
    _template(env, {"fields": [{"name": "cl", "type": "number"}]})
    _propose(env, [_cl("2.31", "", "was 2.31 L/h"), _cl("2.41", "B", "was 2.41 L/h")])
    rows = {r.value: r for r in _records(env, src, "cl")}
    assert rows["2.41"].verify_status == "fail"
    _approve(env, rows["2.31"].id)
    extract_tools.decide_records(PharmState(session_id=SID), env.ctx, {
        "record_id": rows["2.41"].id, "decision": "reject", "reason": "fails", "confirm": True})
    return rows


def test_decided_rows_are_frozen_in_sql(env, papers):
    rows = _decided(env, papers)
    con = env.store._conn
    with pytest.raises(sqlite3.IntegrityError, match="immutable|verification"):
        con.execute("UPDATE evidence_record SET status='approved' WHERE id=?", (rows["2.41"].id,))
    with pytest.raises(sqlite3.IntegrityError, match="decided record is immutable"):
        con.execute("UPDATE evidence_record SET decision_reason='edited' WHERE id=?", (rows["2.41"].id,))
    with pytest.raises(sqlite3.IntegrityError, match="decided record is immutable"):
        con.execute("DELETE FROM evidence_record WHERE id=?", (rows["2.41"].id,))
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        con.execute("UPDATE evidence_screen SET decided_verdict='exclude' WHERE status='approved'")
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        con.execute("DELETE FROM evidence_screen WHERE status='approved'")


def test_an_approved_row_that_failed_verification_is_refused_in_sql(env, papers):
    src = _include(env, papers, names=("paper_d",))["paper_d"]
    _template(env, {"fields": [{"name": "cl", "type": "number"}]})
    _propose(env, [_cl("2.41", "B", "was 2.41 L/h")])
    planted = _records(env, src, "cl")[0]
    with pytest.raises(sqlite3.IntegrityError, match="verification"):
        env.store._conn.execute("UPDATE evidence_record SET status='approved' WHERE id=?", (planted.id,))


def test_export_leaves_out_an_approved_row_that_did_not_pass(env, papers):
    rows = _decided(env, papers)
    con = env.store._conn
    con.execute("DROP TRIGGER IF EXISTS trg_evrec_decided")          # a ledger edited by hand
    con.execute("DROP TRIGGER IF EXISTS trg_evrec_unverified")
    con.execute("UPDATE evidence_record SET status='approved' WHERE id=?", (rows["2.41"].id,))
    res = export_tools.export_evidence_dataset(PharmState(session_id=SID), replace(env.ctx, actor="alice"), {})
    with open(res.result["long_path"], newline="") as fh:
        values = [r["value"] for r in csv.DictReader(fh)]
    assert values == ["2.31"]
    assert [x["record_id"] for x in res.result["left_out"]] == [rows["2.41"].id]


def test_the_decide_path_still_works(env, papers):
    rows = _decided(env, papers)
    with pytest.raises(EvidenceStateError):
        env.store.decide_record(rows["2.31"].id, "rejected", "bob", "again")
