"""Round-2 attack regressions on the ledger: inclusion, actors and export files.

- A paper the reviewer EXCLUDED under the current criteria feeds nothing: its records cannot be
  approved, and records approved earlier are left out of the export (and listed).
- Extraction reads only sources decided 'include' under the CURRENT criteria hash; never a fallback
  to an earlier screen (whose same-second ordering is not even well defined).
- The ledger names the authenticated caller (ToolContext.actor) for exports, configs and uploads; an
  'actor' argument (a model's choice on a chat turn) is ignored, and saving criteria / a template is
  a human-only tool.
- Two exports in the same second never overwrite each other's files.
"""
from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.config import settings
from app.core.audit import AuditChain
from app.core.pharmstate import PharmState
from app.core.provenance import file_sha256
from app.evidence import export_tools, extract_tools, llm_access, screen_tools
from app.evidence import store as evstore
from app.evidence.config_tools import validate_criteria
from app.tools.base import HumanOnlyToolError, ToolContext
from app.tools.builtins import default_registry
from tests import evidence_fixtures as fx
from tests.test_evidence_extract import INCLUDE_ALL, _include, _template

SID = "sess_r2_ledger"


@pytest.fixture(scope="module")
def papers(tmp_path_factory) -> dict[str, Path]:
    return fx.build_all(tmp_path_factory.mktemp("papers"))


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    st = evstore.configure(":memory:")
    llm_access.set_llm(None)
    yield SimpleNamespace(store=st, ctx=ToolContext(data_dir=str(tmp_path), actor="alice"), sid=SID, dir=tmp_path)
    llm_access.set_llm(None)


def _state(env) -> PharmState:
    return PharmState(session_id=env.sid)


def _decide(env, rec_id: str, decision: str = "approve"):
    return extract_tools.decide_records(_state(env), env.ctx, {"record_id": rec_id, "decision": decision,
                                                                 "reason": "checked", "confirm": True})


def _passing(env, source_id: str) -> list:
    return [r for r in env.store.list_records(env.sid, status="proposed", source_id=source_id)
            if r.verify_status == "pass"]


def _rescreen(env, criteria: list[dict], decisions: dict[str, str]) -> None:
    """Save *criteria*, screen every source under them and decide each source as *decisions* says."""
    env.store.save_config(env.sid, "criteria", validate_criteria(criteria), "alice")
    screen_tools.screen_publications(_state(env), env.ctx, {})
    current = env.store.get_config(env.sid, "criteria")["body_hash"]
    for scr in env.store.list_screens(env.sid):
        if scr.criteria_hash == current and scr.status == "proposed":
            screen_tools.decide_screening(_state(env), env.ctx, {"screen_id": scr.id, "confirm": True,
                                                                  "decision": decisions[scr.source_id],
                                                                  "reason": "reviewed"})


V2 = [{**INCLUDE_ALL[0], "text": "Reports Cmax, AUC or clearance (revised)"}]
V3 = [{**INCLUDE_ALL[0], "text": "Reports Cmax, AUC or clearance (third)"}]


# ------------------------------------------------------------- inclusion ---

def test_excluded_paper_cannot_be_approved_and_is_left_out_of_the_export(env, papers):
    sources = _include(env, papers, names=("paper_a", "paper_b"), sid=env.sid)
    _template(env, sid=env.sid)
    extract_tools.extract_evidence(_state(env), env.ctx, {})
    a, b = sources["paper_a"], sources["paper_b"]
    first_a, first_b = _passing(env, a.id)[0], _passing(env, b.id)[0]
    _decide(env, first_a.id)
    _decide(env, first_b.id)
    _rescreen(env, V2, {a.id: "exclude", b.id: "include"})
    with pytest.raises(ValueError, match="not included"):
        _decide(env, _passing(env, a.id)[0].id)
    _decide(env, _passing(env, b.id)[0].id)                 # an included paper is unaffected
    out = export_tools.export_evidence_dataset(_state(env), env.ctx, {})
    rows = list(csv.DictReader(open(out.result["long_path"], encoding="utf-8")))
    assert rows and {r["source_id"] for r in rows} == {b.id}
    assert [x["record_id"] for x in out.result["left_out"]] == [first_a.id]
    assert "left out" in out.action


def test_export_refuses_when_every_approved_record_is_excluded(env, papers):
    sources = _include(env, papers, names=("paper_a",), sid=env.sid)
    _template(env, sid=env.sid)
    extract_tools.extract_evidence(_state(env), env.ctx, {})
    _decide(env, _passing(env, sources["paper_a"].id)[0].id)
    _rescreen(env, V2, {sources["paper_a"].id: "exclude"})
    with pytest.raises(ValueError, match="not currently included"):
        export_tools.export_evidence_dataset(_state(env), env.ctx, {})


def test_extraction_needs_a_decision_under_the_current_criteria(env, papers):
    sources = _include(env, papers, names=("paper_a",), sid=env.sid)
    _template(env, sid=env.sid)
    _rescreen(env, V2, {sources["paper_a"].id: "exclude"})          # later decision: exclude
    env.store.save_config(env.sid, "criteria", validate_criteria(V3), "alice")   # saved, never screened
    with pytest.raises(ValueError, match="no included publications"):
        extract_tools.extract_evidence(_state(env), env.ctx, {})


# ----------------------------------------------------------------- actors ---

def test_export_records_the_authenticated_actor_not_an_argument(env, papers):
    sources = _include(env, papers, names=("paper_a",), sid=env.sid)
    _template(env, sid=env.sid)
    extract_tools.extract_evidence(_state(env), env.ctx, {})
    _decide(env, _passing(env, sources["paper_a"].id)[0].id)
    reg = default_registry()
    _, res = reg.execute("export_evidence_dataset", state=_state(env), ctx=replace(env.ctx, actor=""),
                         args={"actor": "Dr. Jane Reviewer (PI)"}, audit=AuditChain(), timestamp="t", actor="bob")
    assert res.result["dataset"]["exported_by"] == "bob"


def test_config_save_is_human_only_and_records_the_caller(env):
    reg = default_registry()
    assert reg.get("save_evidence_config").human_only
    assert "save_evidence_config" not in {t.name for t in reg.for_agent("evidence")}
    args = {"kind": "criteria", "body": INCLUDE_ALL, "actor": "Dr. Jane Reviewer (PI)"}
    with pytest.raises(HumanOnlyToolError):
        reg.execute("save_evidence_config", state=_state(env), ctx=env.ctx, args=args, audit=AuditChain(),
                    timestamp="t", actor="bob")
    reg.execute("save_evidence_config", state=_state(env), ctx=env.ctx, args=args, audit=AuditChain(),
                timestamp="t", actor="bob", allow_human_only=True)
    assert env.store.get_config(env.sid, "criteria")["saved_by"] == "bob"


def test_ingest_records_the_authenticated_uploader(env, papers):
    import hashlib
    import shutil

    from app.evidence import ingest
    data = papers["paper_a"].read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    dest = ingest.evidence_dir(env.dir) / f"{sha}.pdf"
    shutil.copyfile(papers["paper_a"], dest)
    _, res = default_registry().execute(
        "ingest_evidence_source", state=_state(env), ctx=replace(env.ctx, actor=""), audit=AuditChain(),
        args={"path": str(dest), "sha256": sha, "actor": "Dr. Jane Reviewer (PI)"}, timestamp="t", actor="bob")
    assert res.result["source"]["uploaded_by"] == "bob"


# ----------------------------------------------------------- export files ---

def test_same_second_exports_keep_both_files(env, papers, monkeypatch):
    monkeypatch.setattr(export_tools, "_stamp", lambda: "20261002T085310Z")
    sources = _include(env, papers, names=("paper_a",), sid=env.sid)
    _template(env, sid=env.sid)
    extract_tools.extract_evidence(_state(env), env.ctx, {})
    passing = _passing(env, sources["paper_a"].id)
    assert len(passing) >= 2
    _decide(env, passing[0].id)
    first = export_tools.export_evidence_dataset(_state(env), env.ctx, {}).result
    _decide(env, passing[1].id)
    second = export_tools.export_evidence_dataset(_state(env), env.ctx, {}).result
    assert first["long_path"] != second["long_path"] and first["wide_path"] != second["wide_path"]
    for res in (first, second):
        assert file_sha256(Path(res["long_path"])) == res["long_sha256"]
        assert file_sha256(Path(res["wide_path"])) == res["wide_sha256"]
