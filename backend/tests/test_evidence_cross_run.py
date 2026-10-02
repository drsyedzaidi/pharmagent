"""Conflicts across extraction runs: a new proposal that disagrees with an earlier run's live proposal
for the same source, field and group is flagged 'duplicate' (not only when one is approved), and the
earlier run's proposal is flagged 'superseded' so the reviewer sees which one is current."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.config import settings
from app.core.pharmstate import PharmState
from app.evidence import extract_tools, llm_access
from app.evidence import models as M
from app.evidence import store as evstore
from app.tools.base import ToolContext
from tests import evidence_fixtures as fx
from tests.test_evidence_extract import SID, SchemaLLM, _extract, _include, _records, _template

CL_231 = {"field": "cl", "value": "2.31", "unit": "L/h", "page": 1, "kind": "text", "group": "",
          "snippet": "was 2.31 L/h"}
CL_480 = {"field": "CL", "value": "48.0", "unit": "L", "page": 1, "kind": "text", "group": " ",
          "snippet": "volume of distribution was 48.0"}


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


def _codes(rec) -> set[str]:
    return {f["code"] for f in rec.flags}


def _run(env, *records: dict) -> None:
    llm_access.set_llm(SchemaLLM({"records": list(records)}))
    _extract(env)


def _by_value(env, src) -> dict:
    return {r.value: r for r in _records(env, src) if r.field.casefold() == "cl"}


def test_conflict_with_an_earlier_runs_proposal_is_flagged_and_superseded(env, papers):
    src = _include(env, papers, names=("paper_d",))["paper_d"]
    _template(env, {"fields": [{"name": "cl", "type": "number", "unit": "L/h"}]})
    _run(env, CL_231)
    first = _by_value(env, src)["2.31"]
    assert M.FLAG_DUPLICATE not in _codes(first) and M.FLAG_SUPERSEDED not in _codes(first)
    _run(env, CL_480)                                    # a second run proposes a different value
    rows = _by_value(env, src)
    old, new = rows["2.31"], rows["48.0"]
    assert old.status == new.status == "proposed"
    assert M.FLAG_DUPLICATE in _codes(new), new.flags
    assert M.FLAG_DUPLICATE in _codes(old), old.flags
    assert M.FLAG_SUPERSEDED in _codes(old) and M.FLAG_SUPERSEDED not in _codes(new)
    sup = next(f for f in old.flags if f["code"] == M.FLAG_SUPERSEDED)
    assert sup["severity"] == "warn" and sup["origin"] == "script" and new.id in sup["message"]


def test_superseded_survives_reverify_and_conflict_is_recomputed(env, papers):
    src = _include(env, papers, names=("paper_d",))["paper_d"]
    _template(env, {"fields": [{"name": "cl", "type": "number", "unit": "L/h"}]})
    _run(env, CL_231)
    _run(env, CL_480)
    extract_tools.verify_records(PharmState(session_id=env.sid), env.ctx, {})
    rows = _by_value(env, src)
    assert M.FLAG_SUPERSEDED in _codes(rows["2.31"]) and M.FLAG_SUPERSEDED not in _codes(rows["48.0"])
    assert M.FLAG_DUPLICATE in _codes(rows["2.31"]) and M.FLAG_DUPLICATE in _codes(rows["48.0"])


def test_re_proposing_the_earlier_claim_makes_it_current_again(env, papers):
    src = _include(env, papers, names=("paper_d",))["paper_d"]
    _template(env, {"fields": [{"name": "cl", "type": "number", "unit": "L/h"}]})
    _run(env, CL_231)
    _run(env, CL_480)
    _run(env, CL_231)                                    # same claim as run 1: not re-inserted, current again
    rows = _by_value(env, src)
    assert M.FLAG_SUPERSEDED not in _codes(rows["2.31"]) and M.FLAG_SUPERSEDED in _codes(rows["48.0"])


def test_same_value_in_a_later_run_is_not_a_conflict(env, papers):
    src = _include(env, papers, names=("paper_d",))["paper_d"]
    _template(env, {"fields": [{"name": "cl", "type": "number", "unit": "L/h"}]})
    _run(env, CL_231)
    _run(env, {**CL_231, "snippet": "CL/F was 2.31"})
    rows = [r for r in _records(env, src) if r.field == "cl"]
    assert len(rows) == 2 and not any(M.FLAG_DUPLICATE in _codes(r) for r in rows)
    assert sum(M.FLAG_SUPERSEDED in _codes(r) for r in rows) == 1


def test_decided_records_are_never_marked(env, papers):
    src = _include(env, papers, names=("paper_d",))["paper_d"]
    _template(env, {"fields": [{"name": "cl", "type": "number", "unit": "L/h"}]})
    _run(env, CL_231)
    first = _by_value(env, src)["2.31"]
    env.store.decide_record(first.id, "rejected", "alice", "wrong")
    _run(env, CL_480)
    rows = _by_value(env, src)
    assert rows["2.31"].flags == first.flags and M.FLAG_DUPLICATE not in _codes(rows["48.0"])
