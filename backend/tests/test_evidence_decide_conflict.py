"""decide_records re-checks conflicts at decision time: approving a value for a source/field/group
that already holds a DIFFERENT approved value needs an explicit allow_conflict (the approved record
is immutable, so the override is the only way past a mistake); the same value needs nothing."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.config import settings
from app.core.pharmstate import PharmState
from app.evidence import extract_tools, llm_access
from app.evidence import store as evstore
from app.tools.base import ToolContext
from tests import evidence_fixtures as fx
from tests.test_evidence_extract import SID, SchemaLLM, _extract, _include, _records, _template


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


def _decide(env, record_id, **extra):
    args = {"record_id": record_id, "decision": "approve", "reason": "checked", "confirm": True, **extra}
    return extract_tools.decide_records(PharmState(session_id=env.sid), replace(env.ctx, actor="alice"), args)


def test_conflicting_approval_needs_an_explicit_override(env, papers):
    sources = _include(env, papers, names=("paper_d",))
    _template(env, {"fields": [{"name": "cl", "type": "number", "unit": "L/h"}]})
    llm_access.set_llm(SchemaLLM({"records": [
        {"field": "cl", "value": "2.31", "unit": "L/h", "page": 1, "kind": "text", "group": "",
         "snippet": "was 2.31 L/h"},
        {"field": "cl", "value": "48.0", "unit": "L", "page": 1, "kind": "text", "group": "",
         "snippet": "volume of distribution was 48.0"}]}))
    _extract(env)
    rows = {r.value: r for r in _records(env, sources["paper_d"], "cl")}
    assert set(rows) >= {"2.31", "48.0"}
    _decide(env, rows["2.31"].id)
    with pytest.raises(ValueError, match="already holds"):
        _decide(env, rows["48.0"].id)
    assert env.store.get_record(rows["48.0"].id).status == "proposed"
    out = _decide(env, rows["48.0"].id, allow_conflict=True)
    assert out.result["record"]["status"] == "approved"
