"""The reviewer decides inclusion, not merely approve/reject of the model's verdict: 'include' or
'exclude' is stored as decided_verdict (status approved when it agrees with the proposal, rejected
when it overrides it); 'unsure' can only be resolved that way; extraction reads decided includes;
counts report the effective inclusion."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.config import settings
from app.core.pharmstate import apply_writes
from app.evidence import llm_access, screen_tools
from app.evidence import store as evstore
from app.evidence.models import effective_verdict
from app.tools.base import ToolContext
from tests import evidence_fixtures as fx
from tests.test_evidence_extract import TEMPLATE, _extract, _template
from tests.test_evidence_screen import SID, _by_source, _ingest_all, _save_criteria, _screen


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

CRIT = [{"id": "C1", "text": "human adult PK", "kind": "inclusion", "required": True, "keywords": ["human", "adult"]},
        {"id": "C2", "text": "Cmax", "kind": "inclusion", "required": True, "keywords": ["Cmax", "AUC"]},
        {"id": "E1", "text": "in vitro", "kind": "exclusion", "required": False, "keywords": ["in vitro", "rat"]}]


def _decide(env, state, screen_id, decision, reason="reviewed"):
    args = {"screen_id": screen_id, "decision": decision, "reason": reason, "confirm": True}
    res = screen_tools.decide_screening(state, replace(env.ctx, actor="alice"), args)
    return res, apply_writes(state, "evidence", res.writes)


def test_reviewer_includes_an_unsure_and_overrides_an_exclude(env, papers):
    _save_criteria(env, CRIT)
    sources = _ingest_all(env, papers, ("paper_a", "paper_c", "paper_d"))
    res, state = _screen(env)
    rows = _by_source(res, sources)
    unsure = next(r for r in rows.values() if r["verdict"] == "unsure")
    excl = rows["paper_c"]
    assert excl["verdict"] == "exclude"
    with pytest.raises(ValueError, match="include or exclude"):
        _decide(env, state, unsure["id"], "approve")
    out, state = _decide(env, state, unsure["id"], "include")
    assert out.result["screen"]["decided_verdict"] == "include" and out.result["screen"]["status"] == "rejected"
    out, state = _decide(env, state, excl["id"], "include", reason="in vitro data are wanted")
    assert out.result["screen"]["decided_verdict"] == "include"
    sr = state.screening_results
    assert sr["included"] == 2 and sr["excluded"] == 0 and sr["undecided"] == 1


def test_approve_and_reject_still_map_to_an_inclusion(env, papers):
    _save_criteria(env, CRIT)
    sources = _ingest_all(env, papers, ("paper_a", "paper_c"))
    res, state = _screen(env)
    rows = _by_source(res, sources)
    out, state = _decide(env, state, rows["paper_a"]["id"], "approve")
    assert out.result["screen"]["decided_verdict"] == "include" and out.result["screen"]["status"] == "approved"
    out, state = _decide(env, state, rows["paper_c"]["id"], "reject")
    assert out.result["screen"]["decided_verdict"] == "include"     # rejecting an exclusion = include it


def test_extraction_reads_decided_includes_only(env, papers):
    _save_criteria(env, CRIT)
    sources = _ingest_all(env, papers, ("paper_a", "paper_c"))
    res, state = _screen(env)
    rows = _by_source(res, sources)
    _decide(env, state, rows["paper_a"]["id"], "exclude", reason="duplicate cohort")
    _decide(env, state, rows["paper_c"]["id"], "include", reason="wanted")
    _template(env, TEMPLATE, sid=env.sid)
    out, _ = _extract(env)
    assert {r["source_id"] for r in out.result["records"]} == {sources["paper_c"].id}


def test_effective_verdict_of_legacy_rows():
    class S:
        def __init__(self, verdict, status, decided=None):
            self.verdict, self.status, self.decided_verdict = verdict, status, decided
    assert effective_verdict(S("include", "approved")) == "include"
    assert effective_verdict(S("include", "rejected")) == "exclude"
    assert effective_verdict(S("exclude", "rejected")) == "include"
    assert effective_verdict(S("unsure", "approved")) is None
    assert effective_verdict(S("include", "proposed")) is None
    assert effective_verdict(S("unsure", "rejected", "include")) == "include"
