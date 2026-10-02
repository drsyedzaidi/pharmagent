"""When the configured model fails during extraction (timeout, transport error, invalid answer) the
keyless table rules still run, and the fallback is reported: per source (llm_error), in the tool
summary, and in the extraction_results mirror (llm_fallback) that the Records card renders, where
it survives later re-verification and decisions. The client timeout is configurable."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from app.config import settings
from app.core import llm as llm_mod
from app.core.pharmstate import apply_writes
from app.evidence import extract_tools, llm_access
from app.evidence import store as evstore
from app.tools.base import ToolContext
from tests import evidence_fixtures as fx
from tests.test_evidence_extract import SID, _extract, _include, _records, _template


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


def _timeout_client(monkeypatch, seen: list) -> llm_mod.OpenAICompatLLM:
    def fake_post(*a, **k):
        seen.append(k.get("timeout"))
        raise requests.exceptions.ReadTimeout("Read timed out. (read timeout=120.0)")
    monkeypatch.setattr(llm_mod.requests, "post", fake_post)
    return llm_mod.OpenAICompatLLM(base_url="http://127.0.0.1:11434/v1", model="gemma4:31b", api_key=None)


def test_timeout_fallback_is_reported_and_survives_decisions(env, papers, monkeypatch):
    seen: list = []
    _include(env, papers, names=("paper_d",))
    _template(env)
    llm_access.set_llm(_timeout_client(monkeypatch, seen))
    res, state = _extract(env)
    row = res.result["sources"][0]
    assert row["method"] == "table" and "timed out" in row["llm_error"]
    assert "fell back" in res.summary and "timed out" in res.summary
    fb = state.extraction_results["llm_fallback"]
    assert fb and fb[0]["source_id"] == row["source_id"] and "timed out" in fb[0]["error"]
    rec = next(r for r in _records(env) if r.verify_status == "pass")
    out = extract_tools.decide_records(state, replace(env.ctx, actor="alice"),
                                       {"record_id": rec.id, "decision": "reject", "reason": "x", "confirm": True})
    state = apply_writes(state, "evidence", out.writes)
    assert state.extraction_results["llm_fallback"] == fb


def test_keyless_extraction_reports_no_fallback(env, papers):
    _include(env, papers, names=("paper_a",))
    _template(env)
    res, state = _extract(env)
    assert res.result["sources"][0]["llm_error"] == "" and state.extraction_results["llm_fallback"] == []
    assert "fell back" not in res.summary


def test_llm_timeout_is_configurable(monkeypatch):
    seen: list = []
    client = _timeout_client(monkeypatch, seen)
    monkeypatch.setattr(settings, "llm_timeout_s", 600.0)
    assert client.extract_json("s", "u", {"type": "object"}) is None
    assert seen == [600.0] and "timed out" in (client.last_error or "")
