"""Commit 5 of the provenance corrections: reports render the run provenance recorded at session creation (catalog
version, canonical-JSON digest, status) and never substitute the current server's provenance."""
from __future__ import annotations

import pytest
from docx import Document

from app.config import settings
from app.core.llm import MockLLM
from app.core.orchestrator import Orchestrator
from app.core.pharmstate import PharmState
from app.core.provenance import ProvenanceQuarantined
from app.core.store import SessionStore
from app.tools.base import ToolContext
from app.tools.regulatory_tools import generate_272
from app.tools.report_tools import generate_report

DIGEST = "conventions_catalog_canonical_json_sha256"


@pytest.fixture
def orch(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    return Orchestrator(llm=MockLLM(), store=SessionStore(str(tmp_path / "s.db")))


def _text(path: str) -> str:
    return "\n".join(p.text for p in Document(path).paragraphs)


def _render(fn, state: PharmState, tmp_path) -> str:
    res = fn(state, ToolContext(data_dir=str(tmp_path)), {})
    path = res.result.get("report_path") or res.result.get("regulatory_report_path") or res.writes.get(
        "regulatory_report_path")
    return _text(path)


@pytest.mark.parametrize("fn", [generate_report, generate_272], ids=["report", "module-2.7.2"])
def test_a_verified_session_report_shows_the_recorded_identity_and_status(orch, tmp_path, fn):
    state = orch.create_session().state
    rec = dict(state.run_provenance)
    text = _render(fn, state, tmp_path)
    assert rec["conventions_catalog_version"] in text and rec[DIGEST] in text
    assert "Run provenance status: verified" in text and rec["git_sha"] in text


@pytest.mark.parametrize("fn", [generate_report, generate_272], ids=["report", "module-2.7.2"])
def test_the_report_never_substitutes_the_current_servers_provenance(orch, tmp_path, monkeypatch, fn):
    state = orch.create_session().state
    import app.core.provenance as prov_mod
    monkeypatch.setattr(prov_mod, "collect_provenance",
                        lambda *a, **k: {**dict(state.run_provenance), "git_sha": "CURRENT-SERVER", DIGEST: "f" * 64})
    text = _render(fn, state, tmp_path)
    assert "CURRENT-SERVER" not in text and "f" * 64 not in text
    assert dict(state.run_provenance)[DIGEST] in text


@pytest.mark.parametrize("fn", [generate_report, generate_272], ids=["report", "module-2.7.2"])
def test_a_legacy_session_report_says_provenance_was_not_recorded(orch, tmp_path, fn):
    from app.core.provenance import collect_provenance

    state = PharmState(session_id="old", run_provenance=None, run_provenance_status="legacy_unverified")
    text = _render(fn, state, tmp_path)
    assert "Run provenance status: legacy_unverified" in text and "not recorded" in text
    assert collect_provenance()[DIGEST] not in text                 # nothing substituted from this server


@pytest.mark.parametrize("fn", [generate_report, generate_272], ids=["report", "module-2.7.2"])
def test_a_mismatched_state_is_refused_even_on_a_direct_call(tmp_path, fn):
    state = PharmState(session_id="x", run_provenance=(("git_sha", "a"),), run_provenance_status="mismatch")
    with pytest.raises(ProvenanceQuarantined):
        fn(state, ToolContext(data_dir=str(tmp_path)), {})


def test_a_state_with_no_status_reports_provenance_as_not_recorded(tmp_path):
    text = _render(generate_report, PharmState(), tmp_path)
    assert "Run provenance status: not recorded" in text
