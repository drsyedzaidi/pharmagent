"""Security review MEDIUM 2: the briefing memo carries the run-provenance label (status, assurance, catalog version,
canonical-JSON digest) traced to the genesis audit entry, without weakening the memo's untraced-number check."""
from __future__ import annotations

import pytest
from docx import Document

from app.config import settings
from app.core.llm import MockLLM
from app.core.orchestrator import Orchestrator
from app.core.store import SessionStore

DIGEST = "conventions_catalog_canonical_json_sha256"


@pytest.fixture
def orch(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    return Orchestrator(llm=MockLLM(), store=SessionStore(str(tmp_path / "s.db")))


def _memo_text(orch, sess) -> str:
    from tests.memo_fixtures import full_chain, full_state

    sess.state = full_state().model_copy(update={"session_id": sess.id, "run_provenance": sess.state.run_provenance})
    sess.audit = full_chain(genesis=(sess.id, dict(sess.state.run_provenance)) if sess.state.run_provenance else None)
    orch.run_tool(sess.id, "build_briefing_memo", "report", {}, actor="alice")
    path = orch.get_session(sess.id).state.memo_path
    return "\n".join(p.text for p in Document(path).paragraphs)


def test_a_verified_memo_shows_status_assurance_catalog_version_and_digest(orch):
    sess = orch.create_session()
    rec = dict(sess.state.run_provenance)
    text = _memo_text(orch, sess)
    assert "Run provenance: verified" in text and "hash_only" in text
    assert f"v{rec['conventions_catalog_version']}" in text and rec[DIGEST] in text
    assert "[#0]" in text                                            # traced to the genesis audit entry


def test_the_untraced_number_check_still_refuses_a_stray_number(orch, monkeypatch):
    from app.compute import memo_build
    from app.compute.memo import MemoTraceError

    sess = orch.create_session()
    original = memo_build.provenance_blocks

    def leaky(state, trace):
        blocks, values = original(state, trace)
        from app.compute.memo_model import Para
        return blocks + (Para("stray 12345"),), values

    monkeypatch.setattr(memo_build, "provenance_blocks", leaky)
    with pytest.raises(MemoTraceError):
        _memo_text(orch, sess)
