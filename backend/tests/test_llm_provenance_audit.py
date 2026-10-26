"""F-10 / F-25: the LLM that acts on a session is what the audit trail and provenance say it is."""
from __future__ import annotations

import itertools
import os
import subprocess
import sys
from pathlib import Path

import pytest

from app.config import settings
from app.core.llm import MockLLM
from app.core.orchestrator import Orchestrator
from app.core.store import SessionStore

BACKEND = Path(__file__).resolve().parent.parent


class FakeRemoteLLM(MockLLM):
    """Stands in for an OpenAI-compatible server: only provider/model/base_url matter here."""
    model = "gpt-test"
    base_url = "https://llm.example.org:8443/v1"
    api_key = "sk-must-never-be-logged"


def _orch() -> Orchestrator:
    return Orchestrator(llm=MockLLM(), clock=lambda c=itertools.count(): f"t{next(c)}",
                        store=SessionStore(":memory:"))


def test_provenance_records_the_orchestrators_actual_llm_not_settings(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(settings, "model", "some-configured-model")
    sess = _orch().create_session()
    prov = dict(sess.state.run_provenance)
    assert prov["llm_provider"] == "mock" and prov["llm_model"] == "mock"
    genesis = sess.audit.entries[0]
    assert genesis.action == "session_created" and sess.audit.verify()


def test_importing_config_does_not_create_the_data_dir(tmp_path):
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("root ignores read-only directories")
    cwd = tmp_path / "ro"
    cwd.mkdir()
    cwd.chmod(0o555)
    try:
        done = subprocess.run([sys.executable, "-c", "import app.config"], cwd=cwd,
                              env={**os.environ, "PYTHONPATH": str(BACKEND)},
                              capture_output=True, text=True)
    finally:
        cwd.chmod(0o755)
    assert done.returncode == 0, done.stderr
    assert not (cwd / "data").exists()


def test_llm_switch_is_audited_on_every_open_session_and_chain_still_verifies():
    orch = _orch()
    a, b = orch.create_session(owner="alice"), orch.create_session(owner="bob")
    n_a, n_b = len(a.audit.entries), len(b.audit.entries)

    orch.set_llm(FakeRemoteLLM())

    for sess, before in ((a, n_a), (b, n_b)):
        entries = sess.audit.entries
        assert len(entries) == before + 1
        e = entries[-1]
        assert e.action == "llm_config_changed" and e.tool == "llm_config"
        assert sess.audit.verify()
        assert orch.audit_status(sess.id)["ok"] is True
    stored = orch.store.get(a.id)["audit"][-1]
    assert stored["action"] == "llm_config_changed"
    # genesis provenance is not rewritten
    assert dict(a.state.run_provenance)["llm_provider"] == "mock"
    # the secret never reaches the chain's plaintext fields
    assert "sk-must-never" not in repr(a.audit.to_list())


def test_llm_change_entry_carries_provider_model_and_host_only():
    from app.core.orchestrator import describe_llm
    d = describe_llm(FakeRemoteLLM())
    assert d == {"provider": "openai-compatible", "model": "gpt-test", "base_url_host": "llm.example.org"}
    assert describe_llm(MockLLM()) == {"provider": "mock", "model": "mock", "base_url_host": None}


def test_unchanged_llm_does_not_add_an_entry():
    orch = _orch()
    sess = orch.create_session()
    n = len(sess.audit.entries)
    orch.set_llm(MockLLM())
    assert len(sess.audit.entries) == n


def test_each_chat_turn_records_the_acting_model():
    orch = _orch()
    sess = orch.create_session()
    orch.set_llm(FakeRemoteLLM())
    orch.chat(sess.id, "hello there")
    turn = [e for e in sess.audit.entries if e.tool == "chat" and e.action.startswith("chat_turn")]
    assert len(turn) == 1 and "gpt-test" in turn[0].action and "openai-compatible" in turn[0].action
    assert sess.audit.verify()
