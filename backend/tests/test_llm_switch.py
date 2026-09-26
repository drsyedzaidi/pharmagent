"""Runtime LLM switching: mock / local (Ollama) / openai (ChatGPT) / anthropic.
Secrets stay in memory; a failed probe never replaces the live LLM."""
from __future__ import annotations

import itertools
import json

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.core import llm as llm_mod
from app.core import llm_config
from app.core.llm import MockLLM, OpenAICompatLLM
from app.core.llm_config import DEFAULT_MODELS, LlmChoice, LlmManager, build_llm
from app.core.orchestrator import Orchestrator
from app.core.store import SessionStore


def _err(r) -> str:
    """Error text regardless of envelope shape ({detail} or {error:{message}})."""
    j = r.json()
    return str(j.get("detail") or (j.get("error") or {}).get("message") or j)


class _Resp:
    def __init__(self, payload, status=200):
        self._p, self.status_code = payload, status

    def json(self):
        return self._p


@pytest.fixture
def client(monkeypatch, tmp_path):
    from app.config import settings
    for k, v in {"llm_provider": "auto", "llm_base_url": None, "llm_api_key": None,
                 "anthropic_api_key": None, "use_mock_llm": False, "model": "m",
                 "api_token": None}.items():
        monkeypatch.setattr(settings, k, v)
    main.orch = Orchestrator(llm=MockLLM(), clock=lambda c=itertools.count(): f"t{next(c)}",
                             store=SessionStore(":memory:"))
    main.llm_manager = LlmManager(tmp_path)
    return TestClient(main.app)


@pytest.fixture
def fake_http(monkeypatch):
    state = {"chat_ok": True, "tags": ["qwen2.5:7b", "nomic-embed-text:latest", "qwen2.5:0.5b"], "posts": []}

    def post(url, json=None, headers=None, timeout=None):  # noqa: A002
        state["posts"].append({"url": url, "json": json, "headers": headers})
        if not state["chat_ok"]:
            return _Resp({"error": "model not found"}, 404)
        return _Resp({"choices": [{"message": {"role": "assistant", "content": "nca"}}]})

    def get(url, timeout=None):
        return _Resp({"models": [{"name": n} for n in state["tags"]]})

    monkeypatch.setattr(llm_mod.requests, "post", post)
    monkeypatch.setattr(llm_config.requests, "get", get)
    return state


# ── choice / builder ────────────────────────────────────────────────────────

def test_choice_public_view_never_contains_the_key():
    c = LlmChoice("openai", "gpt-4o-mini", None, "sk-secret")
    pub = c.public()
    assert pub["has_key"] is True and "sk-secret" not in json.dumps(pub)
    assert pub["base_url"] == "https://api.openai.com/v1"


def test_local_defaults_to_ollama_and_needs_no_key():
    llm = build_llm(LlmChoice("local", "qwen2.5:7b"))
    assert isinstance(llm, OpenAICompatLLM) and llm.base_url.endswith("11434/v1") and llm.api_key is None


def test_openai_and_anthropic_require_a_key():
    with pytest.raises(ValueError, match="API key"):
        build_llm(LlmChoice("openai", "gpt-4o-mini"))
    with pytest.raises(ValueError, match="API key"):
        build_llm(LlmChoice("anthropic", "claude-opus-4-8"))


def test_unknown_provider_rejected():
    with pytest.raises(ValueError, match="unknown provider"):
        build_llm(LlmChoice("gemini", "x"))


def test_anthropic_choice_builds_a_real_client_with_the_given_key():
    llm = build_llm(LlmChoice("anthropic", "claude-opus-4-8", None, "sk-ant-test"))
    assert llm.model == "claude-opus-4-8"


# ── manager persistence ─────────────────────────────────────────────────────

def test_manager_persists_choice_without_the_key(tmp_path, monkeypatch):
    from app.config import settings
    m = LlmManager(tmp_path)

    class _Orch:
        def set_llm(self, llm):
            self.llm = llm
    m.apply(LlmChoice("openai", "gpt-4o-mini", None, "sk-secret"), _Orch())
    raw = (tmp_path / "llm_settings.json").read_text()
    assert "sk-secret" not in raw and '"provider": "openai"' in raw
    # simulate a relaunch: the process env has no key -> it must be re-entered
    monkeypatch.setattr(settings, "llm_api_key", None)
    monkeypatch.setattr(settings, "anthropic_api_key", None)
    again = LlmManager(tmp_path)
    assert again.choice.provider == "openai" and again.choice.model == "gpt-4o-mini"
    assert again.choice.api_key is None


def test_activate_applies_a_persisted_local_choice_on_startup(tmp_path, monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "llm_provider", "auto")
    monkeypatch.setattr(settings, "llm_base_url", None)
    monkeypatch.setattr(settings, "anthropic_api_key", None)
    (tmp_path / "llm_settings.json").write_text(
        json.dumps({"provider": "local", "model": "qwen2.5:7b", "base_url": None}))

    class _Orch:
        llm = None

        def set_llm(self, llm):
            self.llm = llm
    o = _Orch()
    LlmManager(tmp_path).activate(o)
    assert isinstance(o.llm, OpenAICompatLLM) and o.llm.model == "qwen2.5:7b"
    assert settings.llm_label == "openai:qwen2.5:7b@http://127.0.0.1:11434/v1"


def test_activate_falls_back_when_the_persisted_choice_needs_a_missing_key(tmp_path, monkeypatch):
    from app.config import settings
    for k in ("llm_api_key", "anthropic_api_key", "llm_base_url"):
        monkeypatch.setattr(settings, k, None)
    monkeypatch.setattr(settings, "llm_provider", "auto")
    (tmp_path / "llm_settings.json").write_text(
        json.dumps({"provider": "openai", "model": "gpt-4o-mini", "base_url": None}))

    class _Orch:
        llm = None

        def set_llm(self, llm):
            self.llm = llm
    o = _Orch()
    m = LlmManager(tmp_path)
    m.activate(o)
    assert o.llm is None and m.choice.provider == "mock"     # startup not broken


def test_resolve_keeps_the_held_key_when_only_the_model_changes(tmp_path):
    m = LlmManager(tmp_path)
    m.choice = LlmChoice("openai", "gpt-4o-mini", None, "sk-held")
    c = m.resolve({"provider": "openai", "model": "gpt-4o"})
    assert c.api_key == "sk-held" and c.model == "gpt-4o"
    c2 = m.resolve({"provider": "local"})           # switching provider drops it
    assert c2.api_key is None and c2.model == DEFAULT_MODELS["local"]


# ── HTTP ────────────────────────────────────────────────────────────────────

def test_get_lists_providers_and_local_models_without_secrets(client, fake_http):
    r = client.get("/api/llm").json()
    assert r["providers"] == ["mock", "local", "openai", "anthropic"]
    assert r["local_models"] == ["qwen2.5:0.5b", "qwen2.5:7b"]
    assert r["current"]["provider"] == "mock" and "api_key" not in r["current"]


def test_put_switches_to_local_and_health_reflects_it(client, fake_http):
    r = client.put("/api/llm", json={"provider": "local", "model": "qwen2.5:7b"})
    assert r.status_code == 200, r.text
    assert r.json()["current"] == {"provider": "local", "model": "qwen2.5:7b",
                                   "base_url": "http://127.0.0.1:11434/v1", "has_key": False}
    assert client.get("/api/health").json()["llm"] == "openai:qwen2.5:7b@http://127.0.0.1:11434/v1"
    assert isinstance(main.orch.llm, OpenAICompatLLM) and main.orch.supervisor.llm is main.orch.llm
    # the probe hit the local server, keyless
    assert fake_http["posts"][-1]["url"].startswith("http://127.0.0.1:11434/v1")
    assert "Authorization" not in fake_http["posts"][-1]["headers"]


def test_put_openai_sends_bearer_and_never_echoes_the_key(client, fake_http):
    r = client.put("/api/llm", json={"provider": "openai", "model": "gpt-4o-mini", "api_key": "sk-abc"})
    assert r.status_code == 200
    assert "sk-abc" not in r.text and r.json()["current"]["has_key"] is True
    assert fake_http["posts"][-1]["headers"]["Authorization"] == "Bearer sk-abc"
    assert fake_http["posts"][-1]["url"] == "https://api.openai.com/v1/chat/completions"
    assert "sk-abc" not in client.get("/api/llm").text


def test_failed_probe_keeps_the_current_llm(client, fake_http):
    fake_http["chat_ok"] = False
    r = client.put("/api/llm", json={"provider": "local", "model": "nope:1b"})
    assert r.status_code == 400 and "did not answer" in _err(r)
    assert isinstance(main.orch.llm, MockLLM)
    assert client.get("/api/llm").json()["current"]["provider"] == "mock"


def test_missing_key_is_a_400(client, fake_http):
    r = client.put("/api/llm", json={"provider": "anthropic", "model": "claude-opus-4-8"})
    assert r.status_code == 400 and "API key" in _err(r)


def test_test_only_probes_without_switching(client, fake_http):
    r = client.put("/api/llm", json={"provider": "local", "model": "qwen2.5:7b", "test_only": True})
    assert r.status_code == 200 and r.json()["tested"]["provider"] == "local"
    assert r.json()["current"]["provider"] == "mock" and isinstance(main.orch.llm, MockLLM)


def test_back_to_mock_needs_no_probe_network(client, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("mock must not call the network")
    monkeypatch.setattr(llm_mod.requests, "post", boom)
    r = client.put("/api/llm", json={"provider": "mock"})
    assert r.status_code == 200 and r.json()["current"]["provider"] == "mock"


def test_switch_requires_the_bearer_token_when_auth_is_on(client, fake_http, monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "api_token", "t0k")
    assert client.put("/api/llm", json={"provider": "local"}).status_code == 401
    assert client.get("/api/llm").status_code == 401
    ok = client.put("/api/llm", json={"provider": "local", "model": "qwen2.5:7b"},
                    headers={"Authorization": "Bearer t0k"})
    assert ok.status_code == 200
