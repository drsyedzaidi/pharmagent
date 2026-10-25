"""F-02: PUT /api/llm must never send a held API key to a caller-chosen URL,
and must refuse private/loopback/link-local targets that are not allowlisted."""
from __future__ import annotations

import itertools

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.core import llm as llm_mod
from app.core import llm_config
from app.core.llm import MockLLM
from app.core.llm_config import LlmChoice, LlmManager
from app.core.orchestrator import Orchestrator
from app.core.store import SessionStore

EVIL = "https://evil.example.com/v1"


class _Resp:
    status_code = 200

    def json(self):
        return {"choices": [{"message": {"role": "assistant", "content": "nca"}}]}


@pytest.fixture
def env(monkeypatch, tmp_path):
    from app.config import settings
    for k, v in {"llm_provider": "auto", "llm_base_url": None, "llm_api_key": None,
                 "anthropic_api_key": None, "use_mock_llm": False, "model": "m",
                 "api_token": None}.items():
        monkeypatch.setattr(settings, k, v)
    monkeypatch.delenv("PHARMAGENT_LLM_BASE_URL_ALLOWLIST", raising=False)
    # "evil.example.com" resolves to a public address; no real DNS in tests
    monkeypatch.setattr(llm_config, "_resolve_host", lambda host: ["93.184.216.34"])
    posts: list[dict] = []

    def post(url, json=None, headers=None, timeout=None):  # noqa: A002
        posts.append({"url": url, "headers": headers or {}})
        return _Resp()

    monkeypatch.setattr(llm_mod.requests, "post", post)
    main.orch = Orchestrator(llm=MockLLM(), clock=lambda c=itertools.count(): f"t{next(c)}",
                             store=SessionStore(":memory:"))
    main.llm_manager = LlmManager(tmp_path)
    return TestClient(main.app), posts, settings


def _hold_openai_key(settings, monkeypatch=None, key="sk-held"):
    main.llm_manager.choice = LlmChoice("openai", "gpt-4o-mini", None, key)


def test_stored_key_not_reused_for_new_base_url(env):
    _, _, settings = env
    _hold_openai_key(settings)
    with pytest.raises(ValueError, match="API key"):
        main.llm_manager.resolve({"provider": "openai", "base_url": EVIL})


def test_env_key_not_reused_for_new_base_url(env, monkeypatch):
    _, _, settings = env
    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(settings, "llm_api_key", "sk-env")
    with pytest.raises(ValueError, match="API key"):
        main.llm_manager.resolve({"provider": "openai", "base_url": EVIL})


def test_probe_never_sees_reused_key_for_changed_url(env, monkeypatch):
    client, posts, settings = env
    _hold_openai_key(settings)
    seen: list[LlmChoice] = []
    monkeypatch.setattr(main, "probe", lambda llm: seen.append(llm) or {"ok": True, "detail": "x"})
    r = client.put("/api/llm", json={"provider": "openai", "base_url": EVIL, "test_only": True})
    assert r.status_code == 400
    assert seen == [] and posts == []


def test_caller_supplied_key_is_accepted_for_new_url(env):
    _, _, settings = env
    _hold_openai_key(settings)
    c = main.llm_manager.resolve({"provider": "openai", "base_url": EVIL, "api_key": "sk-new"})
    assert c.api_key == "sk-new"


def test_same_origin_reuses_stored_key_despite_formatting(env):
    _, _, settings = env
    _hold_openai_key(settings)
    for url in ("", "https://API.openai.com/v1", "https://api.openai.com:443/v1/"):
        assert main.llm_manager.resolve({"provider": "openai", "base_url": url}).api_key == "sk-held"


@pytest.mark.parametrize("url", [
    "http://169.254.169.254/latest", "http://10.0.0.5:8080/v1", "http://127.0.0.1:9999/v1",
    "http://localhost:8000/v1", "http://[::1]:8000/v1", "file:///etc/passwd", "ftp://x.example.com",
    "http://user:pw@api.openai.com/v1",
])
def test_disallowed_target_rejected_with_400(env, url):
    client, posts, _ = env
    r = client.put("/api/llm", json={"provider": "local", "base_url": url, "test_only": True})
    assert r.status_code == 400 and posts == []


def test_hostname_resolving_to_private_address_rejected(env, monkeypatch):
    client, posts, _ = env
    monkeypatch.setattr(llm_config, "_resolve_host", lambda host: ["10.1.2.3"])
    r = client.put("/api/llm", json={"provider": "local", "base_url": "http://rebind.example.com/v1",
                                     "test_only": True})
    assert r.status_code == 400 and posts == []


def test_allowlist_env_permits_private_target(env, monkeypatch):
    client, posts, _ = env
    monkeypatch.setenv("PHARMAGENT_LLM_BASE_URL_ALLOWLIST", "http://10.0.0.5:8080, http://lan.test:1234")
    r = client.put("/api/llm", json={"provider": "local", "model": "q", "base_url": "http://10.0.0.5:8080/v1",
                                     "test_only": True})
    assert r.status_code == 200 and posts[0]["url"].startswith("http://10.0.0.5:8080")


def test_default_local_and_openai_flows_still_resolve(env, monkeypatch):
    _, _, settings = env
    m = main.llm_manager
    assert m.resolve({"provider": "local", "model": "q"}).effective_base_url == settings.DEFAULT_OLLAMA_URL
    assert m.resolve({"provider": "local", "model": "q", "base_url": "http://localhost:11434/v1"})
    assert m.resolve({"provider": "openai", "api_key": "sk-x"}).effective_base_url == llm_config.OPENAI_API_URL
    monkeypatch.setattr(settings, "llm_base_url", "http://gpu.lan:8000/v1")
    monkeypatch.setattr(llm_config, "_resolve_host", lambda host: ["192.168.1.9"])
    assert m.resolve({"provider": "local", "model": "q", "base_url": "http://gpu.lan:8000/v1"})
