"""Free / local LLM support: an OpenAI-compatible chat-completions adapter
(Ollama, LM Studio, OpenRouter, Groq, ...) selectable by configuration.

No network: `requests.post` is monkeypatched with a fake that records the
request and returns a canned OpenAI-shaped response.
"""
from __future__ import annotations

import json

import pytest

from app.core import llm as llm_mod
from app.core.llm import MockLLM, OpenAICompatLLM, get_llm
from app.tools.base import Tool
from app.tools.builtins import default_registry


class _Resp:
    def __init__(self, payload: dict, status: int = 200) -> None:
        self._payload, self.status_code = payload, status

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


@pytest.fixture
def fake_post(monkeypatch):
    calls: list[dict] = []
    replies: list[dict] = []

    def post(url, json=None, headers=None, timeout=None):  # noqa: A002 - mirrors requests.post
        calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        return _Resp(replies.pop(0) if replies else {"choices": [{"message": {"content": ""}}]})

    monkeypatch.setattr(llm_mod.requests, "post", post)
    return calls, replies


def _chat(content: str | None = None, tool_calls: list | None = None) -> dict:
    msg: dict = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {"choices": [{"message": msg, "finish_reason": "tool_calls" if tool_calls else "stop"}]}


def _tc(name: str, args: dict) -> dict:
    return {"id": "call_1", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


# ── wire format ─────────────────────────────────────────────────────────────

def test_tool_to_openai_shape():
    t = default_registry().get("compute_nca")
    o = t.to_openai()
    assert o["type"] == "function"
    assert o["function"]["name"] == "compute_nca"
    assert o["function"]["parameters"] == t.input_schema
    assert o["function"]["description"] == t.description


def test_select_tool_posts_openai_chat_completion_with_tools(fake_post):
    calls, replies = fake_post
    replies.append(_chat(tool_calls=[_tc("compute_nca", {"dose_group": "all"})]))
    llm = OpenAICompatLLM(base_url="http://127.0.0.1:11434/v1", model="qwen2.5:7b", api_key=None)
    tools = default_registry().for_agent("nca")
    out = llm.select_tool("nca", "compute NCA", tools, {"dataset_id": "ds_1"})

    assert out == {"name": "compute_nca", "input": {"dose_group": "all"}}
    req = calls[0]
    assert req["url"] == "http://127.0.0.1:11434/v1/chat/completions"
    body = req["json"]
    assert body["model"] == "qwen2.5:7b"
    assert body["tools"] == [t.to_openai() for t in tools]
    assert body["messages"][0]["role"] == "system" and "nca" in body["messages"][0]["content"]
    assert body["messages"][-1] == {"role": "user", "content": "compute NCA"}
    assert "Authorization" not in (req["headers"] or {})     # keyless local server


def test_api_key_goes_in_bearer_header(fake_post):
    calls, replies = fake_post
    replies.append(_chat(content="data_manager"))
    OpenAICompatLLM(base_url="https://openrouter.ai/api/v1", model="m", api_key="sk-x").classify(
        "load csv", ["data_manager", "nca"], {})
    assert calls[0]["headers"]["Authorization"] == "Bearer sk-x"


def test_text_only_reply_means_no_tool(fake_post):
    _, replies = fake_post
    replies.append(_chat(content="All done, nothing more to run."))
    llm = OpenAICompatLLM(base_url="http://x/v1", model="m", api_key=None)
    assert llm.select_tool("nca", "x", default_registry().for_agent("nca"), {}) is None


def test_malformed_tool_arguments_do_not_crash_the_turn(fake_post):
    _, replies = fake_post
    replies.append({"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
        {"id": "c", "type": "function", "function": {"name": "compute_nca", "arguments": "{not json"}}]}}]})
    llm = OpenAICompatLLM(base_url="http://x/v1", model="m", api_key=None)
    out = llm.select_tool("nca", "x", default_registry().for_agent("nca"), {})
    assert out == {"name": "compute_nca", "input": {}}


def test_unknown_tool_name_is_ignored(fake_post):
    _, replies = fake_post
    replies.append(_chat(tool_calls=[_tc("rm_rf", {})]))
    llm = OpenAICompatLLM(base_url="http://x/v1", model="m", api_key=None)
    assert llm.select_tool("nca", "x", default_registry().for_agent("nca"), {}) is None


def test_classify_matches_an_option_and_falls_back_to_first(fake_post):
    _, replies = fake_post
    llm = OpenAICompatLLM(base_url="http://x/v1", model="m", api_key=None)
    replies.append(_chat(content="I think: NCA"))
    assert llm.classify("auc", ["data_manager", "nca"], {"nca": "..."}) == "nca"
    replies.append(_chat(content="no idea"))
    assert llm.classify("auc", ["data_manager", "nca"], {}) == "data_manager"


@pytest.mark.parametrize("status,needle", [
    (401, "check the API key"), (403, "check the API key"),
    (404, "not found"), (429, "quota"), (500, "server error"), (422, "tool calling"),
])
def test_http_errors_say_what_to_fix(monkeypatch, status, needle):
    def post(url, json=None, headers=None, timeout=None):  # noqa: A002
        return _Resp({"error": "boom"}, status=status)
    monkeypatch.setattr(llm_mod.requests, "post", post)
    llm = OpenAICompatLLM(base_url="http://x/v1", model="m", api_key=None)
    with pytest.raises(RuntimeError, match=needle):
        llm.classify("x", ["a"], {})


# ── provider selection ──────────────────────────────────────────────────────

def _settings(monkeypatch, **kw):
    from app.config import settings
    defaults = {"llm_provider": "auto", "llm_base_url": None, "llm_api_key": None,
                "anthropic_api_key": None, "use_mock_llm": False, "model": "m"}
    defaults.update(kw)
    for k, v in defaults.items():
        monkeypatch.setattr(settings, k, v)
    return settings


def test_auto_is_mock_without_any_key_or_url(monkeypatch):
    s = _settings(monkeypatch)
    assert s.llm_provider_effective == "mock" and isinstance(get_llm(), MockLLM)


def test_auto_picks_openai_compat_when_a_base_url_is_set(monkeypatch):
    s = _settings(monkeypatch, llm_base_url="http://127.0.0.1:11434/v1", model="qwen2.5:7b")
    assert s.llm_provider_effective == "openai"
    llm = get_llm()
    assert isinstance(llm, OpenAICompatLLM) and llm.model == "qwen2.5:7b"
    assert llm.base_url == "http://127.0.0.1:11434/v1"


def test_explicit_openai_provider_defaults_to_local_ollama_url(monkeypatch):
    s = _settings(monkeypatch, llm_provider="openai")
    assert s.llm_provider_effective == "openai"
    assert get_llm().base_url == "http://127.0.0.1:11434/v1"


def test_explicit_mock_overrides_a_configured_url(monkeypatch):
    s = _settings(monkeypatch, llm_provider="mock", llm_base_url="http://x/v1")
    assert s.llm_provider_effective == "mock" and s.llm_is_mock


def test_anthropic_key_still_wins_in_auto(monkeypatch):
    s = _settings(monkeypatch, anthropic_api_key="sk-ant", llm_base_url="http://x/v1")
    assert s.llm_provider_effective == "anthropic"


def test_trailing_slash_on_base_url_is_tolerated(fake_post):
    calls, replies = fake_post
    replies.append(_chat(content="a"))
    OpenAICompatLLM(base_url="http://x/v1/", model="m", api_key=None).classify("q", ["a"], {})
    assert calls[0]["url"] == "http://x/v1/chat/completions"


def test_tool_protocol_still_satisfied():
    t = Tool("t", "d", "nca", {"type": "object", "properties": {}}, lambda s, c, a: None)
    assert t.to_openai()["function"]["parameters"] == {"type": "object", "properties": {}}
