"""LLM client.

Three responsibilities only — everything quantitative is a tool:
  1. classify(message, options)  -> route to an agent
  2. select_tool(agent, ...)     -> pick the next tool (or None to stop)
  3. extract_json(system, user, schema) -> one schema-shaped proposal from
     untrusted document text (evidence pipeline), or None; the mock always
     answers None so keyless runs stay on the deterministic path

`RealLLM` uses Anthropic Claude (tool-use). `OpenAICompatLLM` talks to any
OpenAI-compatible chat-completions server with function calling — a FREE,
LOCAL model through Ollama or LM Studio, or a hosted free tier (OpenRouter,
Groq). `MockLLM` is deterministic and keyless, so the platform and the test
suite run with no external calls. The active client is chosen by
configuration (see app.config.Settings.llm_provider_effective).
"""
from __future__ import annotations

import json
import logging
from typing import Any, Protocol

import requests

from app.config import settings
from app.core.llm_mock import MockLLM
from app.tools.base import Tool

log = logging.getLogger("pharmagent")


class LLM(Protocol):
    def classify(self, message: str, options: list[str], descriptions: dict[str, str]) -> str: ...
    def select_tool(self, agent: str, message: str, tools: list[Tool],
                    state_summary: dict[str, Any]) -> dict[str, Any] | None: ...
    def extract_json(self, system: str, user: str, schema: dict[str, Any], *,
                     max_tokens: int = 2048) -> dict[str, Any] | None: ...


EMIT_TOOL = "emit"
EMIT_DESCRIPTION = "Return the answer as a single JSON object matching the schema."

__all__ = ["LLM", "MockLLM", "RealLLM", "OpenAICompatLLM", "get_llm", "is_loopback",
           "LOCAL_REASONING_EFFORT", "EMIT_TOOL", "EMIT_DESCRIPTION"]


class RealLLM:
    """Anthropic Claude client."""

    last_error: str | None = None   # message of the last failed extract_json call

    def __init__(self, *, api_key: str | None = None, model: str | None = None) -> None:
        import anthropic
        self.model = model or settings.model
        self._client = anthropic.Anthropic(api_key=api_key or settings.anthropic_api_key)

    def classify(self, message: str, options: list[str], descriptions: dict[str, str]) -> str:
        roster = "\n".join(f"- {o}: {descriptions.get(o, '')}" for o in options)
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=64,
            system=("You route a pharmacometrics request to ONE specialist agent. "
                    "Reply with only the agent key, nothing else.\n" + roster),
            messages=[{"role": "user", "content": message}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text").strip().lower()
        for o in options:
            if o in text:
                return o
        return options[0] if options else "data_manager"

    def select_tool(self, agent: str, message: str, tools: list[Tool],
                    state_summary: dict[str, Any]) -> dict[str, Any] | None:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=settings.max_tokens,
            system=(f"You are the {agent} agent in a pharmacometrics platform. "
                    "Use a tool to make progress, or answer directly if the task is done. "
                    f"Current state: {state_summary}"),
            tools=[t.to_anthropic() for t in tools],
            messages=[{"role": "user", "content": message}],
        )
        for block in resp.content:
            if block.type == "tool_use":
                return {"name": block.name, "input": block.input}
        return None

    def extract_json(self, system: str, user: str, schema: dict[str, Any], *,
                     max_tokens: int = 2048) -> dict[str, Any] | None:
        """Forced ``emit`` tool call; its input is the schema-shaped answer. Any failure
        (API error, text-only reply, another tool) is None, never an exception; a transport
        failure's message is kept in ``last_error`` so a caller can say why it fell back."""
        self.last_error = None
        try:
            resp = self._client.messages.create(
                model=self.model, max_tokens=max_tokens, system=system,
                tools=[{"name": EMIT_TOOL, "description": EMIT_DESCRIPTION, "input_schema": schema}],
                tool_choice={"type": "tool", "name": EMIT_TOOL},
                messages=[{"role": "user", "content": user}],
            )
        except Exception as exc:  # noqa: BLE001 - the tool falls back deterministically
            self.last_error = str(exc)[:200]
            log.warning("llm_extract_failed", extra={"model": self.model, "error": str(exc)[:200]})
            return None
        for block in resp.content:
            if block.type == "tool_use" and block.name == EMIT_TOOL and isinstance(block.input, dict):
                return block.input
        return None


LOCAL_REASONING_EFFORT = "none"


def is_loopback(url: str) -> bool:
    """True for a server on this machine (Ollama, LM Studio), false for hosted APIs."""
    return "127.0.0.1" in url or "localhost" in url or "[::1]" in url


class OpenAICompatLLM:
    """OpenAI chat-completions client (function calling) over plain HTTP.

    Works unchanged against Ollama (``http://127.0.0.1:11434/v1``, no key),
    LM Studio (``http://127.0.0.1:1234/v1``), OpenRouter, Groq, vLLM, or the
    OpenAI API itself. The model must support tool calling (Ollama: gemma4, qwen3, qwen2.5,
    llama3.1/3.2, mistral-nemo, ...); a model that only answers in prose
    simply selects no tool (the turn ends with an idle hint).

    Only ``requests.post`` is used, so tests fake it without a server.
    """

    TIMEOUT_S = 120.0   # default; settings.llm_timeout_s overrides (large local models)
    last_error: str | None = None   # message of the last failed extract_json call

    def __init__(self, *, base_url: str, model: str, api_key: str | None,
                 reasoning_effort: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        # Sent on ROUTING calls only. "none" switches off hidden reasoning in
        # local thinking models (qwen3 on Ollama): with classify's 32-token
        # budget they spend everything thinking and return empty content. Tool
        # selection keeps the model's default because qwen3 picks the right
        # calculator 6/6 with thinking on but 4/6 with it off (measured
        # 2026-09-30). Never set for hosted OpenAI: it rejects the field for
        # non-reasoning models.
        self.reasoning_effort = reasoning_effort

    # -- transport ----------------------------------------------------------
    def _chat(self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None = None,
              max_tokens: int = 512, quick: bool = False,
              tool_choice: dict[str, Any] | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {"model": self.model, "messages": messages,
                                "max_tokens": max_tokens, "temperature": 0}
        if tools:
            body["tools"] = tools
        if tool_choice:
            body["tool_choice"] = tool_choice
        if quick and self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        resp = requests.post(f"{self.base_url}/chat/completions", json=body, headers=headers,
                             timeout=settings.llm_timeout_s or self.TIMEOUT_S)
        if resp.status_code >= 400:
            raise RuntimeError(self._http_error(resp.status_code))
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"LLM server {self.base_url} returned no choices: {data}")
        return choices[0].get("message") or {}

    def _http_error(self, status: int) -> str:
        """A message that says what to fix, not just the status code."""
        where = f"{self.base_url} (model {self.model!r})"
        if status in (401, 403):
            return f"authentication failed at {where}: check the API key"
        if status == 404:
            return (f"model {self.model!r} not found at {self.base_url}: pull/load it "
                    "(Ollama: `ollama pull <model>`) or pick another name")
        if status == 429:
            return f"rate limit / quota exceeded at {where}"
        if status >= 500:
            return f"LLM server error HTTP {status} at {where}"
        return f"HTTP {status} from {where}; does the model support tool calling?"

    # -- LLM protocol -------------------------------------------------------
    def classify(self, message: str, options: list[str], descriptions: dict[str, str]) -> str:
        roster = "\n".join(f"- {o}: {descriptions.get(o, '')}" for o in options)
        msg = self._chat([
            {"role": "system", "content": (
                "You route a pharmacometrics request to ONE specialist agent. "
                "Reply with only the agent key, nothing else.\n" + roster)},
            {"role": "user", "content": message},
        ], max_tokens=32, quick=True)
        text = (msg.get("content") or "").strip().lower()
        for o in options:
            if o in text:
                return o
        fallback = options[0] if options else "data_manager"
        log.warning("llm_classify_no_match",
                    extra={"model": self.model, "reply": text[:80], "fallback": fallback})
        return fallback

    def select_tool(self, agent: str, message: str, tools: list[Tool],
                    state_summary: dict[str, Any]) -> dict[str, Any] | None:
        msg = self._chat([
            {"role": "system", "content": (
                f"You are the {agent} agent in a pharmacometrics platform. "
                "Call a tool to make progress, or answer in plain text if the task is "
                f"done. Current state: {state_summary}")},
            {"role": "user", "content": message},
        ], tools=[t.to_openai() for t in tools], max_tokens=settings.max_tokens)
        known = {t.name for t in tools}
        for call in msg.get("tool_calls") or []:
            fn = call.get("function") or {}
            name = fn.get("name")
            if name not in known:
                log.warning("llm_unknown_tool", extra={"tool": name, "agent": agent})
                continue
            raw = fn.get("arguments") or {}
            if isinstance(raw, str):
                try:
                    args = json.loads(raw) if raw.strip() else {}
                except json.JSONDecodeError:
                    log.warning("llm_bad_tool_arguments", extra={"tool": name})
                    args = {}
            else:
                args = dict(raw)
            return {"name": name, "input": args if isinstance(args, dict) else {}}
        return None

    def extract_json(self, system: str, user: str, schema: dict[str, Any], *,
                     max_tokens: int = 2048) -> dict[str, Any] | None:
        """Forced ``emit`` function call (``tool_choice``); its arguments are the answer.
        Transport errors, a prose-only reply, another function or unparsable / non-object
        arguments all yield None so the calling tool falls back deterministically (a transport
        failure's message is kept in ``last_error``)."""
        self.last_error = None
        tool = {"type": "function", "function": {"name": EMIT_TOOL, "description": EMIT_DESCRIPTION,
                                                 "parameters": schema}}
        try:
            msg = self._chat([{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                             tools=[tool], max_tokens=max_tokens,
                             tool_choice={"type": "function", "function": {"name": EMIT_TOOL}})
        except (RuntimeError, ValueError, requests.RequestException) as exc:
            self.last_error = str(exc)[:200]
            log.warning("llm_extract_failed", extra={"model": self.model, "error": str(exc)[:200]})
            return None
        for call in msg.get("tool_calls") or []:
            fn = call.get("function") or {}
            if fn.get("name") != EMIT_TOOL:
                continue
            raw = fn.get("arguments")
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except json.JSONDecodeError:
                    log.warning("llm_bad_extract_arguments", extra={"model": self.model})
                    return None
            return raw if isinstance(raw, dict) else None
        return None


def get_llm() -> LLM:
    provider = settings.llm_provider_effective
    if provider == "anthropic":
        return RealLLM()
    if provider == "openai":
        url = settings.llm_base_url or settings.DEFAULT_OLLAMA_URL
        return OpenAICompatLLM(base_url=url, model=settings.model, api_key=settings.llm_api_key,
                               reasoning_effort=LOCAL_REASONING_EFFORT if is_loopback(url) else None)
    return MockLLM()
