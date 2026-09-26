"""LLM client.

Two responsibilities only — everything quantitative is a tool:
  1. classify(message, options)  -> route to an agent
  2. select_tool(agent, ...)     -> pick the next tool (or None to stop)

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
import re
from typing import Any, Protocol

import requests

from app.config import settings
from app.tools.base import Tool

log = logging.getLogger("pharmagent")


class LLM(Protocol):
    def classify(self, message: str, options: list[str], descriptions: dict[str, str]) -> str: ...
    def select_tool(self, agent: str, message: str, tools: list[Tool],
                    state_summary: dict[str, Any]) -> dict[str, Any] | None: ...


_PATH_RE = re.compile(r"[\w./~-]+\.(?:csv|xpt|sas7bdat)", re.IGNORECASE)


class MockLLM:
    """Deterministic, keyless. Drives the core NCA flow heuristically."""

    def classify(self, message: str, options: list[str], descriptions: dict[str, str]) -> str:
        return options[0] if options else "data_manager"

    def select_tool(self, agent: str, message: str, tools: list[Tool],
                    state_summary: dict[str, Any]) -> dict[str, Any] | None:
        s = state_summary
        if agent == "data_manager":
            if not s.get("dataset_id"):
                m = _PATH_RE.search(message or "")
                return {"name": "load_dataset", "input": {"path": m.group(0)}} if m else None
            # `data_quality` in the summary is the quality_flags LIST: an empty
            # list means "profiled, nothing flagged", not "not profiled" — a
            # falsy check here re-profiled clean datasets MAX_TOOL_STEPS times.
            if s.get("data_quality") is None:
                return {"name": "profile_pk_dataset", "input": {}}
            return None
        if agent == "nca":
            return None if s.get("nca_parameters") else {"name": "compute_nca", "input": {}}
        if agent == "be":
            return None if s.get("be_results") else {"name": "run_bioequivalence", "input": {}}
        if agent == "dose_prop":
            return None if s.get("dose_prop_results") else {"name": "run_dose_proportionality", "input": {}}
        if agent == "compartmental":
            return None if s.get("compartmental_results") else {"name": "fit_compartmental", "input": {}}
        if agent == "poppk":
            return None if s.get("poppk_results") else {"name": "run_poppk", "input": {}}
        if agent == "modeler":
            return None if s.get("pk_model_results") else {"name": "fit_pk_model", "input": {"compare": True}}
        if agent == "qc":
            return None if s.get("qc_verdict") else {"name": "run_qc", "input": {}}
        if agent == "report":
            return None if s.get("report_path") else {"name": "generate_report", "input": {}}
        if agent == "statistician":
            return None if s.get("stats_advice") else {"name": "recommend_statistics", "input": {}}
        if agent == "simulator":
            low = (message or "").lower()
            # Every choice here is a PROPOSAL (expensive+proposable tools): the
            # keyless mock never confirms on the human's behalf and cannot
            # compose a `design`/`params` from prose -- the human fills/approves.
            if any(k in low for k in ("simulation-estimation", "simulation estimation",
                                      "simest", "sim-est")):
                return {"name": "run_simest", "input": {}}
            if "bootstrap" in low:
                return {"name": "run_bootstrap", "input": {}}
            if any(k in low for k in ("importance resampling", "sampling importance", " sir")):
                return {"name": "run_sir", "input": {}}
            if any(k in low for k in ("likelihood profil", "profile likelihood")):
                return {"name": "run_profile", "input": {}}
            return None
        return None


class RealLLM:
    """Anthropic Claude client."""

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


class OpenAICompatLLM:
    """OpenAI chat-completions client (function calling) over plain HTTP.

    Works unchanged against Ollama (``http://127.0.0.1:11434/v1``, no key),
    LM Studio (``http://127.0.0.1:1234/v1``), OpenRouter, Groq, vLLM, or the
    OpenAI API itself. The model must support tool calling (Ollama: qwen2.5,
    llama3.1/3.2, mistral-nemo, ...); a model that only answers in prose
    simply selects no tool (the turn ends with an idle hint).

    Only ``requests.post`` is used, so tests fake it without a server.
    """

    TIMEOUT_S = 120.0   # local 7B models on CPU can take a while per call

    def __init__(self, *, base_url: str, model: str, api_key: str | None) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key

    # -- transport ----------------------------------------------------------
    def _chat(self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None = None,
              max_tokens: int = 512) -> dict[str, Any]:
        body: dict[str, Any] = {"model": self.model, "messages": messages,
                                "max_tokens": max_tokens, "temperature": 0}
        if tools:
            body["tools"] = tools
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        resp = requests.post(f"{self.base_url}/chat/completions", json=body, headers=headers,
                             timeout=self.TIMEOUT_S)
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
        ], max_tokens=32)
        text = (msg.get("content") or "").strip().lower()
        for o in options:
            if o in text:
                return o
        return options[0] if options else "data_manager"

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


def get_llm() -> LLM:
    provider = settings.llm_provider_effective
    if provider == "anthropic":
        return RealLLM()
    if provider == "openai":
        return OpenAICompatLLM(base_url=settings.llm_base_url or settings.DEFAULT_OLLAMA_URL,
                               model=settings.model, api_key=settings.llm_api_key)
    return MockLLM()
