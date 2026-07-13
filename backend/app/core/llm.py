"""LLM client.

Two responsibilities only — everything quantitative is a tool:
  1. classify(message, options)  -> route to an agent
  2. select_tool(agent, ...)     -> pick the next tool (or None to stop)

`RealLLM` uses Anthropic Claude (tool-use). `MockLLM` is deterministic and
keyless, so the platform and the test suite run with no external calls. The
active client is chosen automatically by configuration.
"""
from __future__ import annotations

import re
from typing import Any, Protocol

from app.config import settings
from app.tools.base import Tool


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
            if not s.get("data_quality"):
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
        return None


class RealLLM:
    """Anthropic Claude client."""

    def __init__(self) -> None:
        import anthropic
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    def classify(self, message: str, options: list[str], descriptions: dict[str, str]) -> str:
        roster = "\n".join(f"- {o}: {descriptions.get(o, '')}" for o in options)
        resp = self._client.messages.create(
            model=settings.model,
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
            model=settings.model,
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


def get_llm() -> LLM:
    return MockLLM() if settings.llm_is_mock else RealLLM()
