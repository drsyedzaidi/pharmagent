"""Process-wide handle on the orchestrator's current LLM for the evidence tools.

Tools receive no LLM through ``ToolContext``; the orchestrator calls ``set_llm`` from
``__init__`` and ``set_llm`` (UI provider switch). ``llm_available`` is False for the
keyless mock and for any client lacking ``extract_json``, which keeps every tool on its
deterministic path (keyword screening, table extraction) in tests and keyless demos.
"""
from __future__ import annotations

from typing import Any

from app.core.llm import MockLLM

_llm: Any | None = None


def set_llm(llm: Any | None) -> None:
    global _llm
    _llm = llm


def current_llm() -> Any | None:
    return _llm


def llm_available() -> bool:
    """True only for a real client that implements ``extract_json``."""
    llm = _llm
    return llm is not None and not isinstance(llm, MockLLM) and callable(getattr(llm, "extract_json", None))
