"""MockLLM must profile a dataset once. A clean dataset has quality_flags == []
which is falsy; the old check re-selected profile_pk_dataset every step of the
turn (6 identical "Profiled ..." replies in the chat)."""
from __future__ import annotations

from app.core.llm import MockLLM
from app.tools.builtins import default_registry


def test_clean_profiled_dataset_is_not_profiled_again():
    tools = default_registry().for_agent("data_manager")
    summary = {"dataset_id": "ds_1", "data_quality": []}     # profiled, no flags
    assert MockLLM().select_tool("data_manager", "load dataset", tools, summary) is None


def test_unprofiled_dataset_is_profiled():
    tools = default_registry().for_agent("data_manager")
    summary = {"dataset_id": "ds_1", "data_quality": None}
    choice = MockLLM().select_tool("data_manager", "profile it", tools, summary)
    assert choice and choice["name"] == "profile_pk_dataset"


def test_full_turn_profiles_exactly_once():
    from pathlib import Path

    from app.agents.definitions import AGENTS
    from app.core.audit import AuditChain
    from app.core.pharmstate import PharmState
    from app.tools.base import ToolContext
    sample = str(Path(__file__).parent.parent / "sample_data" / "oral_pk.csv")
    res = AGENTS["data_manager"].run_turn(
        state=PharmState(), message=f"load and profile the dataset {sample}", llm=MockLLM(),
        registry=default_registry(), ctx=ToolContext(), audit=AuditChain(), clock=lambda: "t0")
    names = [c["tool"] for c in res.tool_calls]
    assert names.count("profile_pk_dataset") == 1, names
