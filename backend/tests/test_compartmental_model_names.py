"""fit_compartmental `models` must accept the names an LLM will actually send.

qwen2.5:7b sent ["1-compartment"] (the schema had no enum); the compute layer
knows only "1cmt"/"2cmt", so nothing was fitted and the summary read
"0/12 subjects converged" — a silent no-op dressed as a failed fit.
"""
from __future__ import annotations

import pytest

from app.tools.builtins import default_registry
from app.tools.compartmental_tools import normalize_model_names


@pytest.mark.parametrize("raw,expected", [
    (["1-compartment"], ("1cmt",)),
    (["one-compartment"], ("1cmt",)),
    (["1 compartment", "2 compartment"], ("1cmt", "2cmt")),
    (["1cmt", "2cmt"], ("1cmt", "2cmt")),
    (["Two-Cmt"], ("2cmt",)),
    (["1-cmt oral"], ("1cmt",)),
    ("1cmt", ("1cmt",)),                       # a bare string, not a list
    ([], ("1cmt", "2cmt")),                    # empty -> defaults
    (None, ("1cmt", "2cmt")),
])
def test_names_are_normalized(raw, expected):
    assert normalize_model_names(raw) == expected


def test_unknown_name_is_rejected_loudly():
    with pytest.raises(ValueError, match="1cmt.*2cmt"):
        normalize_model_names(["3-compartment"])


def test_duplicates_collapse():
    assert normalize_model_names(["1cmt", "1-compartment"]) == ("1cmt",)


def test_schema_advertises_the_accepted_values():
    schema = default_registry().get("fit_compartmental").input_schema
    assert schema["properties"]["models"]["items"]["enum"] == ["1cmt", "2cmt"]


def test_tool_fits_with_llm_style_name(monkeypatch):
    from pathlib import Path

    from app.core.llm import MockLLM
    from app.core.orchestrator import Orchestrator
    from app.core.store import SessionStore
    sample = str(Path(__file__).parent.parent / "sample_data" / "oral_pk.csv")
    o = Orchestrator(llm=MockLLM(), store=SessionStore(":memory:"))
    sid = o.create_session().id
    o.chat(sid, f"load and profile the dataset {sample}")
    out = o.run_tool(sid, "fit_compartmental", "compartmental", {"models": ["1-compartment"]})
    r = out["state"]["compartmental_results"]
    assert r["n_converged"] == 12 and r["model_selection_counts"] == {"1cmt": 12}
