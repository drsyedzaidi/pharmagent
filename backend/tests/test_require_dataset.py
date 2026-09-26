"""Tools resolve their DataFrame through one helper. An LLM often echoes a
dataset_id from an earlier turn or invents one; that must fall back to the
session's loaded dataset instead of surfacing a bare KeyError ('ds_1')."""
from __future__ import annotations

import pandas as pd
import pytest

from app.core.pharmstate import PharmState
from app.tools.base import ToolContext, require_dataset


def _ctx() -> ToolContext:
    return ToolContext(dataset_store={"ds_real": pd.DataFrame({"ID": [1], "TIME": [0], "DV": [1.0]})})


def test_explicit_loaded_id_is_used():
    dsid, df = require_dataset(_ctx(), PharmState(dataset_id="ds_real"), {"dataset_id": "ds_real"})
    assert dsid == "ds_real" and len(df) == 1


def test_stale_or_invented_id_falls_back_to_the_session_dataset():
    dsid, _ = require_dataset(_ctx(), PharmState(dataset_id="ds_real"), {"dataset_id": "ds_1"})
    assert dsid == "ds_real"


def test_no_args_uses_the_session_dataset():
    dsid, _ = require_dataset(_ctx(), PharmState(dataset_id="ds_real"))
    assert dsid == "ds_real"


def test_nothing_loaded_is_a_clear_error():
    with pytest.raises(ValueError, match="no dataset loaded"):
        require_dataset(ToolContext(), PharmState(), {"dataset_id": "ds_1"})
    with pytest.raises(ValueError, match="no dataset loaded"):
        require_dataset(_ctx(), PharmState(dataset_id="ds_gone"))


def test_tool_survives_an_echoed_stale_dataset_id():
    from pathlib import Path

    from app.core.llm import MockLLM
    from app.core.orchestrator import Orchestrator
    from app.core.store import SessionStore
    sample = str(Path(__file__).parent.parent / "sample_data" / "oral_pk.csv")
    o = Orchestrator(llm=MockLLM(), store=SessionStore(":memory:"))
    sid = o.create_session().id
    o.chat(sid, f"load and profile the dataset {sample}")
    out = o.run_tool(sid, "fit_compartmental", "compartmental",
                     {"dataset_id": "ds_1", "models": ["1cmt"]})
    assert out["state"]["compartmental_results"]["n_converged"] == 12
