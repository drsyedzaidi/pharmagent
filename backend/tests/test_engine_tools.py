"""Tool-wiring tests for run_engine_comparison — fast (uses the mock engine).

Exercises registration, the guard branches, and the audit-safe payload without a
live external fit.
"""
import json

import numpy as np
import pandas as pd
import pytest

from app.core.pharmstate import AGENT_WRITE_FIELDS, PharmState, apply_writes
from app.tools.base import ToolContext
from app.tools.builtins import default_registry
from app.tools.engine_tools import run_engine_comparison


def _dataset() -> pd.DataFrame:
    rows = []
    for sid in range(1, 5):
        rows.append({"ID": sid, "TIME": 0.0, "DV": np.nan, "AMT": 100.0, "EVID": 1, "CMT": 1})
        for t, c in zip([0.5, 1, 2, 4, 8, 12, 24], [0.8, 1.2, 1.4, 1.2, 0.7, 0.4, 0.1]):
            rows.append({"ID": sid, "TIME": t, "DV": c * (1 + 0.04 * sid),
                         "AMT": np.nan, "EVID": 0, "CMT": 2})
    return pd.DataFrame(rows)


@pytest.fixture
def loaded():
    ctx = ToolContext(dataset_store={"d1": _dataset()})
    state = PharmState(dataset_id="d1")
    return state, ctx


def test_tool_is_registered():
    tool = default_registry().get("run_engine_comparison")
    assert tool.agent == "modeler"
    assert "engine" in tool.description.lower()


def test_state_write_access_includes_field():
    assert "engine_comparison_results" in AGENT_WRITE_FIELDS["modeler"]
    st = apply_writes(PharmState(), "modeler", {"engine_comparison_results": {"status": "ok"}})
    assert st.engine_comparison_results == {"status": "ok"}


def test_no_dataset_is_graceful():
    res = run_engine_comparison(PharmState(), ToolContext(), {"model_key": "oral_1cmt"})
    assert res.writes["engine_comparison_results"]["status"] == "no_model" \
        or res.writes["engine_comparison_results"]["status"] == "no_dataset"


def test_unknown_model_is_graceful(loaded):
    state, ctx = loaded
    res = run_engine_comparison(state, ctx, {"candidates": [{"model_key": "not_a_model"}]})
    assert res.writes["engine_comparison_results"]["status"] == "unknown_model"


def test_runs_writes_and_payload_is_json_safe(loaded):
    state, ctx = loaded
    res = run_engine_comparison(
        state, ctx,
        {"candidates": [{"model_key": "oral_1cmt", "iiv_params": ["CL", "V"]}],
         "engines": ["mock"]})
    payload = res.writes["engine_comparison_results"]
    assert payload["status"] == "ok"
    assert payload["winner"] is not None
    assert payload["prediction_ranking"] and "pred_rmse" in payload["prediction_ranking"][0]
    # winner-choosing metric must be prediction-based, documented as non-OFV
    assert "within-engine only" in payload["note"]
    # the whole payload must be serializable for the audit chain (raw stripped)
    json.dumps(payload)
    assert all("raw" not in r for r in payload["results"])


def test_compare_mode_fallback_recovers_best_model(loaded):
    # A compare-mode pk_model_results carries `best_model`, not `model_key`.
    from app.tools.engine_tools import _resolve_candidates
    state, _ctx = loaded
    state.pk_model_results = {"status": "ok", "mode": "compare",
                             "best_model": "oral_1cmt", "ranking": []}
    specs = _resolve_candidates({}, state)
    assert len(specs) == 1 and specs[0].model_key == "oral_1cmt"


def test_write_access_applies_through_apply_writes(loaded):
    state, ctx = loaded
    res = run_engine_comparison(
        state, ctx,
        {"candidates": [{"model_key": "oral_1cmt", "iiv_params": ["CL", "V"]}], "engines": ["mock"]})
    new_state = apply_writes(state, "modeler", res.writes)
    assert new_state.engine_comparison_results["status"] == "ok"
