"""Keyless chat (MockLLM) reaches the briefing memo, the exposure-response tools and the
statistician's indirect-comparison hint. The desktop app ships this path: with no
key, each request must reach its tool, or stop with a message naming what is missing.
The mock never confirms an expensive tool."""
from __future__ import annotations

import itertools

import pytest

from app.core.llm import MockLLM
from app.core.orchestrator import Orchestrator
from app.core.store import SessionStore
from tests.memo_fixtures import full_state
from tests.test_er_tools import LOGIT_ARGS, TOX_ARGS, _frame


@pytest.fixture
def orch(tmp_path, monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    return Orchestrator(llm=MockLLM(), clock=lambda c=itertools.count(): f"t{next(c)}",
                        store=SessionStore(":memory:"))


def _tools(out) -> list[str]:
    return [c["tool"] for c in out["tool_calls"]]


def _with_data(orch, fits: bool = False) -> str:
    sid = orch.create_session().id
    sess = orch.get_session(sid)
    sess.ctx.dataset_store["d1"] = _frame()
    sess.state = sess.state.model_copy(update={"dataset_id": "d1"})
    if fits:
        for args in (LOGIT_ARGS, TOX_ARGS):
            orch.run_tool(sid, "fit_exposure_response", "er_dose", args, actor="alice")
    return sid


# ── report ─────────────────────────────────────────────────────────────────────
def test_a_memo_request_builds_the_briefing_memo_once(orch):
    sid = orch.create_session().id
    orch.get_session(sid).state = full_state().model_copy(update={"session_id": sid})
    out = orch.chat(sid, "write a briefing memo", actor="alice")
    assert out["agent"] == "report" and _tools(out) == ["build_briefing_memo"]
    assert out["state"]["memo_path"] and "Built briefing memo" in out["messages"][0]
    again = orch.chat(sid, "update the briefing memo please", actor="alice")
    assert _tools(again) == []                                     # a memo exists: nothing re-run


def test_a_plain_report_request_still_generates_the_docx_report(orch):
    sid = orch.create_session().id
    orch.get_session(sid).state = full_state().model_copy(update={"session_id": sid})
    out = orch.chat(sid, "generate the report docx", actor="alice")
    assert out["agent"] == "report" and _tools(out) == ["generate_report"]


# ── exposure-response ──────────────────────────────────────────────────────────
def test_an_er_fit_request_reaches_the_fit_tool_which_names_the_missing_columns(orch):
    sid = _with_data(orch)
    out = orch.chat(sid, "fit a logistic exposure-response model for the adverse event", actor="alice")
    assert out["agent"] == "er_dose" and _tools(out) == ["fit_exposure_response"]
    assert "column" in out["messages"][0].lower()                 # the refusal names what to supply
    assert out["state"]["er_results"] is None


def test_an_er_fit_request_with_a_stored_fit_runs_nothing_and_lists_the_fits(orch):
    sid = _with_data(orch, fits=True)
    out = orch.chat(sid, "Cox hazard ratio for exposure-response", actor="alice")
    assert out["agent"] == "er_dose" and _tools(out) == []
    assert "efficacy" in out["messages"][0] and "toxicity" in out["messages"][0]


def test_a_dose_selection_request_reaches_select_optimal_dose(orch):
    sid = _with_data(orch)
    out = orch.chat(sid, "select the optimal dose with Project Optimus", actor="alice")
    # a refusal writes nothing, so the stateless mock re-picks it once and the loop's repeat guard stops
    assert out["agent"] == "er_dose" and _tools(out)[0] == "select_optimal_dose" and len(_tools(out)) <= 2
    assert "missing E-R fit" in out["messages"][0]                 # the documented refusal: no fits yet
    with_fits = _with_data(orch, fits=True)
    out = orch.chat(with_fits, "select the optimal dose with Project Optimus", actor="alice")
    assert _tools(out)[0] == "select_optimal_dose" and out["state"]["dose_selection_results"] is None
    assert "exposure_mapping" in out["messages"][0]                # names the input the mock cannot supply


def test_a_bootstrap_request_is_only_proposed_never_confirmed(orch):
    sid = _with_data(orch, fits=True)
    out = orch.chat(sid, "bootstrap the exposure-response logistic model", actor="alice")
    assert out["agent"] == "er_dose"
    assert out["pending_tool"]["tool"] == "bootstrap_exposure_response"
    assert "confirm" not in out["pending_tool"]["args"]
    assert "bootstrap" not in out["state"]["er_results"]["fits"]["efficacy"]   # nothing computed


def test_mock_skips_dose_selection_when_one_exists():
    choice = MockLLM().select_tool("er_dose", "optimal dose", [], {"dose_selection": "present"})
    assert choice is None


# ── statistician ───────────────────────────────────────────────────────────────
def test_an_indirect_comparison_request_names_the_inputs_instead_of_failing(orch):
    sid = orch.create_session().id
    out = orch.chat(sid, "Bucher indirect comparison of A vs C through placebo", actor="alice")
    assert out["agent"] == "statistician" and _tools(out) == []
    msg = out["messages"][0]
    assert "no dataset loaded" not in msg and "both direct effects" in msg and "HR" in msg
