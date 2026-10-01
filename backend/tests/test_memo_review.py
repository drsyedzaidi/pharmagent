"""Reviewer hook: the adversarial reviewer re-derives the memo's value table from the
CURRENT state and flags any number in the stored memo text that state no longer
accounts for (an edited memo, or an analysis re-run after the memo was built)."""
import itertools
from pathlib import Path

import pytest

from app.compute import adversarial
from app.core.llm import MockLLM
from app.core.orchestrator import Orchestrator
from app.core.pharmstate import PharmState
from app.core.store import SessionStore
from app.tools.base import ToolContext
from app.tools.builtins import default_registry
from tests.memo_fixtures import full_chain, full_state

THEOPH = str(Path(__file__).parent.parent / "sample_data" / "theoph_pk.csv")


def _state_with_memo(tmp_path) -> PharmState:
    state, _ = default_registry().execute(
        "build_briefing_memo", state=full_state(), ctx=ToolContext(data_dir=str(tmp_path)),
        args={}, audit=full_chain(), timestamp="t", actor="u")
    return state


def _memo_findings(state: PharmState):
    r = adversarial.review(state.model_dump(), None, None)
    return r, [f for f in r["findings"] if f["id"].startswith("memo-")]


def test_a_memo_built_from_the_current_state_raises_no_finding(tmp_path):
    r, found = _memo_findings(_state_with_memo(tmp_path))
    assert found == [] and r["checked"]["memo"] is True


def test_no_memo_means_nothing_to_check():
    r = adversarial.review(full_state().model_dump(), None, None)
    assert r["checked"]["memo"] is False
    assert not [f for f in r["findings"] if f["id"].startswith("memo-")]


def test_an_injected_number_is_a_high_finding_that_blocks_the_goal(tmp_path):
    state = _state_with_memo(tmp_path)
    memo = {**state.memo_results, "memo_text": state.memo_results["memo_text"] + "\nCL/F rose to 7.77."}
    r, found = _memo_findings(state.model_copy(update={"memo_results": memo}))
    assert [f["id"] for f in found] == ["memo-untraced-numbers"]
    f = found[0]
    assert f["severity"] == "HIGH" and "7.77" in f["evidence"]
    assert r["goal_met"] is False and r["counts"]["HIGH"] >= 1


def test_a_stale_memo_is_flagged_when_the_analysis_changes_afterwards(tmp_path):
    state = _state_with_memo(tmp_path)
    # -234.5 is ALSO the SCM base OFV, so plain membership would still excuse the old figure;
    # the pair-by-pair manifest comparison is what catches this
    nl = {**state.nlme_results, "ofv": -999.9}
    r, found = _memo_findings(state.model_copy(update={"nlme_results": nl}))
    assert [f["id"] for f in found] == ["memo-stale"]
    assert "nlme_results.ofv=-234.5" in found[0]["evidence"]
    assert found[0]["severity"] == "HIGH" and r["goal_met"] is False
    assert "re-run" in found[0]["claim"] + found[0]["suggested_action"]


def test_a_stale_number_that_appears_nowhere_else_is_also_untraced(tmp_path):
    state = _state_with_memo(tmp_path)
    nl = {**state.nlme_results, "condition_number": 77.7}       # 38.2 occurs only here
    _, found = _memo_findings(state.model_copy(update={"nlme_results": nl}))
    assert sorted(f["id"] for f in found) == ["memo-stale", "memo-untraced-numbers"]
    assert "38.2" in next(f for f in found if f["id"] == "memo-untraced-numbers")["evidence"]


def test_a_memo_whose_state_was_cleared_is_flagged_not_crashed(tmp_path):
    state = _state_with_memo(tmp_path)
    r, found = _memo_findings(PharmState(memo_results=state.memo_results))
    assert sorted(f["id"] for f in found) == ["memo-stale", "memo-untraced-numbers"]


@pytest.mark.parametrize("memo", [{"status": "error"}, {"status": "ok"},
                                  {"status": "ok", "memo_text": 5}, {}])
def test_malformed_or_absent_memo_results_are_ignored(memo):
    r = adversarial.review(PharmState(memo_results=memo or None).model_dump(), None, None)
    assert not [f for f in r["findings"] if f["id"].startswith("memo-")]


def test_review_loop_end_to_end_clean_then_tampered(tmp_path, monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    counter = itertools.count()
    orch = Orchestrator(llm=MockLLM(), clock=lambda: f"t{next(counter)}", store=SessionStore(":memory:"))
    sid = orch.create_session().id
    orch.start_workflow(sid, "nca_full", {"path": THEOPH})
    orch.run_tool(sid, "build_briefing_memo", "report", {})

    clean = orch.review_loop(sid)
    assert not [f for f in clean["findings"] if f["id"].startswith("memo-")]
    assert clean["checked"]["memo"] is True

    sess = orch.get_session(sid)
    memo = {**sess.state.memo_results, "memo_text": sess.state.memo_results["memo_text"] + "\nAUCinf 555.5"}
    sess.state = sess.state.model_copy(update={"memo_results": memo})
    dirty = orch.review_loop(sid)
    assert [f["id"] for f in dirty["findings"] if f["id"].startswith("memo-")] == ["memo-untraced-numbers"]
    assert dirty["goal_met"] is False


def test_memo_alone_never_counts_as_having_checked_anything():
    """A memo with no analysis behind it must stay INCOMPLETE, not 'GOAL MET'."""
    memo = {"status": "ok", "memo_text": "no numbers here", "value_table": []}
    r = adversarial.review(PharmState(memo_results=memo).model_dump(), None, None)
    assert r["checked"]["memo"] is True and r["nothing_checked"] is True
    assert r["status"] == "INCOMPLETE" and r["goal_met"] is False


def test_corrupt_state_with_a_memo_present_is_a_finding_not_a_crash(tmp_path):
    state = _state_with_memo(tmp_path)
    broken = state.model_copy(update={"forest_results": {"status": "ok", "rows": ["a"]}})
    r, found = _memo_findings(broken)
    assert [f["id"] for f in found] == ["memo-unverifiable"] and r["goal_met"] is False
