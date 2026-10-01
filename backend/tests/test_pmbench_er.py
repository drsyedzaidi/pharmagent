"""PharmacometricsBench exposure-response pack (opt-in; the frozen v0 set is untouched).

Ground truth for every task is the output of the validated compute function on the
generated data (logistic OR, KM median, Cox HR, Optimus dose selection), so an
agent that calls the tool scores 1.0 and one that free-hands the statistic does not.
"""
import json

import pytest

from pharmacometricsbench.agents import naive_agent, oracle_agent
from pharmacometricsbench.generators import build_er_taskset, build_taskset
from pharmacometricsbench.grading import grade_task, score_report
from pharmacometricsbench.llm import MockLLM, make_llm_agent
from pharmacometricsbench.tool_agent import TOOLS, execute_tool

KINDS = {"logistic_or", "km_median", "cox_hr", "optimal_dose"}


@pytest.fixture(scope="module")
def tasks():
    return build_er_taskset()


def _report(agent, tasks):
    return score_report([grade_task(t, agent(t)) for t in tasks])


def test_pack_has_one_task_per_analysis_and_is_reproducible(tasks):
    assert len(tasks) == 4 and {t.category for t in tasks} == {"er"}
    assert {t.dataset["analysis"] for t in tasks} == KINDS
    assert [t.to_dict() for t in tasks] == [t.to_dict() for t in build_er_taskset()]
    for t in tasks:
        assert t.targets and t.prompt and t.oracle.startswith("app.compute.")


def test_frozen_v0_set_is_unchanged_unless_the_pack_is_requested():
    assert len(build_taskset(per_category=6)) == 30
    assert "er" not in {t.category for t in build_taskset(per_category=6)}
    with_er = build_taskset(per_category=6, include_er=True)
    assert len(with_er) == 34 and sum(t.category == "er" for t in with_er) == 4
    assert [t.to_dict() for t in with_er[:30]] == [t.to_dict() for t in build_taskset(per_category=6)]


def test_oracle_agent_scores_perfect(tasks):
    report = _report(oracle_agent, tasks)
    assert report["overall"] == 1.0, report


def test_naive_agent_is_discriminated(tasks):
    assert _report(naive_agent, tasks)["overall"] <= _report(oracle_agent, tasks)["overall"] - 0.15


def test_llm_round_trip_is_lossless_for_the_oracle_strategy(tasks):
    agent = make_llm_agent(MockLLM(strategy="oracle"))
    assert _report(agent, tasks)["overall"] == 1.0


def test_targets_are_the_headline_statistics(tasks):
    names = {t.dataset["analysis"]: {x.name for x in t.targets} for t in tasks}
    assert {"or_per_unit", "or_per_sd"} <= names["logistic_or"]
    assert "km_median" in names["km_median"]
    assert {"hr_per_unit", "hr_per_sd"} <= names["cox_hr"]
    assert {"selected_dose", "utility_at_selected"} <= names["optimal_dose"]


def test_the_tool_using_agent_can_reach_the_pack_through_a_tool(tasks):
    assert "exposure_response" in {t["name"] for t in TOOLS}
    for task in tasks:
        out = execute_tool("exposure_response", json.loads(json.dumps(task.dataset)))
        assert "error" not in out, out
        graded = grade_task(task, out)
        assert graded["score"] == 1.0, (task.task_id, graded)
