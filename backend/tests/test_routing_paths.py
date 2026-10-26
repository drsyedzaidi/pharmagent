"""File paths and names are arguments, not intent: their words must not route to an agent."""
import pytest

from app.agents.supervisor import Supervisor, score
from app.core.llm import MockLLM

SUP = Supervisor(MockLLM())


@pytest.mark.parametrize("path", [
    "/home/u/pharmagent-be-qc/data/oral_pk.csv",
    "/data/qc_results.csv",
    "C:\\studies\\review\\report_final.csv",
    "~/checklist/diagnostic.csv",
])
def test_path_words_add_no_agent_to_load_plan(path):
    plan = SUP.route_plan(f"load {path}")
    assert [a for a, _ in plan] == ["data_manager"]


@pytest.mark.parametrize("path", ["/x/qc_data.csv", "/review/a.csv", "data/report.csv"])
def test_path_words_score_nothing_for_other_agents(path):
    s = score(f"load {path}")
    assert s["qc"] == 0 and s["report"] == 0


def test_real_intent_beside_a_path_still_routes():
    plan = SUP.route_plan("load /data/study.csv then run the QC checklist")
    assert [a for a, _ in plan] == ["data_manager", "qc"]
    assert score("run QC on /data/study.csv")["qc"] >= 1
