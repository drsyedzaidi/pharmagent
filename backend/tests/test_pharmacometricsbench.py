"""PharmacometricsBench v0 harness self-test + regression gate.

The oracle agent calls the validated compute tools, so it must reproduce ground
truth exactly and score a perfect 1.0 on every category. If this ever drops, a
task is malformed or a tool's behaviour changed — either way it is caught here.
The naive (tool-free) agent must score materially lower, proving the benchmark
discriminates correct process from eyeballing.
"""
import pytest

from pharmacometricsbench.agents import naive_agent, oracle_agent
from pharmacometricsbench.generators import build_taskset
from pharmacometricsbench.grading import grade_task, score_report, within_tolerance
from pharmacometricsbench.spec import Target


@pytest.fixture(scope="module")
def tasks():
    return build_taskset(per_category=6)


def _report(agent, tasks):
    return score_report([grade_task(t, agent(t)) for t in tasks])


def test_taskset_shape(tasks):
    assert len(tasks) == 30
    cats = {t.category for t in tasks}
    assert cats == {"nca", "be", "dp", "compartmental", "exposure"}
    for t in tasks:
        assert t.targets and t.prompt and t.oracle


def test_oracle_agent_scores_perfect(tasks):
    report = _report(oracle_agent, tasks)
    assert report["overall"] == 1.0, report
    for cat, score in report["by_category"].items():
        assert score == 1.0, (cat, score)


def test_naive_agent_is_discriminated(tasks):
    naive = _report(naive_agent, tasks)
    oracle = _report(oracle_agent, tasks)
    # The tool-free agent must be materially worse than the tool-calling oracle.
    assert naive["overall"] <= oracle["overall"] - 0.15, (naive, oracle)


def test_taskset_is_reproducible():
    a = build_taskset(per_category=4)
    b = build_taskset(per_category=4)
    assert [t.to_dict() for t in a] == [t.to_dict() for t in b]


@pytest.mark.parametrize("pred,value,rule,expected", [
    (1.00, 1.00, {"type": "rel", "rel": 0.05}, True),
    (1.10, 1.00, {"type": "rel", "rel": 0.05}, False),
    (1.90, 1.00, {"type": "twofold"}, True),
    (2.10, 1.00, {"type": "twofold"}, False),
    (True, True, {"type": "exact"}, True),
    (False, True, {"type": "exact"}, False),
    (None, 1.00, {"type": "rel", "rel": 0.05}, False),
])
def test_tolerance_rules(pred, value, rule, expected):
    assert within_tolerance(pred, Target("x", value, rule)) is expected


def test_runner_surfaces_errors_and_parse_gaps(tasks):
    # A crashing agent must be counted as errored (not silently a wrong answer);
    # an agent that omits a target must be counted as a parse gap.
    from pharmacometricsbench.runner import run_agent

    def boom(_task):
        raise RuntimeError("kaboom")

    def drops_a_target(task):
        ans = oracle_agent(task)
        ans.pop(next(iter(ans)), None)  # omit one graded target
        return ans

    err = run_agent(boom, tasks)
    assert err["n_error"] == len(tasks) and err["overall"] == 0.0

    gap = run_agent(drops_a_target, tasks)
    assert gap["n_error"] == 0 and gap["n_parse_gap"] == len(tasks)
