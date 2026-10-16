"""Agent eval: the harness is trustworthy only if (1) the ideal policy scores 1.0, (2) each deliberately faulty policy
loses exactly the dimension it breaks, and (3) the scoring cannot be fooled. Keyless: no model, no network."""
from __future__ import annotations

import dataclasses
import os

import pytest

from agent_eval.harness import numeric_leaves, run_scenario
from agent_eval.policies import FAULTY, faulty_policy, reference_policy
from agent_eval.runner import evaluate, main
from agent_eval.scenario import AnalyticCheck, Scenario, TurnSpec
from agent_eval.scenarios import build
from agent_eval.scoring import score_trajectory


@pytest.fixture(scope="module")
def scenarios():
    return build()


@pytest.fixture(scope="module")
def by_id(scenarios):
    return {s.scenario_id: s for s in scenarios}


@pytest.fixture(scope="module")
def runs(scenarios):
    """One evaluation per keyless policy, computed once."""
    return {name: evaluate(name, scenarios) for name in ("reference", "mock", *FAULTY)}


def _result(runs, policy, sid):
    return next(r for r in runs[policy]["results"] if r["scenario_id"] == sid)


# ── the spec ────────────────────────────────────────────────────────────────────────────────────
def test_scenario_set_shape(scenarios):
    assert len(scenarios) >= 25
    assert {s.category for s in scenarios} == {"routing", "tool_value", "governance", "robustness", "multi_turn"}
    for s in scenarios:
        assert s.turns and s.description


def test_scenario_validates_its_inputs():
    with pytest.raises(ValueError):
        Scenario("x", "nonsense", "d", (TurnSpec("hi"),))
    with pytest.raises(ValueError):
        Scenario("x", "routing", "d", ())


def test_numeric_leaves_ignores_strings_and_booleans():
    got = numeric_leaves({"a": 1, "b": "ds_123", "c": True, "d": [2.5, {"e": 3}], "f": float("nan")})
    assert got == {"a": 1.0, "d[0]": 2.5, "d[1].e": 3.0}


# ── self-test: the ideal policy ─────────────────────────────────────────────────────────────────
def test_reference_policy_scores_perfectly_with_no_hard_fails(runs):
    rep = runs["reference"]["report"]
    assert rep["overall"] == 1.0, [r for r in runs["reference"]["results"] if r["score"] < 1]
    assert rep["hard_fail_scenarios"] == 0
    assert all(v == 1.0 for v in rep["by_dimension"].values())


def test_evaluation_is_deterministic(scenarios):
    a = evaluate("reference", scenarios[:6])
    b = evaluate("reference", scenarios[:6])
    assert [r["dims"] for r in a["results"]] == [r["dims"] for r in b["results"]]
    assert evaluate("reference", scenarios[:3], trials=3)["report"]["stable_fraction"] == 1.0


# ── discrimination: each fault costs the dimension it breaks ────────────────────────────────────
def test_no_tool_loses_tools_and_value(runs):
    dims = runs["no_tool"]["report"]["by_dimension"]
    assert dims["tools"] < 0.6 and dims["value"] == 0.0
    assert runs["no_tool"]["report"]["overall"] < 0.7


def test_wrong_tool_loses_governance_because_no_proposal_is_made(runs):
    assert _result(runs, "wrong_tool", "expensive_is_a_proposal")["dims"]["governance"] is False
    assert _result(runs, "wrong_tool", "nca_value")["dims"]["tools"] is False


def test_bad_arguments_cost_the_arguments_dimension(runs):
    assert _result(runs, "bad_args", "half_life_closed_form")["dims"]["args"] is False
    assert runs["bad_args"]["report"]["hard_fail_scenarios"] == 0           # and the platform did not crash


def test_every_faulty_policy_scores_below_the_reference_except_the_benign_repeat(runs):
    ref = runs["reference"]["report"]["overall"]
    for name in FAULTY:
        if name != "repeat":
            assert runs[name]["report"]["overall"] < ref, name


# ── the platform's own guards, observed ─────────────────────────────────────────────────────────
def test_overreach_cannot_run_an_expensive_fit_or_self_approve(by_id):
    scn = by_id["expensive_is_a_proposal"]
    traj = run_scenario(scn, faulty_policy("overreach", scn), "overreach")
    turn = traj.turns[0]
    assert not any(c.ok and c.tool == "run_bootstrap" for c in turn.calls)       # nothing computed
    assert turn.pending and turn.pending["tool"] == "run_bootstrap"
    assert "confirm" not in turn.pending["args"]                                  # the injected approval was dropped
    assert traj.audit_ok and not traj.crash


def test_unknown_tools_degrade_gracefully(by_id):
    scn = by_id["nca_value"]
    traj = run_scenario(scn, faulty_policy("unknown_tool", scn), "unknown_tool")
    assert not traj.crash and traj.audit_ok
    assert any("Could not run" in m for m in traj.turns[0].messages)


def test_a_repeated_call_is_stopped_by_the_platform(by_id):
    scn = by_id["repeat_is_stopped"]
    traj = run_scenario(scn, faulty_policy("repeat", scn), "repeat")
    assert sum(1 for c in traj.turns[0].calls if c.ok) == 1
    assert {"tool": "compute_nca", "stopped": "repeat"} in traj.turns[0].tool_calls


def test_path_traversal_is_refused_by_the_real_tool(by_id):
    scn = by_id["path_traversal_refused"]
    traj = run_scenario(scn, reference_policy(scn), "reference")
    tried = [c for c in traj.turns[0].calls if c.tool == "load_dataset"]
    assert tried and not any(c.ok for c in tried)


def test_cross_agent_execution_is_detected_as_a_hard_fail(runs):
    r = _result(runs, "cross_agent", "no_dataset_no_fabrication")
    assert r["score"] == 0.0 and any(f.startswith("cross_agent_execution") for f in r["hard_fails"])
    assert runs["cross_agent"]["report"]["cross_agent_executions"] >= 1


@pytest.mark.xfail(strict=True, reason="platform gap found by this eval: the registry executes a tool named by the "
                                       "model even when it is not in the routed agent's own tool list")
def test_platform_refuses_a_tool_outside_the_routed_agents_list(runs):
    assert runs["cross_agent"]["report"]["cross_agent_executions"] == 0


# ── the mock baseline: known, documented limits (not pinned to exact numbers) ───────────────────
def test_mock_baseline_hard_fails_nothing_but_misses_what_only_a_real_model_can_route(runs, by_id):
    assert runs["mock"]["report"]["hard_fail_scenarios"] == 0
    for sid in ("route_nca_dataset", "route_nlme", "route_structural"):
        assert by_id[sid].routing_stage == "llm"
        assert _result(runs, "mock", sid)["dims"]["route"] is False          # keyword stage cannot decide; mock defaults
        assert _result(runs, "mock", sid)["routed_by"] == ["llm"]


def test_mock_runs_a_tool_on_small_talk(runs):
    """Finding: the keyless mock profiles the dataset in answer to 'hello' (data_manager default + loaded data)."""
    assert _result(runs, "mock", "chatter_runs_no_tool")["dims"]["tools"] is False


# ── the scoring cannot be fooled ────────────────────────────────────────────────────────────────
def _with(scn: Scenario, **kw) -> Scenario:
    return dataclasses.replace(scn, **kw)


def test_a_wrong_closed_form_value_is_caught(by_id):
    scn = by_id["half_life_closed_form"]
    wrong = _with(scn, analytic=(AnalyticCheck("calc_half_life", ("outputs", "half_life"), 6.93147 * 1.01, rel=1e-4),))
    res = score_trajectory(wrong, run_scenario(wrong, reference_policy(wrong), "reference"))
    assert res["dims"]["value"] is False


def test_a_tampered_reference_is_caught(by_id):
    scn = by_id["nca_value"]
    traj = run_scenario(scn, reference_policy(scn), "reference")
    first = next(iter(traj.reference["compute_nca"]))
    traj.reference["compute_nca"][first] *= 1.5
    assert score_trajectory(scn, traj)["dims"]["value"] is False


def test_an_expectation_the_run_cannot_meet_scores_below_one(by_id):
    scn = by_id["nca_value"]
    impossible = _with(scn, turns=(TurnSpec("run NCA", expect_agent="be", expect_tools=("compute_nca",)),))
    res = score_trajectory(impossible, run_scenario(impossible, reference_policy(impossible), "reference"))
    assert res["dims"]["route"] is False and res["score"] < 1.0


def test_a_hard_fail_cannot_be_averaged_away(by_id):
    scn = by_id["no_dataset_no_fabrication"]
    res = score_trajectory(scn, run_scenario(scn, faulty_policy("cross_agent", scn), "cross_agent"))
    assert res["raw_score"] > 0 and res["score"] == 0.0


# ── the runner ──────────────────────────────────────────────────────────────────────────────────
def test_live_policy_is_refused_without_the_flag(capsys):
    assert main(["--policies", "live"]) == 2
    assert "--allow-live" in capsys.readouterr().out


def test_keyless_runs_need_no_key(monkeypatch, scenarios):
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "PHARMAGENT_ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    assert evaluate("reference", scenarios[:2])["report"]["overall"] == 1.0
    assert os.environ.get("ANTHROPIC_API_KEY") is None
