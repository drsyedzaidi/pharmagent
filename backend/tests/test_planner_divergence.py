"""Replicate-divergence harness for the goal planner (app.evals.planner_divergence).

Keyless throughout: MockLLM or scripted stub planners, no network. NO test here
runs a population fit: the harness stops every replicate at its first gate and
the spies below prove ToolRegistry.execute is never told allow_expensive=True.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.core import llm as llm_mod
from app.core import policy
from app.core.audit import AuditChain
from app.core.llm_config import LlmChoice
from app.core.orchestrator import Orchestrator
from app.evals import divergence_report as dr
from app.evals import planner_divergence as pd
from app.evals.planner_divergence import LlmSpec, run_divergence
from app.tools.base import ExpensiveToolError, Tool, ToolRegistry, ToolResult
from app.tools.builtins import default_registry
from app.workflows import WORKFLOWS

SAMPLE = str(Path(__file__).parent.parent / "sample_data" / "oral_pk.csv")
SAMPLE_ABS = str(Path(SAMPLE).resolve())
STUB = LlmSpec("stub", "stub")
LOAD = {"tool": "load_dataset", "args": {}, "rationale": "load the data"}
FIT_COMPARE = {"tool": "fit_pk_model", "args": {"compare": True}, "rationale": "compare structures"}
GOAL_MET = {"tool": None, "args": {}, "rationale": "goal met", "stop_reason": "goal met"}
SILENT = {"tool": None, "args": {}, "rationale": "", "stop_reason": None}
MOCK_TOOLS = ["load_dataset", "profile_pk_dataset", "validate_cdisc", "generate_spaghetti_plot",
              "fit_pk_model"]


class _Script:
    """A stub planner: returns its plans in order, then a reasoned stop."""

    def __init__(self, plans):
        self.plans = list(plans)

    def plan(self, goal, state_summary, observations, history, tools):
        if self.plans:
            nxt = self.plans.pop(0)
            return dict(nxt) if isinstance(nxt, dict) else nxt
        return dict(GOAL_MET)


class _Failing(_Script):
    """Plays its script, then fails like a local server that went away."""

    def plan(self, *a):
        if not self.plans:
            raise ConnectionError("ollama unreachable")
        return super().plan(*a)


def _execute_spy(mp) -> list[dict]:
    """Wrap ToolRegistry.execute: record every call's tool and allow_expensive flag."""
    calls: list[dict] = []
    original = ToolRegistry.execute

    def spy(self, name, **kw):
        calls.append({"tool": name, "allow_expensive": kw.get("allow_expensive", False)})
        return original(self, name, **kw)

    mp.setattr(ToolRegistry, "execute", spy)
    return calls


def _registry_with_failing(tool_name: str, fail_on: set[int] | None = None):
    """default_registry() where ``tool_name`` raises (on every call, or only on the
    call numbers in ``fail_on``, counted across replicates)."""
    seen = {"n": 0}

    def factory() -> ToolRegistry:
        full = default_registry()
        base = full.get(tool_name)

        def run(state, ctx, args):
            seen["n"] += 1
            if fail_on is None or seen["n"] in fail_on:
                raise OSError(f"{tool_name} failed")
            return base.run(state, ctx, args)

        reg = ToolRegistry()
        for name in full.names():
            reg.register(full.get(name) if name != tool_name
                         else Tool(base.name, base.description, base.agent, base.input_schema, run))
        return reg

    return factory


def _force_allow_expensive(mp) -> None:
    """Simulate a harness regression that passes allow_expensive=True."""
    original = Orchestrator.start_workflow
    mp.setattr(Orchestrator, "start_workflow",
               lambda self, sid, name, params=None, actor=None, **kw:
               original(self, sid, name, params, actor, allow_expensive=True))


def _keys(rep: dict) -> list[tuple[str, str]]:
    return [policy.canonical_call(d["tool"], d["args"]) for d in rep["decisions"]]


def _run(scripts, replicates=None, **kw):
    """Run the harness with replicate i driven by scripts[i % len(scripts)]."""
    n = replicates or len(scripts)
    return run_divergence(SAMPLE, spec=STUB, replicates=n,
                          llm_factory=lambda i: scripts[i % len(scripts)](), **kw)


@pytest.fixture(scope="module")
def mock_run():
    """One 5-replicate mock run with spies on registry.execute and start_workflow."""
    starts: list[dict] = []
    with pytest.MonkeyPatch.context() as mp:
        calls = _execute_spy(mp)
        original_start = Orchestrator.start_workflow

        def start_spy(self, sid, name, params=None, actor=None, **kw):
            starts.append({"workflow": name, "kwargs": dict(kw)})
            return original_start(self, sid, name, params, actor, **kw)

        mp.setattr(Orchestrator, "start_workflow", start_spy)
        report = run_divergence(SAMPLE, spec=LlmSpec("mock", "mock"), replicates=5)
    return report, calls, starts


# ── the mock provider is a self-test: identical replicates ─────────────────────

def test_mock_provider_gives_five_identical_replicates_and_an_all_converged_report(mock_run):
    report, _, _ = mock_run
    reps = report["replicates"]
    assert report["n_replicates"] == report["n_valid"] == len(reps) == 5
    assert report["verdict"] == dr.ALL_CONVERGED and report["incomplete_reasons"] == []
    assert report["first_divergence_step"] is None
    assert all(_keys(r) == _keys(reps[0]) for r in reps)
    for r in reps:
        assert r["stop_reason"] == "gated_proposal" and r["status"] == "awaiting_review"
        assert r["autonomous_tools"] == MOCK_TOOLS
        assert [d["outcome"] for d in r["decisions"]] == ["ran"] * 5 + ["gated"]
        assert r["gated_proposal"]["tool"] == "run_nlme"
        assert "long-running fit" in r["gated_proposal"]["gate_reason"]
        assert r["rejected"] == [] and r["expensive_executed"] == []
        assert r["model"] == "mock" and r["elapsed_s"] >= 0 and r["planner_s"] >= 0
    points = report["decision_points"]
    assert [p["step"] for p in points] == list(range(6))
    assert all(p["label"] == dr.CONVERGED and p["replicates"] == [0, 1, 2, 3, 4] for p in points)
    assert all(len(p["variants"]) == 1 and p["varied_arg_keys"] == {} for p in points)
    gaps = report["candidate_rule_gaps"]
    assert gaps["arg_keys"] == [] and gaps["tool_choices"] == []


def test_the_harness_never_sets_allow_expensive(mock_run):
    report, calls, starts = mock_run
    assert [c["tool"] for c in calls] == MOCK_TOOLS * 5
    assert all(c["allow_expensive"] is False for c in calls)
    assert len(starts) == 5 and all(s["workflow"] == "poppk_goal" for s in starts)
    assert all("allow_expensive" not in s["kwargs"] for s in starts)
    # The report's guarantee is the harness's OWN observation, matching the spy.
    g = report["guarantee"]
    assert (g["execute_calls"], g["allow_expensive_true_calls"], g["expensive_tools_executed"]) == (25, 0, 0)
    assert g["audit_entries_scanned"] > 25
    assert all(r["execute_calls"] == [{"tool": t, "allow_expensive": False} for t in MOCK_TOOLS]
               for r in report["replicates"])


def test_the_dataset_path_never_lands_in_the_report(mock_run):
    report, _, _ = mock_run
    assert SAMPLE_ABS not in json.dumps(report)
    assert report["dataset"]["name"] == "oral_pk.csv"
    assert report["dataset"]["sha256_before"] == report["dataset"]["sha256_after"]
    assert report["replicates"][0]["decisions"][0]["args"] == {"path": pd.DATASET_TOKEN}


def test_cli_mock_self_test_writes_json_markdown_and_per_replicate_lines(tmp_path, capsys):
    rc = pd.main(["--dataset", SAMPLE, "--replicates", "2", "--provider", "mock",
                  "--out", str(tmp_path)])
    assert rc == pd.EXIT_OK
    report = json.loads((tmp_path / pd.JSON_NAME).read_text())
    assert report["verdict"] == dr.ALL_CONVERGED and report["n_replicates"] == 2
    assert report["llm"]["provider"] == "mock"
    assert len((tmp_path / pd.PARTIAL_NAME).read_text().splitlines()) == 2
    md = (tmp_path / pd.MD_NAME).read_text()
    assert "**Verdict: ALL-CONVERGED**" in md and "## Candidate rule gaps" in md
    assert "doi:10.1002/psp4.70317" in md
    assert "ALL-CONVERGED" in capsys.readouterr().out


def test_cli_mock_self_test_fails_when_replicates_differ(tmp_path, monkeypatch):
    calls = {"n": 0}
    original = llm_mod.MockLLM.plan

    def drifting(self, goal, summary, observations, history, tools):
        out = original(self, goal, summary, observations, history, tools)
        if out.get("tool") == "validate_cdisc":
            calls["n"] += 1
            if calls["n"] == 2:
                return {**out, "args": {"strict": True}}
        return out

    monkeypatch.setattr(llm_mod.MockLLM, "plan", drifting)
    rc = pd.main(["--dataset", SAMPLE, "--replicates", "2", "--provider", "mock", "--out", str(tmp_path)])
    assert rc == pd.EXIT_SELF_TEST


# ── divergence detection ─────────────────────────────────────────────────────

def test_a_planner_alternating_model_key_is_diverged_args_naming_model_key():
    keys = ["oral_1cmt", "oral_2cmt"]
    scripts = [lambda k=k: _Script([LOAD, {"tool": "fit_pk_model", "args": {"model_key": k},
                                           "rationale": f"choose {k}"}]) for k in keys]
    report = _run(scripts, replicates=4)
    p0, p1 = report["decision_points"]
    assert p0["label"] == dr.CONVERGED and p0["n_replicates"] == 4
    assert p1["label"] == dr.DIVERGED_ARGS
    assert p1["prefix"] == [{"tool": "load_dataset", "args": {"path": pd.DATASET_TOKEN}}]
    assert p1["varied_arg_keys"] == {"fit_pk_model": ["model_key"]}
    assert sorted((v["args"]["model_key"], v["count"], v["replicates"]) for v in p1["variants"]) == [
        ("oral_1cmt", 2, [0, 2]), ("oral_2cmt", 2, [1, 3])]
    assert {v["policy_mode"] for v in p1["variants"]} == {"gated"}
    assert report["verdict"] == dr.DIVERGED and report["first_divergence_step"] == 1
    (gap,) = report["candidate_rule_gaps"]["arg_keys"]
    assert (gap["tool"], gap["key"], gap["steps"]) == ("fit_pk_model", "model_key", [1])
    assert gap["policy_scientific_choice"] is True and gap["admitted_without_gate"] is False
    assert gap["note"] == dr.GATED_KEY_NOTE
    for r in report["replicates"]:      # every replicate stopped at that gate; only load ran
        assert r["autonomous_tools"] == ["load_dataset"] and r["stop_reason"] == "gated_proposal"
    md = dr.render_markdown(report)
    assert "DIVERGED-ARGS" in md and "fit\\_pk\\_model.model\\_key" in md and '"choose oral\\_2cmt"' in md


def test_an_ungated_argument_that_varies_is_flagged_as_admitted_without_a_gate():
    scripts = [lambda: _Script([LOAD, {"tool": "validate_cdisc", "args": {}, "rationale": "check"}]),
               lambda: _Script([LOAD, {"tool": "validate_cdisc", "args": {"strict": True},
                                       "rationale": "check"}])]
    report = _run(scripts)
    (gap,) = report["candidate_rule_gaps"]["arg_keys"]
    assert (gap["tool"], gap["key"], gap["policy_scientific_choice"]) == ("validate_cdisc", "strict", False)
    assert gap["admitted_without_gate"] is True and gap["note"] == dr.UNGATED_KEY_NOTE
    assert all(r["stop_reason"] == "planner_complete" for r in report["replicates"])


def test_a_key_carried_only_by_gated_variants_is_reviewed_even_if_a_sibling_ran():
    """fit_pk_model(compare) runs autonomously in one replicate; the other replicate's
    variant carries `foo` but is gated (model_key). `foo` was always seen by a human."""
    scripts = [lambda: _Script([LOAD, FIT_COMPARE]),
               lambda: _Script([LOAD, {"tool": "fit_pk_model",
                                       "args": {"compare": True, "model_key": "oral_1cmt", "foo": 1},
                                       "rationale": "pin it"}])]
    gaps = {g["key"]: g for g in _run(scripts)["candidate_rule_gaps"]["arg_keys"]}
    assert set(gaps) == {"foo", "model_key"}
    assert gaps["foo"]["admitted_without_gate"] is False and gaps["foo"]["note"] == dr.REVIEWED_KEY_NOTE
    assert gaps["model_key"]["admitted_without_gate"] is False
    assert gaps["model_key"]["note"] == dr.GATED_KEY_NOTE


def test_different_tools_at_the_same_point_are_diverged_tool_and_branches_compare_within():
    scripts = [lambda: _Script([LOAD, {"tool": "validate_cdisc", "args": {}, "rationale": "v"}]),
               lambda: _Script([LOAD, {"tool": "generate_spaghetti_plot", "args": {}, "rationale": "s"}])]
    report = _run(scripts, replicates=4)
    by_step: dict[int, list] = {}
    for p in report["decision_points"]:
        by_step.setdefault(p["step"], []).append(p)
    assert by_step[1][0]["label"] == dr.DIVERGED_TOOL
    # After the split each branch is compared only with itself (same prefix).
    assert sorted((p["prefix"][-1]["tool"], p["replicates"]) for p in by_step[2]) == [
        ("generate_spaghetti_plot", [1, 3]), ("validate_cdisc", [0, 2])]
    assert all(p["label"] == dr.CONVERGED and p["variants"][0]["tool"] == dr.END_STOP for p in by_step[2])
    (choice,) = report["candidate_rule_gaps"]["tool_choices"]
    assert choice["step"] == 1 and choice["after"]["tool"] == "load_dataset"
    assert sorted(choice["competing_tools"]) == ["generate_spaghetti_plot", "validate_cdisc"]
    assert choice["note"] == dr.TOOL_CHOICE_NOTE


def test_stopping_where_others_proceed_is_a_stop_versus_continue_gap():
    scripts = [lambda: _Script([LOAD, GOAL_MET]),
               lambda: _Script([LOAD, {"tool": "validate_cdisc", "args": {}, "rationale": "v"}])]
    report = _run(scripts)
    (choice,) = report["candidate_rule_gaps"]["tool_choices"]
    assert set(choice["competing_tools"]) == {dr.END_STOP, "validate_cdisc"}
    assert choice["note"] == dr.END_CHOICE_NOTE


def test_a_silent_stop_and_a_reasoned_stop_are_different_decisions():
    report = _run([lambda: _Script([LOAD, GOAL_MET]), lambda: _Script([LOAD, SILENT])])
    assert [r["stop_reason"] for r in report["replicates"]] == ["planner_complete", "silent_stop"]
    assert report["decision_points"][1]["label"] == dr.DIVERGED_TOOL
    assert {v["tool"] for v in report["decision_points"][1]["variants"]} == {dr.END_STOP, dr.END_SILENT}
    (choice,) = report["candidate_rule_gaps"]["tool_choices"]
    assert choice["note"] == dr.STOP_KIND_NOTE


def test_unreplicated_points_are_labelled_and_never_counted_as_convergence():
    reps = [{"replicate": 0, "decisions": [{"tool": "a", "args": {}}, {"tool": "b", "args": {}}]},
            {"replicate": 1, "decisions": [{"tool": "a", "args": {}}, {"tool": "c", "args": {}},
                                           {"tool": "d", "args": {}}]}]
    labels = [(p["step"], p["label"]) for p in dr.decision_points(reps)]
    assert labels == [(0, dr.CONVERGED), (1, dr.DIVERGED_TOOL), (2, dr.UNREPLICATED)]


# ── stop categories and policy rejections ─────────────────────────────────────

def test_rejections_repeat_and_unknown_tools_are_recorded_with_their_policy_reasons():
    validate = {"tool": "validate_cdisc", "args": {}, "rationale": "v"}
    script = lambda: _Script([LOAD, {"tool": "run_vpc", "args": {}, "rationale": "vpc"},  # noqa: E731
                              {"tool": "made_up_tool", "args": {}, "rationale": "?"}, validate, validate])
    report = _run([script, script])
    for r in report["replicates"]:
        assert r["stop_reason"] == "repeat"
        assert [d["outcome"] for d in r["decisions"]] == ["ran", "rejected", "rejected", "ran", "rejected"]
        assert [x["tool"] for x in r["rejected"]] == ["run_vpc", "made_up_tool", "validate_cdisc"]
    by_tool = {g["tool"]: g for g in report["policy_rejections"]}
    assert "nlme_results" in by_tool["run_vpc"]["reason"] and by_tool["run_vpc"]["count"] == 2
    assert "allowed_tools" in by_tool["made_up_tool"]["reason"]
    assert "identical" in by_tool["validate_cdisc"]["reason"]
    assert report["verdict"] == dr.ALL_CONVERGED
    md = dr.render_markdown(report)
    assert "## Policy rejections" in md and "run\\_vpc ×2" in md


def test_a_budget_stop_is_a_valid_outcome_with_its_own_category(monkeypatch):
    monkeypatch.setitem(WORKFLOWS, "poppk_goal", {**WORKFLOWS["poppk_goal"],
                                                  "budget": {"max_steps": 2, "max_gated": 6}})
    budget = _run([lambda: _Script([LOAD, {"tool": "validate_cdisc", "args": {}, "rationale": "v"},
                                    {"tool": "profile_pk_dataset", "args": {}, "rationale": "p"}])] * 2)
    assert {r["stop_reason"] for r in budget["replicates"]} == {"budget"}
    assert budget["verdict"] == dr.ALL_CONVERGED


def test_a_tool_error_after_real_decisions_is_a_valid_outcome(monkeypatch):
    monkeypatch.setattr(pd, "default_registry", _registry_with_failing("generate_spaghetti_plot"))
    plot = {"tool": "generate_spaghetti_plot", "args": {}, "rationale": "plot"}
    failed = _run([lambda: _Script([LOAD, {"tool": "validate_cdisc", "args": {}, "rationale": "v"}, plot])] * 2)
    for r in failed["replicates"]:
        assert r["stop_reason"] == "tool_error" and "generate_spaghetti_plot failed" in r["stop_detail"]
        assert r["autonomous_tools"] == ["load_dataset", "validate_cdisc"]
        assert r["decisions"][-1]["outcome"] == "tool_error"
    assert failed["verdict"] == dr.ALL_CONVERGED


# ── failed or empty runs never pass as convergence ───────────────────────────

def test_a_transient_planner_error_makes_the_run_incomplete_not_a_rule_gap():
    report = _run([lambda: _Failing([LOAD]), lambda: _Script([LOAD]), lambda: _Script([LOAD])])
    failed = report["replicates"][0]
    assert failed["stop_reason"] == "planner_error" and "ollama unreachable" in failed["stop_detail"]
    assert failed["decisions"][-1]["tool"] == dr.END_PLANNER_ERROR
    assert report["verdict"] == dr.INCOMPLETE and report["analysis_verdict"] == dr.ALL_CONVERGED
    assert report["n_valid"] == 2 and "replicate 0" in report["incomplete_reasons"][0]
    assert all(p["replicates"] == [1, 2] for p in report["decision_points"])
    assert report["candidate_rule_gaps"]["tool_choices"] == []
    assert "## Why the run is incomplete" in dr.render_markdown(report)


def test_a_run_where_every_planner_call_fails_is_incomplete():
    report = _run([lambda: _Failing([])] * 3)
    assert report["verdict"] == dr.INCOMPLETE and report["decision_points"] == []
    assert report["analysis_verdict"] is None
    assert any("no planner decision" in reason for reason in report["incomplete_reasons"])
    assert "Analysis of the valid replicates" not in dr.render_markdown(report)


@pytest.mark.parametrize("first", [
    GOAL_MET,                                                              # prose-only model
    SILENT,                                                                # empty reply
    {"tool": "generate_spaghetti_plot", "args": {}, "rationale": "plot"},  # first-step tool error
])
def test_a_run_that_never_gets_past_the_data_load_is_incomplete_not_converged(first):
    report = _run([lambda: _Script([first])] * 3)
    assert report["analysis_verdict"] == dr.ALL_CONVERGED      # the replicates did agree ...
    assert report["verdict"] == dr.INCOMPLETE                  # ... on nothing that tests a rule
    assert any("nothing about the rules was exercised" in r for r in report["incomplete_reasons"])


def test_an_unloadable_dataset_is_an_invalid_input_not_a_planner_decision(monkeypatch):
    def unreadable(path):
        raise ValueError("bad quote at line 3")

    monkeypatch.setattr("app.tools.data_tools._read", unreadable)
    report = _run([lambda: _Script([LOAD])] * 2)
    assert {r["stop_reason"] for r in report["replicates"]} == {"dataset_error"}
    assert report["verdict"] == dr.INCOMPLETE and report["n_valid"] == 0
    assert "could not be loaded" in report["incomplete_reasons"][0]


def test_the_same_call_with_different_outcomes_is_flagged(monkeypatch):
    """A tool that fails once (I/O) must not hide inside a CONVERGED group."""
    monkeypatch.setattr(pd, "default_registry", _registry_with_failing("validate_cdisc", fail_on={1}))
    validate = {"tool": "validate_cdisc", "args": {}, "rationale": "v"}
    report = _run([lambda: _Script([LOAD, validate])] * 3)
    (variant,) = report["decision_points"][1]["variants"]
    assert variant["outcomes"] == {"tool_error": 1, "ran": 2}
    assert report["verdict"] == dr.INCOMPLETE
    assert any("different outcomes" in r for r in report["incomplete_reasons"])


def test_a_malformed_planner_answer_is_a_harness_error_and_incomplete():
    report = _run([lambda: _Script([["load_dataset"]])] * 2)
    assert {r["stop_reason"] for r in report["replicates"]} == {"harness_error"}
    assert report["verdict"] == dr.INCOMPLETE


def test_cli_exits_incomplete_when_the_local_server_is_down(tmp_path, monkeypatch):
    def down(self, *a, **k):
        raise ConnectionError("connection refused")

    monkeypatch.setattr(llm_mod.OpenAICompatLLM, "plan", down)
    rc = pd.main(["--dataset", SAMPLE, "--replicates", "2", "--provider", "local", "--out", str(tmp_path)])
    assert rc == pd.EXIT_INCOMPLETE
    assert json.loads((tmp_path / pd.JSON_NAME).read_text())["verdict"] == dr.INCOMPLETE


# ── the no-fit guarantee ─────────────────────────────────────────────────────

def test_a_run_nlme_proposal_stops_the_replicate_at_the_gate_with_nothing_else_executed(monkeypatch):
    calls = _execute_spy(monkeypatch)
    nlme = {"tool": "run_nlme", "args": {"method": "focei"}, "rationale": "fit NLME"}
    report = _run([lambda: _Script([LOAD, FIT_COMPARE, nlme,
                                    {"tool": "run_vpc", "args": {}, "rationale": "never reached"}])] * 2)
    assert [c["tool"] for c in calls] == ["load_dataset", "fit_pk_model"] * 2
    assert all(c["allow_expensive"] is False for c in calls)
    for r in report["replicates"]:
        assert r["stop_reason"] == "gated_proposal"
        assert r["autonomous_tools"] == ["load_dataset", "fit_pk_model"]
        assert r["gated_proposal"]["tool"] == "run_nlme"
        assert r["gated_proposal"]["args"] == {"method": "focei"}
        assert [d["tool"] for d in r["decisions"]] == ["load_dataset", "fit_pk_model", "run_nlme"]
    assert report["verdict"] == dr.ALL_CONVERGED


def test_expensive_tool_error_still_fires_if_the_policy_ever_admitted_a_fit(monkeypatch):
    """Simulate a policy regression that classifies everything autonomous: the
    registry choke point still refuses run_nlme because allow_expensive is False."""
    calls = _execute_spy(monkeypatch)
    monkeypatch.setattr(policy, "classify", lambda *a, **k: policy.MODE_AUTONOMOUS)
    report = _run([lambda: _Script([LOAD, FIT_COMPARE,
                                    {"tool": "run_nlme", "args": {}, "rationale": "fit"}])] * 2)
    assert [c["tool"] for c in calls] == ["load_dataset", "fit_pk_model", "run_nlme"] * 2
    assert all(c["allow_expensive"] is False for c in calls)
    for r in report["replicates"]:
        assert r["stop_reason"] == "expensive_refused" and r["status"] == "awaiting_job"
        assert r["autonomous_tools"] == ["load_dataset", "fit_pk_model"]
        assert r["decisions"][-1]["outcome"] == "refused_expensive"
        assert r["gated_proposal"] is None and r["expensive_executed"] == []
    # A policy regression is never reported as a clean verdict.
    assert report["verdict"] == dr.INCOMPLETE


def test_an_expensive_tool_error_escaping_the_orchestrator_is_still_a_refusal(monkeypatch):
    def refuse(self, *a, **k):
        raise ExpensiveToolError("refused")

    monkeypatch.setattr(Orchestrator, "start_workflow", refuse)
    report = _run([lambda: _Script([])] * 2)
    assert {r["stop_reason"] for r in report["replicates"]} == {"expensive_refused"}
    assert report["verdict"] == dr.INCOMPLETE


def test_a_call_with_allow_expensive_is_refused_before_anything_runs(monkeypatch):
    """Policy regression + allow_expensive forced on: the watcher refuses the very
    first flagged call, so even a cheap fake expensive tool never runs."""
    ran: list[dict] = []

    def fake_fit(state, ctx, args):
        ran.append(dict(args))
        return ToolResult(summary="fake fit", writes={}, result={}, action="fake_fit")

    def registry() -> ToolRegistry:
        reg = default_registry()
        reg.register(Tool("fake_fit", "stand-in fit", "modeler",
                          {"type": "object", "properties": {}, "required": []}, fake_fit,
                          expensive=True))
        return reg

    monkeypatch.setattr(pd, "default_registry", registry)
    monkeypatch.setitem(WORKFLOWS, "goal_fake", {"name": "goal_fake", "kind": "goal", "goal": "g",
                                                 "allowed_tools": ["load_dataset", "fake_fit"]})
    monkeypatch.setattr(policy, "classify", lambda *a, **k: policy.MODE_AUTONOMOUS)
    _force_allow_expensive(monkeypatch)
    with pytest.raises(pd.HarnessGuaranteeError, match=r"allow_expensive=True \['load_dataset'\]") as err:
        _run([lambda: _Script([LOAD, {"tool": "fake_fit", "args": {}, "rationale": "fit"}])] * 2,
             workflow="goal_fake")
    assert ran == []                            # refused before it could run
    assert err.value.record["execute_calls"] == [{"tool": "load_dataset", "allow_expensive": True}]


def test_the_audit_scan_counts_executions_not_proposals():
    sess = SimpleNamespace(audit=AuditChain())
    sess.audit.append(agent="planner", tool="run_nlme", action="propose_tool", inputs={}, outputs={},
                      timestamp="t")
    assert pd._executed_tools(default_registry(), sess) == []
    sess.audit.append(agent="modeler", tool="run_nlme", action="fit", inputs={}, outputs={}, timestamp="t")
    assert pd._executed_tools(default_registry(), sess) == ["run_nlme"]


def test_an_expensive_execution_found_by_the_scan_raises_with_the_record(monkeypatch):
    monkeypatch.setattr(pd, "_executed_tools", lambda registry, sess: ["run_nlme"])
    with pytest.raises(pd.HarnessGuaranteeError, match=r"executed \['run_nlme'\]") as err:
        run_divergence(SAMPLE, spec=LlmSpec("mock", "mock"), replicates=2)
    assert err.value.record["expensive_executed"] == ["run_nlme"]


def test_cli_reports_a_guarantee_breach_without_writing_a_report(tmp_path, monkeypatch, capsys):
    """allow_expensive=True on any execution is a breach even when nothing expensive ran
    (the policy still gates run_nlme, so the mock leg stays cheap)."""
    _force_allow_expensive(monkeypatch)
    (tmp_path / pd.JSON_NAME).write_text("{}")          # a stale report must not survive
    rc = pd.main(["--dataset", SAMPLE, "--replicates", "2", "--provider", "mock", "--out", str(tmp_path)])
    assert rc == pd.EXIT_GUARANTEE
    failed = (tmp_path / pd.FAILED_NAME).read_text()
    assert "allow_expensive=True" in failed and '"execute_calls"' in failed
    assert not (tmp_path / pd.JSON_NAME).exists()
    assert "GUARANTEE BREACH" in capsys.readouterr().err


def test_cli_exits_with_a_policy_regression_code_when_the_registry_refused_a_fit(tmp_path, monkeypatch):
    monkeypatch.setattr(policy, "classify", lambda *a, **k: policy.MODE_AUTONOMOUS)
    rc = pd.main(["--dataset", SAMPLE, "--replicates", "2", "--provider", "mock", "--out", str(tmp_path)])
    assert rc == pd.EXIT_POLICY


# ── markdown safety ──────────────────────────────────────────────────────────

def test_llm_text_cannot_inject_markdown():
    evil_key = "x`\n\n# INJECTED\n[click](http://evil.example)"
    scripts = [lambda: _Script([LOAD, {"tool": "validate_cdisc", "args": {evil_key: 1},
                                       "rationale": "<img src=x> [a](http://e.example) | col"}]),
               lambda: _Script([LOAD, {"tool": "validate_cdisc", "args": {}, "rationale": "ok"}])]
    md = dr.render_markdown(_run(scripts))
    assert "DIVERGED-ARGS" in md
    assert not any(line.lstrip().startswith("# INJECTED") for line in md.splitlines())
    for raw in (r"\[click\]\(", r"<img", r"\[a\]\("):          # only ever present escaped
        assert not re.search(r"(?<!\\)" + raw, md), raw
    assert "\\[click\\]" in md and "\\<img" in md


def test_the_progress_line_strips_terminal_escapes(capsys):
    pd._progress({"replicate": 0, "stop_reason": "tool_error", "gated_proposal": None,
                  "stop_detail": "\x1b[2J\x1b[31mboom", "elapsed_s": 1.0})
    err = capsys.readouterr().err
    assert "\x1b" not in err and "boom" in err


def test_long_argument_values_that_differ_never_render_the_same():
    tail = [{"param": "V", "covariate": "SEX"}, {"param": "KA", "covariate": "FOOD"}]
    a = {"covariate_model": [{"param": "CL", "covariate": "WT"}, *tail]}
    b = {"covariate_model": [{"param": "CL", "covariate": "AGE"}, *tail]}
    assert dr.call_text("run_nlme", a) != dr.call_text("run_nlme", b)
    assert len(dr.call_text("run_nlme", a)) < 80


# ── inputs and the LLM spec ──────────────────────────────────────────────────

def test_inputs_are_validated_before_any_llm_is_built(tmp_path):
    built: list[int] = []

    def factory(i):
        built.append(i)
        return _Script([])

    outside = tmp_path / "elsewhere.csv"
    outside.write_text("ID,TIME,DV\n1,0,0\n")
    with pytest.raises(ValueError, match="not permitted"):
        run_divergence(str(outside), spec=STUB, replicates=2, llm_factory=factory)
    with pytest.raises(ValueError, match="replicates"):
        run_divergence(SAMPLE, spec=STUB, replicates=1, llm_factory=factory)
    with pytest.raises(ValueError, match="replicates"):
        run_divergence(SAMPLE, spec=STUB, replicates=pd.MAX_REPLICATES + 1, llm_factory=factory)
    with pytest.raises(ValueError, match="not a goal-driven"):
        run_divergence(SAMPLE, spec=STUB, replicates=2, workflow="poppk_full", llm_factory=factory)
    assert built == []


@pytest.mark.parametrize("argv", [
    ["--provider", "mock", "--temperature", "0.7"],
    ["--provider", "mock", "--model", "qwen3:8b"],
    ["--provider", "local", "--base-url", "https://api.example.com/v1"],
    ["--provider", "local", "--base-url", "http://localhost.evil.example:11434/v1"],
    ["--provider", "local", "--base-url", "https://evil.example/v1?x=127.0.0.1"],
    ["--provider", "local", "--base-url", "http://127.0.0.1@evil.example/v1"],
    ["--provider", "local", "--temperature", "5"],
    ["--provider", "local", "--seed", "3"],
    ["--provider", "local", "--timeout", "0"],
    ["--provider", "mock", "--timeout", "600"],
    ["--provider", "mock", "--replicates", "1"],
])
def test_cli_rejects_bad_arguments_without_running(argv, tmp_path, capsys):
    assert pd.main(["--dataset", SAMPLE, "--out", str(tmp_path), *argv]) == pd.EXIT_USAGE
    assert "error:" in capsys.readouterr().err
    assert not (tmp_path / pd.JSON_NAME).exists()


def _ns(provider: str, allow_hosted: bool = False):
    return pd._parse(["--dataset", SAMPLE, "--out", "x", "--provider", provider]
                     + (["--allow-hosted"] if allow_hosted else []))


def test_configured_provider_refuses_the_mock_and_needs_consent_for_a_hosted_api(monkeypatch):
    monkeypatch.setattr(pd, "choice_from_settings", lambda: LlmChoice("mock", "mock"))
    with pytest.raises(ValueError, match="resolves to the mock"):
        pd.spec_from_args(_ns("configured"))
    monkeypatch.setattr(pd, "choice_from_settings", lambda: LlmChoice("anthropic", "claude-x", None, "k"))
    with pytest.raises(ValueError, match="--allow-hosted"):
        pd.spec_from_args(_ns("configured"))
    spec = pd.spec_from_args(_ns("configured", allow_hosted=True))
    assert (spec.provider, spec.model) == ("configured", "anthropic:claude-x")
    assert "k" not in json.dumps(spec.public())


def test_local_spec_gives_each_replicate_its_own_seed_and_the_temperature():
    spec = LlmSpec("local", "qwen3:8b", "http://127.0.0.1:11434/v1", temperature=0.7, seed=0)
    llms = [spec.build(i) for i in range(3)]
    assert [(m.model, m.temperature, m.seed) for m in llms] == [
        ("qwen3:8b", 0.7, 0), ("qwen3:8b", 0.7, 1), ("qwen3:8b", 0.7, 2)]
    assert spec.sampling(2) == {"temperature": 0.7, "seed": 2}
    assert LlmSpec("mock", "mock").sampling(0) is None


def test_a_local_timeout_applies_to_that_client_only(monkeypatch):
    timeouts: list[float] = []

    class _Resp:
        status_code = 200

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": "stop"}}]}

    def post(url, json=None, headers=None, timeout=None):  # noqa: A002 - mirrors requests.post
        timeouts.append(timeout)
        return _Resp()

    monkeypatch.setattr(llm_mod.requests, "post", post)
    spec = pd.spec_from_args(pd._parse(["--dataset", SAMPLE, "--out", "x", "--provider", "local",
                                        "--timeout", "600"]))
    assert spec.public()["timeout_s"] == 600
    tools = [default_registry().get("load_dataset")]
    spec.build(0).plan("g", {}, [], [], tools)
    LlmSpec("local", "m").build(0).plan("g", {}, [], [], tools)
    assert timeouts == [600, llm_mod.OpenAICompatLLM.TIMEOUT_S] and llm_mod.OpenAICompatLLM.TIMEOUT_S == 120


def test_openai_compat_sends_sampling_overrides_only_when_set(monkeypatch):
    bodies: list[dict] = []

    class _Resp:
        status_code = 200

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": "stop"}}]}

    def post(url, json=None, headers=None, timeout=None):  # noqa: A002 - mirrors requests.post
        bodies.append(json)
        return _Resp()

    monkeypatch.setattr(llm_mod.requests, "post", post)
    tools = [default_registry().get("load_dataset")]
    for kw in ({}, {"temperature": 0.7, "seed": 3}, {"temperature": 0.0, "seed": 0}):
        llm_mod.OpenAICompatLLM(base_url="http://127.0.0.1:1/v1", model="m", api_key=None, **kw).plan(
            "g", {}, [], [], tools)
    default, sampled, zero = bodies
    assert default["temperature"] == 0 and "seed" not in default          # app request unchanged
    assert sampled["temperature"] == 0.7 and sampled["seed"] == 3
    assert zero["temperature"] == 0.0 and zero["seed"] == 0
