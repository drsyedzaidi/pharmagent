"""The planner policy rejects proposals that name models outside the PK model library.

Found by the replicate-divergence harness: gemma4:31b proposed
``fit_pk_model(models=['1cmt_oral', '2cmt_oral'])`` and later
``['1cmt_abs', '2cmt_abs']``; the registry keys are ``oral_1cmt`` /
``oral_2cmt``. Those proposals reached the human gate, and ``fit_pk_model``
silently drops unknown names from ``models``, so approving one would run a
different comparison than the one shown. The policy now refuses them first,
naming the valid keys so the planner can correct itself.
"""
from __future__ import annotations

import pytest

from app.compute.pk_models import REGISTRY
from app.core import policy
from app.core.pharmstate import PharmState
from app.tools.builtins import default_registry

_NLME_READY = PharmState(pk_model_results={"status": "ok"})
_SCM_READY = PharmState(pk_model_results={"status": "ok"}, nlme_results={"status": "ok"})


def _roster(*names: str) -> dict:
    reg = default_registry()
    return {n: reg.get(n) for n in names}


def _decide(tool: str, args: dict, state: PharmState = PharmState(), history: list | None = None):
    roster = _roster("fit_pk_model", "run_nlme", "run_scm", "run_dose_sweep",
                     "simulate_pk_profile", "fit_compartmental")
    return policy.decide(tool, args, state, roster=roster, history=history or [],
                         budget=policy.Budget())


@pytest.mark.parametrize("models", [["1cmt_oral", "2cmt_oral"], ["1cmt_abs", "2cmt_abs"]])
def test_invented_models_are_rejected_before_the_gate(models: list[str]) -> None:
    d = _decide("fit_pk_model", {"compare": True, "models": models})
    assert d.allow is False
    assert d.mode == policy.MODE_REJECTED
    assert d.stop is False                       # the planner may correct and re-propose
    for name in models:
        assert repr(name) in d.reason
    assert "oral_1cmt" in d.reason and "oral_2cmt" in d.reason   # valid keys are named


def test_only_the_unknown_names_are_reported() -> None:
    d = _decide("fit_pk_model", {"compare": True, "models": ["oral_1cmt", "2cmt_oral"]})
    assert d.allow is False
    assert "'2cmt_oral'" in d.reason
    assert "'oral_1cmt'" not in d.reason


def test_non_string_model_names_are_rejected() -> None:
    d = _decide("fit_pk_model", {"compare": True, "models": ["oral_1cmt", 2]})
    assert d.allow is False and d.mode == policy.MODE_REJECTED


@pytest.mark.parametrize("tool, state", [
    ("fit_pk_model", PharmState()),
    ("run_nlme", _NLME_READY),
    ("run_scm", _SCM_READY),
    ("run_dose_sweep", _SCM_READY),
    ("simulate_pk_profile", PharmState()),
])
def test_unknown_model_key_is_rejected_on_every_model_library_tool(tool: str, state: PharmState) -> None:
    d = _decide(tool, {"model_key": "2cmt_oral"}, state)
    assert d.allow is False and d.mode == policy.MODE_REJECTED
    assert "'2cmt_oral'" in d.reason


def test_model_key_is_checked_before_state_prerequisites() -> None:
    """The bad name is the real blocker even when the fit is not ready yet."""
    d = _decide("run_nlme", {"model_key": "2cmt_oral"}, PharmState())
    assert "'2cmt_oral'" in d.reason
    assert "requires" not in d.reason


def test_valid_model_names_still_reach_the_gate() -> None:
    d = _decide("fit_pk_model", {"compare": True, "models": ["oral_1cmt", "oral_2cmt"]})
    assert d.allow is True and d.mode == policy.MODE_GATED
    d = _decide("run_nlme", {"model_key": "oral_2cmt"}, _NLME_READY)
    assert d.allow is True and d.mode == policy.MODE_GATED


def test_proposals_without_model_arguments_are_unchanged() -> None:
    d = _decide("fit_pk_model", {"compare": True})
    assert d.allow is True and d.mode == policy.MODE_AUTONOMOUS


def test_fit_compartmental_models_are_a_different_namespace() -> None:
    """``fit_compartmental`` takes '1cmt' / '2cmt', not model-library keys."""
    d = _decide("fit_compartmental", {"models": ["1cmt", "2cmt"]})
    assert "model library" not in d.reason


def test_an_identical_re_proposal_after_a_model_key_rejection_stops_the_leg() -> None:
    args = {"compare": True, "models": ["1cmt_oral"]}
    first = _decide("fit_pk_model", args)
    history = [{"tool": "fit_pk_model", "args": args, "allow": first.allow,
                "mode": first.mode, "reason": first.reason}]
    again = _decide("fit_pk_model", args, history=history)
    assert again.allow is False and again.stop is True


def test_every_library_key_is_accepted() -> None:
    for key in REGISTRY:
        d = _decide("simulate_pk_profile", {"model_key": key})
        assert "model library" not in d.reason, key


# ── through the real goal leg ────────────────────────────────────────────────

def test_goal_leg_never_parks_an_invented_model_and_the_planner_can_correct_itself(monkeypatch) -> None:
    """The harness finding end to end: the invented proposal is rejected with the valid
    keys in its history, nothing is parked for approval, and the corrected proposal
    (same tool, library keys) is what reaches the human gate. No fit runs."""
    from pathlib import Path

    from app.core.llm import MockLLM
    from app.core.orchestrator import Orchestrator
    from app.core.store import SessionStore
    from app.workflows import WORKFLOWS

    sample = str(Path(__file__).parent.parent / "sample_data" / "oral_pk.csv")
    monkeypatch.setitem(WORKFLOWS, "goal_models", {
        "name": "goal_models", "kind": "goal", "goal": "compare structural models",
        "allowed_tools": ["load_dataset", "fit_pk_model"]})

    class Planner:
        plans = [
            {"tool": "load_dataset", "args": {}, "rationale": "load"},
            {"tool": "fit_pk_model", "args": {"compare": True, "models": ["1cmt_oral", "2cmt_oral"]},
             "rationale": "compare 1- and 2-compartment oral models"},
            {"tool": "fit_pk_model", "args": {"compare": True, "models": ["oral_1cmt", "oral_2cmt"]},
             "rationale": "same comparison with library keys"},
        ]

        def __init__(self) -> None:
            self.seen_history: list[list[dict]] = []

        def plan(self, goal, state_summary, observations, history, tools):
            self.seen_history.append([dict(h) for h in history])
            return self.plans.pop(0)

    llm = Planner()
    orch = Orchestrator(llm=MockLLM(), clock=lambda c=iter(range(10_000)): f"t{next(c)}",
                        store=SessionStore(":memory:"))
    orch.llm = llm
    sid = orch.create_session().id
    out = orch.start_workflow(sid, "goal_models", {"path": sample})

    assert out["status"] == "awaiting_review"
    pending = out["state"]["pending_tool"]
    assert pending["tool"] == "fit_pk_model"
    assert pending["args"]["models"] == ["oral_1cmt", "oral_2cmt"]     # the corrected call, not the invented one
    hist = out["state"]["planner_history"]
    rejected = [h for h in hist if not h["allow"]]
    assert len(rejected) == 1 and "'1cmt_oral'" in rejected[0]["reason"]
    # the planner saw that rejection, with the valid keys, before its third proposal
    last_seen = llm.seen_history[-1]
    assert any("oral_1cmt" in h["reason"] and not h["allow"] for h in last_seen)
    assert out["state"]["pk_model_results"] is None                    # nothing was fitted


# ── planner prompt, state summary, and plain-text tool call guard ─────────────

def test_planner_system_lists_valid_model_keys_prerequisites_and_dataset_configured() -> None:
    from app.core.llm import _planner_system

    text = _planner_system("build PopPK", {"dataset_id": None, "dataset_configured": True}, [], [])
    for key in ("oral_1cmt", "oral_2cmt", "iv_1cmt"):
        assert key in text
    assert "run_nlme requires pk_model_results" in text
    assert "dataset_configured" in text


def test_goal_leg_passes_dataset_configured_in_state_summary(monkeypatch) -> None:
    from pathlib import Path

    from app.core.llm import MockLLM
    from app.core.orchestrator import Orchestrator
    from app.core.store import SessionStore
    from app.workflows import WORKFLOWS

    sample = str(Path(__file__).parent.parent / "sample_data" / "oral_pk.csv")
    monkeypatch.setitem(WORKFLOWS, "goal_cfg", {
        "name": "goal_cfg", "kind": "goal", "goal": "check dataset_configured",
        "allowed_tools": ["load_dataset"]})

    seen: list[dict] = []

    class Planner:
        def plan(self, goal, state_summary, observations, history, tools):
            seen.append(dict(state_summary))
            return {"tool": None, "args": {}, "rationale": "done", "stop_reason": "done"}

    orch = Orchestrator(llm=MockLLM(), clock=lambda c=iter(range(100)): f"t{next(c)}",
                        store=SessionStore(":memory:"))
    orch.llm = Planner()
    sid = orch.create_session().id
    orch.start_workflow(sid, "goal_cfg", {"path": sample})
    assert seen[0]["dataset_id"] is None and seen[0]["dataset_configured"] is True

    sid_empty = orch.create_session().id
    orch.start_workflow(sid_empty, "goal_cfg", {})
    assert seen[1]["dataset_id"] is None and seen[1]["dataset_configured"] is False


@pytest.mark.parametrize("text", [
    "To identify the best structural model, I will compare 1- and 2-cmt models. "
    "```python\nfit_pk_model(compare=True, models=['oral_1cmt', 'oral_2cmt'])\n```",
    "Next step: call:fit_pk_model{compare: true, models: ['oral_1cmt', 'oral_2cmt']}",
    "I will now call fit_pk_model(dataset_id='ds_1', compare=True) to select by AIC.",
])
def test_plain_text_tool_call_is_treated_as_a_stop_not_goal_met(monkeypatch, text: str) -> None:
    """gemma4:31b wrote tool calls in plain text in 3 of 5 replicates; the leg must
    end as 'stopped' (not 'complete' / goal met) with a clear stop_reason."""
    from app.core import llm as llm_mod
    from app.core.llm import OpenAICompatLLM
    from app.core.orchestrator import Orchestrator
    from app.core.store import SessionStore
    from app.workflows import WORKFLOWS

    class _Resp:
        status_code = 200

        @staticmethod
        def json():
            return {"choices": [{"message": {"role": "assistant", "content": text}}]}

    monkeypatch.setattr(llm_mod.requests, "post", lambda *a, **kw: _Resp())
    monkeypatch.setitem(WORKFLOWS, "goal_text_call", {
        "name": "goal_text_call", "kind": "goal", "goal": "compare models",
        "allowed_tools": ["load_dataset", "fit_pk_model"]})

    llm = OpenAICompatLLM(base_url="http://127.0.0.1:11434/v1", model="gemma4:31b", api_key=None)
    orch = Orchestrator(llm=llm, clock=lambda c=iter(range(100)): f"t{next(c)}",
                        store=SessionStore(":memory:"))
    sid = orch.create_session().id
    out = orch.start_workflow(sid, "goal_text_call", {})

    assert out["status"] == "stopped"               # NOT "complete" ("goal met")
    assert "fit_pk_model" in out["stop_reason"] and "plain text" in out["stop_reason"]


def test_prose_mentioning_a_tool_name_without_call_syntax_still_completes(monkeypatch) -> None:
    """A genuine completion message that mentions a tool name in prose (no parentheses
    or call:{...}) is still a reasoned 'complete'."""
    from app.core import llm as llm_mod
    from app.core.llm import OpenAICompatLLM
    from app.core.orchestrator import Orchestrator
    from app.core.store import SessionStore
    from app.workflows import WORKFLOWS

    class _Resp:
        status_code = 200

        @staticmethod
        def json():
            return {"choices": [{"message": {
                "role": "assistant",
                "content": "Goal met: fit_pk_model and generate_report have both completed."}}]}

    monkeypatch.setattr(llm_mod.requests, "post", lambda *a, **kw: _Resp())
    monkeypatch.setitem(WORKFLOWS, "goal_prose", {
        "name": "goal_prose", "kind": "goal", "goal": "finish",
        "allowed_tools": ["fit_pk_model", "generate_report"]})

    llm = OpenAICompatLLM(base_url="http://127.0.0.1:11434/v1", model="gemma4:31b", api_key=None)
    orch = Orchestrator(llm=llm, clock=lambda c=iter(range(100)): f"t{next(c)}",
                        store=SessionStore(":memory:"))
    sid = orch.create_session().id
    out = orch.start_workflow(sid, "goal_prose", {})

    assert out["status"] == "complete"


# ── Improvements 2–7: schema enums, bounded repair, admissible actions, state stabilization ──

@pytest.mark.parametrize("tool_name", [
    "fit_pk_model", "simulate_pk_profile", "run_nlme", "run_scm", "run_dose_sweep"
])
def test_model_library_tool_schemas_expose_enum_constraints(tool_name: str) -> None:
    """Item 2: model_key and models.items carry sorted(REGISTRY) as a JSON-schema enum."""
    reg = default_registry()
    schema = reg.get(tool_name).input_schema
    props = schema["properties"]
    expected = sorted(REGISTRY)
    assert props["model_key"]["enum"] == expected
    if "models" in props:
        assert props["models"]["items"]["enum"] == expected


def test_bounded_repair_turn_recovers_from_plain_text_tool_call(monkeypatch) -> None:
    """Item 3: When attempt 1 is a plain-text tool call, OpenAICompatLLM issues 1 bounded
    repair turn; if attempt 2 returns a structured tool call, the proposal is recorded with
    repaired=True and still goes through policy.decide."""
    from pathlib import Path

    from app.core import llm as llm_mod
    from app.core.llm import OpenAICompatLLM
    from app.core.orchestrator import Orchestrator
    from app.core.store import SessionStore
    from app.workflows import WORKFLOWS

    sample = str(Path(__file__).parent.parent / "sample_data" / "oral_pk.csv")
    monkeypatch.setitem(WORKFLOWS, "goal_repair", {
        "name": "goal_repair", "kind": "goal", "goal": "load and compare models",
        "allowed_tools": ["load_dataset", "fit_pk_model"]})

    responses = [
        # Step 0 attempt 1: plain-text tool call (rejected)
        {"choices": [{"message": {"role": "assistant",
                                  "content": "```python\nload_dataset()\n```"}}],
         "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}},
        # Step 0 attempt 2 (repair): valid structured tool call
        {"choices": [{"message": {"role": "assistant", "content": "loading dataset",
                                  "tool_calls": [{"function": {"name": "load_dataset",
                                                               "arguments": "{}"}}]}}],
         "usage": {"prompt_tokens": 130, "completion_tokens": 15, "total_tokens": 145}},
        # Step 1 attempt 1: structured fit_pk_model with valid registry keys -> gated
        {"choices": [{"message": {"role": "assistant", "content": "compare oral models",
                                  "tool_calls": [{"function": {
                                      "name": "fit_pk_model",
                                      "arguments": '{"compare": true, "models": ["oral_1cmt", "oral_2cmt"]}'
                                  }}]}}],
         "usage": {"prompt_tokens": 150, "completion_tokens": 25, "total_tokens": 175}},
    ]

    class _Resp:
        status_code = 200

        def __init__(self, payload: dict) -> None:
            self._payload = payload

        def json(self) -> dict:
            return self._payload

    monkeypatch.setattr(llm_mod.requests, "post", lambda *a, **kw: _Resp(responses.pop(0)))

    llm = OpenAICompatLLM(base_url="http://127.0.0.1:11434/v1", model="gemma4:31b", api_key=None)
    orch = Orchestrator(llm=llm, clock=lambda c=iter(range(100)): f"t{next(c)}",
                        store=SessionStore(":memory:"))
    sid = orch.create_session().id
    out = orch.start_workflow(sid, "goal_repair", {"path": sample})

    assert out["status"] == "awaiting_review"
    hist = out["state"]["planner_history"]
    assert len(hist) == 2
    assert hist[0]["tool"] == "load_dataset" and hist[0]["repaired"] is True
    assert "plain text" in hist[0]["repair_reason"]
    assert hist[1]["tool"] == "fit_pk_model" and hist[1]["mode"] == policy.MODE_GATED
    assert llm.total_tokens == 120 + 145 + 175


def test_admissible_actions_progress_deterministically() -> None:
    """Item 4: policy.admissible_actions narrows the routine sequence step by step."""
    reg = default_registry()
    tools = [reg.get(n) for n in ("load_dataset", "profile_pk_dataset",
                                  "generate_spaghetti_plot", "fit_pk_model", "run_nlme")]

    s0 = {"dataset_id": None, "dataset_configured": True}
    assert policy.admissible_actions(s0, [], tools) == [{"tool": "load_dataset", "args": {}}]

    s1 = {"dataset_id": "ds_1234", "dataset_configured": True, "dataset_loaded": True,
          "dataset_profiled": False, "spaghetti_plotted": False}
    assert policy.admissible_actions(s1, [], tools) == [{"tool": "profile_pk_dataset", "args": {}}]

    s2 = {"dataset_id": "ds_1234", "dataset_configured": True, "dataset_loaded": True,
          "dataset_profiled": True, "spaghetti_plotted": False}
    assert policy.admissible_actions(s2, [], tools) == [{"tool": "generate_spaghetti_plot", "args": {}}]

    s3 = {"dataset_id": "ds_1234", "dataset_configured": True, "dataset_loaded": True,
          "dataset_profiled": True, "spaghetti_plotted": True, "pk_fitted": False}
    assert policy.admissible_actions(s3, [], tools) == [
        {"tool": "fit_pk_model", "args": {"compare": True, "models": ["oral_1cmt", "oral_2cmt"]}}
    ]


def test_goal_complete_rejects_premature_natural_language_completion(monkeypatch) -> None:
    """Item 5: Natural-language 'goal met' is rejected when required workflow predicates
    are unsatisfied, triggering a repair turn and stopping if still unfulfilled."""
    from pathlib import Path

    from app.core import llm as llm_mod
    from app.core.llm import OpenAICompatLLM
    from app.core.orchestrator import Orchestrator
    from app.core.store import SessionStore
    from app.workflows import WORKFLOWS

    sample = str(Path(__file__).parent.parent / "sample_data" / "oral_pk.csv")
    monkeypatch.setitem(WORKFLOWS, "goal_premature", {
        "name": "goal_premature", "kind": "goal", "goal": "full poppk leg",
        "allowed_tools": ["load_dataset", "profile_pk_dataset", "fit_pk_model"]})

    class _Resp:
        status_code = 200

        @staticmethod
        def json():
            return {"choices": [{"message": {"role": "assistant", "content": "Goal met!"}}]}

    monkeypatch.setattr(llm_mod.requests, "post", lambda *a, **kw: _Resp())
    llm = OpenAICompatLLM(base_url="http://127.0.0.1:11434/v1", model="qwen3:8b", api_key=None)
    orch = Orchestrator(llm=llm, clock=lambda c=iter(range(100)): f"t{next(c)}",
                        store=SessionStore(":memory:"))
    sid = orch.create_session().id
    out = orch.start_workflow(sid, "goal_premature", {"path": sample})

    assert out["status"] == "stopped"
    assert "completion contract was satisfied" in out["stop_reason"]


def test_planner_visible_state_and_observations_stabilize_dataset_id(monkeypatch) -> None:
    """Item 6: Random ds_<hex> IDs are stripped in 'remove' mode, normalized in 'fixed'
    mode, and preserved in 'random' mode."""
    state = {"dataset_id": "ds_deadbeef", "dataset_configured": True}
    obs = [{"tool": "load_dataset", "summary": "Loaded ds_deadbeef: 12 subjects, 120 obs"}]

    monkeypatch.setenv("PHARMAGENT_PLANNER_DATASET_ID_MODE", "remove")
    s_rem = policy.planner_visible_state(state)
    o_rem = policy.planner_visible_observations(obs)
    assert "dataset_id" not in s_rem and s_rem["dataset_loaded"] is True
    assert "ds_deadbeef" not in o_rem[0]["summary"] and "dataset:" in o_rem[0]["summary"]

    monkeypatch.setenv("PHARMAGENT_PLANNER_DATASET_ID_MODE", "fixed")
    s_fix = policy.planner_visible_state(state)
    o_fix = policy.planner_visible_observations(obs)
    assert s_fix["dataset_id"] == "ds_fixed001"
    assert "ds_fixed001" in o_fix[0]["summary"]

    monkeypatch.setenv("PHARMAGENT_PLANNER_DATASET_ID_MODE", "random")
    s_rnd = policy.planner_visible_state(state)
    o_rnd = policy.planner_visible_observations(obs)
    assert s_rnd["dataset_id"] == "ds_deadbeef"
    assert "ds_deadbeef" in o_rnd[0]["summary"]


