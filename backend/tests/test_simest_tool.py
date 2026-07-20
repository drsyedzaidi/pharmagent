"""Tool-wiring tests for run_simest (agent registration, confirm gate,
no-clobber on rejection, fit_fn wiring) -- via a monkeypatched
`app.compute.nlme.population_fit`, so NO test in this file runs a real fit.

`population_fit` is imported lazily inside the tool function (matching
`run_nlme`'s own convention), so patching the module attribute before calling
the tool intercepts every call the tool makes.
"""
from __future__ import annotations

import json

import pytest

from app.agents.definitions import AGENTS, DESCRIPTIONS
from app.agents.supervisor import KEYWORDS, Supervisor, score
from app.core.pharmstate import AGENT_WRITE_FIELDS, PharmState, apply_writes
from app.tools.base import ToolContext
from app.tools.builtins import default_registry
from app.tools.simest_tools import run_simest

MODEL_KEY = "oral_1cmt"


def _nlme(*, cov_effects=None) -> dict:
    return {
        "status": "ok", "model_key": MODEL_KEY,
        "theta": {"CL": 4.0, "V": 40.0, "KA": 1.0},
        "omega_cv_pct": {"CL": 30.0, "V": 20.0},
        "sigma": {"prop": 0.1, "add": 0.3},
        "iiv_params": ["CL", "V"], "error_model": "combined",
        "covariate_effects": cov_effects or [],
    }


def _design() -> dict:
    return {"n_subjects": 10, "obs_t": [0.5, 1, 2, 4, 8], "dose": 100.0, "n_doses": 1}


def _fake_population_fit(monkeypatch, *, calls=None):
    def fake(model_key, subjects, *, method="focei", iiv_params=None, error_model="proportional",
             max_iter=200, seed=20250614, compute_uncertainty=True, covariate_model=None):
        if calls is not None:
            calls.append({"method": method, "seed": seed, "n_subjects": len(subjects)})
        return {"theta": {"CL": 4.0, "V": 40.0}, "theta_rse_pct": {"CL": 8.0, "V": 6.0},
               "converged": True}
    monkeypatch.setattr("app.compute.nlme.population_fit", fake)


@pytest.fixture
def state():
    return PharmState(dataset_id="d1", nlme_results=_nlme())


# ── SAFETY: the LLM chat path can never reach run_simest ────────────────────

def test_tool_is_registered_under_simulator_not_modeler():
    tool = default_registry().get("run_simest")
    assert tool.agent == "simulator"


def test_simulator_is_absent_from_agents_and_descriptions():
    # The actual mechanism that makes agent="simulator" unreachable from chat:
    # AGENTS/DESCRIPTIONS have no "simulator" entry.
    assert "simulator" not in AGENTS
    assert "simulator" not in DESCRIPTIONS


def test_simulator_is_absent_from_supervisor_keywords_and_unroutable():
    assert "simulator" not in KEYWORDS
    # score() can only ever produce keys present in KEYWORDS.
    assert "simulator" not in score("run a simulation estimation study now")


def test_supervisor_route_can_never_return_simulator(monkeypatch):
    # Even forcing the LLM-classification fallback path (by making every
    # keyword score tie at zero) cannot select "simulator", because its
    # candidate list is `list(DESCRIPTIONS)`, which excludes it.
    class _StubLLM:
        def classify(self, message, choices, descriptions):
            assert "simulator" not in choices
            return choices[0]
    sup = Supervisor(_StubLLM())
    agent_name, method = sup.route("")
    assert agent_name != "simulator"


def test_state_write_access_includes_simest_results():
    assert "simest_results" in AGENT_WRITE_FIELDS["simulator"]
    assert "simest_results" not in AGENT_WRITE_FIELDS.get("modeler", set())
    st = apply_writes(PharmState(), "simulator", {"simest_results": {"status": "ok"}})
    assert st.simest_results == {"status": "ok"}


# ── confirm gate: unconditional, never threshold-gated ──────────────────────

def test_confirm_false_is_rejected_without_running_a_fit(state, monkeypatch):
    calls = []
    _fake_population_fit(monkeypatch, calls=calls)
    res = run_simest(state, ToolContext(), {"design": _design()})  # no "confirm" key
    assert res.result["status"] == "confirm_required"
    assert res.writes == {}
    assert calls == []  # population_fit was NEVER called


def test_confirm_explicitly_false_is_also_rejected(state, monkeypatch):
    calls = []
    _fake_population_fit(monkeypatch, calls=calls)
    res = run_simest(state, ToolContext(), {"confirm": False, "design": _design()})
    assert res.result["status"] == "confirm_required"
    assert calls == []


def test_confirm_true_with_a_tiny_design_still_requires_confirm(state, monkeypatch):
    # The historical porous threshold gate (n_rep * n_subjects > 400) is gone
    # entirely -- confirm is unconditional regardless of how small the design is.
    calls = []
    _fake_population_fit(monkeypatch, calls=calls)
    tiny = {"n_subjects": 2, "obs_t": [1.0], "dose": 100.0}
    res = run_simest(state, ToolContext(), {"design": tiny})  # still no confirm
    assert res.result["status"] == "confirm_required"
    assert calls == []


# ── no-clobber: a rejection must not overwrite a prior successful run ───────

def test_rejection_does_not_clobber_a_previous_successful_run(state, monkeypatch):
    _fake_population_fit(monkeypatch)
    good = {"status": "ok", "n_rep_completed": 5}
    state.simest_results = good
    res = run_simest(state, ToolContext(), {"design": _design()})  # confirm missing -> rejected
    assert res.writes == {}  # the prior good result in `state` is untouched
    assert state.simest_results == good


def test_no_nlme_is_rejected_without_writes(monkeypatch):
    calls = []
    _fake_population_fit(monkeypatch, calls=calls)
    res = run_simest(PharmState(), ToolContext(), {"confirm": True, "design": _design()})
    assert res.result["status"] == "no_nlme"
    assert res.writes == {}
    assert calls == []


def test_invalid_design_is_rejected_without_writes(state, monkeypatch):
    calls = []
    _fake_population_fit(monkeypatch, calls=calls)
    res = run_simest(state, ToolContext(), {"confirm": True, "design": {"n_subjects": 1}})
    assert res.result["status"] == "invalid_design"
    assert res.writes == {}
    assert calls == []


def test_covariate_model_is_rejected_without_writes(monkeypatch):
    calls = []
    _fake_population_fit(monkeypatch, calls=calls)
    state = PharmState(nlme_results=_nlme(cov_effects=[
        {"param": "CL", "covariate": "SEX", "kind": "categorical", "levels": ["F"],
         "coefficient": {"F": -0.2}}]))
    res = run_simest(state, ToolContext(), {"confirm": True, "design": _design()})
    assert res.result["status"] == "covariates_unsupported"
    assert res.writes == {}
    assert calls == []


# ── successful run: writes state, wraps fit_fn correctly ───────────────────

def test_confirmed_run_writes_simest_results_and_calls_the_fake_fit(state, monkeypatch):
    calls = []
    _fake_population_fit(monkeypatch, calls=calls)
    res = run_simest(state, ToolContext(), {"confirm": True, "design": _design(), "n_rep": 2})
    assert res.writes["simest_results"]["status"] in ("ok", "partial")
    assert len(calls) == 2  # one population_fit call per completed replicate
    assert all(c["method"] == "focei" for c in calls)
    assert res.result["n_rep_completed"] == 2


def test_method_saem_is_threaded_through(state, monkeypatch):
    calls = []
    _fake_population_fit(monkeypatch, calls=calls)
    run_simest(state, ToolContext(), {"confirm": True, "design": _design(),
                                      "n_rep": 1, "method": "saem"})
    assert calls[0]["method"] == "saem"


def test_citation_unverified_flag_present_in_summary(state, monkeypatch):
    _fake_population_fit(monkeypatch)
    res = run_simest(state, ToolContext(), {"confirm": True, "design": _design(), "n_rep": 1})
    assert "CITATION UNVERIFIED" in res.summary


def test_payload_is_json_safe(state, monkeypatch):
    _fake_population_fit(monkeypatch)
    res = run_simest(state, ToolContext(), {"confirm": True, "design": _design(), "n_rep": 2})
    json.dumps(res.writes["simest_results"])
    json.dumps(res.result)


def test_a_real_fit_exception_from_population_fit_does_not_crash_the_tool(state, monkeypatch):
    def crashing(*a, **kw):
        raise RuntimeError("estimator blew up")
    monkeypatch.setattr("app.compute.nlme.population_fit", crashing)
    res = run_simest(state, ToolContext(), {"confirm": True, "design": _design(), "n_rep": 2})
    assert res.writes["simest_results"]["status"] == "not_evaluable"
