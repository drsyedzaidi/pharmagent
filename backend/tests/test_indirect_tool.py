"""indirect_comparison tool: registry wiring, audit, state ownership, routing."""
import json
import math

import pytest

from app.agents.supervisor import Supervisor
from app.core.audit import AuditChain, hash_payload
from app.core.llm import MockLLM
from app.core.pharmstate import AGENT_WRITE_FIELDS, PharmState, PharmStateError, apply_writes
from app.tools.base import ToolContext
from app.tools.builtins import default_registry

AB = {"estimate": 0.50, "se": 0.20}
CB = {"estimate": 0.80, "se": 0.30}


def _run(state, args, audit=None):
    audit = audit if audit is not None else AuditChain()
    new_state, res = default_registry().execute(
        "indirect_comparison", state=state, ctx=ToolContext(), args=args,
        audit=audit, timestamp="t0", actor="tester")
    return new_state, res, audit


def test_registered_cheap_and_owned_by_the_statistician():
    tool = default_registry().get("indirect_comparison")
    assert tool.agent == "statistician"
    assert tool.expensive is False and tool.proposable is False
    props = tool.input_schema["properties"]
    assert {"ab", "cb", "scale"} <= set(props)
    assert set(tool.input_schema["required"]) == {"ab", "cb", "scale"}


def test_runs_through_the_choke_point_audits_and_writes_state():
    state, res, audit = _run(PharmState(), {"ab": AB, "cb": CB, "scale": "HR"})
    assert res.result["estimate"] == pytest.approx(0.625, abs=1e-12)
    assert res.result["se"] == pytest.approx(math.sqrt(0.13), abs=1e-12)
    comps = state.indirect_results["comparisons"]
    assert len(comps) == 1 and comps[0] == res.result
    # exactly one audit entry, whose output hash is the hash of the stored comparison
    assert [e.tool for e in audit.entries] == ["indirect_comparison"]
    assert audit.entries[0].agent == "statistician"
    assert audit.entries[0].outputs_hash == hash_payload(comps[0])
    assert audit.entries[0].actor == "tester"
    assert audit.verify()
    json.dumps(state.model_dump())


def test_successive_comparisons_accumulate_without_mutating_prior_state():
    s1, _, audit = _run(PharmState(), {"ab": AB, "cb": CB, "scale": "HR"})
    s2, _, _ = _run(s1, {"ab": {"estimate": 5.0, "se": 0.3}, "cb": {"estimate": 2.0, "se": 0.4},
                         "scale": "MD", "treatment_a": "Drug A", "treatment_c": "Drug C"}, audit)
    assert len(s1.indirect_results["comparisons"]) == 1       # prior state untouched
    scales = [c["scale"] for c in s2.indirect_results["comparisons"]]
    assert scales == ["HR", "MD"]
    assert s2.indirect_results["comparisons"][1]["contrast"] == "Drug A vs Drug C via B"


def test_history_is_bounded():
    state = PharmState()
    audit = AuditChain()
    for _ in range(25):
        state, _, _ = _run(state, {"ab": AB, "cb": CB, "scale": "HR"}, audit)
    assert len(state.indirect_results["comparisons"]) == 20


def test_invalid_input_raises_value_error_and_writes_nothing():
    audit = AuditChain()
    with pytest.raises(ValueError, match="se must be > 0"):
        _run(PharmState(), {"ab": {"estimate": 0.5, "se": 0.0}, "cb": CB, "scale": "HR"}, audit)
    assert audit.entries == []


def test_ci_style_input_and_summary_text():
    _, res, _ = _run(PharmState(), {
        "ab": {"estimate": 0.5, "ci_lower": 0.3, "ci_upper": 0.8333, "ci_level": 0.90},
        "cb": CB, "scale": "hr", "ci_level": 0.90})
    assert res.result["scale"] == "HR" and res.result["ci_level"] == 0.90
    assert res.result["inputs"]["ab"]["ci_level"] == 0.90
    assert "A vs C via B" in res.summary and "HR" in res.summary
    assert "A vs B input CI 90%" in res.summary


def test_output_ci_level_never_silently_reinterprets_an_input_ci():
    """A 90% CI pasted from a paper with only the top-level ci_level=0.90 set used to be
    read as a 95% CI (SE ~16% too small). The ambiguous call is refused."""
    leg = {"estimate": 0.62, "ci_lower": 0.45, "ci_upper": 0.88}
    audit = AuditChain()
    with pytest.raises(ValueError, match="ab.ci_level"):
        _run(PharmState(), {"ab": leg, "cb": {"estimate": 0.8, "se": 0.1}, "scale": "HR",
                            "ci_level": 0.90}, audit)
    assert audit.entries == []
    _, res, _ = _run(PharmState(), {"ab": {**leg, "ci_level": 0.90}, "cb": {"estimate": 0.8, "se": 0.1},
                                    "scale": "HR", "ci_level": 0.90})
    assert res.result["inputs"]["ab"]["se"] == pytest.approx(
        (math.log(0.88) - math.log(0.45)) / (2 * 1.6448536269514722), rel=1e-9)


def test_summary_keeps_a_fractional_ci_level():
    _, res, _ = _run(PharmState(), {"ab": {"estimate": 0.5, "ci_lower": 0.3, "ci_upper": 0.8333,
                                           "ci_level": 0.975}, "cb": CB, "scale": "HR", "ci_level": 0.975})
    assert "(97.5% CI " in res.summary and "input CI 97.5%" in res.summary
    assert "98%" not in res.summary


def test_a_defaulted_input_ci_level_is_echoed_as_assumed():
    _, res, _ = _run(PharmState(), {"ab": {"estimate": 0.62, "ci_lower": 0.45, "ci_upper": 0.88},
                                    "cb": CB, "scale": "HR"})
    assert res.result["inputs"]["ab"]["ci_level_source"] == "default"
    assert "A vs B input CI assumed 95%" in res.summary


def test_schema_says_which_ci_level_is_which():
    props = default_registry().get("indirect_comparison").input_schema["properties"]
    assert "OUTPUT" in props["ci_level"]["description"]
    assert "THIS" in props["ab"]["properties"]["ci_level"]["description"]


def test_only_the_statistician_may_write_indirect_results():
    assert "indirect_results" in AGENT_WRITE_FIELDS["statistician"]
    with pytest.raises(PharmStateError):
        apply_writes(PharmState(), "nca", {"indirect_results": {"comparisons": []}})


@pytest.mark.parametrize("msg", [
    "run a Bucher indirect comparison of drug A vs drug C via placebo",
    "do an adjusted indirect treatment comparison",
])
def test_supervisor_routes_indirect_comparison_to_the_statistician(msg):
    assert Supervisor(MockLLM()).route(msg)[0] == "statistician"


def test_null_optional_arguments_fall_back_to_defaults():
    _, res, _ = _run(PharmState(), {"ab": AB, "cb": CB, "scale": "HR", "ci_level": None,
                                    "treatment_a": None, "treatment_c": ""})
    assert res.result["ci_level"] == 0.95 and res.result["contrast"] == "A vs C via B"
