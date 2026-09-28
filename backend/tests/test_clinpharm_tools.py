"""Clinical-pharmacology calculator tools: validation at the boundary, state
writes (last result + bounded history), routing to the `clinpharm` agent, and
the chat path (cheap tools, never expensive)."""
from __future__ import annotations

import pytest

from app.agents.definitions import AGENTS, DESCRIPTIONS
from app.agents.supervisor import KEYWORDS, Supervisor
from app.core.llm import MockLLM
from app.core.pharmstate import AGENT_WRITE_FIELDS, PharmState
from app.tools.base import ToolContext
from app.tools.builtins import default_registry
from app.tools.clinpharm_tools import TOOLS, calc_half_life, calc_renal_function

TOOL_NAMES = [t.name for t in TOOLS]


def _run(name: str, args: dict, state: PharmState | None = None):
    reg = default_registry()
    st = state or PharmState()
    return reg.execute(name, state=st, ctx=ToolContext(), args=args,
                       audit=__import__("app.core.audit", fromlist=["AuditChain"]).AuditChain(),
                       timestamp="t0", actor="t")


# ── registration / routing ──────────────────────────────────────────────────

def test_all_tools_registered_under_clinpharm_and_cheap():
    reg = default_registry()
    for n in TOOL_NAMES:
        t = reg.get(n)
        assert t.agent == "clinpharm" and not t.expensive and not t.proposable, n
    assert "clinpharm" in AGENTS and "clinpharm" in DESCRIPTIONS and "clinpharm" in KEYWORDS
    assert {"clinpharm_results", "clinpharm_history"} <= AGENT_WRITE_FIELDS["clinpharm"]


def test_supervisor_routes_calculator_questions():
    sup = Supervisor(MockLLM())
    for msg in ("what is the half-life if ke is 0.1", "creatinine clearance for a 70 kg 60 year old man",
                "loading dose to reach 10 mg/L with V 40 L", "sample size for a bioequivalence study with 25% CV",
                "convert 12 mg/L to micromolar, MW 305", "scale clearance allometrically from 70 kg to 20 kg"):
        assert sup.route(msg)[0] == "clinpharm", msg
    assert sup.route("compute NCA AUC and Cmax")[0] == "nca"
    assert sup.route("run a bioequivalence assessment test vs reference")[0] == "be"


def test_mock_llm_picks_a_calculator_by_keyword():
    tools = default_registry().for_agent("clinpharm")
    m = MockLLM()
    assert m.select_tool("clinpharm", "half-life from ke 0.2", tools, {})["name"] == "calc_half_life"
    assert m.select_tool("clinpharm", "creatinine clearance", tools, {})["name"] == "calc_renal_function"
    assert m.select_tool("clinpharm", "sample size for BE", tools, {})["name"] == "calc_be_sample_size"
    assert m.select_tool("clinpharm", "hello", tools, {}) is None


# ── validation ──────────────────────────────────────────────────────────────

def test_missing_required_input_is_a_clear_error():
    with pytest.raises(ValueError, match="missing required input: tau"):
        _run("calc_accumulation", {"ke": 0.1})
    with pytest.raises(ValueError, match="must be a number"):
        calc_half_life(PharmState(), ToolContext(), {"half_life": "six"})
    with pytest.raises(ValueError, match="sex must be"):
        calc_renal_function(PharmState(), ToolContext(), {"age": 60, "scr_mg_dl": 1.0})
    with pytest.raises(ValueError, match="give half_life"):
        calc_half_life(PharmState(), ToolContext(), {})


# ── results + state ─────────────────────────────────────────────────────────

def test_half_life_writes_result_and_history():
    st, res = _run("calc_half_life", {"half_life": 6})
    r = st.clinpharm_results
    assert r["tool"] == "calc_half_life" and r["status"] == "ok"
    assert abs(r["outputs"]["ke"] - 0.1155) < 1e-3 and r["outputs"]["half_life"] == 6
    assert "ln2" in r["formula"] and "t½" in res.summary
    assert st.clinpharm_history[-1] is r


def test_history_is_bounded_and_immutable():
    st = PharmState()
    for i in range(25):
        st, _ = _run("calc_half_life", {"half_life": i + 1}, st)
    assert len(st.clinpharm_history) == 20
    assert st.clinpharm_history[-1]["inputs"]["half_life"] == 25


def test_renal_function_with_dose_adjustment():
    st, res = _run("calc_renal_function", {"age": 40, "weight_kg": 80, "scr_mg_dl": 1.0, "sex": "male",
                                           "normal_dose": 100, "reference_crcl": 120})
    o = st.clinpharm_results["outputs"]
    assert abs(o["crcl_cockcroft_gault"] - 8000 / 72) < 1e-3
    assert 90 < o["egfr_ckd_epi_2021"] < 130
    assert abs(o["adjusted_dose"] - 100 * min(8000 / 72 / 120, 1)) < 1e-3
    assert "not a substitute" in st.clinpharm_results["note"]


def test_dose_regimen_both_parts():
    st, _ = _run("calc_dose_regimen", {"target_conc": 2, "volume": 50, "cavg_ss": 1, "clearance": 5, "tau": 12,
                                       "bioavailability": 0.5})
    o = st.clinpharm_results["outputs"]
    assert o["loading_dose"] == 200 and o["maintenance_dose"] == 120 and o["dose_rate"] == 10


def test_be_sample_size_accepts_percent_cv():
    st, _ = _run("calc_be_sample_size", {"cv_intra": 25})
    assert st.clinpharm_results["inputs"]["cv_intra"] == 0.25
    assert 24 <= st.clinpharm_results["outputs"]["total"] <= 34


def test_quick_one_compartment_oral_profile():
    st, res = _run("quick_one_compartment", {"route": "oral", "dose": 100, "volume": 30, "ka": 1.2, "half_life": 4.6})
    o = st.clinpharm_results["outputs"]
    assert len(o["profile"]) == 121 and o["profile"][0]["c"] == 0
    assert o["cmax"] > 0 and o["tmax"] > 0 and "Cmax" in res.summary


def test_conversion_round_trip():
    st, _ = _run("convert_concentration", {"mg_per_l": 12, "molar_mass": 305.4})
    um = st.clinpharm_results["outputs"]["umol_per_l"]
    st2, _ = _run("convert_concentration", {"umol_per_l": um, "molar_mass": 305.4})
    assert abs(st2.clinpharm_results["outputs"]["mg_per_l"] - 12) < 1e-3


def test_allometric_default_exponent_by_kind():
    st, _ = _run("calc_allometric", {"value": 10, "from_bw": 70, "to_bw": 20, "kind": "V"})
    assert st.clinpharm_results["inputs"]["exponent"] == 1.0
    st, _ = _run("calc_allometric", {"value": 10, "from_bw": 70, "to_bw": 20})
    assert st.clinpharm_results["inputs"]["exponent"] == 0.75


# ── chat path end to end (MockLLM composes nothing; the tool reports what it needs) ──

def test_chat_turn_reaches_a_calculator():
    from app.core.orchestrator import Orchestrator
    from app.core.store import SessionStore
    o = Orchestrator(llm=MockLLM(), store=SessionStore(":memory:"))
    sid = o.create_session().id
    out = o.chat(sid, "what is the half-life if ke is 0.1")
    assert out["agent"] == "clinpharm"
    assert out["tool_calls"] and out["tool_calls"][0]["tool"] == "calc_half_life"


# ── HTTP calculator endpoint ────────────────────────────────────────────────

def test_calc_endpoint_runs_and_rejects_non_calculators():
    import itertools

    from fastapi.testclient import TestClient

    import app.main as main
    from app.core.orchestrator import Orchestrator
    from app.core.store import SessionStore
    main.orch = Orchestrator(llm=MockLLM(), clock=lambda c=itertools.count(): f"t{next(c)}",
                             store=SessionStore(":memory:"))
    c = TestClient(main.app)
    sid = c.post("/api/sessions").json()["id"]
    r = c.post(f"/api/sessions/{sid}/calc", json={"tool": "calc_half_life", "args": {"half_life": 6}})
    assert r.status_code == 200 and r.json()["result"]["outputs"]["half_life"] == 6
    assert r.json()["state"]["clinpharm_results"]["tool"] == "calc_half_life"
    assert c.post(f"/api/sessions/{sid}/calc", json={"tool": "run_nlme", "args": {}}).status_code == 400
    assert c.post(f"/api/sessions/{sid}/calc", json={"tool": "nope", "args": {}}).status_code == 404
    bad = c.post(f"/api/sessions/{sid}/calc", json={"tool": "calc_accumulation", "args": {"ke": 0.1}})
    assert bad.status_code == 400 and "tau" in bad.text
