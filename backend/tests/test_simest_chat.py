"""run_simest is reachable from the agent (chat) path -- as a PROPOSAL.

The guardrail "never run a real NLME fit synchronously from a chat turn, and
never from an automated loop" is preserved by construction:

  * `simulator` is now a routable agent, so a chat message about a
    simulation-estimation design check reaches the simulator, whose LLM turn
    can select `run_simest` and compose its `design` from the message.
  * `run_simest` is `expensive=True` (registry refuses it on the synchronous
    chat path, exactly like run_nlme) AND `proposable=True`: instead of being
    silently skipped, the refused call is recorded as `state.pending_tool`
    (audited), and NOTHING is computed.
  * The pharmacometrician approves or rejects the proposal. Approval is the
    human `confirm` the tool requires, and the run is submitted to the
    JobManager (admission-controlled) via `Orchestrator.run_tool`.
  * Bootstrap / SIR / profiling -- the other real-fit tools that used to rely
    only on the agent being unroutable -- are now `expensive=True` too, so
    they are still refused on the chat path (skipped, not proposed).

No test here runs a real fit: `app.compute.nlme.population_fit` is patched.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from app.agents.base import Agent
from app.agents.definitions import AGENTS, DESCRIPTIONS
from app.agents.supervisor import KEYWORDS, Supervisor
from app.core.audit import AuditChain
from app.core.llm import MockLLM
from app.core.orchestrator import Orchestrator
from app.core.pharmstate import AGENT_WRITE_FIELDS, PharmState
from app.core.store import SessionStore
from app.tools.base import ToolContext
from app.tools.builtins import default_registry

SAMPLE = str(Path(__file__).parent.parent / "sample_data" / "oral_pk.csv")
MODEL_KEY = "oral_1cmt"
DESIGN = {"n_subjects": 10, "obs_t": [0.5, 1, 2, 4, 8], "dose": 100.0, "n_doses": 1}
MESSAGE = "run a simulation-estimation check of this sampling design"


def _nlme() -> dict:
    return {
        "status": "ok", "model_key": MODEL_KEY,
        "theta": {"CL": 4.0, "V": 40.0, "KA": 1.0},
        "omega_cv_pct": {"CL": 30.0, "V": 20.0},
        "sigma": {"prop": 0.1, "add": 0.3},
        "iiv_params": ["CL", "V"], "error_model": "combined",
        "covariate_effects": [],
    }


class _PickTool:
    """Stub LLM: selects `tool_name` with `args` once, then stops."""

    def __init__(self, tool_name: str, args: dict | None = None) -> None:
        self.tool_name, self.args, self._served = tool_name, args or {}, False

    def classify(self, message, options, descriptions):
        return options[0] if options else "data_manager"

    def select_tool(self, agent, message, tools, state_summary):
        if self._served:
            return None
        self._served = True
        return {"name": self.tool_name, "input": dict(self.args)}


def _forbid_real_fit(monkeypatch):
    def boom(*a, **k):  # pragma: no cover - the assertion IS that this never runs
        raise AssertionError("a real population_fit was invoked from the chat path")
    monkeypatch.setattr("app.compute.nlme.population_fit", boom)


def _fake_fit(monkeypatch, calls: list | None = None):
    def fake(model_key, subjects, *, method="focei", iiv_params=None, error_model="proportional",
             max_iter=200, seed=20250614, compute_uncertainty=True, covariate_model=None):
        if calls is not None:
            calls.append({"method": method, "n_subjects": len(subjects)})
        return {"theta": {"CL": 4.0, "V": 40.0}, "theta_rse_pct": {"CL": 8.0, "V": 6.0},
                "converged": True}
    monkeypatch.setattr("app.compute.nlme.population_fit", fake)


def _orch(llm=None, store=None) -> Orchestrator:
    return Orchestrator(llm=llm or MockLLM(), clock=lambda c=itertools.count(): f"t{next(c)}",
                        store=store or SessionStore(":memory:"))


def _session_with_fit(orch: Orchestrator) -> str:
    sid = orch.create_session().id
    sess = orch.get_session(sid)
    sess.state = sess.state.model_copy(update={"dataset_id": "d1", "nlme_results": _nlme()})
    return sid


# ── routing: simulator is a real agent now ──────────────────────────────────

def test_simulator_is_a_routable_agent():
    assert "simulator" in AGENTS and "simulator" in DESCRIPTIONS and "simulator" in KEYWORDS
    assert AGENTS["simulator"].name == "simulator"


def test_supervisor_routes_simest_requests_to_simulator():
    sup = Supervisor(MockLLM())
    for msg in (MESSAGE, "simest: is 5 samples per subject enough?",
                "check the precision of this sampling design by simulation estimation"):
        assert sup.route(msg)[0] == "simulator", msg
    # unrelated routing is untouched
    assert sup.route("compute NCA AUC and Cmax")[0] == "nca"
    assert sup.route("load this csv dataset and profile it")[0] == "data_manager"


def test_run_simest_is_in_the_simulator_tool_list():
    names = {t.name for t in default_registry().for_agent("simulator")}
    assert "run_simest" in names


def test_supervisor_can_write_pending_tool():
    assert "pending_tool" in AGENT_WRITE_FIELDS["supervisor"]
    assert "pending_tool" not in AGENT_WRITE_FIELDS["simulator"]


# ── flags: expensive everywhere a real fit runs; proposable only for simest ─

def test_real_fit_tools_are_expensive():
    reg = default_registry()
    for name in ("run_simest", "run_bootstrap", "run_sir", "run_profile"):
        assert reg.get(name).expensive, name


def test_only_run_simest_is_proposable():
    reg = default_registry()
    assert reg.get("run_simest").proposable
    for name in ("run_bootstrap", "run_sir", "run_profile", "run_nlme", "run_scm",
                 "run_engine_comparison", "fit_pk_model", "simulate_pk_profile"):
        assert not reg.get(name).proposable, name


# ── chat turn: proposal, never execution ────────────────────────────────────

def test_chat_proposes_run_simest_and_computes_nothing(monkeypatch):
    _forbid_real_fit(monkeypatch)
    orch = _orch(llm=_PickTool("run_simest", {"design": DESIGN, "n_rep": 3}))
    sid = _session_with_fit(orch)
    out = orch.chat(sid, MESSAGE, actor="alice")

    assert out["agent"] == "simulator"
    pending = out["pending_tool"]
    assert pending["tool"] == "run_simest" and pending["agent"] == "simulator"
    assert pending["args"] == {"design": DESIGN, "n_rep": 3}
    assert pending["proposed_by"] == "alice"
    assert out["state"]["pending_tool"] == pending
    assert out["state"]["simest_results"] is None
    assert any(c.get("proposed") == "expensive" and c["tool"] == "run_simest"
               for c in out["tool_calls"])
    assert any("approve" in m.lower() for m in out["messages"])
    # the proposal itself is on the tamper-evident chain
    sess = orch.get_session(sid)
    assert sess.audit.verify()
    entries = sess.audit.to_list()
    assert any(e["tool"] == "run_simest" and e["action"] == "propose_tool" for e in entries)


def test_proposal_does_not_inject_confirm_on_the_llms_behalf(monkeypatch):
    _forbid_real_fit(monkeypatch)
    orch = _orch(llm=_PickTool("run_simest", {"design": DESIGN, "confirm": True}))
    sid = _session_with_fit(orch)
    out = orch.chat(sid, MESSAGE)
    # the LLM saying confirm=true is not a human confirmation: still only a proposal
    assert out["pending_tool"]["tool"] == "run_simest"
    assert out["state"]["simest_results"] is None


def test_non_proposable_expensive_tools_are_still_skipped_on_chat(monkeypatch):
    _forbid_real_fit(monkeypatch)
    for name in ("run_bootstrap", "run_sir", "run_profile"):
        res = Agent(name="simulator", system_prompt="").run_turn(
            state=PharmState(nlme_results=_nlme()), message="x", llm=_PickTool(name),
            registry=default_registry(), ctx=ToolContext(), audit=AuditChain(),
            clock=lambda: "t0", actor="t")
        assert any(c.get("skipped") == "expensive" for c in res.tool_calls), name
        assert res.proposal is None, name
        assert res.state.pending_tool is None, name


def test_mock_llm_proposes_run_simest_for_the_simulator_agent():
    choice = MockLLM().select_tool("simulator", MESSAGE, default_registry().for_agent("simulator"),
                                   {"nlme_results": "present"})
    assert choice and choice["name"] == "run_simest"
    assert MockLLM().select_tool("simulator", "hello", [], {}) is None


# ── decision: reject / approve ──────────────────────────────────────────────

def test_reject_clears_the_proposal_and_audits_it(monkeypatch):
    _forbid_real_fit(monkeypatch)
    orch = _orch(llm=_PickTool("run_simest", {"design": DESIGN}))
    sid = _session_with_fit(orch)
    orch.chat(sid, MESSAGE)
    out = orch.reject_pending_tool(sid, actor="bob", reason="wrong design")
    assert out["status"] == "rejected" and out["tool"] == "run_simest"
    assert out["state"]["pending_tool"] is None
    assert out["state"]["simest_results"] is None
    sess = orch.get_session(sid)
    assert sess.audit.verify()
    last = sess.audit.to_list()[-1]
    assert last["action"] == "reject_tool" and last["actor"] == "bob"
    assert last["reason"] == "wrong design"


def test_decision_without_a_proposal_raises():
    orch = _orch()
    sid = orch.create_session().id
    with pytest.raises(KeyError):
        orch.reject_pending_tool(sid, actor="bob")
    with pytest.raises(KeyError):
        orch.take_pending_tool(sid, actor="bob")


def test_take_returns_the_confirmed_call_and_clears_it_once(monkeypatch):
    _forbid_real_fit(monkeypatch)
    orch = _orch(llm=_PickTool("run_simest", {"design": DESIGN, "n_rep": 2}))
    sid = _session_with_fit(orch)
    orch.chat(sid, MESSAGE)
    call = orch.take_pending_tool(sid, actor="bob", reason="go")
    assert call["tool"] == "run_simest" and call["agent"] == "simulator"
    # human approval IS the tool's required confirm
    assert call["args"] == {"design": DESIGN, "n_rep": 2, "confirm": True}
    assert orch.get_session(sid).state.pending_tool is None
    last = orch.get_session(sid).audit.to_list()[-1]
    assert last["action"] == "approve_tool" and last["actor"] == "bob" and last["reason"] == "go"
    # a second approval (double click) has nothing to take -> cannot double-submit
    with pytest.raises(KeyError):
        orch.take_pending_tool(sid, actor="bob")


def test_approved_call_runs_through_run_tool_and_writes_results(monkeypatch):
    calls: list = []
    _fake_fit(monkeypatch, calls)
    orch = _orch(llm=_PickTool("run_simest", {"design": DESIGN, "n_rep": 2}))
    sid = _session_with_fit(orch)
    orch.chat(sid, MESSAGE)
    call = orch.take_pending_tool(sid, actor="bob")
    out = orch.run_tool(sid, call["tool"], call["agent"], call["args"], actor="bob")
    assert out["state"]["simest_results"]["status"] == "ok"
    assert out["state"]["simest_results"]["n_rep_completed"] == 2
    assert calls and all(c["n_subjects"] == DESIGN["n_subjects"] for c in calls)
    assert out["audit_ok"]


def test_a_new_proposal_replaces_the_old_one(monkeypatch):
    _forbid_real_fit(monkeypatch)
    orch = _orch(llm=_PickTool("run_simest", {"design": DESIGN, "n_rep": 1}))
    sid = _session_with_fit(orch)
    orch.chat(sid, MESSAGE)
    orch.llm = _PickTool("run_simest", {"design": DESIGN, "n_rep": 4})
    out = orch.chat(sid, MESSAGE)
    assert out["pending_tool"]["args"]["n_rep"] == 4
    # the abandoned proposal is audited, not silently dropped
    actions = [e["action"] for e in orch.get_session(sid).audit.to_list()]
    assert actions[-2:] == ["supersede_tool", "propose_tool"]
    assert orch.get_session(sid).audit.verify()


def test_pending_tool_survives_restart(monkeypatch, tmp_path):
    _forbid_real_fit(monkeypatch)
    db = str(tmp_path / "s.db")
    orch = _orch(llm=_PickTool("run_simest", {"design": DESIGN}), store=SessionStore(db))
    sid = _session_with_fit(orch)
    orch.chat(sid, MESSAGE)
    again = _orch(store=SessionStore(db))
    assert again.get_session(sid).state.pending_tool["tool"] == "run_simest"


# ── automated loops still cannot run it ─────────────────────────────────────

def test_skill_replay_still_refuses_run_simest(monkeypatch):
    _forbid_real_fit(monkeypatch)
    from app.core.skills import Skill
    orch = _orch()
    orch.skills.save(Skill(name="simest-probe", description="p", goal="p",
                           steps=[{"agent": "simulator", "tool": "run_simest",
                                   "args": {"design": DESIGN, "confirm": True}}],
                           source_session=None, owner="alice", created_at="t0", version=1))
    out = orch.run_skill("simest-probe", dataset_path=SAMPLE, owner="alice")
    step = [e for e in out["executed"] if e["tool"] == "run_simest"]
    assert step and step[0]["status"] == "error"
    assert out["state"]["simest_results"] is None


# ── HTTP ────────────────────────────────────────────────────────────────────

@pytest.fixture
def client(monkeypatch):
    from fastapi.testclient import TestClient

    import app.main as main
    main.orch = _orch(llm=_PickTool("run_simest", {"design": DESIGN, "n_rep": 2}))
    yield TestClient(main.app), main


def test_http_chat_then_approve_runs_as_a_job(client, monkeypatch):
    calls: list = []
    _fake_fit(monkeypatch, calls)
    c, main = client
    sid = c.post("/api/sessions").json()["id"]
    sess = main.orch.get_session(sid)
    sess.state = sess.state.model_copy(update={"dataset_id": "d1", "nlme_results": _nlme()})

    r = c.post(f"/api/sessions/{sid}/chat", json={"message": MESSAGE}).json()
    assert r["pending_tool"]["tool"] == "run_simest"
    assert r["state"]["simest_results"] is None and not calls

    r = c.post(f"/api/sessions/{sid}/chat/pending_tool", json={"approve": True}).json()
    assert r["status"] == "running" and r["kind"] == "chat_run_simest" and "job_id" in r
    job = _poll(c, sid, r["job_id"])
    assert job["status"] == "done"
    assert job["result"]["state"]["simest_results"]["status"] == "ok"
    assert calls
    # approving twice cannot submit twice
    assert c.post(f"/api/sessions/{sid}/chat/pending_tool", json={"approve": True}).status_code == 400


def test_http_reject_runs_inline_and_computes_nothing(client, monkeypatch):
    _forbid_real_fit(monkeypatch)
    c, main = client
    sid = c.post("/api/sessions").json()["id"]
    sess = main.orch.get_session(sid)
    sess.state = sess.state.model_copy(update={"dataset_id": "d1", "nlme_results": _nlme()})
    c.post(f"/api/sessions/{sid}/chat", json={"message": MESSAGE})
    r = c.post(f"/api/sessions/{sid}/chat/pending_tool",
               json={"approve": False, "reason": "no"}).json()
    assert r["status"] == "rejected" and r["state"]["pending_tool"] is None
    assert c.post(f"/api/sessions/{sid}/chat/pending_tool",
                  json={"approve": False}).status_code == 400


def _poll(c, sid, job_id, tries=200):
    import time
    for _ in range(tries):
        j = c.get(f"/api/sessions/{sid}/jobs/{job_id}").json()
        if j["status"] in ("done", "error"):
            return j
        time.sleep(0.05)
    raise AssertionError("job did not finish")
