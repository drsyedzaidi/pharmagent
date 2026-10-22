"""Tests for Part 2 Agentic Autonomy upgrades:
1. Rich, privacy-safe diagnostic state summaries (`_state_summary`)
2. Intra-turn & cross-agent observation scratchpad (`turn_observations`)
3. Multi-agent pipeline handoff execution in `Orchestrator.chat` (`route_plan`)
4. Closed-loop deterministic remediation in `Orchestrator.review_loop(auto_remediate=True)`
"""
from __future__ import annotations

import itertools
from pathlib import Path

from app.agents.definitions import AGENTS
from app.core.audit import AuditChain
from app.core.llm import MockLLM
from app.core.orchestrator import Orchestrator
from app.core.pharmstate import PharmState
from app.core.store import SessionStore
from app.tools.base import ToolContext
from app.tools.builtins import default_registry

SAMPLE = str(Path(__file__).parent.parent / "sample_data" / "oral_pk.csv")


def _orch(llm=None) -> Orchestrator:
    counter = itertools.count()
    return Orchestrator(
        llm=llm or MockLLM(),
        clock=lambda: f"t{next(counter)}",
        store=SessionStore(":memory:"),
    )


def test_rich_state_summary_exposes_diagnostics_never_raw_rows():
    state = PharmState(
        dataset_id="ds_1",
        dataset_metadata={
            "n_subjects": 6,
            "columns": ["ID", "TIME", "DV", "AMT"],
            "detected_roles": {"ID": "ID", "TIME": "TIME", "DV": "DV", "AMT": "AMT"},
        },
        nca_parameters=[
            {"subject": 1, "Cmax": 10.0, "lambda_z": 0.1, "pct_AUC_extrap": 5.0},
            {"subject": 2, "Cmax": 12.0, "lambda_z": None, "pct_AUC_extrap": 25.0},
        ],
        pk_model_results={
            "status": "ok",
            "mode": "compare",
            "best": {"model_key": "oral_1cmt", "n_subjects": 6, "n_converged": 6},
            "ranking": [
                {"model_key": "oral_1cmt", "aic": 100.0},
                {"model_key": "oral_2cmt", "aic": 104.5},
            ],
        },
        nlme_results={
            "status": "ok",
            "converged": True,
            "condition_number": 42.0,
            "shrinkage_pct": {"CL": 12.0, "V": 35.0},
            "theta_rse_pct": {"CL": 8.0, "V": 15.0},
        },
        review_results={
            "status": "FINDINGS BLOCK GOAL",
            "goal_met": False,
            "counts": {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 0, "LOW": 0},
            "findings": [
                {"id": "nca-extrap-2", "severity": "HIGH", "resolved": False},
                {"id": "nlme-shrink-V", "severity": "MEDIUM", "resolved": False},
            ],
        },
    )
    summary = AGENTS["nca"]._state_summary(
        state, observations=[{"agent": "data_manager", "tool": "load_dataset", "status": "ok"}]
    )
    assert summary["n_subjects"] == 6
    assert summary["columns"] == ["ID", "TIME", "DV", "AMT"]
    assert summary["nca_diagnostics"] == {
        "n_subjects": 2,
        "missing_lambda_z": 1,
        "high_extrap_gt20pct": 1,
    }
    assert summary["pk_model_diagnostics"] == {
        "mode": "compare",
        "best_model": "oral_1cmt",
        "n_subjects": 6,
        "n_converged": 6,
        "aic_delta_runner_up": 4.5,
    }
    assert summary["nlme_diagnostics"] == {
        "converged": True,
        "condition_number": 42.0,
        "max_shrinkage_pct": 35.0,
        "max_theta_rse_pct": 15.0,
    }
    assert summary["review_diagnostics"] == {
        "status": "FINDINGS BLOCK GOAL",
        "goal_met": False,
        "counts": {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 0, "LOW": 0},
        "blocking_ids": ["nca-extrap-2"],
    }
    assert summary["turn_observations"] == [
        {"agent": "data_manager", "tool": "load_dataset", "status": "ok"}
    ]


def test_turn_observations_are_fed_back_across_steps_in_a_turn():
    """An agent executing multiple tools in a single turn receives its prior step's
    observation in state_summary['turn_observations']."""
    seen_summaries: list[dict] = []

    class _TwoStepLLM:
        def __init__(self) -> None:
            self.step = 0

        def select_tool(self, agent, message, tools, state_summary):
            seen_summaries.append(state_summary)
            self.step += 1
            if self.step == 1:
                return {"name": "list_pk_models", "input": {"family": "pk"}}
            if self.step == 2:
                return {"name": "list_pk_models", "input": {"family": "pd"}}
            return None

    res = AGENTS["modeler"].run_turn(
        state=PharmState(),
        message="list pk then pd models",
        llm=_TwoStepLLM(),
        registry=default_registry(),
        ctx=ToolContext(),
        audit=AuditChain(),
        clock=lambda: "t0",
        actor="tester",
    )
    assert len(res.tool_calls) == 2
    assert seen_summaries[0]["turn_observations"] is None
    assert len(seen_summaries[1]["turn_observations"]) == 1
    assert seen_summaries[1]["turn_observations"][0]["tool"] == "list_pk_models"
    assert seen_summaries[2]["turn_observations"][1]["tool"] == "list_pk_models"


def test_multi_agent_chat_handoff_executes_pipeline_in_one_turn():
    """A compound prompt ('load <csv> then compute NCA and then run the QC checklist
    and then generate the report') chains across data_manager -> nca -> qc -> report
    in a single chat turn, with full audit and skill command logging."""
    orch = _orch()
    sid = orch.create_session(owner="alice").id
    out = orch.chat(
        sid,
        f"load {SAMPLE} and then compute NCA and then run the QC checklist and then generate the report",
        actor="alice",
    )
    assert out["agents_executed"] == ["data_manager", "nca", "qc", "report"]
    executed_tools = [tc["tool"] for tc in out["tool_calls"] if "summary" in tc]
    assert executed_tools == [
        "load_dataset",
        "profile_pk_dataset",
        "compute_nca",
        "run_qc",
        "generate_report",
    ]
    assert out["state"]["nca_parameters"] is not None
    assert out["state"]["qc_verdict"] is not None
    assert out["state"]["report_path"] is not None
    sess = orch.get_session(sid)
    assert sess.audit.verify() is True
    # Skill capture from this single multi-agent chat turn works out of the box
    skill = orch.capture_skill(sid, "autonomous-nca-report", actor="alice")["skill"]
    assert [s["tool"] for s in skill["steps"]] == ["compute_nca", "run_qc", "generate_report"]


def test_multi_agent_handoff_stops_at_human_proposal_gate():
    """If a multi-agent prompt reaches a proposable expensive tool (e.g. bootstrap),
    execution halts cleanly at the proposal gate without running downstream steps."""
    orch = _orch()
    sid = orch.create_session(owner="alice").id
    out = orch.chat(
        sid,
        f"load {SAMPLE} and then run NCA and then bootstrap the parameter uncertainty and then write the report",
        actor="alice",
    )
    assert out["agents_executed"] == ["data_manager", "nca", "simulator"]
    assert out["pending_tool"] is not None
    assert out["pending_tool"]["tool"] == "run_bootstrap"
    assert out["state"]["report_path"] is None


def test_review_loop_auto_remediate_resolves_incomplete_and_stale_memo(tmp_path, monkeypatch):
    """When auto_remediate=True, review_loop deterministically runs compute_nca if
    status is INCOMPLETE on a loaded dataset, and rebuilds a stale briefing memo
    when analysis state updated after the memo was built."""
    from app.config import settings
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    orch = _orch()
    sid = orch.create_session(owner="alice").id
    orch.run_tool(sid, "load_dataset", "data_manager", {"path": SAMPLE}, actor="alice")

    # 1. Dataset is loaded, but no analysis has run -> pass 1 is INCOMPLETE,
    #    auto_remediate runs compute_nca, pass 2 achieves GOAL MET.
    loop_res = orch.review_loop(sid, max_iter=3, actor="alice", auto_remediate=True)
    assert loop_res["goal_met"] is True
    assert loop_res["status"] == "GOAL MET"
    assert loop_res["iterations"] == 2
    assert loop_res["remediations"] == [{"tool": "compute_nca", "reason": "status=INCOMPLETE"}]
    assert loop_res["audit_ok"] is True

    # 2. Build a briefing memo from full_state, then update an analysis value so
    #    memo-stale fires without untraced numbers.
    from tests.memo_fixtures import full_chain, full_state
    sess = orch.get_session(sid)
    sess.state = full_state().model_copy(update={"session_id": sess.id,    # keep its run-provenance record
                                               "run_provenance": sess.state.run_provenance})
    sess.audit = full_chain(genesis=(sess.id, dict(sess.state.run_provenance)))
    orch.run_tool(sid, "build_briefing_memo", "report", {}, actor="alice")
    nl = {**sess.state.nlme_results, "ofv": -220.2}
    sess.state = sess.state.model_copy(update={"nlme_results": nl})
    # Verify without auto_remediate it fails with memo-stale
    without_rem = orch.review_loop(sid, max_iter=3, actor="alice", auto_remediate=False)
    assert without_rem["goal_met"] is False
    assert [f["id"] for f in without_rem["findings"] if f["id"].startswith("memo-")] == ["memo-stale"]

    # With auto_remediate=True, it rebuilds the briefing memo and converges!
    with_rem = orch.review_loop(sid, max_iter=3, actor="alice", auto_remediate=True)
    assert with_rem["remediations"] == [{"tool": "build_briefing_memo", "reason": "finding=memo-stale"}]
    assert not [f for f in with_rem["findings"] if f["id"].startswith("memo-")]
