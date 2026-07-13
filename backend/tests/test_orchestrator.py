"""End-to-end orchestration: routing, the NCA workflow, and the review gate.

Runs entirely with the deterministic MockLLM (no API key), proving the whole
spine: routing → schema-privacy → tools → PharmState → audit → review gate →
report.
"""
import itertools
from pathlib import Path

from app.agents.supervisor import Supervisor
from app.core.llm import MockLLM
from app.core.orchestrator import Orchestrator
from app.core.store import SessionStore

SAMPLE = str(Path(__file__).parent.parent / "sample_data" / "oral_pk.csv")


def _orch():
    counter = itertools.count()
    return Orchestrator(llm=MockLLM(), clock=lambda: f"t{next(counter)}",
                        store=SessionStore(":memory:"))


def test_supervisor_routing():
    sup = Supervisor(MockLLM())
    assert sup.route("compute NCA AUC and Cmax")[0] == "nca"
    assert sup.route("load this csv dataset and profile it")[0] == "data_manager"
    assert sup.route("run a QC review checklist")[0] == "qc"
    assert sup.route("generate the report docx")[0] == "report"


def test_workflow_pauses_at_review_gate():
    orch = _orch()
    sid = orch.create_session().id
    out = orch.start_workflow(sid, "nca_full", {"path": SAMPLE})
    assert out["status"] == "awaiting_review"
    st = out["state"]
    assert st["dataset_id"] is not None
    assert st["dataset_metadata"]["n_subjects"] == 12
    assert st["nca_parameters"] and len(st["nca_parameters"]) == 12
    assert st["qc_verdict"] in {"PASS", "CONDITIONAL PASS", "FAIL"}
    assert out["audit_ok"] is True
    # report not yet generated (gated)
    assert st["report_path"] is None


def test_workflow_resume_completes_and_reports():
    orch = _orch()
    sid = orch.create_session().id
    orch.start_workflow(sid, "nca_full", {"path": SAMPLE})
    out = orch.resume_workflow(sid, approve=True)
    assert out["status"] == "complete"
    assert out["audit_ok"] is True
    report = out["state"]["report_path"]
    assert report and Path(report).exists()


def test_privacy_metadata_only():
    """dataset_metadata must not carry raw row values."""
    orch = _orch()
    sid = orch.create_session().id
    out = orch.start_workflow(sid, "nca_full", {"path": SAMPLE})
    meta = out["state"]["dataset_metadata"]
    assert "privacy" in meta
    # metadata has columns + aggregate summaries, never a 'rows'/'data' payload
    assert "rows" not in meta and "data" not in meta


def test_persistence_survives_restart(tmp_path):
    """A session (state + audit chain) reloads from the DB after a 'restart'."""
    db = str(tmp_path / "pa.db")
    o1 = Orchestrator(llm=MockLLM(), clock=lambda c=itertools.count(): f"t{next(c)}",
                      store=SessionStore(db))
    sid = o1.create_session().id
    o1.start_workflow(sid, "nca_full", {"path": SAMPLE})
    # fresh orchestrator over the SAME db file == process restart
    o2 = Orchestrator(llm=MockLLM(), clock=lambda c=itertools.count(): f"t{next(c)}",
                      store=SessionStore(db))
    s = o2.get_session(sid)
    assert s.state.nca_parameters and len(s.state.nca_parameters) == 12
    assert s.state.qc_verdict in {"PASS", "CONDITIONAL PASS", "FAIL"}
    assert s.audit.verify() is True            # hash chain intact across reload
    assert sid in o2.sessions
    # dataset re-hydrated from disk so analysis can continue
    assert s.state.dataset_id in s.ctx.dataset_store


def test_session_ownership_enforced(tmp_path):
    from app.core.orchestrator import AccessError
    o = Orchestrator(llm=MockLLM(), clock=lambda c=itertools.count(): f"t{next(c)}",
                     store=SessionStore(str(tmp_path / "o.db")))
    sid = o.create_session(owner="alice").id
    assert o.get_session(sid, owner="alice").owner == "alice"
    import pytest as _pytest
    with _pytest.raises(AccessError):
        o.get_session(sid, owner="bob")


def test_chat_routes_and_loads():
    orch = _orch()
    sid = orch.create_session().id
    res = orch.chat(sid, f"load and profile the dataset {SAMPLE}")
    assert res["agent"] == "data_manager"
    assert res["state"]["dataset_id"] is not None
    assert res["state"]["data_quality"] is not None
