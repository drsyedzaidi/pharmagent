"""F-06: a gate approval is bound to the specific gate it was issued for."""
from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from app.core.llm import MockLLM
from app.core.orchestrator import Orchestrator, StaleGateError
from app.core.store import SessionStore
from app.workflows import WORKFLOWS

SAMPLE = str(Path(__file__).parent.parent / "sample_data" / "oral_pk.csv")

TWO_GATES = {
    "name": "two_gates",
    "description": "test-only template with two cheap human gates",
    "steps": [
        {"agent": "data_manager", "tool": "load_dataset", "label": "Load dataset"},
        {"agent": "data_manager", "tool": "profile_pk_dataset", "label": "Profile data", "gate": True},
        {"agent": "data_manager", "tool": "validate_cdisc", "label": "Validate structure", "gate": True},
        {"agent": "data_manager", "tool": "generate_spaghetti_plot", "label": "Spaghetti plot"},
    ],
}


@pytest.fixture
def orch(monkeypatch):
    monkeypatch.setitem(WORKFLOWS, "two_gates", TWO_GATES)
    return Orchestrator(llm=MockLLM(), clock=lambda c=itertools.count(): f"t{next(c)}",
                        store=SessionStore(":memory:"))


def _paused(orch):
    sid = orch.create_session().id
    first = orch.start_workflow(sid, "two_gates", {"path": SAMPLE})
    assert first["status"] == "awaiting_review"
    return sid, first["review"]


def test_pending_review_carries_unique_gate_ids(orch):
    sid, gate1 = _paused(orch)
    assert gate1["gate_id"]
    second = orch.resume_workflow(sid, approve=True, actor="alice", gate_id=gate1["gate_id"])
    assert second["status"] == "awaiting_review"
    assert second["review"]["gate_id"] and second["review"]["gate_id"] != gate1["gate_id"]


def test_second_submit_for_consumed_gate_is_refused_and_leaves_gate_two_pending(orch):
    sid, gate1 = _paused(orch)
    second = orch.resume_workflow(sid, approve=True, actor="alice", gate_id=gate1["gate_id"])
    gate2 = second["review"]
    with pytest.raises(StaleGateError, match="refetch"):
        orch.resume_workflow(sid, approve=True, actor="alice", gate_id=gate1["gate_id"])
    sess = orch.get_session(sid)
    assert sess.pending_review == gate2
    assert not any(e.action.startswith("approved after step 2") for e in sess.audit.entries)
    done = orch.resume_workflow(sid, approve=True, actor="alice", gate_id=gate2["gate_id"])
    assert done["status"] == "complete"


def test_http_resume_requires_gate_id_and_returns_409_for_a_consumed_gate(orch, monkeypatch):
    from fastapi.testclient import TestClient

    import app.main as main
    monkeypatch.setattr(main, "orch", orch)
    client = TestClient(main.app)
    sid = client.post("/api/sessions").json()["id"]
    gate1 = client.post(f"/api/sessions/{sid}/workflow/start",
                        json={"workflow": "two_gates", "params": {"path": SAMPLE}}).json()["review"]
    assert client.post(f"/api/sessions/{sid}/workflow/resume", json={"approve": True}).status_code == 422
    ok = client.post(f"/api/sessions/{sid}/workflow/resume",
                     json={"approve": True, "gate_id": gate1["gate_id"]})
    assert ok.status_code == 200 and ok.json()["status"] == "awaiting_review"
    stale = client.post(f"/api/sessions/{sid}/workflow/resume",
                        json={"approve": True, "gate_id": gate1["gate_id"]})
    assert stale.status_code == 409
    assert orch.get_session(sid).pending_review == ok.json()["review"]
