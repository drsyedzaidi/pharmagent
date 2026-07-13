"""Skills: capture a session's analysis sequence, store/version it, replay on new data.

Covers the distill rules, the SkillStore, capture from a real session command log,
and end-to-end replay reproducing the analysis on a fresh dataset.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.llm import MockLLM
from app.core.orchestrator import Orchestrator
from app.core.skills import Skill, SkillStore, distill_steps
from app.core.store import SessionStore

SAMPLE = str(Path(__file__).parent.parent / "sample_data" / "oral_pk.csv")


def _orch():
    counter = itertools.count()
    return Orchestrator(llm=MockLLM(), clock=lambda: f"t{next(counter)}",
                        store=SessionStore(":memory:"))


# ── distill ─────────────────────────────────────────────────────────────────────

def test_distill_drops_io_and_profiling():
    log = [
        {"agent": "data_manager", "tool": "load_dataset", "args": {"path": "x"}},
        {"agent": "data_manager", "tool": "profile_pk_dataset", "args": {}},
        {"agent": "data_manager", "tool": "validate_cdisc", "args": {}},
        {"agent": "data_manager", "tool": "generate_spaghetti_plot", "args": {}},
        {"agent": "nca", "tool": "compute_nca", "args": {}},
        {"agent": "reviewer", "tool": "adversarial_review", "args": {}},
    ]
    steps = distill_steps(log)
    tools = [s["tool"] for s in steps]
    assert tools == ["compute_nca", "adversarial_review"]


def test_distill_collapses_consecutive_duplicates():
    log = [
        {"agent": "modeler", "tool": "fit_pk_model", "args": {"model_key": "oral_1cmt"}},
        {"agent": "modeler", "tool": "fit_pk_model", "args": {"model_key": "oral_2cmt"}},
        {"agent": "nca", "tool": "compute_nca", "args": {}},
    ]
    steps = distill_steps(log)
    # the two consecutive fits collapse to the last (winning) one
    assert [s["tool"] for s in steps] == ["fit_pk_model", "compute_nca"]
    assert steps[0]["args"] == {"model_key": "oral_2cmt"}


# ── SkillStore ──────────────────────────────────────────────────────────────────

def _skill(name="s1", steps=None):
    return Skill(name=name, description="d", goal="g",
                 steps=steps or [{"agent": "nca", "tool": "compute_nca", "args": {}}],
                 source_session="sess_x", owner=None, created_at="t0", version=1)


def test_store_save_get_list_delete():
    store = SkillStore(":memory:")
    store.save(_skill("alpha"))
    store.save(_skill("beta"))
    assert {s.name for s in store.list()} == {"alpha", "beta"}
    assert store.get("alpha").description == "d"
    assert store.delete("alpha") is True
    assert store.get("alpha") is None
    assert store.delete("alpha") is False


def test_store_resave_bumps_version():
    store = SkillStore(":memory:")
    store.save(_skill("v"))
    assert store.get("v").version == 1
    store.save(_skill("v"))            # re-capture under the same name
    assert store.get("v").version == 2


def test_skill_markdown_renders_steps():
    s = _skill("mdtest", steps=[
        {"agent": "nca", "tool": "compute_nca", "args": {}},
        {"agent": "modeler", "tool": "fit_pk_model", "args": {"model_key": "oral_1cmt"}},
    ])
    md = s.to_markdown()
    assert "# Skill: mdtest" in md
    assert "compute_nca" in md and "fit_pk_model" in md
    assert "oral_1cmt" in md


# ── capture + replay via the orchestrator ────────────────────────────────────────

def _run_nca_session(orch):
    sid = orch.create_session(owner="alice").id
    orch.start_workflow(sid, "nca_full", {"path": SAMPLE}, actor="alice")  # pauses at QC gate
    orch.resume_workflow(sid, approve=True, actor="alice", reason="ok")    # finishes
    return sid


def test_capture_requires_commands():
    orch = _orch()
    sid = orch.create_session().id
    with pytest.raises(ValueError):
        orch.capture_skill(sid, "empty")


def test_capture_then_inspect_steps():
    orch = _orch()
    sid = _run_nca_session(orch)
    out = orch.capture_skill(sid, "nca-pipeline", description="single-dose oral",
                             goal="zero CRITICAL/HIGH", actor="alice")
    tools = [s["tool"] for s in out["skill"]["steps"]]
    assert "compute_nca" in tools
    assert "adversarial_review" in tools
    assert "load_dataset" not in tools          # dropped by distill
    assert out["skill"]["source_session"] == sid


def test_replay_reproduces_nca_on_fresh_session():
    orch = _orch()
    sid = _run_nca_session(orch)
    orch.capture_skill(sid, "replayme", actor="alice")
    out = orch.run_skill("replayme", dataset_path=SAMPLE, owner="alice", actor="alice")
    assert out["session_id"] != sid                       # a brand-new session
    assert out["state"]["nca_parameters"]                 # analysis reproduced
    assert out["state"]["review_results"] is not None     # reviewer replayed too
    assert all(s["status"] == "ok" for s in out["executed"])
    assert out["audit_ok"] is True


def test_run_unknown_skill_raises():
    orch = _orch()
    with pytest.raises(KeyError):
        orch.run_skill("nope", dataset_path=SAMPLE)


# ── HTTP surface ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def client():
    from app.main import app
    return TestClient(app)


def _http_nca_session(client) -> str:
    sid = client.post("/api/sessions").json()["id"]
    from app.main import orch
    orch.start_workflow(sid, "nca_full", {"path": SAMPLE})
    orch.resume_workflow(sid, approve=True)
    return sid


def test_http_capture_list_get_run_delete(client):
    sid = _http_nca_session(client)
    name = "http-skill-xyz"
    # capture
    cap = client.post(f"/api/sessions/{sid}/capture-skill",
                      json={"name": name, "description": "demo", "goal": "clean"})
    assert cap.status_code == 200
    assert any(s["tool"] == "compute_nca" for s in cap.json()["skill"]["steps"])
    # list + get
    assert name in [s["name"] for s in client.get("/api/skills").json()["skills"]]
    assert client.get(f"/api/skills/{name}").json()["goal"] == "clean"
    # markdown export
    md = client.get(f"/api/skills/{name}/markdown")
    assert md.status_code == 200 and md.text.startswith("# Skill:")
    # run on a fresh dataset
    run = client.post(f"/api/skills/{name}/run", json={"dataset_path": SAMPLE})
    assert run.status_code == 200
    assert run.json()["state"]["nca_parameters"]
    # delete (cleanup)
    assert client.delete(f"/api/skills/{name}").status_code == 200
    assert client.get(f"/api/skills/{name}").status_code == 404


def test_http_run_unknown_skill_404(client):
    assert client.post("/api/skills/does-not-exist/run",
                       json={"dataset_path": SAMPLE}).status_code == 404
