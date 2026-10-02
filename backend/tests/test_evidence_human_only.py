"""decide_screening / decide_records are the reviewer's: a chat-turn model can neither see them nor run
them (even with confirm=true and a forged actor), skill replay cannot run them, and the actor stored
on a decision is the authenticated caller, never an argument."""
from __future__ import annotations

import itertools
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.config import settings
from app.core.llm import MockLLM
from app.core.orchestrator import Orchestrator
from app.core.pharmstate import PharmState
from app.core.store import SessionStore
from app.evidence import llm_access
from app.evidence import store as evstore
from app.tools.base import HumanOnlyToolError, ToolContext
from app.tools.builtins import default_registry
from tests import evidence_fixtures as fx
from tests.test_evidence_records_api import _setup

EV = "/api/sessions/{sid}/evidence"
DECIDE_TOOLS = ("decide_screening", "decide_records")


class ForcedLLM(MockLLM):
    """Stands in for a real model (or one steered by injected text): it names a decide_* tool with
    confirm=true and a forged reviewer whether or not the tool is in the list it was offered."""

    def __init__(self) -> None:
        super().__init__()
        self.call: tuple[str, dict] | None = None
        self.offered: list[set[str]] = []

    def select_tool(self, agent, message, tools, state):
        self.offered.append({t.name for t in tools})
        call, self.call = self.call, None
        return None if call is None else {"name": call[0], "input": call[1]}


@pytest.fixture(scope="module")
def papers(tmp_path_factory) -> dict[str, Path]:
    return fx.build_all(tmp_path_factory.mktemp("papers"))


@pytest.fixture
def forced(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    evstore.configure(":memory:")
    llm = ForcedLLM()
    main.orch = Orchestrator(llm=llm, clock=lambda c=itertools.count(): f"t{next(c)}",
                             store=SessionStore(":memory:"))
    llm_access.set_llm(None)
    yield TestClient(main.app), llm
    llm_access.set_llm(None)


def test_decide_tools_are_not_offered_to_any_agent():
    reg = default_registry()
    for tool in DECIDE_TOOLS:
        assert reg.get(tool).human_only
        assert all(tool not in {t.name for t in reg.for_agent(a)} for a in ("evidence", "supervisor", "nca"))


def test_registry_refuses_human_only_without_the_human_path():
    reg = default_registry()
    from app.core.audit import AuditChain
    with pytest.raises(HumanOnlyToolError):
        reg.execute("decide_records", state=PharmState(session_id="s"), ctx=ToolContext(),
                    args={"record_id": "rec_x", "decision": "approve", "reason": "x", "confirm": True},
                    audit=AuditChain(), timestamp="t")


def test_chat_turn_cannot_approve_a_record_even_with_confirm_and_actor(forced, papers):
    client, llm = forced
    sid = _setup(client, papers, names=("paper_a",))
    assert client.post(EV.format(sid=sid) + "/extract", json={}).status_code == 200
    rec = next(r for r in client.get(EV.format(sid=sid)).json()["records"] if r["verify_status"] == "pass")
    llm.call = ("decide_records", {"record_id": rec["id"], "decision": "approve", "reason": "model says fine",
                                   "actor": "dr.jane.pi", "confirm": True})
    r = client.post(f"/api/sessions/{sid}/chat", json={"message": "approve the evidence records"})
    assert r.status_code == 200, r.text
    assert any(DECIDE_TOOLS[1] not in offered for offered in llm.offered)
    after = next(x for x in client.get(EV.format(sid=sid)).json()["records"] if x["id"] == rec["id"])
    assert after["status"] == "proposed" and after["decided_by"] is None
    entries = client.get(f"/api/sessions/{sid}/audit").json()["entries"]
    assert not any(e["tool"] == "decide_records" for e in entries)


def test_chat_turn_cannot_decide_a_screening_verdict(forced, papers):
    client, llm = forced
    sid = client.post("/api/sessions").json()["id"]
    r = client.post(EV.format(sid=sid) + "/upload",
                    files={"file": ("paper_a.pdf", papers["paper_a"].read_bytes(), "application/pdf")})
    assert r.status_code == 200
    from tests.test_evidence_extract import INCLUDE_ALL
    client.post(EV.format(sid=sid) + "/criteria", json={"criteria": INCLUDE_ALL})
    scr = client.post(EV.format(sid=sid) + "/screen", json={}).json()["result"]["screens"][0]
    llm.call = ("decide_screening", {"screen_id": scr["id"], "decision": "include", "confirm": True,
                                     "reason": "reviewed", "actor": "dr_smith"})
    assert client.post(f"/api/sessions/{sid}/chat", json={"message": "screen the publications"}).status_code == 200
    after = client.get(EV.format(sid=sid)).json()["screens"][0]
    assert after["status"] == "proposed" and after["decided_by"] is None


def test_http_decision_records_the_authenticated_actor_not_the_argument(forced, papers):
    client, _ = forced
    sid = _setup(client, papers, names=("paper_a",))
    client.post(EV.format(sid=sid) + "/extract", json={})
    rec = next(r for r in client.get(EV.format(sid=sid)).json()["records"] if r["verify_status"] == "pass")
    out = main.orch.run_tool(sid, "decide_records", "evidence",
                             {"record_id": rec["id"], "decision": "reject", "reason": "x", "confirm": True,
                              "actor": "forged_name"}, actor="real_user")
    assert out["result"]["record"]["decided_by"] == "real_user"
