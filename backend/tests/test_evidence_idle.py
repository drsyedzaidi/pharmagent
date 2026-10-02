"""S4: the evidence agent's idle hint names the pipeline step that is waiting on the human,
the state summary exposes the evidence counts the keyless mock keys on, and
``MockLLM.select_tool('evidence', ...)`` only ever picks the three pipeline tools — never a
decide_* tool — and only once publications exist.
"""
from __future__ import annotations

from app.agents.base import Agent, idle_hint
from app.core.audit import AuditChain
from app.core.llm import MockLLM
from app.core.pharmstate import PharmState
from app.tools.base import Tool, ToolContext
from app.tools.builtins import default_registry


class _NoTool:
    def select_tool(self, agent, message, tools, state_summary):
        return None


def _turn(state: PharmState, llm=None, message: str = "x") -> list[str]:
    res = Agent(name="evidence", system_prompt="").run_turn(
        state=state, message=message, llm=llm or _NoTool(), registry=default_registry(),
        ctx=ToolContext(), audit=AuditChain(), clock=lambda: "t0", actor="t")
    return res.messages


def _summary(n: int, criteria: bool = True, template: bool = True) -> dict:
    return {"sources": n, "parsed": n, "no_text": 0, "failed": 0, "screens": 0, "records": 0,
            "datasets": 0, "criteria_hash": "c" * 64 if criteria else None,
            "template_hash": "t" * 64 if template else None, "updated_at": "t0"}


# ── idle hints ───────────────────────────────────────────────────────────────

def test_no_sources_points_at_the_upload():
    msgs = _turn(PharmState())
    assert len(msgs) == 1
    m = msgs[0]
    assert m.startswith("[evidence]") and "no publications uploaded" in m.lower()
    assert "no action taken" not in m.lower()


def test_sources_without_criteria_or_screens_name_the_next_step():
    m = idle_hint("evidence", PharmState(evidence_summary=_summary(3, criteria=False)))
    assert "3 publication" in m and "criteria" in m.lower()
    m = idle_hint("evidence", PharmState(evidence_summary=_summary(3)))
    assert "screen" in m.lower() and "await your decision" not in m.lower()


def test_proposed_screens_await_the_decision():
    st = PharmState(evidence_summary=_summary(3),
                    screening_results={"criteria_hash": "c" * 64, "n_screened": 3, "proposed": 2,
                                       "approved": 1, "rejected": 0, "method": "keyword",
                                       "verdicts": {"include": 2, "exclude": 1, "unsure": 0}})
    m = _turn(st)[0]
    assert "2 screening verdict" in m and "await your decision" in m.lower()


def test_failed_verification_is_called_out_and_cannot_be_approved():
    st = PharmState(evidence_summary=_summary(2),
                    screening_results={"proposed": 0, "approved": 2, "rejected": 0},
                    extraction_results={"template_hash": "t" * 64, "n_records": 5, "proposed": 3,
                                        "approved": 2, "rejected": 0, "verify_fail": 1, "flagged": 1})
    m = _turn(st)[0]
    low = m.lower()
    assert "1 proposed record" in low and "failed verification" in low
    assert "3 extracted record" in low and "await your decision" in low


def test_approved_records_suggest_the_export():
    st = PharmState(evidence_summary=_summary(2),
                    screening_results={"proposed": 0, "approved": 2, "rejected": 0},
                    extraction_results={"proposed": 0, "approved": 4, "rejected": 1, "verify_fail": 0})
    m = _turn(st)[0]
    assert "4 approved record" in m and "export the evidence dataset" in m.lower()


def test_after_export_the_hint_names_the_dataset():
    st = PharmState(evidence_summary=_summary(2), screening_results={"proposed": 0},
                    extraction_results={"proposed": 0, "approved": 4, "verify_fail": 0},
                    evidence_dataset_path="/data/evidence/exports/evidence_s_1.csv")
    m = idle_hint("evidence", st)
    assert "evidence_s_1.csv" in m and "no action taken" not in m.lower()


def test_approved_screens_but_no_template_asks_for_the_template():
    st = PharmState(evidence_summary=_summary(2, template=False),
                    screening_results={"proposed": 0, "approved": 2, "rejected": 0})
    m = idle_hint("evidence", st)
    assert "template" in m.lower() and "extract" in m.lower()


# ── state summary ────────────────────────────────────────────────────────────

def test_state_summary_exposes_evidence_keys():
    agent = Agent(name="evidence", system_prompt="")
    s = agent._state_summary(PharmState())
    assert s["evidence_sources"] == 0 and s["screening_results"] is None
    assert s["extraction_results"] is None and s["evidence_dataset_path"] is None
    st = PharmState(evidence_summary=_summary(2), screening_results={"proposed": 1},
                    extraction_results={"proposed": 0}, evidence_dataset_path="/x.csv")
    s = agent._state_summary(st)
    assert s["evidence_sources"] == 2 and s["screening_results"] == "present"
    assert s["extraction_results"] == "present" and s["evidence_dataset_path"] == "/x.csv"
    assert "evidence_summary" not in s           # counts only, never the mirror itself


# ── keyless mock: evidence branch ────────────────────────────────────────────

def _stub(name: str) -> Tool:
    return Tool(name, "d", "evidence", {"type": "object", "properties": {}}, lambda s, c, a: None)


TOOLS = [_stub(n) for n in ("ingest_evidence_source", "save_evidence_config", "screen_publications",
                            "decide_screening", "extract_evidence", "verify_records", "decide_records",
                            "export_evidence_dataset")]


def _pick(message: str, **summary) -> dict | None:
    base = {"evidence_sources": 1, "screening_results": None, "extraction_results": None,
            "evidence_dataset_path": None}
    return MockLLM().select_tool("evidence", message, TOOLS, {**base, **summary})


def test_mock_screens_only_when_sources_exist():
    assert _pick("screen the publications") == {"name": "screen_publications", "input": {}}
    assert _pick("screen the publications against my inclusion criteria") == {
        "name": "screen_publications", "input": {}}
    assert _pick("screen the publications", evidence_sources=0) is None
    assert _pick("screen the publications", evidence_sources=None) is None


def test_mock_screening_already_done_stops():
    assert _pick("screen the publications", screening_results="present") is None


def test_mock_extract_and_export_follow_the_pipeline():
    assert _pick("extract the evidence from the included papers") == {"name": "extract_evidence", "input": {}}
    assert _pick("extract the evidence", extraction_results="present") is None
    assert _pick("export the evidence dataset") == {"name": "export_evidence_dataset", "input": {}}
    assert _pick("export the evidence dataset", evidence_dataset_path="/x.csv") is None


def test_mock_never_selects_decide_or_config_tools():
    for msg in ("approve the screening verdicts", "decide the records", "reject record rec_1",
                "save my criteria", "verify the records", "hello"):
        out = _pick(msg)
        assert out is None, (msg, out)


def test_mock_only_picks_tools_the_registry_offers():
    only_config = [t for t in TOOLS if t.name in ("ingest_evidence_source", "save_evidence_config")]
    out = MockLLM().select_tool("evidence", "screen the publications", only_config, {"evidence_sources": 2})
    assert out is None


def test_keyless_turn_before_the_screen_tool_exists_returns_the_hint_not_an_error():
    st = PharmState(evidence_summary=_summary(2))
    msgs = _turn(st, llm=MockLLM(), message="screen the publications")
    assert len(msgs) == 1 and msgs[0].startswith("[evidence]")
    assert "could not run" not in msgs[0].lower()
