"""When an agent's turn selects no tool, the reply must say what IS loaded and
what to do next — not the bare "[agent] no action taken." that a desktop user
read as "the app is broken" after typing "load dataset" / "Profile PK data set"
with nothing uploaded.
"""
from __future__ import annotations

from app.agents.base import Agent, idle_hint
from app.core.audit import AuditChain
from app.core.pharmstate import PharmState
from app.tools.base import ToolContext
from app.tools.builtins import default_registry


class _NoTool:
    def select_tool(self, agent, message, tools, state_summary):
        return None


def _turn(agent: str, state: PharmState) -> list[str]:
    res = Agent(name=agent, system_prompt="").run_turn(
        state=state, message="x", llm=_NoTool(), registry=default_registry(),
        ctx=ToolContext(), audit=AuditChain(), clock=lambda: "t0", actor="t")
    return res.messages


def test_data_manager_with_nothing_loaded_points_at_the_upload_button():
    msgs = _turn("data_manager", PharmState())
    assert len(msgs) == 1
    m = msgs[0].lower()
    assert "no dataset" in m and "upload" in m and "path" in m
    assert "no action taken" not in m


def test_data_manager_with_a_loaded_dataset_names_it_and_the_next_steps():
    st = PharmState(dataset_id="ds_1", dataset_metadata={"n_subjects": 12, "n_records": 132},
                    data_quality={"quality_flags": []})
    m = _turn("data_manager", st)[0]
    assert "ds_1" in m and "12 subjects" in m
    assert "nca" in m.lower()


def test_nca_already_computed_says_so_instead_of_no_action():
    st = PharmState(dataset_id="ds_1", nca_parameters=[{"ID": i} for i in range(12)])
    m = _turn("nca", st)[0]
    assert "already" in m.lower() and "12" in m
    assert "qc" in m.lower() or "report" in m.lower()


def test_nca_without_a_dataset_explains_the_prerequisite():
    m = _turn("nca", PharmState())[0]
    assert "no dataset" in m.lower() and "upload" in m.lower()


def test_generic_agent_gets_a_state_aware_fallback():
    st = PharmState(dataset_id="ds_1", nca_parameters=[{"ID": 1}])
    m = idle_hint("qc", st)
    assert m.startswith("[qc]") and "no action taken" not in m
    assert "ds_1" in m


def test_hint_is_a_single_line_without_raw_rows():
    st = PharmState(dataset_id="ds_1", nca_parameters=[{"ID": 1, "AUC": 3.3}])
    for agent in ("data_manager", "nca", "be", "poppk", "modeler", "simulator", "qc", "report"):
        m = idle_hint(agent, st)
        assert "\n" not in m and "AUC" not in m, agent
