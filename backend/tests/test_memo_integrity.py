"""Briefing-memo integrity: user-supplied inputs are labelled as inputs, every CI shows
its own level, results from a previously loaded dataset never mix with the current
one, and the exposure-response / dose-selection decision is not silently dropped."""
from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from app.compute import adversarial
from app.compute.indirect import bucher_indirect
from app.compute.memo import AuditTrace
from app.compute.memo_build import METHOD_NOTE, METHOD_NOTE_WITH_INPUTS, build_memo
from app.compute.memo_model import Recorder, Table
from app.core.audit import AuditChain
from app.core.llm import MockLLM
from app.core.orchestrator import Orchestrator
from app.core.pharmstate import PharmState
from app.core.store import SessionStore
from app.tools.base import ToolContext
from app.tools.builtins import default_registry
from tests.memo_fixtures import full_chain, full_state, full_trail

DATA = Path(__file__).parent.parent / "sample_data"


def _indirect_state(comp: dict) -> PharmState:
    return PharmState(indirect_results={"comparisons": [comp]})


def _rows(memo, first: str):
    return [r for b in memo.blocks() if isinstance(b, Table) and b.headers[0] == first for r in b.rows]


# ── user-supplied inputs ───────────────────────────────────────────────────────
def test_direct_legs_are_recorded_as_inputs_and_labelled_user_supplied():
    memo = build_memo(full_state(), full_trail())
    kinds = {v.key: v.kind for v in memo.values}
    assert kinds["indirect_results.0.inputs.ab.estimate"] == "input"
    assert kinds["indirect_results.0.inputs.ab.se"] == "input"           # an SE the user gave
    assert kinds["indirect_results.0.estimate"] == "number"              # the computed result
    rows = {r[0]: r for r in _rows(memo, "Comparison")}
    assert rows["Indirect"][1] == "computed"
    assert rows["Direct Drug A vs Placebo"][1] == "user-supplied"
    text = memo.text()
    assert METHOD_NOTE_WITH_INPUTS in text and "No number was written by a language model" not in text


def test_a_memo_without_inputs_keeps_the_all_computed_note():
    state = full_state().model_copy(update={"indirect_results": None})
    assert METHOD_NOTE in build_memo(state, full_trail()).text()


def test_each_direct_ci_shows_its_own_level_not_the_output_level():
    comp = bucher_indirect({"estimate": 0.62, "ci_lower": 0.45, "ci_upper": 0.88, "ci_level": 0.90},
                           {"estimate": 0.8, "se": 0.1}, scale="HR", ci_level=0.975)
    memo = build_memo(_indirect_state(comp), AuditTrace())
    rows = {r[0]: r for r in _rows(memo, "Comparison")}
    assert rows["Indirect"][4] == "97.5% [#?]"                            # the output interval, not 98%
    ab = rows["Direct A vs B"]
    assert ab[3] == "0.450 to 0.880 [#?]" and ab[4] == "90% [#?]"        # the published interval's level
    assert "SE derived from its CI" in ab[1]
    kinds = {v.key: v.kind for v in memo.values}
    assert kinds["indirect_results.0.inputs.ab.se"] == "number"           # derived, so computed


def test_fractional_ci_levels_keep_their_decimal_in_every_memo_section():
    comp = bucher_indirect({"estimate": 0.62, "ci_lower": 0.45, "ci_upper": 0.88, "ci_level": 0.975},
                           {"estimate": 0.8, "se": 0.1}, scale="HR", ci_level=0.995)
    rows = {r[0]: r for r in _rows(build_memo(_indirect_state(comp), AuditTrace()), "Comparison")}
    assert rows["Indirect"][4] == "99.5% [#?]" and rows["Direct A vs B"][4] == "97.5% [#?]"
    state = full_state()
    fits = {k: {**v, "ci_level": 0.975} for k, v in state.er_results["fits"].items()}
    forest = {**state.forest_results, "ci_level": 0.975}
    memo = build_memo(state.model_copy(update={"er_results": {**state.er_results, "fits": fits},
                                               "forest_results": forest}), full_trail())
    # edited fits no longer match their audited outputs, so they are tagged unknown [#?]
    assert {r[8] for r in _rows(memo, "Fit")} == {"97.5% [#?]"}
    assert "with 97.5% [#8] confidence intervals" in memo.text()


def test_a_defaulted_input_ci_level_is_marked_assumed():
    comp = bucher_indirect({"estimate": 0.62, "ci_lower": 0.45, "ci_upper": 0.88},
                           {"estimate": 0.8, "se": 0.1}, scale="HR")
    rows = {r[0]: r for r in _rows(build_memo(_indirect_state(comp), AuditTrace()), "Comparison")}
    assert rows["Direct A vs B"][4] == "95% [#?] assumed"


# ── results from a previously loaded dataset ───────────────────────────────────
def _chain_then_reload() -> AuditChain:
    chain = full_chain()
    chain.append(agent="data_manager", tool="load_dataset", action="load_dataset", inputs={}, outputs={},
                 timestamp="t")
    return chain


def test_results_computed_before_the_latest_load_are_withheld_not_mixed_in():
    memo = build_memo(full_state(), AuditTrace(_chain_then_reload().to_list()))
    keys = [s.key for s in memo.sections]
    assert keys == ["dataset", "indirect"]                       # metadata + the dataset-free comparison
    assert not any(v.key.startswith(("data_quality", "nca_summary", "nlme_results", "er_results"))
                   for v in memo.values)
    text = memo.text()
    assert "Results withheld" in text and "Non-compartmental analysis: computed on a previously loaded" in text
    assert "Data-quality profile: computed on a previously loaded" in text
    assert "Population fit: not run." not in text                # withheld is not 'not run'


def test_without_an_audit_trail_nothing_is_withheld():
    assert build_memo(full_state(), AuditTrace()).withheld == ()


def test_orchestrated_reload_of_another_file_does_not_mix_datasets(tmp_path, monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    orch = Orchestrator(llm=MockLLM(), clock=lambda c=itertools.count(): f"t{next(c)}",
                        store=SessionStore(":memory:"))
    sid = orch.create_session().id
    orch.run_tool(sid, "load_dataset", "data_manager", {"path": str(DATA / "theoph_pk.csv")})
    for tool, agent in (("profile_pk_dataset", "data_manager"), ("compute_nca", "nca"), ("run_qc", "qc")):
        orch.run_tool(sid, tool, agent, {})
    orch.run_tool(sid, "load_dataset", "data_manager", {"path": str(DATA / "oral_pk.csv")})
    res = orch.run_tool(sid, "build_briefing_memo", "report", {})["result"]
    assert res["sections_included"] == ["dataset"]
    assert {w["key"] for w in res["results_withheld"]} == {"data_quality", "nca_summary", "qc"}
    assert "BLQ observations" not in res["memo_text"] and "NCA:" not in res["memo_text"]
    assert res["dataset_id"] == orch.get_session(sid).state.dataset_id


def test_reviewer_flags_a_memo_built_on_another_dataset(tmp_path):
    state, _ = default_registry().execute(
        "build_briefing_memo", state=full_state().model_copy(update={"dataset_id": "ds_old"}),
        ctx=ToolContext(data_dir=str(tmp_path)), args={}, audit=full_chain(), timestamp="t", actor="u")
    ok = adversarial.review(state.model_dump(), None, None)
    assert not [f for f in ok["findings"] if f["id"] == "memo-dataset-changed"]
    moved = adversarial.review(state.model_copy(update={"dataset_id": "ds_new"}).model_dump(), None, None)
    found = [f for f in moved["findings"] if f["id"] == "memo-dataset-changed"]
    assert len(found) == 1 and found[0]["severity"] == "HIGH" and "ds_old" in found[0]["evidence"]


# ── exposure-response and dose selection ───────────────────────────────────────
def test_er_fits_and_the_dose_decision_are_in_the_memo_with_trace_tags():
    memo = build_memo(full_state(), full_trail())
    assert {"er", "dose_selection"} <= {s.key for s in memo.sections}
    heads = "\n".join(memo.headlines)
    assert "efficacy OR per SD 16.901 [#13]" in heads and "Dose selection: dose 200 [#17] selected." in heads
    text = memo.text()
    assert "Selected dose 200 [#17]: utility 0.425 [#17]" in text
    assert "weight w 1.000 [#17] (declared)" in text
    assert "bootstrap percentile interval 7.441 to 45.103 [#16]" in text


def test_er_sections_are_listed_not_run_when_absent():
    state = full_state().model_copy(update={"er_results": None, "dose_selection_results": None})
    text = build_memo(state, full_trail()).text()
    assert "Exposure-response: not run." in text and "Dose selection: not run." in text


def test_a_free_text_fit_label_with_digits_is_withheld():
    state = full_state()
    fits = {"AE rate 87 pct": state.er_results["fits"]["toxicity"]}
    memo = build_memo(state.model_copy(update={"er_results": {"fits": fits}}), full_trail())
    assert "87" not in memo.text() and "label withheld (contains digits)" in memo.text()


def test_no_feasible_dose_is_reported_not_dropped():
    ds = {**full_state().dose_selection_results, "status": "no_feasible_dose", "selected": None,
          "message": "No dose satisfies the toxicity cap."}
    text = build_memo(full_state().model_copy(update={"dose_selection_results": ds}), full_trail()).text()
    assert "No dose was selected: No dose satisfies the toxicity cap." in text


# ── recorder: missing values are never recorded ────────────────────────────────
@pytest.mark.parametrize("call", [
    lambda c: c.span("k", None, 1.0), lambda c: c.span("k", 1.0, float("nan")),
    lambda c: c.frac("k", None, 3), lambda c: c.count("k", None), lambda c: c.sig("k", float("inf")),
    lambda c: c.num("k", None), lambda c: c.inputs().num("k", float("nan"))])
def test_missing_inputs_render_a_dash_and_record_nothing(call):
    rec = Recorder()
    assert call(rec.at(3)) == "-"
    assert rec.values == ()
