"""Exposure-response provenance in the briefing memo: each fit (and its bootstrap) is
tagged with the audit entry of the run that produced IT, a fit computed on a previously
loaded dataset is withheld on its own, and a dose selection is withheld when it was
computed from such a fit -- even when the selection itself ran after the reload."""
from __future__ import annotations

import itertools
import re
from pathlib import Path

import pytest

from app.compute.memo_build import build_memo
from app.compute.memo_model import Table
from app.core.llm import MockLLM
from app.core.orchestrator import Orchestrator
from app.core.store import SessionStore
from tests.memo_fixtures import IDX, full_state, full_trail
from tests.test_er_tools import COX_ARGS, LOGIT_ARGS, SEL, TOX_ARGS, _frame

DATA = Path(__file__).parent.parent / "sample_data"


def _rows(memo, first: str):
    return [r for b in memo.blocks() if isinstance(b, Table) and b.headers[0] == first for r in b.rows]


# ── fixture: one audit entry per fit ───────────────────────────────────────────
def test_each_fit_and_its_bootstrap_carry_their_own_audit_entry():
    memo = build_memo(full_state(), full_trail())
    fits = {r[0]: r for r in _rows(memo, "Fit")}
    assert fits["efficacy"][6] == f"16.901 [#{IDX['er_eff']}]"
    assert fits["toxicity"][6] == f"2.457 [#{IDX['er_tox']}]"
    assert fits["pfs"][6] == f"2.104 [#{IDX['er_pfs']}]"
    assert f"bootstrap percentile interval 7.441 to 45.103 [#{IDX['er_boot']}]" in memo.text()
    assert f"Selected dose 200 [#{IDX['dose']}]" in memo.text()


# ── a real session: fits, a reload, more fits ──────────────────────────────────
@pytest.fixture
def orch(tmp_path, monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    return Orchestrator(llm=MockLLM(), clock=lambda c=itertools.count(): f"t{next(c)}",
                        store=SessionStore(":memory:"))


def _session(orch) -> str:
    sid = orch.create_session().id
    sess = orch.get_session(sid)
    sess.ctx.dataset_store["d1"] = _frame()
    sess.state = sess.state.model_copy(update={"dataset_id": "d1"})
    return sid


def _reload(orch, sid: str) -> None:
    """load_dataset of another file (a real audit entry), whose frame is then replaced by
    E-R data so that fits can run on the NEW dataset."""
    orch.run_tool(sid, "load_dataset", "data_manager", {"path": str(DATA / "theoph_pk.csv")})
    sess = orch.get_session(sid)
    sess.ctx.dataset_store[sess.state.dataset_id] = _frame(seed=99)


def _fit(orch, sid: str, args: dict) -> int:
    orch.run_tool(sid, "fit_exposure_response", "er_dose", args)
    return _last_index(orch, sid, "fit_exposure_response")


def _last_index(orch, sid: str, tool: str) -> int:
    return max(e.index for e in orch.get_session(sid).audit.entries if e.tool == tool)


def _memo(orch, sid: str) -> dict:
    return orch.run_tool(sid, "build_briefing_memo", "report", {})["result"]


def _tags(text: str, label: str) -> set[str]:
    line = next(ln for ln in text.splitlines() if ln.startswith(f"{label} |"))
    return set(re.findall(r"\[#(\d+)\]", line))


def test_a_fit_from_the_previous_dataset_is_withheld_and_the_current_one_is_tagged_with_its_own_entry(orch):
    sid = _session(orch)
    _fit(orch, sid, LOGIT_ARGS)                                   # efficacy on dataset A
    _reload(orch, sid)
    tox = _fit(orch, sid, TOX_ARGS)                               # toxicity on dataset B
    res = _memo(orch, sid)
    text = res["memo_text"]
    assert [w["key"] for w in res["results_withheld"]] == ["er_results.fits.efficacy"]
    assert not any(ln.startswith("efficacy |") for ln in text.splitlines())
    assert _tags(text, "toxicity") == {str(tox)}


def test_a_fit_and_its_bootstrap_are_tagged_separately(orch):
    sid = _session(orch)
    fit = _fit(orch, sid, LOGIT_ARGS)
    _fit(orch, sid, COX_ARGS)                                     # a later fit of another label
    orch.run_tool(sid, "bootstrap_exposure_response", "er_dose",
                  {"label": "efficacy", "confirm": True, "n_boot": 40})
    boot = _last_index(orch, sid, "bootstrap_exposure_response")
    text = _memo(orch, sid)["memo_text"]
    assert _tags(text, "efficacy") == {str(fit)}
    assert re.search(rf"bootstrap percentile interval \S+ to \S+ \[#{boot}\]", text)


def test_a_dose_selection_run_after_a_reload_on_stale_fits_is_withheld(orch):
    sid = _session(orch)
    _fit(orch, sid, LOGIT_ARGS)
    _fit(orch, sid, TOX_ARGS)
    _reload(orch, sid)
    orch.run_tool(sid, "select_optimal_dose", "er_dose", SEL)     # runs on the stale fits
    res = _memo(orch, sid)
    keys = {w["key"] for w in res["results_withheld"]}
    assert {"er_results.fits.efficacy", "er_results.fits.toxicity", "dose_selection_results"} <= keys
    assert "Selected dose" not in res["memo_text"] and "dose_selection" not in res["sections_included"]


def test_a_selection_whose_fits_were_refit_on_the_new_dataset_is_still_withheld(orch):
    sid = _session(orch)
    _fit(orch, sid, LOGIT_ARGS)
    _fit(orch, sid, TOX_ARGS)
    _reload(orch, sid)
    orch.run_tool(sid, "select_optimal_dose", "er_dose", SEL)     # computed from the dataset-A fits
    _fit(orch, sid, LOGIT_ARGS)
    _fit(orch, sid, TOX_ARGS)                                     # both refit on dataset B
    res = _memo(orch, sid)
    assert {w["key"] for w in res["results_withheld"]} == {"dose_selection_results"}
    assert "er" in res["sections_included"]


def test_a_selection_on_current_fits_is_kept(orch):
    sid = _session(orch)
    _reload(orch, sid)
    _fit(orch, sid, LOGIT_ARGS)
    _fit(orch, sid, TOX_ARGS)
    orch.run_tool(sid, "select_optimal_dose", "er_dose", SEL)
    res = _memo(orch, sid)
    sel = _last_index(orch, sid, "select_optimal_dose")
    assert res["results_withheld"] == [] and "Selected dose" in res["memo_text"]
    assert f"[#{sel}]" in res["memo_text"]


# ── without an audit match: the dataset a fit was computed on still decides ────
def test_a_fit_stamped_with_another_dataset_is_withheld_and_its_digit_label_is_not_printed():
    state = full_state()
    fits = {**state.er_results["fits"],
            "AE rate 87 pct": {**state.er_results["fits"]["toxicity"], "dataset_id": "ds_old"}}
    state = state.model_copy(update={"dataset_id": "ds_new", "er_results": {"fits": fits}})
    memo = build_memo(state, full_trail())
    assert memo.withheld[-1] == ("er_results.fits.AE rate 87 pct",
                                 "Exposure-response fit (label withheld: contains digits)")
    assert "87" not in memo.text() and "efficacy" in {r[0] for r in _rows(memo, "Fit")}


def test_an_unmatched_fit_is_still_withheld_when_every_e_r_run_predates_the_reload():
    """Edited after its run, a fit matches no audited output; it is still dated by the
    newest E-R entry, which here is older than the latest load_dataset."""
    from app.compute.memo import AuditTrace
    from tests.memo_fixtures import full_chain
    chain = full_chain()
    chain.append(agent="data_manager", tool="load_dataset", action="load_dataset", inputs={}, outputs={},
                 timestamp="t")
    state = full_state()
    edited = {k: {**f, "n": 151} for k, f in state.er_results["fits"].items()}
    memo = build_memo(state.model_copy(update={"er_results": {"fits": edited}}), AuditTrace(chain.to_list()))
    keys = {k for k, _ in memo.withheld}
    assert {"er_results.fits.efficacy", "er_results.fits.pfs", "dose_selection_results"} <= keys
    assert "er" not in {s.key for s in memo.sections}
