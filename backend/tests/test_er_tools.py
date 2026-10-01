"""Exposure-response tool layer: registration and routing, dataset extraction,
state writes, the expensive/proposable bootstrap contract, and dose selection
end to end through the registry (the single choke point every call funnels
through). Compute correctness lives in test_er_models / test_er_survival /
test_er_bootstrap / test_optimus; this file tests the wiring.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from app.agents.definitions import AGENTS, DESCRIPTIONS
from app.agents.supervisor import Supervisor
from app.core.audit import AuditChain
from app.core.llm import MockLLM
from app.core.pharmstate import AGENT_WRITE_FIELDS, PharmState, PharmStateError, apply_writes
from app.tools.base import ExpensiveToolError, ToolContext
from app.tools.builtins import default_registry

REG = default_registry()
TOOLS = ("fit_exposure_response", "bootstrap_exposure_response", "select_optimal_dose")


def _frame(n=150, seed=3) -> pd.DataFrame:
    """Long-format PK-style data (3 rows per subject) with subject-level E-R columns."""
    rng = np.random.default_rng(seed)
    dose = rng.choice([50.0, 100.0, 200.0], n)
    auc = np.round(2.0 * dose * np.exp(rng.normal(0.0, 0.3, n)), 1)
    eff = (rng.random(n) < expit(-3.0 + 0.015 * auc)).astype(int)
    ae = (rng.random(n) < expit(-5.0 + 0.012 * auc)).astype(int)
    t_ev = rng.exponential(1.0 / (0.02 * np.exp(0.004 * (auc - 200.0))))
    t_ce = rng.uniform(10.0, 80.0, n)
    rows = []
    for i in range(n):
        for k, tm in enumerate((0.0, 1.0, 4.0)):
            rows.append({"ID": f"S{i:03d}", "TIME": tm, "DV": float(rng.random()), "DOSE": dose[i],
                         "AUC": auc[i], "EFF": eff[i], "AE": ae[i], "TTE": round(float(min(t_ev[i], t_ce[i])), 2),
                         "EVT": int(t_ev[i] <= t_ce[i]), "WT": 60.0 + (i % 7) * 5.0})
    return pd.DataFrame(rows)


def _state(df: pd.DataFrame | None = None, **kw) -> tuple[PharmState, ToolContext]:
    return (PharmState(dataset_id="d1", **kw),
            ToolContext(dataset_store={"d1": _frame() if df is None else df}))


def _run(name, state, ctx, args, *, allow_expensive=False):
    return REG.execute(name, state=state, ctx=ctx, args=args, audit=AuditChain(), timestamp="t0",
                       actor="tester", allow_expensive=allow_expensive)


LOGIT_ARGS = {"endpoint": "binary", "exposure_column": "AUC", "response_column": "EFF", "label": "efficacy"}
TOX_ARGS = {"endpoint": "binary", "exposure_column": "AUC", "response_column": "AE", "label": "toxicity"}
COX_ARGS = {"endpoint": "time_to_event", "exposure_column": "AUC", "time_column": "TTE", "event_column": "EVT",
            "label": "pfs"}


def _with_fits(**extra):
    state, ctx = _state()
    for args in (LOGIT_ARGS, TOX_ARGS):
        state, _ = _run("fit_exposure_response", state, ctx, args)
    return state.model_copy(update=extra) if extra else state, ctx


# ── registration, routing, write whitelist ───────────────────────────────────

def test_tools_are_registered_under_er_dose_with_the_right_admission_flags():
    for n in TOOLS:
        assert REG.get(n).agent == "er_dose"
    assert not REG.get("fit_exposure_response").expensive
    assert not REG.get("select_optimal_dose").expensive
    boot = REG.get("bootstrap_exposure_response")
    assert boot.expensive and boot.proposable
    assert "er_dose" in AGENTS and "er_dose" in DESCRIPTIONS
    from app.agents.supervisor import anchored
    assert anchored("logistic regression of response on AUC") == "er_dose"


def test_write_whitelist_covers_exactly_the_er_slots():
    assert {"er_results", "er_history", "dose_selection_results"} <= AGENT_WRITE_FIELDS["er_dose"]
    for other in ("simulator", "modeler", "nca"):
        assert "er_results" not in AGENT_WRITE_FIELDS[other]
    with pytest.raises(PharmStateError):
        apply_writes(PharmState(), "er_dose", {"nlme_results": {}})
    with pytest.raises(PharmStateError):
        apply_writes(PharmState(), "nca", {"er_results": {}})


@pytest.mark.parametrize("msg", [
    "fit a logistic exposure-response model for the adverse event",
    "what is the hazard ratio per SD of AUC from a Cox model",
    "Kaplan-Meier curves by exposure quartile",
    "select the optimal dose using a utility index (Project Optimus)",
    # one E-R phrase against one pre-existing keyword ('bootstrap', 'auc') is a tie for
    # the classifier (tests/test_routing_overlap.py); these name two E-R phrases
    "bootstrap the exposure-response logistic model",
    "odds ratio of response per unit AUC from a logistic model",
])
def test_supervisor_routes_er_requests_to_er_dose(msg):
    assert Supervisor(MockLLM()).route(msg)[0] == "er_dose", msg


def test_unrelated_routing_is_untouched():
    sup = Supervisor(MockLLM())
    assert sup.route("compute NCA AUC and Cmax")[0] == "nca"
    assert sup.route("assess dose proportionality with the power model")[0] == "dose_prop"
    assert sup.route("run a bioequivalence assessment test vs reference")[0] == "be"


# ── fit_exposure_response: binary ────────────────────────────────────────────

def test_logistic_fit_writes_a_labelled_entry_and_a_history_record():
    state, ctx = _state()
    new, res = _run("fit_exposure_response", state, ctx, LOGIT_ARGS)
    fit = new.er_results["fits"]["efficacy"]
    assert fit["status"] == "ok" and fit["model"] == "logistic" and fit["label"] == "efficacy"
    assert new.er_results["last_label"] == "efficacy"
    assert fit["n"] == 150 and fit["data"]["subject_column"] == "ID" and fit["data"]["exposure_source"] == "dataset"
    assert len(fit["input_fingerprint"]) == 64 and len(fit["curve"]["prob"]) == 25
    assert new.er_history[-1]["kind"] == "fit" and new.er_history[-1]["label"] == "efficacy"
    assert "OR per SD" in res.summary
    json.dumps(res.result)                                         # audit payload is JSON-safe


@pytest.mark.parametrize("args", [LOGIT_ARGS, COX_ARGS])
def test_summary_keeps_a_fractional_ci_level(args):
    state, ctx = _state()
    _, res = _run("fit_exposure_response", state, ctx, {**args, "ci_level": 0.975})
    assert "(97.5% CI " in res.summary and "98%" not in res.summary


def test_fits_with_different_labels_coexist_and_a_refit_replaces_only_its_label():
    state, ctx = _with_fits()
    assert set(state.er_results["fits"]) == {"efficacy", "toxicity"}
    again, _ = _run("fit_exposure_response", state, ctx, {**LOGIT_ARGS, "ci_level": 0.9})
    assert set(again.er_results["fits"]) == {"efficacy", "toxicity"}
    assert again.er_results["fits"]["efficacy"]["ci_level"] == 0.9
    assert again.er_results["fits"]["toxicity"] == state.er_results["fits"]["toxicity"]
    assert len(again.er_history) == 3


def test_history_is_bounded_newest_last():
    state, ctx = _state()
    for i in range(25):
        state, _ = _run("fit_exposure_response", state, ctx, {**LOGIT_ARGS, "label": f"f{i}"})
    assert len(state.er_history) == 20 and state.er_history[-1]["label"] == "f24"
    fits = state.er_results["fits"]
    assert len(fits) == 20 and "f24" in fits and "f0" not in fits         # stored fits are bounded too


def test_a_refused_fit_writes_nothing_and_never_overwrites_a_prior_fit():
    state, ctx = _with_fits()
    df = ctx.dataset_store["d1"].copy()
    df["EFF"] = (df["AUC"] > df["AUC"].median()).astype(int)       # perfect separation
    ctx2 = ToolContext(dataset_store={"d1": df})
    new, res = _run("fit_exposure_response", state, ctx2, LOGIT_ARGS)
    assert res.result["status"] == "separation" and res.writes == {}
    assert new.er_results == state.er_results


def test_column_problems_are_clear_value_errors():
    state, ctx = _state()
    with pytest.raises(ValueError, match="endpoint"):
        _run("fit_exposure_response", state, ctx, {"exposure_column": "AUC"})
    with pytest.raises(ValueError, match="response_column"):
        _run("fit_exposure_response", state, ctx, {"endpoint": "binary", "exposure_column": "AUC"})
    with pytest.raises(ValueError, match="'NOPE'"):
        _run("fit_exposure_response", state, ctx, {**LOGIT_ARGS, "response_column": "NOPE"})
    with pytest.raises(ValueError, match="vary within a subject"):
        _run("fit_exposure_response", state, ctx, {**LOGIT_ARGS, "exposure_column": "DV"})
    df = ctx.dataset_store["d1"].copy()
    df["EFF"] = df["EFF"].map({0: "N", 1: "Y"})
    with pytest.raises(ValueError, match="non-numeric"):
        _run("fit_exposure_response", state, ToolContext(dataset_store={"d1": df}), LOGIT_ARGS)
    with pytest.raises(ValueError, match="no dataset"):
        _run("fit_exposure_response", PharmState(), ToolContext(), LOGIT_ARGS)


def test_the_exposure_cannot_also_be_a_cox_covariate():
    state, ctx = _state()
    with pytest.raises(ValueError, match="must not include the exposure"):
        _run("fit_exposure_response", state, ctx, {**COX_ARGS, "covariate_columns": ["AUC"]})
    with pytest.raises(ValueError, match="same name as the exposure"):
        _run("fit_exposure_response", state, ctx, {**COX_ARGS, "exposure_label": "WT", "covariate_columns": ["WT"]})


def test_exposure_from_state_nca_when_the_dataset_has_no_exposure_column():
    df = _frame().drop(columns=["AUC"])
    truth = _frame().groupby("ID", sort=False)["AUC"].first()
    nca = [{"subject": sid, "dose": 100.0, "AUC_inf": float(v)} for sid, v in truth.items()]
    nca.pop(0)                                                       # one subject without NCA exposure
    state = PharmState(dataset_id="d1", nca_parameters=nca)
    ctx = ToolContext(dataset_store={"d1": df})
    new, _ = _run("fit_exposure_response", state, ctx,
                  {"endpoint": "binary", "exposure_metric": "AUC_inf", "response_column": "EFF", "label": "eff"})
    fit = new.er_results["fits"]["eff"]
    assert fit["data"]["exposure_source"] == "nca" and fit["data"]["n_missing_exposure"] == 1
    assert fit["n"] == 149 and fit["n_dropped_non_finite"] == 1 and fit["exposure_label"] == "AUC_inf"
    with pytest.raises(ValueError, match="exposure_metric"):
        _run("fit_exposure_response", state, ctx, {"endpoint": "binary", "response_column": "EFF"})
    with pytest.raises(ValueError, match="not in the NCA results"):
        _run("fit_exposure_response", state, ctx,
             {"endpoint": "binary", "exposure_metric": "NOPE", "response_column": "EFF"})


# ── fit_exposure_response: time to event ─────────────────────────────────────

def test_time_to_event_fit_stores_cox_km_groups_and_logrank():
    state, ctx = _state()
    new, res = _run("fit_exposure_response", state, ctx, {**COX_ARGS, "covariate_columns": ["WT"]})
    e = new.er_results["fits"]["pfs"]
    assert e["model"] == "cox" and e["cox"]["ties"] == "efron" and e["n"] == 150
    assert [c["name"] for c in e["cox"]["covariates"]] == ["AUC", "WT"]
    assert e["km"]["median"]["estimate"] is not None or e["km"]["n_events"] > 0
    g = e["km_by_exposure_group"]
    assert g["n_groups"] == 4 and len(g["groups"]) == 4 and g["logrank"]["status"] == "ok"
    assert "HR per SD" in res.summary and "efron" in res.summary
    breslow, _ = _run("fit_exposure_response", state, ctx, {**COX_ARGS, "ties": "breslow"})
    assert breslow.er_results["fits"]["pfs"]["cox"]["ties"] == "breslow"
    json.dumps(res.result)


# ── bootstrap_exposure_response: the run_bootstrap contract ──────────────────

def test_bootstrap_is_refused_on_the_synchronous_path_and_requires_confirm():
    state, ctx = _with_fits()
    with pytest.raises(ExpensiveToolError):
        _run("bootstrap_exposure_response", state, ctx, {"label": "efficacy", "confirm": True})
    new, res = _run("bootstrap_exposure_response", state, ctx, {"label": "efficacy"}, allow_expensive=True)
    assert res.result["status"] == "confirm_required" and res.writes == {}


def test_bootstrap_rejections_never_overwrite_a_completed_run():
    state, ctx = _with_fits()
    done, _ = _run("bootstrap_exposure_response", state, ctx,
                   {"label": "efficacy", "confirm": True, "n_boot": 60, "seed": 4}, allow_expensive=True)
    before = done.er_results["fits"]["efficacy"]["bootstrap"]
    for args, status in (({"label": "efficacy"}, "confirm_required"),
                         ({"label": "nope", "confirm": True}, "no_fit")):
        new, res = _run("bootstrap_exposure_response", done, ctx, args, allow_expensive=True)
        assert res.result["status"] == status and res.writes == {}
        assert new.er_results["fits"]["efficacy"]["bootstrap"] == before


def test_bootstrap_attaches_to_the_stored_fit_seeded_and_reproducible():
    state, ctx = _with_fits()
    args = {"label": "efficacy", "confirm": True, "n_boot": 80, "seed": 123}
    a, res = _run("bootstrap_exposure_response", state, ctx, args, allow_expensive=True)
    b, _ = _run("bootstrap_exposure_response", state, ctx, args, allow_expensive=True)
    boot = a.er_results["fits"]["efficacy"]["bootstrap"]
    assert boot["status"] == "ok" and boot["seed"] == 123 and boot["n_ok"] > 0
    assert boot == b.er_results["fits"]["efficacy"]["bootstrap"]
    assert "draws" not in res.result and res.result["seed"] == 123       # audit payload stays small
    assert len(boot["draws"]) <= 500
    assert a.er_results["fits"]["toxicity"] == state.er_results["fits"]["toxicity"]
    assert a.er_history[-1]["kind"] == "bootstrap"


def test_bootstrap_detects_that_the_data_changed_since_the_fit():
    state, ctx = _with_fits()
    df = ctx.dataset_store["d1"].copy()
    df.loc[df["ID"] == "S000", "AUC"] = 999.0
    new, res = _run("bootstrap_exposure_response", state, ToolContext(dataset_store={"d1": df}),
                    {"label": "efficacy", "confirm": True, "n_boot": 40}, allow_expensive=True)
    assert res.result["status"] == "data_changed" and res.writes == {}


def test_bootstrap_stratifies_and_handles_cox():
    state, ctx = _with_fits()
    new, res = _run("bootstrap_exposure_response", state, ctx,
                    {"label": "toxicity", "confirm": True, "n_boot": 60, "stratify_by": "DOSE"},
                    allow_expensive=True)
    assert new.er_results["fits"]["toxicity"]["bootstrap"]["stratified"] is True
    state2, _ = _run("fit_exposure_response", state, ctx, COX_ARGS)
    new2, _ = _run("bootstrap_exposure_response", state2, ctx,
                   {"label": "pfs", "confirm": True, "n_boot": 40}, allow_expensive=True)
    assert new2.er_results["fits"]["pfs"]["bootstrap"]["hr_per_sd"]["lo"] is not None


# ── select_optimal_dose ──────────────────────────────────────────────────────

LINEAR = {"type": "linear", "reference_dose": 100.0, "reference_exposure": 200.0}
SEL = {"doses": [25.0, 50.0, 100.0, 150.0, 200.0, 300.0], "exposure_mapping": LINEAR, "utility_weight": 1.0,
       "n_draws": 300, "seed": 5}


def test_selection_end_to_end_writes_dose_selection_results():
    state, ctx = _with_fits()
    new, res = _run("select_optimal_dose", state, ctx, SEL)
    out = new.dose_selection_results
    assert out["status"] == "ok" and out["selected"]["dose"] in SEL["doses"]
    assert out["utility"]["w_source"] == "declared" and out["uncertainty"]["seed"] == 5
    assert out["uncertainty"]["source"] == {"efficacy": "asymptotic_mvn", "toxicity": "asymptotic_mvn"}
    assert new.er_history[-1]["kind"] == "dose_selection"
    assert "selected" in res.summary.lower() and "w=" in res.summary
    json.dumps(res.result)


def test_selection_uses_bootstrap_draws_when_the_fit_carries_them():
    state, ctx = _with_fits()
    state, _ = _run("bootstrap_exposure_response", state, ctx,
                    {"label": "toxicity", "confirm": True, "n_boot": 80}, allow_expensive=True)
    new, _ = _run("select_optimal_dose", state, ctx, SEL)
    assert new.dose_selection_results["uncertainty"]["source"]["toxicity"] == "bootstrap"


def test_the_utility_weight_is_required_and_a_refusal_writes_nothing():
    state, ctx = _with_fits()
    args = {k: v for k, v in SEL.items() if k != "utility_weight"}
    new, res = _run("select_optimal_dose", state, ctx, args)
    assert res.result["status"] == "utility_not_declared" and res.writes == {}
    ack, res2 = _run("select_optimal_dose", state, ctx, {**args, "utility": "default_w=1"})
    assert ack.dose_selection_results["utility"]["w_source"] == "default_acknowledged"


def test_missing_fits_are_reported_with_the_available_labels():
    state, ctx = _state()
    _, res = _run("select_optimal_dose", state, ctx, SEL)
    assert res.result["status"] == "no_fit" and res.writes == {}
    state2, _ = _run("fit_exposure_response", state, ctx, LOGIT_ARGS)
    _, res2 = _run("select_optimal_dose", state2, ctx, SEL)
    assert res2.result["status"] == "no_fit" and "efficacy" in res2.result["message"]
    state3, _ = _run("fit_exposure_response", state, ctx, COX_ARGS)
    _, res3 = _run("select_optimal_dose", state3, ctx, {**SEL, "efficacy_label": "pfs", "toxicity_label": "pfs"})
    assert res3.result["status"] == "invalid_input"                        # Cox has no dose-level probability


def test_exposure_per_dose_can_come_from_the_dose_proportionality_results():
    state, ctx = _with_fits()
    dp = {"status": "ok", "dose_levels": [50.0, 100.0, 200.0],
          "parameters": {"AUC": {"intercept": float(np.log(2.0)), "slope": 1.0}}}
    state = state.model_copy(update={"dose_prop_results": dp})
    args = {k: v for k, v in SEL.items() if k != "exposure_mapping"}
    new, _ = _run("select_optimal_dose", state, ctx, {**args, "exposure_from": "dose_proportionality"})
    rows = {r["dose"]: r for r in new.dose_selection_results["doses"]}
    assert rows[100.0]["exposure"] == pytest.approx(200.0) and rows[300.0]["dose_extrapolated"] is True
    bad, res = _run("select_optimal_dose", state.model_copy(update={"dose_prop_results": None}), ctx,
                    {**args, "exposure_from": "dose_proportionality"})
    assert res.result["status"] == "invalid_input" and res.writes == {}
    _, res2 = _run("select_optimal_dose", state, ctx, {**args, "exposure_from": "dose_proportionality",
                                                       "dp_parameter": "Cmax"})
    assert res2.result["status"] == "invalid_input"
    with pytest.raises(ValueError, match="exposure_mapping"):
        _run("select_optimal_dose", state, ctx, args)


def test_no_feasible_dose_is_recorded_as_a_result():
    state, ctx = _with_fits()
    new, res = _run("select_optimal_dose", state, ctx, {**SEL, "toxicity_cap": 0.0001})
    assert res.result["status"] == "no_feasible_dose"
    assert new.dose_selection_results["status"] == "no_feasible_dose"
