"""Non-parametric bootstrap for parameter uncertainty.

Every test injects a FAKE fit_fn -- no estimator runs here. That is what makes
the expensive method testable at all, and it lets the statistical behaviour be
checked against closed-form answers instead of against another estimator.
"""
import math

import numpy as np
import pytest

from app.compute.bootstrap import (
    _MIN_OK_FOR_CI,
    _asymptotic_ci,
    _resample_indices,
    run_bootstrap,
)

N = 40


def _subjects(seed=0, n=N):
    rng = np.random.default_rng(seed)
    vals = rng.normal(3.0, 0.6, n)
    return [{"subject": i, "_cl": float(vals[i])} for i in range(n)], vals


def _mean_fit(rep, seed):
    """theta['CL'] = mean of the resampled subjects -> the bootstrap
    distribution of CL is the bootstrap distribution of a sample mean, whose
    SE is known in closed form."""
    return {"status": "ok", "converged": True,
            "theta": {"CL": float(np.mean([s["_cl"] for s in rep]))},
            "iiv_params": [], "omega_cv_pct": {}, "sigma": {}}


def _nlme(vals):
    return {"status": "ok", "theta": {"CL": float(vals.mean())},
            "theta_rse_pct": {"CL": 5.0}, "iiv_params": [],
            "omega_cv_pct": {}, "omega_rse_pct": {}}


# ── statistical correctness ─────────────────────────────────────────────────

def test_bootstrap_se_of_a_mean_matches_the_closed_form():
    """The nonparametric bootstrap SE of a sample mean converges to
    s_n/sqrt(n) with s_n the SAMPLE sd (1/n normalisation) -- NOT the
    population sd. This is the benchmark that says the resampling itself is
    right, independent of any PK model."""
    subs, vals = _subjects()
    r = run_bootstrap("m", subs, _nlme(vals), fit_fn=_mean_fit, n_boot=2000,
                      seed=1, params=("CL",))
    expect = float(np.std(vals, ddof=0)) / math.sqrt(len(vals))
    assert r["parameters"][0]["boot_se"] == pytest.approx(expect, rel=0.06)


def test_stratification_removes_between_stratum_variance():
    """Stratifying on a variable that tracks the outcome must NARROW the
    interval, because the between-stratum component is no longer resampled.
    Constructed adversarially: the strata are the low half and high half."""
    subs, vals = _subjects()
    order = np.argsort(vals)
    lab = {int(i): (1 if k < len(vals) // 2 else 2) for k, i in enumerate(order)}
    kw = dict(fit_fn=_mean_fit, n_boot=1500, seed=1, params=("CL",))
    plain = run_bootstrap("m", subs, _nlme(vals), **kw)
    strat = run_bootstrap("m", subs, _nlme(vals),
                          strata=[lab[i] for i in range(len(vals))], **kw)
    assert strat["parameters"][0]["boot_se"] < 0.8 * plain["parameters"][0]["boot_se"]
    assert strat["stratified"] is True and strat["n_strata"] == 2


def test_stratified_resample_preserves_every_stratum_size():
    """An unstratified draw can omit a small arm entirely; the stratified one
    must not, or the replicate is a different design."""
    rng = np.random.default_rng(3)
    strata = [1] * 30 + [2] * 8 + [3] * 2          # one very small arm
    idx = _resample_indices(len(strata), rng, strata)
    got = [strata[i] for i in idx]
    assert len(idx) == len(strata)
    for s in (1, 2, 3):
        assert got.count(s) == strata.count(s)


def test_unstratified_resample_draws_n_with_replacement():
    rng = np.random.default_rng(3)
    idx = _resample_indices(20, rng, None)
    assert len(idx) == 20 and all(0 <= i < 20 for i in idx)
    assert len(set(idx)) < 20                       # with replacement -> ties


def test_asymptotic_interval_is_multiplicative_on_the_log_scale():
    """RSE% here is 100*SE on the LOG scale, so the comparison interval must be
    est*exp(+/-1.96*rse/100). Building it additively would compare the
    bootstrap against a different, wrong interval."""
    nl = {"theta": {"CL": 4.0}, "theta_rse_pct": {"CL": 20.0},
          "omega_cv_pct": {}, "omega_rse_pct": {}}
    est, lo, hi = _asymptotic_ci(nl, "CL")
    assert est == 4.0
    assert lo == pytest.approx(4.0 * math.exp(-1.959963984540054 * 0.20))
    assert hi == pytest.approx(4.0 * math.exp(+1.959963984540054 * 0.20))
    assert lo > 0.0                                  # multiplicative -> positive


def test_comparison_reports_interval_width_ratio():
    subs, vals = _subjects()
    r = run_bootstrap("m", subs, _nlme(vals), fit_fn=_mean_fit, n_boot=300,
                      seed=1, params=("CL",))
    c = r["comparison"][0]
    assert c["parameter"] == "CL"
    assert c["boot_width"] > 0 and c["asymptotic_width"] > 0
    assert c["width_ratio_boot_over_asymptotic"] == pytest.approx(
        c["boot_width"] / c["asymptotic_width"], rel=1e-6)


# ── failure accounting (the simest lesson: never silently drop) ──────────────

def test_failed_replicates_are_counted_not_dropped():
    subs, vals = _subjects()
    state = {"n": 0}

    def flaky(rep, seed):
        state["n"] += 1
        if state["n"] % 3 == 0:
            return {"status": "ok", "converged": False, "theta": {"CL": 3.0}}
        return _mean_fit(rep, seed)

    r = run_bootstrap("m", subs, _nlme(vals), fit_fn=flaky, n_boot=90, seed=1,
                      params=("CL",))
    assert r["n_failed"] > 0
    assert r["n_ok"] + r["n_failed"] == r["n_completed"]
    assert r["failure_reasons"]["not_converged"] == r["n_failed"]
    assert r["success_rate"] == pytest.approx(r["n_ok"] / r["n_completed"], rel=1e-6)


def test_low_success_rate_is_flagged_as_optimistic():
    """Fits fail preferentially on awkward resamples, so surviving replicates
    understate the uncertainty. That has to be said, not buried."""
    subs, vals = _subjects()
    state = {"n": 0}

    def mostly_fail(rep, seed):
        state["n"] += 1
        if state["n"] % 2 == 0:
            raise RuntimeError("boom")
        return _mean_fit(rep, seed)

    r = run_bootstrap("m", subs, _nlme(vals), fit_fn=mostly_fail, n_boot=80,
                      seed=1, params=("CL",))
    assert r["success_rate"] < 0.8
    assert any("optimistic" in n for n in r["notes"])
    assert r["failure_reasons"]["RuntimeError"] > 0


def test_too_few_successes_refuses_to_report_a_ci():
    subs, vals = _subjects()

    def always_fail(rep, seed):
        raise RuntimeError("no")

    r = run_bootstrap("m", subs, _nlme(vals), fit_fn=always_fail, n_boot=30,
                      seed=1, params=("CL",))
    assert r["status"] == "too_few_successful_fits"
    assert "parameters" not in r
    assert r["min_required"] == _MIN_OK_FOR_CI


def test_unstratified_run_warns_about_representativeness():
    subs, vals = _subjects()
    r = run_bootstrap("m", subs, _nlme(vals), fit_fn=_mean_fit, n_boot=60,
                      seed=1, params=("CL",))
    assert any("unstratified" in n for n in r["notes"])


def test_low_replicate_count_points_at_the_stability_trace():
    subs, vals = _subjects()
    r = run_bootstrap("m", subs, _nlme(vals), fit_fn=_mean_fit, n_boot=60,
                      seed=1, params=("CL",))
    assert any("stability" in n for n in r["notes"])


# ── stability trace ─────────────────────────────────────────────────────────

def test_stability_trace_is_reported_at_increasing_replicate_counts():
    """A CI still moving at the final count has not converged, whatever the
    nominal replicate number."""
    subs, vals = _subjects()
    r = run_bootstrap("m", subs, _nlme(vals), fit_fn=_mean_fit, n_boot=300,
                      seed=1, params=("CL",))
    marks = [s["n_replicates"] for s in r["stability"]]
    assert marks == sorted(marks) and marks[-1] == r["n_ok"]
    assert all(m >= _MIN_OK_FOR_CI for m in marks)
    last = r["stability"][-1]["parameters"][0]
    assert last["lo"] == r["parameters"][0]["boot_lo"]
    assert last["hi"] == r["parameters"][0]["boot_hi"]


# ── contract / degenerate input ─────────────────────────────────────────────

def test_deterministic_for_a_fixed_seed():
    subs, vals = _subjects()
    kw = dict(fit_fn=_mean_fit, n_boot=120, seed=99, params=("CL",))
    a = run_bootstrap("m", subs, _nlme(vals), **kw)
    b = run_bootstrap("m", subs, _nlme(vals), **kw)
    assert a["parameters"] == b["parameters"]
    assert a["stability"] == b["stability"]


def test_default_parameters_cover_thetas_and_iiv():
    subs, vals = _subjects()
    nl = dict(_nlme(vals), iiv_params=["CL"], omega_cv_pct={"CL": 30.0},
              omega_rse_pct={"CL": 20.0})

    def fit(rep, seed):
        return {"status": "ok", "converged": True,
                "theta": {"CL": float(np.mean([s["_cl"] for s in rep]))},
                "iiv_params": ["CL"], "omega_cv_pct": {"CL": 30.0}, "sigma": {}}

    r = run_bootstrap("m", subs, nl, fit_fn=fit, n_boot=60, seed=1)
    assert {p["parameter"] for p in r["parameters"]} == {"CL", "omega_CL"}


def test_rejects_misaligned_strata():
    subs, vals = _subjects()
    with pytest.raises(ValueError, match="must align"):
        run_bootstrap("m", subs, _nlme(vals), fit_fn=_mean_fit, n_boot=5,
                      strata=[1, 2, 3])


def test_rejects_an_impossible_ci_level():
    subs, vals = _subjects()
    with pytest.raises(ValueError, match="ci_level"):
        run_bootstrap("m", subs, _nlme(vals), fit_fn=_mean_fit, n_boot=5,
                      ci_level=1.5)


def test_needs_a_converged_fit_and_enough_subjects():
    subs, vals = _subjects()
    assert run_bootstrap("m", subs[:1], _nlme(vals), fit_fn=_mean_fit,
                         n_boot=5)["status"] == "insufficient_subjects"
    assert run_bootstrap("m", subs, {"status": "no_fit"}, fit_fn=_mean_fit,
                         n_boot=5)["status"] == "no_fit"


def test_result_is_json_safe():
    import json
    subs, vals = _subjects()
    r = run_bootstrap("m", subs, _nlme(vals), fit_fn=_mean_fit, n_boot=60,
                      seed=1, params=("CL",))
    s = json.dumps(r)
    assert "NaN" not in s and "Infinity" not in s


# ── tool layer ──────────────────────────────────────────────────────────────

def test_bootstrap_tool_is_not_llm_reachable():
    """A bootstrap is hundreds of real NLME fits. The guardrail is "never
    submit a real NLME/SCM fit from an automated loop", so this tool must be
    unreachable from a chat turn -- HTTP endpoint only, like run_simest."""
    from app.agents.definitions import AGENTS, DESCRIPTIONS
    from app.agents.supervisor import KEYWORDS
    from app.tools.builtins import default_registry

    tool = default_registry()._tools["run_bootstrap"]
    assert tool.agent == "simulator"
    # Supervisor.route can only return a key in KEYWORDS, or via its LLM
    # fallback a key in DESCRIPTIONS; Agent.run_turn scopes tools to the routed
    # agent. "simulator" in none of them => no chat path can reach this tool.
    assert "simulator" not in AGENTS
    assert "simulator" not in DESCRIPTIONS
    assert "simulator" not in KEYWORDS


def test_bootstrap_tool_requires_confirm():
    from app.core.pharmstate import PharmState
    from app.tools.base import ToolContext
    from app.tools.bootstrap_tools import run_bootstrap as tool_run

    res = tool_run(PharmState(), ToolContext(), {})
    assert res.result["status"] == "confirm_required"
    assert res.writes == {}, "a rejection must not write state"


def test_bootstrap_tool_rejections_never_overwrite_a_completed_run():
    """An admission-style rejection destroying a finished multi-hour run is the
    defect the simest tests guard; the same must hold here."""
    from app.core.pharmstate import PharmState
    from app.tools.base import ToolContext
    from app.tools.bootstrap_tools import run_bootstrap as tool_run

    prior = {"status": "ok", "n_ok": 200}
    st = PharmState(bootstrap_results=prior)
    for args in ({}, {"confirm": True}):          # confirm_required, then no_fit
        res = tool_run(st, ToolContext(), args)
        assert res.writes == {}
    assert st.bootstrap_results == prior


def test_bootstrap_results_is_writable_by_the_simulator_agent():
    from app.core.pharmstate import AGENT_WRITE_FIELDS
    assert "bootstrap_results" in AGENT_WRITE_FIELDS["simulator"]


def test_covariate_spec_is_restated_for_each_replicate():
    """Each replicate must estimate the SAME structure. Dropping the covariate
    model would bootstrap a different (covariate-free) model, and the intervals
    would not describe the fit being reported."""
    from app.tools.bootstrap_tools import _cov_spec
    nl = {"covariate_effects": [
        {"param": "CL", "covariate": "EGFR", "kind": "power", "center": 90.0},
        {"param": "CL", "covariate": "SEX", "kind": "categorical"}]}
    spec = _cov_spec(nl)
    assert spec == [{"param": "CL", "covariate": "EGFR", "kind": "power", "center": 90.0},
                    {"param": "CL", "covariate": "SEX", "kind": "categorical"}]
    assert _cov_spec({"covariate_effects": []}) is None
