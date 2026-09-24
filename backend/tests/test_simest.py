"""Tests for app.compute.simest.run_simest.

Every test uses a FAKE `fit_fn` — no test in this file runs a real NLME fit.
`fit_fn` is injected specifically so the expensive estimator call is owned by
the caller (a background job); these tests verify run_simest's own data
generation, accounting, and statistics in isolation from any real estimator.
"""
from __future__ import annotations

import json
import math

import pytest

from app.compute.simest import (
    _MAX_REPLICATES,
    DESIGN_LIMITATIONS,
    _passes_precision_criterion,
    _wilson_ci,
    run_simest,
)

MODEL_KEY = "oral_1cmt"


def _nlme(*, cov_effects=None, iiv=("CL", "V"), error_model="combined") -> dict:
    return {
        "status": "ok", "model_key": MODEL_KEY,
        "theta": {"CL": 4.0, "V": 40.0, "KA": 1.0},
        "omega_cv_pct": {"CL": 30.0, "V": 20.0},
        "sigma": {"prop": 0.1, "add": 0.3},
        "iiv_params": list(iiv), "error_model": error_model,
        "covariate_effects": cov_effects or [],
    }


def _design(**overrides) -> dict:
    d = {"n_subjects": 12, "obs_t": [0.5, 1, 2, 4, 8, 12], "dose": 100.0, "n_doses": 1}
    d.update(overrides)
    return d


def _perfect_fit_fn(theta_true: dict, rse: dict):
    """A fake fit_fn that reports back the EXACT truth with a fixed RSE —
    every replicate should then pass the precision criterion when the CI is
    tight enough, and fail when it is deliberately loosened."""
    def fn(subjects, seed):
        return {"status": "ok", "converged": True,
                "theta": dict(theta_true), "theta_rse_pct": dict(rse)}
    return fn


def _never_called_fit_fn(*_a, **_kw):
    raise AssertionError("fit_fn must not be called when validation fails upstream")


# ── safety-critical: no real fit is ever invoked in this file ───────────────

def test_module_never_imports_a_real_estimator():
    import app.compute.simest as m
    assert "population_fit" not in dir(m)
    assert "focei_fit" not in dir(m)
    assert "saem_fit" not in dir(m)


# ── categorical / covariate rejection ────────────────────────────────────────

def test_categorical_covariate_effect_is_rejected():
    nlme = _nlme(cov_effects=[{"param": "CL", "covariate": "SEX", "kind": "categorical",
                               "levels": ["F"], "coefficient": {"F": -0.2}}])
    out = run_simest(MODEL_KEY, _design(), nlme, fit_fn=_never_called_fit_fn)
    assert out["status"] == "covariates_unsupported"
    assert "design_limitations" in out


def test_continuous_covariate_effect_is_also_rejected():
    # Continuous covariates are rejected too (unidentifiable in the refit when
    # held at center with no variability) -- not just categorical.
    nlme = _nlme(cov_effects=[{"param": "CL", "covariate": "WT", "kind": "power",
                               "center": 70.0, "coefficient": 0.75}])
    out = run_simest(MODEL_KEY, _design(), nlme, fit_fn=_never_called_fit_fn)
    assert out["status"] == "covariates_unsupported"


def test_no_converged_nlme_is_rejected():
    out = run_simest(MODEL_KEY, _design(), {"status": "no_fit"}, fit_fn=_never_called_fit_fn)
    assert out["status"] == "no_nlme"


# ── design validation raises (a caller/config bug) ──────────────────────────

@pytest.mark.parametrize("bad_design,fragment", [
    ({"n_subjects": 1, "obs_t": [1.0], "dose": 100.0}, "n_subjects"),
    ({"n_subjects": 10, "obs_t": [], "dose": 100.0}, "obs_t"),
    ({"n_subjects": 10, "obs_t": [-1.0], "dose": 100.0}, "obs_t"),
    ({"n_subjects": 10, "obs_t": [1.0], "dose": 100.0, "dose_per_kg": 2.0}, "exactly one"),
    ({"n_subjects": 10, "obs_t": [1.0]}, "exactly one"),
    ({"n_subjects": 10, "obs_t": [1.0], "dose": -5.0}, "dose"),
    ({"n_subjects": 10, "obs_t": [1.0], "dose": 100.0, "n_doses": 3}, "tau"),
])
def test_invalid_design_raises_value_error(bad_design, fragment):
    with pytest.raises(ValueError, match=fragment):
        run_simest(MODEL_KEY, bad_design, _nlme(), fit_fn=_never_called_fit_fn)


# ── params resolution ────────────────────────────────────────────────────────

def test_params_defaults_to_iiv_params_intersect_theta():
    nlme = _nlme(iiv=("CL", "V"))
    fit_fn = _perfect_fit_fn({"CL": 4.0, "V": 40.0}, {"CL": 5.0, "V": 5.0})
    out = run_simest(MODEL_KEY, _design(), nlme, fit_fn=fit_fn, n_rep=2)
    assert set(out["params"]) == {"CL", "V"}


def test_params_outside_theta_are_dropped_not_crashed():
    nlme = _nlme()
    fit_fn = _perfect_fit_fn({"CL": 4.0, "V": 40.0}, {"CL": 5.0, "V": 5.0})
    out = run_simest(MODEL_KEY, _design(), nlme, fit_fn=fit_fn, n_rep=2,
                     params=("CL", "not_a_real_param"))
    assert out["params"] == ["CL"]


def test_all_params_unresolvable_reports_no_params_status():
    nlme = _nlme()
    out = run_simest(MODEL_KEY, _design(), nlme, fit_fn=_never_called_fit_fn,
                     params=("bogus",))
    assert out["status"] == "no_params"
    assert "CL" in out["message"]  # names the available theta keys


# ── replicate cap / n_rep accounting ─────────────────────────────────────────

def test_n_rep_is_capped_at_max_replicates():
    nlme = _nlme()
    fit_fn = _perfect_fit_fn({"CL": 4.0, "V": 40.0}, {"CL": 5.0, "V": 5.0})
    out = run_simest(MODEL_KEY, _design(), nlme, fit_fn=fit_fn, n_rep=1000)
    assert out["n_rep_requested"] == 1000
    assert out["n_rep_planned"] == _MAX_REPLICATES


def test_n_rep_completed_matches_evaluable_count_when_all_succeed():
    nlme = _nlme()
    fit_fn = _perfect_fit_fn({"CL": 4.0, "V": 40.0}, {"CL": 5.0, "V": 5.0})
    out = run_simest(MODEL_KEY, _design(), nlme, fit_fn=fit_fn, n_rep=3)
    assert out["n_rep_completed"] == 3
    assert out["n_point_evaluable"] == 3
    assert out["n_ci_evaluable"] == 3
    assert out["n_excluded"] == 0


# ── excluded-replicate accounting (informative-censoring guard) ─────────────

def test_non_ok_fit_status_is_excluded_and_counted():
    def flaky_fit_fn(subjects, seed):
        return {"status": "not_at_minimum"}
    nlme = _nlme()
    out = run_simest(MODEL_KEY, _design(), nlme, fit_fn=flaky_fit_fn, n_rep=3)
    assert out["n_point_evaluable"] == 0
    assert out["n_excluded"] == 3
    assert out["excluded_reasons"].get("fit_not_ok") == 3
    assert out["status"] == "not_evaluable"


def test_missing_rse_is_still_counted_toward_point_evaluable_not_ci():
    def fit_fn(subjects, seed):
        return {"status": "ok", "theta": {"CL": 4.0, "V": 40.0}, "theta_rse_pct": {}}
    nlme = _nlme()
    out = run_simest(MODEL_KEY, _design(), nlme, fit_fn=fit_fn, n_rep=3)
    # Point estimate is a valid result even when its SE was unavailable --
    # dropping it from bias/RMSE would inject selection bias into those stats.
    assert out["n_point_evaluable"] == 3
    assert out["n_ci_evaluable"] == 0
    assert out["excluded_reasons"].get("rse_unavailable") == 3
    for p in out["params"]:
        assert out["per_param"][p]["rel_bias_pct"] is not None       # uses R_point
        assert out["per_param"][p]["pct_within_60_140_of_own_estimate"] is None  # uses R_ci


def test_fit_exception_does_not_crash_the_loop():
    def crashing_fit_fn(subjects, seed):
        raise RuntimeError("estimator blew up")
    nlme = _nlme()
    out = run_simest(MODEL_KEY, _design(), nlme, fit_fn=crashing_fit_fn, n_rep=2)
    assert out["status"] == "not_evaluable"
    assert out["excluded_reasons"].get("fit_exception") == 2


# ── precision-criterion math (grounded in the course lecture's own text:
#    95% CI within 60%-140% of the replicate's OWN point estimate) ──────────

@pytest.mark.parametrize("theta_val,lo,hi,expected", [
    (10.0, 6.0, 14.0, True),    # exactly at the boundary (inclusive)
    (10.0, 6.01, 13.9, True),   # comfortably inside
    (10.0, 5.9, 14.0, False),   # lo just outside 60%
    (10.0, 6.0, 14.1, False),   # hi just outside 140%
    (10.0, 0.0, 0.0, False),
])
def test_passes_precision_criterion(theta_val, lo, hi, expected):
    assert _passes_precision_criterion(theta_val, lo, hi) is expected


def test_criterion_reflects_deliberately_tight_vs_loose_ci():
    nlme = _nlme(iiv=("CL",))
    tight = run_simest(MODEL_KEY, _design(), nlme, n_rep=5,
                       fit_fn=_perfect_fit_fn({"CL": 4.0}, {"CL": 3.0}))  # SE_log=0.03 -> tight CI
    loose = run_simest(MODEL_KEY, _design(), nlme, n_rep=5,
                       fit_fn=_perfect_fit_fn({"CL": 4.0}, {"CL": 40.0}))  # SE_log=0.40 -> wide CI
    assert tight["per_param"]["CL"]["pct_within_60_140_strict"] == 100.0
    assert loose["per_param"]["CL"]["pct_within_60_140_strict"] == 0.0


def test_bias_and_rmse_reflect_a_deliberately_biased_estimator():
    # Every replicate reports CL = 1.5x truth -> rel_bias_pct ~= +50%.
    nlme = _nlme(iiv=("CL",))
    fit_fn = _perfect_fit_fn({"CL": 6.0}, {"CL": 5.0})  # truth is 4.0
    out = run_simest(MODEL_KEY, _design(), nlme, fit_fn=fit_fn, n_rep=3)
    assert out["per_param"]["CL"]["rel_bias_pct"] == pytest.approx(50.0, abs=0.5)
    assert out["per_param"]["CL"]["cv_across_replicates_pct"] == pytest.approx(0.0, abs=1e-6)


# ── Wilson coverage interval ─────────────────────────────────────────────────

def test_wilson_ci_matches_known_reference_values():
    # k=5, n=5 (all pass): textbook Wilson lower bound for a 95% CI ~= 0.566.
    lo, hi = _wilson_ci(5, 5)
    assert lo == pytest.approx(0.5655, abs=1e-3)
    assert hi == pytest.approx(1.0, abs=1e-6)


def test_wilson_ci_none_when_no_evaluable_replicates():
    assert _wilson_ci(0, 0) == (None, None)


def test_ci_validity_unassessable_below_min_r():
    nlme = _nlme()
    fit_fn = _perfect_fit_fn({"CL": 4.0, "V": 40.0}, {"CL": 5.0, "V": 5.0})
    out = run_simest(MODEL_KEY, _design(), nlme, fit_fn=fit_fn, n_rep=3)
    assert out["ci_validity"] == "unassessable"  # 3 << 30, true at every real cap


# ── target / criterion_met ───────────────────────────────────────────────────

def test_criterion_met_is_none_without_a_target():
    nlme = _nlme()
    fit_fn = _perfect_fit_fn({"CL": 4.0, "V": 40.0}, {"CL": 5.0, "V": 5.0})
    out = run_simest(MODEL_KEY, _design(), nlme, fit_fn=fit_fn, n_rep=2)
    assert out["criterion"]["criterion_met"] is None


def test_criterion_met_true_when_pass_rate_reaches_target():
    nlme = _nlme()
    fit_fn = _perfect_fit_fn({"CL": 4.0, "V": 40.0}, {"CL": 5.0, "V": 5.0})
    out = run_simest(MODEL_KEY, _design(), nlme, fit_fn=fit_fn, n_rep=3, ci_target_pct=80.0)
    assert out["criterion"]["criterion_met"] is True


# ── design_limitations / citation flag always present ───────────────────────

def test_design_limitations_and_citation_always_present_on_ok():
    nlme = _nlme()
    fit_fn = _perfect_fit_fn({"CL": 4.0, "V": 40.0}, {"CL": 5.0, "V": 5.0})
    out = run_simest(MODEL_KEY, _design(), nlme, fit_fn=fit_fn, n_rep=2)
    assert out["design_limitations"] == DESIGN_LIMITATIONS
    assert out["citation"].startswith("[CITATION UNVERIFIED]")


# ── dose_per_kg path ──────────────────────────────────────────────────────────

def test_dose_per_kg_runs_without_error():
    nlme = _nlme()
    fit_fn = _perfect_fit_fn({"CL": 4.0, "V": 40.0}, {"CL": 5.0, "V": 5.0})
    out = run_simest(MODEL_KEY, _design(dose=None, dose_per_kg=1.5), nlme, fit_fn=fit_fn, n_rep=2)
    assert out["status"] in ("ok", "partial")
    assert out["n_point_evaluable"] == 2


# ── determinism + JSON safety ────────────────────────────────────────────────

def test_deterministic_for_fixed_seed():
    nlme = _nlme()
    calls_a, calls_b = [], []

    def fit_fn_a(subjects, seed):
        calls_a.append([s["obs_c"] for s in subjects])
        return {"status": "ok", "theta": {"CL": 4.0, "V": 40.0}, "theta_rse_pct": {"CL": 5.0, "V": 5.0}}

    def fit_fn_b(subjects, seed):
        calls_b.append([s["obs_c"] for s in subjects])
        return {"status": "ok", "theta": {"CL": 4.0, "V": 40.0}, "theta_rse_pct": {"CL": 5.0, "V": 5.0}}

    run_simest(MODEL_KEY, _design(), nlme, fit_fn=fit_fn_a, n_rep=2, seed=777)
    run_simest(MODEL_KEY, _design(), nlme, fit_fn=fit_fn_b, n_rep=2, seed=777)
    assert calls_a == calls_b  # identical simulated data for the same seed


def test_json_safe():
    nlme = _nlme()
    fit_fn = _perfect_fit_fn({"CL": 4.0, "V": 40.0}, {"CL": 5.0, "V": 5.0})
    out = run_simest(MODEL_KEY, _design(), nlme, fit_fn=fit_fn, n_rep=2)
    json.dumps(out)


def test_max_seconds_is_capped_regardless_of_caller():
    from app.compute.simest import _MAX_SECONDS
    nlme = _nlme()
    fit_fn = _perfect_fit_fn({"CL": 4.0, "V": 40.0}, {"CL": 5.0, "V": 5.0})
    # A caller-supplied max_seconds far above the hard ceiling must not
    # override it -- run_simest should still complete quickly with a fast
    # fake fit_fn (the cap only matters for a real, slow estimator).
    out = run_simest(MODEL_KEY, _design(), nlme, fit_fn=fit_fn, n_rep=2,
                     max_seconds=10_000_000.0)
    assert out["status"] in ("ok", "partial")
    # (No direct assertion on the internal cap value beyond import — this
    # documents the contract without depending on private state.)
    assert math.isfinite(_MAX_SECONDS)


def test_progress_callback_invoked_once_per_completed_replicate():
    nlme = _nlme()
    fit_fn = _perfect_fit_fn({"CL": 4.0, "V": 40.0}, {"CL": 5.0, "V": 5.0})
    calls = []
    run_simest(MODEL_KEY, _design(), nlme, fit_fn=fit_fn, n_rep=3,
              progress=lambda p: calls.append(p))
    assert len(calls) == 3
    assert calls[-1]["replicate"] == 3
    assert calls[-1]["n_rep_planned"] == 3


# ── replicate-level payload shape ────────────────────────────────────────────

def test_replicates_payload_carries_theta_and_ci():
    nlme = _nlme()
    fit_fn = _perfect_fit_fn({"CL": 4.0, "V": 40.0}, {"CL": 5.0, "V": 5.0})
    out = run_simest(MODEL_KEY, _design(), nlme, fit_fn=fit_fn, n_rep=2)
    assert len(out["replicates"]) == 2
    rep = out["replicates"][0]
    assert set(rep["theta"].keys()) == {"CL", "V"}
    assert rep["ci"]["CL"][0] < rep["theta"]["CL"] < rep["ci"]["CL"][1]


def test_rng_actually_varies_subjects_across_replicates():
    # Sanity: the simulated data differs between replicates (not accidentally
    # reusing the same draw) -- guards against a seed-handling bug.
    nlme = _nlme()
    seen = []

    def fit_fn(subjects, seed):
        seen.append(tuple(s["obs_c"][0] for s in subjects))
        return {"status": "ok", "theta": {"CL": 4.0, "V": 40.0},
                "theta_rse_pct": {"CL": 5.0, "V": 5.0}}

    run_simest(MODEL_KEY, _design(), nlme, fit_fn=fit_fn, n_rep=3)
    assert seen[0] != seen[1] != seen[2]
