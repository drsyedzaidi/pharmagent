"""Optimus-style dose selection against efficacy and toxicity E-R models.

Ground truth is brute-force arithmetic (expit of the declared coefficients,
U = P_eff - w * P_tox, argmax over the grid) so the selection logic is checked
independently of the logistic estimator, which is validated against R elsewhere.
"""
import json

import numpy as np
import pytest
from scipy.special import expit

from app.compute.optimus import select_dose

DOSES = [25.0, 50.0, 100.0, 150.0, 200.0, 300.0]
LINEAR = {"type": "linear", "reference_dose": 100.0, "reference_exposure": 200.0}   # AUC = 2 * dose


def _fit(a, b, *, sd_a=0.0, sd_b=0.0, rho=0.0, label="AUC", rng=(20.0, 700.0), draws=None):
    fit = {"status": "ok", "model": "logistic", "ci_level": 0.95, "exposure_label": label,
           "exposure_range": list(rng),
           "coef": {"intercept": {"estimate": a}, "slope": {"estimate": b}},
           "cov": [[sd_a ** 2, rho * sd_a * sd_b], [rho * sd_a * sd_b, sd_b ** 2]]}
    if draws is not None:
        fit["bootstrap"] = {"status": "ok", "draws": draws, "seed": 1}
    return fit


EFF = _fit(-2.0, 0.020)
TOX = _fit(-5.0, 0.015)


def _brute(w, cap=None, eff=(-2.0, 0.020), tox=(-5.0, 0.015)):
    e = 2.0 * np.array(DOSES)
    pe, pt = expit(eff[0] + eff[1] * e), expit(tox[0] + tox[1] * e)
    u = pe - w * pt
    ok = np.ones_like(u, dtype=bool) if cap is None else pt <= cap
    return pe, pt, u, ok


def _run(**kw):
    args = dict(doses=DOSES, exposure_mapping=LINEAR, efficacy_fit=EFF, toxicity_fit=TOX,
                utility_weight=1.0, n_draws=400, seed=11)
    args.update(kw)
    return select_dose(**args)


# ── point estimates and selection ────────────────────────────────────────────

def test_table_matches_brute_force_and_selects_the_argmax():
    r = _run(utility_weight=1.5)
    pe, pt, u, _ = _brute(1.5)
    assert r["status"] == "ok"
    for row, a, b, c in zip(r["doses"], pe, pt, u):
        assert row["p_eff"] == pytest.approx(a, rel=1e-9)
        assert row["p_tox"] == pytest.approx(b, rel=1e-9)
        assert row["utility"] == pytest.approx(c, rel=1e-9)
    assert r["selected"]["dose"] == DOSES[int(np.argmax(u))]
    assert r["selected"]["utility"] == pytest.approx(float(u.max()), rel=1e-9)


def test_interior_optimum_is_not_flagged_as_a_grid_edge():
    r = _run(utility_weight=1.5)
    assert r["selected"]["at_grid_edge"] is None
    assert not any("grid" in w.lower() for w in r["warnings"])


def test_toxicity_cap_excludes_doses_and_changes_the_selection():
    free = _run(utility_weight=0.2)
    pe, pt, u, ok = _brute(0.2, cap=0.15)
    capped = _run(utility_weight=0.2, toxicity_cap=0.15)
    assert not ok.all() and ok.any()
    assert capped["selected"]["dose"] == DOSES[int(np.argmax(np.where(ok, u, -np.inf)))]
    assert capped["selected"]["p_tox"] <= 0.15
    assert capped["selected"]["dose"] <= free["selected"]["dose"]
    assert [row["feasible"] for row in capped["doses"]] == ok.tolist()


def test_no_feasible_dose_is_reported_not_guessed():
    r = _run(toxicity_cap=0.001)
    assert r["status"] == "no_feasible_dose" and "selected" not in r
    assert len(r["doses"]) == len(DOSES) and not any(d["feasible"] for d in r["doses"])


@pytest.mark.parametrize("w,edge", [(0.0, "max"), (50.0, "min")])
def test_selection_at_the_grid_edge_carries_an_extrapolation_warning(w, edge):
    r = _run(utility_weight=w)
    assert r["selected"]["at_grid_edge"] == edge
    assert any("edge of the dose grid" in x.lower() and "extrapolat" in x.lower() for x in r["warnings"])


# ── the declared utility is never defaulted silently ─────────────────────────

def test_utility_weight_is_required():
    r = _run(utility_weight=None)
    assert r["status"] == "utility_not_declared"
    assert "utility_weight" in r["message"] and "default_w=1" in r["message"]
    assert "selected" not in r


def test_default_weight_needs_the_explicit_acknowledgement_and_is_echoed():
    r = _run(utility_weight=None, utility="default_w=1")
    assert r["status"] == "ok"
    assert r["utility"]["w"] == 1.0 and r["utility"]["w_source"] == "default_acknowledged"
    assert r["utility"]["declared"] == "default_w=1"
    declared = _run(utility_weight=2.0)
    assert declared["utility"]["w_source"] == "declared" and declared["utility"]["w"] == 2.0
    assert "P_eff" in declared["utility"]["formula"] and "P_tox" in declared["utility"]["formula"]


@pytest.mark.parametrize("kw", [{"utility_weight": -1.0}, {"utility_weight": float("nan")},
                                {"utility_weight": None, "utility": "w=1"}])
def test_invalid_utility_declarations_are_refused(kw):
    assert _run(**kw)["status"] in ("invalid_input", "utility_not_declared")


# ── uncertainty propagation ──────────────────────────────────────────────────

def test_probability_optimal_sums_to_one_and_is_concentrated_when_certain():
    certain = _run(utility_weight=1.5)                     # zero covariance -> every draw identical
    sel = certain["selected"]["dose"]
    probs = {d["dose"]: d["prob_optimal"] for d in certain["doses"]}
    assert probs[sel] == pytest.approx(1.0) and sum(probs.values()) + certain["prob_none_feasible"] == pytest.approx(1.0)
    noisy = _run(utility_weight=1.5, efficacy_fit=_fit(-2.0, 0.020, sd_a=0.4, sd_b=0.004),
                 toxicity_fit=_fit(-5.0, 0.015, sd_a=0.5, sd_b=0.004), n_draws=1500)
    p2 = [d["prob_optimal"] for d in noisy["doses"]]
    assert sum(p2) + noisy["prob_none_feasible"] == pytest.approx(1.0)
    assert max(p2) < 0.95 and sum(1 for p in p2 if p > 0.02) >= 2


def test_utility_band_brackets_the_point_estimate_and_widens_with_uncertainty():
    noisy = _run(efficacy_fit=_fit(-2.0, 0.020, sd_a=0.2, sd_b=0.002),
                 toxicity_fit=_fit(-5.0, 0.015, sd_a=0.2, sd_b=0.002), n_draws=1500)
    for d in noisy["doses"]:
        assert d["utility_lo"] <= d["utility"] <= d["utility_hi"]
        assert d["p_eff_lo"] <= d["p_eff"] <= d["p_eff_hi"]
    certain = _run()
    assert all(d["utility_hi"] - d["utility_lo"] == pytest.approx(0.0, abs=1e-9) for d in certain["doses"])


def test_probability_of_feasibility_under_a_cap_reflects_uncertainty():
    r = _run(toxicity_fit=_fit(-5.0, 0.015, sd_a=0.5, sd_b=0.004), toxicity_cap=0.2, n_draws=1500)
    pf = [d["prob_feasible"] for d in r["doses"]]
    assert pf[0] > pf[-1] and any(0.0 < p < 1.0 for p in pf)


def test_seed_is_echoed_and_the_run_is_reproducible():
    kw = dict(efficacy_fit=_fit(-2.0, 0.020, sd_a=0.4, sd_b=0.004), n_draws=300)
    a, b, c = _run(seed=7, **kw), _run(seed=7, **kw), _run(seed=8, **kw)
    assert a["uncertainty"]["seed"] == 7 and a == b
    assert [d["prob_optimal"] for d in a["doses"]] != [d["prob_optimal"] for d in c["doses"]]


def test_bootstrap_draws_are_used_when_present_and_reported():
    rng = np.random.default_rng(1)
    draws = np.column_stack([rng.normal(-2.0, 0.3, 300), rng.normal(0.020, 0.003, 300)]).tolist()
    r = _run(efficacy_fit=_fit(-2.0, 0.020, draws=draws))
    u = r["uncertainty"]
    assert u["source"] == {"efficacy": "bootstrap", "toxicity": "asymptotic_mvn"}
    assert u["n_draws"] == 300 and u["seed"] == 11


# ── exposure mapping ─────────────────────────────────────────────────────────

def test_power_and_linear_scaling_formulas():
    pw = _run(exposure_mapping={"type": "power", "reference_dose": 100.0, "reference_exposure": 200.0,
                                "exponent": 1.2})
    for row in pw["doses"]:
        assert row["exposure"] == pytest.approx(200.0 * (row["dose"] / 100.0) ** 1.2, rel=1e-9)
    lin = _run()
    for row in lin["doses"]:
        assert row["exposure"] == pytest.approx(2.0 * row["dose"], rel=1e-9)


def test_by_dose_mapping_must_cover_the_whole_grid():
    full = {"type": "by_dose", "values": {str(d): 2.0 * d for d in DOSES}}
    assert _run(exposure_mapping=full)["status"] == "ok"
    partial = {"type": "by_dose", "values": {"25": 50.0, "50": 100.0}}
    r = _run(exposure_mapping=partial)
    assert r["status"] == "invalid_input" and "100" in r["message"]


def test_dose_proportionality_mapping_uses_the_fitted_power_model():
    m = {"type": "dose_proportionality", "intercept": float(np.log(2.0)), "slope": 1.0, "dose_levels": [25.0, 100.0]}
    r = _run(exposure_mapping=m)
    for row in r["doses"]:
        assert row["exposure"] == pytest.approx(2.0 * row["dose"], rel=1e-9)
    assert [d["dose_extrapolated"] for d in r["doses"]] == [False, False, False, True, True, True]
    assert any("dose" in w.lower() and "studied" in w.lower() for w in r["warnings"])


def test_exposures_outside_the_observed_er_range_are_flagged():
    r = _run(efficacy_fit=_fit(-2.0, 0.020, rng=(60.0, 250.0)))
    flags = {d["dose"]: d["exposure_extrapolated"] for d in r["doses"]}
    assert flags[25.0] is True and flags[100.0] is False and flags[300.0] is True
    assert any("observed exposure range" in w.lower() for w in r["warnings"])


def test_a_single_mapping_requires_matching_exposure_metrics():
    r = _run(toxicity_fit=_fit(-5.0, 0.015, label="Cmax"))
    assert r["status"] == "exposure_metric_mismatch"
    both = {"efficacy": LINEAR, "toxicity": {"type": "linear", "reference_dose": 100.0, "reference_exposure": 30.0}}
    ok = _run(toxicity_fit=_fit(-5.0, 0.015, label="Cmax", rng=(5.0, 120.0)), exposure_mapping=both)
    assert ok["status"] == "ok"
    row = next(d for d in ok["doses"] if d["dose"] == 100.0)
    assert row["exposure"] == pytest.approx(200.0)               # efficacy metric
    assert row["exposure_tox"] == pytest.approx(30.0)            # toxicity metric


# ── input validation ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("doses", [[100.0], [], [50.0, 50.0, 100.0], [-10.0, 50.0], [0.0, 50.0], [10.0, float("nan")]])
def test_bad_dose_grids_are_refused(doses):
    assert _run(doses=doses)["status"] == "invalid_input"


def test_non_logistic_or_failed_fits_are_refused():
    assert _run(efficacy_fit={"status": "separation"})["status"] == "invalid_input"
    assert _run(toxicity_fit={"status": "ok", "model": "cox"})["status"] == "invalid_input"


@pytest.mark.parametrize("kw", [{"toxicity_cap": 0.0}, {"toxicity_cap": 1.5}, {"n_draws": 5}])
def test_bad_cap_or_draw_count_is_refused(kw):
    assert _run(**kw)["status"] == "invalid_input"


def test_result_is_json_safe_and_states_the_plug_in_limitation():
    r = _run(efficacy_fit=_fit(-2.0, 0.020, sd_a=0.3, sd_b=0.003))
    s = json.dumps(r)
    assert "NaN" not in s and "Infinity" not in s
    assert any("typical exposure" in n.lower() for n in r["notes"])
