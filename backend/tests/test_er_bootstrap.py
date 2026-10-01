"""Seeded subject-level bootstrap for the exposure-response fits (logistic and Cox).

Same contract as ``app.compute.bootstrap``: whole subjects are resampled, failed
replicates are COUNTED never silently dropped, and the seed is an input that is
echoed so a result can be reproduced from the audit record.
"""
import json

import numpy as np
import pytest

from app.compute.er_bootstrap import bootstrap_cox, bootstrap_logistic
from app.compute.er_models import fit_logistic_er


def _logistic_data(n=400, seed=21, a=-3.0, b=0.03):
    rng = np.random.default_rng(seed)
    x = np.exp(rng.normal(np.log(100.0), 0.4, n))
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-(a + b * x)))).astype(int)
    return x.tolist(), y.tolist()


def _cox_data(n=300, seed=4, beta=0.5):
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, n)
    t_ev = rng.exponential(1.0 / (0.1 * np.exp(beta * x)))
    t_ce = rng.exponential(12.0, n)
    return np.minimum(t_ev, t_ce).tolist(), (t_ev <= t_ce).astype(int).tolist(), x.tolist()


# ── logistic ─────────────────────────────────────────────────────────────────

def test_seed_is_echoed_and_the_run_is_reproducible():
    x, y = _logistic_data()
    a = bootstrap_logistic(x, y, n_boot=120, seed=99)
    b = bootstrap_logistic(x, y, n_boot=120, seed=99)
    c = bootstrap_logistic(x, y, n_boot=120, seed=100)
    assert a["status"] == "ok" and a["seed"] == 99
    assert a["slope"] == b["slope"] and a["draws"] == b["draws"]
    assert a["slope"] != c["slope"]


def test_percentile_interval_brackets_the_mle_and_se_tracks_the_wald_se():
    x, y = _logistic_data()
    fit = fit_logistic_er(x, y)
    r = bootstrap_logistic(x, y, n_boot=500, seed=5)
    s = r["slope"]
    assert s["lo"] < fit["coef"]["slope"]["estimate"] < s["hi"]
    wald = fit["coef"]["slope"]["se"]
    assert 0.75 * wald < s["se"] < 1.35 * wald
    # OR per unit is the exponentiated slope interval; per SD uses the ORIGINAL sample SD
    sd = fit["or_per_sd"]["sd"]
    assert r["or_per_unit"]["lo"] == pytest.approx(np.exp(s["lo"]), rel=1e-6)
    assert r["or_per_sd"]["hi"] == pytest.approx(np.exp(s["hi"] * sd), rel=1e-6)


def test_probability_band_is_ordered_and_on_the_requested_grid():
    x, y = _logistic_data()
    grid = [60.0, 100.0, 150.0]
    r = bootstrap_logistic(x, y, n_boot=200, seed=2, grid=grid)
    cur = r["curve"]
    assert cur["exposure"] == grid
    for lo, med, hi in zip(cur["lo"], cur["prob_median"], cur["hi"]):
        assert 0.0 <= lo <= med <= hi <= 1.0


def test_failed_replicates_are_counted_not_dropped():
    # 14 subjects / 5 events: many resamples are separated or single-class
    x = [3.1, 4.0, 2.2, 5.5, 6.1, 7.7, 3.9, 8.2, 9.1, 4.4, 10.5, 6.6, 5.1, 7.0]
    y = [0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1]
    r = bootstrap_logistic(x, y, n_boot=300, seed=1)
    assert r["status"] == "ok"
    assert r["n_failed"] > 0 and r["n_ok"] + r["n_failed"] == r["n_completed"] == 300
    assert sum(r["failure_reasons"].values()) == r["n_failed"]
    assert any("converge" in n.lower() or "failed" in n.lower() for n in r["notes"])
    assert len(r["draws"]) == r["n_ok"]


def test_a_refused_base_fit_is_passed_through_not_bootstrapped():
    r = bootstrap_logistic(list(range(1, 13)), [0] * 6 + [1] * 6, n_boot=50, seed=1)
    assert r["status"] == "separation" and "draws" not in r


def test_misaligned_strata_and_bad_ci_level_are_rejected():
    x, y = _logistic_data(80)
    with pytest.raises(ValueError, match="strata"):
        bootstrap_logistic(x, y, n_boot=30, seed=1, strata=[1, 2, 3])
    with pytest.raises(ValueError, match="ci_level"):
        bootstrap_logistic(x, y, n_boot=30, seed=1, ci_level=1.2)


def test_stratified_run_is_flagged_and_replicate_cap_is_enforced():
    x, y = _logistic_data(120)
    r = bootstrap_logistic(x, y, n_boot=5000, seed=3, strata=y)
    assert r["stratified"] is True and r["n_boot_requested"] == 1000


def test_result_is_json_safe():
    x, y = _logistic_data(150)
    s = json.dumps(bootstrap_logistic(x, y, n_boot=60, seed=1))
    assert "NaN" not in s and "Infinity" not in s


# ── Cox ──────────────────────────────────────────────────────────────────────

def test_cox_bootstrap_brackets_the_point_estimate_and_is_seeded():
    t, e, x = _cox_data()
    a = bootstrap_cox(t, e, {"x": x}, n_boot=150, seed=8)
    b = bootstrap_cox(t, e, {"x": x}, n_boot=150, seed=8)
    assert a["status"] == "ok" and a["seed"] == 8 and a == b
    h = a["hr_per_unit"]
    assert h["lo"] < h["estimate"] < h["hi"]
    assert a["hr_per_sd"]["estimate"] == pytest.approx(h["estimate"] ** a["sd_exposure"], rel=1e-6)
    assert a["n_ok"] + a["n_failed"] == a["n_completed"]


def test_cox_bootstrap_passes_a_refused_base_fit_through():
    r = bootstrap_cox([1, 2, 3, 4], [0, 0, 0, 0], {"x": [1, 2, 3, 4]}, n_boot=30, seed=1)
    assert r["status"] == "insufficient_events"
