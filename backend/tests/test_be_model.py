"""The 2x2 crossover and parallel estimators (app.compute.be_model).

Crossover model: log(y) with FIXED effects sequence, subject within sequence, period and formulation (the model EMA
CPMP/EWP/QWP/1401/98 Rev. 1 sec. 4.1.8 names; ICH M13A sec. 2.2.3.2 asks for tests of the same four effects).
Each result is checked three ways that do not share code with the implementation:
  1. the closed-form 2x2 solution (Chow & Liu): estimate = (mean d_RT - mean d_TR) / 2, with d the within-subject
     period difference, and the pooled within-sequence variance of d on n1 + n2 - 2 df;
  2. a generic dummy-variable least-squares fit (numpy.linalg.lstsq + pseudo-inverse);
  3. scipy for the parallel design.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from app.compute.be_model import crossover_2x2, parallel_two_group


def _make(n_tr, n_rt, *, seed=1, form=-0.04, per=0.15, sd_b=0.30, sd_w=0.12):
    """Raw positive concentrations for n_tr subjects in sequence TR and n_rt in RT (test_first=True for TR)."""
    rng = np.random.default_rng(seed)
    n = n_tr + n_rt
    test_first = np.array([True] * n_tr + [False] * n_rt)
    subj = rng.normal(0.0, sd_b, n)
    e1, e2 = rng.normal(0.0, sd_w, n), rng.normal(0.0, sd_w, n)
    y_first = 4.0 + subj + np.where(test_first, form / 2, -form / 2) - per / 2 + e1
    y_second = 4.0 + subj + np.where(test_first, -form / 2, form / 2) + per / 2 + e2
    log_t = np.where(test_first, y_first, y_second)
    log_r = np.where(test_first, y_second, y_first)
    return np.exp(log_t), np.exp(log_r), test_first


def _closed_form(test, ref, test_first, alpha=0.10):
    lt, lr = np.log(test), np.log(ref)
    y1 = np.where(test_first, lt, lr)
    y2 = np.where(test_first, lr, lt)
    d = y2 - y1
    d_tr, d_rt = d[test_first], d[~test_first]
    n1, n2 = len(d_tr), len(d_rt)
    est = 0.5 * (d_rt.mean() - d_tr.mean())
    s2 = (((d_tr - d_tr.mean()) ** 2).sum() + ((d_rt - d_rt.mean()) ** 2).sum()) / (n1 + n2 - 2)
    se = 0.5 * math.sqrt(s2 * (1 / n1 + 1 / n2))
    df = n1 + n2 - 2
    t = stats.t.ppf(1 - alpha / 2, df)
    return {"est": est, "se": se, "df": df, "mse": s2 / 2, "lo": 100 * math.exp(est - t * se),
            "hi": 100 * math.exp(est + t * se), "gmr": 100 * math.exp(est)}


def _dummy_ols(test, ref, test_first):
    """Formulation coefficient and SE from y = subject + period + formulation (full-rank dummy coding)."""
    n = len(test)
    rows, y = [], []
    for i in range(n):
        for period in (0, 1):
            is_test = (period == 0) == bool(test_first[i])
            subj = np.zeros(n)
            subj[i] = 1.0
            rows.append(np.concatenate([subj, [period - 0.5, 0.5 if is_test else -0.5]]))
            y.append(math.log(test[i] if is_test else ref[i]))
    X, y = np.array(rows), np.array(y)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    df = len(y) - np.linalg.matrix_rank(X)
    mse = float(resid @ resid) / df
    cov = mse * np.linalg.pinv(X.T @ X)
    return float(beta[-1]), float(math.sqrt(cov[-1, -1])), df, mse


# ── crossover: three independent routes agree ──────────────────────────────────────────────────
@pytest.mark.parametrize("n_tr, n_rt, seed", [(6, 6, 1), (7, 5, 2), (13, 11, 3), (4, 20, 4)])
def test_matches_the_closed_form_solution_balanced_and_unbalanced(n_tr, n_rt, seed):
    test, ref, tf = _make(n_tr, n_rt, seed=seed)
    got, want = crossover_2x2(test, ref, tf), _closed_form(test, ref, tf)
    assert got["df"] == want["df"] == n_tr + n_rt - 2
    for key, ref_key in (("estimate", "est"), ("se", "se"), ("mse", "mse"), ("ci_lower_raw", "lo"),
                         ("ci_upper_raw", "hi"), ("gmr_raw", "gmr")):
        assert math.isclose(got[key], want[ref_key], rel_tol=1e-10), key


@pytest.mark.parametrize("n_tr, n_rt, seed", [(6, 6, 5), (9, 4, 6), (12, 12, 7)])
def test_matches_a_generic_dummy_variable_least_squares_fit(n_tr, n_rt, seed):
    test, ref, tf = _make(n_tr, n_rt, seed=seed, per=0.4)
    got = crossover_2x2(test, ref, tf)
    est, se, df, mse = _dummy_ols(test, ref, tf)
    assert got["df"] == df
    assert math.isclose(got["estimate"], est, rel_tol=1e-9, abs_tol=1e-12)
    assert math.isclose(got["se"], se, rel_tol=1e-9)
    assert math.isclose(got["mse"], mse, rel_tol=1e-9)


def test_it_is_not_the_paired_t_test_when_sequences_are_unbalanced_or_a_period_effect_exists():
    test, ref, tf = _make(8, 4, seed=8, per=0.8)
    got = crossover_2x2(test, ref, tf)
    diff = np.log(test) - np.log(ref)
    paired = stats.ttest_rel(np.log(test), np.log(ref)).confidence_interval(0.90)
    assert got["df"] == len(diff) - 2 and got["df"] != len(diff) - 1
    assert not math.isclose(got["ci_lower_raw"], 100 * math.exp(paired.low), rel_tol=1e-4)
    # the period effect is removed from the residual, so the model SE is smaller than the paired SE here
    assert got["se"] < diff.std(ddof=1) / math.sqrt(len(diff))


def test_swapping_the_roles_of_test_and_reference_negates_the_estimate_and_keeps_the_se():
    test, ref, tf = _make(7, 7, seed=9)
    flipped = crossover_2x2(ref, test, ~tf)           # swap the roles of both products and both sequences
    base = crossover_2x2(test, ref, tf)
    assert math.isclose(flipped["estimate"], -base["estimate"], rel_tol=1e-10)
    assert math.isclose(flipped["se"], base["se"], rel_tol=1e-10)


def test_alpha_changes_the_interval_and_the_90_percent_interval_is_narrower_than_95():
    test, ref, tf = _make(8, 8, seed=10)
    ci90 = crossover_2x2(test, ref, tf, alpha=0.10)
    ci95 = crossover_2x2(test, ref, tf, alpha=0.05)
    assert ci95["ci_lower_raw"] < ci90["ci_lower_raw"] < ci90["ci_upper_raw"] < ci95["ci_upper_raw"]


# ── ANOVA table (ICH M13A 2.2.3.2: tests of sequence, subject within sequence, period and formulation) ──
def test_the_anova_table_has_the_four_effects_and_the_residual_with_the_right_df():
    test, ref, tf = _make(6, 6, seed=11)
    rows = {r["source"]: r for r in crossover_2x2(test, ref, tf)["anova"]}
    assert list(rows) == ["sequence", "subject(sequence)", "period", "formulation", "residual"]
    assert [rows[k]["df"] for k in rows] == [1, 10, 1, 1, 10]
    assert rows["sequence"]["error_term"] == "subject(sequence)"
    assert rows["period"]["error_term"] == rows["formulation"]["error_term"] == "residual"


def test_anova_sums_of_squares_add_up_to_the_total_for_a_balanced_design():
    test, ref, tf = _make(6, 6, seed=12)
    rows = crossover_2x2(test, ref, tf)["anova"]
    y = np.concatenate([np.log(test), np.log(ref)])
    total = float(((y - y.mean()) ** 2).sum())
    assert math.isclose(sum(r["ss"] for r in rows), total, rel_tol=1e-9)


def test_the_formulation_f_test_is_the_square_of_the_t_statistic_with_matching_p_value():
    test, ref, tf = _make(9, 7, seed=13)
    out = crossover_2x2(test, ref, tf)
    form = next(r for r in out["anova"] if r["source"] == "formulation")
    assert math.isclose(form["f"], (out["estimate"] / out["se"]) ** 2, rel_tol=1e-8)
    assert math.isclose(form["p"], 2 * stats.t.sf(abs(out["estimate"] / out["se"]), out["df"]), rel_tol=1e-8)


def test_the_sequence_effect_is_tested_against_subject_within_sequence():
    test, ref, tf = _make(6, 6, seed=14)
    rows = {r["source"]: r for r in crossover_2x2(test, ref, tf)["anova"]}
    assert math.isclose(rows["sequence"]["f"], rows["sequence"]["ms"] / rows["subject(sequence)"]["ms"], rel_tol=1e-10)
    assert math.isclose(rows["sequence"]["p"], stats.f.sf(rows["sequence"]["f"], 1, 10), rel_tol=1e-8)


def test_means_and_intra_subject_cv_are_reported():
    test, ref, tf = _make(6, 6, seed=15)
    out = crossover_2x2(test, ref, tf)
    assert math.isclose(out["geo_mean_test"], math.exp(np.log(test).mean()), rel_tol=1e-10)
    assert math.isclose(out["geo_mean_ref"], math.exp(np.log(ref).mean()), rel_tol=1e-10)
    assert math.isclose(out["arith_mean_test"], test.mean(), rel_tol=1e-10)
    assert math.isclose(out["cv_intra_pct"], 100 * math.sqrt(math.exp(out["mse"]) - 1), abs_tol=5e-5)   # shown to 4 dp
    assert out["n_per_sequence"] == {"TR": 6, "RT": 6}


# ── what it refuses and when it cannot estimate ────────────────────────────────────────────────
def test_one_sequence_only_is_refused_because_period_and_formulation_cannot_be_separated():
    test, ref, _ = _make(6, 0, seed=16)
    with pytest.raises(ValueError, match="both sequences"):
        crossover_2x2(test, ref, np.array([True] * 6))


def test_too_few_subjects_gives_a_result_that_says_not_estimable():
    out = crossover_2x2(np.array([100.0, 90.0]), np.array([95.0, 99.0]), np.array([True, False]))
    assert out["estimable"] is False and out["ci_lower_raw"] is None and out["df"] == 0
    three = crossover_2x2(np.array([100.0, 90.0, 80.0]), np.array([95.0, 99.0, 70.0]), np.array([True, False, False]))
    assert three["estimable"] is True and three["df"] == 1


@pytest.mark.parametrize("bad", [
    dict(test=[1.0, 2.0], ref=[1.0], tf=[True, False]),
    dict(test=[1.0, -2.0, 3.0], ref=[1.0, 2.0, 3.0], tf=[True, False, True]),
    dict(test=[1.0, float("nan"), 3.0], ref=[1.0, 2.0, 3.0], tf=[True, False, True]),
    dict(test=[1.0, 2.0, 3.0], ref=[1.0, 2.0, 3.0], tf=[True, False]),
    dict(test=[], ref=[], tf=[]),
])
def test_malformed_input_is_a_value_error(bad):
    with pytest.raises(ValueError):
        crossover_2x2(np.array(bad["test"]), np.array(bad["ref"]), np.array(bad["tf"]))


# ── parallel ───────────────────────────────────────────────────────────────────────────────────
def _two_groups(seed=20, n1=14, n2=11):
    rng = np.random.default_rng(seed)
    return np.exp(rng.normal(4.0, 0.30, n1)), np.exp(rng.normal(4.05, 0.45, n2))


def test_parallel_pooled_matches_scipy_equal_variance_t():
    a, b = _two_groups()
    got = parallel_two_group(a, b)
    sc = stats.ttest_ind(np.log(a), np.log(b), equal_var=True).confidence_interval(0.90)
    assert got["df"] == len(a) + len(b) - 2 and got["variance"] == "pooled"
    assert math.isclose(got["ci_lower_raw"], 100 * math.exp(sc.low), rel_tol=1e-10)
    assert math.isclose(got["ci_upper_raw"], 100 * math.exp(sc.high), rel_tol=1e-10)


def test_parallel_welch_matches_scipy_unequal_variance_t_with_satterthwaite_df():
    a, b = _two_groups()
    got = parallel_two_group(a, b, variance="welch")
    res = stats.ttest_ind(np.log(a), np.log(b), equal_var=False)
    sc = res.confidence_interval(0.90)
    assert got["variance"] == "welch" and math.isclose(got["df"], res.df, rel_tol=1e-9)
    assert math.isclose(got["ci_lower_raw"], 100 * math.exp(sc.low), rel_tol=1e-10)
    assert math.isclose(got["ci_upper_raw"], 100 * math.exp(sc.high), rel_tol=1e-10)


def test_parallel_rejects_an_unknown_variance_option_and_tiny_groups():
    a, b = _two_groups()
    with pytest.raises(ValueError, match="variance"):
        parallel_two_group(a, b, variance="equal-ish")
    assert parallel_two_group(a[:1], b)["estimable"] is False
