"""Cross-check of the interim BE estimator against an independent implementation: scipy.stats.ttest_rel (paired) and
ttest_ind with equal variances (parallel), whose confidence_interval() is computed by scipy's own t-test code, not
by app.compute.bioequivalence. The interval is taken on the log scale at 90% and back-transformed.

This pins the interim estimator only. It is not the cross-check of the final (period/sequence) model, which still
needs an external reference such as R or SAS PROC GLM."""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from app.compute.be_contract import assess_average_be


def _data(seed, n, sd_between=0.3, sd_within=0.15, ratio=0.97):
    rng = np.random.default_rng(seed)
    subj = rng.normal(0.0, sd_between, n)
    ref = np.exp(np.log(100.0) + subj + rng.normal(0.0, sd_within, n))
    test = np.exp(np.log(100.0 * ratio) + subj + rng.normal(0.0, sd_within, n))
    return test, ref


def _recs(test, ref, *, parallel=False):
    out = []
    for i, (t, r) in enumerate(zip(test, ref)):
        out += [{"subject": f"T{i}" if parallel else i, "treatment": "T", "params": {"Cmax": float(t)}},
                {"subject": f"R{i}" if parallel else i, "treatment": "R", "params": {"Cmax": float(r)}}]
    return out


@pytest.mark.parametrize("seed, n", [(1, 12), (2, 18), (3, 24), (4, 36), (5, 60)])
def test_the_legacy_paired_estimator_matches_scipy_paired_t(seed, n):
    """be_one_parameter(paired=True) is no longer the contract's crossover estimator (that is the 2x2 model in
    app.compute.be_model, checked in test_be_model.py); it is still used by PharmacometricsBench."""
    from app.compute.bioequivalence import be_one_parameter

    test, ref = _data(seed, n)
    sc = stats.ttest_rel(np.log(test), np.log(ref)).confidence_interval(0.90)
    p = be_one_parameter(test.tolist(), ref.tolist(), paired=True)
    assert math.isclose(p["ci_lower_raw"], 100 * math.exp(sc.low), rel_tol=1e-12)
    assert math.isclose(p["ci_upper_raw"], 100 * math.exp(sc.high), rel_tol=1e-12)
    assert math.isclose(p["gmr_pct"], round(100 * math.exp(np.mean(np.log(test / ref))), 4), rel_tol=1e-12)


@pytest.mark.parametrize("seed, n", [(11, 12), (12, 20), (13, 40)])
def test_parallel_matches_scipy_pooled_two_sample_t(seed, n):
    test, ref = _data(seed, n)
    sc = stats.ttest_ind(np.log(test), np.log(ref), equal_var=True).confidence_interval(0.90)
    p = assess_average_be(_recs(test, ref, parallel=True), design="parallel", profile="ema_2010", test="T",
                          reference="R")["parameters"]["Cmax"]
    assert math.isclose(p["ci_lower_raw"], 100 * math.exp(sc.low), rel_tol=1e-12)
    assert math.isclose(p["ci_upper_raw"], 100 * math.exp(sc.high), rel_tol=1e-12)
