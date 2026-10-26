"""Covariance step: a noisy finite-difference Hessian must not hide the SEs.

Seed-2024 covariate cohort (test_nlme): the OFV is clearly curved in every
direction (dOFV +5.7 at a quarter step along the "weak" eigenvector), but the
central-difference Hessian at the default step came out with a slightly negative
eigenvalue (-9 vs +1e4 max), so the floor kicked in, cond ~1e9, no RSEs. The same
Hessian at step 0.1 is positive definite with cond ~280. The eigenvalue spread
across steps is inner-solver noise, not non-identifiability.
"""
from __future__ import annotations

import math

import numpy as np

from app.compute import nlme
from app.compute.nlme import _COV_STEP, _parameter_uncertainty, population_fit
from tests import test_nlme as tn

_GOOD = np.diag([40.0, 100.0, 1000.0, 2000.0])
_NOISY = np.diag([-9.0, 100.0, 1000.0, 2000.0])      # one noise-negative eigenvalue


def _spec():
    pop = tn._make_population(n=3)
    prepared = nlme._prepare_subjects(pop)
    return nlme._PopSpec(nlme.get_model("oral_1cmt"), ["CL"], "proportional", [])


def test_retries_other_steps_when_first_hessian_is_not_positive_definite(monkeypatch) -> None:
    spec = _spec()
    n = spec.n_theta + spec.n_omega + 1
    steps = []

    def fake(fun, x, step=0.0):
        steps.append(step)
        return np.diag(np.where(step == _COV_STEP, -9.0, 500.0) * np.ones(n))

    monkeypatch.setattr(nlme, "_numeric_hessian", fake)
    out = _parameter_uncertainty(spec, lambda x: 0.0, np.zeros(n))
    assert steps[0] == _COV_STEP and len(steps) >= 2
    assert out["theta_rse_pct"]["CL"] is not None
    assert out["cov_note"] == ""


def test_does_not_retry_when_first_hessian_is_clean(monkeypatch) -> None:
    spec = _spec()
    n = spec.n_theta + spec.n_omega + 1
    steps = []
    monkeypatch.setattr(nlme, "_numeric_hessian",
                        lambda f, x, step=0.0: steps.append(step) or 500.0 * np.eye(n))
    _parameter_uncertainty(spec, lambda x: 0.0, np.zeros(n))
    assert steps == [_COV_STEP]


def test_keeps_honest_note_when_every_step_is_bad(monkeypatch) -> None:
    spec = _spec()
    n = spec.n_theta + spec.n_omega + 1
    monkeypatch.setattr(nlme, "_numeric_hessian",
                        lambda f, x, step=0.0: np.diag([-9.0] + [500.0] * (n - 1)))
    out = _parameter_uncertainty(spec, lambda x: 0.0, np.zeros(n))
    assert out["cov_note"] != ""


def test_seed_2024_covariate_cohort_now_gets_standard_errors() -> None:
    pop = tn._make_cov_population(seed=2024, n=40, beta_wt=0.75)
    r = population_fit(
        tn.MODEL_KEY, pop, method="focei", max_iter=30, compute_uncertainty=True,
        covariate_model=[{"param": "CL", "covariate": "WT",
                          "kind": "power", "center": 70.0}])
    rse = r["covariate_effects"][0]["rse_pct"]
    assert isinstance(rse, float) and math.isfinite(rse) and rse > 0.0, r["cov_note"]
    assert r["condition_number"] < 1e3
