"""SAEM default budget and stopping rule (follow-up to the honest-converged flag).

The old rule demanded three consecutive single-iteration steps under 1e-3. The
theta M-step is not averaged, so each iteration jitters by ~2-4e-3 no matter how
long the run: the rule fired only by luck (30-subject cohort, seed 7: converged
at max_iter=500, not at 1000). The rule now compares window means.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.compute import nlme
from app.compute.nlme import (
    _SAEM_DEFAULT_ITER,
    _saem_stable,
    population_fit,
    saem_fit,
)
from app.tools.pkmodel_tools import _nlme_stop_note
from tests import test_nlme as tn


def _hist(n: int, noise: float, drift: float = 0.0, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    base = np.array([5.0, 50.0, 1.0, 0.09, 0.04, 0.1])
    return [base * (1.0 + drift * k + noise * rng.normal(size=base.size))
            for k in range(n)]


def test_jittery_but_stationary_trajectory_counts_as_stable() -> None:
    ok, drift = _saem_stable(_hist(200, noise=3e-3), window=50)
    assert ok, drift


def test_drifting_trajectory_is_not_stable() -> None:
    ok, drift = _saem_stable(_hist(200, noise=1e-3, drift=1e-4), window=50)
    assert not ok and drift > 3e-3


def test_too_short_history_is_not_stable() -> None:
    ok, drift = _saem_stable(_hist(99, noise=0.0), window=50)
    assert not ok and drift is None


def test_saem_default_budget_is_applied_only_when_max_iter_omitted(monkeypatch) -> None:
    seen = []
    monkeypatch.setattr(nlme, "saem_fit",
                        lambda *a, **k: seen.append(k["max_iter"]) or {})
    population_fit("oral_1cmt", [], method="saem")
    population_fit("oral_1cmt", [], method="saem", max_iter=80)
    assert seen == [_SAEM_DEFAULT_ITER, 80]
    assert _SAEM_DEFAULT_ITER >= 1000


def test_focei_default_budget_unchanged(monkeypatch) -> None:
    seen = []
    monkeypatch.setattr(nlme, "focei_fit",
                        lambda *a, **k: seen.append(k["max_iter"]) or {})
    population_fit("oral_1cmt", [], method="focei")
    assert seen == [200]


def test_short_budget_reports_why_saem_did_not_converge() -> None:
    pop = tn._make_population(n=8)
    r = saem_fit("oral_1cmt", pop, iiv_params=["CL", "V"],
                 error_model="proportional", max_iter=30, compute_uncertainty=False)
    assert r["converged"] is False
    assert r["stop"]["reason"] == "budget_too_short"
    assert r["stop"]["iterations"] == 30
    assert "max_iter" in _nlme_stop_note(r)


def test_met_rule_is_reported(monkeypatch) -> None:
    pop = tn._make_population(n=8)
    monkeypatch.setattr(nlme, "_saem_stable", lambda h, window, tol=nlme._SAEM_TOL: (True, 1e-4))
    r = saem_fit("oral_1cmt", pop, iiv_params=["CL", "V"],
                 error_model="proportional", max_iter=300, compute_uncertainty=False)
    assert r["stop"]["reason"] == "stability_rule_met"
    assert r["iterations"] < 300
    assert _nlme_stop_note(r) == ""


@pytest.mark.parametrize("stop,needle", [
    ({"reason": "budget_exhausted", "iterations": 500, "drift": 4e-3}, "raise max_iter"),
    ({"reason": "not_stabilising", "iterations": 1000, "drift": 0.2}, "more iterations"),
    ({"reason": "non_finite_ofv", "iterations": 100, "drift": None}, "objective"),
])
def test_stop_note_text(stop, needle) -> None:
    note = _nlme_stop_note({"method": "SAEM", "converged": False, "stop": stop})
    assert needle in note


# ── negative controls: the rule must NOT call these converged ────────────────
# (a rule that merely accepts any finite OFV would pass all of them)

def test_oscillating_trajectory_is_not_stable() -> None:
    """Symmetric 5% oscillation: window means cancel, spread must not."""
    base = np.array([5.0, 50.0, 1.0, 0.09, 0.04, 0.1])
    hist = [base * (1.0 + 0.05 * np.sin(2 * np.pi * k / 10)) for k in range(300)]
    ok, drift = _saem_stable(hist, window=50)
    assert not ok and drift >= 3e-3


def test_slow_monotone_drift_of_1e3_per_iteration_is_not_stable() -> None:
    ok, drift = _saem_stable(_hist(300, noise=1e-3, drift=1e-3), window=50)
    assert not ok and drift > 3e-3


def test_stationary_noise_at_real_saem_scale_is_stable() -> None:
    """Positive control at the measured per-iteration jitter (up to ~1% sd)."""
    ok, _ = _saem_stable(_hist(200, noise=1e-2), window=50)
    assert ok
