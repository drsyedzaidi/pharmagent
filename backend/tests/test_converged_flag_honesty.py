"""converged=True must mean the optimiser actually converged (review F-03).

A finite OFV is not convergence: Powell never returns a point worse than its
start, and SAEM's finite-OFV fallback overrode its own stability test.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from app.compute import compartmental, nlme


def _subjects():
    return [{"subject": i, "doses": [{"time": 0.0, "amt": 100.0}],
             "obs_t": [1.0, 4.0, 8.0, 24.0],
             "obs_c": [9.0 + 0.3 * i, 7.0, 5.0 - 0.1 * i, 2.0],
             "wt": 70.0, "cov": {}} for i in (1, 2, 3, 4)]


def test_saem_with_tiny_iteration_budget_reports_not_converged():
    res = nlme.saem_fit("oral_1cmt", _subjects(), iiv_params=["CL"],
                        error_model="proportional", max_iter=3,
                        compute_uncertainty=False)
    assert res["converged"] is False


def test_focei_with_unsuccessful_minimize_reports_not_converged(monkeypatch):
    real = nlme.minimize

    def fake(fun, x0, *a, **k):
        r = real(fun, x0, *a, **k)
        return SimpleNamespace(x=r.x, success=False, nit=r.nit, fun=r.fun)

    monkeypatch.setattr(nlme, "minimize", fake)
    res = nlme.focei_fit("oral_1cmt", _subjects(), iiv_params=["CL"],
                         error_model="proportional", max_iter=5,
                         compute_uncertainty=False)
    assert res["converged"] is False
    assert "ofv_improved" in res


def test_compartmental_fit_with_unsuccessful_solver_reports_not_converged(monkeypatch):
    real = compartmental.least_squares

    def fake(*a, **k):
        r = real(*a, **k)
        r.success = False
        return r

    monkeypatch.setattr(compartmental, "least_squares", fake)
    t = np.array([0.5, 1, 2, 4, 8, 12, 24.0])
    c = 10 * (np.exp(-0.15 * t) - np.exp(-1.2 * t))
    out = compartmental.fit_one_subject(t, c, 100.0, model="1cmt")
    assert out["converged"] is False


def test_auto_arbitration_prefers_converged_candidate(monkeypatch):
    bad = {"converged": False, "ofv": 100.0}
    good = {"converged": True, "ofv": 200.0}
    monkeypatch.setattr(nlme, "focei_fit", lambda *a, **k: dict(bad))
    monkeypatch.setattr(nlme, "_focei_saem_fit", lambda *a, **k: dict(good))
    out = nlme._auto_fit("oral_1cmt", _subjects(), iiv_params=["CL"],
                         error_model="proportional", max_iter=5, seed=1,
                         compute_uncertainty=False, covariate_model=None)
    assert out["converged"] is True
