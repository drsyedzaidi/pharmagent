"""Reference validation on the Theophylline dataset.

Drives the SHIPPING compute path (dataset -> _build_subjects -> NCA + FOCE-I +
SAEM) and checks the estimates against published 1-compartment literature
consensus (tests/reference/references.py) plus cross-method internal
consistency. This is the suite that answers "does PharmAgent reproduce values a
pharmacometrician would expect?" — distinct from test_nlme.py, which checks
self-recovery of a simulated truth.

If ``THEOPHYLLINE['tool_reference']`` is populated with the user's own
NONMEM/Monolix/nlmixr2 estimates, an additional exact-concordance test runs.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from app.compute.nlme import population_fit
from app.tools.pkmodel_tools import _build_subjects, _roles
from tests.reference.references import THEOPHYLLINE

_DATA = Path(__file__).resolve().parents[2] / "sample_data" / THEOPHYLLINE["dataset"]
_LIT = THEOPHYLLINE["literature"]


def _gm(xs: list[float]) -> float:
    return float(np.exp(np.mean(np.log(xs))))


def _in_band(value: float, name: str) -> bool:
    lo, hi = _LIT[name][1]
    return lo <= value <= hi


@pytest.fixture(scope="module")
def subjects() -> list[dict[str, Any]]:
    df = pd.read_csv(_DATA)
    roles = _roles(df, type("S", (), {"dataset_metadata": None})())
    subs, _multi, _pd = _build_subjects(df, roles)
    assert len(subs) == 12, "expected the 12-subject Theophylline cohort"
    return subs


@pytest.fixture(scope="module")
def nca_clf(subjects) -> dict[str, float]:
    """NCA-style CL/F = Dose/AUC_inf (lin-trap + terminal extrapolation) and
    terminal t1/2, summarized across subjects."""
    clf, thalf = [], []
    for s in subjects:
        t = np.asarray(s["obs_t"], float)
        c = np.asarray(s["obs_c"], float)
        dose = float(s["doses"][-1]["amt"])
        auc = float(np.trapezoid(c, t))
        slope = np.polyfit(t[-3:], np.log(c[-3:]), 1)[0]
        ke = -slope
        if ke > 0:
            clf.append(dose / (auc + c[-1] / ke))
            thalf.append(math.log(2) / ke)
    return {"clf_geomean": _gm(clf), "t_half_median": float(np.median(thalf))}


@pytest.fixture(scope="module")
def focei(subjects) -> dict[str, Any]:
    return population_fit("oral_1cmt", subjects, method="focei", max_iter=40,
                          compute_uncertainty=False)


@pytest.fixture(scope="module")
def saem(subjects) -> dict[str, Any]:
    return population_fit("oral_1cmt", subjects, method="saem", max_iter=120,
                          seed=20250614, compute_uncertainty=False)


# ── NCA vs published ──────────────────────────────────────────────────────────

def test_nca_clf_matches_literature(nca_clf):
    assert _in_band(nca_clf["clf_geomean"], "CL"), \
        f"NCA CL/F {nca_clf['clf_geomean']:.3f} outside {_LIT['CL'][1]}"


def test_nca_thalf_matches_literature(nca_clf):
    assert _in_band(nca_clf["t_half_median"], "t_half"), \
        f"NCA t1/2 {nca_clf['t_half_median']:.2f} outside {_LIT['t_half'][1]}"


# ── FOCE-I vs published ───────────────────────────────────────────────────────

def test_focei_structural_params_match_literature(focei):
    th = focei["theta"]
    assert _in_band(th["CL"], "CL"), f"CL {th['CL']:.3f} outside {_LIT['CL'][1]}"
    assert _in_band(th["V"], "V"), f"V {th['V']:.3f} outside {_LIT['V'][1]}"
    assert _in_band(th["KA"], "KA"), f"KA {th['KA']:.3f} outside {_LIT['KA'][1]}"


def test_focei_variability_matches_literature(focei):
    assert _in_band(focei["omega_cv_pct"]["CL"], "iiv_cl_pct")
    assert _in_band(float(focei["sigma"]["prop"]), "sigma_prop")
    assert focei["converged"] is True


# ── SAEM vs published ─────────────────────────────────────────────────────────

def test_saem_structural_params_match_literature(saem):
    th = saem["theta"]
    assert _in_band(th["CL"], "CL"), f"CL {th['CL']:.3f} outside {_LIT['CL'][1]}"
    assert _in_band(th["V"], "V"), f"V {th['V']:.3f} outside {_LIT['V'][1]}"
    assert _in_band(th["KA"], "KA"), f"KA {th['KA']:.3f} outside {_LIT['KA'][1]}"


# ── cross-method internal consistency ─────────────────────────────────────────

def test_clearance_agrees_across_methods(nca_clf, focei, saem):
    """NCA CL/F, FOCE-I CL and SAEM CL should agree within 20% — three
    independent routes to the same clearance is strong internal evidence."""
    cl_nca = nca_clf["clf_geomean"]
    cl_foce = focei["theta"]["CL"]
    cl_saem = saem["theta"]["CL"]
    assert abs(cl_foce - cl_nca) / cl_nca < 0.20, (cl_foce, cl_nca)
    assert abs(cl_saem - cl_nca) / cl_nca < 0.20, (cl_saem, cl_nca)
    assert abs(cl_foce - cl_saem) / cl_saem < 0.20, (cl_foce, cl_saem)


def test_volume_agrees_across_methods(focei, saem):
    v_foce, v_saem = focei["theta"]["V"], saem["theta"]["V"]
    assert abs(v_foce - v_saem) / v_saem < 0.20, (v_foce, v_saem)


# ── CWRES on the SAME converged fit (no new fit — reuses the `focei` fixture) ─

def test_cwres_from_focei_theophylline_is_plausible(subjects, focei):
    """CWRES computed post-hoc from the real 12-subject FOCE-I fit must be a
    well-behaved standardized residual: finite, roughly centered, roughly
    unit-scaled, and every subject's stored EBE reused (no re-optimization —
    `posthoc_residuals` must never re-run the inner solver when eta is
    already on the fitted result)."""
    from app.compute.nlme import cv_pct_to_omega2, posthoc_residuals

    omega2 = {p: cv_pct_to_omega2(cv) for p, cv in focei["omega_cv_pct"].items()}
    sigma = focei["sigma"]
    etas = {r["subject"]: r["eta"] for r in focei["individual"]}

    out = posthoc_residuals(
        "oral_1cmt", subjects, theta=focei["theta"], omega2=omega2,
        sigma_prop=float(sigma.get("prop") or 0.0), sigma_add=float(sigma.get("add") or 0.0),
        iiv_params=focei["iiv_params"], error_model=focei["error_model"],
        covariate_effects=focei.get("covariate_effects"), etas=etas)

    assert out["summary"]["n"] > 0
    assert out["summary"]["n_etas_resolved"] == 0  # every subject's EBE was reused
    assert out["summary"]["n_etas_reused"] == out["summary"]["n_subjects_used"]
    cwres = out["cwres"]
    assert all(math.isfinite(v) for v in cwres)
    assert abs(out["summary"]["cwres_mean"]) < 1.0
    assert 0.3 < out["summary"]["cwres_sd"] < 3.0


# ── optional: exact concordance with the user's own tool run ──────────────────

@pytest.mark.skipif(THEOPHYLLINE["tool_reference"] is None,
                    reason="no tool_reference set — populate references.py to enable")
def test_matches_user_tool_reference(focei):
    ref = THEOPHYLLINE["tool_reference"]
    rel_tol = float(ref.get("rel_tol", 0.15))
    for p in ("CL", "V", "KA"):
        if p not in ref:
            continue
        got, want = focei["theta"][p], float(ref[p])
        assert abs(got - want) / want < rel_tol, \
            f"{p}: PharmAgent {got:.3f} vs {ref['tool']} {want:.3f} (>{rel_tol:.0%})"
