"""Tests for app.compute.compartmental — validated against closed-form values."""
from __future__ import annotations

import math

import numpy as np
import pytest

from app.compute.compartmental import (
    conc_1cmt_oral,
    conc_2cmt_oral,
    fit_compartmental,
    fit_one_subject,
)

# Known truth used to simulate a 1-cmt oral profile.
KA_TRUE = 1.0
CL_TRUE = 5.0
V_TRUE = 50.0
DOSE = 100.0
TIMES = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 24.0])


def _closed_form_1cmt(t, dose, ka, CL, V):
    """Independent reference implementation of the 1-cmt oral solution."""
    ke = CL / V
    return (dose * ka) / (V * (ka - ke)) * (math.exp(-ke * t) - math.exp(-ka * t))


def test_conc_1cmt_oral_at_t0_is_zero():
    # C(0) = factor * (exp(0) - exp(0)) = 0 exactly.
    c0 = conc_1cmt_oral(0.0, DOSE, KA_TRUE, CL_TRUE, V_TRUE)
    assert float(c0) == pytest.approx(0.0, abs=1e-12)


def test_conc_1cmt_oral_matches_closed_form():
    # Spot-check a few times against the hand reference.
    for t in (0.5, 2.0, 8.0):
        got = float(conc_1cmt_oral(t, DOSE, KA_TRUE, CL_TRUE, V_TRUE))
        want = _closed_form_1cmt(t, DOSE, KA_TRUE, CL_TRUE, V_TRUE)
        assert got == pytest.approx(want, rel=1e-9)


def test_conc_1cmt_oral_is_vectorized():
    out = conc_1cmt_oral(TIMES, DOSE, KA_TRUE, CL_TRUE, V_TRUE)
    assert isinstance(out, np.ndarray)
    assert out.shape == TIMES.shape


def test_conc_2cmt_oral_at_t0_is_zero():
    # C(0) = A + B - (A+B) = 0 exactly, regardless of params.
    c0 = conc_2cmt_oral(0.0, DOSE, 1.0, 5.0, 30.0, 4.0, 20.0)
    assert float(c0) == pytest.approx(0.0, abs=1e-12)


def test_fit_1cmt_recovers_known_params():
    conc = conc_1cmt_oral(TIMES, DOSE, KA_TRUE, CL_TRUE, V_TRUE)
    fit = fit_one_subject(TIMES, conc, DOSE, model="1cmt")

    assert fit["converged"] is True
    assert fit["model"] == "1cmt"
    assert fit["n_obs"] == len(TIMES)
    # Near-perfect fit to noiseless data.
    assert fit["r_squared"] > 0.999

    assert fit["params"]["CL"] == pytest.approx(CL_TRUE, rel=0.05)
    assert fit["params"]["V"] == pytest.approx(V_TRUE, rel=0.10)
    assert fit["params"]["ka"] == pytest.approx(KA_TRUE, rel=0.10)


def test_fit_one_subject_too_few_points_returns_false_not_crash():
    # 1cmt needs n_params+1 = 4 positive obs; give only 3.
    time = np.array([0.5, 1.0, 2.0])
    conc = np.array([1.0, 1.5, 1.2])
    fit = fit_one_subject(time, conc, DOSE, model="1cmt")

    assert fit["converged"] is False
    assert fit["params"] == {}
    assert fit["n_obs"] == 3


def test_fit_compartmental_groups_and_selects():
    conc = conc_1cmt_oral(TIMES, DOSE, KA_TRUE, CL_TRUE, V_TRUE)
    records = []
    for sid in ("S1", "S2"):
        for t, c in zip(TIMES, conc):
            records.append({"ID": sid, "TIME": float(t), "DV": float(c)})

    out = fit_compartmental(
        records,
        id_col="ID",
        time_col="TIME",
        dv_col="DV",
        dose_by_subject={"S1": DOSE, "S2": DOSE},
        models=("1cmt", "2cmt"),
    )

    assert out["n_subjects"] == 2
    assert len(out["individual_fits"]) == 2
    # Sorted by str(subject).
    subjects = [f["subject"] for f in out["individual_fits"]]
    assert subjects == ["S1", "S2"]

    for f in out["individual_fits"]:
        assert f["converged"] is True
        assert "all_models" in f
        assert len(f["all_models"]) == 2
        # best model recorded in selection map
        assert out["model_selection"][f["subject"]] == f["model"]
        # selected fit has the lowest AIC among its converged attempts
        aics = [m["aic"] for m in f["all_models"]
                if m["converged"] and m["aic"] is not None]
        assert f["aic"] == pytest.approx(min(aics))
