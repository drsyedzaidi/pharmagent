"""compute.clinpharm — ported from PharmKit's pk.test.ts (same analytic references)."""
from __future__ import annotations

import math

import pytest

from app.compute import clinpharm as cp

close = lambda a, b, tol=1e-6: abs(a - b) < tol  # noqa: E731


def test_ke_and_half_life_are_inverse():
    assert close(cp.ke_from_half_life(6), math.log(2) / 6)
    assert close(cp.half_life_from_ke(cp.ke_from_half_life(6)), 6)


def test_ke_from_two_terminal_points():
    assert close(cp.ke_from_two_points(10, 0, 5, 4), math.log(2) / 4)
    with pytest.raises(ValueError):
        cp.ke_from_two_points(10, 4, 5, 0)


def test_accumulation_ratio_one_half_life_interval_is_2():
    ke = cp.ke_from_half_life(12)
    assert close(cp.accumulation_ratio(ke, 12), 2, 1e-9)


def test_time_to_90pct_steady_state():
    ke = cp.ke_from_half_life(1)
    assert close(cp.time_to_fraction_of_steady_state(ke, 0.9), math.log(10) / math.log(2), 1e-9)
    with pytest.raises(ValueError):
        cp.time_to_fraction_of_steady_state(ke, 1.0)


def test_iv_bolus_c0_and_decay():
    assert close(cp.iv_bolus_conc(100, 50, 0.1, 0), 2)
    assert close(cp.iv_bolus_conc(100, 50, 0.1, 10), 2 * math.exp(-1))


def test_iv_infusion_reaches_plateau():
    dose, t_inf, v, ke = 100, 100, 10, 0.2
    plateau = (dose / t_inf) / (v * ke)
    assert close(cp.iv_infusion_conc(dose, t_inf, v, ke, 100), plateau, 1e-3)


def test_oral_peak_at_analytic_tmax_and_flip_flop_limit():
    ka, ke = 1.2, 0.15
    tmax = cp.oral_tmax(ka, ke)
    peak = cp.oral_one_comp_conc(100, 30, ka, ke, tmax)
    assert peak > cp.oral_one_comp_conc(100, 30, ka, ke, tmax - 0.05)
    assert peak > cp.oral_one_comp_conc(100, 30, ka, ke, tmax + 0.05)
    c = cp.oral_one_comp_conc(100, 30, 0.3, 0.3, 5)
    assert math.isfinite(c) and c > 0


def test_sample_profile_has_n_plus_one_points():
    pts = cp.sample_profile(lambda t: 2 * t, 10, n=5)
    assert len(pts) == 6 and pts[-1] == {"t": 10, "c": 20}


def test_cockcroft_gault_reference_case_and_female_factor():
    assert close(cp.cockcroft_gault(40, 80, 1.0, "male"), 8000 / 72)
    assert close(cp.cockcroft_gault(40, 80, 1.0, "female"), 8000 / 72 * 0.85, 1e-9)


def test_ckd_epi_2021_sensible_at_kappa():
    e = cp.ckd_epi_2021(50, 0.9, "male")
    assert 90 < e < 110


def test_renal_dose_adjustment_scales_by_clearance_ratio():
    assert close(cp.renal_dose_adjustment(100, 60, 120, 1), 50, 1e-9)
    assert close(cp.renal_dose_adjustment(100, 130, 120, 1), 100, 1e-9)
    assert close(cp.renal_dose_adjustment(100, 60, 120, 0.5), 75, 1e-9)   # half renal


def test_mass_molar_round_trip():
    mw = 305.4
    assert close(cp.molar_to_mass(cp.mass_to_molar(12, mw), mw), 12, 1e-9)


def test_allometric_scaling_cl_exponent():
    assert close(cp.allometric_scale(10, 1, 10, 0.75), 10 * 10 ** 0.75, 1e-9)


def test_dose_formulas():
    assert close(cp.loading_dose(2.0, 50.0, 0.5), 200.0)
    assert close(cp.maintenance_dose(1.0, 5.0, 12.0, 0.5), 120.0)
    assert close(cp.clearance_from_auc(100, 1000, 0.8), 0.08)
    assert close(cp.volume_from_dose_c0(100, 2), 50)


def test_be_sample_size_plausible_and_monotone():
    r = cp.be_sample_size(0.25, gmr=0.95)
    assert r["total"] % 2 == 0 and 24 <= r["total"] <= 34       # literature ≈ 28
    assert cp.be_sample_size(0.35, gmr=0.95)["total"] > cp.be_sample_size(0.2, gmr=0.95)["total"]
    assert cp.be_sample_size(0.25, gmr=1.0)["total"] < cp.be_sample_size(0.25, gmr=0.9)["total"]
    with pytest.raises(ValueError):
        cp.be_sample_size(0.25, gmr=1.3)
