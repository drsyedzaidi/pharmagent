"""Pins for the lambda_z terminal-window convention (catalog entry `nca.lambda_z_window`): the numerical decline
threshold from issue #12, and deterministic tie behaviour between candidate windows."""
from __future__ import annotations

import numpy as np
import pytest

from app.compute.nca import _MIN_LAMBDA_Z, _best_lambda_z

T = np.array([0.0, 1, 2, 3, 4, 5, 6])


def _expo(rate: float, c0: float = 100.0) -> np.ndarray:
    return c0 * np.exp(-rate * T)


def test_the_decline_threshold_is_one_in_a_billion_per_time_unit():
    assert _MIN_LAMBDA_Z == 1e-9


def test_a_flat_terminal_phase_gives_no_lambda_z():
    assert _best_lambda_z(T, np.full(7, 50.0), tmax=0.0) == {}


def test_a_decline_below_the_threshold_gives_no_lambda_z():
    assert _best_lambda_z(T, _expo(5e-10), tmax=0.0) == {}


def test_a_decline_above_the_threshold_is_accepted():
    fit = _best_lambda_z(T, _expo(2e-9), tmax=0.0)
    assert fit and fit["lambda_z"] == pytest.approx(2e-9, rel=1e-3)


def test_equal_fits_prefer_the_longest_window():
    """A perfectly log-linear tail makes every window fit equally well; a later, shorter window replaces the best only
    when strictly better, so the longest window (starting right after Tmax) is chosen."""
    fit = _best_lambda_z(T, _expo(0.3), tmax=0.0)
    assert fit["lambda_z_n_points"] == 6 and fit["lambda_z_intervals"] == [1.0, 6.0]
    assert fit["lambda_z"] == pytest.approx(0.3, rel=1e-9)


def test_a_clearly_better_shorter_window_wins():
    conc = np.array([100.0, 40.0, 30.0, 20.0, 10.0, 5.0, 2.5])      # exact halving every 1 h from t=3 onward
    fit = _best_lambda_z(T, conc, tmax=0.0)
    assert fit["lambda_z_intervals"][0] == 3.0 and fit["lambda_z_n_points"] == 4
    assert fit["lambda_z"] == pytest.approx(np.log(2), rel=1e-9)


def test_the_chosen_window_always_ends_at_the_last_positive_point():
    assert _best_lambda_z(T, _expo(0.3), tmax=0.0)["lambda_z_intervals"][1] == 6.0


def test_fewer_than_three_points_after_tmax_gives_no_lambda_z():
    assert _best_lambda_z(T[:3], _expo(0.3)[:3], tmax=0.0) == {}


def test_every_window_without_a_positive_adjusted_r2_gives_no_lambda_z():
    """Every window declines here, but none fits: adjusted R-squared is -0.05 (4 points) and -0.97 (3 points)."""
    assert _best_lambda_z(np.array([0.0, 1, 2, 3, 4]), np.array([100.0, 10, 3, 9, 2.5]), tmax=0.0) == {}


def test_a_barely_positive_adjusted_r2_is_accepted():
    fit = _best_lambda_z(np.array([0.0, 1, 2, 3, 4]), np.array([100.0, 10, 2, 8, 1.5]), tmax=0.0)
    assert fit and 0 < fit["lambda_z_r2_adj"] < 0.01


def test_a_zero_after_tmax_is_left_out_of_the_window():
    conc = _expo(0.3)
    conc[3] = 0.0
    fit = _best_lambda_z(T, conc, tmax=0.0)
    assert fit["lambda_z_n_points"] == 5 and 3.0 not in fit["lambda_z_pts_t"]
    assert fit["lambda_z"] == pytest.approx(0.3, rel=1e-9)
