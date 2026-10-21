"""Behavioural pins for conventions-catalog clauses that no existing test covered directly (clause-by-clause review of
the catalog). Each test pins current behaviour; it does not by itself prove the convention is the right one."""
from __future__ import annotations

import math

import numpy as np
import pytest

from app.compute.bioequivalence import be_one_parameter
from app.compute.dosing import parse_blq_flag
from app.compute.nca import Profile, _auc_intervals, nca_subject


# nca.linear_up_log_down
def test_a_rising_segment_is_integrated_linearly():
    assert _auc_intervals(np.array([0.0, 1.0]), np.array([4.0, 8.0]))[0] == pytest.approx(6.0)


def test_a_flat_segment_is_integrated_linearly():
    assert _auc_intervals(np.array([0.0, 1.0, 2.0]), np.array([4.0, 4.0, 4.0]))[0] == pytest.approx(8.0)


def test_a_declining_positive_segment_is_integrated_logarithmically():
    assert _auc_intervals(np.array([0.0, 1.0]), np.array([8.0, 4.0]))[0] == pytest.approx(4.0 / math.log(2.0))


def test_a_decline_to_zero_is_integrated_linearly():
    assert _auc_intervals(np.array([0.0, 1.0]), np.array([4.0, 0.0]))[0] == pytest.approx(2.0)


# nca.measured_zero: an unflagged 0 between positive values is kept and integrated, not dropped
def test_a_measured_zero_is_kept_in_the_profile():
    t = np.array([0.0, 1.0, 2.0, 4.0, 8.0, 12.0])
    c = np.array([0.0, 5.0, 8.0, 0.0, 2.0, 1.0])
    kept = nca_subject(Profile("Z", t, c, 100.0))
    dropped = nca_subject(Profile("Z", np.delete(t, 3), np.delete(c, 3), 100.0))
    assert kept["AUC_last"] < dropped["AUC_last"]


# nca.blq_flag_codes
@pytest.mark.parametrize("value,flagged", [
    (True, True), (False, False), (np.bool_(True), True), (1, True), (0, False), (1.0, True), (0.0, False),
    ("1.0", True), ("y", True), ("No", False), (None, False), ("", False), (".", False), ("NA", False),
    ("nan", False), (float("nan"), False),
])
def test_flag_values_parse_as_documented(value, flagged):
    assert parse_blq_flag(value) is flagged


@pytest.mark.parametrize("value", [2, -1, 0.5, "maybe", "BLQ"])
def test_other_flag_values_raise(value):
    with pytest.raises(ValueError):
        parse_blq_flag(value)


# be.average_tost: dropping applies to the parallel design too, and only the values used are counted
def test_parallel_be_drops_nonpositive_and_missing_values():
    r = be_one_parameter([100.0, 0.0, None, 110.0, -3.0, 90.0], [100.0, 105.0, None, 95.0], paired=False)
    assert (r["n_test"], r["n_ref"]) == (3, 3)
