"""Tests for app.compute.dosing.time_after_dose (TAD).

TAD is defined as elapsed time since the most recent PRIOR dose administration
— the diagnostic-plot convention (pmplots' TAD, NONMEM's ``$INPUT TAD``).
Validated against the course reference dataset's own TAD column in the
implementation notes; the convention here matches it to rounding.
"""
import numpy as np
import pytest

from app.compute.dosing import time_after_dose


def _doses(times) -> list[dict]:
    return [{"time": float(t), "amt": 100.0} for t in times]


def test_tad_is_none_before_first_dose():
    out = time_after_dose([-1.0, 0.0], _doses([1.0, 13.0]))
    assert out == [None, None]


def test_tad_relative_to_most_recent_prior_dose():
    # Single dose at t=0: TAD == TIME.
    out = time_after_dose([0.5, 1.0, 4.0, 8.0], _doses([0.0]))
    assert out == [0.5, 1.0, 4.0, 8.0]


def test_tad_resets_at_each_subsequent_dose():
    # Doses at 0, 12, 24 (q12h): TAD resets at each boundary.
    obs_t = [0.5, 11.9, 12.02, 23.5, 24.05]
    out = time_after_dose(obs_t, _doses([0.0, 12.0, 24.0]))
    assert out == pytest.approx([0.5, 11.9, 0.02, 11.5, 0.05], abs=1e-9)


def test_tad_at_exact_dose_time_is_zero():
    # An observation coincident with a dose time belongs to that dose (right
    # side of the interval), matching TAD=0 at the dose event itself.
    out = time_after_dose([0.0, 12.0], _doses([0.0, 12.0]))
    assert out == [0.0, 0.0]


def test_tad_ignores_amt_and_reads_only_time():
    out = time_after_dose([5.0], [{"time": 0.0, "amt": 0.0}])
    assert out == [5.0]


def test_tad_no_doses_returns_all_none():
    out = time_after_dose([1.0, 2.0], [])
    assert out == [None, None]


def test_tad_empty_obs_returns_empty():
    assert time_after_dose([], _doses([0.0])) == []


def test_tad_unsorted_dose_times_handled():
    # dose_events() output is typically sorted, but the function must not
    # assume it — sort internally rather than trust caller ordering.
    out = time_after_dose([13.0], _doses([12.0, 0.0]))
    assert out == [1.0]


def test_tad_matches_course_reference_convention():
    # Mirrors a real MAD (q12h) subject's TAD column from the Week 3/6/7
    # course dataset (sad-mad-renal-nonmem.csv): dose at 0, 12.02, 24.00,
    # 36.06 ...; observation at 24.05 has TAD 0.05 against the most recent
    # prior dose at 24.00 (not the earlier one at 12.02).
    dose_times = [0.0, 12.02, 24.00, 36.06, 47.87, 60.07, 72.09]
    obs_t = [0.87, 1.40, 11.91, 24.05, 49.01]
    out = time_after_dose(obs_t, _doses(dose_times))
    expected = [0.87, 1.40, 11.91, 0.05, 1.14]
    np.testing.assert_allclose([v for v in out], expected, atol=1e-9)


def test_tad_length_matches_obs_t():
    obs_t = np.linspace(0, 100, 37)
    out = time_after_dose(obs_t, _doses([0.0, 24.0, 48.0, 72.0]))
    assert len(out) == len(obs_t)
    assert all(v is None or v >= 0.0 for v in out)
