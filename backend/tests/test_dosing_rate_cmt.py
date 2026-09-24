"""dose_events carries infusion RATE and dosing CMT through to the simulator.

The simulator has always accepted per-dose ``rate``/``cmt``; the dataset bridge
dropped both and collapsed a subject to one dose amount. These tests pin the
new behaviour AND the inertness of the old paths.
"""
from __future__ import annotations

import pytest

from app.compute.dosing import dose_events

KW = dict(time_col="TIME", amt_col="AMT", ii_col="II", addl_col="ADDL")


def _rows(*recs):
    base = {"TIME": 0.0, "AMT": 0.0, "II": None, "ADDL": None, "RATE": 0.0, "CMT": None}
    return [{**base, **r} for r in recs]


# ── inertness: nothing changes for datasets without RATE/CMT ──────────────────

def test_plain_dataset_output_is_unchanged():
    rows = _rows({"TIME": 0.0, "AMT": 100.0}, {"TIME": 24.0, "AMT": 100.0})
    assert dose_events(rows, **KW) == [{"time": 0.0, "amt": 100.0},
                                       {"time": 24.0, "amt": 100.0}]


def test_addl_ii_expansion_is_unchanged():
    rows = _rows({"TIME": 0.0, "AMT": 50.0, "II": 12.0, "ADDL": 3})
    ev = dose_events(rows, **KW)
    assert [e["time"] for e in ev] == [0.0, 12.0, 24.0, 36.0]
    assert {e["amt"] for e in ev} == {50.0}
    assert all(set(e) == {"time", "amt"} for e in ev)      # no stray keys


def test_zero_and_missing_amounts_are_skipped():
    rows = _rows({"TIME": 0.0, "AMT": 0.0}, {"TIME": 1.0, "AMT": None},
                 {"TIME": 2.0, "AMT": 10.0})
    assert dose_events(rows, **KW) == [{"time": 2.0, "amt": 10.0}]


def test_events_are_sorted_by_time():
    rows = _rows({"TIME": 48.0, "AMT": 10.0}, {"TIME": 0.0, "AMT": 10.0})
    assert [e["time"] for e in dose_events(rows, **KW)] == [0.0, 48.0]


# ── per-record amount (previously: last row overwrote every dose) ─────────────

def test_within_subject_dose_change_is_preserved():
    rows = _rows({"TIME": 0.0, "AMT": 100.0}, {"TIME": 24.0, "AMT": 300.0})
    assert [e["amt"] for e in dose_events(rows, **KW)] == [100.0, 300.0]


# ── RATE ─────────────────────────────────────────────────────────────────────

def test_rate_is_carried_when_positive():
    rows = _rows({"TIME": 0.0, "AMT": 100.0, "RATE": 50.0})
    ev = dose_events(rows, **KW, rate_col="RATE")
    assert ev == [{"time": 0.0, "amt": 100.0, "rate": 50.0}]


def test_zero_or_missing_rate_stays_a_bolus():
    for rate in (0.0, None, ""):
        rows = _rows({"TIME": 0.0, "AMT": 100.0, "RATE": rate})
        assert dose_events(rows, **KW, rate_col="RATE") == [{"time": 0.0, "amt": 100.0}]


def test_rate_column_ignored_when_not_requested():
    rows = _rows({"TIME": 0.0, "AMT": 100.0, "RATE": 50.0})
    assert dose_events(rows, **KW) == [{"time": 0.0, "amt": 100.0}]


def test_rate_applies_to_every_addl_expanded_dose():
    rows = _rows({"TIME": 0.0, "AMT": 100.0, "RATE": 50.0, "II": 24.0, "ADDL": 2})
    ev = dose_events(rows, **KW, rate_col="RATE")
    assert len(ev) == 3 and all(e["rate"] == 50.0 for e in ev)


# ── CMT: 1-based dataset -> 0-based simulator index ──────────────────────────

def test_cmt_is_converted_from_one_based_to_zero_based():
    """NONMEM CMT=1 is the FIRST compartment (index 0). Passing it through
    unconverted would dose index 1 — an oral model's central compartment —
    turning an extravascular dose into an IV bolus."""
    rows = _rows({"TIME": 0.0, "AMT": 100.0, "CMT": 1})
    assert dose_events(rows, **KW, cmt_col="CMT")[0]["cmt"] == 0
    rows = _rows({"TIME": 0.0, "AMT": 100.0, "CMT": 2})
    assert dose_events(rows, **KW, cmt_col="CMT")[0]["cmt"] == 1


@pytest.mark.parametrize("bad", [0, -1, None, ""])
def test_cmt_below_one_is_ignored_not_converted(bad):
    """CMT=0 would become -1 and silently dose the LAST compartment."""
    rows = _rows({"TIME": 0.0, "AMT": 100.0, "CMT": bad})
    assert "cmt" not in dose_events(rows, **KW, cmt_col="CMT")[0]


def test_mixed_route_subject_keeps_per_dose_compartment():
    """The vorlizumab shape: an IV dose into central and an SC dose into depot
    within one dataset."""
    rows = _rows({"TIME": 0.0, "AMT": 68.0, "CMT": 2, "RATE": 816.0},
                 {"TIME": 28.0, "AMT": 200.0, "CMT": 1})
    ev = dose_events(rows, **KW, rate_col="RATE", cmt_col="CMT")
    assert ev[0] == {"time": 0.0, "amt": 68.0, "rate": 816.0, "cmt": 1}
    assert ev[1] == {"time": 28.0, "amt": 200.0, "cmt": 0}


# ── end-to-end: the carried fields actually change the simulation ────────────

def test_infusion_and_bolus_differ_in_the_simulator():
    from app.compute.pk_models import REGISTRY
    from app.compute.pk_simulate import simulate
    m = REGISTRY["iv_1cmt"]
    p = {"CL": 5.0, "V": 30.0}
    obs = [0.5, 1.0, 4.0]
    bolus = simulate(m, p, [{"time": 0.0, "amt": 100.0}], obs)["cp"]
    inf = simulate(m, p, [{"time": 0.0, "amt": 100.0, "rate": 100.0}], obs)["cp"]
    # during a 1 h infusion the concentration is still climbing, so it must be
    # below the bolus curve at 0.5 h
    assert inf[0] < bolus[0]


def test_dose_compartment_choice_changes_the_profile():
    from app.compute.pk_models import REGISTRY
    from app.compute.pk_simulate import simulate
    m = REGISTRY["oral_1cmt"]                      # [depot, central]
    p = {"CL": 5.0, "V": 30.0, "KA": 1.0}
    obs = [0.25]
    depot = simulate(m, p, [{"time": 0.0, "amt": 100.0, "cmt": 0}], obs)["cp"]
    central = simulate(m, p, [{"time": 0.0, "amt": 100.0, "cmt": 1}], obs)["cp"]
    assert central[0] > depot[0]                   # IV-style entry is immediate


def test_rate_role_is_detected_and_excluded_from_covariates():
    from app.core.schema_extractor import detect_roles
    roles = detect_roles(["ID", "TIME", "DV", "AMT", "RATE", "CMT", "WT"])
    assert roles["RATE"] == "RATE" and roles["CMT"] == "CMT"
    assert roles.get("WT") is None                 # still a covariate, not a role
