"""NCA summary carries full descriptive statistics per parameter (N, mean, SD,
CV%, median, min, max, geometric mean, geo-CV%), overall and per dose group —
the standard NCA summary table, not just geomean/geoCV of five parameters."""
from __future__ import annotations

import math

import numpy as np

from app.compute.nca import descriptive_stats, summarize_by_dose


def _rows():
    return [
        {"subject": "1", "dose": 100, "Cmax": 10.0, "AUC_inf": 100.0, "t_half": 4.0, "Tmax": 1.0},
        {"subject": "2", "dose": 100, "Cmax": 20.0, "AUC_inf": 150.0, "t_half": 5.0, "Tmax": 2.0},
        {"subject": "3", "dose": 100, "Cmax": 40.0, "AUC_inf": None, "t_half": 6.0, "Tmax": 1.0},
        {"subject": "4", "dose": 200, "Cmax": 30.0, "AUC_inf": 300.0, "t_half": 4.5, "Tmax": 1.5},
        {"subject": "5", "dose": 200, "Cmax": 50.0, "AUC_inf": 320.0, "t_half": 5.5, "Tmax": 2.0},
    ]


def _param(group, name):
    return next(p for p in group["parameters"] if p["parameter"] == name)


def test_overall_group_matches_numpy():  # stats rounded to 4 dp
    groups = descriptive_stats(_rows())
    allg = next(g for g in groups if g["group"] == "all")
    assert allg["n"] == 5
    cmax = _param(allg, "Cmax")
    v = np.array([10, 20, 40, 30, 50], float)
    assert cmax["n"] == 5
    assert math.isclose(cmax["mean"], v.mean(), abs_tol=1e-3)
    assert math.isclose(cmax["sd"], v.std(ddof=1), abs_tol=1e-3)
    assert math.isclose(cmax["cv_pct"], 100 * v.std(ddof=1) / v.mean(), abs_tol=1e-3)
    assert cmax["median"] == 30 and cmax["min"] == 10 and cmax["max"] == 50
    assert math.isclose(cmax["geomean"], float(np.exp(np.log(v).mean())), abs_tol=1e-3)
    assert math.isclose(cmax["geocv_pct"], 100 * math.sqrt(math.exp(np.log(v).var(ddof=1)) - 1), abs_tol=1e-3)


def test_missing_values_are_excluded_from_n():
    allg = next(g for g in descriptive_stats(_rows()) if g["group"] == "all")
    auc = _param(allg, "AUC_inf")
    assert auc["n"] == 4 and auc["min"] == 100 and auc["max"] == 320


def test_per_dose_groups_only_when_more_than_one_dose():
    groups = descriptive_stats(_rows())
    labels = [g["group"] for g in groups]
    assert labels == ["all", 100, 200]
    d200 = next(g for g in groups if g["group"] == 200)
    assert d200["n"] == 2 and _param(d200, "Cmax")["mean"] == 40
    single = descriptive_stats([r for r in _rows() if r["dose"] == 100])
    assert [g["group"] for g in single] == ["all"]


def test_single_observation_has_no_sd_or_cv():
    g = descriptive_stats([_rows()[0]])[0]
    p = _param(g, "Cmax")
    assert p["n"] == 1 and p["sd"] is None and p["cv_pct"] is None and p["geocv_pct"] is None
    assert p["mean"] == 10 and p["geomean"] == 10


def test_parameters_follow_the_canonical_order_and_skip_absent_ones():
    names = [p["parameter"] for p in descriptive_stats(_rows())[0]["parameters"]]
    assert names == ["Cmax", "Tmax", "AUC_inf", "t_half"]


def test_steady_state_parameters_are_covered():
    rows = [{"subject": "1", "dose": 10, "AUC_tau": 50.0, "Cavg": 2.0, "fluctuation_pct": 80.0,
             "accumulation_ratio": 1.5, "swing_pct": 120.0, "Ctrough": 1.0},
            {"subject": "2", "dose": 10, "AUC_tau": 70.0, "Cavg": 3.0, "fluctuation_pct": 90.0,
             "accumulation_ratio": 1.7, "swing_pct": 150.0, "Ctrough": 1.5}]
    names = [p["parameter"] for p in descriptive_stats(rows)[0]["parameters"]]
    assert names == ["Ctrough", "Cavg", "AUC_tau", "accumulation_ratio", "fluctuation_pct", "swing_pct"]


def test_summary_embeds_descriptive_block():
    s = summarize_by_dose(_rows())
    assert "descriptive" in s and s["descriptive"][0]["group"] == "all"
    assert s["by_dose"]  # the legacy geomean table is untouched


def test_geomean_skips_non_positive_values():
    g = descriptive_stats([{"subject": "1", "dose": 1, "Cmax": 0.0}, {"subject": "2", "dose": 1, "Cmax": 4.0}])[0]
    p = _param(g, "Cmax")
    assert p["n"] == 2 and p["mean"] == 2.0 and p["geomean"] == 4.0
