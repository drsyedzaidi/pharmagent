"""KM / log-rank exposure groups when the exposure has ties on the quantile edges.

Exposure often comes as dose levels or rounded values; quantile edges then land ON
data values. Groups must be the bins that really exist (right-closed, like
``pd.qcut``), named in order from the lowest, with the real group count reported.
"""
from __future__ import annotations

import numpy as np

from app.tools.er_data import Analysis
from app.tools.er_fit import _exposure_groups


def _analysis(x: list[float], seed: int = 1) -> Analysis:
    rng = np.random.default_rng(seed)
    n = len(x)
    return Analysis(endpoint="time_to_event", subjects=[f"S{i}" for i in range(n)], exposure=list(x),
                    time=[float(v) for v in rng.uniform(1.0, 50.0, n)],
                    event=[float(v) for v in rng.random(n) < 0.6])


def test_tied_edges_keep_the_lowest_group_and_do_not_pool_distinct_levels():
    x = [50.0] * 30 + [100.0] * 20 + [200.0] * 20 + [400.0] * 10
    g = _exposure_groups(_analysis(x), 4, 0.95)
    names = [grp["group"] for grp in g["groups"]]
    assert names[0] == "Q1 (lowest)" and names[-1].endswith("(highest)")
    spans = [(grp["exposure_min"], grp["exposure_max"], grp["n"]) for grp in g["groups"]]
    assert spans == [(50.0, 50.0, 30), (100.0, 100.0, 20), (200.0, 200.0, 20), (400.0, 400.0, 10)]
    assert g["n_groups"] == len(g["groups"]) == 4
    assert g["logrank"]["df"] == g["n_groups"] - 1


def test_collapsed_quantiles_report_the_real_group_count_and_a_note():
    x = [50.0] * 60 + [100.0] * 20                      # quartile edges 50, 50, 50 -> only two bins exist
    g = _exposure_groups(_analysis(x), 4, 0.95)
    assert [grp["group"] for grp in g["groups"]] == ["Q1 (lowest)", "Q2 (highest)"]
    assert g["n_groups"] == 2 and g["requested_groups"] == 4
    assert g["logrank"]["df"] == 1
    assert "tied" in g["note"]


def test_continuous_exposure_still_gives_the_requested_groups_without_a_note():
    x = list(np.linspace(10.0, 400.0, 80))
    g = _exposure_groups(_analysis(x), 4, 0.95)
    assert g["n_groups"] == 4 and [grp["n"] for grp in g["groups"]] == [20, 20, 20, 20]
    assert "note" not in g
