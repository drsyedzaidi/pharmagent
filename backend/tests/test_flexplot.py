"""Tests for the deterministic Flexplot-style plot geometry (``app.compute.flexplot``).

Covers variable classification, the loess / linear fits and their confidence
bands, deterministic jitter, quantile binning, crossbar summaries, KDE density,
histogram, the top-level ``flexplot`` builder across all four plot kinds
(scatter / dotplot / histogram / density), faceting, colour grouping, the ghost
line, and degenerate inputs. Assertions are numeric-tolerance based (AAA style).

Note on loess: it is a tricube-weighted **local linear** smoother, so it is
exact on a straight line and tracks gentle curves, but (like any local-linear
loess) it oversmooths a full sine period at the default span and has boundary
bias on high curvature — the tolerances below reflect that honestly.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from app.compute.flexplot import (
    _MAX_PANELS,
    _assign_bin,
    _classify_variable,
    _histogram,
    _jitter,
    _kde_density,
    _linear_fit_ci,
    _loess_fit,
    _quantile_bins,
    _summary_crossbar,
    flexplot,
    plottable_variables,
)


# --------------------------------------------------------------------------- #
# classification
# --------------------------------------------------------------------------- #
def test_numeric_high_cardinality_is_continuous():
    assert _classify_variable(pd.Series([1, 2, 3, 4, 5, 6, 7])) == "continuous"


def test_numeric_low_cardinality_is_categorical():
    # flexplot's "< 5 unique -> factor" rule (e.g. 0/1 sex codes)
    assert _classify_variable(pd.Series([0, 1, 0, 1, 0, 1])) == "categorical"


def test_string_is_categorical():
    assert _classify_variable(pd.Series(["a", "b", "a", "c"])) == "categorical"


def test_empty_numeric_column_falls_back_to_dtype():
    assert _classify_variable(pd.Series([], dtype=float)) == "continuous"
    assert _classify_variable(pd.Series([], dtype=object)) == "categorical"


# --------------------------------------------------------------------------- #
# loess
# --------------------------------------------------------------------------- #
def test_loess_is_exact_on_a_straight_line():
    x = np.linspace(0, 10, 60)
    y = 2 * x + 1
    gx, gy, lo, hi = _loess_fit(x, y)
    assert np.max(np.abs(gy - (2 * gx + 1))) < 1e-6


def test_loess_tracks_a_gentle_curve_and_beats_a_line():
    # half sine period — within the reach of span=0.75 local-linear loess
    x = np.linspace(0, math.pi, 120)
    y = np.sin(x)
    gx, gy, lo, hi = _loess_fit(x, y)
    loess_err = float(np.max(np.abs(gy - np.sin(gx))))
    lgx, lgy, _, _ = _linear_fit_ci(x, y)
    line_err = float(np.max(np.abs(np.interp(gx, lgx, lgy) - np.sin(gx))))
    assert loess_err < 0.2
    assert loess_err < line_err  # the smoother is genuinely nonlinear


def test_loess_band_covers_the_truth_on_a_noisy_line():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 200)
    y = 2 * x + 1 + rng.normal(0, 0.5, 200)
    gx, gy, lo, hi = _loess_fit(x, y)
    truth = 2 * gx + 1
    covered = np.mean((lo <= truth) & (truth <= hi))
    assert covered > 0.9


def test_loess_returns_none_below_three_points():
    assert _loess_fit(np.array([1.0, 2.0]), np.array([1.0, 2.0])) is None


# --------------------------------------------------------------------------- #
# linear fit + CI
# --------------------------------------------------------------------------- #
def test_linear_fit_recovers_slope_and_brackets_it():
    rng = np.random.default_rng(1)
    x = np.linspace(0, 10, 200)
    y = 3 * x + 2 + rng.normal(0, 1, 200)
    gx, gy, lo, hi = _linear_fit_ci(x, y)
    # slope recovered
    assert gy[-1] - gy[0] == pytest.approx(3 * (gx[-1] - gx[0]), rel=0.05)
    # true mean line inside the band almost everywhere
    truth = 3 * gx + 2
    assert np.mean((lo <= truth) & (truth <= hi)) > 0.9


def test_linear_ci_band_shrinks_with_n():
    rng = np.random.default_rng(2)
    xa = np.linspace(0, 10, 20)
    xb = np.linspace(0, 10, 1000)
    _, _, lo_a, hi_a = _linear_fit_ci(xa, 3 * xa + rng.normal(0, 1, 20))
    _, _, lo_b, hi_b = _linear_fit_ci(xb, 3 * xb + rng.normal(0, 1, 1000))
    width_a = hi_a[len(hi_a) // 2] - lo_a[len(lo_a) // 2]
    width_b = hi_b[len(hi_b) // 2] - lo_b[len(lo_b) // 2]
    assert width_b < width_a


# --------------------------------------------------------------------------- #
# jitter
# --------------------------------------------------------------------------- #
def test_jitter_is_deterministic_for_a_fixed_seed():
    base = np.array([0.0, 1, 2, 0, 1, 2])
    a = _jitter(base, width=0.2, seed=1)
    b = _jitter(base, width=0.2, seed=1)
    assert np.array_equal(a, b)
    c = _jitter(base, width=0.2, seed=2)
    assert not np.array_equal(a, c)


def test_jitter_stays_within_width():
    base = np.array([0.0, 1, 2, 3, 4])
    out = _jitter(base, width=0.2, seed=7)
    assert np.all(np.abs(out - base) <= 0.2 + 1e-9)


# --------------------------------------------------------------------------- #
# quantile bins
# --------------------------------------------------------------------------- #
def test_quantile_bins_are_monotone_and_labelled():
    edges, labels = _quantile_bins(np.arange(100.0), n_bins=4)
    assert np.all(np.diff(edges) > 0)
    assert len(labels) == len(edges) - 1
    assert labels[0].startswith("[")   # first bin is a closed interval
    assert labels[1].startswith("(")


def test_quantile_bins_dedupe_ties():
    # a mostly-constant column collapses to fewer bins rather than erroring
    edges, labels = _quantile_bins(np.array([1.0] * 90 + [2.0] * 10), n_bins=4)
    assert edges.size >= 2
    assert np.all(np.diff(edges) > 0)


# --------------------------------------------------------------------------- #
# crossbars
# --------------------------------------------------------------------------- #
def test_crossbar_mean_se_matches_formula():
    z = np.array([1.0, 2, 3, 4, 5])
    center, lo, hi, n = _summary_crossbar(z, center="mean_se")
    se = z.std(ddof=1) / math.sqrt(n)   # standard error of the mean = s/sqrt(n)
    assert center == pytest.approx(3.0)
    assert lo == pytest.approx(3.0 - 1.96 * se)
    assert hi == pytest.approx(3.0 + 1.96 * se)


def test_crossbar_median_iqr_matches_quantiles():
    z = np.array([1.0, 2, 3, 4, 5, 6, 7, 8])
    center, lo, hi, n = _summary_crossbar(z, center="median_iqr")
    assert center == pytest.approx(np.median(z))
    assert lo == pytest.approx(np.quantile(z, 0.25))
    assert hi == pytest.approx(np.quantile(z, 0.75))


# --------------------------------------------------------------------------- #
# density + histogram
# --------------------------------------------------------------------------- #
def test_density_integrates_to_about_one():
    sample = np.random.default_rng(0).normal(0, 1, 500)
    gx, gy = _kde_density(sample)
    assert np.trapezoid(gy, gx) == pytest.approx(1.0, abs=0.05)


def test_histogram_counts_sum_to_n():
    edges, counts = _histogram(np.arange(100.0), n_bins=10)
    assert sum(counts) == 100
    assert len(edges) == len(counts) + 1


# --------------------------------------------------------------------------- #
# top-level builder
# --------------------------------------------------------------------------- #
@pytest.fixture
def demo() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({
        "satisfaction": rng.normal(50, 15, n),
        "interests": rng.normal(30, 20, n),
        "sex": rng.choice(["M", "F"], n),
        "wt": rng.uniform(40, 90, n),
        "grp3": rng.choice(["a", "b", "c"], n),
    })
    df["satisfaction"] = df["satisfaction"] + 0.5 * df["interests"]
    return df


def test_scatter_has_fit_band_and_shared_axes(demo):
    p = flexplot(demo, y="satisfaction", x="interests", fit="loess")
    assert p["kind"] == "scatter"
    assert len(p["cells"]) == 1
    fit = p["cells"][0]["fit"]
    assert fit is not None
    assert len(fit["x"]) == len(fit["y"]) == len(fit["lo"]) == len(fit["hi"])
    assert p["x_range"][0] < p["x_range"][1]


def test_scatter_fit_none_omits_curve(demo):
    p = flexplot(demo, y="satisfaction", x="interests", fit="none")
    assert p["cells"][0]["fit"] is None


def test_color_by_splits_into_series(demo):
    p = flexplot(demo, y="satisfaction", x="interests", color_by="sex")
    assert p["summary"]["n_groups"] == 2
    assert len(p["cells"]) == 2
    assert {c["group"] for c in p["cells"]} == {"M", "F"}
    assert [e["color_index"] for e in p["legend"]] == [0, 1]


def test_categorical_panel_makes_one_facet_per_level(demo):
    p = flexplot(demo, y="satisfaction", x="interests", panel_by="grp3")
    assert p["summary"]["n_panels"] == 3
    assert all(pm["bin_range"] is None for pm in p["panels"])


def test_continuous_panel_is_quantile_binned(demo):
    p = flexplot(demo, y="satisfaction", x="interests", panel_by="wt")
    assert p["var_types"]["panel_by"] == "continuous_binned"
    assert p["summary"]["n_panels"] >= 2
    assert all(pm["bin_range"] is not None for pm in p["panels"])


def test_ghost_line_present_only_with_multiple_panels(demo):
    with_ghost = flexplot(demo, y="satisfaction", x="interests", panel_by="wt", ghost=True)
    assert with_ghost["ghost_line"] is not None
    no_panel = flexplot(demo, y="satisfaction", x="interests", ghost=True)
    assert no_panel["ghost_line"] is None  # ghost needs >= 2 panels


def test_dotplot_has_crossbars_and_jittered_points(demo):
    p = flexplot(demo, y="satisfaction", x="sex", center="mean_se")
    assert p["kind"] == "dotplot"
    cell = p["cells"][0]
    assert len(cell["crossbars"]) == 2  # M and F
    assert p["x_categories"] == sorted(["M", "F"])


def test_histogram_and_density_are_univariate(demo):
    h = flexplot(demo, y="satisfaction")
    assert h["kind"] == "histogram"
    assert h["cells"][0]["bins"] is not None
    d = flexplot(demo, y="satisfaction", geom="density")
    assert d["kind"] == "density"
    assert d["cells"][0]["density"] is not None


def test_categorical_outcome_raises(demo):
    with pytest.raises(ValueError, match="continuous"):
        flexplot(demo, y="sex", x="interests")


def test_empty_dataframe_returns_zero_n_not_crash():
    df = pd.DataFrame({"a": pd.Series([], dtype=float), "b": pd.Series([], dtype=float)})
    p = flexplot(df, y="a", x="b")
    assert p["summary"]["n"] == 0
    assert p["summary"]["y_mean"] is None
    assert p["x_range"][0] < p["x_range"][1]  # never a zero-width axis


def test_single_member_group_cell_does_not_crash():
    df = pd.DataFrame({
        "y": [1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "x": [1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "g": ["A"] * 9 + ["B"],
    })
    p = flexplot(df, y="y", x="x", color_by="g")
    cell_b = next(c for c in p["cells"] if c["group"] == "B")
    assert cell_b["n"] == 1
    assert cell_b["fit"] is None


def test_unknown_variable_raises(demo):
    with pytest.raises(ValueError, match="not in the dataset"):
        flexplot(demo, y="satisfaction", x="nope")


# --------------------------------------------------------------------------- #
# plottable_variables (metadata only)
# --------------------------------------------------------------------------- #
def test_plottable_variables_types_from_metadata():
    meta = {
        "columns": [
            {"name": "TIME", "dtype": "float64", "role": "TIME",
             "summary": {"n": 96, "min": 0.0, "max": 24.0}},
            {"name": "SEX", "dtype": "object", "role": "",
             "summary": {"n_unique": 2}},
        ],
    }
    out = plottable_variables(meta)
    by_name = {v["name"]: v for v in out}
    assert by_name["TIME"]["type"] == "continuous"
    assert by_name["TIME"]["min"] == 0.0 and by_name["TIME"]["max"] == 24.0
    assert by_name["SEX"]["type"] == "categorical"
    assert by_name["SEX"]["n_unique"] == 2


def test_plottable_variables_empty_metadata():
    assert plottable_variables(None) == []
    assert plottable_variables({}) == []


def test_plottable_variables_low_cardinality_numeric_is_categorical():
    # DOSE-like numeric code with < 5 distinct values -> categorical, matching
    # the server's authoritative classification (n_unique now carried in metadata).
    meta = {"columns": [
        {"name": "DOSE", "dtype": "int64", "role": "",
         "summary": {"n": 120, "n_unique": 2, "min": 100.0, "max": 300.0}},
        {"name": "CONC", "dtype": "float64", "role": "DV",
         "summary": {"n": 120, "n_unique": 96, "min": 0.1, "max": 9.0}},
    ]}
    by_name = {v["name"]: v for v in plottable_variables(meta)}
    assert by_name["DOSE"]["type"] == "categorical"
    assert by_name["DOSE"]["n_unique"] == 2
    assert by_name["CONC"]["type"] == "continuous"


# --------------------------------------------------------------------------- #
# hardening / boundary behaviors (from adversarial review)
# --------------------------------------------------------------------------- #
def test_ci_out_of_range_raises(demo):
    for bad in (1.0, 1.5, 0.0, -0.5, 95.0):
        with pytest.raises(ValueError, match="ci must be"):
            flexplot(demo, y="satisfaction", x="interests", ci=bad)


def test_valid_ci_keeps_fit_arrays_parallel(demo):
    fit = flexplot(demo, y="satisfaction", x="interests", fit="linear", ci=0.99)["cells"][0]["fit"]
    assert len(fit["x"]) == len(fit["y"]) == len(fit["lo"]) == len(fit["hi"])


def test_n_bins_is_clamped_not_unbounded(demo):
    p = flexplot(demo, y="satisfaction", n_bins=5_000_000)
    # capped to _MAX_BINS=200 -> at most 201 edges, never a multi-GB allocation
    assert len(p["cells"][0]["bins"]["edges"]) <= 201


def test_bin_edge_ties_fall_in_the_lower_right_closed_bin():
    edges, labels = _quantile_bins(np.arange(0, 100.0), n_bins=4)
    interior = edges[2]
    b = int(_assign_bin(np.array([interior]), edges)[0])
    # right-closed (a, b]: an interior edge belongs to the bin that ends at it
    assert labels[b].endswith(f"{interior:.1f}]") or labels[b].endswith(f"{interior:.1f}k]")


def test_color_by_above_max_groups_drops_the_aesthetic():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "y": rng.normal(0, 1, n),
        "x": np.arange(n, dtype=float),
        "many": [f"g{i % 9}" for i in range(n)],   # 9 > _MAX_GROUPS
    })
    p = flexplot(df, y="y", x="x", color_by="many")
    assert p["echo"]["color_by"] is None
    assert p["groups"] == [""]
    assert p["summary"]["n_groups"] == 1


def test_point_cap_subsamples_but_reports_true_n():
    n = 8000
    x = np.arange(n, dtype=float)
    df = pd.DataFrame({"y": 2 * x + 1, "x": x})
    cell = flexplot(df, y="y", x="x")["cells"][0]
    assert len(cell["points"]["x"]) <= 3000     # shipped points capped
    assert cell["n"] == n                        # true count preserved


def test_continuous_panel_capped_to_four():
    rng = np.random.default_rng(1)
    n = 400
    df = pd.DataFrame({"y": rng.normal(0, 1, n), "x": np.arange(n, dtype=float),
                       "cov": rng.uniform(0, 100, n)})
    p = flexplot(df, y="y", x="x", panel_by="cov", n_bins=10)
    assert p["summary"]["n_panels"] == 4         # min(n_bins, 4)


def test_categorical_panel_cardinality_is_capped():
    rng = np.random.default_rng(2)
    n = 600
    df = pd.DataFrame({"y": rng.normal(0, 1, n), "x": np.arange(n, dtype=float),
                       "id": [f"s{i % 50}" for i in range(n)]})   # 50 levels
    p = flexplot(df, y="y", x="x", panel_by="id")
    assert p["summary"]["n_panels"] == _MAX_PANELS


def test_ghost_line_entries_are_well_formed(demo):
    p = flexplot(demo, y="satisfaction", x="interests", panel_by="wt", ghost=True)
    assert p["ghost_line"]
    for g in p["ghost_line"]:
        assert set(g) >= {"group", "color_index", "x", "y"}
        assert len(g["x"]) == len(g["y"]) and len(g["x"]) > 1


def test_density_cell_has_parallel_xy(demo):
    d = flexplot(demo, y="satisfaction", geom="density")["cells"][0]["density"]
    assert d is not None
    assert len(d["x"]) == len(d["y"]) and len(d["x"]) > 1


def test_dotplot_with_color_splits_into_group_cells(demo):
    p = flexplot(demo, y="satisfaction", x="sex", color_by="grp3")
    assert p["kind"] == "dotplot"
    assert p["summary"]["n_groups"] == 3
    # one cell per colour group, each carrying its own crossbars
    assert len(p["cells"]) == 3
    assert all(c["crossbars"] for c in p["cells"])
