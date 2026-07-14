"""Flexplot-style exploratory plot geometry (a jamovi/flexplot reimplementation).

Pure, deterministic compute on pandas/numpy/scipy only — no statsmodels, no R,
no ggplot2. The server precomputes **all** plot geometry (points, fit bands,
crossbars, density curves, jitter offsets, ghost lines); the browser only maps
data coordinates to pixels. This keeps the statistics in one audited place and
lets the frontend stay a thin inline-SVG renderer, matching the rest of the app
(``compute.vpc`` / ``compute.diagnostics``).

Faithfulness notes (documented deviations from the R ``flexplot`` package):

* **Loess** is a tricube-weighted **local linear** regression (Cleveland 1979).
  R/ggplot loess defaults to local quadratic; local linear is used here for
  numerical robustness. The pointwise confidence band uses the hat-vector norm
  with a residual-variance sigma and ``dof = n - trace(L)``; this is the
  classical Cleveland inference and is *approximate* relative to R's exact
  ``delta1/delta2`` degrees of freedom.
* **Linear** fit confidence band is the exact ``predict.lm`` mean-response
  interval (scipy ``linregress`` + Student-t).
* **Jitter** for categorical axes uses a fixed ``numpy.random.default_rng``
  seed so dot coordinates are reproducible run to run (the absolute offsets
  differ from R, which is cosmetic).
* **KDE** bandwidth is R's ``nrd0`` rule for parity with flexplot's density.

Every returned float is rounded to ``_ROUND_DP``; non-finite values are dropped
pairwise so a degenerate column never poisons the summary. Empty / single-point
inputs return an ``n == 0`` (or degenerate) payload rather than raising.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

# Decimal places for all reported floats (mirrors compute.vpc / compute.diagnostics).
_ROUND_DP = 6
# Fixed RNG seed for reproducible categorical-axis jitter.
_DEFAULT_SEED = 20250614
# A numeric column with fewer distinct values than this is treated as categorical
# on an axis (flexplot's "< 5 unique -> ordered factor" rule).
_MIN_CONTINUOUS_LEVELS = 5
# Drop the colour aesthetic above this many groups (unreadable legend, per flexplot).
_MAX_GROUPS = 6
# Cap points shipped per cell so a large dataset cannot bloat the payload; the
# subsample is deterministic (evenly spaced indices) and ``n`` still reports the
# true count. The same cap bounds the rows fed to the (super-linear) loess fit so
# a large scatter cannot hang the worker.
_POINT_CAP = 3000
# Facet / categorical-axis cardinality caps: a high-cardinality categorical
# column (e.g. a subject id) must not explode compute or the payload. Levels
# beyond the cap are dropped (rows in them are excluded), mirroring the
# ``_MAX_GROUPS`` colour cap.
_MAX_PANELS = 12
_MAX_X_CATEGORIES = 30
# Upper bound on histogram / quantile bins (guards against an OOM request).
_MAX_BINS = 200
# Loess default neighbourhood fraction (ggplot/flexplot default span).
_LOESS_SPAN = 0.75
# Grid resolution for fit / density curves.
_FIT_GRID = 100
_KDE_GRID = 200


# --------------------------------------------------------------------------- #
# Variable classification
# --------------------------------------------------------------------------- #
def _classify_variable(s: pd.Series) -> str:
    """Return ``"continuous"`` or ``"categorical"`` for a column.

    Continuous = numerically parseable for (nearly) all non-null values AND at
    least ``_MIN_CONTINUOUS_LEVELS`` distinct values. Everything else — strings,
    factors, and low-cardinality numeric codes (e.g. 0/1 sex) — is categorical.
    """
    non_null = s.dropna()
    if non_null.empty:
        # No data to inspect (empty column or empty filtered subset): fall back
        # to the declared dtype so an empty numeric column is still continuous.
        return "continuous" if pd.api.types.is_numeric_dtype(s) else "categorical"
    num = pd.to_numeric(non_null, errors="coerce")
    frac_numeric = float(num.notna().mean())
    if frac_numeric >= 0.9 and int(num.dropna().nunique()) >= _MIN_CONTINUOUS_LEVELS:
        return "continuous"
    return "categorical"


def _round(v: float | None) -> float | None:
    """Round a float to the reporting precision; map non-finite to ``None``."""
    if v is None:
        return None
    f = float(v)
    if not np.isfinite(f):
        return None
    return round(f, _ROUND_DP)


def _round_list(a: Any) -> list[float]:
    return [round(float(v), _ROUND_DP) for v in np.asarray(a, float) if np.isfinite(v)]


# --------------------------------------------------------------------------- #
# Binning
# --------------------------------------------------------------------------- #
def _fmt_edge(v: float) -> str:
    """Compact numeric label for a bin edge."""
    a = abs(v)
    if a >= 1000:
        return f"{v/1000:.1f}k"
    if a >= 1:
        return f"{v:.1f}"
    if a == 0:
        return "0"
    return f"{v:.3g}"


def _bin_interval_label(lo: float, hi: float, *, first: bool) -> str:
    """R ``cut``-style interval label: ``[a, b]`` for the first bin else ``(a, b]``."""
    left = "[" if first else "("
    return f"{left}{_fmt_edge(lo)}, {_fmt_edge(hi)}]"


def _quantile_bins(x: np.ndarray, n_bins: int) -> tuple[np.ndarray, list[str]]:
    """Quantile (equal-count) bin edges + interval labels.

    Edges are ``numpy.quantile`` at ``linspace(0, 1, n_bins+1)`` (linear
    interpolation == R type-7). Tied quantiles are de-duplicated, so a spiky
    column simply yields fewer bins.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.array([]), []
    n_bins = max(1, int(n_bins))
    edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size < 2:
        lo = float(x.min())
        hi = float(x.max())
        edges = np.array([lo, hi if hi > lo else lo + 1.0])
    labels = [
        _bin_interval_label(edges[i], edges[i + 1], first=(i == 0))
        for i in range(edges.size - 1)
    ]
    return edges, labels


def _assign_bin(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Index each value into ``[0, len(edges)-2]``; values on/below the first
    edge fall in bin 0, values on/above the last edge in the last bin.

    ``right=True`` so a value exactly on an interior edge lands in the lower
    ``(a, b]`` bin — matching the right-closed interval strip labels.
    """
    idx = np.digitize(values, edges[1:-1], right=True)
    return np.clip(idx, 0, len(edges) - 2)


# --------------------------------------------------------------------------- #
# Fits
# --------------------------------------------------------------------------- #
def _linear_fit_ci(
    x: np.ndarray, y: np.ndarray, *, ci: float = 0.95, grid: int = _FIT_GRID
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Ordinary least squares fit with the exact mean-response confidence band.

    ``se(x0) = sigma * sqrt(1/n + (x0 - xbar)^2 / Sxx)`` and the band half-width
    is ``t_{1-a/2, n-2} * se`` — identical to ``predict.lm(interval="confidence")``.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = x.size
    if n < 2 or np.ptp(x) == 0:
        return None
    res = stats.linregress(x, y)
    gx = np.linspace(float(x.min()), float(x.max()), grid)
    yhat = res.intercept + res.slope * gx
    if n <= 2:
        return gx, yhat, yhat.copy(), yhat.copy()
    xbar = float(x.mean())
    sxx = float(np.sum((x - xbar) ** 2))
    sse = float(np.sum((y - (res.intercept + res.slope * x)) ** 2))
    sigma2 = sse / (n - 2)
    if sxx <= 0:
        band = np.zeros_like(gx)
    else:
        se = np.sqrt(sigma2 * (1.0 / n + (gx - xbar) ** 2 / sxx))
        band = float(stats.t.ppf((1.0 + ci) / 2.0, n - 2)) * se
    return gx, yhat, yhat - band, yhat + band


def _loess_local_row(x0: float, xv: np.ndarray, q: int) -> np.ndarray | None:
    """The linear-smoother weight row ``l`` such that ``yhat(x0) = l . y``.

    Tricube weights over the ``q`` nearest neighbours; weighted local-linear
    design ``[1, x]``. Returns ``None`` if the local design is singular.
    """
    d = np.abs(xv - x0)
    h = np.partition(d, q - 1)[q - 1]
    if h <= 0:
        h = d.max()
    if h <= 0:
        h = 1.0
    u = np.clip(d / h, 0.0, 1.0)
    w = (1.0 - u ** 3) ** 3
    design = np.column_stack([np.ones_like(xv), xv])  # n x 2
    xtw = design.T * w                                 # 2 x n
    xtwx = xtw @ design                                # 2 x 2
    try:
        inv = np.linalg.inv(xtwx)
    except np.linalg.LinAlgError:
        return None
    return np.array([1.0, x0]) @ inv @ xtw             # length n


def _loess_fit(
    x: np.ndarray,
    y: np.ndarray,
    *,
    span: float = _LOESS_SPAN,
    ci: float = 0.95,
    grid: int = _FIT_GRID,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Tricube-weighted local-linear loess with a pointwise confidence band.

    Falls back to ``None`` for fewer than 3 finite points (caller may retry with
    a linear fit). The band uses ``Var(yhat(x0)) = sigma^2 ||l(x0)||^2`` with
    ``sigma^2 = RSS / (n - trace(L))`` — Cleveland's classical loess inference.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = x.size
    if n < 3 or np.ptp(x) == 0:
        return None
    order = np.argsort(x, kind="stable")
    xs_data = x[order]
    ys_data = y[order]
    q = int(np.clip(np.ceil(span * n), 3, n))

    gx = np.linspace(xs_data[0], xs_data[-1], grid)
    ys = np.full(grid, np.nan)
    norms = np.full(grid, np.nan)
    for i, x0 in enumerate(gx):
        row = _loess_local_row(x0, xs_data, q)
        if row is not None:
            ys[i] = float(row @ ys_data)
            norms[i] = float(np.sqrt(np.sum(row ** 2)))

    # Residual variance + effective degrees of freedom via the hat diagonal.
    fitted = np.empty(n)
    lev = np.empty(n)
    for i in range(n):
        row = _loess_local_row(xs_data[i], xs_data, q)
        if row is None:
            fitted[i] = ys_data[i]
            lev[i] = 0.0
        else:
            fitted[i] = float(row @ ys_data)
            lev[i] = float(row[i])
    rss = float(np.sum((ys_data - fitted) ** 2))
    trace_l = float(np.sum(lev))
    dof = max(n - trace_l, 1.0)
    sigma = float(np.sqrt(rss / dof)) if dof > 0 else 0.0
    tval = float(stats.t.ppf((1.0 + ci) / 2.0, dof))

    # Interpolate any singular grid gaps so the returned curve is dense.
    good = np.isfinite(ys)
    if not good.any():
        return None
    if not good.all():
        ys = np.interp(gx, gx[good], ys[good])
        norms = np.interp(gx, gx[good], norms[good])
    lo = ys - tval * sigma * norms
    hi = ys + tval * sigma * norms
    return gx, ys, lo, hi


def _fit_curve(
    x: np.ndarray, y: np.ndarray, *, fit: str, ci: float
) -> dict[str, list[float]] | None:
    """Dispatch to the requested fit; loess degrades to linear when it cannot fit."""
    if fit == "none":
        return None
    result = None
    if fit == "loess":
        result = _loess_fit(x, y, ci=ci)
        if result is None:
            result = _linear_fit_ci(x, y, ci=ci)
    elif fit == "linear":
        result = _linear_fit_ci(x, y, ci=ci)
    if result is None:
        return None
    gx, gy, lo, hi = result
    return {
        "x": _round_list(gx),
        "y": _round_list(gy),
        "lo": _round_list(lo),
        "hi": _round_list(hi),
    }


# --------------------------------------------------------------------------- #
# Jitter, crossbars, density, histogram
# --------------------------------------------------------------------------- #
def _jitter(base: np.ndarray, *, width: float, seed: int) -> np.ndarray:
    """Deterministic uniform jitter in ``[-width, width]`` added to ``base``.

    Same ``(base order, width, seed)`` -> identical output (asserted in tests).
    """
    base = np.asarray(base, float)
    if width <= 0 or base.size == 0:
        return base
    rng = np.random.default_rng(seed)
    return base + rng.uniform(-width, width, base.size)


def _summary_crossbar(
    z: np.ndarray, *, center: str
) -> tuple[float, float, float, int] | None:
    """Return ``(center, lo, hi, n)`` for a category's outcome values.

    * ``median_iqr``: median with the 25th/75th percentiles.
    * ``mean_se``:    mean +/- 1.96 * SEM, where SEM = sd / sqrt(n) (ddof=1 sd).
    * ``mean_sd``:    mean +/- sd (ddof=1).
    """
    z = np.asarray(z, float)
    z = z[np.isfinite(z)]
    n = int(z.size)
    if n == 0:
        return None
    if center == "median_iqr":
        return float(np.median(z)), float(np.quantile(z, 0.25)), float(np.quantile(z, 0.75)), n
    mean = float(z.mean())
    sd = float(z.std(ddof=1)) if n > 1 else 0.0
    if center == "mean_sd":
        return mean, mean - sd, mean + sd, n
    se = sd / np.sqrt(n) if n > 0 else 0.0   # standard error of the mean
    return mean, mean - 1.96 * se, mean + 1.96 * se, n


def _kde_density(sample: np.ndarray, *, grid: int = _KDE_GRID) -> tuple[np.ndarray, np.ndarray] | None:
    """Gaussian KDE evaluated on a grid over the data range, R ``nrd0`` bandwidth."""
    s = np.asarray(sample, float)
    s = s[np.isfinite(s)]
    n = s.size
    if n < 2:
        return None
    std = float(s.std(ddof=1))
    iqr = float(np.subtract(*np.quantile(s, [0.75, 0.25])))
    spread = min(std, iqr / 1.34) if iqr > 0 else std
    if spread <= 0:
        spread = std
    if spread <= 0:
        return None
    bw = 0.9 * spread * n ** (-0.2)  # Silverman's rule-of-thumb (R nrd0)
    if bw <= 0 or std <= 0:
        return None
    kde = stats.gaussian_kde(s, bw_method=bw / std)
    gx = np.linspace(float(s.min()), float(s.max()), grid)
    gy = kde(gx)
    return gx, gy


def _histogram(y: np.ndarray, *, n_bins: int) -> tuple[list[float], list[int]]:
    """``numpy.histogram`` edges + integer counts (counts sum to the finite n)."""
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return [], []
    counts, edges = np.histogram(y, bins=max(1, int(n_bins)))
    return _round_list(edges), [int(c) for c in counts]


# --------------------------------------------------------------------------- #
# Point capping
# --------------------------------------------------------------------------- #
def _cap_indices(n: int) -> np.ndarray | None:
    """Deterministic evenly-spaced subsample indices when ``n > _POINT_CAP``."""
    if n <= _POINT_CAP:
        return None
    return np.linspace(0, n - 1, _POINT_CAP).astype(int)


# --------------------------------------------------------------------------- #
# Top-level builder
# --------------------------------------------------------------------------- #
def _resolve_groups(series: pd.Series | None) -> list[str]:
    if series is None:
        return [""]
    levels = [str(v) for v in pd.Series(series.dropna().unique())]
    levels.sort()
    return levels or [""]


def flexplot(  # noqa: C901 - a single cohesive builder; helpers are extracted
    df: pd.DataFrame,
    *,
    y: str,
    x: str | None = None,
    color_by: str | None = None,
    panel_by: str | None = None,
    fit: str = "loess",
    geom: str = "points",
    center: str = "median_iqr",
    ghost: bool = False,
    log_y: bool = False,
    jitter: float = 0.2,
    n_bins: int = 10,
    ci: float = 0.95,
    seed: int = _DEFAULT_SEED,
) -> dict[str, Any]:
    """Build the full FlexplotData payload for one outcome / predictor selection.

    ``y`` is the outcome (required); ``x`` the optional predictor. Plot kind is
    resolved from the resolved variable types:

    ======================  ================================================
    inputs                  kind
    ======================  ================================================
    y only                  ``histogram`` (or ``density`` if ``geom="density"``)
    continuous y, cont. x   ``scatter`` (+ fit band, optional ghost line)
    continuous y, cat. x    ``dotplot`` (jittered points + crossbars)
    ======================  ================================================

    ``color_by`` splits into colour series; ``panel_by`` facets (a continuous
    panel is quantile-binned into ``min(n_bins, ...)`` panels). Unsupported
    combinations raise ``ValueError`` with an actionable message.
    """
    for name in (y, x, color_by, panel_by):
        if name is not None and name not in df.columns:
            raise ValueError(f"variable {name!r} is not in the dataset")

    # Boundary validation — fail loud on an invalid confidence level (a t-quantile
    # at ci>=1 is infinite and would silently drop the band); clamp the benign
    # display knobs so a hostile n_bins cannot OOM the worker.
    if not 0.0 < ci < 1.0:
        raise ValueError(f"ci must be between 0 and 1 (exclusive), got {ci!r}")
    n_bins = int(min(max(n_bins, 1), _MAX_BINS))
    jitter = float(max(jitter, 0.0))

    yt = _classify_variable(df[y])
    xt = _classify_variable(df[x]) if x else None

    if x is None:
        if yt != "continuous":
            raise ValueError(
                f"a distribution plot needs a continuous outcome; {y!r} is categorical"
            )
        kind = "density" if geom == "density" else "histogram"
    elif yt == "continuous" and xt == "continuous":
        kind = "scatter"
    elif yt == "continuous" and xt == "categorical":
        kind = "dotplot"
    else:
        raise ValueError(
            "unsupported combination for this release: outcome must be continuous "
            f"(got {y!r}={yt}); predictor {x!r}={xt}. Categorical outcomes "
            "(logistic / association plots) are not yet available."
        )

    color_by = color_by or None
    panel_by = panel_by or None
    # Colour aesthetic only for bivariate plots and within the legibility cap.
    groups = _resolve_groups(df[color_by]) if (color_by and kind in ("scatter", "dotplot")) else [""]
    if len(groups) > _MAX_GROUPS:
        color_by = None
        groups = [""]

    # ------------------------------------------------------------------ panels
    panel_type: str | None = None
    panel_ids: list[str] = ["p0"]
    panel_meta: list[dict[str, Any]] = [{"id": "p0", "strip": "", "bin_range": None}]
    panel_assign: pd.Series | None = None  # row -> panel id
    if panel_by:
        panel_type = _classify_variable(df[panel_by])
        if panel_type == "continuous":
            pv = pd.to_numeric(df[panel_by], errors="coerce").to_numpy(dtype=float)
            edges, labels = _quantile_bins(pv, n_bins=min(n_bins, 4) if n_bins > 4 else n_bins)
            if edges.size >= 2:
                bins = _assign_bin(pv, edges)
                panel_ids = [f"p{i}" for i in range(len(labels))]
                panel_meta = [
                    {
                        "id": f"p{i}",
                        "strip": f"{panel_by}:\n{labels[i]}",
                        "bin_range": [_round(edges[i]), _round(edges[i + 1])],
                    }
                    for i in range(len(labels))
                ]
                panel_assign = pd.Series(
                    [f"p{b}" if np.isfinite(pv[k]) else None for k, b in enumerate(bins)],
                    index=df.index,
                )
                panel_type = "continuous_binned"
        else:
            levels = _resolve_groups(df[panel_by])[:_MAX_PANELS]
            panel_ids = [f"p{i}" for i in range(len(levels))]
            panel_meta = [
                {"id": f"p{i}", "strip": f"{panel_by}:\n{levels[i]}", "bin_range": None}
                for i in range(len(levels))
            ]
            level_to_id = {lev: f"p{i}" for i, lev in enumerate(levels)}
            panel_assign = df[panel_by].astype("object").map(
                lambda v: level_to_id.get(str(v)) if pd.notna(v) else None
            )

    # ------------------------------------------------------------------ cells
    y_num = pd.to_numeric(df[y], errors="coerce")
    x_is_cat = kind == "dotplot"
    if x is not None:
        if x_is_cat:
            x_levels = _resolve_groups(df[x])[:_MAX_X_CATEGORIES]
            x_index_map = {lev: i for i, lev in enumerate(x_levels)}
        else:
            x_num = pd.to_numeric(df[x], errors="coerce")

    cells: list[dict[str, Any]] = []
    all_x: list[float] = []
    all_y: list[float] = []

    def _group_mask(g: str) -> pd.Series:
        if color_by and g != "":
            return df[color_by].astype(str) == g
        return pd.Series(True, index=df.index)

    def _panel_mask(pid: str) -> pd.Series:
        if panel_assign is None:
            return pd.Series(True, index=df.index)
        return panel_assign == pid

    for pid in panel_ids:
        for g in groups:
            mask = _group_mask(g) & _panel_mask(pid)
            sub_y = y_num[mask]
            cell: dict[str, Any] = {
                "panel": pid,
                "group": g,
                "points": {"x": [], "y": []},
                "fit": None,
                "crossbars": [],
                "bins": None,
                "density": None,
                "n": 0,
            }
            if kind in ("histogram", "density"):
                yv = sub_y.to_numpy(dtype=float)
                yv = yv[np.isfinite(yv)]
                cell["n"] = int(yv.size)
                if kind == "histogram":
                    edges, counts = _histogram(yv, n_bins=n_bins)
                    cell["bins"] = {"edges": edges, "counts": counts}
                    all_y.extend(counts)  # y-axis is count
                    all_x.extend(edges)
                else:
                    dens = _kde_density(yv)
                    if dens is not None:
                        dx, dy = dens
                        cell["density"] = {"x": _round_list(dx), "y": _round_list(dy)}
                        all_x.extend(_round_list(dx))
                        all_y.extend(_round_list(dy))
                cells.append(cell)
                continue

            # bivariate: pair x,y dropping non-finite / unmapped rows
            if x_is_cat:
                xs_raw = df[x][mask].astype("object")
                pairs = [
                    (x_index_map[str(xv)], yv)
                    for xv, yv in zip(xs_raw, sub_y, strict=True)
                    if pd.notna(xv) and str(xv) in x_index_map and np.isfinite(yv)
                ]
            else:
                xs_raw = x_num[mask].to_numpy(dtype=float)
                ys_raw = sub_y.to_numpy(dtype=float)
                finite = np.isfinite(xs_raw) & np.isfinite(ys_raw)
                pairs = list(zip(xs_raw[finite].tolist(), ys_raw[finite].tolist(), strict=True))

            cell["n"] = len(pairs)
            if not pairs:
                cells.append(cell)
                continue

            px = np.array([p[0] for p in pairs], dtype=float)
            py = np.array([p[1] for p in pairs], dtype=float)

            if x_is_cat:
                # crossbar per category (before jitter), then jittered dots
                for lev, ci_ in x_index_map.items():
                    zv = py[px == ci_]
                    cb = _summary_crossbar(zv, center=center)
                    if cb is not None:
                        c_center, c_lo, c_hi, c_n = cb
                        cell["crossbars"].append({
                            "group_x": lev,
                            "x_index": ci_,
                            "center": _round(c_center),
                            "lo": _round(c_lo),
                            "hi": _round(c_hi),
                            "n": c_n,
                        })
                plot_x = _jitter(px, width=jitter, seed=seed)
            else:
                plot_x = px
                # Fit on a deterministically capped subsample: the loess leverage
                # loop is O(n^2), so an uncapped large scatter would hang the
                # worker. 3000 evenly-spaced points reproduce a smooth trend.
                fcap = _cap_indices(len(pairs))
                if fcap is not None:
                    fit_curve = _fit_curve(px[fcap], py[fcap], fit=fit, ci=ci)
                else:
                    fit_curve = _fit_curve(px, py, fit=fit, ci=ci)
                cell["fit"] = fit_curve

            # deterministic point cap
            cap = _cap_indices(len(pairs))
            if cap is not None:
                cell_x = plot_x[cap]
                cell_y = py[cap]
            else:
                cell_x = plot_x
                cell_y = py
            cell["points"] = {"x": _round_list(cell_x), "y": _round_list(cell_y)}
            all_x.extend(px.tolist())
            all_y.extend(py.tolist())
            cells.append(cell)

    # ------------------------------------------------------------------ ghost line
    ghost_line = None
    if ghost and kind == "scatter" and fit != "none" and len(panel_ids) > 1:
        ghost_line = []
        for gi, g in enumerate(groups):
            mask = _group_mask(g)
            gx_raw = x_num[mask].to_numpy(dtype=float)
            gy_raw = y_num[mask].to_numpy(dtype=float)
            finite = np.isfinite(gx_raw) & np.isfinite(gy_raw)
            gx_fin, gy_fin = gx_raw[finite], gy_raw[finite]
            gcap = _cap_indices(gx_fin.size)   # bound the O(n^2) loess fit
            if gcap is not None:
                gx_fin, gy_fin = gx_fin[gcap], gy_fin[gcap]
            curve = _fit_curve(gx_fin, gy_fin, fit=fit, ci=ci)
            if curve is not None:
                color_index = _group_color_index(color_by, groups, gi)
                ghost_line.append({
                    "group": g,
                    "color_index": color_index,
                    "x": curve["x"],
                    "y": curve["y"],
                })
        if not ghost_line:
            ghost_line = None

    # ------------------------------------------------------------------ ranges + rollup
    fx = np.array([v for v in all_x if np.isfinite(v)], dtype=float)
    fy = np.array([v for v in all_y if np.isfinite(v)], dtype=float)
    # include fit / crossbar extents so bands are never clipped
    for cell in cells:
        if cell["fit"]:
            fy = np.concatenate([fy, cell["fit"]["lo"], cell["fit"]["hi"]])
        for cb in cell["crossbars"]:
            fy = np.concatenate([fy, [v for v in (cb["lo"], cb["hi"]) if v is not None]])
    x_range = [_round(float(fx.min())), _round(float(fx.max()))] if fx.size else [0.0, 1.0]
    y_range = [_round(float(fy.min())), _round(float(fy.max()))] if fy.size else [0.0, 1.0]
    if x_range[0] == x_range[1]:
        x_range = [x_range[0], x_range[0] + 1.0]
    if y_range[0] == y_range[1]:
        y_range = [y_range[0], y_range[0] + 1.0]

    total_n = int(sum(c["n"] for c in cells))
    legend = [
        {
            "id": g,
            "label": f"{color_by}={g}" if (color_by and g != "") else (color_by or ""),
            "color_index": _group_color_index(color_by, groups, gi),
        }
        for gi, g in enumerate(groups)
    ]

    x_axis = "count" if kind == "histogram" else (y if kind == "density" else (x or ""))
    y_axis = "density" if kind == "density" else (y if kind in ("scatter", "dotplot") else y)
    if kind == "histogram":
        x_axis, y_axis = y, "count"

    x_type = "categorical" if x_is_cat else ("continuous" if x else None)
    return {
        "kind": kind,
        "echo": {
            "x": x, "y": y, "color_by": color_by, "panel_by": panel_by,
            "fit": fit, "geom": geom, "center": center, "ghost": ghost,
            "n_bins": int(n_bins), "ci": _round(ci), "log_y": bool(log_y),
            "jitter": _round(jitter), "seed": int(seed),
        },
        "var_types": {
            "x": x_type,
            "y": yt,
            "color_by": "categorical" if color_by else None,
            "panel_by": panel_type,
        },
        "x_label": x_axis,
        "y_label": y_axis,
        "x_range": x_range,
        "y_range": y_range,
        "log_scale": bool(log_y),
        "x_categories": x_levels if x_is_cat else None,
        "groups": groups,
        "legend": legend,
        "panels": panel_meta,
        "cells": cells,
        "ghost_line": ghost_line,
        "summary": {
            "n": total_n,
            "x_mean": _round(float(fx.mean())) if fx.size else None,
            "y_mean": _round(float(fy.mean())) if fy.size else None,
            "n_groups": len(groups),
            "n_panels": len(panel_ids),
        },
    }


def _group_color_index(color_by: str | None, groups: list[str], gi: int) -> int:
    """Palette index for a colour series (0 when there is no colour aesthetic)."""
    if not color_by or groups == [""]:
        return 0
    return gi % 10


# --------------------------------------------------------------------------- #
# Variable listing for the picker (metadata-only, privacy-safe)
# --------------------------------------------------------------------------- #
def plottable_variables(metadata: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Derive a typed variable list from dataset *metadata* only — never raw rows.

    Uses the schema built by ``schema_extractor.extract_schema``: a numeric
    ``summary`` carries ``min``/``max`` (and now ``n_unique``); a categorical
    summary carries only ``n_unique``. The picker type therefore matches the
    server's authoritative classification — a numeric column with fewer than
    ``_MIN_CONTINUOUS_LEVELS`` distinct values is reported ``categorical`` (the
    same rule ``_classify_variable`` applies when the plot is built).
    """
    cols = (metadata or {}).get("columns") or []
    out: list[dict[str, Any]] = []
    for c in cols:
        summ = c.get("summary") or {}
        is_numeric = "min" in summ and "max" in summ
        n_unique = summ.get("n_unique")
        if is_numeric and n_unique is not None and n_unique < _MIN_CONTINUOUS_LEVELS:
            vtype = "categorical"
        else:
            vtype = "continuous" if is_numeric else "categorical"
        out.append({
            "name": c.get("name"),
            "dtype": c.get("dtype", ""),
            "role": c.get("role", ""),
            "type": vtype,
            "is_numeric": is_numeric,
            "n_unique": n_unique,
            "min": summ.get("min"),
            "max": summ.get("max"),
        })
    return out
