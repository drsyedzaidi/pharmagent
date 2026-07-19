"""Shared numerical-parity harness — the P0 de-risk for the browser-native port.

The SAME analysis code runs under both CPython (the shipping backend interpreter)
and Pyodide/WASM; ``compare.py`` then diffs the two output JSONs to prove — not
assume — that PharmAgent's compute reproduces identical numbers in the browser.
See ``docs/WASM_BROWSER_NATIVE_SPEC.md`` §7 (P0) and §6 (numerical-parity risk).

It imports the EXACT shipping modules (``app.compute.nlme``,
``app.tools.pkmodel_tools``, ``app.compute.flexplot``) so parity is tested on the
real validated path — the same one ``tests/reference/test_theophylline.py`` drives
— not on a re-implementation. Layers, in increasing sensitivity:

    micro : scipy.linalg.expm + integrate.solve_ivp + stats (raw float/BLAS parity)
    nca   : deterministic NCA CL/F + t1/2 on the Theophylline cohort
    flex  : flexplot loess geometry (custom loess + t-quantile CI + gaussian_kde)
    focei : the validated FOCE-I population fit  (iterative optimizer path parity)
    saem  : the validated seeded SAEM population fit (deterministic MCMC parity)

If ``micro`` matches but ``focei``/``saem`` drift, the cause is optimizer-path
sensitivity (likely Pyodide's older scipy/pandas pins); if ``micro`` drifts, the
cause is fundamental WASM float/BLAS divergence. That separation is the point.
"""
from __future__ import annotations

import math
import platform
from typing import Any

import numpy as np
import pandas as pd
import scipy
from scipy import stats
from scipy.integrate import solve_ivp
from scipy.linalg import expm

# FOCE-I / SAEM iteration counts. "full" mirrors the reference suite
# (test_theophylline.py: focei max_iter=40, saem max_iter=120); "quick" is a
# faster smoke that still exercises the whole path.
_ITERS = {"full": (40, 120), "quick": (15, 40)}
_SAEM_SEED = 20250614  # matches the reference suite


def _versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
    }


def _micro() -> dict[str, Any]:
    """Deterministic, optimizer-free parity probes for the numerically-sensitive
    primitives PharmAgent leans on (matrix exponential, stiff ODE, t-quantile,
    KDE). No RNG except numpy's PCG64, whose stream is stable across versions."""
    a = np.array([[-0.3, 0.1, 0.0], [0.1, -0.4, 0.2], [0.0, 0.2, -0.5]])
    e = expm(a)
    sol = solve_ivp(lambda t, y: -0.5 * y, (0.0, 10.0), [10.0],
                    t_eval=[2.5, 5.0, 10.0], rtol=1e-9, atol=1e-12)
    sample = np.random.default_rng(0).normal(0.0, 1.0, 500)
    kde = stats.gaussian_kde(sample)
    return {
        "expm_trace": float(np.trace(e)),
        "expm_flat": [float(v) for v in e.ravel()],
        "ode_y": [float(v) for v in sol.y[0]],
        "t_ppf_975_8": float(stats.t.ppf(0.975, 8)),
        "kde_at_0": float(kde(0.0)[0]),
    }


def _subjects(csv_path: str) -> list[dict[str, Any]]:
    """Build the 12-subject Theophylline cohort via the SHIPPING code path."""
    from app.tools.pkmodel_tools import _build_subjects, _roles

    df = pd.read_csv(csv_path)
    state = type("_S", (), {"dataset_metadata": None})()
    roles = _roles(df, state)
    subs, _multi, _pd = _build_subjects(df, roles)
    return subs


def _nca(subjects: list[dict[str, Any]]) -> dict[str, Any]:
    """NCA CL/F = Dose/AUC_inf and terminal t1/2 (the reference computation)."""
    clf: list[float] = []
    thalf: list[float] = []
    for s in subjects:
        t = np.asarray(s["obs_t"], float)
        c = np.asarray(s["obs_c"], float)
        dose = float(s["doses"][-1]["amt"])
        auc = float(np.trapezoid(c, t))
        slope = float(np.polyfit(t[-3:], np.log(c[-3:]), 1)[0])
        ke = -slope
        if ke > 0:
            clf.append(dose / (auc + c[-1] / ke))
            thalf.append(math.log(2) / ke)
    return {
        "clf_geomean": float(np.exp(np.mean(np.log(clf)))),
        "t_half_median": float(np.median(thalf)),
        "n": len(clf),
    }


def _flex(csv_path: str) -> dict[str, Any]:
    """Flexplot DV~TIME scatter — exercises the custom loess + t-quantile CI +
    (indirectly) the scipy.stats path added for the visualization feature."""
    from app.compute.flexplot import flexplot

    df = pd.read_csv(csv_path)
    payload = flexplot(df, y="DV", x="TIME", fit="loess")
    fit = payload["cells"][0]["fit"]
    mid = len(fit["y"]) // 2
    return {
        "n": payload["summary"]["n"],
        "y_mean": payload["summary"]["y_mean"],
        "fit_len": len(fit["y"]),
        "fit_y_mid": fit["y"][mid],
        "fit_lo_mid": fit["lo"][mid],
        "fit_hi_mid": fit["hi"][mid],
    }


def _fit(subjects: list[dict[str, Any]], method: str, max_iter: int,
         seed: int | None = None) -> dict[str, Any]:
    from app.compute.nlme import population_fit

    kw: dict[str, Any] = {"method": method, "max_iter": max_iter,
                          "compute_uncertainty": False}
    if seed is not None:
        kw["seed"] = seed
    r = population_fit("oral_1cmt", subjects, **kw)
    return {
        "theta": {k: float(v) for k, v in r["theta"].items()},
        "omega_cv_pct": {k: float(v) for k, v in r["omega_cv_pct"].items()},
        "sigma": {k: (float(v) if v is not None else None) for k, v in r["sigma"].items()},
        "ofv": float(r["ofv"]),
        "converged": bool(r["converged"]),
    }


def run(csv_path: str, quick: bool = False) -> dict[str, Any]:
    """Run every parity layer and return one JSON-serializable result dict."""
    subjects = _subjects(csv_path)
    foce_iter, saem_iter = _ITERS["quick" if quick else "full"]
    return {
        "mode": "quick" if quick else "full",
        "n_subjects": len(subjects),
        "versions": _versions(),
        "micro": _micro(),
        "nca": _nca(subjects),
        "flex": _flex(csv_path),
        "focei": _fit(subjects, "focei", foce_iter),
        "saem": _fit(subjects, "saem", saem_iter, seed=_SAEM_SEED),
    }
