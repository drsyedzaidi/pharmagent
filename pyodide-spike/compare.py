"""Diff results_cpython.json vs results_pyodide.json and render the parity verdict.

    python pyodide-spike/compare.py

Exit 0 if every metric is within tolerance (GREEN LIGHT for the port), 1 otherwise.
Tolerances widen with optimizer sensitivity: deterministic primitives (micro/nca)
are near-exact; iterative fits (focei/saem) allow a few percent because Pyodide's
older scipy/pandas pins can nudge the optimizer path. A failure here is exactly the
information P0 exists to surface BEFORE any UI work — see docs/WASM_BROWSER_NATIVE_SPEC.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent

# (label, key-path, relative tolerance). Negative ints index into lists.
_METRICS: list[tuple[str, list[Any], float]] = [
    ("micro.expm_trace",     ["micro", "expm_trace"],        1e-9),
    ("micro.ode_y[final]",   ["micro", "ode_y", -1],         1e-6),
    ("micro.t_ppf(.975,8)",  ["micro", "t_ppf_975_8"],       1e-9),
    ("micro.kde_at_0",       ["micro", "kde_at_0"],          1e-6),
    ("nca.clf_geomean",      ["nca", "clf_geomean"],         1e-4),
    ("nca.t_half_median",    ["nca", "t_half_median"],       1e-4),
    ("flex.y_mean",          ["flex", "y_mean"],             1e-4),
    ("flex.fit_y_mid",       ["flex", "fit_y_mid"],          1e-3),
    ("flex.fit_hi_mid",      ["flex", "fit_hi_mid"],         1e-3),
    ("focei.theta.CL",       ["focei", "theta", "CL"],       2e-2),
    ("focei.theta.V",        ["focei", "theta", "V"],        2e-2),
    ("focei.theta.KA",       ["focei", "theta", "KA"],       2e-2),
    ("focei.omega_cv.CL",    ["focei", "omega_cv_pct", "CL"], 5e-2),
    ("focei.sigma.prop",     ["focei", "sigma", "prop"],     5e-2),
    ("focei.ofv",            ["focei", "ofv"],               1e-2),
    ("saem.theta.CL",        ["saem", "theta", "CL"],        3e-2),
    ("saem.theta.V",         ["saem", "theta", "V"],         3e-2),
    ("saem.theta.KA",        ["saem", "theta", "KA"],        3e-2),
]


def _get(d: Any, keypath: list[Any]) -> Any:
    for k in keypath:
        d = d[k]
    return d


def _reldiff(a: float | None, b: float | None) -> float:
    if a is None or b is None:
        return 0.0 if a == b else float("inf")
    scale = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / scale


def main() -> int:
    cp = json.loads((_HERE / "results_cpython.json").read_text())
    py = json.loads((_HERE / "results_pyodide.json").read_text())

    print("=" * 78)
    print("PharmAgent WASM parity — CPython vs Pyodide")
    print("=" * 78)
    for env, r in (("cpython", cp), ("pyodide", py)):
        v = r["versions"]
        print(f"  {env:8s}  numpy {v['numpy']:8s}  scipy {v['scipy']:8s}  "
              f"pandas {v['pandas']:8s}  ({r['mode']})")
    print(f"  converged  cpython {cp['focei']['converged']}/{cp['saem']['converged']}"
          f"   pyodide {py['focei']['converged']}/{py['saem']['converged']}")
    print("-" * 78)
    print(f"  {'metric':22s} {'cpython':>14s} {'pyodide':>14s} {'rel.diff':>11s}  {'tol':>7s}  ok")
    print("-" * 78)

    failures = 0
    for label, keypath, tol in _METRICS:
        try:
            a = _get(cp, keypath)
            b = _get(py, keypath)
        except (KeyError, IndexError, TypeError):
            print(f"  {label:22s} {'MISSING':>14s} {'MISSING':>14s}")
            failures += 1
            continue
        rd = _reldiff(a, b)
        ok = rd <= tol
        failures += 0 if ok else 1
        af = f"{a:.6g}" if isinstance(a, (int, float)) else str(a)
        bf = f"{b:.6g}" if isinstance(b, (int, float)) else str(b)
        print(f"  {label:22s} {af:>14s} {bf:>14s} {rd:>11.2e}  {tol:>7.0e}  "
              f"{'PASS' if ok else 'FAIL'}")

    # convergence must agree and both be True
    conv_ok = (cp["focei"]["converged"] == py["focei"]["converged"] == True  # noqa: E712
               and cp["saem"]["converged"] == py["saem"]["converged"] == True)  # noqa: E712
    if not conv_ok:
        failures += 1

    print("-" * 78)
    if failures == 0 and conv_ok:
        print("  VERDICT: GREEN — numerical parity holds. The port is de-risked.")
        return 0
    print(f"  VERDICT: RED — {failures} metric(s) out of tolerance"
          f"{'' if conv_ok else ' + convergence mismatch'}.")
    print("  Investigate BEFORE any UI work. If only focei/saem drift while micro/nca")
    print("  pass, suspect Pyodide's older scipy/pandas pins nudging the optimizer path.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
