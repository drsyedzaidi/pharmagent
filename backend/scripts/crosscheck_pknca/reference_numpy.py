#!/usr/bin/env python3
"""Textbook NCA re-implementation (NumPy only, imports nothing from the app). A PRE-FLIGHT reference, not an
independent engine: it follows the same stated conventions as PharmAgent (linear-up/log-down, terminal slope = the
window of >= 3 points after Tmax ending at the last measurable point with the strictly largest adjusted R^2), so
agreement shows PharmAgent implements its own stated method, and says nothing about whether the method matches
PKNCA's. PKNCA (run_pknca.R) is the independent reference.

    python3 reference_numpy.py nca_input.csv outdir        # writes outdir/numpy_reference.csv
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np

FIELDS = ["ID", "Cmax", "Tmax", "AUClast", "AUCinf.obs", "lambda.z", "half.life", "lambda.z.n.points",
          "adj.r.squared", "tlast", "clast.obs"]


def auc_to(t: np.ndarray, c: np.ndarray, end: int) -> float:
    total = 0.0
    for i in range(end):
        dt = t[i + 1] - t[i]
        if c[i + 1] < c[i] and c[i + 1] > 0:
            total += dt * (c[i] - c[i + 1]) / math.log(c[i] / c[i + 1])
        else:
            total += dt * (c[i] + c[i + 1]) / 2.0
    return total


def best_terminal(t: np.ndarray, c: np.ndarray, imax: int, ilast: int):
    idx = [i for i in range(imax + 1, ilast + 1) if c[i] > 0]
    best = None
    for s in range(len(idx) - 2):
        win = idx[s:]
        x, y = t[win], np.log(c[win])
        slope, icpt = np.polyfit(x, y, 1)
        if slope >= 0:
            continue
        res = float(np.sum((y - (slope * x + icpt)) ** 2))
        tot = float(np.sum((y - y.mean()) ** 2))
        k = len(win)
        r2 = 1 - res / tot if tot > 0 else 0.0
        adj = 1 - (1 - r2) * (k - 1) / (k - 2)
        if best is None or adj > best[1]:
            best = (-slope, adj, k)
    return best


def subject(sid: str, t: np.ndarray, c: np.ndarray) -> dict:
    order = np.argsort(t, kind="stable")
    t, c = t[order], c[order]
    imax = int(np.argmax(c))
    pos = np.where(c > 0)[0]
    if pos.size == 0:                        # nothing measurable: Cmax/Tmax of the zeros, no AUC, no terminal phase
        return {"ID": sid, "Cmax": c[imax], "Tmax": t[imax]}
    ilast = int(pos[-1])
    auclast = auc_to(t, c, ilast)
    out = {"ID": sid, "Cmax": c[imax], "Tmax": t[imax], "AUClast": auclast, "tlast": t[ilast], "clast.obs": c[ilast]}
    tz = best_terminal(t, c, imax, ilast)
    if tz:
        lam, adj, k = tz
        out.update({"AUCinf.obs": auclast + c[ilast] / lam, "lambda.z": lam, "half.life": math.log(2) / lam,
                    "lambda.z.n.points": k, "adj.r.squared": adj})
    return out


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    rows = [r for r in csv.DictReader(open(argv[1], encoding="utf-8")) if r["EVID"] == "0"]
    ids = sorted({r["ID"] for r in rows})
    out = Path(argv[2])
    out.mkdir(parents=True, exist_ok=True)
    with (out / "numpy_reference.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        for sid in ids:
            mine = [r for r in rows if r["ID"] == sid]
            res = subject(sid, np.array([float(r["TIME"]) for r in mine]), np.array([float(r["DV"]) for r in mine]))
            w.writerow({k: ("" if res.get(k) is None else res[k]) for k in FIELDS})
    print(f"{len(ids)} subjects -> {out / 'numpy_reference.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
