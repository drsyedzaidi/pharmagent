#!/usr/bin/env python3
"""Compare two simulation files (id, time, cp), scenario by scenario. Reports; never decides what a gap means.

    python3 compare_sims.py out/pharmagent_sim.csv out/mrgsolve_sim.csv [--tol 1e-5]

Files carry id, time, cp and (PK/PD scenarios) eff. The concentration is always compared; the response (eff) is
compared as well, as a second block, when either file holds it. Per scenario: the normalised maximum absolute
difference max|a-b| / max|b| over the profile, the relative difference in Cmax, and the relative difference in AUC
(trapezoid on the shared grid). A scenario is OUT when any exceeds --tol.
Exit 0 all within tolerance, 1 at least one scenario out, 2 an input is missing, unreadable or does not line up.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

DEFAULT_TOL = 1e-5


def load(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    by: dict[str, list[tuple[float, float]]] = {}
    with path.open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            by.setdefault(r["id"], []).append((float(r["time"]), float(r["cp"])))
    return {k: (np.array([t for t, _ in v]), np.array([c for _, c in v])) for k, v in by.items()}


def load_eff(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """The response series (scenarios whose eff column is filled); empty for a PK-only file."""
    by: dict[str, list[tuple[float, float]]] = {}
    with path.open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("eff") not in (None, ""):
                by.setdefault(r["id"], []).append((float(r["time"]), float(r["eff"])))
    return {k: (np.array([t for t, _ in v]), np.array([c for _, c in v])) for k, v in by.items()}


def compare(mine: dict, ref: dict, tol: float = DEFAULT_TOL) -> dict:
    rows = []
    for sid in sorted(set(mine) & set(ref)):
        (ta, ca), (tb, cb) = mine[sid], ref[sid]
        if ta.shape != tb.shape or not np.allclose(ta, tb, rtol=0, atol=1e-6):
            rows.append({"id": sid, "status": "GRID_MISMATCH", "nmax": None, "cmax": None, "auc": None})
            continue
        scale = float(np.max(np.abs(cb))) or 1.0
        nmax = float(np.max(np.abs(ca - cb)) / scale)
        cmax = abs(float(ca.max()) - float(cb.max())) / (abs(float(cb.max())) or 1.0)
        auc_a, auc_b = float(np.trapezoid(ca, ta)), float(np.trapezoid(cb, tb))
        auc = abs(auc_a - auc_b) / (abs(auc_b) or 1.0)
        status = "match" if max(nmax, cmax, auc) <= tol else "OUT"
        rows.append({"id": sid, "status": status, "nmax": nmax, "cmax": cmax, "auc": auc})
    return {"rows": rows, "only_mine": sorted(set(mine) - set(ref)), "only_ref": sorted(set(ref) - set(mine))}


def report(res: dict, tol: float) -> tuple[str, bool]:
    rows = res["rows"]
    bad = [r for r in rows if r["status"] != "match"]
    lines = [f"{len(rows)} scenarios compared at tolerance {tol:g}: {len(rows) - len(bad)} match, {len(bad)} not"]
    ok = [r for r in rows if r["nmax"] is not None]
    if ok:
        w = max(ok, key=lambda r: r["nmax"])
        lines.append(f"worst normalised max difference {w['nmax']:.3g} ({w['id']}); "
                     f"worst Cmax {max(r['cmax'] for r in ok):.3g}; worst AUC {max(r['auc'] for r in ok):.3g}")
    for r in sorted(bad, key=lambda r: -(r["nmax"] or 9e9))[:15]:
        lines.append(f"  {r['status']:<13} {r['id']:<28} normalised {r['nmax']}  Cmax {r['cmax']}  AUC {r['auc']}")
    if res["only_mine"] or res["only_ref"]:
        lines.append(f"only in PharmAgent: {res['only_mine']}; only in the reference: {res['only_ref']}")
    return "\n".join(lines), not bad and not res["only_mine"] and not res["only_ref"]


def main(argv: list[str]) -> int:
    args = [a for a in argv[1:] if not a.startswith("--tol")]
    tol = DEFAULT_TOL
    for i, a in enumerate(argv[1:], 1):
        if a == "--tol" and i + 1 < len(argv):
            tol = float(argv[i + 1])
            args = [x for x in args if x != argv[i + 1]]
    if len(args) != 2:
        print(__doc__)
        return 2
    paths = [Path(a) for a in args]
    for p in paths:
        if not p.exists():
            print(f"NOT PRODUCED YET: {p}")
            return 2
    try:
        text, ok = report(compare(load(paths[0]), load(paths[1]), tol), tol)
        eff_a, eff_b = load_eff(paths[0]), load_eff(paths[1])
        if eff_a or eff_b:
            text_e, ok_e = report(compare(eff_a, eff_b, tol), tol)
            text, ok = f"CONCENTRATION\n{text}\n\nRESPONSE (eff)\n{text_e}", ok and ok_e
    except (OSError, KeyError, ValueError) as exc:
        print(f"cannot read the inputs: {exc}")
        return 2
    print(text)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
