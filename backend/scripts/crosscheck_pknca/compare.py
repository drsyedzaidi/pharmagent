#!/usr/bin/env python3
"""Compare PharmAgent's NCA with one or more reference result files, subject by subject, and say plainly where
they differ. It never decides what a difference MEANS: it reports it, and the tolerances are the stated ones below.

    python3 compare.py out/pharmagent.csv out/numpy_reference.csv out/pknca_default.csv out/pknca_matched.csv

Exit 0: every compared parameter is within tolerance for every reference. 1: at least one is not.
2: a file is missing/unreadable (a reference you have not produced yet is reported, not guessed).
Tolerances: values PharmAgent rounds to 6 decimals, so rel 1e-5 (+abs 2e-6) for Cmax, Tmax, AUClast, AUCinf;
rel 1e-4 for lambda.z and half.life; exact for the number of terminal points; abs 1e-4 for adjusted R^2.
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

PARAMS = {                    # name: (kind, tolerance)
    "Cmax": ("rel", 1e-5), "Tmax": ("rel", 1e-5), "AUClast": ("rel", 1e-5), "AUCinf.obs": ("rel", 1e-5),
    "lambda.z": ("rel", 1e-4), "half.life": ("rel", 1e-4), "lambda.z.n.points": ("exact", 0.0),
    "adj.r.squared": ("abs", 1e-4),
}
ABS_FLOOR = 2e-6


def load(path: Path) -> dict[str, dict[str, float | None]]:
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    out: dict[str, dict[str, float | None]] = {}
    for r in rows:
        vals: dict[str, float | None] = {}
        for k, v in r.items():
            if k == "ID":
                continue
            try:
                f = float(v)
                vals[k] = None if math.isnan(f) else f
            except (TypeError, ValueError):
                vals[k] = None
        out[r["ID"]] = vals
    return out


def within(kind: str, tol: float, a: float, b: float) -> bool:
    if kind == "exact":
        return a == b
    if kind == "abs":
        return abs(a - b) <= tol
    return abs(a - b) <= tol * abs(b) + ABS_FLOOR


def compare(mine: dict, ref: dict) -> dict:
    """Return {'rows': [...], 'only_mine': [...], 'only_ref': [...]} for one reference."""
    rows = []
    for sid in sorted(set(mine) & set(ref)):
        for p, (kind, tol) in PARAMS.items():
            a, b = mine[sid].get(p), ref[sid].get(p)
            if a is None and b is None:
                status, rel = "both_missing", None
            elif a is None or b is None:
                status, rel = "one_missing", None
            else:
                status = "match" if within(kind, tol, a, b) else "DIFFERS"
                rel = abs(a - b) / abs(b) if b else abs(a - b)
            rows.append({"id": sid, "param": p, "pharmagent": a, "reference": b, "status": status, "rel_diff": rel})
    return {"rows": rows, "only_mine": sorted(set(mine) - set(ref)), "only_ref": sorted(set(ref) - set(mine))}


def summarize(name: str, res: dict) -> tuple[str, bool]:
    rows = res["rows"]
    bad = [r for r in rows if r["status"] in ("DIFFERS", "one_missing")]
    lines = [f"## PharmAgent vs {name}", f"- compared {len(rows)} values over {len({r['id'] for r in rows})} subjects; "
             f"{len(bad)} outside tolerance"]
    for p in PARAMS:
        sub = [r for r in rows if r["param"] == p and r["rel_diff"] is not None]
        if sub:
            worst = max(sub, key=lambda r: r["rel_diff"])
            ok = sum(r["status"] == "match" for r in sub)
            lines.append(f"- {p}: {ok}/{len(sub)} within tolerance, worst relative difference {worst['rel_diff']:.3g}"
                         f" ({worst['id']})")
    if res["only_mine"] or res["only_ref"]:
        lines.append(f"- subjects only in PharmAgent: {res['only_mine']}; only in the reference: {res['only_ref']}")
    for r in bad[:25]:
        lines.append(f"  - {r['id']} {r['param']}: PharmAgent {r['pharmagent']} vs {name} {r['reference']}")
    if len(bad) > 25:
        lines.append(f"  - ... and {len(bad) - 25} more")
    return "\n".join(lines), not bad and not res["only_mine"] and not res["only_ref"]


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    paths = [Path(a) for a in argv[1:]]
    missing = [str(p) for p in paths if not p.exists()]
    for m in missing:
        print(f"NOT PRODUCED YET: {m}")
    if paths[0] in [Path(m) for m in missing]:
        return 2
    mine = load(paths[0])
    clean = True
    produced = 0
    for p in paths[1:]:
        if not p.exists():
            continue
        produced += 1
        text, ok = summarize(p.stem, compare(mine, load(p)))
        print(text + "\n")
        clean = clean and ok
    if produced == 0:
        return 2
    return 0 if clean and not missing else (1 if not clean else 2)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
