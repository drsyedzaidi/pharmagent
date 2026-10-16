#!/usr/bin/env python3
"""For the noise-free stress subjects the true AUC(0-inf) is known analytically (Dose / (V * ke)). Print each engine's
AUCinf.obs against it, so a difference between engines can be judged against the truth and not only against each other.

    python3 truth_check.py stress_truth.json out_stress_zero/pharmagent.csv out_stress_zero/numpy_reference.csv \
        [more.csv ...]
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


def load(path: Path) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for r in csv.DictReader(path.open(encoding="utf-8")):
        try:
            out[r["ID"]] = float(r["AUCinf.obs"])
        except (KeyError, ValueError):
            out[r["ID"]] = None
    return out


def table(truth_path: Path, files: list[Path]) -> list[dict]:
    truth = json.loads(truth_path.read_text(encoding="utf-8"))["subjects"]
    loaded = {f.stem if f.stem != "pharmagent" else "PharmAgent": load(f) for f in files}
    rows = []
    for sid, t in truth.items():
        true = t["analytic_auc_inf_noise_free"]
        if t["cv_noise"] != 0 or true is None:
            continue
        row = {"id": sid, "true": true}
        for name, vals in loaded.items():
            v = vals.get(sid)
            row[name] = None if v is None else v
            row[name + "_err_pct"] = None if v is None else 100.0 * (v - true) / true
        rows.append(row)
    return rows


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    rows = table(Path(argv[1]), [Path(a) for a in argv[2:]])
    names = [k for k in rows[0] if k not in ("id", "true") and not k.endswith("_err_pct")] if rows else []
    print(f"{'subject':<28}{'true AUCinf':>12}" + "".join(f"{n:>26}" for n in names))
    for r in rows:
        cells = ""
        for n in names:
            cell = "n/a" if r[n] is None else f"{r[n]:.3f} ({r[n + '_err_pct']:+.1f}%)"
            cells += f"{cell:>26}"
        print(f"{r['id']:<28}{r['true']:>12.3f}{cells}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
