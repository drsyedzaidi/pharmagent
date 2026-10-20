#!/usr/bin/env python3
"""Deterministic synthetic stress table for NCA cross-checks: known generating parameters, awkward BLQ and noise.

    python3 make_stress_table.py [outdir]     # writes stress_zero.csv, stress_dropped.csv, stress_truth.json

One subject per scenario, all dosed 100 mg once. One-compartment oral model C(t) = D*ka/(V*(ka-ke)) *
(exp(-ke*(t-tlag)) - exp(-ka*(t-tlag))), V = 20 L, LLOQ = 0.05. A value below LLOQ is BLQ. Two codings of the SAME data,
because real datasets differ and an NCA tool can treat them differently:
  stress_zero.csv     BLQ values written as DV = 0 with BLQ = 1 (as the pharmaverse pre-dose sample is coded)
  stress_dropped.csv  BLQ rows absent (the value is treated as missing)
Columns: ID, TIME, DV, AMT, EVID, BLQ (same layout as nca_input.csv). The truth file records what generated each
subject and which behaviour the scenario is designed to expose; it does NOT contain expected results.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

SEED, DOSE, V, LLOQ = 20261016, 100.0, 20.0, 0.05
TIMES = [0.0, 0.25, 0.5, 1, 2, 3, 4, 6, 8, 12, 16, 24, 36, 48, 72]
COLUMNS = ["ID", "TIME", "DV", "AMT", "EVID", "BLQ"]

# id: (ka, ke, tlag, cv_noise, design note, post-process)
SCENARIOS: dict[str, dict] = {
    "S01_clean": dict(ka=1.5, ke=0.10, tlag=0.0, cv=0.0, note="noise-free; terminal slope = ke exactly"),
    "S02_noisy_terminal": dict(ka=1.5, ke=0.10, tlag=0.0, cv=0.25,
                               note="25% lognormal noise; the terminal window is ambiguous"),
    "S03_trailing_blq": dict(ka=1.5, ke=0.25, tlag=0.0, cv=0.0, note="late samples fall below LLOQ: trailing BLQ"),
    "S04_trailing_blq_noisy": dict(ka=1.5, ke=0.25, tlag=0.0, cv=0.20, note="trailing BLQ plus noise"),
    "S05_middle_blq": dict(ka=1.5, ke=0.08, tlag=0.0, cv=0.0,
                           note="one assay failure at 16 h between quantifiable values", force_blq=[16.0]),
    "S06_leading_blq": dict(ka=1.2, ke=0.10, tlag=2.5, cv=0.0, note="absorption lag: BLQ before the first rise"),
    "S07_tmax_last": dict(ka=0.03, ke=0.30, tlag=0.0, cv=0.0, note="Tmax at the last sample: no terminal phase to fit"),
    "S08_two_points_after_tmax": dict(ka=0.35, ke=0.30, tlag=0.0, cv=0.0, note="only two samples after Tmax before BLQ",
                                      force_blq=[16.0, 24.0, 36.0, 48.0, 72.0]),
    "S09_all_blq": dict(ka=1.5, ke=0.10, tlag=0.0, cv=0.0, note="every sample below LLOQ", scale=1e-4),
    "S10_long_halflife": dict(ka=2.0, ke=0.02, tlag=0.0, cv=0.15, note="long half-life, noisy, many terminal points"),
    "S11_terminal_bump": dict(ka=1.5, ke=0.12, tlag=0.0, cv=0.05,
                              note="secondary peak at 24 h (re-absorption)", bump=True),
}


def profile(rng: np.random.Generator, p: dict) -> list[tuple[float, float]]:
    ka, ke, tlag, cv = p["ka"], p["ke"], p["tlag"], p["cv"]
    out = []
    for t in TIMES:
        te = t - tlag
        c = 0.0 if te <= 0 else DOSE * ka / (V * (ka - ke)) * (np.exp(-ke * te) - np.exp(-ka * te))
        if p.get("bump") and t >= 24:
            c += 0.35 * np.exp(-0.08 * (t - 24))
        c *= p.get("scale", 1.0)
        if cv and c > 0:
            c *= float(np.exp(rng.normal(0.0, np.sqrt(np.log(1 + cv * cv)))))
        out.append((t, c))
    return out


def build() -> tuple[list[dict], list[dict], dict]:
    rng = np.random.default_rng(SEED)
    zero, dropped, truth = [], [], {"seed": SEED, "dose": DOSE, "V": V, "LLOQ": LLOQ, "times": TIMES, "subjects": {}}
    for sid, p in SCENARIOS.items():
        prof = profile(rng, p)
        dose_row = {"ID": sid, "TIME": "0", "DV": "", "AMT": f"{DOSE:g}", "EVID": "1", "BLQ": "0"}
        zero.append(dose_row)
        dropped.append(dose_row)
        n_blq = 0
        for t, c in prof:
            blq = c < LLOQ or t in p.get("force_blq", [])
            n_blq += blq
            base = {"ID": sid, "TIME": f"{t:g}", "AMT": "0", "EVID": "0"}
            if blq:
                zero.append({**base, "DV": "0", "BLQ": "1"})
            else:
                row = {**base, "DV": f"{c:.6g}", "BLQ": "0"}
                zero.append(row)
                dropped.append(row)
        truth["subjects"][sid] = {"ka": p["ka"], "ke": p["ke"], "tlag": p["tlag"], "cv_noise": p["cv"],
                                  "design": p["note"], "n_obs": len(prof), "n_blq": int(n_blq),
                                  "analytic_auc_inf_noise_free": DOSE / (V * p["ke"]) if "scale" not in p else None}
    return zero, dropped, truth


def write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main(argv: list[str]) -> int:
    out = Path(argv[1]) if len(argv) > 1 else Path(__file__).resolve().parent
    zero, dropped, truth = build()
    write(out / "stress_zero.csv", zero)
    write(out / "stress_dropped.csv", dropped)
    (out / "stress_truth.json").write_text(json.dumps(truth, indent=1) + "\n", encoding="utf-8")
    print(f"{len(SCENARIOS)} scenarios; zero-coded {len(zero)} rows, dropped {len(dropped)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
