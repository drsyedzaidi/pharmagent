"""Deterministic datasets for the R cross-check of the 2x2 crossover model and the Welch parallel estimator.

Crossover files: subject,sequence,period,treatment,Cmax (two rows per subject). Generative model on the log scale:
subject effect + period effect + formulation effect + residual; sequence sizes are deliberately unbalanced in most
cases and a period effect is present, which is exactly where the paired t-test is wrong and the 2x2 model is not.
Run:  python make_fixtures.py   (rewrites the CSVs in this folder)"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# case, n_TR, n_RT, seed, formulation effect, period effect, sd between, sd within
CROSSOVER = [
    ("xo_balanced_12", 6, 6, 101, -0.04, 0.10, 0.30, 0.12),
    ("xo_unbalanced_7_5", 7, 5, 102, 0.06, 0.25, 0.35, 0.15),
    ("xo_unbalanced_13_11", 13, 11, 103, -0.10, 0.15, 0.40, 0.20),
    ("xo_extreme_4_20", 4, 20, 104, 0.02, 0.30, 0.30, 0.10),
    ("xo_strong_period_8_4", 8, 4, 105, -0.04, 0.80, 0.30, 0.12),
    ("xo_large_36_36", 36, 36, 106, -0.02, 0.05, 0.45, 0.25),
]
# case, n_test, n_ref, seed, sd_test, sd_ref, mean shift
PARALLEL = [
    ("par_unequal_var_14_11", 14, 11, 201, 0.30, 0.50, 0.05),
    ("par_unequal_n_24_9", 24, 9, 202, 0.25, 0.40, -0.03),
]


def write_crossover() -> list[tuple[str, str]]:
    cases = []
    for name, n_tr, n_rt, seed, form, per, sd_b, sd_w in CROSSOVER:
        rng = np.random.default_rng(seed)
        n = n_tr + n_rt
        tf = [True] * n_tr + [False] * n_rt
        subj = rng.normal(0.0, sd_b, n)
        rows = []
        for i in range(n):
            for period in (1, 2):
                is_test = (period == 1) == tf[i]
                y = 4.0 + subj[i] + (per / 2 if period == 2 else -per / 2) + (form / 2 if is_test else -form / 2)
                y += rng.normal(0.0, sd_w)
                rows.append({"subject": f"S{i + 1:02d}", "sequence": "TR" if tf[i] else "RT", "period": period,
                             "treatment": "T" if is_test else "R", "Cmax": repr(float(np.exp(y)))})
        with (HERE / f"{name}.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["subject", "sequence", "period", "treatment", "Cmax"])
            w.writeheader()
            w.writerows(rows)
        cases.append((name, "crossover"))
    return cases


def write_parallel() -> list[tuple[str, str]]:
    cases = []
    for name, n1, n2, seed, sd1, sd2, shift in PARALLEL:
        rng = np.random.default_rng(seed)
        t = np.exp(rng.normal(4.0 + shift, sd1, n1))
        r = np.exp(rng.normal(4.0, sd2, n2))
        with (HERE / f"{name}.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["subject", "treatment", "Cmax"])
            for i, v in enumerate(t):
                w.writerow([f"T{i + 1:02d}", "T", repr(float(v))])
            for i, v in enumerate(r):
                w.writerow([f"R{i + 1:02d}", "R", repr(float(v))])
        cases.append((name, "parallel"))
    return cases


if __name__ == "__main__":
    cases = write_crossover() + write_parallel()
    with (HERE / "cases.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["case", "design"])
        w.writerows(cases)
    print(f"wrote {len(cases)} cases")
