#!/usr/bin/env python3
"""Build the population scenarios and the SHARED standard-normal draws both engines consume.

    python3 make_pop.py [outdir]        # writes design.csv, doses.csv, obs.csv, draws.csv

Random numbers cannot be compared across engines, so the draws are fixed here and handed to both: each subject gets one
z per random effect and one z per observation. Everything after that (eta = sqrt(omega2) z, lognormal realisation,
weight scaling, the structural simulation, the residual-error model, the BLQ flag) is deterministic and can be compared.
The z are clipped to +-2.5 and the residual sigmas kept small so no simulated concentration goes negative (PharmAgent
would resample, consuming extra draws); run_pharmagent_pop.py fails loudly if that ever happens.

design.csv  id, model, wt, theta, iiv, omega2, error, sigma_prop, sigma_add, lloq, n_subjects
doses.csv   id, time, amt, rate        obs.csv  id, time        draws.csv  id, subject, kind(eta|eps), idx, z
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

SEED = 20261016
CLIP = 2.5
N_SUBJECTS = 30

# id, model, wt, iiv {param: omega2}, error, sigma_prop, sigma_add, lloq, doses [(t, amt, rate)], obs times
SCENARIOS = [
    ("pop_iv1", "iv_1cmt", 70, {"CL": 0.09, "V": 0.04}, "combined", 0.10, 0.0001, None,
     [(0, 100, 0)], [0.5, 1, 2, 4, 8, 12, 24, 48]),
    ("pop_oral1", "oral_1cmt", 70, {"CL": 0.09, "V": 0.04, "KA": 0.16}, "proportional", 0.15, 0.0, None,
     [(0, 100, 0)], [0.25, 0.5, 1, 2, 4, 8, 12, 24]),
    ("pop_iv2_infusion_100kg", "iv_2cmt", 100, {"CL": 0.09, "VC": 0.04}, "additive", 0.0, 0.0001, None,
     [(0, 100, 25)], [1, 2, 4, 6, 8, 12, 24, 48]),
    ("pop_mm", "iv_1cmt_mm", 70, {"VMAX": 0.09, "V": 0.04}, "combined", 0.10, 0.0001, None,
     [(0, 1000, 0)], [1, 2, 3, 4, 6, 8]),
    ("pop_lag_blq", "oral_1cmt_lag", 70, {"CL": 0.09}, "proportional", 0.15, 0.0, 0.3,
     [(0, 100, 0), (12, 100, 0)], [0.5, 1, 2, 4, 8, 12, 16, 24, 36, 48]),
]


def write(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def build() -> dict[str, list[dict]]:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from app.compute.pk_models import REGISTRY
    rng = np.random.default_rng(SEED)
    design, doses, obs, draws = [], [], [], []
    for sid, model, wt, iiv, err, sp, sa, lloq, dd, tt in SCENARIOS:
        spec = REGISTRY[model]
        theta = ";".join(f"{k}={float(v):g}" for k, v in spec.defaults.items() if k in spec.params)
        design.append({"id": sid, "model": model, "wt": f"{wt:g}", "theta": theta,
                       "iiv": ",".join(iiv), "omega2": ";".join(f"{k}={v:g}" for k, v in iiv.items()),
                       "error": err, "sigma_prop": f"{sp:g}", "sigma_add": f"{sa:g}",
                       "lloq": "" if lloq is None else f"{lloq:g}", "n_subjects": N_SUBJECTS})
        doses += [{"id": sid, "time": f"{t:g}", "amt": f"{a:g}", "rate": f"{r:g}"} for t, a, r in dd]
        obs += [{"id": sid, "time": f"{t:g}"} for t in tt]
        for s in range(1, N_SUBJECTS + 1):
            z = np.clip(rng.standard_normal(len(iiv) + len(tt)), -CLIP, CLIP)
            for i, v in enumerate(z):
                kind, idx = ("eta", i) if i < len(iiv) else ("eps", i - len(iiv))
                draws.append({"id": sid, "subject": s, "kind": kind, "idx": idx, "z": repr(float(v))})
    return {"design": design, "doses": doses, "obs": obs, "draws": draws}


def main(argv: list[str]) -> int:
    out = Path(argv[1]) if len(argv) > 1 else Path(__file__).parent
    t = build()
    write(out / "design.csv", t["design"], ["id", "model", "wt", "theta", "iiv", "omega2", "error", "sigma_prop",
                                              "sigma_add", "lloq", "n_subjects"])
    write(out / "doses.csv", t["doses"], ["id", "time", "amt", "rate"])
    write(out / "obs.csv", t["obs"], ["id", "time"])
    write(out / "draws.csv", t["draws"], ["id", "subject", "kind", "idx", "z"])
    print(f"{len(t['design'])} scenarios x {N_SUBJECTS} subjects, {len(t['draws'])} draws -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
