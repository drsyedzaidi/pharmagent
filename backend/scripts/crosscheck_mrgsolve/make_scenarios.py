#!/usr/bin/env python3
"""Build the scenario tables both engines simulate. Deterministic; the only inputs are PharmAgent's model defaults.

    python3 make_scenarios.py [outdir] [pd]   # writes scenarios.csv, doses.csv, obs.csv (PK set, or the PK/PD set)

scenarios.csv  id, model, wt, params        params = "CL=5;V=50;KA=1" (the UNSCALED structural values)
doses.csv      id, time, amt, rate          one row per dose (rate > 0 = zero-order infusion of amt/rate hours)
obs.csv        id, time                     the observation grid (shared: both engines use identical times)

Design (ten structural PK models x dosing situations):
  single    : one dose of 100 at t=0, 48 h
  multiple  : 100 every 12 h x 5, 72 h
  weight    : single dose in a 100 kg subject (allometric scaling, centred at 70 kg)
  infusion  : IV models only, 100 over 4 h (rate 25), 48 h
  saturate  : Michaelis-Menten models only, a 2000 dose so elimination saturates, 96 h
PK/PD set ("pd", the nine PK/PD models, written to pd/): single (100, 96 h), multiple (100 every 24 h x 5, 168 h),
weight (100 kg, 96 h) and high (1000, 96 h, response near its maximum).
The observation grid is PharmAgent's own: 160 points over 0..tmax joined with the dose times.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND))

PD_MODELS = ["pkpd_direct_linear", "pkpd_direct_emax", "pkpd_direct_sigmoid", "pkpd_effect_emax",
             "pkpd_idr1_inhib_kin", "pkpd_idr2_inhib_kout", "pkpd_idr3_stim_kin", "pkpd_idr4_stim_kout",
             "pkpd_transit_lifespan"]
MODELS = ["iv_1cmt", "iv_2cmt", "iv_3cmt", "oral_1cmt", "oral_1cmt_lag", "oral_2cmt", "oral_1cmt_transit",
          "iv_1cmt_mm", "oral_1cmt_mm", "iv_1cmt_mixed"]
MM = {"iv_1cmt_mm", "oral_1cmt_mm", "iv_1cmt_mixed"}


def grid(tmax: float, dose_times: list[float], n_points: int = 160) -> list[float]:
    return sorted(set(np.linspace(0.0, tmax, n_points).tolist() + dose_times))


def build(kind: str = "pk") -> tuple[list[dict], list[dict], list[dict]]:
    from app.compute.pk_models import REGISTRY
    scen, doses, obs = [], [], []

    def add(sid: str, model: str, wt: float, regimen: list[tuple[float, float, float]], tmax: float) -> None:
        spec = REGISTRY[model]
        params = ";".join(f"{k}={float(v):g}" for k, v in spec.defaults.items() if k in spec.params)
        scen.append({"id": sid, "model": model, "wt": f"{wt:g}", "params": params})
        for t, amt, rate in regimen:
            doses.append({"id": sid, "time": f"{t:g}", "amt": f"{amt:g}", "rate": f"{rate:g}"})
        for t in grid(tmax, [r[0] for r in regimen]):
            obs.append({"id": sid, "time": repr(round(t, 6))})

    if kind == "pd":
        for m in PD_MODELS:                       # response models need a longer horizon than concentration
            add(f"{m}:single", m, 70, [(0, 100, 0)], 96)
            add(f"{m}:multiple", m, 70, [(24 * k, 100, 0) for k in range(5)], 168)
            add(f"{m}:weight", m, 100, [(0, 100, 0)], 96)
            add(f"{m}:high", m, 70, [(0, 1000, 0)], 96)       # drives the response toward its maximum
        return scen, doses, obs
    for m in MODELS:
        iv = REGISTRY[m].is_iv
        add(f"{m}:single", m, 70, [(0, 100, 0)], 48)
        add(f"{m}:multiple", m, 70, [(12 * k, 100, 0) for k in range(5)], 72)
        add(f"{m}:weight", m, 100, [(0, 100, 0)], 48)
        if iv:
            add(f"{m}:infusion", m, 70, [(0, 100, 25)], 48)
        if m in MM:
            add(f"{m}:saturate", m, 70, [(0, 2000, 0)], 96)
    return scen, doses, obs


def write(rows: list[dict], path: Path, fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main(argv: list[str]) -> int:
    kind = "pd" if "pd" in argv[1:] else "pk"
    pos = [a for a in argv[1:] if a != "pd"]
    out = Path(pos[0]) if pos else (Path(__file__).parent / ("pd" if kind == "pd" else ""))
    out.mkdir(parents=True, exist_ok=True)
    scen, doses, obs = build(kind)
    write(scen, out / "scenarios.csv", ["id", "model", "wt", "params"])
    write(doses, out / "doses.csv", ["id", "time", "amt", "rate"])
    write(obs, out / "obs.csv", ["id", "time"])
    print(f"{len(scen)} scenarios, {len(doses)} dose rows, {len(obs)} observation rows -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
