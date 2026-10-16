#!/usr/bin/env python3
"""Compare PharmAgent's and mrgsolve's population simulations on the SHARED draws.

    python3 compare_pop.py <pharmagent dir> <mrgsolve dir>

Reads pharmagent_pop.csv / pharmagent_eta.csv and mrgsolve_pop.csv / mrgsolve_eta.csv. Stated tolerances (set before the
R run, from the structural cross-check's ~1e-6 agreement):
  eta    absolute 1e-12 (sqrt(omega2) * z is plain arithmetic in both engines)
  ipred  relative 1e-5   dv  relative 1e-5 (dv = ipred + sd*z carries the ipred difference, scaled by 1 + 0.1*z <= 1.25)
  blq    exact; a flag may differ only where |dv - lloq| / lloq < 1e-5 (a value sitting on the threshold), counted apart
Exit 0 all inside, 1 outside, 2 a reference file is missing.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ETA_TOL, REL_TOL, BLQ_EDGE = 1e-12, 1e-5, 1e-5


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-300)


def compare(pa_pop: list[dict], mr_pop: list[dict], pa_eta: list[dict], mr_eta: list[dict],
            lloq: dict[str, float] | None = None) -> dict:
    lloq = lloq or {}
    out = {"n_obs": 0, "n_eta": 0, "max_eta_abs": 0.0, "max_ipred_rel": 0.0, "max_dv_rel": 0.0,
           "blq_flips": 0, "blq_edge": 0, "worst": {}}
    if len(pa_pop) != len(mr_pop) or len(pa_eta) != len(mr_eta):
        raise ValueError(f"row counts differ: pop {len(pa_pop)}/{len(mr_pop)}, eta {len(pa_eta)}/{len(mr_eta)}")
    for a, b in zip(pa_eta, mr_eta, strict=True):
        if (a["id"], a["subject"], a["param"]) != (b["id"], b["subject"], b["param"]):
            raise ValueError(f"eta rows misaligned at {a} vs {b}")
        d = abs(float(a["eta"]) - float(b["eta"]))
        out["n_eta"] += 1
        out["max_eta_abs"] = max(out["max_eta_abs"], d)
    for a, b in zip(pa_pop, mr_pop, strict=True):
        if (a["id"], a["subject"]) != (b["id"], b["subject"]) or abs(float(a["time"]) - float(b["time"])) > 1e-9:
            raise ValueError(f"observation rows misaligned at {a} vs {b}")
        out["n_obs"] += 1
        ri, rd = rel(float(a["ipred"]), float(b["ipred"])), rel(float(a["dv"]), float(b["dv"]))
        for key, v in (("max_ipred_rel", ri), ("max_dv_rel", rd)):
            if v > out[key]:
                out[key] = v
                out["worst"][key] = f'{a["id"]} subject {a["subject"]} t={a["time"]}'
        if a["blq"] != b["blq"]:
            lq = lloq.get(a["id"])
            if lq and rel(float(b["dv"]), lq) < BLQ_EDGE:
                out["blq_edge"] += 1
            else:
                out["blq_flips"] += 1
    return out


def passed(r: dict) -> bool:
    return r["max_eta_abs"] <= ETA_TOL and r["max_ipred_rel"] <= REL_TOL and r["max_dv_rel"] <= REL_TOL \
        and r["blq_flips"] == 0


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    pa, mr = Path(argv[1]), Path(argv[2])
    files = [pa / "pharmagent_pop.csv", pa / "pharmagent_eta.csv", mr / "mrgsolve_pop.csv", mr / "mrgsolve_eta.csv"]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        print("reference not produced:", *missing, sep="\n  ")
        return 2
    design = load_rows(Path(__file__).parent / "design.csv")
    lloq = {d["id"]: float(d["lloq"]) for d in design if d["lloq"]}
    r = compare(*(load_rows(f) for f in files), lloq=lloq)
    print(f"{r['n_obs']} observations, {r['n_eta']} etas")
    print(f"  eta   max abs  {r['max_eta_abs']:.2e}   (tol {ETA_TOL:.0e})")
    w = r["worst"]
    print(f"  ipred max rel  {r['max_ipred_rel']:.2e}   (tol {REL_TOL:.0e})  worst: {w.get('max_ipred_rel', '-')}")
    print(f"  dv    max rel  {r['max_dv_rel']:.2e}   (tol {REL_TOL:.0e})  worst: {w.get('max_dv_rel', '-')}")
    print(f"  BLQ flips {r['blq_flips']} (edge-of-threshold, excused: {r['blq_edge']})")
    ok = passed(r)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
