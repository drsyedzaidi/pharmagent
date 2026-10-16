#!/usr/bin/env python3
"""Simulate every scenario with PharmAgent's own ODE simulator (app.compute.pk_simulate.simulate).

    python3 run_pharmagent_sim.py <dir with scenarios.csv/doses.csv/obs.csv> <outdir>      # writes pharmagent_sim.csv

Deterministic; scipy only; no R, no network, no LLM. Unrounded concentrations (the plotting helper rounds them).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND))


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    src, out = Path(argv[1]), Path(argv[2])
    out.mkdir(parents=True, exist_ok=True)
    from app.compute.pk_models import get_model
    from app.compute.pk_simulate import simulate
    scen, doses, obs = read(src / "scenarios.csv"), read(src / "doses.csv"), read(src / "obs.csv")
    rows = []
    for s in scen:
        model = get_model(s["model"])
        params = {k: float(v) for k, v in (kv.split("=") for kv in s["params"].split(";"))}
        d = [{"time": float(r["time"]), "amt": float(r["amt"]), "rate": float(r["rate"])}
             for r in doses if r["id"] == s["id"]]
        times = [float(r["time"]) for r in obs if r["id"] == s["id"]]
        cp = simulate(model, params, d, times, wt=float(s["wt"]))["cp"]
        rows += [{"id": s["id"], "time": repr(t), "cp": repr(float(c))} for t, c in zip(times, cp, strict=True)]
    with (out / "pharmagent_sim.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "time", "cp"], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"{len(scen)} scenarios, {len(rows)} points -> {out / 'pharmagent_sim.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
