#!/usr/bin/env python3
"""Run PharmAgent's REAL population-simulation function on the shared draws.

    python3 run_pharmagent_pop.py <pop dir> <outdir>      # writes pharmagent_pop.csv and pharmagent_eta.csv

app.compute.nlme.simulate_replicate_subject is called as production calls it; only its random-number generator is
replaced by ScriptedRng, which hands back the shared z in the order the function asks for them (one vector for the
etas, then one standard normal per observation). The runner FAILS if any draw is left unused or the function asks for
more, or if it had to resample a negative concentration, because either would mean the comparison's draw-order
assumption is wrong. No R, no network, no LLM. IPRED comes from the module's own _predict (a private helper; used
here for diagnosis only).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BACKEND))


class ScriptedRng:
    """Stands in for numpy.random.Generator: returns the shared z, scaled the way the caller asks."""

    def __init__(self, z_eta: list[float], z_eps: list[float]) -> None:
        self.z_eta, self.z_eps = list(z_eta), list(z_eps)
        self.eta_used = self.eps_used = 0

    def normal(self, loc, scale):
        scale = np.asarray(scale, dtype=float)
        if scale.size != len(self.z_eta) or self.eta_used:
            raise RuntimeError("eta requested in an unexpected shape or more than once")
        self.eta_used = scale.size
        return loc + scale * np.array(self.z_eta)

    def standard_normal(self):
        if self.eps_used >= len(self.z_eps):
            raise RuntimeError("more residual draws requested than there are observations (a resample happened)")
        z = self.z_eps[self.eps_used]
        self.eps_used += 1
        return z


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def kv(s: str) -> dict[str, float]:
    return {k: float(v) for k, v in (p.split("=") for p in s.split(";") if p)}


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    src, out = Path(argv[1]), Path(argv[2])
    out.mkdir(parents=True, exist_ok=True)
    from app.compute.nlme import _PopSpec, _predict, _Subject, simulate_replicate_subject
    from app.compute.pk_models import get_model
    design, doses, obs, draws = (read(src / f"{n}.csv") for n in ("design", "doses", "obs", "draws"))
    rows, etas = [], []
    for d in design:
        sid = d["id"]
        iiv = [p for p in d["iiv"].split(",") if p]
        om2 = kv(d["omega2"])
        theta = kv(d["theta"])
        dd = [{"time": float(r["time"]), "amt": float(r["amt"]), "rate": float(r["rate"])}
              for r in doses if r["id"] == sid]
        tt = [float(r["time"]) for r in obs if r["id"] == sid]
        lloq = float(d["lloq"]) if d["lloq"] else None
        for s in range(1, int(d["n_subjects"]) + 1):
            z = [r for r in draws if r["id"] == sid and int(r["subject"]) == s]
            rng = ScriptedRng([float(r["z"]) for r in sorted(z, key=lambda r: int(r["idx"])) if r["kind"] == "eta"],
                              [float(r["z"]) for r in sorted(z, key=lambda r: int(r["idx"])) if r["kind"] == "eps"])
            try:
                res = simulate_replicate_subject(
                    d["model"], theta, om2, float(d["sigma_prop"]), float(d["sigma_add"]), iiv, d["error"], dd, tt,
                    float(d["wt"]), rng, lloq=lloq)
            except RuntimeError as exc:
                raise SystemExit(f"{sid} subject {s}: {exc}") from exc
            if not res["ok"] or res["n_resampled"] or res["n_negative_draws"]:
                raise SystemExit(f"{sid} subject {s}: not a clean draw {res['ok'], res['n_resampled']}")
            if rng.eta_used != len(iiv) or rng.eps_used != len(tt):
                raise SystemExit(f"{sid} subject {s}: draws used {rng.eta_used}/{rng.eps_used}, "
                                 f"expected {len(iiv)}/{len(tt)}")
            spec = _PopSpec(get_model(d["model"]), iiv, d["error"])
            subj = _Subject({"subject": "x", "doses": dd, "obs_t": tt, "obs_c": [1.0] * len(tt), "wt": float(d["wt"])})
            ipred = _predict(spec, subj, {**get_model(d["model"]).defaults, **theta},
                             np.array([res["eta"][p] for p in iiv]))
            blq = res["obs_blq"] or [False] * len(tt)
            rows += [{"id": sid, "subject": s, "time": repr(t), "ipred": repr(float(f)), "dv": repr(float(y)),
                      "blq": int(b)} for t, f, y, b in zip(res["obs_t"], ipred, res["obs_c"], blq, strict=True)]
            etas += [{"id": sid, "subject": s, "param": p, "eta": repr(float(v))} for p, v in res["eta"].items()]
    for name, data, fields in (("pharmagent_pop.csv", rows, ["id", "subject", "time", "ipred", "dv", "blq"]),
                               ("pharmagent_eta.csv", etas, ["id", "subject", "param", "eta"])):
        with (out / name).open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
            w.writeheader()
            w.writerows(data)
    print(f"{len(design)} scenarios, {len(rows)} observations, {len(etas)} etas -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
