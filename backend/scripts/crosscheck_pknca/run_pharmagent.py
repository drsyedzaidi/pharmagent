#!/usr/bin/env python3
"""Run PharmAgent's own compute_nca tool (through the orchestrator, default role detection) on the same table.

    python3 run_pharmagent.py scripts/crosscheck_pknca/nca_input.csv scripts/crosscheck_pknca/out

Deterministic arithmetic on 6 subjects; no model fit, no network, no LLM (MockLLM is only the orchestrator's
placeholder). Writes pharmagent.csv with the columns the comparer expects, and pharmagent_roles.json so the role
detection that was actually used is on record.
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND))

FIELDS = ["ID", "Cmax", "Tmax", "AUClast", "AUCinf.obs", "lambda.z", "half.life", "lambda.z.n.points",
          "adj.r.squared", "tlast", "clast.obs"]
MAP = {"Cmax": "Cmax", "Tmax": "Tmax", "AUClast": "AUC_last", "AUCinf.obs": "AUC_inf", "lambda.z": "lambda_z",
       "half.life": "t_half", "lambda.z.n.points": "lambda_z_n_points", "adj.r.squared": "lambda_z_r2_adj",
       "tlast": "Tlast", "clast.obs": "Clast"}


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    src, outdir = Path(argv[1]), Path(argv[2])
    outdir.mkdir(parents=True, exist_ok=True)
    from app.config import settings
    tmp = Path(tempfile.mkdtemp(prefix="crosscheck_"))
    settings.data_dir = tmp                                   # the dataset-path guard only loads from data/
    shutil.copy(src, tmp / "nca_input.csv")
    from app.core.llm import MockLLM
    from app.core.orchestrator import Orchestrator
    from app.core.store import SessionStore
    orch = Orchestrator(llm=MockLLM(), store=SessionStore(":memory:"))
    sid = orch.create_session().id
    loaded = orch.run_tool(sid, "load_dataset", "data_manager", {"path": str(tmp / "nca_input.csv")})
    roles = loaded["state"]["dataset_metadata"]["detected_roles"]
    orch.run_tool(sid, "compute_nca", "nca", {})
    params = orch.get_session(sid).state.nca_parameters
    with (outdir / "pharmagent.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        for p in params:
            w.writerow({"ID": p["subject"], **{k: ("" if p.get(v) is None else p[v]) for k, v in MAP.items()}})
    (outdir / "pharmagent_roles.json").write_text(json.dumps(roles, indent=1), encoding="utf-8")
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"{len(params)} subjects -> {outdir / 'pharmagent.csv'}; roles used: {roles}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
