"""Runnable demo — a real cross-engine comparison on synthetic data.

    cd backend && python -m app.engines.demo

Fits one candidate (oral_1cmt) with the native FOCE-I engine and two mock
engines (an oracle and a biased one), then prints the two tables that make the
scientific point: the cross-engine winner is chosen on prediction accuracy, while
OFV/AIC/BIC are reported ONLY within each engine.
"""
from __future__ import annotations

import numpy as np

from app.compute.pk_models import get_model
from app.compute.pk_simulate import simulate

from .base import CandidateSpec
from .ensemble import build_ensemble
from .mock import MockEngineAdapter
from .native import PharmAgentAdapter
from .nlmixr2 import Nlmixr2Adapter
from .runner import run_matrix_subjects
from .select import select_winner


def _synthetic_subjects(n: int = 6) -> list[dict]:
    rng = np.random.default_rng(7)
    model = get_model("oral_1cmt")
    base = dict(model.defaults)
    times = np.array([0.5, 1, 2, 3, 4, 6, 8, 12, 24], dtype=float)
    subjects = []
    for sid in range(1, n + 1):
        p = {k: base[k] * float(np.exp(rng.normal(0, 0.22))) for k in base}
        cp = simulate(model, p, [{"time": 0.0, "amt": 100.0}], times)["cp"]
        obs = cp * np.exp(rng.normal(0, 0.06, size=cp.size))
        subjects.append({"subject": f"S{sid}", "doses": [{"time": 0.0, "amt": 100.0}],
                         "obs_t": times.tolist(), "obs_c": obs.tolist(), "wt": 70.0})
    return subjects


def main() -> None:
    subjects = _synthetic_subjects()
    spec = CandidateSpec(model_key="oral_1cmt", iiv_params=["CL", "V"])
    adapters = [PharmAgentAdapter(method="focei")]
    nlmixr2 = Nlmixr2Adapter()
    if nlmixr2.available():
        adapters.append(nlmixr2)                          # real external R engine
    else:
        adapters.append(MockEngineAdapter("nlmixr2_like", bias=0.05))  # stand-in
    adapters.append(MockEngineAdapter("monolix_like", bias=0.25))      # worse stand-in
    matrix = run_matrix_subjects(subjects, [spec], adapters)
    # Add a consensus engine (geometric mean of the converged fits) so it competes
    # in the ranking — motivated by ensembles beating single methods (Käser 2026).
    ensemble = build_ensemble(matrix["results"], spec.model_key, subjects)
    results = matrix["results"] + ([ensemble] if ensemble else [])
    sel = select_winner(results)

    print(f"\nCross-engine comparison — oral_1cmt, {len(subjects)} subjects, "
          f"{matrix['n_available']}/{matrix['n_engines']} engines available\n")

    print("PREDICTION RANKING (engine-agnostic — this picks the winner)")
    print(f"  {'engine':<16}{'pred_rmse':>10}{'vpc_cov90':>11}{'pred_r2':>9}{'|bias|':>9}")
    for r in sel["prediction_ranking"]:
        print(f"  {r.engine:<16}{_f(r.pred_rmse):>10}{_f(r.vpc_coverage90):>11}"
              f"{_f(r.pred_r2):>9}{_f(abs(r.pred_bias) if r.pred_bias is not None else None):>9}")
    # Show engines that did NOT produce a usable fit, so a failed/absent engine is
    # visible rather than silently dropped from the ranking.
    skipped = [r for r in matrix["results"] if r.status != "ok" or not r.converged]
    for r in skipped:
        why = r.message or ("did not converge" if r.status == "ok" else r.status)
        print(f"  {r.engine:<16}{'—':>10}{'—':>11}{'—':>9}{'—':>9}   ({why})")

    print("\nWITHIN-ENGINE LIKELIHOOD (never compared across engines)")
    for eng, rows in sel["within_engine_likelihood"].items():
        for row in rows:
            print(f"  {eng:<16} {row['model']:<12} ofv={_f(row['ofv'])}  "
                  f"aic={_f(row['aic'])}  bic={_f(row['bic'])}")

    w = sel["winner"]
    print(f"\nWINNER: {w.engine}  (metric: {sel['selection_metric']})")
    print(f"NOTE: {sel['note']}\n")


def _f(v, nd: int = 4) -> str:
    return "—" if v is None else f"{v:.{nd}f}"


if __name__ == "__main__":
    main()
