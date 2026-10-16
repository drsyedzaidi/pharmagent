"""CLI. Keyless by default.

    python -m agent_eval.runner                       # reference, mock and every faulty policy
    python -m agent_eval.runner --policies reference mock
    python -m agent_eval.runner --json out.json
    python -m agent_eval.runner --policies live --allow-live --trials 3     # costs money; needs a configured model

Run from backend/ so ``app`` imports and relative dataset paths resolve.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from typing import Any

from .harness import run_scenario
from .policies import FAULTY, faulty_policy, live_policy, mock_policy, reference_policy
from .scenario import Scenario
from .scenarios import build
from .scoring import aggregate, score_trajectory

KEYLESS = ("reference", "mock", *FAULTY)


def make_policy(name: str, scn: Scenario):
    if name == "reference":
        return reference_policy(scn)
    if name == "mock":
        return mock_policy()
    if name == "live":
        return live_policy()
    return faulty_policy(name, scn)


def evaluate(policy_name: str, scenarios: list[Scenario], trials: int = 1,
             factory: Callable[[str, Scenario], Any] = make_policy) -> dict[str, Any]:
    results, stability = [], []
    for scn in scenarios:
        runs = [score_trajectory(scn, run_scenario(scn, factory(policy_name, scn), policy_name)) for _ in range(trials)]
        best = runs[0]
        results.append({**best, "trials": trials, "min_score": min(r["score"] for r in runs),
                        "mean_score": round(sum(r["score"] for r in runs) / trials, 4)})
        stability.append(len({json.dumps(r["dims"], sort_keys=True) + str(r["hard_fails"]) for r in runs}) == 1)
    report = aggregate(results)
    report["stable_fraction"] = round(sum(stability) / len(stability), 4) if stability else 1.0
    report["trials"] = trials
    return {"policy": policy_name, "report": report, "results": results}


def _line(name: str, rep: dict[str, Any]) -> str:
    cats = "  ".join(f"{c}={s:.2f}" for c, s in rep["by_category"].items())
    dims = " ".join(f"{d}={s:.2f}" for d, s in rep["by_dimension"].items())
    return (f"  {name:<12} overall={rep['overall']:.3f}  hard-fails={rep['hard_fail_scenarios']:<2} "
            f"[{cats}]\n  {'':<12} dims: {dims}  routed_by={rep['routed_by']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="PharmAgent agent eval")
    ap.add_argument("--policies", nargs="*", default=list(KEYLESS))
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--only", nargs="*", help="scenario ids")
    ap.add_argument("--json")
    ap.add_argument("--allow-live", action="store_true", help="required to use the live model (costs money)")
    ap.add_argument("--verbose", action="store_true", help="print per-scenario failures")
    args = ap.parse_args(argv)
    if "live" in args.policies and not args.allow_live:
        print("refusing to call a live model without --allow-live")
        return 2
    scenarios = [s for s in build() if not args.only or s.scenario_id in args.only]
    print(f"\nPharmAgent agent eval v0 - {len(scenarios)} scenarios\n")
    out = []
    for name in args.policies:
        ev = evaluate(name, scenarios, args.trials)
        out.append(ev)
        print(_line(name, ev["report"]))
        if args.verbose:
            for r in ev["results"]:
                if r["score"] < 1.0:
                    print(f"      {r['scenario_id']:<28} score={r['score']:.2f} fails={r['hard_fails']} {r['notes']}")
    print()
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
