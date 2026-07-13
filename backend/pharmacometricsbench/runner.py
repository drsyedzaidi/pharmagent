"""CLI: generate a task set, run agents over it, print a leaderboard.

    python -m pharmacometricsbench.runner --generate --per-category 6
    python -m pharmacometricsbench.runner --run oracle naive

Run from the backend/ directory so ``app`` imports resolve.
"""
from __future__ import annotations

import argparse
import os
from collections.abc import Callable

from .agents import AGENTS
from .generators import build_taskset
from .grading import grade_task, score_report
from .spec import Task, dump_tasks, load_tasks

_HERE = os.path.dirname(__file__)
_DEFAULT_TASKS = os.path.join(_HERE, "tasks", "v0.jsonl")


def run_agent(agent: Callable[[Task], dict], tasks: list[Task]) -> dict:
    results = []
    n_error = 0
    for task in tasks:
        errored = False
        try:
            answer = agent(task)
        except Exception as exc:  # a crash is a zero, not a harness failure
            answer = {"_error": str(exc)}
            errored = True
            n_error += 1
        graded = grade_task(task, answer)
        graded["_errored"] = errored
        results.append(graded)
    report = score_report(results)
    # Surface how the zeros arose so a crash or a parse gap is not mistaken for a
    # merely-wrong answer: errored = agent raised; parse_gap = answered but a
    # graded target was absent from the answer (e.g. an LLM's output failed to parse).
    report["n_error"] = n_error
    report["n_parse_gap"] = sum(1 for r in results if not r["_errored"] and r["missing"])
    return report


def _fmt(name: str, report: dict) -> str:
    cats = "  ".join(f"{c}={s:.2f}" for c, s in report["by_category"].items())
    flags = []
    if report.get("n_error"):
        flags.append(f"{report['n_error']} errored")
    if report.get("n_parse_gap"):
        flags.append(f"{report['n_parse_gap']} parse-gap")
    extra = f"   ({', '.join(flags)})" if flags else "   (0 errored, 0 parse-gap)"
    return f"  {name:<8} overall={report['overall']:.3f}   [{cats}]{extra}"


def main() -> None:
    ap = argparse.ArgumentParser(description="PharmacometricsBench v0 runner")
    ap.add_argument("--generate", action="store_true", help="regenerate the task set")
    ap.add_argument("--per-category", type=int, default=6)
    ap.add_argument("--tasks", default=_DEFAULT_TASKS)
    ap.add_argument("--run", nargs="*", default=["oracle", "naive", "llm"],
                    help="agent names to evaluate (default: oracle naive llm)")
    args = ap.parse_args()

    if args.generate or not os.path.exists(args.tasks):
        os.makedirs(os.path.dirname(args.tasks), exist_ok=True)
        tasks = build_taskset(per_category=args.per_category)
        dump_tasks(tasks, args.tasks)
        print(f"generated {len(tasks)} tasks -> {args.tasks}")

    tasks = load_tasks(args.tasks)
    print(f"\nPharmacometricsBench v0 — {len(tasks)} tasks, "
          f"{len({t.category for t in tasks})} categories\n")
    for name in args.run:
        agent = AGENTS.get(name)
        if agent is None:
            print(f"  {name:<8} (unknown agent — skipped)")
            continue
        print(_fmt(name, run_agent(agent, tasks)))
    print()


if __name__ == "__main__":
    main()
