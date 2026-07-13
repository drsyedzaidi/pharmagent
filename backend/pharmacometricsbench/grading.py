"""Deterministic grading and scoring.

Each target is pass/fail under its tolerance rule; a task's score is the mean
over its targets; the overall score is the mean over *categories* (equal weight
per category, so an easy category with many tasks cannot dominate).
"""
from __future__ import annotations

import math
from typing import Any

from .spec import Target, Task


def within_tolerance(pred: Any, target: Target) -> bool:
    """True if a predicted value satisfies the target's tolerance rule."""
    if pred is None:
        return False
    rule = target.tol
    kind = rule["type"]

    if kind == "exact":
        # Booleans / categorical answers. Compare loosely across bool/str forms.
        return _norm(pred) == _norm(target.value)

    # Numeric rules below.
    try:
        p = float(pred)
        t = float(target.value)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(p):
        return False

    if kind == "abs":
        return abs(p - t) <= float(rule["abs"])
    if kind == "rel":
        if t == 0.0:
            return abs(p - t) <= float(rule.get("abs_floor", 1e-9))
        return abs(p - t) / abs(t) <= float(rule["rel"])
    if kind == "twofold":
        if t == 0.0 or p <= 0.0:
            return p == t
        ratio = p / t
        return 0.5 <= ratio <= 2.0
    raise ValueError(f"unknown tolerance type: {kind}")


def _norm(v: Any) -> Any:
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"true", "yes", "pass"}:
            return True
        if s in {"false", "no", "fail"}:
            return False
        return s
    return v


def grade_task(task: Task, answer: dict[str, Any]) -> dict[str, Any]:
    """Grade one agent answer (a name->value dict) against a task."""
    per_target: dict[str, bool] = {}
    for tgt in task.targets:
        per_target[tgt.name] = within_tolerance(answer.get(tgt.name), tgt)
    score = sum(per_target.values()) / len(per_target) if per_target else 0.0
    return {
        "task_id": task.task_id,
        "category": task.category,
        "per_target": per_target,
        "score": score,
        "missing": [t.name for t in task.targets if t.name not in answer],
    }


def score_report(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate graded results into a leaderboard-style report."""
    by_cat: dict[str, list[float]] = {}
    for r in results:
        by_cat.setdefault(r["category"], []).append(r["score"])

    cat_scores = {c: sum(v) / len(v) for c, v in by_cat.items()}
    overall = sum(cat_scores.values()) / len(cat_scores) if cat_scores else 0.0
    return {
        "overall": round(overall, 4),
        "by_category": {c: round(s, 4) for c, s in sorted(cat_scores.items())},
        "n_tasks": len(results),
        "n_categories": len(cat_scores),
    }
