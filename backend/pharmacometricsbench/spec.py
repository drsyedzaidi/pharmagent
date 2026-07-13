"""Task specification: the on-disk JSONL format for benchmark tasks.

A Task carries everything a grader needs and everything an agent sees. The agent
is handed ``prompt`` + ``dataset``; the grader uses ``targets`` (each an expected
value plus a tolerance rule). ``meta`` holds provenance (true simulation params,
oracle function) and must never be required to answer the task.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

# Tolerance rule shapes (the "tol" dict on a Target):
#   {"type": "rel",  "rel": 0.05}   -> |pred-true| / |true| <= rel
#   {"type": "abs",  "abs": 1.0}    -> |pred-true| <= abs
#   {"type": "twofold"}             -> 0.5 <= pred/true <= 2.0  (PK GMFE band)
#   {"type": "exact"}               -> pred == true  (booleans / categories)
TolRule = dict[str, Any]


@dataclass
class Target:
    """One graded quantity within a task."""
    name: str
    value: Any                 # ground-truth value (float or bool)
    tol: TolRule
    unit: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Target:
        return cls(name=d["name"], value=d["value"], tol=d["tol"], unit=d.get("unit"))


@dataclass
class Task:
    task_id: str
    category: str              # nca | be | dp | compartmental
    prompt: str
    dataset: dict[str, Any]    # inputs the agent is allowed to see
    targets: list[Target]
    oracle: str                # provenance: which compute fn produced the truth
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "category": self.category,
            "prompt": self.prompt,
            "dataset": self.dataset,
            "targets": [t.to_dict() for t in self.targets],
            "oracle": self.oracle,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Task:
        return cls(
            task_id=d["task_id"],
            category=d["category"],
            prompt=d["prompt"],
            dataset=d["dataset"],
            targets=[Target.from_dict(t) for t in d["targets"]],
            oracle=d["oracle"],
            meta=d.get("meta", {}),
        )


def dump_tasks(tasks: list[Task], path: str) -> None:
    """Write tasks as JSON Lines (one task per line)."""
    with open(path, "w") as fh:
        for t in tasks:
            fh.write(json.dumps(t.to_dict()) + "\n")


def load_tasks(path: str) -> list[Task]:
    """Read a JSONL task file."""
    tasks: list[Task] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                tasks.append(Task.from_dict(json.loads(line)))
    return tasks
