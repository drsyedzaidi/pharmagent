"""PharmacometricsBench v0 — a reproducible eval for agentic pharmacometrics.

Ground truth for every task is defined as *the output a validated PharmAgent
compute tool produces on the provided data*. The benchmark therefore measures
one thing precisely: does the agent reproduce the tool's numbers, or does it
free-hand them? An agent that calls the deterministic tools scores 1.0; an agent
that eyeballs or mis-integrates does not. This directly encodes the platform
thesis — *agents decide, tools execute*.

v0 covers four deterministic categories whose oracles need no fitted NLME model
and no external data: NCA, bioequivalence, dose-proportionality, and
one-compartment structural PK. TDM/forecast, PBPK, and identifiability are
deferred (they need fixtures or literature answer keys) — see
pharmagent/PHARMACOMETRICSBENCH.md.
"""

from .grading import grade_task, score_report, within_tolerance
from .spec import Target, Task, dump_tasks, load_tasks

__all__ = [
    "Task", "Target", "load_tasks", "dump_tasks",
    "grade_task", "score_report", "within_tolerance",
]
