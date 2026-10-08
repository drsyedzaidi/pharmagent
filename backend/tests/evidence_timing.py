"""Wall-clock limits for performance assertions that stay honest under a line tracer.

``scripts/evidence_coverage.py`` (and pytest-cov, a debugger, ``python -m trace``) install a line tracer that
slows pure-Python code 2-10x, which turns a correct 4 s assertion into a failure that says nothing about
the code. ``limit(seconds)`` returns *seconds* unchanged when no tracer or monitoring tool is active (the
strict limit every normal run uses) and *seconds* x TRACE_SLOWDOWN when one is, detected explicitly rather
than guessed from the environment. A regression of the kind these tests guard against (a quadratic loop,
work repeated per term) is orders of magnitude, not a small factor, so the scaled limit still catches it;
such tests should also assert the work done (a call count) where that is possible, which no tracer distorts.
"""
from __future__ import annotations

import sys
import threading

TRACE_SLOWDOWN = 4.0
_TOOL_IDS = range(6)                      # sys.monitoring tool ids (debugger, coverage, profiler, optimizer ...)


def tracing_active() -> bool:
    """True when a sys.settrace / threading.settrace hook or a sys.monitoring tool is installed."""
    if sys.gettrace() is not None or threading.gettrace() is not None:
        return True
    monitoring = getattr(sys, "monitoring", None)
    return monitoring is not None and any(monitoring.get_tool(i) is not None for i in _TOOL_IDS)


def limit(seconds: float, slowdown: float = TRACE_SLOWDOWN) -> float:
    """The wall-clock limit to assert against: strict normally, scaled while a tracer is active. *slowdown*: the
    measured tracer factor of a code path that pays more than the default per line (r7 round 3: the two-column checks
    of a 64 000-character line run about 7x slower under ``scripts/evidence_coverage.py``); the normal limit is the same
    whatever it is."""
    return seconds * slowdown if tracing_active() else seconds
