"""PharmAgent agent eval: does the real orchestrator route, choose tools, pass arguments, stop and stay inside its
governance when driven by a model?

Companion to ``pharmacometricsbench`` (which grades final NUMBERS). In PharmAgent the model cannot free-hand a number:
final messages are tool summaries. What a model controls is routing, tool choice, arguments, when to stop, and
whether it tries something it must not. This package measures exactly those, on the real Orchestrator, keylessly by
default (scripted and faulty policies), with an opt-in live model.
"""
from .harness import run_scenario
from .scenario import Scenario, TurnSpec
from .scoring import aggregate, score_trajectory

__all__ = ["Scenario", "TurnSpec", "aggregate", "run_scenario", "score_trajectory"]
