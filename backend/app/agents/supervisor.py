"""Supervisor — Level 0 routing.

Two-stage: (1) weighted keyword scoring over domain terms; (2) if ambiguous,
fall back to LLM classification. When a workflow is active, the orchestrator
bypasses this and routes by the template's task plan instead.
"""
from __future__ import annotations

from app.agents.definitions import DESCRIPTIONS
from app.core.llm import LLM

KEYWORDS: dict[str, list[str]] = {
    "data_manager": ["load", "dataset", "upload", "profile", "quality", "cdisc",
                     "spaghetti", "csv", "xpt", "column"],
    "nca": ["nca", "noncompartmental", "non-compartmental", "auc", "cmax", "tmax",
            "half-life", "half life", "trapezoidal", "lambda", "clearance"],
    "be": ["bioequivalence", "bioequivalent", "be ", "gmr", "geometric mean ratio",
           "test reference", "test/reference", "90% ci", "80-125", "abe"],
    "dose_prop": ["dose proportionality", "dose-proportionality", "proportional",
                  "power model", "linearity", "dose linearity", "dose escalation"],
    "compartmental": ["compartmental", "compartment", "one-compartment", "two-compartment",
                      "1-compartment", "2-compartment", "model fit", "fit model",
                      "structural model", "ka ", "absorption rate"],
    "poppk": ["population pk", "poppk", "pop pk", "iiv", "inter-individual",
              "interindividual", "typical value", "mixed effects", "mixed-effects",
              "two-stage", "covariate"],
    "modeler": ["structural model", "model library", "fit a model", "fit model",
                "model selection", "three-compartment", "3-compartment", "transit",
                "michaelis", "menten", "indirect response", "turnover", "emax model",
                "pkpd model", "pk/pd", "effect compartment", "model fitting", "best model"],
    "qc": ["qc", "quality control", "diagnostic", "checklist", "review", "verify"],
    "report": ["report", "docx", "document", "methods section", "write up", "writeup"],
}


def score(message: str) -> dict[str, int]:
    low = (message or "").lower()
    return {agent: sum(low.count(k) for k in kws) for agent, kws in KEYWORDS.items()}


class Supervisor:
    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def route(self, message: str) -> tuple[str, str]:
        """Return (agent_name, routing_method)."""
        scores = score(message)
        best = max(scores, key=scores.get)
        top = scores[best]
        # ambiguous: no clear winner or a tie at the top
        winners = [a for a, s in scores.items() if s == top]
        if top == 0 or len(winners) > 1:
            choice = self.llm.classify(message, list(DESCRIPTIONS), DESCRIPTIONS)
            return choice, "llm"
        return best, "keyword"
