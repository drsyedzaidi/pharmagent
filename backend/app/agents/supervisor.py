"""Supervisor — Level 0 routing.

Three stages: (1) a few unambiguous anchors decide outright -- an artifact the user
wants (a memo or briefing) goes to the report agent and a named indirect-comparison
method (Bucher, indirect comparison, common comparator) to the statistician, because
such requests always also name other agents' domain nouns (the analyses a memo
summarises, the hazard / odds ratio scale of an indirect comparison); (2) keyword
scoring over domain terms (raw substring counts, as at commit 54bc72d); (3) if no
keyword matches or the top score is tied, LLM classification over every agent. When a
workflow is active, the orchestrator bypasses this and routes by the template's task
plan instead. tests/test_routing_baseline.py pins routing against commit 54bc72d.
"""
from __future__ import annotations

import re

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
    "simulator": ["simulation-estimation", "simulation estimation", "simest", "sim-est",
                  "sampling design", "study design", "design check", "precision check",
                  "simulate", "dose sweep", "trial simulation", "clinical trial simulation",
                  "target attainment", "probability of target", "pediatric",
                  "special population", "renal impairment",
                  "bootstrap", "sampling importance", "importance resampling",
                  "likelihood", "likelihood profil", "profile likelihood", "parameter uncertainty"],
    "qc": ["qc", "quality control", "diagnostic", "checklist", "review", "verify"],
    "clinpharm": ["half-life", "half life", "elimination rate", "rate constant", " ke ", "ke is",
                  "accumulation", "steady state", "steady-state", "loading dose", "maintenance dose",
                  "creatinine", "crcl", "egfr", "ckd-epi", "cockcroft", "renal dose", "renal function",
                  "allometric", "allometry", "body weight scaling", "micromolar", "µm", "umol",
                  "mg/l", "molar mass", "molecular weight", "sample size", "how many subjects",
                  "% cv", "cv of", "intra-subject", "creatinine clearance", "year old", "target concentration",
                  "dose to reach", "kg to", "scale clearance", "scale volume",
                  "one-compartment profile", "1-compartment profile"],
    "statistician": ["statistic", "parametric", "nonparametric", "non-parametric",
                     "normality", "shapiro", "t-test", " t test", "wilcoxon", "mann-whitney",
                     "mann whitney", "kruskal", "anova", "which test", "what test",
                     "how to analyze", "how to analyse", "how should i analyze",
                     "how should i analyse", "analysis plan", "log-transform",
                     "log transform", "geometric mean", "skew", "distribution"],
    "report": ["report", "docx", "document", "methods section", "write up", "writeup"],
}

#: Agents whose keywords are only exposure / dataset nouns (AUC, Cmax, "dataset"): an
#: exposure-response request always names those, so they never block er_dose.
NOUN_AGENTS = frozenset({"nca", "data_manager"})

#: Checked before anything else; whole words, so 'memory' and 'debriefing' never anchor.
#: The pre-existing agents' KEYWORDS are exactly those of commit 54bc72d, and er_dose has
#: no scoring keywords at all, so a request that names no new phrase routes exactly as at
#: 54bc72d by construction.
ANCHORS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("report", re.compile(r"\b(?:memos?|memorand(?:um|a)|briefings?)\b")),
    ("statistician", re.compile(
        r"\bbucher\b|\bindirect(?:ly)?[\s-]+(?:treatment[\s-]+)?compar\w*"
        r"|\bindirect[\s-]+(?:hazard|odds|risk)[\s-]+ratios?\b|\bitc\b"
        r"|\badjusted[\s-]+indirect\b|\bcommon[\s-]+comparator\b")),
)

_DASH = r"[\s\-‐-―]"   # space, hyphen and the unicode dashes (en, em, ...)

#: An exposure-response request: a binary or time-to-event fit, its bootstrap, or a dose
#: selection. Decides for er_dose only when no pre-existing agent other than the noun
#: agents scores, and no continuous PD model or NLME-uncertainty method is named.
ER_ANCHOR = re.compile(
    rf"\bexposure{_DASH}+response\b|\be{_DASH}?r{_DASH}+(?:model|analys[ie]s|relationship|curves?|slope|fit)\b"
    rf"|\ber{_DASH}+analys[ie]s\b|\bkaplan\b|\bkm{_DASH}+(?:curves?|plots?|estimat\w*)\b"
    rf"|\bcox{_DASH}+(?:model|regression|proportional|ph)\b|\bproportional{_DASH}+hazards?\b"
    rf"|\blogistic{_DASH}+(?:regression|model|curves?|fits?|e{_DASH}?r)\b|\blogit\b"
    rf"|\btime{_DASH}+to{_DASH}+event\b|\boptimus\b|\boptimal{_DASH}+dos(?:e|es|ing)\b")

#: Continuous PD models belong to the modeler; NLME uncertainty methods to the simulator.
ER_YIELD = re.compile(
    rf"\be{_DASH}?max\b|\bsigmoid\w*|\bturnover\b|\bindirect{_DASH}+response\b"
    rf"|\beffect{_DASH}+compartment\b|\bpk{_DASH}*/?{_DASH}*pd\b|\bpd{_DASH}+model\b|\bhill\b"
    rf"|\bsir\b|\bparameter{_DASH}+uncertainty\b|\bprofil\w*{_DASH}+(?:the{_DASH}+)?likelihood\b"
    rf"|\bpop{_DASH}?pk\b|\bnlme\b|\bfoce\w*|\bsaem\b|\bpopulation{_DASH}+pk\b")

def score(message: str) -> dict[str, int]:
    low = (message or "").lower()
    return {agent: sum(low.count(k) for k in kws) for agent, kws in KEYWORDS.items()}


def _simulator_only_bootstrap(low: str) -> bool:
    """An exposure-response bootstrap is er_dose's own (proposed) tool, so the simulator's
    'bootstrap' keyword alone does not block the anchor; ER_YIELD already stops it when a
    popPK / NLME fit is named."""
    return [k for k in KEYWORDS["simulator"] if k in low] == ["bootstrap"]


def anchored(message: str) -> str | None:
    """The agent an anchor decides outright, or None."""
    low = (message or "").lower()
    agent = next((a for a, pattern in ANCHORS if pattern.search(low)), None)
    if agent is not None:
        return agent
    if not ER_ANCHOR.search(low) or ER_YIELD.search(low):
        return None
    # The exposure-response phrase's own words ('proportional' in 'proportional hazards')
    # must not count as another agent claiming the request.
    rest = ER_ANCHOR.sub(" ", low)
    blocking = {a for a, v in score(rest).items() if v and a not in NOUN_AGENTS}
    if blocking == {"simulator"} and _simulator_only_bootstrap(rest):
        blocking = set()
    return "er_dose" if not blocking else None


class Supervisor:
    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def route(self, message: str) -> tuple[str, str]:
        """Return (agent_name, routing_method)."""
        agent = anchored(message)
        if agent is not None:
            return agent, "keyword"
        scores = score(message)
        best = max(scores, key=scores.get)
        top = scores[best]
        # ambiguous: no clear winner or a tie at the top
        winners = [a for a, s in scores.items() if s == top]
        if top == 0 or len(winners) > 1:
            choice = self.llm.classify(message, list(DESCRIPTIONS), DESCRIPTIONS)
            return choice, "llm"
        return best, "keyword"
