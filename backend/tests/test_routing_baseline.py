"""Routing must not regress against the supervisor at commit 54bc72d.

The exposure-response agent, the briefing memo and the Bucher indirect comparison were
added after 54bc72d. The supervisor keeps the 54bc72d scorer unchanged (raw lower-case
substring counts, tie or zero -> classifier over every agent) and adds only:
  * two outright anchors, checked before scoring: report <- 'memo' / 'briefing';
    statistician <- 'bucher' / 'indirect (treatment) comparison(s)' /
    'adjusted indirect' / 'common comparator';
  * the er_dose agent, whose keywords are specific multi-word phrases only;
  * the new statistician and report keywords those anchors name.

Acceptance bar: a request routes exactly as at 54bc72d -- same agent, same method, same
classifier roster -- unless it matches a new anchor or names a new er_dose phrase.

How the expectations were derived: ``BASELINE_CASES`` and ``RECHECK_PROBES`` were run
through the 54bc72d supervisor, obtained with ``git show 54bc72d:backend/app/agents/
supervisor.py`` and loaded into a temporary module (``importlib.util.
spec_from_file_location``) next to the current ``app`` package; the results are
hard-coded below and ``test_expectations_match_the_54bc72d_supervisor`` re-derives them
when git and the commit are available. ``test_every_baseline_keyword_routes_as_at_54bc72d``
sweeps every pre-existing keyword through both supervisors.

Known baseline behaviour, out of scope here and kept as-is (54bc72d routes shown):
  'run noncompartmental analysis'   -> compartmental (substring 'compartmental')
  'non-compartmental'               -> compartmental (substring 'compartmental')
  'one-compartment profile'         -> compartmental (2 vs clinpharm 1)
  'fit a three-compartment model'   -> classifier   (compartmental/modeler tie)
  'quality control of the analysis' -> classifier   (data_manager/qc tie)
"""
from __future__ import annotations

import importlib.util
import subprocess
import tempfile
from pathlib import Path

import pytest

from app.agents.definitions import DESCRIPTIONS
from app.agents.supervisor import KEYWORDS, Supervisor

BASELINE_COMMIT = "54bc72d"
LLM = "<classifier>"      # expectation marker: the classifier decides, offered every agent


class _NeverLLM:
    def classify(self, message, options, descriptions):
        raise AssertionError(f"keyword routing should have decided: {message!r}")


class _Recorder:
    """Deterministic classifier that records the roster it was offered."""

    def __init__(self) -> None:
        self.options: list[str] | None = None

    def classify(self, message, options, descriptions):
        self.options = list(options)
        return options[0]


#: (phrase, agent the 54bc72d supervisor chose by keyword -- and the request's intent)
BASELINE_CASES: list[tuple[str, str]] = [
    ("load this csv dataset and profile it", "data_manager"),
    ("upload the xpt file", "data_manager"),
    ("check data quality of the dataset", "data_manager"),
    ("plot spaghetti curves for each subject", "data_manager"),
    ("convert to cdisc format", "data_manager"),
    ("compute NCA AUC and Cmax", "nca"),
    ("estimate terminal lambda and auc", "nca"),
    ("what is the tmax", "nca"),
    ("linear-up log-down trapezoidal auc", "nca"),
    ("compute NCA AUC and Cmax and export to docx", "nca"),
    ("run a bioequivalence assessment test vs reference", "be"),
    ("compute the gmr and 90% ci", "be"),
    ("is it bioequivalent within 80-125", "be"),
    ("average bioequivalence abe for test/reference", "be"),
    ("assess dose proportionality with the power model", "dose_prop"),
    ("check dose linearity across the dose escalation", "dose_prop"),
    ("is exposure proportional to dose", "dose_prop"),
    ("fit a two-compartment model", "compartmental"),
    ("one-compartment fit per subject", "compartmental"),
    ("estimate ka with a compartmental model", "compartmental"),
    ("estimate the absorption rate", "compartmental"),
    ("run a population pk analysis", "poppk"),
    ("estimate iiv and typical value", "poppk"),
    ("poppk with covariate search", "poppk"),
    ("mixed-effects model with inter-individual variability", "poppk"),
    ("compare the model library and pick the best model", "modeler"),
    ("fit a transit absorption model", "modeler"),
    ("michaelis-menten elimination", "modeler"),
    ("turnover model for the biomarker", "modeler"),
    ("effect compartment pkpd model", "modeler"),
    ("simulate the Emax model over a dose sweep", "simulator"),
    ("fit the emax model and simulate a dose sweep", "simulator"),
    ("simulate an emax model dose sweep for the pediatric population", "simulator"),
    ("simulate an emax curve", "simulator"),
    ("simulate the emax response for 3 doses", "simulator"),
    ("run a simulation-estimation design check", "simulator"),
    ("clinical trial simulation of target attainment", "simulator"),
    ("pediatric dose simulation", "simulator"),
    ("bootstrap the parameter uncertainty", "simulator"),
    ("profile likelihood for CL", "simulator"),
    ("simulate renal impairment exposure", "simulator"),
    ("run a qc checklist", "qc"),
    ("verify the diagnostic plots", "qc"),
    ("run a QC review checklist", "qc"),
    ("compute half-life from ke is 0.1", "clinpharm"),
    ("loading dose for target concentration", "clinpharm"),
    ("cockcroft-gault creatinine clearance", "clinpharm"),
    ("convert 5 mg/l to micromolar", "clinpharm"),
    ("accumulation at steady state", "clinpharm"),
    ("sample size for intra-subject cv of 25%", "clinpharm"),
    ("which test should I use, parametric or non-parametric?", "statistician"),
    ("how should I analyze this data", "statistician"),
    ("check normality with shapiro", "statistician"),
    ("log-transform skewed distribution", "statistician"),
    ("anova vs kruskal", "statistician"),
    ("generate the report docx", "report"),
    ("write up the methods section", "report"),
    ("generate the report", "report"),
    ("create the report document", "report"),
]

#: Requests earlier rewrites regressed; the value is what 54bc72d did (``LLM`` = a tie,
#: settled by the classifier over the full roster).
RECHECK_PROBES: list[tuple[str, str]] = [
    ("convert 10umol to ng/ml", "clinpharm"),
    ("what is 10µm in ng/ml", "clinpharm"),
    ("dose for 70kg to 20kg patient", "clinpharm"),
    ("scale 70kg to 20kg clearance", LLM),                       # nca 'clearance' = clinpharm 'kg to'
    ("simulate the emax model", LLM),                            # simulator = modeler
    ("simulate the indirect response model for 3 doses", LLM),   # simulator = modeler
    ("bootstrap the emax model parameters", LLM),                # simulator = modeler
    ("simulate the effect compartment model", LLM),              # simulator = modeler = compartmental
    ("profile likelihood of the emax model", "simulator"),       # 'likelihood' x2
]

#: Agents and tools added after 54bc72d: exposure-response, memo, indirect comparison.
NEW_CASES: list[tuple[str, str]] = [
    ("fit a Cox model of time to progression on AUC and give the hazard ratio per SD", "er_dose"),
    ("Kaplan-Meier curves by exposure quartile with a log-rank test", "er_dose"),
    ("logistic exposure-response of the adverse event: odds ratio per SD of Cmax", "er_dose"),
    ("select the optimal dose with a utility of efficacy minus toxicity", "er_dose"),
    ("Bucher indirect comparison of the hazard ratios for A vs C through placebo", "statistician"),
    ("indirect comparison: odds ratio A vs B 0.6, C vs B 0.8", "statistician"),
    ("adjusted indirect comparison of hazard ratio 0.7 and 0.9 via common comparator", "statistician"),
    ("briefing memo summarising the Cox hazard ratio and the optimal dose", "report"),
    ("make a briefing memo of the exposure-response and dose selection", "report"),
    ("memo with the indirect comparison result", "report"),
    ("write a memo on the NCA AUC and Cmax", "report"),
]


def _route(sup_cls, message: str):
    rec = _Recorder()
    return sup_cls(rec).route(message), rec.options


@pytest.mark.parametrize("message,agent", BASELINE_CASES + NEW_CASES)
def test_keyword_routing(message, agent):
    assert Supervisor(_NeverLLM()).route(message) == (agent, "keyword")


@pytest.mark.parametrize("message,expected", RECHECK_PROBES)
def test_recheck_probes_route_as_at_54bc72d(message, expected):
    (agent, how), options = _route(Supervisor, message)
    if expected == LLM:
        assert how == "llm" and options == list(DESCRIPTIONS), message
    else:
        assert (agent, how) == (expected, "keyword") and options is None, message


def test_the_table_spans_every_agent():
    covered = {a for _, a in BASELINE_CASES + NEW_CASES} | {"reviewer"}
    assert covered == set(DESCRIPTIONS)


def test_reviewer_is_reached_through_the_classifier_with_every_agent_offered():
    """The reviewer has no keywords (as at 54bc72d): a request without any keyword goes to
    the classifier with the full roster, and the classifier's choice is final."""
    seen = {}

    class _Pick:
        def classify(self, message, options, descriptions):
            seen["options"] = list(options)
            return "reviewer"

    assert Supervisor(_Pick()).route("please challenge and refute these results") == ("reviewer", "llm")
    assert seen["options"] == list(DESCRIPTIONS)


# ── 54bc72d supervisor, loaded from git ──────────────────────────────────────

def _load_baseline_supervisor():
    repo = Path(__file__).resolve().parents[2]
    try:
        src = subprocess.run(
            ["git", "-C", str(repo), "show", f"{BASELINE_COMMIT}:backend/app/agents/supervisor.py"],
            capture_output=True, text=True, check=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    path = Path(tempfile.mkdtemp(prefix="supervisor_54bc72d_")) / "supervisor_54bc72d.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("supervisor_54bc72d", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASELINE = _load_baseline_supervisor()

#: 54bc72d KEYWORDS, used when git is unavailable (identical to the commit's table).
FALLBACK_KEYWORDS: dict[str, list[str]] = {
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

BASELINE_KEYWORDS: dict[str, list[str]] = BASELINE.KEYWORDS if BASELINE is not None else FALLBACK_KEYWORDS
SWEEP_TEMPLATES = ("{kw}", "please {kw} for this dataset")
SWEEP = sorted({(t.format(kw=kw), agent) for agent, kws in BASELINE_KEYWORDS.items()
                for kw in kws for t in SWEEP_TEMPLATES})

#: Sweep phrases allowed to route differently: they match a new anchor or contain a new
#: er_dose phrase. None of the 54bc72d keywords does (the new vocabulary is disjoint),
#: so the list is empty; ``test_sweep_exceptions_are_exactly_the_listed_ones`` fails if
#: a future keyword makes one -- add it here with its reason.
SWEEP_EXCEPTIONS: dict[str, str] = {}


def _hits_new_vocabulary(phrase: str) -> bool:
    from app.agents.supervisor import ANCHORS, ER_ANCHOR
    low = phrase.lower()
    return any(p.search(low) for _, p in ANCHORS) or bool(ER_ANCHOR.search(low))


def test_fallback_keywords_match_the_commit():
    if BASELINE is None:
        pytest.skip(f"git or commit {BASELINE_COMMIT} unavailable")
    assert BASELINE.KEYWORDS == FALLBACK_KEYWORDS


def test_pre_existing_keyword_lists_are_exactly_those_of_54bc72d():
    # Anchored phrases (memo, briefing, Bucher, indirect comparison) are deliberately not
    # keywords: a raw substring count would let 'memory' or 'indirect treatment effect'
    # score for report / statistician without the anchor firing.
    for agent, kws in FALLBACK_KEYWORDS.items():
        assert KEYWORDS[agent] == kws, agent
    # er_dose has no scoring keywords: it is reached only through ER_ANCHOR, after the
    # 54bc72d keyword routing has had its say. That is what makes a request without a
    # new phrase route exactly as at 54bc72d.
    assert set(KEYWORDS) == set(FALLBACK_KEYWORDS)


def test_sweep_exceptions_are_exactly_the_listed_ones():
    derived = {phrase for phrase, _ in SWEEP if _hits_new_vocabulary(phrase)}
    assert derived == set(SWEEP_EXCEPTIONS)


@pytest.mark.parametrize("phrase,agent", SWEEP)
def test_every_baseline_keyword_routes_as_at_54bc72d(phrase, agent):
    if BASELINE is None:
        pytest.skip(f"git or commit {BASELINE_COMMIT} unavailable")
    if phrase in SWEEP_EXCEPTIONS:
        pytest.skip(SWEEP_EXCEPTIONS[phrase])
    assert _route(Supervisor, phrase) == _route(BASELINE.Supervisor, phrase), (phrase, agent)


def test_expectations_match_the_54bc72d_supervisor():
    if BASELINE is None:
        pytest.skip(f"git or commit {BASELINE_COMMIT} unavailable")
    sup = BASELINE.Supervisor(_NeverLLM())
    for message, agent in BASELINE_CASES:
        assert sup.route(message) == (agent, "keyword"), message
    for message, expected in RECHECK_PROBES:
        (agent, how), options = _route(BASELINE.Supervisor, message)
        if expected == LLM:
            assert how == "llm" and options == list(DESCRIPTIONS), message
        else:
            assert (agent, how) == (expected, "keyword"), message
