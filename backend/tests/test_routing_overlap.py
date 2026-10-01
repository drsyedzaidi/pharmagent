"""Supervisor routing where the new agents share vocabulary with existing ones.

An indirect comparison always names its effect scale (hazard / odds / risk ratio) and a
memo names the analyses it summarises, so both are anchored outright before scoring:
report <- 'memo' / 'briefing'; statistician <- 'bucher', 'indirect (treatment)
comparison(s)', 'adjusted indirect', 'common comparator'. Everything else is the
54bc72d scorer: raw substring counts, a tie or no keyword -> the classifier over every
agent. The exposure-response agent owns only specific multi-word phrases ('cox model',
'hazard ratio', 'logistic regression', ...), never a bare generic word, so 'wilcoxon'
(which contains 'cox') stays with the statistician.

An unambiguous exposure-response phrase ('exposure-response', 'cox model', 'kaplan',
'logistic regression', 'optimal dose', ...) anchors er_dose too, unless the request also
names a continuous PD model, a report artifact, a test choice or a simulation. In that
case er_dose neither anchors nor scores, so the request routes exactly as at 54bc72d.
"""
from __future__ import annotations

import pytest

from app.agents.definitions import DESCRIPTIONS
from app.agents.supervisor import KEYWORDS, Supervisor
from app.core.llm import MockLLM


class _NeverLLM:
    def classify(self, message, options, descriptions):  # pragma: no cover - must not be called
        raise AssertionError(f"keyword routing should have decided: {message!r}")


CASES = [
    # indirect comparison: anchored to the statistician whatever scale it names
    ("Bucher indirect comparison of the hazard ratios for A vs C through placebo", "statistician"),
    ("indirect comparison: odds ratio A vs B 0.6, C vs B 0.8", "statistician"),
    ("adjusted indirect comparison of hazard ratio 0.7 and 0.9 via common comparator", "statistician"),
    ("indirect comparison of hazard ratios: A vs placebo HR 0.7, B vs placebo HR 0.8", "statistician"),
    ("indirect treatment comparison, odds ratio scale", "statistician"),
    ("indirect comparisons of the Cox model hazard ratios", "statistician"),
    ("indirect comparison of risk ratios via common comparator", "statistician"),
    ("run a Bucher indirect comparison of drug A vs drug C via placebo", "statistician"),
    ("do an adjusted indirect treatment comparison", "statistician"),
    # wilcoxon contains 'cox' but er_dose has no bare 'cox'
    ("responder analysis with Wilcoxon test", "statistician"),
    ("wilcoxon", "statistician"),
    ("use a wilcoxon test for these two groups", "statistician"),
    ("mann-whitney or wilcoxon?", "statistician"),
    # Cox / Kaplan-Meier / logistic E-R phrasings with specific phrases reach er_dose
    ("fit a Cox model of time to progression on AUC and give the hazard ratio per SD", "er_dose"),
    ("what is the hazard ratio per SD of AUC from a Cox model", "er_dose"),
    ("Cox proportional hazards regression of time to event on Cmax", "er_dose"),
    ("hazard ratio from a cox regression", "er_dose"),
    ("Kaplan-Meier curves by exposure quartile", "er_dose"),
    ("Kaplan-Meier curves by exposure quartile with a log-rank test", "er_dose"),
    ("time-to-event analysis with kaplan-meier and logrank", "er_dose"),
    ("fit a logistic exposure-response model for the adverse event", "er_dose"),
    ("logistic exposure-response of the adverse event: odds ratio per SD of Cmax", "er_dose"),
    ("logistic regression of response on AUC: odds ratio per SD", "er_dose"),
    ("select the optimal dose with a utility of efficacy minus toxicity", "er_dose"),
    ("select the optimal dose using a utility index (Project Optimus)", "er_dose"),
    ("bootstrap the exposure-response logistic model", "er_dose"),
    # report: an artifact request wins over the analyses it names
    ("briefing memo summarising the Cox hazard ratio and the optimal dose", "report"),
    ("make a briefing memo of the exposure-response and dose selection", "report"),
    ("memo with the indirect comparison result", "report"),
    ("write a briefing for the team on the Kaplan-Meier result", "report"),
    # existing agents are not taken over by er_dose
    ("fit an indirect response model to the PD data", "modeler"),
    ("fit pk/pd model with Emax response curve", "modeler"),
    ("pediatric dose selection", "simulator"),
    ("dose selection based on target attainment", "simulator"),
    ("simulate survival of the cohort", "simulator"),
    ("likelihood ratio test for the logistic slope", "simulator"),
]


@pytest.mark.parametrize("message,agent", CASES)
def test_overlapping_vocabulary_routes_by_intent(message, agent):
    assert Supervisor(_NeverLLM()).route(message) == (agent, "keyword")


#: 'odds ratio' / 'hazard ratio' alone are not exposure-response phrases (an indirect
#: comparison or a BE report names them too), so these route exactly as at 54bc72d.
RATIO_ONLY = [
    ("odds ratio of response per unit AUC", "nca"),
    ("hazard ratio per unit Cmax", "nca"),
]


@pytest.mark.parametrize("message,agent", RATIO_ONLY)
def test_a_ratio_word_alone_routes_as_at_54bc72d(message, agent):
    assert Supervisor(_NeverLLM()).route(message) == (agent, "keyword")


def test_compare_indirectly_without_an_anchor_goes_to_the_classifier():
    # A PD model can act 'indirectly', so 'compare ... indirectly' is not an anchor.
    seen = {}

    class _Spy:
        def classify(self, message, options, descriptions):
            seen["options"] = list(options)
            return "CLASSIFIER"
    msg = "compare drug A and drug B indirectly using their hazard ratios"
    assert Supervisor(_Spy()).route(msg) == ("CLASSIFIER", "llm")
    assert seen["options"] == list(DESCRIPTIONS)


#: Probes from the round-3 routing audits. Each routes exactly as at 54bc72d (no new
#: phrase involved, or an exception term makes er_dose step aside) or reaches the agent
#: that owns the new capability.
AUDIT_CASES = [
    # substrings of the anchored words never route
    ("the job ran out of memory during the bootstrap", "simulator"),
    ("the debriefing showed AUC was higher in fed state", "nca"),
    ("indirect treatment effect of food on AUC", "nca"),
    # anchors accept plurals, hyphens and the adverb form
    ("summarise as two memos", "report"),
    ("write two briefings for the team", "report"),
    ("an indirect-comparison of hazard ratios A vs C", "statistician"),
    ("Run an ITC of A vs B through placebo using the published hazard ratios", "statistician"),
    ("Compute the indirect hazard ratio of drug A versus drug B through their placebo arms",
     "statistician"),
    # exposure-response requests that name one or two exposure metrics
    ("logistic regression of response on AUC", "er_dose"),
    ("logistic regression of response on AUC and Cmax", "er_dose"),
    ("kaplan-meier by Cmax quartile", "er_dose"),
    ("cox model of time to progression vs AUC", "er_dose"),
    ("bootstrap the exposure-response logistic model", "er_dose"),
    ("find the optimal dose from efficacy and toxicity", "er_dose"),
    # an exception term: routes as at 54bc72d
    ("write the methods section for the logistic regression", "report"),
    ("generate the report with the exposure-response results", "report"),
    ("which test should I use for the odds ratio", "statistician"),
    ("simulate response probability at the optimal dose", "simulator"),
    ("profile likelihood of the exposure-response model", "simulator"),
]


@pytest.mark.parametrize("message,agent", AUDIT_CASES)
def test_round3_audit_probes(message, agent):
    assert Supervisor(_NeverLLM()).route(message) == (agent, "keyword")


def test_a_continuous_pd_model_is_never_routed_to_er_dose():
    # er_dose fits binary and time-to-event endpoints only.
    class _Spy:
        def classify(self, message, options, descriptions):
            return "CLASSIFIER"
    assert Supervisor(_Spy()).route("fit an Emax exposure-response model") == ("CLASSIFIER", "llm")


@pytest.mark.parametrize("word", ["cox", "hazard", "odds", "survival", "utility", "responder",
                                  "logistic", "response model", "dose selection", "adverse event"])
def test_er_dose_owns_no_generic_word(word):
    from app.agents.supervisor import ER_ANCHOR
    assert "er_dose" not in KEYWORDS
    assert ER_ANCHOR.search(word) is None


def test_no_keyword_at_all_still_asks_the_classifier_with_every_agent():
    agent, how = Supervisor(MockLLM()).route("hello there")
    assert how == "llm" and agent == "data_manager"


#: Round-4 audit probes (the final routing design). 'BASE' means the request names no new
#: phrase that applies, so it must route exactly as at commit 54bc72d.
ROUND4 = [
    ("Which statistical test should I use to compare time to event between arms?", "BASE"),
    ("Should I use a Wilcoxon or t-test on the time-to-event endpoint?", "BASE"),
    ("How do I analyze time-to-event data with tied events?", "er_dose"),
    ("Is a log-rank test appropriate here or a Wilcoxon test?", "BASE"),
    ("Fit an indirect-response model to the exposure-response data for cortisol", "BASE"),
    ("Add an effect-compartment delay to the exposure-response model of heart rate", "BASE"),
    ("Fit a sigmoidal exposure-response curve for QTc change", "BASE"),
    ("Fit an E-max exposure-response model for QTc", "BASE"),
    ("Build a PK\u2013PD exposure-response model for blood pressure", "BASE"),
    ("Run a simest for the exposure-response logistic model to check the sampling design", "BASE"),
    ("Check the sampling design for the exposure-response study", "BASE"),
    ("Run SIR on the exposure-response model for parameter uncertainty", "BASE"),
    ("Profile the likelihood for the exposure-response logistic model slope", "BASE"),
    ("Bootstrap the popPK model then use it in the exposure-response analysis", "BASE"),
    ("Generate reports for the exposure-response analysis", "BASE"),
    ("Draft the exposure-response documentation for the submission", "BASE"),
    ("Compare a direct Emax model with one where the drug acts indirectly via turnover", "BASE"),
    ("Was the suboptimal dose in cohort 2 explained by low AUC?", "BASE"),
    ("Patients on the suboptimal dose had lower Cmax?", "BASE"),
    ("Plot KM curves of PFS by AUC tertile", "er_dose"),
    ("Fit a logistic curve of probability of nausea vs Cmax", "er_dose"),
    ("Logit model of responder probability vs AUC", "er_dose"),
    ("Perform an ER analysis of AUC vs response rate", "er_dose"),
    ("E-R relationship of Cmax with grade 3 neutropenia", "er_dose"),
    ("Bootstrap the E-R slope to get a confidence interval", "er_dose"),
    ("Cox proportional hazards regression of time to event on Cmax", "er_dose"),
    ("the job ran out of memory during the bootstrap", "BASE"),
    ("convert 10umol to ng/ml", "BASE"),
    ("simulate the emax model", "BASE"),
    ("write the methods section for the logistic regression", "BASE"),
]


@pytest.mark.parametrize("message,expected", ROUND4)
def test_round4_audit_probes(message, expected):
    import subprocess
    import types

    class _Spy:
        def classify(self, message, options, descriptions):
            return "CLASSIFIER"

    new = Supervisor(_Spy()).route(message)
    if expected != "BASE":
        assert new == (expected, "keyword")
        return
    src = subprocess.run(["git", "show", "54bc72d:backend/app/agents/supervisor.py"],
                         capture_output=True, text=True)
    if src.returncode != 0:
        pytest.skip("git or commit 54bc72d unavailable")
    old = types.ModuleType("supervisor_54bc72d")
    exec(compile(src.stdout, "supervisor_54bc72d", "exec"), old.__dict__)
    assert new == old.Supervisor(_Spy()).route(message)
