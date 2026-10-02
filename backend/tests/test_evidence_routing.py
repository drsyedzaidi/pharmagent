"""The evidence agent is reached by an outright anchor; everything else routes as at d68842c.

The anchor names only evidence-pipeline nouns (publications, PDFs, systematic review,
evidence, inclusion/exclusion criteria, extraction template). Bare 'screen', 'review',
'extract' and 'dataset' are not anchored, so the QC, NCA and data_manager requests that
use them route exactly as before (checked against the d68842c supervisor loaded from git).
"""
from __future__ import annotations

import importlib.util
import subprocess
import tempfile
from pathlib import Path

import pytest

from app.agents.definitions import AGENTS, DESCRIPTIONS
from app.agents.supervisor import ANCHORS, KEYWORDS, Supervisor, anchored
from app.core.llm import MockLLM

PRE_EVIDENCE_COMMIT = "d68842c"


class _NeverLLM:
    def classify(self, message, options, descriptions):  # pragma: no cover - must not be called
        raise AssertionError(f"keyword routing should have decided: {message!r}")


class _Recorder:
    def __init__(self) -> None:
        self.options: list[str] | None = None

    def classify(self, message, options, descriptions):
        self.options = list(options)
        return options[0]


POSITIVES = [
    "screen the publications",
    "screening these PDFs",
    "systematic review of PK studies",
    "extract the evidence from the approved papers",
    "export the evidence dataset",
    "edit the extraction template",
    "screen the publications against my inclusion criteria",
    "apply the exclusion criteria to the uploaded pdf",
    "meta-analysis of the included studies",
]

#: Requests that share words with the evidence pipeline but belong to older agents.
NEGATIVES = [
    "review the dataset",
    "extract Cmax",
    "quality control review",
    "load dataset",
    "screening visit pk",
    "write the report document",
    "verify the diagnostic plots",
    "profile the dataset quality",
    "compute NCA AUC and Cmax",
]


@pytest.mark.parametrize("message", POSITIVES)
def test_evidence_requests_are_anchored(message):
    assert Supervisor(_NeverLLM()).route(message) == ("evidence", "keyword")


@pytest.mark.parametrize("message", NEGATIVES)
def test_negatives_never_reach_the_evidence_agent(message):
    assert anchored(message) != "evidence"
    rec = _Recorder()
    agent, _ = Supervisor(rec).route(message)
    assert agent != "evidence"


def test_evidence_is_in_the_classifier_roster_and_agent_table():
    assert "evidence" in DESCRIPTIONS and "evidence" in AGENTS
    assert AGENTS["evidence"].name == "evidence"
    assert "PROPOSAL" in AGENTS["evidence"].system_prompt
    rec = _Recorder()
    Supervisor(rec).route("hello there")
    assert rec.options == list(DESCRIPTIONS) and "evidence" in rec.options
    assert Supervisor(MockLLM()).route("hello there") == ("data_manager", "llm")


def test_evidence_owns_no_scoring_keyword_and_one_anchor():
    assert "evidence" not in KEYWORDS
    assert [a for a, _ in ANCHORS].count("evidence") == 1
    assert [a for a, _ in ANCHORS][-1] == "evidence"      # appended; earlier anchors keep priority


def test_memo_and_bucher_anchors_still_win_over_evidence_words():
    assert Supervisor(_NeverLLM()).route("briefing memo of the evidence dataset") == ("report", "keyword")
    assert Supervisor(_NeverLLM()).route("bucher indirect comparison from the publications") == (
        "statistician", "keyword")


@pytest.mark.parametrize("phrase,agent", [(k, a) for a, ks in KEYWORDS.items() for k in ks])
def test_no_pre_existing_keyword_phrase_is_anchored_to_evidence(phrase, agent):
    assert anchored(phrase) != "evidence"


# ── d68842c supervisor, loaded from git ──────────────────────────────────────

def _load_pre_evidence_supervisor():
    repo = Path(__file__).resolve().parents[2]
    try:
        src = subprocess.run(
            ["git", "-C", str(repo), "show", f"{PRE_EVIDENCE_COMMIT}:backend/app/agents/supervisor.py"],
            capture_output=True, text=True, check=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    tmp = Path(tempfile.mkdtemp()) / "supervisor_pre_evidence.py"
    tmp.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("supervisor_pre_evidence", tmp)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("message", NEGATIVES)
def test_negatives_route_exactly_as_at_d68842c(message):
    old = _load_pre_evidence_supervisor()
    if old is None:
        pytest.skip("git or the d68842c commit is unavailable")
    before, after = _Recorder(), _Recorder()
    assert old.Supervisor(before).route(message) == Supervisor(after).route(message)
    assert after.options == before.options       # same roster (both read the live DESCRIPTIONS)
