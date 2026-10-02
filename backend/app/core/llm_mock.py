"""Keyless, deterministic LLM stand-in.

Drives the core flows heuristically from message keywords and the state summary so the
platform, the desktop app and the test suite run with no key and no network call. It
never confirms an expensive tool on the human's behalf and never answers
``extract_json`` (None), which keeps the evidence tools on their deterministic path.
Re-exported from ``app.core.llm`` as ``MockLLM``.
"""
from __future__ import annotations

import re
from typing import Any

from app.tools.base import Tool

_PATH_RE = re.compile(r"[\w./~-]+\.(?:csv|xpt|sas7bdat)", re.IGNORECASE)


_KE_RE = re.compile(r"\bke\s*(?:is|=|of)?\s*([0-9]*\.?[0-9]+)")
_HL_RE = re.compile(r"half[- ]life\s*(?:is|=|of)?\s*([0-9]*\.?[0-9]+)")


def _mock_numbers(low: str, tool: str) -> dict[str, Any]:
    """Best-effort numeric extraction for the keyless mock (half-life / ke only;
    everything else is left for the tool to ask for)."""
    if tool in ("calc_half_life", "calc_accumulation"):
        m = _KE_RE.search(low)
        if m:
            return {"ke": float(m.group(1))}
        m = _HL_RE.search(low)
        if m:
            return {"half_life": float(m.group(1))}
    return {}


_ER_SELECT = ("optimal dose", "dose selection", "select the dose", "optimus", "utility")
_ER_FIT = ("logistic", "odds ratio", "hazard ratio", "cox", "kaplan", "exposure-response",
           "exposure response")
_ER_TTE = ("hazard", "cox", "kaplan", "survival", "time to event", "time-to-event")


def _mock_er_dose(low: str, s: dict[str, Any]) -> dict[str, Any] | None:
    """Keyless exposure-response choices. The mock cannot read column names, the
    doses or the utility weight from prose: the tool's own refusal names what is
    missing. The bootstrap is only ever PROPOSED (expensive+proposable); the mock
    never confirms on the human's behalf."""
    if any(k in low for k in _ER_SELECT):
        return None if s.get("dose_selection") else {"name": "select_optimal_dose", "input": {"doses": []}}
    if "bootstrap" in low:
        return {"name": "bootstrap_exposure_response", "input": {}}
    if any(k in low for k in _ER_FIT):
        if s.get("er_fits"):
            return None
        endpoint = "time_to_event" if any(k in low for k in _ER_TTE) else "binary"
        return {"name": "fit_exposure_response", "input": {"endpoint": endpoint}}
    return None


_DECISION_WORDS = ("approve", "reject", "decide", "decision")
_EVIDENCE_STEPS = (("export", "export_evidence_dataset", "evidence_dataset_path"),
                   ("extract", "extract_evidence", "extraction_results"),
                   ("screen", "screen_publications", "screening_results"))


def _mock_evidence(low: str, s: dict[str, Any], known: set[str]) -> dict[str, Any] | None:
    """Keyless evidence pipeline: screen / extract / export, each once, and only when
    publications exist. decide_* and config tools are never selected -- those are the
    human's and the workspace's; the registry only offers what later stages registered."""
    if not s.get("evidence_sources") or any(k in low for k in _DECISION_WORDS):
        return None
    for word, name, done_key in _EVIDENCE_STEPS:
        if word in low:
            return None if s.get(done_key) or name not in known else {"name": name, "input": {}}
    return None


class MockLLM:
    """Deterministic, keyless. Drives the core NCA flow heuristically."""

    def classify(self, message: str, options: list[str], descriptions: dict[str, str]) -> str:
        return options[0] if options else "data_manager"

    def extract_json(self, system: str, user: str, schema: dict[str, Any], *,
                     max_tokens: int = 2048) -> dict[str, Any] | None:
        return None

    def select_tool(self, agent: str, message: str, tools: list[Tool],
                    state_summary: dict[str, Any]) -> dict[str, Any] | None:
        s = state_summary
        if agent == "data_manager":
            if not s.get("dataset_id"):
                m = _PATH_RE.search(message or "")
                return {"name": "load_dataset", "input": {"path": m.group(0)}} if m else None
            # `data_quality` in the summary is the quality_flags LIST: an empty
            # list means "profiled, nothing flagged", not "not profiled" — a
            # falsy check here re-profiled clean datasets MAX_TOOL_STEPS times.
            if s.get("data_quality") is None:
                return {"name": "profile_pk_dataset", "input": {}}
            return None
        if agent == "nca":
            return None if s.get("nca_parameters") else {"name": "compute_nca", "input": {}}
        if agent == "be":
            return None if s.get("be_results") else {"name": "run_bioequivalence", "input": {}}
        if agent == "dose_prop":
            return None if s.get("dose_prop_results") else {"name": "run_dose_proportionality", "input": {}}
        if agent == "compartmental":
            return None if s.get("compartmental_results") else {"name": "fit_compartmental", "input": {}}
        if agent == "poppk":
            return None if s.get("poppk_results") else {"name": "run_poppk", "input": {}}
        if agent == "modeler":
            return None if s.get("pk_model_results") else {"name": "fit_pk_model", "input": {"compare": True}}
        if agent == "qc":
            return None if s.get("qc_verdict") else {"name": "run_qc", "input": {}}
        if agent == "report":
            low = (message or "").lower()
            if "memo" in low or "briefing" in low:
                return None if s.get("memo_path") else {"name": "build_briefing_memo", "input": {}}
            return None if s.get("report_path") else {"name": "generate_report", "input": {}}
        if agent == "statistician":
            low = (message or "").lower()
            # An indirect comparison needs the two direct effects, which the keyless
            # mock cannot read from prose: stop with the idle hint naming the inputs.
            if any(k in low for k in ("indirect", "bucher", "common comparator")):
                return None
            return None if s.get("stats_advice") else {"name": "recommend_statistics", "input": {}}
        if agent == "er_dose":
            return _mock_er_dose((message or "").lower(), s)
        if agent == "clinpharm":
            low = (message or "").lower()
            # keyword → calculator; the keyless mock passes the numbers it can find
            # as a best effort and the tool names any input still missing.
            table = (("calc_renal_function", ("creatinine", "crcl", "egfr", "ckd", "cockcroft", "renal")),
                     ("calc_be_sample_size", ("sample size", "how many subjects")),
                     ("calc_dose_regimen", ("loading dose", "maintenance dose")),
                     ("calc_accumulation", ("accumulation", "steady state", "steady-state")),
                     ("calc_allometric", ("allometric", "allometry", "body weight scaling")),
                     ("convert_concentration", ("micromolar", "µm", "umol", "mg/l", "molar mass",
                                                "molecular weight")),
                     ("quick_one_compartment", ("compartment profile", "simulate a one")),
                     ("calc_half_life", ("half-life", "half life", "rate constant", " ke ", "ke is",
                                         "elimination rate")))
            for name, keys in table:
                if any(k in low for k in keys):
                    if s.get("last_calculation") == name:
                        return None            # already answered this turn
                    return {"name": name, "input": _mock_numbers(low, name)}
            return None
        if agent == "simulator":
            low = (message or "").lower()
            # Every choice here is a PROPOSAL (expensive+proposable tools): the
            # keyless mock never confirms on the human's behalf and cannot
            # compose a `design`/`params` from prose -- the human fills/approves.
            if any(k in low for k in ("simulation-estimation", "simulation estimation",
                                      "simest", "sim-est")):
                return {"name": "run_simest", "input": {}}
            if "bootstrap" in low:
                return {"name": "run_bootstrap", "input": {}}
            if any(k in low for k in ("importance resampling", "sampling importance", " sir")):
                return {"name": "run_sir", "input": {}}
            if any(k in low for k in ("likelihood profil", "profile likelihood")):
                return {"name": "run_profile", "input": {}}
            return None
        if agent == "evidence":
            return _mock_evidence((message or "").lower(), s, {t.name for t in tools})
        return None
