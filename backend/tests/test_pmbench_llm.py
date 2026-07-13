"""LLM-adapter tests — all keyless via MockLLM.

The pipeline is prompt → text → parse. Two invariants pin it down:

* With the ``oracle`` strategy (a mock that "used the tools"), the round-trip is
  lossless — the parsed answer scores a perfect 1.0. This proves the JSON
  extraction and coercion never lose information.
* With the ``naive`` strategy, the LLM path reproduces the tool-free floor, so it
  matches ``naive_agent`` — proving the plumbing is faithful, not that any real
  model is good.
"""
import pytest

from pharmacometricsbench.agents import naive_agent
from pharmacometricsbench.generators import build_taskset
from pharmacometricsbench.grading import grade_task, score_report
from pharmacometricsbench.llm import (
    MockLLM,
    _extract_json_object,
    build_prompt,
    make_llm_agent,
    parse_answer,
)


def test_parser_prefers_last_json_fence():
    """A verbose model may show a worked example (with its own braces) before the
    committed answer; the LAST fence is the real answer, not an intermediate one."""
    text = (
        "Worked example for the format:\n```json\n{\"Cmax\": 0.0}\n```\n"
        "Now the actual result:\n```json\n{\"Cmax\": 5.2, \"t_half\": 3.1}\n```\n"
    )
    assert _extract_json_object(text) == {"Cmax": 5.2, "t_half": 3.1}


def test_parser_none_on_truncated_response():
    """If the response is cut off before any closing fence (the max_tokens bug),
    extraction returns None rather than a bogus partial — surfaced as a parse-gap."""
    assert _extract_json_object("Step 1: Cmax = 5.2 ...long working, no JSON yet") is None


@pytest.fixture(scope="module")
def tasks():
    return build_taskset(per_category=6)


def _report(agent, tasks):
    return score_report([grade_task(t, agent(t)) for t in tasks])


def test_oracle_strategy_roundtrips_lossless(tasks):
    agent = make_llm_agent(MockLLM(strategy="oracle"))
    assert _report(agent, tasks)["overall"] == 1.0


def test_naive_strategy_matches_naive_agent(tasks):
    agent = make_llm_agent(MockLLM(strategy="naive"))
    llm = _report(agent, tasks)
    direct = _report(naive_agent, tasks)
    # Same underlying math, delivered through the text/JSON channel.
    assert llm["overall"] == pytest.approx(direct["overall"], abs=1e-9)


def test_parser_survives_prose_and_fence(tasks):
    task = tasks[0]  # an NCA task
    text = (
        "Sure! After reviewing the data, my estimates are below.\n"
        '```json\n{"Cmax": 12.5, "AUC_inf": 140.0, "t_half": 5.1}\n```\n'
        "Let me know if you need anything else."
    )
    ans = parse_answer(text, task)
    assert ans == {"Cmax": 12.5, "AUC_inf": 140.0, "t_half": 5.1}


def test_parser_coerces_bool_and_ignores_junk(tasks):
    be_task = next(t for t in tasks if t.category == "be")
    text = '{"gmr_pct": 104.2, "within_limits": "yes", "note": "looks fine"}'
    ans = parse_answer(text, be_task)
    assert ans["within_limits"] is True
    assert ans["gmr_pct"] == 104.2
    assert "note" not in ans  # non-target keys dropped


def test_parser_returns_empty_on_no_json(tasks):
    assert parse_answer("I cannot answer this.", tasks[0]) == {}


def test_build_prompt_embeds_keys_and_data(tasks):
    task = tasks[0]
    p = build_prompt(task)
    assert "Cmax" in p and "AUC_inf" in p and "t_half" in p
    assert "CATEGORY: nca" in p and "DATA_JSON:" in p
