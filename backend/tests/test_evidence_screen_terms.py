"""Keyless screening semantics: keywords are alternatives; a criterion written as text (no keywords)
needs most of its content words, per 'or'-separated part, and says so in its note and reasoning."""
from __future__ import annotations

from app.evidence.screen_rules import criterion_alternatives, keyword_screen

PAGE_B = [{"n": 1, "text": "Exposure of drug Z in 24 adult patients at steady state in a clinical study.",
           "columns": None}]
PAGE_A = [{"n": 1, "text": "Healthy human adult volunteers received drug X; plasma PK was characterised.",
           "columns": None}]
PAGE_HUMAN_DISCUSSION = [{"n": 1, "text": "Healthy adult volunteers. Unlike earlier animal data, exposure was linear.",
                          "columns": None}]


def _crit(text: str, kind: str = "inclusion", keywords: list[str] | None = None) -> dict:
    return {"id": "C1", "text": text, "kind": kind, "required": True, "keywords": keywords or []}


def test_one_word_of_several_does_not_meet_a_text_criterion():
    out = keyword_screen([_crit("human adult PK study")], PAGE_B)
    entry = out["criteria"][0]
    assert entry["met"] is False and entry["note"] == "1 of 3 criterion words found (needs 2)"
    assert out["verdict"] != "include" and "needs most of its words" in out["reasoning"]


def test_most_words_meet_a_text_criterion_with_a_lower_score():
    out = keyword_screen([_crit("human adult PK study")], PAGE_A)
    entry = out["criteria"][0]
    assert entry["met"] is True and entry["note"] == "3 of 3 criterion words found (needs 2)"
    partial = keyword_screen([_crit("human adult renal study")], PAGE_A)
    assert partial["criteria"][0]["met"] is True and partial["score"] < out["score"]


def test_or_in_criterion_text_keeps_alternatives():
    alts, from_keywords = criterion_alternatives(_crit("in-vitro or animal study", kind="exclusion"))
    assert not from_keywords and alts == [["in-vitro"], ["animal"]]
    hit = keyword_screen([_crit("in-vitro or animal study", kind="exclusion")], PAGE_HUMAN_DISCUSSION)
    assert hit["criteria"][0]["met"] is True        # explicit 'or': either part meets it


def test_keywords_stay_alternatives():
    out = keyword_screen([_crit("human adult PK study", keywords=["adult", "children"])], PAGE_B)
    assert out["criteria"][0]["met"] is True and out["criteria"][0]["note"] == ""
