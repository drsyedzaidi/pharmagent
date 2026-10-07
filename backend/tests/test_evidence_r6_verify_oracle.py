"""r6 numeric grammar: the verify() path oracle (prepared; the migration writer enables it).

``R6_VERIFY_ORACLE=1 pytest tests/test_evidence_r6_verify_oracle.py`` runs the stratified sample against the verify() of the
day (about 195 000 records, ~1 minute on one core: run it on a pool when it joins the normal gate). While the callers still
use the old reader it reports the defects the grammar exists to close (a pass where the clean record fails, an outside
character that passes), so only the harness checks run by default. ``R6_VERIFY_STRONG=1`` adds the rule that an OUTSIDE
code point is never a pass.
"""
from __future__ import annotations

import os

import pytest

from tests import evidence_r6_verify_oracle as V

ENABLED = os.environ.get("R6_VERIFY_ORACLE") == "1"


def test_the_harness_runs_against_the_current_verify_and_classifies_every_outcome():
    cps = V.sample_cps(24)
    assert 12 <= len(cps) <= 60
    count, bad = V.run_sample(["cmax_dot", "sci_x_caret", "cell_thousands"], 24, strong=False)
    assert count > 1000
    assert all(isinstance(v, V.Violation) for v in bad)
    assert all(v.message for v in bad)


def test_the_judge_reads_a_pass_a_fail_and_a_wrong_pass_correctly():
    case = V.Case("cmax_dot", 0, ord("m"), "both", "text")
    assert V.judge(case, strong=True) == []                         # a letter before the sentence changes nothing
    e = V._expr("cmax_dot")
    assert V._run("12.5", e.text, e.text, "text").status == "pass"
    assert V._run("12.6", e.text, e.text, "text").status == "fail"


def test_the_sample_holds_listed_and_outside_code_points():
    from tests import evidence_r6_pins as P
    cps = V.sample_cps(150)
    classes = {P.expected_class(cp) for cp in cps}
    assert classes == {"inside", "gap", "context", "line", "outside"}
    assert V.describe(0x200B).startswith("U+200B")


@pytest.mark.skipif(not ENABLED, reason="enabled by the migration writer: R6_VERIFY_ORACLE=1")
def test_verify_never_passes_where_the_clean_record_fails_or_with_another_value():
    count, bad = V.run_sample(None, int(os.environ.get("R6_VERIFY_CPS", "150")), strong=False)
    assert count > 100_000
    assert bad[:10] == []


@pytest.mark.skipif(os.environ.get("R6_VERIFY_STRONG") != "1", reason="strong rule: R6_VERIFY_STRONG=1")
def test_an_outside_code_point_in_the_page_or_the_snippet_is_never_a_pass():
    count, bad = V.run_sample(None, int(os.environ.get("R6_VERIFY_CPS", "150")), strong=True)
    assert bad[:10] == []
