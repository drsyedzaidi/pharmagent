"""r6 numeric grammar: the verify() path oracle (enabled by the r6 migration).

The stratified sample (about 195 000 verify() calls, ~45 s on one core) runs on every full pass of this file: the callers
read numbers with numgrammar only, so a pass where the clean record fails, a pass with another value, and any OUTSIDE code
point in the page or the snippet that passes are all defects. ``R6_VERIFY_ORACLE=0`` skips the long samples (the harness
checks always run); ``R6_VERIFY_CPS`` sets the number of code points (default 150).
"""
from __future__ import annotations

import os

import pytest

from tests import evidence_r6_verify_oracle as V

ENABLED = os.environ.get("R6_VERIFY_ORACLE") != "0"


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
    assert classes == {"inside", "gap", "context", "outside"}                    # r7: the LINE list is deleted
    assert V.describe(0x200B).startswith("U+200B")


@pytest.fixture(scope="module")
def sample():
    """ONE run of the stratified sample (~195 000 verify() calls) with the strong rule; both tests read it."""
    return V.run_sample(None, int(os.environ.get("R6_VERIFY_CPS", "150")), strong=True)


@pytest.mark.skipif(not ENABLED, reason="the long verify() sample is off: R6_VERIFY_ORACLE=0")
def test_verify_never_passes_where_the_clean_record_fails_or_with_another_value(sample):
    count, bad = sample
    assert count > 100_000
    assert [v for v in bad if "outside" not in v.message][:10] == []


@pytest.mark.skipif(not ENABLED, reason="the long verify() sample is off: R6_VERIFY_ORACLE=0")
def test_an_outside_code_point_in_the_page_or_the_snippet_is_never_a_pass(sample):
    assert [v for v in sample[1] if "outside" in v.message][:10] == []
