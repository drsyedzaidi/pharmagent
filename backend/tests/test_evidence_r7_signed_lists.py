"""r7 owner decision 9: ``-11.8 -14.2`` with no delimiter, table structure or explicit range marker stays FLAGGED.

Two signed numbers one gap apart are the shape of a range (``12 -15``): each of them is marked ``ranged`` by the reader
and a record that cites one raises the error flag ``censored_value`` (a range bound is not an exact value). The reader
never guesses that the page means a list. Separate table cells are table structure (no flag); an explicit range marker
(``to``) keeps the flag.
"""
from __future__ import annotations

import pytest

from app.evidence import models as M
from app.evidence.numgrammar import tokens
from tests.evidence_r6_round1_support import cell, run


@pytest.mark.parametrize("text", [
    "SystolicBP \u221211.8 \u221214.2 \u221219.9", "SystolicBP -11.8 -14.2", "SystolicBP \u201311.8* \u201314.2***",
    "change \u221211.8 \u221214.2 mmHg", "\u221211.8 \u221214.2",
])
def test_signed_numbers_one_gap_apart_are_all_marked_as_range_ends(text):
    got = [t for t in tokens(text) if t.value is not None]
    assert len(got) >= 2 and all(t.ranged for t in got), (text, got)
    assert [t.value for t in got][:2] == [-11.8, -14.2]


@pytest.mark.parametrize("value", ["-11.8", "-14.2"])
def test_a_record_citing_one_of_them_raises_censored_value(value):
    page = "SystolicBP \u221211.8 \u221214.2 \u221219.9 in the treated arm"
    got = run(page, page, value)
    assert got.status == "pass" and M.FLAG_CENSORED_VALUE in got.flags, got


def test_an_explicit_range_marker_keeps_the_flag_and_table_cells_are_structure():
    got = run("change from \u221211.8 to \u221214.2 mmHg", "change from \u221211.8 to \u221214.2 mmHg", "-11.8")
    assert got.status == "pass" and M.FLAG_CENSORED_VALUE in got.flags
    rows = [["Outcome", "Arm A", "Arm B"], ["SystolicBP", "\u221211.8", "\u221214.2"]]
    for c, value in ((1, "-11.8"), (2, "-14.2")):
        got = cell(rows, 1, c, value, snippet=rows[1][c])
        assert got.status == "pass" and M.FLAG_CENSORED_VALUE not in got.flags, (value, got)
