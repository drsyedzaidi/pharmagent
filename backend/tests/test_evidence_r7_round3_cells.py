"""r7 round 3: the verifier's table and two-column paths (findings of the round-3 audit).

* a table cell's number is judged by the neighbour-word allow-list over the cells around it (the cells one separator
  away: a sign, an opener, a decimal tail, a power, a slash or an operator letter in the next or previous cell refuses
  it like it does in text); a cell boundary stays structure for a signed list (owner decision 9);
* the page-text span of a table record is the cited cell on the cited row: the row's cells are matched as whole words in
  order, so a row label that is a prefix of another ('50 mg' / '50 mg/kg') or a duplicate value in the row lands on the
  cited column; a tie is no span ('row not located'), never the first match;
* the two-column checks and the row span read each page line once per verification (no quadratic re-normalising);
* the column split of a merged line does not depend on the hash seed: a line with two valid splits is ambiguous.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from app.evidence import models as M
from app.evidence import verify_columns as VCOL
from app.evidence.read_cache import read_cache
from app.evidence.verify import verify
from tests.evidence_r6_round1_support import cell
from tests.evidence_timing import limit

BACKEND = Path(__file__).resolve().parents[1]


def table_doc(rows: list[list[str]], text: str) -> dict:
    return {"decimal_style": ".", "pages": [{"text": text, "columns": [], "tables": [{"rows": rows, "strategy": "lines"}]}]}


def table_rec(value: str, r: int, c: int, snippet: str | None = None) -> dict:
    return {"value": value, "locator": {"page": 1, "kind": "table", "snippet": snippet or value,
                                        "cell": {"table": 0, "row": r, "col": c}}}


# ------------------------------------------------------------------------------ the cell window ---
@pytest.mark.parametrize("prev, cur, nxt, value", [
    ("(-", "12)", "ng/mL", "12"), ("-(", "12)", "ng/mL", "12"), ("- (", "12)", "ng/mL", "12"), ("A", "12", ".5", "12"),
    ("A", "0.", "5", "0"), ("A", "12", "^3", "12"), ("1.2 e", "-3", "B", "-3"), ("x", "10^3", "B", "10^3"),
    ("A", "12", "/15", "12")])
def test_a_neighbouring_cell_is_judged_by_the_neighbour_word_allow_list(prev, cur, nxt, value):
    got = cell([["P", "Q", "R"], [prev, cur, nxt]], 1, 1, value, snippet=cur, page="zzz")
    assert got.status == "fail" and got.reason in (M.REASON_UNRESOLVED_SHAPE, M.REASON_PARTIAL_NUMBER), got


@pytest.mark.parametrize("prev, cur, nxt, value, flag", [
    ("-", "12", "ng/mL", "12", M.FLAG_CENSORED_VALUE), ("A", "12.5", "(3.1)", "12.5", None),
    ("Cmax", "12.5", "13.6", "12.5", None), ("A", "12", "-15", "12", None), ("", "12.5", "", "12.5", None),
    ("A", "12.5", "mg", "12.5", None)])
def test_ordinary_neighbouring_cells_keep_the_number_and_a_cell_boundary_is_structure(prev, cur, nxt, value, flag):
    got = cell([["P", "Q", "R"], [prev, cur, nxt]], 1, 1, value, snippet=cur, page="zzz")
    assert got.status == "pass", got
    assert (flag in got.flags) if flag else M.FLAG_CENSORED_VALUE not in got.flags, got


# ------------------------------------------------------------------------------ the row span ---
def test_the_span_lands_on_the_cited_row_when_a_row_label_prefixes_another_and_cells_wrap():
    rows = [["Dose", "Tmax, h (SD)", "Cmax, ng/mL"], ["50 mg/kg", "1.5\n(0.5)", "120"], ["50 mg", "1.5\n(0.4)", "20"]]
    text = "Dose Tmax, h (SD) Cmax, ng/mL\n50 mg/kg 1.5 120\n(0.5)\n50 mg 1.5 20\n(0.4)\n"
    res = verify(table_rec("1.5", 2, 1), table_doc(rows, text))
    assert res.status == "pass", res
    s = res.locator["span"]
    assert s is not None and text[s[0]:s[1]] == "1.5" and text.rfind("\n", 0, s[0]) == text.index("50 mg 1.5") - 1


def test_the_span_lands_on_the_cited_column_of_a_duplicate_value():
    rows = [["Parameter", "Fasted", "Fed"], ["Tmax (h)", "1.5", "1.5"], ["t1/2 (h)", "21.5", "20.1"]]
    text = "Table 2\nParameter Fasted Fed\nTmax (h) 1.5 1.5\nt1/2 (h) 21.5 20.1\n"
    assert verify(table_rec("1.5", 1, 2), table_doc(rows, text)).locator["span"] == [42, 45]
    assert verify(table_rec("1.5", 1, 1), table_doc(rows, text)).locator["span"] == [38, 41]


def test_a_tie_between_two_lines_is_no_span():
    rows = [["Arm", "N"], ["A", "12"], ["A", "12"]]
    text = "Arm N\nA 12\nA 12\n"
    res = verify(table_rec("12", 2, 1), table_doc(rows, text))
    assert res.status == "pass" and res.locator["span"] is None, res


def test_a_row_fragment_needs_the_row_cells_as_whole_words():
    rows = [["Arm", "Dose", "N"], ["50 mg", "2", "70"]]
    text = "Arm Dose N\n50 mg/kg 281 170\n"              # the row is not on the page text: its cells only prefix words
    assert verify(table_rec("2", 1, 1), table_doc(rows, text)).status == "pass"
    text2 = "Arm Dose N\n50 mg 281 70\n"                # the row's line prints '281': a ruling split the number
    assert verify(table_rec("2", 1, 1), table_doc(rows, text2)).reason == M.REASON_PARTIAL_NUMBER


# ------------------------------------------------------------------------- linear cost per record ---
def _twocol(n: int) -> tuple[dict, dict]:
    left = "Left " + " ".join(["1.5"] * n)
    text = left + " right side words\n"
    return ({"value": "1.5", "locator": {"page": 1, "kind": "text", "snippet": "1.5"}},
            {"decimal_style": ".", "pages": [{"text": text, "columns": [left, "right side words"], "tables": []}]})


def test_the_two_column_checks_normalise_each_line_once(monkeypatch):
    calls: list[int] = []
    real = VCOL.normalise
    monkeypatch.setattr(VCOL, "normalise", lambda s: (calls.append(len(s)), real(s))[1])
    counts = []
    for n in (200, 400):
        calls.clear()
        verify(*_twocol(n))
        counts.append(len(calls))
    assert counts[1] <= counts[0] + 2, counts                  # each distinct line once: no growth with the hits


def test_a_long_merged_line_verifies_in_bounded_time():
    rec, doc = _twocol(16_000)
    t0 = time.perf_counter()
    with read_cache():
        verify(rec, doc)
    assert time.perf_counter() - t0 < limit(1.0)


def test_the_row_span_reads_each_line_once(monkeypatch):
    from app.evidence import verify_table as VT
    calls: list[int] = []
    real = VT.normalise
    monkeypatch.setattr(VT, "normalise", lambda s: (calls.append(len(s)), real(s))[1])
    counts = []
    for n in (200, 400):
        line = " ".join(["1.5"] * n)
        calls.clear()
        verify(table_rec("1.5", 1, 1), table_doc([["Dose", "Tmax"], ["A", "1.5"]], "Dose Tmax\n" + line + "\nA 1.5\n"))
        counts.append(len(calls))
    assert counts[1] <= counts[0] + 2, counts


# -------------------------------------------------------------------------- the hash seed ---
_SEED_POC = """
import json, sys
sys.path.insert(0, '.')
from app.evidence.verify import verify
doc = {'decimal_style': '.', 'pages': [{'text': 'Plasma bupropion levels rose\\n',
       'columns': ['Plasma\\nPlasma bupropion', 'bupropion levels rose\\nlevels rose'], 'tables': []}]}
r = verify({'value': 'Plasma bupropion', 'locator': {'page': 1, 'kind': 'text', 'snippet': 'Plasma bupropion'}}, doc,
           field_type='text')
print(json.dumps([r.status, r.locator.get('column'), r.locator.get('span')]))
"""


def test_the_column_split_does_not_depend_on_the_hash_seed():
    outs = {subprocess.run([sys.executable, "-c", _SEED_POC], cwd=BACKEND, capture_output=True, text=True,
                           env={**os.environ, "PYTHONHASHSEED": seed}, timeout=120).stdout for seed in ("1", "3", "4")}
    assert len(outs) == 1, outs
    assert '"pass", 0' in outs.pop()                              # the ambiguous merged line is read in the column
