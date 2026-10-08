"""r7 owner decision 8: Greek chi (U+03C7) is readable only in the explicit chi-square production.

    chi_square = chi ("\\u00b2" | "2") ["(" ["df" [gap] "=" [gap]] digit{1,3} ")"] [gap] "=" [gap]   + a number

Inside the production chi is a label letter (the number after ``=`` and the degrees of freedom read as usual); anywhere
else chi is outside: it taints its word, the words one separator away and its whole line. No look-alike is admitted: a
capital chi, a Cyrillic ha, a Latin x or an OCR substitution stays what it is.
"""
from __future__ import annotations

import pytest

from app.evidence import models as M
from app.evidence.numgrammar import tokens, unreadable
from tests.evidence_r6_round1_support import run

CHI = "\u03c7"


def V(text: str) -> list[float | None]:
    return [t.value for t in tokens(text)]


@pytest.mark.parametrize("text, want", [
    (f"{CHI}\u00b2 = 3.84", [3.84]), (f"{CHI}2 = 3.84", [3.84]), (f"{CHI}\u00b2=3.84", [3.84]),
    (f"LRT {CHI}\u00b2(1) = 6.63, p < 0.01", [1.0, 6.63, 0.01]),
    (f"{CHI}\u00b2(df=2) = 7.1 and p = 0.03", [2.0, 7.1, 0.03]),
    (f"{CHI}\u00b2(df = 2)= 7.1", [2.0, 7.1]),
    (f"\u0394{CHI}\u00b2 = 12.5 (p = 0.001)", [12.5, 0.001]),
])
def test_the_chi_square_form_reads_its_number_and_leaves_the_line_clean(text, want):
    assert V(text) == want
    assert unreadable(text) is None


@pytest.mark.parametrize("text", [
    f"\u0394OFV > 6.63 ({CHI}\u00b2 df=1, p<0.01): subject",       # the corpus line: no '=' right after the statistic
    f"*p < 0.05 ({CHI}\u00b2 threshold \u0394OFV > 3.84 for df=1)",
    f"compare nested models by \u0394OFV vs {CHI}\u00b2 threshold 3.84",
    f"AUC was 1.2 {CHI} 10^3 ng.h/mL", f"{CHI} = 3.84", f"{CHI}3 = 3.84", f"{CHI}\u00b2(1) 6.63",
    f"{CHI}\u00b2 = x 12", f"{CHI}\u00b2\u00b2 = 3.84", f"{CHI}^2 = 3.84",
    f"{CHI}\u00b2 = 3.84 and 1.2 {CHI} 10^3",                 # one production never clears another chi of the line
    "\u03a7\u00b2 = 3.84", "\u0445\u00b2 = 3.84", "\u03a72 = 3.84",   # capital chi, Cyrillic ha: never chi
])
def test_chi_anywhere_else_and_every_look_alike_leave_the_line_unresolved(text):
    got = tokens(text)
    assert got and all(t.value is None and t.reason == "unresolved_context" for t in got), (text, got)
    assert unreadable(text) is not None


def test_a_latin_x_squared_is_never_read_as_chi():
    """r7 round 3: was '(t,) = tokens("x\u00b2 = 3.84"); (t.value, t.shape) == (3.84, "decimal")'. A variable name before
    '=' is a label or one letter (glue.variable): 'x\u00b2' is an operator letter before a power, so the number after
    it is unresolved, never a chi-square reading (decision 8: a Latin x is never chi) and never a clean 3.84."""
    (t,) = tokens("x\u00b2 = 3.84")
    assert t.value is None and t.reason == "unresolved_shape" and "3.84" in "x\u00b2 = 3.84"[t.start:t.end]
    assert not any(t.value is not None for t in tokens("x\u00b2 = 3.84"))
    assert [x.value for x in tokens("x = 3.84")] == [3.84]           # one letter is still a variable name


def test_verify_reads_a_chi_square_record_and_refuses_one_outside_the_form():
    page = f"Likelihood ratio test {CHI}\u00b2 = 3.84 (p = 0.05) for the covariate"
    got = run(page, f"{CHI}\u00b2 = 3.84 (p = 0.05)", "3.84")
    assert (got.status, got.value_num) == ("pass", 3.84)
    bad = run(f"\u0394OFV > 6.63 ({CHI}\u00b2 df=1, p<0.01): subject", "6.63", "6.63")
    assert (bad.status, bad.reason) == ("fail", M.REASON_UNRESOLVED_CONTEXT)
