"""The verify() path oracle of the r6 numeric grammar (PREPARED: the migration writer enables it).

The tokenizer oracle (evidence_r6_oracle.py) proves what the reader does with every code point. This module proves what
``verify()`` does with the readings: for a stratified sample of code points c, each (expression, boundary) is run as a text
record (the expression inside a sentence) and as a table-cell record (the expression is the cell), with c in the page
only, in the snippet only and in both, and the outcome is judged against the clean outcome:

- never a pass where the clean record fails;
- a pass carries the clean number and the clean literal (c removed when it sits inside the literal's span);
- a fail carries a reason from ``models.REASON_CODES``;
- (strong) a code point of the OUTSIDE class anywhere in the page or the snippet is never a pass: the readability
  pre-check refuses the snippet and the tokenizer taints the page line.

Until the callers are switched the old reader fails the strong rule (that is the defect), so the strong rule is off by
default and ``R6_VERIFY_STRONG=1`` turns it on; the weak rules and the harness itself run today.
"""
from __future__ import annotations

import os
import unicodedata
from collections.abc import Iterator
from typing import NamedTuple

from app.evidence import models as M
from tests import evidence_r6_oracle as O
from tests import evidence_r6_pins as P
from tests.evidence_r5b_support import check

# expression name -> the value a proposal cites from it (a pass is expected where the clean record passes)
CITES = {
    "n_eq_12": "12", "cmax_dot": "12.5", "cmax_comma": "12.5", "auc_thousands": "1234", "auc_thin": "18900",
    "neg_hyphen": "-20", "percent": "45", "e_notation": "0.0012", "sci_x_caret": "1200", "bare_power": "1000000",
    "range": "1.95", "pm": "5.3", "qual_glued": "0.5", "dose_mg": "150", "lancet": "12.4", "degree": "37",
    "mixed_number": "12", "ten_dose": "10", "mult_daily": "2", "glued_unit": "150", "cell_thousands": "3412",
    "cell_12_145": "145", "cell_lancet": "12.4", "cell_stars": "-14.2",
}
WHERES = ("page", "snippet", "both")
MODES = ("text", "cell")


class Case(NamedTuple):
    name: str
    boundary: int
    cp: int
    where: str
    mode: str


class Violation(NamedTuple):
    case: Case
    message: str


def sample_cps(n: int = 150) -> list[int]:
    """Deterministic sample of about n code points: half from the listed ones, half from the outside strata."""
    half = max(n // 2, 1)
    listed, outside = O.non_outside_cps(), O.stratified_outside(2048)
    return sorted(set(listed[:: max(1, len(listed) // half)]) | set(outside[:: max(1, len(outside) // half)]))


def _expr(name: str) -> O.Expr:
    return next(e for e in O.EXPRESSIONS if e.name == name)


def _run(value: str, snippet: str, page: str, mode: str) -> M.VerifyResult:
    return check(value, snippet, page, cell=(mode == "cell"))


def cases(names: list[str] | None = None, cps: list[int] | None = None) -> Iterator[Case]:
    """Every (expression, boundary, code point, where, record kind) of the sample."""
    for name in names or list(CITES):
        for b in _expr(name).cuts:
            for cp in cps or []:
                for where in WHERES:
                    for mode in MODES:
                        yield Case(name, b, cp, where, mode)


def judge(case: Case, strong: bool) -> list[Violation]:
    e = _expr(case.name)
    value = CITES[case.name]
    c = chr(case.cp)
    inserted = O.variants(e, case.boundary, c)
    snippet = inserted if case.where in ("snippet", "both") else e.text
    page = inserted if case.where in ("page", "both") else e.text
    clean = _run(value, e.text, e.text, case.mode)
    got = _run(value, snippet, page, case.mode)
    out: list[str] = []
    if got.status not in ("pass", "fail"):
        out.append(f"status {got.status!r}")
    if got.status == "fail" and got.reason not in M.REASON_CODES:
        out.append(f"reason {got.reason!r} is not a known code")
    if got.status == "pass":
        if clean.status != "pass":
            out.append("passed where the clean record fails")
        else:
            if round(got.number, 12) != round(clean.number, 12):
                out.append(f"passed with another value: {got.number} != {clean.number}")
            literal = (got.literal or "").replace(c, "")
            if literal != (clean.literal or "") and c not in (got.literal or ""):
                out.append(f"literal {got.literal!r} differs from the clean {clean.literal!r}")
    if strong and P.expected_class(case.cp) == "outside" and got.status == "pass":
        out.append("an outside character is never a pass")
    return [Violation(case, m) for m in out]


def run_sample(names: list[str] | None = None, n: int = 150, *, strong: bool | None = None) -> tuple[int, list[Violation]]:
    strong = bool(os.environ.get("R6_VERIFY_STRONG") == "1") if strong is None else strong
    count, bad = 0, []
    for case in cases(names, sample_cps(n)):
        count += 1
        bad += judge(case, strong)
    return count, bad


def describe(cp: int) -> str:
    return f"U+{cp:04X} {unicodedata.name(chr(cp), '?')}"
