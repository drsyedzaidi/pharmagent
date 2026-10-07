"""r6 numeric grammar, layer B: every code point at every boundary of every canonical expression.

Normal gate (a few seconds): the independent oracle on the reference characters, every LISTED code point (209) at
every boundary of every expression read directly, a stratified 2 048-point outside subset through the batch, and the
golden file that pins the 28 class representatives.

Full run (``R6_EXHAUSTIVE=1 pytest -m exhaustive tests/test_evidence_r6_exhaustive.py``, or
``python -m tests.evidence_r6_oracle --workers 12 --log run.log``): all 1 112 064 code points x every pair on a
process pool, ~10 minutes on 12 workers. A failure names the pair and the first offending code points.
"""
from __future__ import annotations

import os
import random
import re
from pathlib import Path

import pytest

from app.evidence.numgrammar import taint, tokens
from tests import evidence_r6_oracle as O
from tests import evidence_r6_pins as P

STRATIFIED = 2048


@pytest.fixture(scope="module")
def golden():
    g = O.load_golden()
    assert g is not None, "tests/evidence_r6_golden.json is missing: python -m tests.evidence_r6_oracle --regen-golden"
    return g


@pytest.fixture(scope="module")
def sample():
    return O.stratified_outside(STRATIFIED)


# ------------------------------------------------------------------ the expression set ---
def test_the_expression_set_covers_the_specified_forms_and_about_260_pairs():
    names = {e.name for e in O.EXPRESSIONS}
    for needed in ("n_eq_12", "cmax_dot", "cmax_comma", "auc_thousands", "auc_thin", "neg_hyphen", "neg_minus",
                   "neg_endash", "percent", "e_notation", "sci_x_caret", "sci_times_caret_neg", "sci_times_sup",
                   "bare_power", "range", "pm", "qual_glued", "qual_spaced", "tilde_operator", "dose_mg", "lancet",
                   "degree", "sbp_table", "mixed_number", "pua_inside", "braille_qualifier", "ten_dose", "mult_daily",
                   "detached", "lookalike_ten", "cid_inside", "cid_lead", "glued_unit", "plus_glued",
                   "cell_thousands", "cell_12_145", "cell_bad_groups", "cell_pm_groups", "cell_lancet", "cell_stars"):
        assert needed in names, needed
    assert 250 <= len(O.PAIRS) <= 320
    assert sum(1 for e in O.EXPRESSIONS if e.mode == "cell") == 6
    for e in O.EXPRESSIONS:
        assert all(0 <= b <= len(e.text) for b in e.cuts) and e.cuts == tuple(sorted(set(e.cuts))), e.name


def test_the_clean_expressions_read_as_documented():
    want = {"n_eq_12": [12.0], "cmax_dot": [12.5], "cmax_comma": [12.5], "auc_thousands": [1234.0], "auc_thin": [18900.0],
            "neg_hyphen": [-20.0], "neg_minus": [-20.0], "neg_endash": [-20.0], "percent": [45.0],
            "e_notation": [0.0012], "sci_x_caret": [1200.0], "sci_times_caret_neg": [0.0012],
            "sci_times_sup": [0.0012], "bare_power": [1e6], "range": [1.95, 2.74], "pm": [5.3, 0.4],
            "qual_glued": [0.5], "qual_spaced": [0.5], "tilde_operator": [45.0], "dose_mg": [150.0],
            "lancet": [12.4], "degree": [37.0], "sbp_table": [-11.8, -14.2], "mixed_number": [None],
            "ten_dose": [None], "mult_daily": [None], "detached": [None], "lookalike_ten": [None],
            "glued_unit": [None], "plus_glued": [None], "pua_inside": [None], "braille_qualifier": [None],
            "cid_inside": [None], "cid_lead": [None], "cell_thousands": [3412.0, 812.0], "cell_12_145": [12.0, 145.0],
            "cell_bad_groups": [None], "cell_pm_groups": [18900.0, 4100.0], "cell_lancet": [12.4],
            "cell_stars": [-14.2], "line_n_eq_12": [12.0], "line_cmax": [12.5], "line_sci": [1200.0]}
    for e in O.EXPRESSIONS:
        assert [t.value for t in O.run(e, e.text)] == want[e.name], e.name
    assert O.EXPRESSIONS[[e.name for e in O.EXPRESSIONS].index("cmax_comma")].style == ","


# ----------------------------------------------------------- the independent oracle ---
def test_the_reference_outside_line_and_cid_insertions_satisfy_the_oracle_at_every_boundary():
    bad = [(O.EXPRESSIONS[i].name, b, O.reference_errors(O.EXPRESSIONS[i], b)[:2]) for i, b in O.PAIRS
           if O.reference_errors(O.EXPRESSIONS[i], b)]
    assert bad == []


def test_every_listed_code_point_at_every_boundary(golden):
    """209 listed code points (inside, context, line, gap, equivalents) x every boundary, read directly."""
    cps = O.non_outside_cps()
    assert len(cps) == 45 + 8 + 4 + 74 + 2 + 76                     # equivalents counted with their target class
    bad = []
    for ei, b in O.PAIRS:
        e = O.EXPRESSIONS[ei]
        for cp in cps:
            bad += [f"{e.name}@{b}: {m}" for m in O.check_class(e, b, cp, golden)]
    assert bad[:10] == []


def test_a_stratified_outside_subset_reads_exactly_like_the_reference_outside_character(sample):
    assert 1000 <= len(sample) <= STRATIFIED
    for must in (0x00, 0x0B, 0x0C, 0x85, 0xAD, 0x200B, 0x200D, 0x2028, 0x2029, 0x202E, 0x2060, 0x2066, 0xFEFF, 0xFFFC,
                 0xFFFD, 0xE000, 0x2800, 0x3000, 0xE0001, 0x0300, 0x20D0, 0x0661, 0x00BE, 0x2460):
        assert must in sample, hex(must)
    bad = []
    for ei, b in O.PAIRS:
        bad += O.batch_errors(O.EXPRESSIONS[ei], b, sample)
    assert bad[:10] == []


def test_the_golden_file_pins_every_class_representative_at_every_boundary(golden):
    fresh = O.build_golden()
    assert set(golden) == set(fresh) and len(golden) == len(O.PAIRS) * 28
    changed = [k for k in fresh if golden[k] != fresh[k]]
    assert changed[:10] == [], ("the reader changed for a class representative: review the diff, then "
                                "`python -m tests.evidence_r6_oracle --regen-golden`")


def test_the_representatives_are_exactly_the_28_behaviours():
    reps = {O.rep_of(chr(cp)) for cp in P.INSIDE_SET | P.CONTEXT_SET}
    assert len(reps) == 28
    assert O.rep_of("7") == "5" and O.rep_of("0") == "0" and O.rep_of("E") == "e" and O.rep_of("μ") == "m"


# ------------------------------------------------------------- the harness is honest ---
def test_batching_reads_each_variant_like_a_single_call():
    rnd = random.Random(6)
    outside = O.outside_cps()
    for ei, b in rnd.sample(O.PAIRS, 25):
        e = O.EXPRESSIONS[ei]
        cps = rnd.sample(outside, 60)
        stride = len(e.text) + 1 + len(O.SEP)
        got = tokens(O.SEP.join(O.variants(e, b, chr(cp)) for cp in cps), e.style)
        for k, cp in enumerate(cps):
            single = [(t.start, t.end, t.value, t.shape, t.parts, t.detail) for t in O.run(e, O.variants(e, b, chr(cp)))]
            batch = [(t.start - k * stride, t.end - k * stride, t.value, t.shape, t.parts, t.detail)
                     for t in got if t.start // stride == k]
            assert batch == single, (e.name, b, hex(cp))


def test_a_cell_reads_like_a_page_when_it_is_one_line():
    for e in O.EXPRESSIONS:
        if e.mode != "cell":
            continue
        for b in e.cuts:
            for ch in ("x", "\u200b", "†", "5", "-", " ", "\u2028"):
                s = O.variants(e, b, ch)
                assert tokens(s, e.style, mode="cell") == tokens(s, e.style, mode="page"), (e.name, b, ch)


@pytest.mark.parametrize("label, patch", [
    ("the line tier is blind to U+2029", lambda: ("LINE_TIER_RE", "\u2029")),
    ("the word tier is blind to U+00AE", lambda: ("WORD_TIER_RE", "®")),
])
def test_the_sweep_catches_a_reader_that_ignores_a_character(label, patch, sample, monkeypatch):
    attr, ch = patch()
    original = getattr(taint, attr)
    monkeypatch.setattr(taint, attr, re.compile(original.pattern.replace("[^", "[^" + ch, 1)))
    found = []
    for ei, b in O.PAIRS[:60]:
        found += O.sweep_pair((ei, b), sample, None, direct=[]).errors
        found += [f"{m}" for m in O.check_class(O.EXPRESSIONS[ei], b, ord("®"), None)] if attr == "WORD_TIER_RE" \
            else []
    assert found, label


def test_the_sweep_catches_a_changed_class_representative(monkeypatch, golden):
    from app.evidence.numgrammar import scan
    monkeypatch.setattr(scan, "QUALIFIERS", scan.QUALIFIERS - {"~"})
    found = [m for ei, b in O.PAIRS[:60] for m in O.check_class(O.EXPRESSIONS[ei], b, 0x7E, golden)]
    assert found


# ------------------------------------------------------------------- the full run ---
@pytest.mark.exhaustive
@pytest.mark.skipif(os.environ.get("R6_EXHAUSTIVE") != "1", reason="set R6_EXHAUSTIVE=1 for the 1.1 M code-point run")
def test_every_code_point_at_every_boundary_on_a_process_pool():
    workers = int(os.environ.get("R6_WORKERS", max(1, min(12, (os.cpu_count() or 2) - 2))))
    log = Path(os.environ["R6_EXHAUSTIVE_LOG"]) if os.environ.get("R6_EXHAUSTIVE_LOG") else None
    result = O.full_run(workers, log)
    assert result["failures"] == [] and result["errors"] == 0, result
    assert result["pairs"] == len(O.PAIRS)
    assert result["variants"] == len(O.PAIRS) * 1_112_064
