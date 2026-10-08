"""r7 writer 2: the two new layer-B sweeps (``unitprod``, ``label``; tests/evidence_r6_unit_spec.py) have teeth.

The normal gate runs them on the stratified sample (test_evidence_r6_scope.py); the exhaustive run takes every code point.
Here each one is shown to catch the reader it guards against: a unit production that accepts any letters, the r6 rule that
read a label's trailing digits as a number before the next word, and the r7 writer-1 citation rule after a label.
"""
from __future__ import annotations

import re

from app.evidence.numgrammar import productions, tokenizer, units
from app.evidence.numgrammar import words as W
from tests import evidence_r6_oracle as O
from tests import evidence_r6_unit_spec as U

SAMPLE = sorted(set(O.non_outside_cps()) | {0x200B, 0xFFFD, 0x2014, 0xE000})


def test_the_specification_reads_the_owner_examples():
    assert U.spec_unit("mg") == 2 and U.spec_unit("m2") == 2 and U.spec_unit("mL/min") == 6 and U.spec_unit("h^-1") == 4
    assert U.spec_unit("OmL") == U.spec_unit("l") == U.spec_unit("mgtid") == U.spec_unit("d") == U.spec_unit("s") == -1
    assert U.spec_unit("mg)x") == 2 and U.spec_unit("mg)2") == -1
    assert U.spec_split("m", "L") and not U.spec_split("mg", "kg")


def test_both_sweeps_pass_on_the_listed_code_points():
    assert U.sweep_unitprod(SAMPLE) == [] and U.sweep_label(SAMPLE) == []


def test_the_unit_sweep_catches_a_production_that_accepts_any_letters(monkeypatch):
    monkeypatch.setattr(units, "UNIT_RE", re.compile("[A-Za-z]+(?=\\Z|[)\\],;:.'(\\[])"))
    assert U.sweep_unitprod(SAMPLE)


def test_the_unit_sweep_catches_a_reader_that_ignores_an_outside_character(monkeypatch):
    original = units.UNIT_RE
    monkeypatch.setattr(units, "UNIT_RE", re.compile(original.pattern.replace("(?=\\Z|", "​?(?=\\Z|", 1)))
    import app.evidence.numgrammar.taint as T
    monkeypatch.setattr(T, "OUTSIDE_RE", re.compile(T.OUTSIDE_RE.pattern.replace("[^", "[^​", 1)))
    assert U.sweep_unitprod([0x200B])


def test_the_label_sweep_catches_label_digits_read_as_a_number_before_the_next_word(monkeypatch):
    def old(self, i, drop=0):
        w = self.text(i)
        return W.ends_with_number(w[:len(w) - drop])
    monkeypatch.setattr(tokenizer.Reader, "number_end", old)
    assert U.sweep_label(SAMPLE)


def test_the_label_sweep_catches_the_citation_rule_that_ignored_labels(monkeypatch):
    monkeypatch.setattr(productions, "_label_end", lambda *args: False)
    assert U.sweep_label(SAMPLE)
