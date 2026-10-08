"""r6 round 2, finding 5: the normal gate fails when ``app/evidence`` (or the oracle) changed after the last full run of the
exhaustive tier. The record is ``tests/evidence_r6_attestation.json`` (see ``evidence_r6_attest.py``)."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tests import evidence_r6_attest as A
from tests import evidence_r6_layer_b as B
from tests import evidence_r6_neighbour_spec as NS
from tests import evidence_r6_oracle as O

HOW = ("the exhaustive tier has not been run on this code: commit, then `R6_EXHAUSTIVE=1 python -m pytest -m exhaustive "
       "tests/test_evidence_r6_exhaustive.py tests/test_evidence_r6_scope.py` (about 16 minutes), then commit "
       "tests/evidence_r6_attestation.json")
GOOD_TOKENS = {"errors": 0, "pairs": 291, "variants": 291 * A.CODE_POINTS, "head": "a" * 40, "dirty": []}
SWEEPS = ("scale", "style", "thin", "text")
GOOD_LAYER_B = {"errors": 0, "code_points": {n: A.CODE_POINTS for n in SWEEPS}, "head": "a" * 40, "dirty": []}


def good(digest: str = "d" * 64) -> dict:
    return copy.deepcopy({"digest": digest, "tokens": GOOD_TOKENS, "layer_b": GOOD_LAYER_B})


def problems(data, **kw):
    return A.problems(data, pairs=291, sweeps=SWEEPS, now="d" * 64, ancestor=lambda head: True, **kw)


# ------------------------------------------------------------------------- the gate ---

def test_both_full_runs_passed_on_exactly_the_code_of_this_checkout():
    got = A.problems(A.load(), pairs=len(O.PAIRS), sweeps=tuple(B.SWEEPS), neighbours=len(NS.vocabulary()))
    assert got == [], f"{HOW}\n" + "\n".join(got)


GOOD_NEIGHBOURS = {"errors": 0, "words": 30825, "variants": 1911150, "head": "a" * 40, "dirty": []}


def test_the_neighbour_sweep_is_required_when_the_gate_asks_for_it_and_every_shortfall_is_reported():
    """r7 writer 3: the third record. Without ``neighbours`` the old two-record check is unchanged."""
    def check(data):
        return A.problems(data, pairs=291, sweeps=SWEEPS, now="d" * 64, ancestor=lambda head: True, neighbours=30825)
    data = {**good(), "neighbours": dict(GOOD_NEIGHBOURS)}
    assert check(data) == [] and problems(good()) == []
    assert any("neighbour-shape sweep" in p and "no passing" in p for p in check(good()))
    for field, value, needle in (("errors", 2, "neighbour-shape sweep covered"), ("words", 5, "neighbour-shape sweep covered"),
                                 ("dirty", ["x"], "neighbours run was made on a dirty"),
                                 ("head", "b" * 40, "different commits")):
        bad = {**good(), "neighbours": {**GOOD_NEIGHBOURS, field: value}}
        assert any(needle in p for p in check(bad)), (field, check(bad))


def test_the_gate_reads_the_committed_file_and_the_inputs_cover_the_reader_and_the_oracle():
    names = {p.relative_to(A.BACKEND).as_posix() for p in A.inputs()}
    assert {"app/evidence/verify.py", "app/evidence/normalize.py", "app/evidence/unit_norm.py",
            "app/evidence/numgrammar/alphabet.py", "app/evidence/numgrammar/tokenizer.py",
            "tests/evidence_r6_oracle.py", "tests/evidence_r6_layer_b.py", "tests/evidence_r6_pins.py",
            "tests/evidence_r6_golden.json", "tests/test_evidence_r6_exhaustive.py",
            "tests/test_evidence_r6_scope.py", "tests/evidence_r6_attest.py"} <= names
    assert "tests/evidence_r6_attestation.json" not in names and not any("__pycache__" in n for n in names)


@pytest.mark.parametrize("module", ["test_evidence_r6_exhaustive.py", "test_evidence_r6_scope.py"])
def test_each_exhaustive_test_writes_the_attestation_when_it_passes_on_a_clean_tree(module):
    src = (Path(__file__).parent / module).read_text("utf-8")
    assert "evidence_r6_attest" in src and "record(" in src


# -------------------------------------------------------------------------- the machinery ---

def test_the_digest_follows_the_content_of_every_input(tmp_path, monkeypatch):
    a, b = tmp_path / "app" / "evidence" / "a.py", tmp_path / "tests" / "evidence_r6_b.py"
    for p in (a, b):
        p.parent.mkdir(parents=True)
        p.write_text("x = 1\n")
    monkeypatch.setattr(A, "BACKEND", tmp_path)
    monkeypatch.setattr(A, "inputs", lambda: [a, b])
    first = A.digest()
    assert A.digest() == first
    a.write_text("x = 2\n")
    assert A.digest() != first
    a.write_text("x = 1\n")
    assert A.digest() == first
    b.write_text("x = 1 \n")
    assert A.digest() != first


def test_a_run_is_recorded_with_the_digest_and_a_record_for_other_code_is_dropped(tmp_path, monkeypatch):
    target = tmp_path / "att.json"
    monkeypatch.setattr(A, "digest", lambda: "one")
    A.record("tokens", {"errors": 0}, target)
    A.record("layer_b", {"errors": 0}, target)
    assert json.loads(target.read_text()) == {"digest": "one", "layer_b": {"errors": 0}, "tokens": {"errors": 0}}
    monkeypatch.setattr(A, "digest", lambda: "two")
    A.record("tokens", {"errors": 0, "again": True}, target)
    assert json.loads(target.read_text()) == {"digest": "two", "tokens": {"errors": 0, "again": True}}
    with pytest.raises(ValueError):
        A.record("other", {}, target)
    assert A.load(tmp_path / "missing.json") is None
    (tmp_path / "bad.json").write_text("[1]")
    assert A.load(tmp_path / "bad.json") is None


def test_a_complete_attestation_of_this_code_has_no_problem():
    assert problems(good()) == []


@pytest.mark.parametrize("mutate, needle", [
    (lambda d: d.update(digest="e" * 64), "changed after the last full run"),
    (lambda d: d.pop("tokens"), "token sweep"),
    (lambda d: d.pop("layer_b"), "layer-B"),
    (lambda d: d["tokens"].update(errors=1), "token sweep covered"),
    (lambda d: d["tokens"].update(pairs=290), "token sweep covered"),
    (lambda d: d["tokens"].update(variants=1), "token sweep covered"),
    (lambda d: d["layer_b"].update(errors=2), "layer-B sweeps covered"),
    (lambda d: d["layer_b"]["code_points"].pop("text"), "layer-B sweeps covered"),
    (lambda d: d["layer_b"]["code_points"].update(style=5), "layer-B sweeps covered"),
    (lambda d: d["tokens"].update(dirty=["backend/app/evidence/verify.py"]), "dirty"),
    (lambda d: d["layer_b"].update(head=""), "dirty or unknown"),
    (lambda d: d["layer_b"].update(head="b" * 40), "different commits"),
])
def test_every_way_the_record_can_fall_short_is_reported(mutate, needle):
    data = good()
    mutate(data)
    assert any(needle in p for p in problems(data)), (needle, problems(data))


def test_no_record_at_all_is_a_problem_with_the_instruction():
    assert any("no attestation" in p for p in problems(None)) and any("no attestation" in p for p in problems({}))


def test_a_run_from_another_branch_is_reported_when_git_can_tell():
    assert any("not an ancestor" in p for p in A.problems(good(), pairs=291, sweeps=SWEEPS, now="d" * 64,
                                                          ancestor=lambda head: False))
    assert A.problems(good(), pairs=291, sweeps=SWEEPS, now="d" * 64, ancestor=lambda head: None) == []


def test_git_ancestor_is_none_for_an_unknown_commit_and_true_for_head():
    import subprocess
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=A.BACKEND, capture_output=True, text=True).stdout.strip()
    assert A.git_ancestor("0" * 40) in (None, False)
    if head:
        assert A.git_ancestor(head) is True
