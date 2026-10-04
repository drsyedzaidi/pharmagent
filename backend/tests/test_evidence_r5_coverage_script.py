"""scripts/evidence_coverage.py: chunked runs merge their hit lines, and the report names what was missed."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evidence_coverage.py"
spec = importlib.util.spec_from_file_location("evidence_coverage_script", SCRIPT)
cov = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cov)


def test_missing_lines_are_reported_as_compact_ranges_and_ignore_none():
    assert cov._missing({3, 4, 5, 9, 12, 13, None}) == "3-5, 9, 12-13"
    assert cov._missing(set()) == ""


def test_merge_data_unions_lines_across_runs_and_rewrites_the_file(tmp_path):
    data = tmp_path / "cov.json"
    first = cov._merge_data({Path("/a.py"): {1, 2}}, data)
    assert first == {Path("/a.py"): {1, 2}} and json.loads(data.read_text()) == {"/a.py": [1, 2]}
    second = cov._merge_data({Path("/a.py"): {2, 3}, Path("/b.py"): {7}}, data)
    assert second == {Path("/a.py"): {1, 2, 3}, Path("/b.py"): {7}}
    assert json.loads(data.read_text()) == {"/a.py": [1, 2, 3], "/b.py": [7]}


def test_without_a_data_file_the_hits_pass_through_unchanged():
    hit = {Path("/a.py"): {1}}
    assert cov._merge_data(hit, None) is hit
