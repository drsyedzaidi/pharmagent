"""The pharmaverse PK fixtures stay exactly what their manifest says: pinned, byte-identical subsets, with the licence
material present. Nothing here fits or simulates anything."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import pytest

FIX = Path(__file__).parent / "fixtures" / "pharmaverse"
MANIFEST = json.loads((FIX / "MANIFEST.json").read_text(encoding="utf-8"))


def _rows(name: str) -> list[dict[str, str]]:
    with (FIX / name).open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_subset_files_match_the_manifest_hashes_and_row_counts():
    for name, meta in MANIFEST["subset"].items():
        data = (FIX / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == meta["sha256"], name
        assert len(_rows(name)) == meta["rows"], name


def test_sources_are_pinned_to_full_commit_shas():
    for src in MANIFEST["sources"].values():
        assert len(src["commit"]) == 40 and set(src["commit"]) <= set("0123456789abcdef")
        assert len(src["sha256"]) == 64


def test_subjects_are_exactly_the_documented_selection():
    both = set(MANIFEST["selection"]["subjects_in_both"])
    edge = MANIFEST["selection"]["pc_only_all_blq_subject"]
    assert len(both) == 6 and edge not in both
    assert {r["USUBJID"] for r in _rows("adppk_subset.csv")} == both
    assert {r["USUBJID"] for r in _rows("pc_subset.csv")} == both | {edge}


def test_the_edge_case_subject_is_entirely_below_the_limit_of_quantification():
    mine = [r for r in _rows("pc_subset.csv") if r["USUBJID"] == MANIFEST["selection"]["pc_only_all_blq_subject"]]
    assert mine and all("BLQ" in r["PCORRES"] for r in mine)


def test_the_other_subjects_have_quantifiable_samples():
    for sid in MANIFEST["selection"]["subjects_in_both"]:
        quant = [r for r in _rows("pc_subset.csv") if r["USUBJID"] == sid and "BLQ" not in r["PCORRES"]]
        assert len(quant) >= MANIFEST["selection"]["min_quantifiable_pc_rows"], sid


def test_adppk_is_nonmem_shaped_per_subject():
    rows = _rows("adppk_subset.csv")
    assert {"USUBJID", "EVID", "NFRLT", "AMT", "DV", "BLQFL", "DOSEA"} <= set(rows[0])
    for sid in MANIFEST["selection"]["subjects_in_both"]:
        mine = [r for r in rows if r["USUBJID"] == sid]
        assert {r["EVID"] for r in mine} == {"0", "1"}                      # observations and dose records
        assert len({r["DOSEA"] for r in mine}) == 1


def test_licence_material_and_notice_are_present():
    for name in ("LICENSE-pharmaversesdtm.md", "LICENSE-pharmaverseadam.md"):
        text = (FIX / name).read_text(encoding="utf-8")
        assert "Apache License" in text and "Version 2.0" in text
    notice = (FIX / "NOTICE.md").read_text(encoding="utf-8")
    for needle in ("Apache License 2.0", "Cytel", "Roche", "GlaxoSmithKline", "byte-identical", "simulated",
                   "not** a PK answer key", *[s["commit"] for s in MANIFEST["sources"].values()]):
        assert needle in notice, needle


SRC = Path(os.environ.get("PHARMAVERSE_SRC", "/nonexistent"))


@pytest.mark.skipif(not ((SRC / "pc_pinned.csv").exists() and (SRC / "adppk_pinned.csv").exists()),
                    reason="set PHARMAVERSE_SRC to a folder holding the pinned pc_pinned.csv and adppk_pinned.csv")
def test_rebuilding_from_the_pinned_sources_reproduces_the_fixtures(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("build_subset", FIX / "build_subset.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main(["x", str(SRC / "pc_pinned.csv"), str(SRC / "adppk_pinned.csv"), str(tmp_path)]) == 0
    for name in ("pc_subset.csv", "adppk_subset.csv"):
        assert (tmp_path / name).read_bytes() == (FIX / name).read_bytes()
    assert hashlib.sha256((SRC / "pc_pinned.csv").read_bytes()).hexdigest() == \
        MANIFEST["sources"]["pc"]["sha256"]
