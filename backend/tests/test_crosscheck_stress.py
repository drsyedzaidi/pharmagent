"""Stress table (noisy terminal phases, trailing/middle/leading BLQ, no terminal phase, all-BLQ): what PharmAgent's NCA
does, pinned. 'Characterisation' tests record CURRENT behaviour so a change cannot go unnoticed. The zero-coded BLQ
defect (AUC_last integrated through BLQ zero rows) is fixed on this branch. Nothing here runs R."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

DIR = Path(__file__).resolve().parents[1] / "scripts" / "crosscheck_pknca"
HASHES = {
    "stress_zero.csv": "65bc99f7befa88dd2269321524f9b684af3a7e04b71dc75d7b51375e4d63818d",
    "stress_dropped.csv": "8e2eb7e98173f2345cea6be65711f716525adc5a52f2e186f8f920e2e244140d",
    "stress_truth.json": "55fd17b68c9fc129681bae81859939ea322ef22eaf283dee41768e94c3812b7b",
}


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


run_pa, ref_np, compare, truth_check, make_stress = (
    _load(n) for n in ("run_pharmagent", "reference_numpy", "compare", "truth_check", "make_stress_table"))


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    out = {}
    for coding in ("zero", "dropped"):
        d = tmp_path_factory.mktemp(coding)
        assert run_pa.main(["x", str(DIR / f"stress_{coding}.csv"), str(d)]) == 0
        assert ref_np.main(["x", str(DIR / f"stress_{coding}.csv"), str(d)]) == 0
        out[coding] = {"pa": compare.load(d / "pharmagent.csv"), "ref": compare.load(d / "numpy_reference.csv"),
                       "pa_file": d / "pharmagent.csv", "ref_file": d / "numpy_reference.csv"}
    return out


# ── the data ────────────────────────────────────────────────────────────────────────────────────
def test_committed_stress_files_are_the_pinned_ones():
    for name, sha in HASHES.items():
        assert hashlib.sha256((DIR / name).read_bytes()).hexdigest() == sha, name


def test_generator_is_deterministic_in_structure():
    zero, dropped, truth = make_stress.build()
    zero2, dropped2, truth2 = make_stress.build()
    assert (zero, dropped, truth) == (zero2, dropped2, truth2)
    assert len(truth["subjects"]) == 11
    assert not any(r["BLQ"] == "1" for r in dropped)                       # dropped coding has no BLQ rows at all
    assert {r["BLQ"] for r in zero} == {"0", "1"}


def test_the_all_blq_scenario_really_is_all_below_the_limit():
    mine = [r for r in _rows(DIR / "stress_zero.csv") if r["ID"] == "S09_all_blq" and r["EVID"] == "0"]
    assert mine and all(r["DV"] == "0" and r["BLQ"] == "1" for r in mine)


# ── blank/dropped coding: PharmAgent == the textbook, and close to the analytic truth ──────────
def test_with_blq_rows_absent_pharmagent_matches_the_textbook_on_every_area(results):
    res = compare.compare(results["dropped"]["pa"], results["dropped"]["ref"])
    areas = [r for r in res["rows"] if r["param"] in ("Cmax", "Tmax", "AUClast", "AUCinf.obs")]
    assert areas and all(r["status"] == "match" for r in areas), [r for r in areas if r["status"] != "match"]


def test_with_blq_rows_absent_noise_free_aucs_are_within_three_percent_of_the_analytic_truth(results):
    rows = truth_check.table(DIR / "stress_truth.json", [results["dropped"]["pa_file"]])
    assert len(rows) >= 5
    for r in rows:
        assert abs(r["PharmAgent_err_pct"]) < 3.0, r


# ── zero coding with a BLQ flag: rule M1 (fixed in this branch; before it these tests were a strict xfail) ──
def test_auclast_stops_at_tlast_when_blq_is_coded_as_zero(results):
    """Every subject except S05 (middle BLQ: the textbook reference integrates the zero, M1 excludes it) and S09 (nothing
    measurable) must agree with the textbook on AUClast and AUCinf."""
    res = compare.compare(results["zero"]["pa"], results["zero"]["ref"])
    bad = [r for r in res["rows"] if r["param"] in ("AUClast", "AUCinf.obs") and r["status"] == "DIFFERS"
           and r["id"] not in ("S05_middle_blq", "S09_all_blq")]
    assert not bad, bad


def test_pharmagent_ends_auc_last_at_its_own_reported_tlast(results):
    """The earlier defect: the excess over the textbook was the triangle from Tlast to the next zero row. Now zero."""
    for sid in ("S03_trailing_blq", "S08_two_points_after_tmax", "S01_clean"):
        pa, ref = results["zero"]["pa"][sid], results["zero"]["ref"][sid]
        assert pa["AUClast"] == pytest.approx(ref["AUClast"], rel=1e-5, abs=2e-6), sid


def test_a_middle_blq_row_is_excluded_so_s05_is_close_to_the_analytic_truth_in_both_codings(results):
    zero = truth_check.table(DIR / "stress_truth.json", [results["zero"]["pa_file"]])
    dropped = truth_check.table(DIR / "stress_truth.json", [results["dropped"]["pa_file"]])
    z = next(r for r in zero if r["id"] == "S05_middle_blq")["PharmAgent_err_pct"]
    d = next(r for r in dropped if r["id"] == "S05_middle_blq")["PharmAgent_err_pct"]
    assert abs(z) < 1.5 and abs(d) < 1.5 and z == pytest.approx(d, abs=0.5)      # was -11.7% before M1 was applied


def test_zero_coded_noise_free_aucs_are_within_three_percent_of_the_analytic_truth(results):
    rows = truth_check.table(DIR / "stress_truth.json", [results["zero"]["pa_file"]])
    assert len(rows) >= 5
    for r in rows:
        assert abs(r["PharmAgent_err_pct"]) < 3.0, r                          # S08 was +6.2% before the fix


def test_characterisation_all_blq_subject(results):
    """Current behaviour, not a verdict: coded as zeros the subject gets AUC_last = 0 (a number); with the rows absent the
    subject silently disappears from the result."""
    assert results["zero"]["pa"]["S09_all_blq"]["AUClast"] == 0.0
    assert "S09_all_blq" not in results["dropped"]["pa"]


def test_characterisation_terminal_window_ties_resolve_to_the_longest_window(results):
    """PharmAgent compares adjusted R^2 after rounding to 6 decimals, so near-exact exponentials tie and the first (longest)
    window wins; a strict float comparison picks 3 points. Both give lambda_z within 1e-3 here."""
    pa, ref = results["zero"]["pa"]["S01_clean"], results["zero"]["ref"]["S01_clean"]
    assert pa["lambda.z.n.points"] == 6 and ref["lambda.z.n.points"] == 3
    assert pa["lambda.z"] == pytest.approx(ref["lambda.z"], rel=1e-3)


# ── against PKNCA's committed outputs (an independent engine; R is not run here) ───────────────
REF = DIR / "reference_results"


def _pknca(folder: str, cfg: str):
    return compare.load(REF / folder / f"{cfg}.csv")


def test_zero_coded_auclast_matches_pknca_defaults_on_every_subject_including_middle_blq(results):
    """PKNCA default: BLQ first=keep, middle=drop, last=keep, AUClast to Tlast: the same as rule M1 for the area."""
    res = compare.compare(results["zero"]["pa"], _pknca("stress_zero", "pknca_default"))
    rows = [r for r in res["rows"] if r["param"] in ("Cmax", "AUClast")]
    assert len(rows) == 22 and all(r["status"] == "match" for r in rows), [r for r in rows if r["status"] != "match"]


def test_the_only_auc_disagreement_with_pknca_matched_is_the_middle_blq_convention(results):
    res = compare.compare(results["zero"]["pa"], _pknca("stress_zero", "pknca_matched"))
    off = {r["id"] for r in res["rows"] if r["param"] == "AUClast" and r["status"] == "DIFFERS"}
    assert off == {"S05_middle_blq"}                   # 'matched' keeps the middle zero; PharmAgent (M1) excludes it


def test_terminal_window_pharmagent_stays_within_one_percent_of_pknca_defaults_lambda_z(results):
    """Convention difference, characterised: PKNCA's default tolerance (1e-4) prefers MORE terminal points than PharmAgent's
    rounded-tie rule; lambda_z differs by < 1% and PharmAgent is not the farther one from the generating ke (noise-free)."""
    ref = _pknca("stress_zero", "pknca_default")
    truth = json.loads((DIR / "stress_truth.json").read_text())["subjects"]
    for sid, pa in results["zero"]["pa"].items():
        if pa.get("lambda.z") is None or ref[sid].get("lambda.z") is None:
            continue
        assert pa["lambda.z"] == pytest.approx(ref[sid]["lambda.z"], rel=1e-2), sid
        if truth[sid]["cv_noise"] == 0 and truth[sid]["ke"] and sid not in ("S11_terminal_bump", "S07_tmax_last"):
            ke = truth[sid]["ke"]
            assert abs(pa["lambda.z"] - ke) <= abs(ref[sid]["lambda.z"] - ke) + 1e-6, sid


def test_characterisation_no_time_zero_sample_pknca_refuses_the_auc_pharmagent_starts_at_the_first_sample(results):
    """Convention difference (F5), not judged: PKNCA returns NA for AUC when no sample is at the interval start; PharmAgent
    integrates from its first sample and does not impute C(0) = 0, so the dose-to-first-sample area is omitted."""
    ref = _pknca("stress_dropped", "pknca_default")
    for sid in ("S01_clean", "S02_noisy_terminal", "S03_trailing_blq"):
        assert ref[sid]["AUClast"] is None and results["dropped"]["pa"][sid]["AUClast"] > 0


def test_pharmaverse_first_run_stays_in_agreement_with_pknca(tmp_path):
    out = tmp_path / "pv"
    assert run_pa.main(["x", str(DIR / "nca_input.csv"), str(out)]) == 0
    mine = compare.load(out / "pharmagent.csv")
    for cfg in ("pknca_default", "pknca_matched"):
        res = compare.compare(mine, _pknca("pharmaverse", cfg))
        assert len(res["rows"]) == 48 and all(r["status"] == "match" for r in res["rows"]), cfg
