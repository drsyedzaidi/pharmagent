"""Stress table (noisy terminal phases, trailing/middle/leading BLQ, no terminal phase, all-BLQ): what PharmAgent's NCA
does, pinned. 'Characterisation' tests record CURRENT behaviour so a fix cannot go unnoticed; the strict xfail states the
behaviour the standard AUClast definition requires. Nothing here runs R."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
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


# ── zero coding: the BLQ flag is detected but the zero rows are used as measurements ──────────
@pytest.mark.xfail(strict=True, reason="PharmAgent integrates AUC_last through trailing BLQ rows coded as zero, past its "
                                       "own reported Tlast; the standard AUClast ends at the last measurable value")
def test_auclast_stops_at_tlast_when_blq_is_coded_as_zero(results):
    res = compare.compare(results["zero"]["pa"], results["zero"]["ref"])
    bad = [r for r in res["rows"] if r["param"] == "AUClast" and r["status"] != "match"]
    assert not bad, bad


def test_characterisation_the_excess_is_exactly_the_triangle_from_tlast_to_the_next_zero_row(results):
    """Pins the mechanism (not a wish): excess AUClast = (next time - Tlast) * Clast / 2, to rounding."""
    rows = _rows(DIR / "stress_zero.csv")
    checked = 0
    for sid in ("S03_trailing_blq", "S08_two_points_after_tmax", "S01_clean"):
        obs = [(float(r["TIME"]), float(r["DV"])) for r in rows if r["ID"] == sid and r["EVID"] == "0"]
        last = [o for o in obs if o[1] > 0][-1]
        nxt = [o for o in obs if o[0] > last[0]][0]
        pa, ref = results["zero"]["pa"][sid], results["zero"]["ref"][sid]
        assert pa["tlast"] == last[0]                                       # PharmAgent REPORTS Tlast correctly...
        excess = pa["AUClast"] - ref["AUClast"]                              # ...but integrates beyond it
        assert excess == pytest.approx((nxt[0] - last[0]) * last[1] / 2.0, abs=2e-5)
        checked += 1
    assert checked == 3


def test_characterisation_middle_blq_zero_is_a_data_point_for_pharmagent_and_the_textbook_alike(results):
    """Both treat a middle BLQ coded 0 as a real zero concentration: AUCinf is ~12% low against the analytic truth when the
    row is present, within 1% when it is dropped. (PKNCA's default drops middle BLQ: the R run adjudicates.)"""
    zero = truth_check.table(DIR / "stress_truth.json", [results["zero"]["pa_file"]])
    dropped = truth_check.table(DIR / "stress_truth.json", [results["dropped"]["pa_file"]])
    z = next(r for r in zero if r["id"] == "S05_middle_blq")["PharmAgent_err_pct"]
    d = next(r for r in dropped if r["id"] == "S05_middle_blq")["PharmAgent_err_pct"]
    assert z < -10.0 and abs(d) < 1.0


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
