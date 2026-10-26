"""PharmAgent's interim BE estimator against base R (stats::t.test, R 4.5.1), on the committed datasets in
tests/fixtures/be_r_crosscheck. R's results (r_results.csv) were produced by crosscheck.R in that folder and are
stored, so this test needs no R. Crossover: paired t.test on log values; parallel: t.test(var.equal = TRUE).

Agreement validates the interim implementation only; it does not establish regulatory correctness (the
period/sequence/formulation model is not implemented)."""
from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from app.compute.be_contract import assess_average_be

D = Path(__file__).parent / "fixtures" / "be_r_crosscheck"
R = {row["case"]: row for row in csv.DictReader((D / "r_results.csv").open())}


def _records(case):
    out = []
    for row in csv.DictReader((D / f"{case}.csv").open()):
        sid = row["subject"] if row["subject"].startswith(("T", "R")) else int(row["subject"])
        out.append({"subject": sid, "treatment": row["treatment"], "params": {"Cmax": float(row["Cmax"])}})
    return out


@pytest.mark.parametrize("case", sorted(c for c, r in R.items() if r["design"] == "crossover"))
def test_the_legacy_paired_estimator_matches_base_r_paired_t(case):
    """The crossover cases here are base R's PAIRED t-test, so they validate the legacy be_one_parameter, which is
    no longer the contract's crossover estimator (see tests/test_be_crosscheck_r_anova.py for that)."""
    from app.compute.bioequivalence import be_one_parameter

    r = R[case]
    rows = _records(case)
    t = [x["params"]["Cmax"] for x in rows if x["treatment"] == "T"]
    ref = [x["params"]["Cmax"] for x in rows if x["treatment"] == "R"]
    p = be_one_parameter(t, ref, paired=True)
    assert math.isclose(p["ci_lower_raw"], float(r["ci_lower_pct"]), rel_tol=1e-10)
    assert math.isclose(p["ci_upper_raw"], float(r["ci_upper_pct"]), rel_tol=1e-10)


@pytest.mark.parametrize("case", sorted(c for c, r in R.items() if r["design"] == "parallel"))
def test_pooled_parallel_estimator_matches_base_r(case):
    r = R[case]
    p = assess_average_be(_records(case), design=r["design"], profile="fda_2026", test="T",
                          reference="R")["parameters"]["Cmax"]
    assert (p["n_test"], p["n_reference"]) == (int(r["n_test"]), int(r["n_reference"]))
    assert math.isclose(p["ci_lower_raw"], float(r["ci_lower_pct"]), rel_tol=1e-10)
    assert math.isclose(p["ci_upper_raw"], float(r["ci_upper_pct"]), rel_tol=1e-10)
    assert math.isclose(p["gmr_pct"], round(float(r["gmr_pct"]), 4), abs_tol=1e-9)


def test_every_r_case_is_covered():
    cases = {row["case"] for row in csv.DictReader((D / "cases.csv").open())}
    assert cases == set(R) and len(cases) == 8
