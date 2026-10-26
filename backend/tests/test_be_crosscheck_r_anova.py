"""Cross-check of the 2x2 crossover model and the Welch parallel estimator against R (lm, drop1, confint, nlme::lme,
t.test), using the datasets in tests/fixtures/be_r_anova.

R's results are stored in r_anova_results.csv, produced by crosscheck_anova.R in that folder. Until that file exists
(R is run only on the owner's go-ahead) the comparison tests SKIP with a visible reason; they are not validation yet.
Agreement validates the implementation against an independent one; it does not by itself establish regulatory
correctness (that also needs the independent biostatistical review named in the catalog entry)."""
from __future__ import annotations

import csv
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from app.compute.be_contract import assess_average_be
from app.compute.be_model import crossover_2x2, parallel_two_group

D = Path(__file__).parent / "fixtures" / "be_r_anova"
RESULTS = D / "r_anova_results.csv"
CASES = list(csv.DictReader((D / "cases.csv").open()))
XO = [c["case"] for c in CASES if c["design"] == "crossover"]
PAR = [c["case"] for c in CASES if c["design"] == "parallel"]
needs_r = pytest.mark.skipif(not RESULTS.exists(), reason="R results not generated yet: R runs only on the "
                             "owner's go-ahead (see crosscheck_anova.R); these comparisons are NOT validation yet")


def _r(case: str) -> dict[str, str]:
    return {row["case"]: row for row in csv.DictReader(RESULTS.open())}[case]


def _crossover_inputs(case: str):
    by: dict[str, dict] = {}
    for row in csv.DictReader((D / f"{case}.csv").open()):
        by.setdefault(row["subject"], {})[row["treatment"]] = (float(row["Cmax"]), int(row["period"]))
    subs = sorted(by)
    return (np.array([by[s]["T"][0] for s in subs]), np.array([by[s]["R"][0] for s in subs]),
            np.array([by[s]["T"][1] == 1 for s in subs]), by)


def _close(a: float, b: str, rel: float = 1e-8) -> bool:
    return math.isclose(a, float(b), rel_tol=rel, abs_tol=1e-12)


# ── fixtures are reproducible (no R needed) ────────────────────────────────────────────────────
def test_the_committed_datasets_are_exactly_what_the_generator_writes(tmp_path):
    spec = importlib.util.spec_from_file_location("make_be_fixtures", D / "make_fixtures.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.HERE = tmp_path
    mod.write_crossover()
    mod.write_parallel()
    for case in XO + PAR:
        assert (tmp_path / f"{case}.csv").read_bytes() == (D / f"{case}.csv").read_bytes(), case


def test_the_datasets_cover_unbalanced_sequences_and_a_period_effect():
    sizes = {c: _crossover_inputs(c)[2] for c in XO}
    assert any(tf.sum() != (~tf).sum() for tf in sizes.values())
    assert len(XO) >= 5 and len(PAR) >= 2
    periods = []
    for case in XO:
        t, r, tf, by = _crossover_inputs(case)
        first = np.array([min(by[s][k][1] for k in ("T", "R")) for s in sorted(by)])
        assert first.min() == 1                                                      # periods are 1 and 2
        lt, lr = np.log(t), np.log(r)
        d = np.where(tf, lr - lt, lt - lr)                         # second period minus first, per subject
        periods.append(d.mean())
    assert max(abs(x) for x in periods) > 0.05                                       # a real period effect is present


# ── comparisons with R ─────────────────────────────────────────────────────────────────────────
@needs_r
@pytest.mark.parametrize("case", XO)
def test_crossover_estimate_se_df_and_interval_match_r_lm(case):
    t, r, tf, _ = _crossover_inputs(case)
    got, want = crossover_2x2(t, r, tf), _r(case)
    assert got["df"] == int(float(want["df"]))
    assert _close(got["estimate"], want["estimate"]) and _close(got["se"], want["se"])
    assert _close(got["mse"], want["mse"])
    assert _close(got["ci_lower_raw"], want["ci_lower_pct"]) and _close(got["ci_upper_raw"], want["ci_upper_pct"])


@needs_r
@pytest.mark.parametrize("case", XO)
def test_crossover_anova_table_matches_r(case):
    t, r, tf, _ = _crossover_inputs(case)
    rows = {x["source"]: x for x in crossover_2x2(t, r, tf)["anova"]}
    want = _r(case)
    for source, key in (("sequence", "ss_sequence"), ("subject(sequence)", "ss_subject"), ("period", "ss_period"),
                        ("formulation", "ss_formulation"), ("residual", "ss_residual")):
        assert _close(rows[source]["ss"], want[key], 1e-7), (source, rows[source]["ss"], want[key])
    for source, f_key, p_key in (("sequence", "f_sequence", "p_sequence"), ("period", "f_period", "p_period"),
                                 ("formulation", "f_formulation", "p_formulation")):
        assert _close(rows[source]["f"], want[f_key], 1e-7) and _close(rows[source]["p"], want[p_key], 1e-6), source


@needs_r
@pytest.mark.parametrize("case", XO)
def test_the_mixed_model_gives_the_same_formulation_estimate_and_se_for_complete_data(case):
    t, r, tf, _ = _crossover_inputs(case)
    got, want = crossover_2x2(t, r, tf), _r(case)
    if want["lme_estimate"] in ("", "NA"):
        pytest.skip("nlme was not available in the R session")
    # the equivalence holds away from the boundary; at a zero REML between-subject SD the mixed model collapses
    assert float(want["lme_between_sd"]) > 1e-3, "REML between-subject variance on the boundary: unsuitable dataset"
    # the estimate agrees to ~1e-11; the SE to the REML optimizer's tolerance (the variance components are found
    # iteratively: the gap grew with subject count, 1e-10 at n=12 to 1.5e-6 at n=72), hence the looser bound
    assert _close(got["estimate"], want["lme_estimate"], 1e-8) and _close(got["se"], want["lme_se"], 1e-5)
    assert got["df"] == int(float(want["lme_df"]))


@needs_r
@pytest.mark.parametrize("case", XO)
def test_the_whole_contract_path_matches_r_lm(case):
    _, _, _, by = _crossover_inputs(case)
    recs = []
    for s, arms in by.items():
        for trt, (val, period) in arms.items():
            recs.append({"subject": s, "treatment": trt, "params": {"Cmax": val}, "period": period})
    p = assess_average_be(recs, design="crossover", profile="fda_2026", test="T",
                          reference="R")["parameters"]["Cmax"]
    want = _r(case)
    assert _close(p["ci_lower_raw"], want["ci_lower_pct"]) and _close(p["ci_upper_raw"], want["ci_upper_pct"])


@needs_r
@pytest.mark.parametrize("case", PAR)
def test_welch_parallel_matches_r_t_test(case):
    rows = list(csv.DictReader((D / f"{case}.csv").open()))
    t = np.array([float(x["Cmax"]) for x in rows if x["treatment"] == "T"])
    r = np.array([float(x["Cmax"]) for x in rows if x["treatment"] == "R"])
    got, want = parallel_two_group(t, r, variance="welch"), _r(case)
    assert _close(got["estimate"], want["estimate"]) and _close(got["se"], want["se"])
    assert _close(got["df"], want["welch_df"], 1e-9)
    assert _close(got["ci_lower_raw"], want["ci_lower_pct"]) and _close(got["ci_upper_raw"], want["ci_upper_pct"])
