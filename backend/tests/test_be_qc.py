"""QC rules, per-profile decision parameters and the subject-level disposition table (BE slice 3).

Sources (texts read 2026-10-23 and 2026-10-26):
- Pre-dose concentration > 5 % of the subject's Cmax in that period: the affected period is dropped.
  FDA, Bioequivalence Studies With Pharmacokinetic Endpoints for Drugs Submitted Under an ANDA (May 2026),
  'Subjects with pre-dose concentrations'; ICH M13A (EMA/CHMP/ICH/953493/2022) sec. 2.2.3.3; EMA CPMP/EWP/QWP/1401/98
  Rev. 1/Corr** 'Carry-over effects' (which exempts endogenous drugs).
- Very low exposure (AUC < 5 % of the geometric mean AUC of the product, computed without the subject): an exception
  that is accepted only in exceptional cases (M13A sec. 2.2.1.1; EMA); never applied automatically here.
- AUC(0-t)/AUC(0-inf) < 80 % in more than 20 % of observations: the validity of the study may need to be discussed
  (M13A sec. 2.2.2.2; EMA); subjects are not excluded for it.
- Cmax at the first post-dose sample: to be discussed (M13A sec. 2.1.8.1; FDA sec. III.C).
- Decision parameters: immediate-release products are judged on Cmax and AUC(0-t) (FDA ANDA guidance, statistical
  information; M13A sec. 2.2.4; EMA sec. 4.1.8). AUC(0-inf) is statistical information for modified-release products
  only (FDA), and an acceptability measure under M13A.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.compute.be_contract import BEContractError, assess_average_be


def _xo(n_tr=7, n_rt=6, seed=5, overrides=None):
    """Crossover records with Cmax, AUC_last and AUC_inf. ``overrides`` maps (subject, treatment) -> dict of keys
    to overwrite on that record (e.g. predose=...) or {'params': {...}} to overwrite parameters."""
    rng = np.random.default_rng(seed)
    n = n_tr + n_rt
    out = []
    for i in range(n):
        base = float(np.exp(rng.normal(0.0, 0.25)))
        tf = i < n_tr
        for trt in ("T", "R"):
            cmax = 100.0 * base * float(np.exp(rng.normal(0.0, 0.08)))
            auc = cmax * 6.0 * float(np.exp(rng.normal(0.0, 0.05)))
            rec = {"subject": f"S{i}", "treatment": trt, "period": (1 if (trt == "T") == tf else 2),
                   "params": {"Cmax": cmax, "AUC_last": auc, "AUC_inf": auc * 1.08}, "predose": 0.0,
                   "cmax_first_point": False}
            for key, val in (overrides or {}).get((f"S{i}", trt), {}).items():
                if key == "params":
                    rec["params"].update(val)
                else:
                    rec[key] = val
            out.append(rec)
    return out


def _par(n=14, seed=9):
    rng = np.random.default_rng(seed)
    out = []
    for arm in ("T", "R"):
        for i in range(n):
            cmax = 100.0 * float(np.exp(rng.normal(0.0, 0.3)))
            out.append({"subject": f"{arm}{i}", "treatment": arm, "predose": 0.0, "cmax_first_point": False,
                        "params": {"Cmax": cmax, "AUC_last": cmax * 6, "AUC_inf": cmax * 6.4}})
    return out


def _run(recs, **kw):
    args = {"design": "crossover", "profile": "fda_2026", "test": "T", "reference": "R"}
    args.update(kw)
    return assess_average_be(recs, **args)


def _find(res, rule, **match):
    return [f for f in res["qc_findings"] if f["rule"] == rule and all(f.get(k) == v for k, v in match.items())]


# ── pre-dose rule ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("profile", ["fda_2026", "ich_m13a", "ema_2010"])
def test_a_predose_above_5_percent_of_cmax_drops_that_period_and_so_the_subject(profile):
    cmax = next(r for r in _xo() if r["subject"] == "S3" and r["treatment"] == "T")["params"]["Cmax"]
    recs = _xo(overrides={("S3", "T"): {"predose": 0.0501 * cmax}})
    res = _run(recs, profile=profile)
    f = _find(res, "predose_gt_5pct_cmax", subject="S3", treatment="T")
    assert len(f) == 1 and f[0]["action"] == "excluded" and f[0]["period"] == 1     # S3 is in sequence TR
    assert f[0]["basis"] and f[0]["detail"]["ratio"] > 0.05
    p = res["parameters"]["Cmax"]
    assert p["n_evaluable"] == 12
    reasons = {(e["subject"], e["treatment"], e["reason"]) for e in res["exclusions"] if e["parameter"] == "Cmax"}
    assert ("S3", "T", "predose_gt_5pct_cmax") in reasons and ("S3", "R", "no_evaluable_counterpart") in reasons


def test_the_predose_boundary_is_exact_five_percent_is_kept_just_above_is_dropped():
    cmax = next(r for r in _xo() if r["subject"] == "S2" and r["treatment"] == "R")["params"]["Cmax"]
    at = _run(_xo(overrides={("S2", "R"): {"predose": cmax * 0.05}}))
    over = _run(_xo(overrides={("S2", "R"): {"predose": cmax * 0.05 * (1 + 1e-9)}}))
    assert not _find(at, "predose_gt_5pct_cmax") and at["parameters"]["Cmax"]["n_evaluable"] == 13
    assert len(_find(over, "predose_gt_5pct_cmax")) == 1 and over["parameters"]["Cmax"]["n_evaluable"] == 12


def test_a_predose_of_zero_or_missing_key_raises_no_finding():
    recs = _xo()
    for r in recs:
        r.pop("predose")
    assert not [f for f in _run(recs)["qc_findings"] if f["rule"].startswith("predose")]


def test_a_missing_predose_sample_is_reported_but_nothing_is_excluded():
    res = _run(_xo(overrides={("S1", "T"): {"predose": None}}))
    f = _find(res, "no_predose_sample", subject="S1", treatment="T")
    assert len(f) == 1 and f[0]["action"] == "flagged" and res["parameters"]["Cmax"]["n_evaluable"] == 13


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf"), "5"])
def test_an_unusable_predose_value_is_flagged_not_silently_ignored(bad):
    res = _run(_xo(overrides={("S1", "T"): {"predose": bad}}))
    assert len(_find(res, "invalid_predose_value", subject="S1")) == 1
    assert res["parameters"]["Cmax"]["n_evaluable"] == 13


def test_flag_only_policy_keeps_the_subject_but_still_reports_the_finding_and_needs_a_justification():
    cmax = next(r for r in _xo() if r["subject"] == "S3" and r["treatment"] == "T")["params"]["Cmax"]
    recs = _xo(overrides={("S3", "T"): {"predose": 0.2 * cmax}})
    with pytest.raises(BEContractError, match="justification"):
        _run(recs, predose_policy="flag_only")
    res = _run(recs, predose_policy="flag_only", predose_justification="endogenous compound; protocol 8.2")
    f = _find(res, "predose_gt_5pct_cmax", subject="S3")
    assert f[0]["action"] == "flagged" and res["parameters"]["Cmax"]["n_evaluable"] == 13
    assert res["predose_policy"] == {"policy": "flag_only", "justification": "endogenous compound; protocol 8.2"}


def test_the_default_policy_is_the_profile_rule_and_an_unknown_policy_is_refused():
    assert _run(_xo())["predose_policy"]["policy"] == "profile_rule"
    with pytest.raises(BEContractError, match="predose_policy"):
        _run(_xo(), predose_policy="ignore")


def test_in_a_parallel_design_a_predose_failure_drops_the_subject():
    recs = _par()
    cmax = recs[3]["params"]["Cmax"]
    recs[3]["predose"] = 0.5 * cmax
    res = _run(recs, design="parallel")
    assert res["parameters"]["Cmax"]["n_test"] == 13
    assert len(_find(res, "predose_gt_5pct_cmax", subject="T3")) == 1


def test_a_prespecified_exclusion_keeps_its_own_reason_when_qc_would_also_exclude():
    cmax = next(r for r in _xo() if r["subject"] == "S3" and r["treatment"] == "T")["params"]["Cmax"]
    recs = _xo(overrides={("S3", "T"): {"predose": 0.3 * cmax}})
    res = _run(recs, prespecified_exclusions=[{"subject": "S3", "reason": "vomiting (protocol 7.3)"}])
    reasons = {e["reason"] for e in res["exclusions"] if e["subject"] == "S3" and e["parameter"] == "Cmax"}
    assert "vomiting (protocol 7.3)" in reasons and "predose_gt_5pct_cmax" not in reasons


# ── flag-only rules ────────────────────────────────────────────────────────────────────────────
def test_very_low_exposure_is_flagged_but_never_excluded():
    recs = _xo(overrides={("S4", "T"): {"params": {"AUC_last": 0.4}}})            # ~ 0.07 % of the test geometric mean
    res = _run(recs)
    f = _find(res, "low_exposure_lt_5pct_gmean", subject="S4", treatment="T")
    assert len(f) == 1 and f[0]["action"] == "flagged" and f[0]["detail"]["ratio"] < 0.05
    assert res["parameters"]["AUC_last"]["n_evaluable"] == 13


def test_the_auc_extrapolation_rule_needs_more_than_a_fifth_of_observations_below_80_percent():
    below = {(f"S{i}", t): {"params": {"AUC_last": 5.0, "AUC_inf": 10.0}} for i in range(3) for t in ("T", "R")}
    few = {k: v for k, v in list(below.items())[:5]}                       # 5 of 26 records = 19.2 %
    assert not _find(_run(_xo(overrides=few)), "auc_coverage_lt_80pct_in_gt_20pct_of_records")
    res = _run(_xo(overrides=below))                                                # 6 of 26 = 23.1 %
    study = _find(res, "auc_coverage_lt_80pct_in_gt_20pct_of_records")
    assert len(study) == 1 and study[0]["scope"] == "study" and study[0]["action"] == "flagged"
    assert study[0]["detail"]["n_below"] == 6 and study[0]["detail"]["n_records"] == 26
    assert res["parameters"]["Cmax"]["n_evaluable"] == 13                   # nobody is excluded for it


def test_cmax_at_the_first_sample_is_flagged_for_discussion():
    res = _run(_xo(overrides={("S5", "R"): {"cmax_first_point": True}}))
    f = _find(res, "cmax_at_first_sample", subject="S5", treatment="R")
    assert len(f) == 1 and f[0]["action"] == "flagged"


def test_the_qc_summary_counts_findings_by_action_and_rule():
    cmax = next(r for r in _xo() if r["subject"] == "S3" and r["treatment"] == "T")["params"]["Cmax"]
    res = _run(_xo(overrides={("S3", "T"): {"predose": 0.3 * cmax}, ("S5", "R"): {"cmax_first_point": True}}))
    s = res["qc_summary"]
    assert s["excluded"] == 1 and s["flagged"] == 1
    assert s["by_rule"] == {"predose_gt_5pct_cmax": 1, "cmax_at_first_sample": 1}


# ── decision parameters ────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("profile", ["fda_2026", "ich_m13a", "ema_2010"])
def test_cmax_and_auc_last_decide_and_auc_inf_is_supportive_for_every_profile(profile):
    res = _run(_xo(), profile=profile)
    assert res["decision_parameters"] == ["Cmax", "AUC_last"] and res["supportive_parameters"] == ["AUC_inf"]
    assert [res["parameters"][k]["role"] for k in ("Cmax", "AUC_last", "AUC_inf")] == ["decision", "decision",
                                                                                     "supportive"]


def test_a_failing_supportive_parameter_does_not_change_the_verdict():
    recs = _xo()
    for r in recs:
        if r["treatment"] == "T":
            r["params"]["AUC_inf"] *= 3.0                                  # test AUC_inf far outside 80-125 %
    res = _run(recs)
    assert res["parameters"]["AUC_inf"]["within_limits"] is False
    assert res["parameters"]["Cmax"]["within_limits"] and res["parameters"]["AUC_last"]["within_limits"]
    assert res["bioequivalent"] is True


def test_naming_auc_inf_as_a_decision_parameter_needs_a_justification_and_then_decides():
    recs = _xo()
    for r in recs:
        if r["treatment"] == "T":
            r["params"]["AUC_inf"] *= 3.0
    with pytest.raises(BEContractError, match="justification"):
        _run(recs, decision_parameters=["Cmax", "AUC_last", "AUC_inf"])
    res = _run(recs, decision_parameters=["Cmax", "AUC_last", "AUC_inf"],
               decision_justification="modified-release product; FDA asks for AUC0-inf")
    assert res["bioequivalent"] is False and res["parameters"]["AUC_inf"]["role"] == "decision"
    assert res["decision_justification"] == "modified-release product; FDA asks for AUC0-inf"


def test_a_decision_parameter_missing_from_the_data_means_no_conclusion():
    recs = _xo()
    for r in recs:
        r["params"].pop("AUC_last")
    res = _run(recs)
    assert res["bioequivalent"] is False and res["decision_gap"] == ["AUC_last"]


@pytest.mark.parametrize("bad", [[], "Cmax", ["Cmax", 3], ["Cmax", "Cmax"], ["NotAParameter"]])
def test_malformed_decision_parameter_lists_are_refused(bad):
    with pytest.raises(BEContractError):
        _run(_xo(), decision_parameters=bad, decision_justification="x")


# ── subject-level disposition table ────────────────────────────────────────────────────────────
def test_every_subject_has_a_disposition_row_with_sequence_periods_and_parameter_status():
    res = _run(_xo())
    rows = {r["subject"]: r for r in res["dispositions"]}
    assert len(rows) == 13 and rows["S0"]["sequence"] == "TR" and rows["S12"]["sequence"] == "RT"
    assert rows["S0"]["periods"] == {"T": 1, "R": 2} and rows["S12"]["periods"] == {"T": 2, "R": 1}
    assert all(r["disposition"] == "analysed" and r["qc_flags"] == [] for r in rows.values())
    assert rows["S0"]["parameters"]["Cmax"] == {"status": "analysed", "reasons": []}


def test_a_predose_failure_shows_as_an_excluded_subject_with_both_reasons():
    cmax = next(r for r in _xo() if r["subject"] == "S3" and r["treatment"] == "T")["params"]["Cmax"]
    res = _run(_xo(overrides={("S3", "T"): {"predose": 0.3 * cmax}}))
    row = {r["subject"]: r for r in res["dispositions"]}["S3"]
    assert row["disposition"] == "excluded"
    assert row["parameters"]["Cmax"]["status"] == "excluded"
    assert set(row["parameters"]["Cmax"]["reasons"]) == {"predose_gt_5pct_cmax", "no_evaluable_counterpart"}
    assert row["qc_flags"] == ["predose_gt_5pct_cmax"]


def test_a_subject_missing_one_parameter_is_partially_analysed():
    res = _run(_xo(overrides={("S4", "T"): {"params": {"AUC_inf": None}}}))
    row = {r["subject"]: r for r in res["dispositions"]}["S4"]
    assert row["disposition"] == "partial"
    assert row["parameters"]["AUC_inf"]["status"] == "excluded" and row["parameters"]["Cmax"]["status"] == "analysed"


def test_flags_appear_on_the_row_of_the_subject_they_concern():
    res = _run(_xo(overrides={("S5", "R"): {"cmax_first_point": True}}))
    row = {r["subject"]: r for r in res["dispositions"]}["S5"]
    assert row["qc_flags"] == ["cmax_at_first_sample"] and row["disposition"] == "analysed"


def test_parallel_dispositions_carry_the_arm_not_a_sequence():
    res = _run(_par(), design="parallel")
    rows = res["dispositions"]
    assert len(rows) == 28 and {r["arm"] for r in rows} == {"T", "R"}
    assert all("sequence" not in r or r["sequence"] is None for r in rows)


def test_a_prespecified_exclusion_is_a_disposition_reason_too():
    res = _run(_xo(), prespecified_exclusions=[{"subject": "S7", "reason": "vomiting (protocol 7.3)"}])
    row = {r["subject"]: r for r in res["dispositions"]}["S7"]
    assert row["disposition"] == "excluded" and "vomiting (protocol 7.3)" in row["parameters"]["Cmax"]["reasons"]


# ── review of 6481cf0 ──────────────────────────────────────────────────────────────────────────
def test_an_override_cannot_drop_a_profile_decision_parameter_the_data_carry():
    recs = _xo()
    for bad in (["AUC_inf"], ["Cmax"], ["AUC_last", "AUC_inf"]):
        with pytest.raises(BEContractError, match="cannot drop"):
            _run(recs, decision_parameters=bad, decision_justification="because")
    ok = _run(recs, decision_parameters=["Cmax", "AUC_last", "AUC_inf"], decision_justification="modified release")
    assert ok["decision_parameters"] == ["Cmax", "AUC_last", "AUC_inf"] and ok["decision_omitted"] == []


def test_an_override_may_omit_a_profile_parameter_the_data_do_not_carry_and_says_so():
    recs = _xo()
    for r in recs:
        r["params"].pop("AUC_last")
    with pytest.raises(BEContractError, match="justification"):
        _run(recs, decision_parameters=["Cmax"])
    res = _run(recs, decision_parameters=["Cmax"], decision_justification="AUC not measured in this study")
    assert res["decision_parameters"] == ["Cmax"] and res["decision_omitted"] == ["AUC_last"]


def test_records_the_analysis_discards_do_not_feed_the_study_level_flag_rules():
    both_below = {("S0", "T"): {"params": {"AUC_last": 5.0, "AUC_inf": 10.0}},
                  ("S0", "R"): {"params": {"AUC_last": 5.0, "AUC_inf": 10.0}}}
    four_more = {(f"S{i}", t): {"params": {"AUC_last": 5.0, "AUC_inf": 10.0}} for i in (1, 2) for t in ("T", "R")}
    over = {**both_below, **four_more}                         # 6 of 26 records: 23 %
    assert _find(_run(_xo(overrides=over)), "auc_coverage_lt_80pct_in_gt_20pct_of_records")
    res = _run(_xo(overrides=over), prespecified_exclusions=[{"subject": "S0", "reason": "protocol 7.3"}])
    assert not _find(res, "auc_coverage_lt_80pct_in_gt_20pct_of_records")          # 4 of 24: 16.7 %
    assert not _find(res, "auc_coverage_lt_80pct", subject="S0")


def test_a_qc_excluded_record_does_not_count_in_the_low_exposure_geometric_mean_or_flags():
    cmax = next(r for r in _xo() if r["subject"] == "S0" and r["treatment"] == "T")["params"]["Cmax"]
    res = _run(_xo(overrides={("S0", "T"): {"predose": 0.3 * cmax, "params": {"AUC_last": 1e-6}}}))
    assert not _find(res, "low_exposure_lt_5pct_gmean", subject="S0")
    assert not _find(res, "auc_coverage_lt_80pct", subject="S0")


def test_a_prespecified_exclusion_that_also_meets_the_qc_rule_is_not_counted_as_a_qc_exclusion():
    cmax = next(r for r in _xo() if r["subject"] == "S0" and r["treatment"] == "T")["params"]["Cmax"]
    res = _run(_xo(overrides={("S0", "T"): {"predose": 0.3 * cmax}}),
               prespecified_exclusions=[{"subject": "S0", "reason": "vomiting (protocol 7.3)"}])
    f = _find(res, "predose_gt_5pct_cmax", subject="S0")[0]
    assert f["action"] == "flagged" and f["detail"]["superseded_by"] == "prespecified_exclusion"
    assert res["qc_summary"]["excluded"] == 0
    assert {e["reason"] for e in res["exclusions"] if e["subject"] == "S0" and e["parameter"] == "Cmax"} >= {
        "vomiting (protocol 7.3)"}


def test_several_predose_samples_are_reported_and_the_rule_uses_the_highest():
    cmax = next(r for r in _xo() if r["subject"] == "S2" and r["treatment"] == "R")["params"]["Cmax"]
    res = _run(_xo(overrides={("S2", "R"): {"predose": 0.2 * cmax, "predose_samples": 2}}))
    multi = _find(res, "multiple_predose_samples", subject="S2")
    assert len(multi) == 1 and multi[0]["action"] == "flagged" and multi[0]["detail"]["n"] == 2
    assert len(_find(res, "predose_gt_5pct_cmax", subject="S2")) == 1


def test_disposition_rows_are_in_natural_subject_order():
    recs = _xo(n_tr=7, n_rt=6)
    for r in recs:
        r["subject"] = f"S{int(r['subject'][1:]) + 1}"                    # S1 .. S13: S10 must follow S9, not S1
    order = [row["subject"] for row in _run(recs)["dispositions"]]
    assert order == [f"S{i}" for i in range(1, 14)]


# ── re-review of e1608b4 ───────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("ident", ["²", "1²", "S²", "٣", "S٣", "１２", "1e3", "", None, -1, "-3", "9" * 5000, "S" + "7" * 4400])
def test_natural_key_never_raises_whatever_the_subject_id(ident):
    from app.compute.be_qc import natural_key
    assert natural_key(ident) is not None


def test_natural_key_orders_numbers_by_value_and_never_ties_distinct_ids():
    import random

    from app.compute.be_qc import natural_key
    ids = ["S10", "S2", "S1", "S٣", "S3", "S²", "S02", "s2", "S9" * 3]
    shuffled = []
    for seed in range(5):
        x = list(ids)
        random.Random(seed).shuffle(x)
        shuffled.append(sorted(x, key=natural_key))
    assert all(order == shuffled[0] for order in shuffled)                 # a total order: no ties
    assert shuffled[0].index("S2") < shuffled[0].index("S10") and shuffled[0].index("S1") < shuffled[0].index("S2")


def test_an_exotic_subject_id_does_not_break_the_disposition_table():
    recs = _xo()
    for r in recs:
        if r["subject"] == "S1":
            r["subject"] = "S²"
    assert len(_run(recs)["dispositions"]) == 13


def test_in_a_crossover_a_qc_excluded_subject_leaves_the_study_rules_with_both_records():
    cmax = next(r for r in _xo() if r["subject"] == "S2" and r["treatment"] == "T")["params"]["Cmax"]
    res = _run(_xo(overrides={("S2", "T"): {"predose": 0.3 * cmax},
                              ("S2", "R"): {"params": {"AUC_last": 1e-6}}}))
    assert not _find(res, "low_exposure_lt_5pct_gmean", subject="S2")
    assert not _find(res, "auc_coverage_lt_80pct", subject="S2")


def test_a_low_exposure_check_that_is_skipped_after_discarding_says_so():
    tiny = {("S0", "T"): {"params": {"AUC_last": 0.01}}}
    assert _find(_run(_xo(n_tr=2, n_rt=1, overrides=tiny)), "low_exposure_lt_5pct_gmean", subject="S0")
    cmax = next(r for r in _xo(n_tr=2, n_rt=1) if r["subject"] == "S1" and r["treatment"] == "T")["params"]["Cmax"]
    res = _run(_xo(n_tr=2, n_rt=1, overrides={**tiny, ("S1", "T"): {"predose": 0.3 * cmax}}))
    skipped = _find(res, "low_exposure_check_skipped", treatment="T")
    assert len(skipped) == 1 and skipped[0]["scope"] == "study" and skipped[0]["action"] == "flagged"
    assert skipped[0]["detail"]["n_remaining"] == 2 and skipped[0]["detail"]["n_discarded"] == 1


def test_the_predose_finding_names_the_time_of_the_sample_it_used():
    cmax = next(r for r in _xo() if r["subject"] == "S2" and r["treatment"] == "R")["params"]["Cmax"]
    res = _run(_xo(overrides={("S2", "R"): {"predose": 0.2 * cmax, "predose_time": -24.0, "predose_samples": 2}}))
    assert _find(res, "predose_gt_5pct_cmax", subject="S2")[0]["detail"]["predose_time"] == -24.0
    assert _find(res, "multiple_predose_samples", subject="S2")[0]["detail"]["time_used"] == -24.0


def test_the_override_error_says_a_parameter_counts_as_carried_if_any_subject_has_it():
    with pytest.raises(BEContractError, match="any subject"):
        _run(_xo(), decision_parameters=["AUC_inf"], decision_justification="x")
