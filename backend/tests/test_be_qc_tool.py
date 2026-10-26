"""The tool derives the QC inputs from the dataset: the pre-dose sample, the first-sample Cmax, and passes the QC
policy arguments to the contract."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.test_be_tool_contract import FULL, TIMES, _call, _frame


def _set_dv(df, subject, trt, time, value):
    out = df.copy()
    mask = (out.ID == subject) & (out.TRT == trt) & (out.TIME == time)
    assert mask.sum() == 1
    out.loc[mask, "DV"] = value
    return out


def _cmax(df, subject, trt):
    return float(df[(df.ID == subject) & (df.TRT == trt)].DV.max())


def _finding(res, rule, **match):
    return [f for f in res["qc_findings"] if f["rule"] == rule and all(f.get(k) == v for k, v in match.items())]


def test_a_clean_dataset_has_a_disposition_row_per_subject_and_no_exclusions_from_qc():
    res = _call(_frame(13), **FULL).result
    assert res["status"] == "ok" and len(res["dispositions"]) == 13
    assert all(r["disposition"] == "analysed" for r in res["dispositions"])
    assert not [f for f in res["qc_findings"] if f["action"] == "excluded"]
    assert res["decision_parameters"] == ["Cmax", "AUC_last"]


def test_a_predose_sample_above_5_percent_of_cmax_drops_that_subject_period():
    df = _frame(13)
    cmax = _cmax(df, 3, "B")
    df = _set_dv(df, 3, "B", 0.0, 0.2 * cmax)
    res = _call(df, **FULL).result
    f = _finding(res, "predose_gt_5pct_cmax", subject=3, treatment="B")
    assert len(f) == 1 and f[0]["action"] == "excluded"
    assert f[0]["detail"]["predose"] == pytest.approx(0.2 * cmax) and f[0]["detail"]["ratio"] > 0.05
    assert res["parameters"]["Cmax"]["n_evaluable"] == 12
    assert {r["subject"]: r for r in res["dispositions"]}[3]["disposition"] == "excluded"


def test_no_sample_at_or_before_time_zero_is_reported_as_a_missing_predose():
    df = _frame(13)
    df = df[~((df.ID == 2) & (df.TRT == "A") & (df.TIME == 0.0))]
    res = _call(df, **FULL).result
    f = _finding(res, "no_predose_sample", subject=2, treatment="A")
    assert len(f) == 1 and f[0]["action"] == "flagged" and res["parameters"]["Cmax"]["n_evaluable"] == 13


def test_flag_only_with_a_justification_keeps_the_subject():
    df = _frame(13)
    df = _set_dv(df, 3, "B", 0.0, 0.2 * _cmax(df, 3, "B"))
    res = _call(df, **FULL, predose_policy="flag_only", predose_justification="endogenous; protocol 8.2").result
    assert res["status"] == "ok" and res["parameters"]["Cmax"]["n_evaluable"] == 13
    assert _finding(res, "predose_gt_5pct_cmax", subject=3)[0]["action"] == "flagged"


def test_a_bad_policy_is_an_argument_refusal():
    res = _call(_frame(13), **FULL, predose_policy="ignore").result
    assert res["status"] == "invalid_arguments" and "predose_policy" in res["message"]


def test_cmax_at_the_first_postdose_sample_is_flagged():
    df = _frame(13)
    first_post = TIMES[1]
    df = _set_dv(df, 5, "A", first_post, 10 * _cmax(df, 5, "A"))
    res = _call(df, **FULL).result
    f = _finding(res, "cmax_at_first_sample", subject=5, treatment="A")
    assert len(f) == 1 and f[0]["action"] == "flagged"


def test_decision_parameter_arguments_reach_the_contract():
    res = _call(_frame(13), **FULL, decision_parameters=["Cmax", "AUC_last", "AUC_inf"],
                decision_justification="modified-release product").result
    assert res["decision_parameters"] == ["Cmax", "AUC_last", "AUC_inf"]
    assert res["parameters"]["AUC_inf"]["role"] == "decision"


def test_the_summary_mentions_the_qc_outcome():
    df = _frame(13)
    df = _set_dv(df, 3, "B", 0.0, 0.2 * _cmax(df, 3, "B"))
    out = _call(df, **FULL)
    assert "1 excluded by QC" in out.summary and np.isfinite(out.result["parameters"]["Cmax"]["gmr_pct"])


# ── subjects and rows lost before the contract still appear in the disposition table ──────────
def test_a_subject_with_too_few_samples_in_one_arm_shows_both_reasons():
    df = _frame(13)
    df = df[~((df.ID == 2) & (df.TRT == "A") & (df.TIME >= 1.0))]            # two samples left on A
    res = _call(df, **FULL).result
    row = {r["subject"]: r for r in res["dispositions"]}[2]
    assert row["disposition"] == "excluded"
    assert set(row["parameters"]["Cmax"]["reasons"]) == {"fewer_than_3_samples", "no_evaluable_counterpart"}


def test_a_subject_lost_in_both_arms_before_the_contract_still_has_a_row():
    df = _frame(13)
    df = df[~((df.ID == 4) & (df.TIME >= 1.0))]                              # two samples left on each arm
    res = _call(df, **FULL).result
    rows = {r["subject"]: r for r in res["dispositions"]}
    assert len(rows) == 13 and rows[4]["disposition"] == "excluded"
    assert rows[4]["parameters"]["Cmax"] == {"status": "excluded", "reasons": ["fewer_than_3_samples"]}


def test_a_dropped_concentration_row_is_a_note_on_the_subject_not_an_exclusion():
    df = _frame(13)
    df = df.astype({"DV": object})
    df.loc[(df.ID == 5) & (df.TRT == "B") & (df.TIME == 2.0), "DV"] = float("nan")
    res = _call(df, **FULL).result
    row = {r["subject"]: r for r in res["dispositions"]}[5]
    assert row["disposition"] == "analysed" and any("missing_dv" in n and "2.0" in n for n in row["notes"])


def test_every_predose_sample_counts_and_the_highest_decides():
    df = _frame(13)
    cmax = _cmax(df, 4, "A")
    early = df[(df.ID == 4) & (df.TRT == "A") & (df.TIME == 0.0)].assign(TIME=-2.0, DV=0.5 * cmax)   # earlier, high
    res = _call(pd.concat([df, early]), **FULL).result
    f = _finding(res, "predose_gt_5pct_cmax", subject=4, treatment="A")
    assert len(f) == 1 and f[0]["detail"]["predose"] == pytest.approx(0.5 * cmax)
    assert len(_finding(res, "multiple_predose_samples", subject=4, treatment="A")) == 1
    assert f[0]["detail"]["predose_time"] == -2.0                          # the sample the rule actually used


def test_disposition_rows_from_the_tool_are_in_natural_order():
    df = _frame(13)
    df = df.assign(ID=df.ID.map(lambda i: f"S{i}"))
    assert [r["subject"] for r in _call(df, **FULL).result["dispositions"]] == [f"S{i}" for i in range(1, 14)]
