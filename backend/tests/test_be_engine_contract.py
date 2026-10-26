"""The statistical engine inside the safety contract: crossover records carry a period, the 2x2 fixed-effects model
replaces the paired t-test, and the result reports what a reviewer needs (model, df, ANOVA, means)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from app.compute.be_contract import BEContractError, assess_average_be
from app.compute.be_model import crossover_2x2

SEED_RNG = 77


def _crossover_records(n_tr=7, n_rt=6, *, per=0.5, form=-0.03, labels=(1, 2), param="Cmax"):
    """Raw values from the same generative model as test_be_model, as contract records with periods."""
    rng = np.random.default_rng(SEED_RNG)
    n = n_tr + n_rt
    tf = np.array([True] * n_tr + [False] * n_rt)
    subj = rng.normal(0, 0.3, n)
    e = rng.normal(0, 0.12, (n, 2))
    y_first = 4.0 + subj + np.where(tf, form / 2, -form / 2) - per / 2 + e[:, 0]
    y_second = 4.0 + subj + np.where(tf, -form / 2, form / 2) + per / 2 + e[:, 1]
    recs = []
    for i in range(n):
        t_val = math.exp(y_first[i] if tf[i] else y_second[i])
        r_val = math.exp(y_second[i] if tf[i] else y_first[i])
        t_per, r_per = (labels[0], labels[1]) if tf[i] else (labels[1], labels[0])
        recs.append({"subject": f"S{i}", "treatment": "T", "params": {param: t_val}, "period": t_per})
        recs.append({"subject": f"S{i}", "treatment": "R", "params": {param: r_val}, "period": r_per})
    return recs, tf


def _run(recs, **kw):
    args = {"design": "crossover", "profile": "fda_2026", "test": "T", "reference": "R",
            "decision_parameters": ["Cmax"], "decision_justification": "unit-test data carry one parameter"}
    args.update(kw)
    return assess_average_be(recs, **args)


def _arrays(recs, param="Cmax"):
    by = {}
    for r in recs:
        by.setdefault(r["subject"], {})[r["treatment"]] = (r["params"][param], r["period"])
    subs = sorted(by, key=lambda s: int(s[1:]))
    t = np.array([by[s]["T"][0] for s in subs])
    r_ = np.array([by[s]["R"][0] for s in subs])
    tf = np.array([by[s]["T"][1] < by[s]["R"][1] for s in subs])
    return t, r_, tf


# ── periods are required and the design must be a real 2x2 ─────────────────────────────────────
def test_crossover_records_without_a_period_are_refused():
    recs, _ = _crossover_records()
    for r in recs:
        r.pop("period")
    with pytest.raises(BEContractError, match="period"):
        _run(recs)


def test_one_missing_period_label_is_refused():
    recs, _ = _crossover_records()
    recs[3].pop("period")
    with pytest.raises(BEContractError, match="period"):
        _run(recs)


def test_a_single_sequence_is_refused_by_the_contract_itself():
    recs, _ = _crossover_records(n_tr=12, n_rt=0)
    with pytest.raises(BEContractError, match="sequence"):
        _run(recs)


def test_more_than_two_periods_is_refused_as_not_a_two_period_crossover():
    recs, _ = _crossover_records()
    recs[0]["period"] = 3
    with pytest.raises(BEContractError, match="two-period"):
        _run(recs)


def test_a_subject_with_both_products_in_one_period_is_refused():
    recs, _ = _crossover_records()
    recs[1]["period"] = recs[0]["period"]
    with pytest.raises(BEContractError, match="same period"):
        _run(recs)


def test_period_labels_may_be_text_and_are_ordered_by_sort_order():
    a, tf = _crossover_records(labels=(1, 2))
    b, _ = _crossover_records(labels=("P1", "P2"))
    pa, pb = _run(a)["parameters"]["Cmax"], _run(b)["parameters"]["Cmax"]
    assert pa["ci_lower_raw"] == pb["ci_lower_raw"] and pa["ci_upper_raw"] == pb["ci_upper_raw"]
    # the interval does not depend on period order, so also pin the order itself through the sequence sizes
    assert pa["n_per_sequence"] == pb["n_per_sequence"] == {"TR": 7, "RT": 6}


# ── the estimate is the 2x2 model, not the paired t-test ───────────────────────────────────────
def test_the_crossover_estimate_is_the_fixed_effects_model_with_n_minus_2_df():
    recs, _ = _crossover_records(7, 6)
    p = _run(recs)["parameters"]["Cmax"]
    t, r, tf = _arrays(recs)
    want = crossover_2x2(t, r, tf)
    assert p["df_error"] == 11 == want["df"]
    assert math.isclose(p["ci_lower_raw"], want["ci_lower_raw"], rel_tol=1e-12)
    assert math.isclose(p["ci_upper_raw"], want["ci_upper_raw"], rel_tol=1e-12)
    assert p["n_per_sequence"] == {"TR": 7, "RT": 6}
    paired_diff = np.log(t) - np.log(r)
    assert not math.isclose(math.log(p["gmr_pct"] / 100), paired_diff.mean(), rel_tol=1e-3)   # not the paired mean


def test_the_result_carries_the_anova_table_means_and_cv_for_review():
    p = _run(_crossover_records()[0])["parameters"]["Cmax"]
    assert [r["source"] for r in p["anova"]] == ["sequence", "subject(sequence)", "period", "formulation", "residual"]
    for key in ("geo_mean_test", "geo_mean_ref", "arith_mean_test", "arith_mean_ref", "se_log", "mse_log",
                "cv_intra_pct"):
        assert p[key] is not None, key


def test_the_model_text_names_the_anova_and_says_validation_is_pending():
    res = _run(_crossover_records()[0])
    assert "ANOVA" in res["model"] and "sequence, subject within sequence, period and formulation" in res["model"]
    assert "interim" in res["model"] and "validation pending" in res["model"]


def test_excluding_every_subject_of_one_sequence_makes_the_parameter_not_estimable():
    recs, _ = _crossover_records(13, 12)
    rt_subjects = sorted({r["subject"] for r in recs if r["subject"] and int(r["subject"][1:]) >= 13})
    out = _run(recs, prespecified_exclusions=[{"subject": s, "reason": "protocol 7.3"} for s in rt_subjects])
    p = out["parameters"]["Cmax"]
    assert p["acceptance"] == "not_estimable" and p["within_limits"] is False and out["bioequivalent"] is False
    assert sum(e["reason"] == "protocol 7.3" for e in out["exclusions"]) == 2 * len(rt_subjects)


def test_a_parameter_missing_one_product_is_excluded_for_that_parameter_only():
    recs, _ = _crossover_records(7, 6)
    recs[0]["params"]["Cmax"] = None
    out = _run(recs)
    p = out["parameters"]["Cmax"]
    assert p["n_evaluable"] == 12
    assert any(e["reason"] == "missing_value" for e in out["exclusions"])
    assert any(e["reason"] == "no_evaluable_counterpart" for e in out["exclusions"])


# ── parallel ───────────────────────────────────────────────────────────────────────────────────
def _parallel_records(n=12, seed=3):
    rng = np.random.default_rng(seed)
    t, r = np.exp(rng.normal(4.0, 0.3, n)), np.exp(rng.normal(4.05, 0.5, n))
    return ([{"subject": f"T{i}", "treatment": "T", "params": {"Cmax": float(v)}} for i, v in enumerate(t)]
            + [{"subject": f"R{i}", "treatment": "R", "params": {"Cmax": float(v)}} for i, v in enumerate(r)])


def test_parallel_defaults_to_pooled_variance_and_records_it():
    out = _run(_parallel_records(), design="parallel")
    assert out["variance"] == "pooled" and out["parameters"]["Cmax"]["df_error"] == 22


def test_parallel_welch_is_available_and_recorded():
    out = _run(_parallel_records(), design="parallel", variance="welch")
    p = out["parameters"]["Cmax"]
    assert out["variance"] == "welch" and p["df_error"] != 22 and isinstance(p["df_error"], float)


def test_an_unknown_variance_option_is_refused_and_crossover_ignores_it():
    with pytest.raises(BEContractError, match="variance"):
        _run(_parallel_records(), design="parallel", variance="equal-ish")
    assert _run(_crossover_records()[0])["variance"] is None


def test_periods_on_parallel_records_are_ignored():
    recs = _parallel_records()
    for r in recs:
        r["period"] = 1
    assert _run(recs, design="parallel")["parameters"]["Cmax"]["acceptance"] in ("within_limits", "outside_limits")


def test_the_csv_export_carries_the_degrees_of_freedom_and_standard_error():
    from app.core.exporters import _be
    from app.core.pharmstate import PharmState

    res = _run(_crossover_records()[0])
    header = _be(PharmState(be_results={"status": "ok", **res})).splitlines()[0]
    assert "df_error" in header and "se_log" in header
