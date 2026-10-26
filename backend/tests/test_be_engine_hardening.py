"""Independent review of 5786bc7 (BE statistical engine): every finding reproduced here first, then fixed.

M1 non-finite period labels bypass the same-period rule; M2 degenerate data gave negative sums of squares and
garbage F/p; M3 stale user-facing text; M4 tests that passed for the wrong reason (period ordering, unbalanced ANOVA
rows against an independent computation); L5 numeric-looking text labels sorted as text; L5 a True label hid a bool
check; L6 numpy integer labels refused; L7 the tool's period check treated a third arm like T/R; L8 a fragile string
match decided a refusal status; L9 test_first and value dtypes."""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from app.compute.be_contract import BEContractError, assess_average_be
from app.compute.be_model import crossover_2x2, parallel_two_group
from tests.test_be_engine_contract import _crossover_records, _run
from tests.test_be_model import _make
from tests.test_be_tool_contract import FULL, _call, _frame


# ── M1 / L5 / L6: period labels ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_period_labels_are_refused_even_when_a_subject_shares_one(bad):
    recs, _ = _crossover_records(7, 7, labels=(1, bad))
    recs[0]["period"] = recs[1]["period"] = bad                  # subject 0: both products in the "same" label
    with pytest.raises(BEContractError, match="finite"):
        _run(recs)


def test_numpy_integer_and_float_period_labels_are_accepted_like_python_numbers():
    plain, _ = _crossover_records(7, 6)
    numpy_labels, _ = _crossover_records(7, 6, labels=(np.int64(1), np.int64(2)))
    floats, _ = _crossover_records(7, 6, labels=(np.float64(1.0), np.float64(2.0)))
    base = _run(plain)["parameters"]["Cmax"]
    for recs in (numpy_labels, floats):
        got = _run(recs)["parameters"]["Cmax"]
        assert got["ci_lower_raw"] == base["ci_lower_raw"] and got["n_per_sequence"] == base["n_per_sequence"]


def test_a_bool_label_is_refused_wherever_it_appears_in_the_data():
    recs, _ = _crossover_records(7, 6, labels=(True, 2))         # True == 1 would otherwise hide inside the set
    with pytest.raises(BEContractError, match="numbers or text"):
        _run(recs)
    recs, _ = _crossover_records(7, 6)
    recs[-1]["period"] = True
    with pytest.raises(BEContractError, match="numbers or text"):
        _run(recs)


@pytest.mark.parametrize("labels", [(2, 10), ("2", "10"), (2.0, 10.0)])
def test_period_labels_are_ordered_by_value_not_by_text(labels):
    """T in the numerically earlier period is sequence TR, however the labels are written."""
    out = _run(_crossover_records(9, 5, labels=labels)[0])["parameters"]["Cmax"]
    assert out["n_per_sequence"] == {"TR": 9, "RT": 5}


def test_period_ordering_is_visible_in_the_sequence_counts():
    a = _run(_crossover_records(9, 5, labels=(1, 2))[0])["parameters"]["Cmax"]
    b = _run(_crossover_records(9, 5, labels=(2, 1))[0])["parameters"]["Cmax"]
    assert a["n_per_sequence"] == {"TR": 9, "RT": 5} and b["n_per_sequence"] == {"TR": 5, "RT": 9}


def test_numbers_mixed_with_text_labels_are_refused():
    recs, _ = _crossover_records(7, 6)
    recs[0]["period"] = "1"
    with pytest.raises(BEContractError):
        _run(recs)


# ── M2: degenerate data ────────────────────────────────────────────────────────────────────────
def test_constant_data_gives_non_negative_sums_of_squares_and_no_garbage_tests():
    t = np.full(5, 2.0)
    out = crossover_2x2(t, t.copy(), np.array([True, True, False, False, True]))
    for row in out["anova"]:
        assert row["ss"] >= 0 and row["ms"] >= 0, row
        if row["error_term"] is not None:
            assert row["f"] is None and row["p"] is None, row


def test_sums_of_squares_are_never_negative_on_near_degenerate_data():
    rng = np.random.default_rng(5)
    for scale in (1e-14, 1e-9, 1e-3):
        t = np.exp(4.0 + rng.normal(0, scale, 8))
        r = np.exp(4.0 + rng.normal(0, scale, 8))
        for row in crossover_2x2(t, r, np.array([True] * 4 + [False] * 4))["anova"]:
            assert row["ss"] >= 0, (scale, row)


# ── M4: unbalanced ANOVA rows against an independent closed-form computation ───────────────────
def _closed_form_anova(test, ref, test_first):
    lt, lr = np.log(test), np.log(ref)
    y1, y2 = np.where(test_first, lt, lr), np.where(test_first, lr, lt)
    tot, d = y1 + y2, y2 - y1
    g1, g2 = test_first, ~test_first
    n1, n2 = int(g1.sum()), int(g2.sum())
    m1, m2 = tot[g1].mean() / 2, tot[g2].mean() / 2
    ss_seq = 2 * n1 * n2 / (n1 + n2) * (m1 - m2) ** 2
    ss_subj = (((tot[g1] - tot[g1].mean()) ** 2).sum() + ((tot[g2] - tot[g2].mean()) ** 2).sum()) / 2
    s2 = (((d[g1] - d[g1].mean()) ** 2).sum() + ((d[g2] - d[g2].mean()) ** 2).sum()) / (n1 + n2 - 2)
    mse = s2 / 2
    var_eff = mse / 2 * (1 / n1 + 1 / n2)                            # Var of the period and formulation estimates
    per_hat = 0.5 * (d[g1].mean() + d[g2].mean())
    form_hat = 0.5 * (d[g2].mean() - d[g1].mean())
    return {"sequence": ss_seq, "subject(sequence)": ss_subj, "period": per_hat ** 2 / var_eff * mse,
            "formulation": form_hat ** 2 / var_eff * mse, "residual": mse * (n1 + n2 - 2), "mse": mse,
            "n": (n1, n2)}


@pytest.mark.parametrize("n_tr, n_rt, seed", [(5, 3, 31), (9, 4, 32), (2, 7, 33), (12, 12, 34), (3, 3, 35)])
def test_all_anova_rows_match_an_independent_closed_form_in_unbalanced_designs(n_tr, n_rt, seed):
    test, ref, tf = _make(n_tr, n_rt, seed=seed, per=0.6, form=0.1, sd_b=0.4)
    rows = {r["source"]: r for r in crossover_2x2(test, ref, tf)["anova"]}
    want = _closed_form_anova(test, ref, tf)
    for source in ("sequence", "subject(sequence)", "period", "formulation", "residual"):
        assert math.isclose(rows[source]["ss"], want[source], rel_tol=1e-9), source
    n = n_tr + n_rt
    assert math.isclose(rows["period"]["f"], rows["period"]["ms"] / rows["residual"]["ms"], rel_tol=1e-12)
    assert math.isclose(rows["period"]["p"], stats.f.sf(rows["period"]["f"], 1, n - 2), rel_tol=1e-9)
    assert math.isclose(rows["subject(sequence)"]["f"], rows["subject(sequence)"]["ms"] / rows["residual"]["ms"],
                        rel_tol=1e-12)


# ── Welch edge cases and dtype validation in be_model ──────────────────────────────────────────
def test_welch_with_two_per_group_and_with_zero_variances():
    got = parallel_two_group(np.array([10.0, 14.0]), np.array([9.0, 13.0]), variance="welch")
    assert got["estimable"] and math.isfinite(got["df"]) and got["df"] > 0
    flat = parallel_two_group(np.array([5.0, 5.0, 5.0]), np.array([4.0, 4.0, 4.0]), variance="welch")
    assert flat["se"] == 0.0 and flat["df"] == 4
    one_flat = parallel_two_group(np.array([5.0, 5.0, 5.0]), np.array([4.0, 5.0, 6.0]), variance="welch")
    assert one_flat["se"] > 0 and math.isfinite(one_flat["df"])


@pytest.mark.parametrize("bad_first", [np.array(["False", "True", "True", "False"]), np.array([0, 1, 2, 1]),
                                       np.array([0.5, 1.0, 0.0, 1.0])])
def test_test_first_must_be_boolean_or_zero_one(bad_first):
    with pytest.raises(ValueError, match="test_first"):
        crossover_2x2(np.array([1.0, 2, 3, 4]), np.array([1.0, 2, 3, 5]), bad_first)


def test_test_first_accepts_zero_one_integers():
    a = crossover_2x2(np.array([1.0, 2, 3, 4]), np.array([1.0, 2.5, 3, 5]), np.array([1, 0, 1, 0]))
    b = crossover_2x2(np.array([1.0, 2, 3, 4]), np.array([1.0, 2.5, 3, 5]), np.array([True, False, True, False]))
    assert a["estimate"] == b["estimate"]


def test_boolean_values_are_not_concentrations():
    with pytest.raises(ValueError, match="positive|numbers"):
        crossover_2x2(np.array([True, True, True]), np.array([1.0, 2.0, 3.0]), np.array([True, False, True]))


def test_an_estimate_too_large_to_back_transform_is_a_value_error():
    with pytest.raises(ValueError, match="range"):
        crossover_2x2(np.array([1e300, 1e300, 1e300, 1e300]), np.array([1e-300] * 4),
                      np.array([True, False, True, False]))


# ── tool: third arm, status codes and the user-facing summary ──────────────────────────────────
def test_a_third_treatment_sharing_a_period_does_not_trigger_the_test_reference_rule():
    df = _frame(13)
    extra = df[df.TRT == "A"].assign(TRT="C", PERIOD=lambda d: d.PERIOD)   # C is in the same period as A
    import pandas as pd
    res = _call(pd.concat([df, extra]), **FULL).result
    assert res["status"] == "ok" and res["ignored_treatments"] == ["C"]


def test_a_treatment_level_named_replicate_does_not_change_the_refusal_status():
    res = _call(_frame(13), **{**FULL, "reference": "replicate"}).result   # a level that is not in the data
    assert "replicate" in res["message"] and res["status"] == "invalid_arguments"


def test_the_summary_describes_the_current_model_not_the_retired_one():
    summary = _call(_frame(13), **FULL).summary
    assert "no period/sequence terms" not in summary
    assert "validation pending" in summary and "interim" in summary


def test_the_unsupported_design_status_comes_from_the_error_type_not_its_wording():
    recs, _ = _crossover_records(7, 6)
    recs[0]["period"] = 3
    with pytest.raises(BEContractError) as exc:
        assess_average_be(recs, design="crossover", profile="fda_2026", test="T", reference="R")
    from app.compute.be_contract import BEUnsupportedDesign
    assert isinstance(exc.value, BEUnsupportedDesign)


# ── re-review of 1052784 ───────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("c", [5.0, 3.3, 1e-7, 7.77, 0.1, 123456.789, 1.0, 2.0, 1e12, 1e-12])
def test_constant_data_never_yields_f_tests_whatever_the_constant(c):
    tf = np.array([True] * 7 + [False] * 6)
    out = crossover_2x2(np.full(13, c), np.full(13, c), tf)
    for row in out["anova"]:
        assert row["ss"] >= 0 and row["ms"] >= 0, (c, row)
        if row["error_term"] is not None:
            assert row["f"] is None and row["p"] is None, (c, row)


@pytest.mark.parametrize("sd_within", [1e-3, 1e-4, 1e-5])
def test_a_very_precise_assay_still_gets_its_f_tests(sd_within):
    test, ref, tf = _make(8, 8, seed=41, per=0.2, form=0.05, sd_b=0.3, sd_w=sd_within)
    rows = {r["source"]: r for r in crossover_2x2(test, ref, tf)["anova"]}
    for source in ("period", "formulation", "subject(sequence)", "sequence"):
        assert rows[source]["f"] is not None and rows[source]["p"] is not None, (sd_within, source)


def test_a_mixed_type_period_column_is_a_refusal_not_a_crash():

    df = _frame(13).astype({"PERIOD": object})
    df.loc[df.index[0], "PERIOD"] = "2"                              # an int/str mix
    res = _call(df, **FULL).result
    assert res["status"] == "invalid_arguments" and "period" in res["message"].lower()


def test_text_period_labels_are_read_as_numbers_only_in_plain_decimal_syntax():
    """'1_0' and non-ASCII digits are text, not numbers: they are ordered as text."""
    out = _run(_crossover_records(9, 5, labels=("1_0", "2"))[0])["parameters"]["Cmax"]
    assert out["n_per_sequence"] == {"TR": 9, "RT": 5}              # '1_0' < '2' as text; as 10 vs 2 it would flip
    arabic = _run(_crossover_records(9, 5, labels=("٣", "٤"))[0])["parameters"]["Cmax"]
    assert arabic["n_per_sequence"] == {"TR": 9, "RT": 5}


def test_plain_decimal_text_with_exponent_and_spaces_is_still_numeric():
    out = _run(_crossover_records(9, 5, labels=(" 2 ", "1e1"))[0])["parameters"]["Cmax"]
    assert out["n_per_sequence"] == {"TR": 9, "RT": 5}              # 2 before 10


def test_object_arrays_of_booleans_are_accepted_as_flags():
    t, r = np.array([1.0, 2, 3, 4]), np.array([1.0, 2.5, 3, 5])
    obj = np.array([True, False, True, False], dtype=object)
    assert crossover_2x2(t, r, obj)["estimate"] == crossover_2x2(t, r, np.array([True, False, True, False]))["estimate"]


def test_a_boolean_hidden_in_a_list_of_values_is_refused():
    with pytest.raises(ValueError, match="boolean"):
        crossover_2x2([True, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], [True, False, True, False])


def test_no_constant_of_any_magnitude_yields_an_f_test():
    """Property check over 400 random constants (1e-13 .. 1e13) and three designs: no F, no negative SS."""
    rng = np.random.default_rng(2026)
    for design in ([True] * 7 + [False] * 6, [True] * 3 + [False] * 9, [True, False] * 6):
        tf = np.array(design)
        for c in np.exp(rng.uniform(-30, 30, 400)):
            for row in crossover_2x2(np.full(tf.size, c), np.full(tf.size, c), tf)["anova"]:
                assert row["ss"] >= 0, (c, row)
                if row["error_term"] is not None:
                    assert row["f"] is None and row["p"] is None, (c, row)
