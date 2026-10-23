"""run_bioequivalence under the slice-1 safety contract: the tool never guesses the reference, never changes the
design, refuses multi-group and unsupported data, and reports every excluded subject."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.pharmstate import PharmState
from app.tools.base import ToolContext
from app.tools.be_tools import run_bioequivalence

TIMES = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0)


def _profile(scale):
    return [0.0] + [scale * (np.exp(-0.2 * t) - np.exp(-1.5 * t)) * 10 for t in TIMES[1:]]


def _frame(n=12, levels=("A", "B"), drop=()):
    rows = []
    for i in range(n):
        for k, trt in enumerate(levels):
            if (i, trt) in drop:
                continue
            scale = 1.0 + 0.02 * i + (0.01 if k else 0.0)
            for t, c in zip(TIMES, _profile(scale)):
                rows.append({"ID": i + 1, "TIME": t, "DV": c, "TRT": trt, "EVID": 0})
    return pd.DataFrame(rows)


def _call(df, **args):
    state = PharmState(dataset_id="d1")
    ctx = ToolContext(dataset_store={"d1": df})
    return run_bioequivalence(state, ctx, args)


FULL = {"design": "crossover", "profile": "fda_2026", "test": "B", "reference": "A"}


@pytest.mark.parametrize("missing", sorted(FULL))
def test_missing_contract_arguments_fail_closed_with_a_named_list(missing):
    args = {k: v for k, v in FULL.items() if k != missing}
    out = _call(_frame(), **args)
    res = out.result
    assert res["status"] == "missing_required_args" and missing in res["missing"]
    assert "bioequivalent" not in res and out.writes["be_results"]["status"] == "missing_required_args"


def test_the_reference_is_never_guessed_from_level_names():
    out = _call(_frame(levels=("X", "Y")), design="crossover", profile="fda_2026")
    assert out.result["status"] == "missing_required_args"
    assert set(out.result["missing"]) == {"test", "reference"}


def test_one_subject_missing_a_period_keeps_the_crossover_design():
    out = _call(_frame(13, drop={(0, "A")}), **FULL)
    res = out.result
    assert res["status"] == "ok" and res["design"] == "crossover"
    assert any(e["subject"] == 1 and e["reason"] == "no_evaluable_counterpart" for e in res["exclusions"])
    assert res["parameters"]["Cmax"]["n_evaluable"] == 12


def test_a_subject_with_too_few_samples_is_in_the_ledger():
    df = _frame(13)
    df = df[~((df.ID == 2) & (df.TRT == "B") & (df.TIME >= 1.0))]  # subject 2 on B keeps 2 samples
    res = _call(df, **FULL).result
    assert any(e["subject"] == 2 and e["treatment"] == "B" and e["reason"] == "fewer_than_3_samples"
               for e in res["exclusions"])


def test_a_group_column_is_refused_as_unsupported():
    df = _frame()
    df["GROUP"] = 1
    res = _call(df, **FULL, group_col="GROUP").result
    assert res["status"] == "unsupported_design" and "multi-group" in res["message"]


def test_a_replicate_dataset_is_refused():
    df = _frame()
    extra = df[(df.ID == 1) & (df.TRT == "A")].assign(PERIOD=3)
    df = pd.concat([df.assign(PERIOD=np.where(df.TRT == "A", 1, 2)), extra])
    res = _call(df, **FULL, period_col="PERIOD").result
    assert res["status"] == "unsupported_design" and "replicate" in res["message"]


def test_an_unknown_level_is_a_validation_refusal_not_a_crash():
    res = _call(_frame(), **{**FULL, "reference": "Z"}).result
    assert res["status"] == "invalid_arguments" and "Z" in res["message"]


def test_the_summary_states_profile_design_and_the_interim_model():
    out = _call(_frame(), **FULL)
    assert "fda_2026" in out.summary and "crossover" in out.summary and "interim" in out.summary
