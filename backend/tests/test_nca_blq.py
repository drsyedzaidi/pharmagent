"""NCA and below-LOQ data. The tool's documented rule (M1): BLQ records are excluded from lambda_z and AUC, leading BLQ
counts as a 0 baseline, and AUC_last ends at the last measurable concentration (Tlast). These tests pin that rule at the
compute level and through the real tool with a BLQ flag (role CENS)."""
from __future__ import annotations

import csv
import math

import numpy as np
import pytest

from app.compute.nca import Profile, nca_subject
from app.config import settings
from app.core.llm import MockLLM
from app.core.orchestrator import Orchestrator
from app.core.store import SessionStore

T = np.array([0.0, 1, 2, 4, 8, 12, 24, 36, 48])
C = np.array([0.0, 5.0, 8.0, 4.0, 2.0, 1.0, 0.25, 0.0, 0.0])


# ── compute level: AUC_last ends at Tlast ───────────────────────────────────────────────────────
def test_trailing_zeros_do_not_extend_auc_last():
    with_tail = nca_subject(Profile("S", T, C, 100.0))
    without = nca_subject(Profile("S", T[:7], C[:7], 100.0))
    assert with_tail["Tlast"] == 24.0 and with_tail["Clast"] == 0.25
    assert with_tail["AUC_last"] == without["AUC_last"]
    assert with_tail["AUC_inf"] == without["AUC_inf"]
    assert with_tail["MRT"] == without["MRT"]
    assert with_tail["CL_F"] == without["CL_F"]


def test_auc_last_equals_the_hand_computed_value():
    """Linear-up/log-down by hand for the profile above, to Tlast = 24 h only."""
    auc = 0.0
    for i in range(6):
        t1, t2, c1, c2 = T[i], T[i + 1], C[i], C[i + 1]
        auc += (t2 - t1) * (c1 + c2) / 2 if c2 >= c1 else (t2 - t1) * (c1 - c2) / math.log(c1 / c2)
    out = nca_subject(Profile("S", T, C, 100.0))
    assert out["AUC_last"] == pytest.approx(auc, abs=1e-5)


def test_a_profile_without_trailing_zeros_is_unchanged():
    a = nca_subject(Profile("S", T[:7], C[:7], 100.0))
    assert a["AUC_last"] > 0 and a["Tlast"] == 24.0 and a["lambda_z_n_points"] >= 3


def test_an_all_zero_profile_still_reports_zero_area():
    out = nca_subject(Profile("S", T[:4], np.zeros(4), 100.0))
    assert out["AUC_last"] == 0.0 and out["Cmax"] == 0.0


# ── through the tool, with a BLQ flag ───────────────────────────────────────────────────────────
@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)

    def _run(rows: list[tuple], with_flag: bool = True) -> dict[str, dict]:
        cols = ["ID", "TIME", "DV", "AMT", "EVID"] + (["BLQ"] if with_flag else [])
        path = tmp_path / "d.csv"
        with path.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(cols)
            for sid in dict.fromkeys(r[0] for r in rows):
                w.writerow([sid, 0, "", 100, 1] + ([0] if with_flag else []))
            for sid, t, dv, blq in rows:
                w.writerow([sid, t, "" if dv is None else dv, 0, 0] + ([blq] if with_flag else []))
        orch = Orchestrator(llm=MockLLM(), store=SessionStore(":memory:"))
        sid = orch.create_session().id
        orch.run_tool(sid, "load_dataset", "data_manager", {"path": str(path)})
        orch.run_tool(sid, "compute_nca", "nca", {})
        return {p["subject"]: p for p in orch.get_session(sid).state.nca_parameters}

    return _run


def _profile(sid, pairs, blq_times=()):
    return [(sid, t, c, 1 if t in blq_times else 0) for t, c in pairs]


PAIRS = list(zip(T.tolist(), C.tolist(), strict=True))


def test_trailing_blq_rows_are_excluded(run):
    flagged = run(_profile("A", PAIRS, blq_times=(36.0, 48.0)))["A"]
    clean = run(_profile("A", PAIRS[:7]))["A"]
    for k in ("AUC_last", "AUC_inf", "lambda_z", "Tlast", "Cmax"):
        assert flagged[k] == clean[k], k


def test_a_middle_blq_row_is_excluded_not_integrated_as_zero(run):
    pairs = [(0.0, 0.0), (1.0, 5.0), (2.0, 8.0), (4.0, 4.0), (8.0, 0.0), (12.0, 1.0), (24.0, 0.25)]
    flagged = run(_profile("B", pairs, blq_times=(8.0,)))["B"]
    dropped = run(_profile("B", [p for p in pairs if p[0] != 8.0]))["B"]
    kept_as_zero = run(_profile("B", pairs), with_flag=False)["B"]
    assert flagged["AUC_last"] == dropped["AUC_last"] and flagged["AUC_inf"] == dropped["AUC_inf"]
    assert flagged["AUC_last"] > kept_as_zero["AUC_last"]            # the zero would have cut the area


def test_leading_blq_rows_count_as_a_zero_baseline_even_when_the_value_is_blank(run):
    pairs = [(0.0, None), (0.5, None), (1.0, 5.0), (2.0, 8.0), (4.0, 4.0), (8.0, 2.0), (12.0, 1.0)]
    leading = run(_profile("C", pairs, blq_times=(0.0, 0.5)))["C"]
    explicit_zero = run([("C", t, 0.0 if c is None else c, 0) for t, c in pairs], with_flag=False)["C"]
    assert leading["AUC_last"] == explicit_zero["AUC_last"] and leading["Cmax"] == 8.0


def test_without_a_flag_trailing_zeros_no_longer_inflate_the_area(run):
    with_tail = run(_profile("D", PAIRS), with_flag=False)["D"]
    clean = run(_profile("D", PAIRS[:7]), with_flag=False)["D"]
    assert with_tail["AUC_last"] == clean["AUC_last"]


def test_an_all_blq_subject_keeps_its_current_reporting(run):
    out = run(_profile("E", [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)], blq_times=(0.0, 1.0, 2.0)))["E"]
    assert out["AUC_last"] == 0.0


def test_the_result_states_the_rule_it_applies(run, tmp_path):
    # the summary text is what a reviewer reads: it must say AUC_last ends at Tlast
    orch = Orchestrator(llm=MockLLM(), store=SessionStore(":memory:"))
    sid = orch.create_session().id
    path = tmp_path / "r.csv"
    path.write_text("ID,TIME,DV,AMT,EVID,BLQ\nA,0,,100,1,0\nA,1,5,0,0,0\nA,2,2,0,0,0\nA,4,1,0,0,0\nA,8,0.5,0,0,0\n")
    orch.run_tool(sid, "load_dataset", "data_manager", {"path": str(path)})
    res = orch.run_tool(sid, "compute_nca", "nca", {})
    assert "Tlast" in res["result"]["nca_summary"]["blq"]["rule"]
