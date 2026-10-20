"""Regression tests for GitHub Issues #7, #8 (tool layer), #9, #10, #11, and #12."""
from __future__ import annotations

import math

import numpy as np
import pytest

from app.compute.nca import Profile, nca_subject, refit_lambda_z_manual, run_nca
from app.compute.nca_ss import nca_ss_subject
from app.config import settings
from app.core.llm import MockLLM
from app.core.orchestrator import Orchestrator
from app.core.store import SessionStore


@pytest.fixture
def run_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)

    def _run(text: str) -> dict:
        path = tmp_path / "input.csv"
        path.write_text(text.strip() + "\n")
        orch = Orchestrator(llm=MockLLM(), store=SessionStore(":memory:"))
        sid = orch.create_session().id
        orch.run_tool(sid, "load_dataset", "data_manager", {"path": str(path)})
        res = orch.run_tool(sid, "compute_nca", "nca", {})
        sess = orch.get_session(sid)
        return {
            "params": {p["subject"]: p for p in sess.state.nca_parameters},
            "summary": res["result"]["nca_summary"],
            "result": res["result"],
            "plot": sess.state.nca_plot_data,
        }

    return _run


# ── Issue #7: explicit repeated dose rows without an II column ──────────────────────────
ISSUE7_NO_II = """
ID,TIME,DV,AMT,EVID
A,0,,100,1
A,12,,100,1
A,24,,100,1
A,36,,100,1
A,48,,100,1
A,12,1.836654,0,0
A,24,2.140251,0,0
A,36,2.190435,0,0
A,48,2.198731,0,0
A,48.5,7.099602,0,0
A,49,8.976663,0,0
A,50,9.306984,0,0
A,52,7.277054,0,0
A,56,4.008779,0,0
A,60,2.200102,0,0
"""

ISSUE7_WITH_II = """
ID,TIME,DV,AMT,EVID,II
A,0,,100,1,12
A,12,,100,1,12
A,24,,100,1,12
A,36,,100,1,12
A,48,,100,1,12
A,12,1.836654,0,0,
A,24,2.140251,0,0,
A,36,2.190435,0,0,
A,48,2.198731,0,0,
A,48.5,7.099602,0,0,
A,49,8.976663,0,0,
A,50,9.306984,0,0,
A,52,7.277054,0,0,
A,56,4.008779,0,0,
A,60,2.200102,0,0,
"""


def test_issue7_explicit_repeated_doses_without_ii_infer_tau_and_match_ss(run_csv):
    no_ii = run_csv(ISSUE7_NO_II)
    with_ii = run_csv(ISSUE7_WITH_II)
    assert no_ii["result"].get("steady_state") is True
    assert no_ii["params"]["A"] == with_ii["params"]["A"]
    assert no_ii["params"]["A"]["tau"] == 12.0
    assert no_ii["params"]["A"]["AUC_tau"] == pytest.approx(65.97, abs=0.01)
    assert no_ii["params"]["A"]["CL_F"] == pytest.approx(1.516, abs=0.01)


def test_issue7_explicit_repeated_doses_with_incomplete_final_interval_fail_closed(run_csv):
    """Repeated dose rows with samples only after the first dose must fail closed, never run single-dose NCA."""
    csv_text = """
ID,TIME,DV,AMT,EVID
A,0,,100,1
A,12,,100,1
A,24,,100,1
A,1,5,0,0
A,2,8,0,0
A,4,4,0,0
A,8,2,0,0
"""
    with pytest.raises(ValueError, match=r"steady-state NCA stopped.*'A'"):
        run_csv(csv_text)


def test_issue7_irregular_repeated_doses_without_ii_fail_closed(run_csv):
    """Non-constant dose spacing without an II column cannot form a steady-state interval and must fail closed."""
    csv_text = """
ID,TIME,DV,AMT,EVID
A,0,,100,1
A,8,,100,1
A,24,,100,1
A,24,2.0,0,0
A,25,6.0,0,0
A,26,8.0,0,0
A,28,5.0,0,0
A,32,3.0,0,0
A,40,2.0,0,0
"""
    with pytest.raises(ValueError, match=r"steady-state NCA stopped.*'A'.*constant interdose interval"):
        run_csv(csv_text)


# ── Issue #8 (tool layer): load_dataset without path raises ValueError, not KeyError ────
def test_issue8_load_dataset_missing_path_raises_value_error():
    orch = Orchestrator(llm=MockLLM(), store=SessionStore(":memory:"))
    sid = orch.create_session().id
    with pytest.raises(ValueError, match="path"):
        orch.run_tool(sid, "load_dataset", "data_manager", {})


# ── Issue #9: single-dose Tmax, Tlast, and MRT are relative to first dose time ──────────
def _issue9_csv(offset: float) -> str:
    obs = [(0, 0), (1, 5), (2, 8), (4, 4), (8, 2), (12, 1), (24, 0.25)]
    lines = ["ID,TIME,DV,AMT,EVID", f"A,{offset:g},,100,1"]
    lines += [f"A,{offset + t:g},{c},0,0" for t, c in obs]
    return "\n".join(lines)


def test_issue9_nonzero_dose_time_reports_time_since_dose_for_tmax_tlast_and_mrt(run_csv):
    at_0 = run_csv(_issue9_csv(0.0))
    at_24 = run_csv(_issue9_csv(24.0))
    p0, p24 = at_0["params"]["A"], at_24["params"]["A"]
    assert p24["Tmax"] == p0["Tmax"] == 2.0
    assert p24["Tlast"] == p0["Tlast"] == 24.0
    assert p24["MRT"] == p0["MRT"] == pytest.approx(7.3488, abs=1e-4)
    assert p24["AUC_last"] == p0["AUC_last"] == pytest.approx(44.346, abs=1e-3)
    assert p24["AUC_inf"] == p0["AUC_inf"] == pytest.approx(46.320, abs=1e-3)
    assert p24["CL_F"] == p0["CL_F"] == pytest.approx(2.1589, abs=1e-4)
    assert p24["lambda_z_intervals"] == p0["lambda_z_intervals"]
    assert at_24["plot"]["subjects"][0]["x"] == at_0["plot"]["subjects"][0]["x"]
    assert at_24["plot"]["subjects"][0]["fit_y"] == at_0["plot"]["subjects"][0]["fit_y"]


def test_issue9_run_nca_accepts_dose_time_by_subject():
    obs = [(0, 0), (1, 5), (2, 8), (4, 4), (8, 2), (12, 1), (24, 0.25)]
    recs_0 = [{"ID": "A", "TIME": float(t), "DV": float(c)} for t, c in obs]
    recs_24 = [{"ID": "A", "TIME": float(t + 24), "DV": float(c)} for t, c in obs]
    out_0 = run_nca(recs_0, id_col="ID", time_col="TIME", dv_col="DV", dose_by_subject={"A": 100.0})
    out_24 = run_nca(recs_24, id_col="ID", time_col="TIME", dv_col="DV",
                     dose_by_subject={"A": 100.0}, dose_time_by_subject={"A": 24.0})
    assert out_24["nca_parameters"][0] == out_0["nca_parameters"][0]


# ── Issue #10: small concentration units preserve significant figures ───────────────────
@pytest.mark.parametrize("scale", [1.0, 1e-7, 1e-9])
def test_issue10_small_concentration_units_preserve_precision_and_consistency(scale, run_csv):
    obs = [(0, 0), (1, 5), (2, 8), (4, 4), (8, 2), (12, 1), (24, 0.25)]
    lines = ["ID,TIME,DV,AMT,EVID", "A,0,,100,1"]
    lines += [f"A,{t},{c * scale:.15g},0,0" for t, c in obs]
    out = run_csv("\n".join(lines))
    p = out["params"]["A"]
    assert p["Cmax"] == pytest.approx(8.0 * scale, rel=1e-6)
    assert p["Clast"] == pytest.approx(0.25 * scale, rel=1e-6)
    assert p["AUC_last"] == pytest.approx(44.346029 * scale, rel=1e-6)
    assert p["AUC_inf"] == pytest.approx(46.320201 * scale, rel=1e-6)
    assert p["Cmax"] > 0 and p["AUC_last"] > 0 and p["AUC_inf"] > 0
    # Derived CL/F must stay consistent with the reported Dose / AUC_inf
    assert p["CL_F"] == pytest.approx(100.0 / p["AUC_inf"], rel=1e-5)
    # Summary geomeans also preserve significant figures
    by_dose = out["summary"]["by_dose"][0]
    assert by_dose["Cmax_geomean"] == pytest.approx(8.0 * scale, rel=1e-3)
    assert by_dose["AUC_last_geomean"] == pytest.approx(44.346029 * scale, rel=1e-3)


def test_issue10_steady_state_small_concentration_units():
    tad = [0, 1, 2, 4, 8, 12]
    conc = [c * 1e-9 for c in (2.0, 6.0, 8.0, 5.0, 3.0, 2.0)]
    r = nca_ss_subject("S", tad, conc, dose=100.0, tau=12.0)
    assert r["Cmax"] == pytest.approx(8e-9, rel=1e-6)
    assert r["Cmin"] == pytest.approx(2e-9, rel=1e-6)
    assert r["AUC_tau"] > 0
    assert r["CL_F"] == pytest.approx(100.0 / r["AUC_tau"], rel=1e-5)


# ── Issue #11: unflagged +/-inf DV never becomes Cmax or poisons AUC ────────────────────
@pytest.mark.parametrize("bad", ["inf", "-inf"])
@pytest.mark.parametrize("with_blq", [False, True])
def test_issue11_unflagged_infinite_dv_is_excluded_in_single_dose_nca(bad, with_blq, run_csv):
    obs = [(0, "0"), (1, "5"), (2, "8"), (4, "4"), (8, "2"), (12, "1"), (24, "0.25")]
    cols = "ID,TIME,DV,AMT,EVID" + (",BLQ" if with_blq else "")
    flag = ",0" if with_blq else ""
    clean_lines = [cols, f"A,0,,100,1{flag}"] + [f"A,{t},{c},0,0{flag}" for t, c in obs]
    inf_lines = clean_lines + [f"A,6,{bad},0,0{flag}"]

    clean = run_csv("\n".join(clean_lines))
    with_inf = run_csv("\n".join(inf_lines))
    p = with_inf["params"]["A"]
    assert p == clean["params"]["A"]
    assert p["Cmax"] == 8.0 and p["Tmax"] == 2.0
    assert math.isfinite(p["AUC_last"]) and math.isfinite(p["AUC_inf"])
    assert math.isfinite(p["MRT"]) and math.isfinite(p["CL_F"])
    assert with_inf["summary"]["blq"]["n_below_loq"] == clean["summary"]["blq"]["n_below_loq"]


@pytest.mark.parametrize("bad", [float("inf"), float("-inf")])
def test_issue11_nca_subject_ignores_nonfinite_conc(bad):
    clean = nca_subject(Profile("A", np.array([0, 1, 2, 4, 8, 12, 24], float),
                                np.array([0, 5, 8, 4, 2, 1, 0.25], float), 100.0))
    dirty = nca_subject(Profile("A", np.array([0, 1, 2, 4, 6, 8, 12, 24], float),
                                np.array([0, 5, 8, 4, bad, 2, 1, 0.25], float), 100.0))
    assert dirty == clean


# ── Issue #12: flat terminal phase is rejected, not accepted on float noise ─────────────
def test_issue12_steady_state_flat_terminal_phase_reports_lambda_z_none():
    r = nca_ss_subject("S", tad=(0, 1, 2, 4, 8, 12), conc=(2, 6, 8, 5, 5, 5), dose=100, tau=12)
    for k in ("lambda_z", "t_half", "Vz_F", "accumulation_ratio", "lambda_z_r2_adj", "lambda_z_n_points"):
        assert r[k] is None, k
    # Interval exposures that do not depend on lambda_z remain finite and valid
    assert r["AUC_tau"] > 0 and r["CL_F"] == pytest.approx(100.0 / r["AUC_tau"], rel=1e-5)


def test_issue12_single_dose_flat_terminal_phase_reports_lambda_z_none():
    r = nca_subject(Profile("S", np.array([0, 1, 2, 4, 8, 12], float),
                            np.array([0, 6, 8, 5, 5, 5], float), 100.0))
    assert r["AUC_last"] == pytest.approx(62.765859, abs=1e-5)
    for k in ("lambda_z", "t_half", "AUC_inf", "pct_AUC_extrap", "MRT", "CL_F", "Vz_F", "Vss",
              "lambda_z_r2_adj", "lambda_z_n_points"):
        assert r[k] is None, k


def test_issue12_negative_adjusted_r2_terminal_window_is_rejected():
    """A terminal window whose fit is worse than a horizontal line (adj R^2 <= 0) is not a terminal phase."""
    r = nca_subject(Profile("S", np.array([0, 1, 2, 4, 8, 12], float),
                            np.array([0, 6, 8, 5.0, 4.0, 4.95], float), 100.0))
    assert r["lambda_z"] is None and r["AUC_inf"] is None


def test_issue12_manual_refit_rejects_flat_points():
    with pytest.raises(ValueError, match="declining trend"):
        refit_lambda_z_manual([4.0, 8.0, 12.0], [5.0, 5.0, 5.0])
