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


# ── a blank, unflagged DV is not a quantifiable value ──────────────────────────────────────────
def test_a_blank_unflagged_dv_does_not_end_the_leading_blq_baseline(run):
    """A row with no DV and no BLQ flag carries no measurement. It must not count as the first quantifiable value,
    or the flagged leading-BLQ row after it would be treated as 'middle' and dropped instead of becoming a 0 baseline."""
    rest = [("F", 1.0, 5.0, 0), ("F", 2.0, 8.0, 0), ("F", 4.0, 4.0, 0), ("F", 8.0, 2.0, 0), ("F", 12.0, 1.0, 0)]
    flagged = run([("F", 0.0, None, 0), ("F", 0.5, None, 1)] + rest)["F"]
    zero_baseline = run([("F", 0.5, 0.0, 0)] + rest, with_flag=False)["F"]
    assert flagged["AUC_last"] == zero_baseline["AUC_last"] and flagged["Cmax"] == 8.0


# ── steady-state NCA applies the same BLQ rule ────────────────────────────────────────────────
@pytest.fixture
def run_ss(tmp_path, monkeypatch):
    """Steady-state dataset: 100 mg every 12 h, doses at 0/12/24/36, last interval 36-48 h. rows are
    (id, tad in the last interval, dv, blq); times are written as 36 + tad."""
    monkeypatch.setattr(settings, "data_dir", tmp_path)

    def _run(rows: list[tuple], with_flag: bool = True, dose_blq: int = 0, dose_dv: object = "") -> dict:
        cols = ["ID", "TIME", "DV", "AMT", "EVID", "II", "ADDL"] + (["BLQ"] if with_flag else [])
        path = tmp_path / "ss.csv"
        with path.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(cols)
            for sid in dict.fromkeys(r[0] for r in rows):
                w.writerow([sid, 0, dose_dv, 100, 1, 12, 3] + ([dose_blq] if with_flag else []))
            for sid, tad, dv, blq in rows:
                w.writerow([sid, 36 + tad, "" if dv is None else dv, 0, 0, "", ""] + ([blq] if with_flag else []))
        orch = Orchestrator(llm=MockLLM(), store=SessionStore(":memory:"))
        sid = orch.create_session().id
        orch.run_tool(sid, "load_dataset", "data_manager", {"path": str(path)})
        res = orch.run_tool(sid, "compute_nca", "nca", {})
        params = {p["subject"]: p for p in orch.get_session(sid).state.nca_parameters}
        return {"params": params, "summary": res["result"]["nca_summary"]}

    return _run


SS_PROFILE = [(0.0, 2.0), (1.0, 6.0), (2.0, 8.0), (4.0, 5.0), (8.0, 3.0), (12.0, 2.0)]


def _ss_rows(sid, pairs, blq_at=(), value_at=None):
    value_at = value_at or {}
    return [(sid, t, value_at.get(t, c), 1 if t in blq_at else 0) for t, c in pairs]


def test_steady_state_middle_blq_row_is_excluded_not_integrated_as_zero(run_ss):
    pairs = [(t, 0.0 if t == 4.0 else c) for t, c in SS_PROFILE]
    flagged = run_ss(_ss_rows("M", pairs, blq_at=(4.0,)))["params"]["M"]
    dropped = run_ss(_ss_rows("M", [p for p in pairs if p[0] != 4.0]), with_flag=False)["params"]["M"]
    kept_as_zero = run_ss(_ss_rows("M", pairs), with_flag=False)["params"]["M"]
    assert flagged["AUC_tau"] == dropped["AUC_tau"]
    assert flagged["AUC_tau"] > kept_as_zero["AUC_tau"]            # the zero would have cut the area


def test_steady_state_trailing_blq_at_tau_still_integrates_the_whole_interval(run_ss):
    """Dropping the BLQ sample at tau must not shrink AUC_tau to [0, 8]: the interval is closed with the documented
    steady-state convention C(tau) = C(0), so the reference is the same profile with C(tau) set to C(0) explicitly."""
    pairs = [(t, 0.0 if t == 12.0 else c) for t, c in SS_PROFILE]
    flagged = run_ss(_ss_rows("T", pairs, blq_at=(12.0,)))["params"]["T"]
    c0 = SS_PROFILE[0][1]
    closed = run_ss(_ss_rows("T", [(t, c0 if t == 12.0 else c) for t, c in pairs]), with_flag=False)["params"]["T"]
    assert flagged["AUC_tau"] == closed["AUC_tau"] and flagged["CL_F"] == closed["CL_F"]


def test_steady_state_leading_blq_row_is_a_zero_baseline_even_when_the_value_is_blank(run_ss):
    flagged = run_ss(_ss_rows("L", SS_PROFILE, blq_at=(0.0,), value_at={0.0: None}))["params"]["L"]
    zero_baseline = run_ss(_ss_rows("L", SS_PROFILE, value_at={0.0: 0.0}), with_flag=False)["params"]["L"]
    assert flagged["AUC_tau"] == zero_baseline["AUC_tau"]


def test_steady_state_keeps_the_dose_schedule(run_ss):
    out = run_ss(_ss_rows("S", SS_PROFILE))
    assert out["params"]["S"]["tau"] == 12.0 and out["params"]["S"]["dose"] == 100.0
    assert out["summary"]["by_dose"]


def test_steady_state_rule_text_describes_what_was_applied(run_ss):
    flagged = run_ss(_ss_rows("S", SS_PROFILE, blq_at=(12.0,)))["summary"]["blq"]["rule"]
    assert "0 baseline" in flagged and "excluded" in flagged and "steady-state" in flagged.lower()
    assert "lambda_z" not in flagged                      # no lambda_z claim: the SS terminal slope is not part of this rule
    unflagged = run_ss(_ss_rows("S", SS_PROFILE), with_flag=False)["summary"]["blq"]["rule"]
    assert "No BLQ flag" in unflagged and "M1" not in unflagged


# ── M1 is scoped to the selected final interval ──────────────────────────────────────────────────
def test_steady_state_m1_is_scoped_to_the_final_interval_not_the_whole_history(run_ss):
    """Quantifiable earlier-interval samples must not turn a BLQ at the start of the final interval into a 'middle'
    record: within the final interval it is leading, so it is a 0 baseline (here the value is blank)."""
    earlier = [(-10.0, 3.0), (-6.0, 2.5)]                      # 26 h and 30 h: an earlier dosing interval
    flagged = run_ss(_ss_rows("W", earlier + SS_PROFILE, blq_at=(0.0,), value_at={0.0: None}))["params"]["W"]
    reference = run_ss(_ss_rows("W", earlier + SS_PROFILE, value_at={0.0: 0.0}), with_flag=False)["params"]["W"]
    assert flagged["AUC_tau"] == reference["AUC_tau"]


def test_steady_state_earlier_interval_samples_do_not_change_the_final_interval_result(run_ss):
    earlier = [(-10.0, 3.0), (-6.0, 2.5)]
    with_history = run_ss(_ss_rows("H", earlier + SS_PROFILE, blq_at=(12.0,), value_at={12.0: 0.0}))["params"]["H"]
    alone = run_ss(_ss_rows("H", SS_PROFILE, blq_at=(12.0,), value_at={12.0: 0.0}))["params"]["H"]
    assert with_history["AUC_tau"] == alone["AUC_tau"]


# ── quantifiable needs a finite TIME as well as a finite DV ───────────────────────────────────────
def test_a_row_without_a_finite_time_does_not_count_as_quantifiable():
    import pandas as pd

    from app.tools.nca_tools import _apply_m1

    obs = pd.DataFrame({"ID": ["A", "A", "A"], "TIME": [1.0, float("nan"), float("nan")],
                        "DV": [float("nan"), 5.0, float("nan")], "BLQ": [1, 0, 1]})
    out = _apply_m1(obs, "ID", "TIME", "DV", "BLQ")
    # nothing here is quantifiable (the only finite DV has no time), so both BLQ rows are baseline zeros
    assert sorted(out.loc[out["BLQ"] == 1, "DV"].tolist()) == [0.0, 0.0]


# ── BLQ records are counted from observation rows only ───────────────────────────────────────────
def test_steady_state_blq_count_ignores_dose_rows_with_a_flag(run_ss):
    out = run_ss(_ss_rows("C", SS_PROFILE, blq_at=(12.0,), value_at={12.0: 0.0}), dose_blq=1)
    assert out["summary"]["blq"]["n_below_loq"] == 1


def test_steady_state_unflagged_zero_count_ignores_dose_rows(run_ss):
    out = run_ss(_ss_rows("C", SS_PROFILE, value_at={12.0: 0.0}), with_flag=False, dose_dv=0)
    assert out["summary"]["blq"]["n_below_loq"] == 1


# ── the dose schedule survives extraction ────────────────────────────────────────────────────────
def _ss_records(blq: bool):
    recs = [{"ID": "S", "TIME": 0, "DV": None, "AMT": 100, "II": 12, "ADDL": 3, "BLQ": 0}]
    for tad, dv in [(-10.0, 3.0), (-6.0, 2.5)] + SS_PROFILE:
        recs.append({"ID": "S", "TIME": 36 + tad, "DV": dv, "AMT": 0, "II": None, "ADDL": None, "BLQ": 0})
    return recs


@pytest.mark.parametrize("use_blq_col", [False, True])
def test_the_dose_schedule_and_last_interval_survive_extraction(use_blq_col):
    from app.compute.dosing import extract_ss_intervals

    kw = {"blq_col": "BLQ"} if use_blq_col else {}
    prof = extract_ss_intervals(_ss_records(use_blq_col), id_col="ID", time_col="TIME", dv_col="DV", amt_col="AMT",
                                ii_col="II", addl_col="ADDL", **kw)["S"]
    assert (prof.n_doses, prof.tau, prof.dose) == (4, 12.0, 100.0)          # 1 dose row + ADDL 3, every 12 h
    assert prof.tad == (0.0, 1.0, 2.0, 4.0, 8.0, 12.0)                      # only the 36-48 h interval, earlier samples out
    assert prof.c0_source == "measured"


# ── endpoint and row-classification boundaries ────────────────────────────────────────────────────
def _extract(recs, **kw):
    from app.compute.dosing import extract_ss_intervals

    return extract_ss_intervals(recs, id_col="ID", time_col="TIME", dv_col="DV", amt_col="AMT",
                                ii_col="II", addl_col="ADDL", blq_col="BLQ", require_complete_interval=True, **kw)


def _obs(sid, tad, dv, blq=0):
    return {"ID": sid, "TIME": 36 + tad, "DV": dv, "AMT": 0, "II": None, "ADDL": None, "BLQ": blq}


def test_trailing_blq_profile_spans_exactly_zero_to_tau():
    recs = [{"ID": "S", "TIME": 0, "DV": None, "AMT": 100, "II": 12, "ADDL": 3, "BLQ": 0}]
    recs += [_obs("S", t, 0.0 if t == 12.0 else c, int(t == 12.0)) for t, c in SS_PROFILE]
    prof = _extract(recs)["S"]
    assert prof.tad[0] == 0.0 and prof.tad[-1] == 12.0
    assert prof.conc[-1] == prof.conc[0]                                   # C(tau) = C(0)
    assert prof.ctau_source == "c0_assumed"


def _physical_doses(last_flag: int):
    """Four explicit dose rows (0/12/24/36 h, no ADDL); the final one is BLQ-flagged with a blank DV."""
    return [{"ID": "P", "TIME": t, "DV": None, "AMT": 100, "II": 12, "ADDL": 0, "BLQ": last_flag if t == 36 else 0}
            for t in (0, 12, 24, 36)]


def test_a_flagged_final_dose_row_is_not_a_concentration():
    obs = [_obs("P", t, c) for t, c in SS_PROFILE if t != 0.0]             # no tad-0 sample: C(0) must come from C(tau)
    flagged = _extract(_physical_doses(1) + obs)["P"]
    plain = _extract(_physical_doses(0) + obs)["P"]
    assert flagged == plain
    assert flagged.n_doses == 4 and flagged.c0_source == "ctau_assumed" and 0.0 not in flagged.conc


def test_a_flagged_final_dose_row_does_not_change_auc_tau(run, tmp_path):
    path = tmp_path / "phys.csv"
    lines = ["ID,TIME,DV,AMT,EVID,II,ADDL,BLQ"]
    for flag in (1, 0):
        sid = f"P{flag}"
        lines += [f"{sid},{t},,100,1,12,0,{flag if t == 36 else 0}" for t in (0, 12, 24, 36)]
        lines += [f"{sid},{36 + t},{c},0,0,,,0" for t, c in SS_PROFILE if t != 0.0]
    path.write_text("\n".join(lines) + "\n")
    orch = Orchestrator(llm=MockLLM(), store=SessionStore(":memory:"))
    sid = orch.create_session().id
    orch.run_tool(sid, "load_dataset", "data_manager", {"path": str(path)})
    orch.run_tool(sid, "compute_nca", "nca", {})
    p = {r["subject"]: r for r in orch.get_session(sid).state.nca_parameters}
    assert p["P1"]["AUC_tau"] == p["P0"]["AUC_tau"] and p["P1"]["Cmin"] == p["P0"]["Cmin"]


@pytest.mark.parametrize("bad", [float("inf"), float("-inf")])
def test_an_infinite_dv_neither_ends_the_leading_phase_nor_enters_the_profile(bad):
    dose = [{"ID": "I", "TIME": 0, "DV": None, "AMT": 100, "II": 12, "ADDL": 3, "BLQ": 0}]
    rest = [_obs("I", t, c) for t, c in SS_PROFILE if t >= 1.0]
    with_inf = _extract(dose + [_obs("I", 0.0, bad), _obs("I", 0.5, None, 1)] + rest)["I"]
    reference = _extract(dose + [_obs("I", 0.5, 0.0)] + rest)["I"]
    assert with_inf == reference
    assert all(math.isfinite(c) for c in with_inf.conc) and 0.0 in with_inf.conc   # the 0.5 h BLQ became a 0 baseline


# ── near-tau samples, unclosable subjects, endpoint provenance ────────────────────────────────────
@pytest.mark.parametrize("eps", [-0.2, 0.2])                     # tolerance is 2% of tau = 0.24 h
def test_a_sample_within_the_tau_tolerance_is_placed_exactly_at_tau(eps, run_ss):
    dose = [{"ID": "N", "TIME": 0, "DV": None, "AMT": 100, "II": 12, "ADDL": 3, "BLQ": 0}]
    near = [_obs("N", 12.0 + eps if t == 12.0 else t, c) for t, c in SS_PROFILE]
    prof = _extract(dose + near)["N"]
    assert prof.tad[0] == 0.0 and prof.tad[-1] == 12.0 and prof.ctau_source == "measured"
    shifted = run_ss(_ss_rows("N", [(12.0 + eps if t == 12.0 else t, c) for t, c in SS_PROFILE]))["params"]["N"]
    exact = run_ss(_ss_rows("N", SS_PROFILE))["params"]["N"]
    assert shifted["AUC_tau"] == exact["AUC_tau"]


UNCLOSABLE = [(1.0, 6.0), (2.0, 8.0), (4.0, 5.0)]                # no sample at 0 or near tau: no interval end known


def test_one_unclosable_subject_is_not_silently_omitted(run_ss):
    rows = _ss_rows("V", SS_PROFILE) + _ss_rows("U", UNCLOSABLE)
    with pytest.raises(ValueError, match=r"U") as err:
        run_ss(rows)
    assert "complete" in str(err.value) and "'V'" not in str(err.value)


def test_all_unclosable_subjects_fail_closed_never_single_dose_nca(run_ss):
    with pytest.raises(ValueError, match=r"steady-state"):
        run_ss(_ss_rows("U", UNCLOSABLE))


def test_the_public_result_records_the_c_tau_equals_c0_assumption(run_ss):
    pairs = [(t, 0.0 if t == 12.0 else c) for t, c in SS_PROFILE]
    out = run_ss(_ss_rows("T", pairs, blq_at=(12.0,)) + _ss_rows("M", SS_PROFILE))
    assert out["params"]["T"]["ctau_source"] == "c0_assumed" and out["params"]["M"]["ctau_source"] == "measured"
    assert out["params"]["T"]["c0_source"] == "measured"
    assert out["summary"]["blq"]["ctau_assumed_c0"] == ["T"]
    assert "C(tau) = C(0)" in out["summary"]["blq"]["rule"]


# ── the complete-interval contract holds without a BLQ column too ─────────────────────────────────
@pytest.mark.parametrize("eps", [-0.2, 0.2])
def test_no_flag_sample_within_the_tau_tolerance_is_placed_exactly_at_tau(eps, run_ss):
    from app.compute.dosing import extract_ss_intervals_checked

    recs = [{"ID": "N", "TIME": 0, "DV": None, "AMT": 100, "II": 12, "ADDL": 3}]
    recs += [{"ID": "N", "TIME": 36 + (12.0 + eps if t == 12.0 else t), "DV": c, "AMT": 0, "II": None, "ADDL": None}
             for t, c in SS_PROFILE]
    prof = extract_ss_intervals_checked(recs, id_col="ID", time_col="TIME", dv_col="DV", amt_col="AMT", ii_col="II",
                                        addl_col="ADDL", require_complete_interval=True)[0]["N"]
    assert prof.tad[0] == 0.0 and prof.tad[-1] == 12.0 and prof.ctau_source == "measured"
    shifted = run_ss(_ss_rows("N", [(12.0 + eps if t == 12.0 else t, c) for t, c in SS_PROFILE]), with_flag=False)
    exact = run_ss(_ss_rows("N", SS_PROFILE), with_flag=False)
    assert shifted["params"]["N"]["AUC_tau"] == exact["params"]["N"]["AUC_tau"]


def test_no_flag_one_unclosable_subject_is_not_silently_omitted(run_ss):
    with pytest.raises(ValueError, match=r"'U'") as err:
        run_ss(_ss_rows("V", SS_PROFILE) + _ss_rows("U", UNCLOSABLE), with_flag=False)
    assert "neither C(0) nor C(tau)" in str(err.value) and "'V'" not in str(err.value)


def test_no_flag_all_unclosable_subjects_fail_closed(run_ss):
    with pytest.raises(ValueError, match=r"steady-state"):
        run_ss(_ss_rows("U", UNCLOSABLE), with_flag=False)


def test_no_flag_result_records_truthful_endpoint_provenance(run_ss):
    missing_tau = [(t, c) for t, c in SS_PROFILE if t != 12.0]
    out = run_ss(_ss_rows("T", missing_tau) + _ss_rows("M", SS_PROFILE), with_flag=False)
    t, m = out["params"]["T"], out["params"]["M"]
    assert (t["c0_source"], t["ctau_source"]) == ("measured", "c0_assumed")
    assert (m["c0_source"], m["ctau_source"]) == ("measured", "measured")
    c0 = SS_PROFILE[0][1]
    closed = run_ss(_ss_rows("T", missing_tau + [(12.0, c0)]), with_flag=False)["params"]["T"]
    assert t["AUC_tau"] == closed["AUC_tau"]                                # integrated over the whole [0, 12]
    assert out["summary"]["blq"]["ctau_assumed_c0"] == ["T"]
    rule = out["summary"]["blq"]["rule"]
    assert "No BLQ flag" in rule and "C(tau) = C(0)" in rule and "T" in rule


# ── strict (complete-interval) path without a BLQ column: same row validations as the BLQ path ────
def _strict(recs):
    from app.compute.dosing import extract_ss_intervals_checked

    return extract_ss_intervals_checked(recs, id_col="ID", time_col="TIME", dv_col="DV", amt_col="AMT", ii_col="II",
                                        addl_col="ADDL", require_complete_interval=True)[0]


def _nf_obs(sid, tad, dv):
    return {"ID": sid, "TIME": 36 + tad, "DV": dv, "AMT": 0, "II": None, "ADDL": None}


def test_no_flag_a_final_dose_row_with_dv_zero_is_not_a_concentration():
    doses = [{"ID": "P", "TIME": t, "DV": 0.0 if t == 36 else None, "AMT": 100, "II": 12, "ADDL": 0}
             for t in (0, 12, 24, 36)]
    blank = [{**d, "DV": None} for d in doses]
    obs = [_nf_obs("P", t, c) for t, c in SS_PROFILE if t != 0.0]
    with_zero, plain = _strict(doses + obs)["P"], _strict(blank + obs)["P"]
    assert with_zero == plain
    assert with_zero.n_doses == 4 and with_zero.c0_source == "ctau_assumed" and 0.0 not in with_zero.conc


@pytest.mark.parametrize("bad", [float("inf"), float("-inf")])
def test_no_flag_an_infinite_dv_never_enters_the_profile(bad):
    dose = [{"ID": "I", "TIME": 0, "DV": None, "AMT": 100, "II": 12, "ADDL": 3}]
    obs = [_nf_obs("I", t, c) for t, c in SS_PROFILE]
    with_inf = _strict(dose + obs + [_nf_obs("I", 0.5, bad)])["I"]
    reference = _strict(dose + obs)["I"]
    assert with_inf == reference and all(math.isfinite(c) for c in with_inf.conc)


@pytest.mark.parametrize("start", [0.0005, 0.001])                 # PRE_DOSE_TOL = 1e-3
def test_no_flag_a_start_within_the_zero_tolerance_is_placed_exactly_at_zero(start):
    dose = [{"ID": "Z", "TIME": 0, "DV": None, "AMT": 100, "II": 12, "ADDL": 3}]
    obs = [_nf_obs("Z", start if t == 0.0 else t, c) for t, c in SS_PROFILE]
    prof = _strict(dose + obs)["Z"]
    assert prof.tad[0] == 0.0 and prof.tad[-1] == 12.0 and prof.c0_source == "measured"
    assert prof == _strict(dose + [_nf_obs("Z", t, c) for t, c in SS_PROFILE])["Z"]


# ── correction cycle on d823217: closure, endpoints, tau-sample choice, dose-time anchor, flag codings ──
TAU = 12.0


def _css(t: float, a: float, ka: float, ke: float) -> float:
    """Exact one-compartment oral steady-state concentration at time-after-dose t (q12h)."""
    return a * (math.exp(-ke * t) / (1 - math.exp(-ke * TAU)) - math.exp(-ka * t) / (1 - math.exp(-ka * TAU)))


# reviewer case 1.1: t1/2 0.5 h, LLOQ 0.5. At steady state AUC_tau equals the single-dose AUC(0-inf) = a (1/ke - 1/ka).
SHORT = {"a": 28.87248, "ka": 2.99949, "ke": math.log(2) / 0.5}
SHORT_AUC_TAU = SHORT["a"] * (1 / SHORT["ke"] - 1 / SHORT["ka"])           # 11.20
SHORT_ROWS = [(0.0, 0.0, 1), (0.25, 6.776, 0), (0.5, 7.992, 0), (1.0, 5.78, 0), (2.0, 1.733, 0),
              (4.0, 0.0, 1), (8.0, 0.0, 1), (12.0, 0.0, 1)]                   # (tad, dv, blq) exactly as reported

# reviewer case 1.2: dose 100, V 10, ka 1.5, ke 0.15 -> AUC_tau = Dose/CL = 66.67 and C(0) = C(12) = 2.2004
LONG = {"a": 100 / 9, "ka": 1.5, "ke": 0.15}
LONG_AUC_TAU = 100 / 1.5


def test_the_short_half_life_case_is_the_analytic_profile():
    for tad, dv, blq in SHORT_ROWS:
        c = _css(tad, **SHORT)
        if blq:
            assert c < 0.5                                                   # flagged exactly where the model is below LLOQ
        else:
            assert c == pytest.approx(dv, abs=5e-4)
    assert SHORT_AUC_TAU == pytest.approx(11.20, abs=0.01)


def test_ss_trailing_blq_is_not_bridged_to_an_assumed_zero_c_tau(run_ss):
    """d823217 dropped the 4/8/12 h BLQ rows, closed the interval with C(tau) = C(0) = 0 and drew one straight line from
    (2 h, 1.733): AUC_tau 18.13 against the analytic 11.20 (+62%), CL/F -38%. A decline to an assumed zero cannot be
    integrated by log-down, so the subject must stop the run with that reason instead of returning a number."""
    with pytest.raises(ValueError, match=r"steady-state") as err:
        run_ss([("A", t, dv, b) for t, dv, b in SHORT_ROWS])
    msg = str(err.value)
    assert "'A'" in msg and "log-down" in msg and "C(tau) = C(0)" in msg


def test_ss_assumed_c_tau_reached_by_log_down_matches_the_analytic_auc(run_ss):
    """Guard: where the assumed end IS integrable under the declared rule (decline to a positive C(0)), the closed
    interval matches the analytic AUC_tau (the 8->12 h segment is mono-exponential, so log-down is exact there)."""
    pairs = [(t, round(_css(t, **LONG), 6)) for t in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)]       # no sample at tau
    p = run_ss(_ss_rows("G", pairs), with_flag=False)["params"]["G"]
    assert p["ctau_source"] == "c0_assumed"
    assert p["AUC_tau"] == pytest.approx(LONG_AUC_TAU, rel=0.02)
    # the same declared rule as a measured C(tau): log-down 8->12 h gives 65.98, a straight line 66.34
    explicit = run_ss(_ss_rows("G", pairs + [(12.0, pairs[0][1])]), with_flag=False)["params"]["G"]
    assert p["AUC_tau"] == explicit["AUC_tau"] == pytest.approx(65.980, abs=1e-3)


def test_ss_one_sample_never_supplies_both_interval_ends(run_ss):
    """Samples at 0.5-11 h only: d823217 took the 11 h sample as C(0) (10%-of-tau window) and copied it to C(tau), so
    one observation became both ends (Cmin = Ctrough = 2.5565, true 2.2004). Neither end is supported: stop the run."""
    pairs = [(t, round(_css(t, **LONG), 6)) for t in (0.5, 1.0, 2.0, 4.0, 8.0, 11.0)]
    for with_flag in (True, False):
        with pytest.raises(ValueError, match=r"neither C\(0\) nor C\(tau\)") as err:
            run_ss(_ss_rows("E", pairs), with_flag=with_flag)
        assert "'E'" in str(err.value)


@pytest.mark.parametrize("with_flag", [True, False])
def test_the_rule_text_names_which_end_was_assumed(with_flag, run_ss):
    no_c0 = [p for p in SS_PROFILE if p[0] != 0.0]
    no_ctau = [p for p in SS_PROFILE if p[0] != 12.0]
    out = run_ss(_ss_rows("Z", no_c0) + _ss_rows("T", no_ctau) + _ss_rows("M", SS_PROFILE), with_flag=with_flag)
    blq = out["summary"]["blq"]
    assert blq["c0_assumed_ctau"] == ["Z"] and blq["ctau_assumed_c0"] == ["T"]
    assert (out["params"]["Z"]["c0_source"], out["params"]["Z"]["ctau_source"]) == ("ctau_assumed", "measured")
    assert "C(0) = C(tau) was assumed" in blq["rule"] and "C(tau) = C(0) was assumed" in blq["rule"]


NEAR_TAU = [(0.0, 2.0), (1.0, 6.0), (2.0, 8.0), (4.0, 5.0), (8.0, 3.0)]


def test_the_tau_sample_is_the_one_closest_to_tau_whatever_the_row_order(run_ss):
    first = run_ss(_ss_rows("O", NEAR_TAU + [(11.8, 2.6), (12.0, 2.0)]))["params"]["O"]
    second = run_ss(_ss_rows("O", NEAR_TAU + [(12.0, 2.0), (11.8, 2.6)]))["params"]["O"]
    assert first == second
    # 12.0 h is C(tau); the 11.8 h sample stays an interior sample at its own time (main's 50.506 in either order)
    assert first["AUC_tau"] == pytest.approx(50.506, abs=1e-3) and first["Cmin"] == 2.0


def test_equidistant_tau_samples_resolve_to_the_earlier_one(run_ss):
    first = run_ss(_ss_rows("Q", NEAR_TAU + [(11.9, 2.3), (12.1, 1.9)]))["params"]["Q"]
    second = run_ss(_ss_rows("Q", NEAR_TAU + [(12.1, 1.9), (11.9, 2.3)]))["params"]["Q"]
    assert first == second
    at_tau = run_ss(_ss_rows("Q", NEAR_TAU + [(12.0, 2.3)]))["params"]["Q"]  # 11.9 h placed at tau, 12.1 h is past tau
    assert first["AUC_tau"] == at_tau["AUC_tau"]


@pytest.mark.parametrize("tad,dv", [(12.0, 2.6), (4.0, 5.5), (0.0, 2.4)])
def test_conflicting_duplicate_concentrations_fail_closed(tad, dv, run_ss):
    with pytest.raises(ValueError, match=r"conflicting") as err:
        run_ss(_ss_rows("D", SS_PROFILE + [(tad, dv)]))
    assert "'D'" in str(err.value)


def test_identical_duplicate_rows_are_collapsed(run_ss):
    dup = run_ss(_ss_rows("D", SS_PROFILE + [(12.0, 2.0), (4.0, 5.0)]))["params"]["D"]
    assert dup == run_ss(_ss_rows("D", SS_PROFILE))["params"]["D"]


REST = [(1.0, 5.0), (2.0, 8.0), (4.0, 4.0), (8.0, 2.0), (12.0, 1.0)]


def test_a_flagged_pre_dose_sample_adds_no_area_before_the_dose(run):
    """Dose at t = 0, flagged blank sample at t = -2 h, none at 0: d823217 integrated from -2 h (42.854). The area
    starts at the dose time with C(0) = 0, the same as a flagged blank at t = 0 (37.854)."""
    pre = run([("P", -2.0, None, 1)] + [("P", t, c, 0) for t, c in REST])["P"]
    at_dose = run([("P", 0.0, None, 1)] + [("P", t, c, 0) for t, c in REST])["P"]
    assert pre["AUC_last"] == at_dose["AUC_last"] == pytest.approx(37.854, abs=1e-3)


def test_an_unflagged_pre_dose_zero_adds_no_area_before_the_dose(run):
    pre = run([("P", -2.0, 0.0, 0)] + [("P", t, c, 0) for t, c in REST], with_flag=False)["P"]
    at_dose = run([("P", 0.0, 0.0, 0)] + [("P", t, c, 0) for t, c in REST], with_flag=False)["P"]
    assert pre["AUC_last"] == at_dose["AUC_last"] == pytest.approx(37.854, abs=1e-3)


@pytest.mark.parametrize("with_flag", [True, False])
def test_a_positive_pre_dose_value_is_not_integrated(with_flag, run):
    with_pre = run([("P", -1.0, 0.3, 0)] + [("P", t, c, 0) for t, c in REST], with_flag=with_flag)["P"]
    without = run([("P", t, c, 0) for t, c in REST], with_flag=with_flag)["P"]
    for k in ("AUC_last", "AUC_inf", "Cmax", "Tmax", "lambda_z"):
        assert with_pre[k] == without[k], k


# ── owner's rule: an unflagged 0 is a measured value, not a quantifiable positive concentration ──────
def test_an_unflagged_measured_zero_does_not_end_the_leading_blq_phase(run):
    flagged = run([("Z", 0.0, 0.0, 0), ("Z", 0.5, 0.0, 1)] + [("Z", t, c, 0) for t, c in REST])
    reference = run([("Z", 0.0, 0.0, 0), ("Z", 0.5, 0.0, 0)] + [("Z", t, c, 0) for t, c in REST], with_flag=False)
    assert flagged["Z"]["AUC_last"] == reference["Z"]["AUC_last"] == pytest.approx(36.604, abs=1e-3)


def test_steady_state_unflagged_zero_does_not_end_the_leading_blq_phase(run_ss):
    flagged = run_ss([("Z", 0.0, 0.0, 0), ("Z", 0.5, None, 1)] + _ss_rows("Z", SS_PROFILE[1:]))["params"]["Z"]
    reference = run_ss([("Z", 0.0, 0.0, 0), ("Z", 0.5, 0.0, 0)] + _ss_rows("Z", SS_PROFILE[1:]),
                       with_flag=False)["params"]["Z"]
    assert flagged["AUC_tau"] == reference["AUC_tau"]


def test_the_rule_text_states_the_measured_zero_convention(run, tmp_path):
    orch = Orchestrator(llm=MockLLM(), store=SessionStore(":memory:"))
    sid = orch.create_session().id
    path = tmp_path / "z.csv"
    path.write_text("ID,TIME,DV,AMT,EVID,BLQ\nA,0,,100,1,0\nA,1,5,0,0,0\nA,2,2,0,0,0\nA,4,1,0,0,0\nA,8,0.5,0,0,0\n")
    orch.run_tool(sid, "load_dataset", "data_manager", {"path": str(path)})
    rule = orch.run_tool(sid, "compute_nca", "nca", {})["result"]["nca_summary"]["blq"]["rule"]
    assert "unflagged 0" in rule and "does not end" in rule and "before the dose" in rule


# ── BLQ flag codings ──────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def run_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)

    def _run(lines: list[str]) -> dict:
        path = tmp_path / "flags.csv"
        path.write_text("\n".join(lines) + "\n")
        orch = Orchestrator(llm=MockLLM(), store=SessionStore(":memory:"))
        sid = orch.create_session().id
        orch.run_tool(sid, "load_dataset", "data_manager", {"path": str(path)})
        res = orch.run_tool(sid, "compute_nca", "nca", {})
        return {"params": {p["subject"]: p for p in orch.get_session(sid).state.nca_parameters},
                "summary": res["result"]["nca_summary"]}

    return _run


def _sd_lines(yes: str, no: str) -> list[str]:
    """Single dose; middle zero at 6 h and trailing zero at 36 h are flagged."""
    obs = [(1, 5), (2, 8), (4, 4), (6, 0), (8, 2), (12, 1), (24, 0.25), (36, 0)]
    return (["ID,TIME,DV,AMT,EVID,BLQ", f"A,0,,100,1,{no}"]
            + [f"A,{t},{c},0,0,{yes if t in (6, 36) else no}" for t, c in obs])


def _ss_lines(yes: str, no: str) -> list[str]:
    """Steady state q12h; the tau sample is flagged (closed with C(tau) = C(0) = 2)."""
    return (["ID,TIME,DV,AMT,EVID,II,ADDL,BLQ", f"S,0,,100,1,12,3,{no}"]
            + [f"S,{36 + t},{0 if t == 12.0 else c},0,0,,,{yes if t == 12.0 else no}" for t, c in SS_PROFILE])


CODINGS = [("Y", ""), ("Y", "N"), ("y", "n"), ("YES", "NO"), ("TRUE", "FALSE"), ("true", ""), ("1", "0"), ("1.0", "")]


@pytest.mark.parametrize("yes,no", CODINGS)
@pytest.mark.parametrize("lines", [_sd_lines, _ss_lines])
def test_supported_blq_flag_codings_mean_the_same_as_1_and_0(yes, no, lines, run_csv):
    coded, reference = run_csv(lines(yes, no)), run_csv(lines("1", "0"))
    assert coded["params"] == reference["params"]
    assert coded["summary"]["blq"]["n_below_loq"] == reference["summary"]["blq"]["n_below_loq"] > 0


@pytest.mark.parametrize("bad", ["2", "BLQ", "-1", "<LLOQ"])
@pytest.mark.parametrize("lines", [_sd_lines, _ss_lines])
def test_an_unknown_blq_flag_value_fails_validation(bad, lines, run_csv):
    with pytest.raises(ValueError, match=r"BLQ") as err:
        run_csv(lines(bad, "0"))
    assert bad in str(err.value)


# ── pinned guarantees (mutation-sensitive) ─────────────────────────────────────────────────────────
def test_m1_is_classified_per_subject(run):
    """Subject A's quantifiable samples must not turn subject B's leading BLQ rows into 'middle' rows."""
    a_rows = [("A", 0.25, 1.0, 0)] + [("A", t, c, 0) for t, c in REST]     # quantifiable before B's 0.5 h BLQ
    b_rows = [("B", 0.0, None, 1), ("B", 0.5, None, 1)] + [("B", t, c, 0) for t, c in REST]
    together = run(a_rows + b_rows)["B"]
    alone = run(b_rows)["B"]
    assert together["AUC_last"] == alone["AUC_last"] == pytest.approx(36.604, abs=1e-3)


def test_a_subject_with_fewer_than_3_usable_samples_stops_the_steady_state_run(run_ss):
    with pytest.raises(ValueError, match=r"fewer than 3 usable samples") as err:
        run_ss(_ss_rows("V", SS_PROFILE) + _ss_rows("F", [(0.0, 2.0), (12.0, 2.0)]))
    assert "'F'" in str(err.value) and "'V'" not in str(err.value)


# ── rule 4 and endpoint selection, pinned in detail ────────────────────────────────────────────────
@pytest.mark.parametrize("pre,anchored", [
    ([(-4.0, None, 1), (-1.0, 0.3, 0)], False),          # the closest pre-dose record is positive: no anchor
    ([(-4.0, 0.3, 0), (-1.0, None, 1)], True),           # the closest is BLQ-flagged: C(dose) = 0
    ([(-1.0, None, 1), (-1.0, 0.3, 0)], False),          # records at the closest time disagree: no anchor
    ([(-1.0, None, 1), (-1.0, 0.0, 0)], True),           # flagged and a measured 0 at the closest time: anchor
])
def test_only_the_closest_pre_dose_records_can_anchor_c0_at_the_dose(pre, anchored, run):
    out = run([("P", t, c, b) for t, c, b in pre] + [("P", t, c, 0) for t, c in REST])["P"]
    assert out["AUC_last"] == pytest.approx(37.854 if anchored else 35.354, abs=1e-3)


def test_the_dose_time_is_the_subjects_first_dose_record(run_csv):
    lines = (["ID,TIME,DV,AMT,EVID,BLQ", "P,10,,100,1,0", "P,8,,0,0,1"]
             + [f"P,{10 + t:g},{c},0,0,0" for t, c in REST])
    assert run_csv(lines)["params"]["P"]["AUC_last"] == pytest.approx(37.854, abs=1e-3)


def test_a_pre_dose_trough_within_the_tolerance_is_c0(run_ss):
    trough = run_ss(_ss_rows("R", [(-0.2, 1.8)] + SS_PROFILE[1:]))["params"]["R"]
    at_zero = run_ss(_ss_rows("R", [(0.0, 1.8)] + SS_PROFILE[1:]))["params"]["R"]
    assert trough == at_zero and trough["c0_source"] == "measured" and trough["Ctrough"] == 1.8


def test_a_pre_dose_sample_that_is_not_c0_is_not_integrated(run_ss):
    extra = run_ss(_ss_rows("X", [(-0.2, 9.0)] + SS_PROFILE))["params"]["X"]
    assert extra == run_ss(_ss_rows("X", SS_PROFILE))["params"]["X"]


def test_a_flagged_and_an_unflagged_record_at_one_time_conflict(run_ss):
    with pytest.raises(ValueError, match=r"conflicting") as err:
        run_ss(_ss_rows("D", SS_PROFILE) + [("D", 12.0, None, 1)])
    assert "BLQ" in str(err.value)


def test_endpoint_distances_within_1e_9_are_ties():
    from app.compute.dosing import _place_endpoints

    placed = _place_endpoints([(11.9, False, 2.3), (12.1 - 1e-12, False, 1.9)], 12.0, 0.24)
    assert placed == [(12.0, False, 2.3)]                  # the earlier sample wins a tie; the later one is past tau
