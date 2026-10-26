"""Statistics / BLQ review findings F-04, F-05, F-15, F-16, F-18 (external review of main fd119a9)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from app.config import settings
from app.core.llm import MockLLM
from app.core.orchestrator import Orchestrator
from app.core.store import SessionStore

TIMES = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0]
CONC_T = [4.0, 8.0, 6.0, 3.0, 1.5, 0.75]
CONC_R = [3.0, 7.0, 6.5, 3.5, 1.7, 0.85]


@pytest.fixture
def orch(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    o = Orchestrator(llm=MockLLM(), store=SessionStore(":memory:"))
    o._tmp = tmp_path  # type: ignore[attr-defined]
    return o


def _load(orch, text: str, name: str = "d.csv"):
    sid = orch.create_session().id
    path = orch._tmp / name
    path.write_text(text)
    orch.run_tool(sid, "load_dataset", "data_manager", {"path": str(path)})
    return sid


# ── F-05: BE / compartmental honour CENS (M1), like compute_nca ──────────────────────────────────────────
# PR #22's safety contract: design, profile, test and reference (and PERIOD for a crossover) are explicit.
BE_X = {"design": "crossover", "profile": "fda_2026", "test": "T", "reference": "R", "period_col": "PERIOD"}
BE_PARALLEL = {"design": "parallel", "profile": "fda_2026", "test": "T", "reference": "R"}


def _be_csv(cens_dv: float) -> str:
    lines = ["ID,TIME,DV,AMT,EVID,TRT,PERIOD,SEQUENCE,CENS"]
    scale = {1: 1.0, 2: 1.2, 3: 0.9, 4: 1.1}
    for sid in (1, 2, 3, 4):
        seq = "TR" if sid <= 2 else "RT"
        for trt, conc in (("T", CONC_T), ("R", CONC_R)):
            per = seq.index(trt) + 1
            d = f"{trt},{per},{seq}"
            lines.append(f"{sid},0,0,100,1,{d},0")
            for t, c in zip(TIMES, conc, strict=True):
                lines.append(f"{sid},{t},{c * scale[sid] * (1.05 if trt == 'T' else 1.0)},0,0,{d},0")
            for t in (24, 48):
                lines.append(f"{sid},{t},{cens_dv},0,0,{d},1")
    return "\n".join(lines) + "\n"


def test_be_censored_rows_are_not_integrated_as_concentrations(orch):
    a = orch.run_tool(_load(orch, _be_csv(0.5), "a.csv"), "run_bioequivalence", "be", BE_X)["result"]
    b = orch.run_tool(_load(orch, _be_csv(0.0), "b.csv"), "run_bioequivalence", "be", BE_X)["result"]
    for p in ("Cmax", "AUC_last", "AUC_inf"):
        assert a["parameters"][p] == b["parameters"][p], p
    # the censored tail must equal the dataset with those rows absent
    no_tail = "\n".join(ln for ln in _be_csv(0.5).splitlines() if not ln.endswith(",1")) + "\n"
    c = orch.run_tool(_load(orch, no_tail, "c.csv"), "run_bioequivalence", "be", BE_X)["result"]
    assert a["parameters"]["AUC_inf"]["gmr_pct"] == c["parameters"]["AUC_inf"]["gmr_pct"]
    assert a["parameters"]["AUC_last"]["gmr_pct"] == c["parameters"]["AUC_last"]["gmr_pct"]


def test_compartmental_censored_rows_are_not_fitted_as_concentrations(orch):
    def csv(cens_dv: float) -> str:
        lines = ["ID,TIME,DV,AMT,EVID,CENS"]
        for sid in (1, 2):
            lines.append(f"{sid},0,0,100,1,0")
            for t in (0.5, 1, 2, 4, 6, 8, 12):
                lines.append(f"{sid},{t},{10 * (math.exp(-0.2 * t) - math.exp(-1.5 * t)) * (1 + 0.1 * sid)},0,0,0")
            for t in (24, 48):
                lines.append(f"{sid},{t},{cens_dv},0,0,1")
        return "\n".join(lines) + "\n"

    ra = orch.run_tool(_load(orch, csv(0.5), "a.csv"), "fit_compartmental", "compartmental", {"models": ["1cmt"]})
    rb = orch.run_tool(_load(orch, csv(0.0), "b.csv"), "fit_compartmental", "compartmental", {"models": ["1cmt"]})
    assert [f["params"] for f in ra["result"]["fits"]] == [f["params"] for f in rb["result"]["fits"]]


# ── F-15: IV infusion MRT/Vss need the infusion-duration correction ─────────────────────────────────────
def _infusion_profile(dose=100.0, cl=10.0, v=20.0, t_inf=2.0):
    import numpy as np
    k = cl / v
    rate = dose / t_inf
    # dense sampling: the linear trapezoid is the only approximation left, so keep its error well under the 0.5 bound
    t = np.round(np.concatenate([np.arange(0.1, 6.0, 0.1), np.arange(6.0, 12.5, 0.5), [14, 16, 20, 24, 30, 36]]), 3)
    during = rate / cl * (1 - np.exp(-k * np.minimum(t, t_inf)))
    c = np.where(t <= t_inf, during, rate / cl * (1 - math.exp(-k * t_inf)) * np.exp(-k * (t - t_inf)))
    return t, c


def test_vss_of_an_iv_infusion_is_corrected_for_infusion_time():
    from app.compute.nca import Profile, nca_subject
    t, c = _infusion_profile()
    out = nca_subject(Profile("S", t, c, 100.0, t_inf=2.0), is_iv=True)
    assert abs(out["Vss"] - 20.0) < 0.5, out["Vss"]
    assert abs(out["MRT"] - 2.0) < 0.1


def test_iv_bolus_default_is_unchanged_by_t_inf():
    from app.compute.nca import Profile, nca_subject
    t, c = _infusion_profile()
    a = nca_subject(Profile("S", t, c, 100.0), is_iv=True)
    b = nca_subject(Profile("S", t, c, 100.0, t_inf=0.0), is_iv=True)
    assert a["MRT"] == b["MRT"] and a["Vss"] == b["Vss"]


def test_compute_nca_derives_infusion_time_from_rate(orch):
    t, c = _infusion_profile()
    lines = ["ID,TIME,DV,AMT,EVID,RATE,ROUTE", "1,0,0,100,1,50,IV"]
    lines += [f"1,{ti},{ci},0,0,0,IV" for ti, ci in zip(t, c, strict=True)]
    sid = _load(orch, "\n".join(lines) + "\n")
    orch.run_tool(sid, "compute_nca", "nca", {})
    params = orch.get_session(sid).state.nca_parameters
    assert abs(params[0]["Vss"] - 20.0) < 0.5, params[0]["Vss"]


# ── F-16: poor terminal fits must not pool into AUC_inf-derived summaries ──────────────────────────────
POOR = [0, 10, 7, 9, 5, 6]          # adj R^2 ~0.04 on 4 points, ~69 % extrapolated: accepted by _best_lambda_z (> 0)
GOOD = [0, 10, 6, 3.6, 1.3, 0.47]   # clean mono-exponential tail
LZ_T = [0, 1, 2, 4, 8, 12]


def _profile(conc):
    import numpy as np

    from app.compute.nca import Profile, nca_subject
    return nca_subject(Profile("S", np.array(LZ_T, float), np.array(conc, float), 100.0))


def test_lambda_z_acceptable_flags_a_low_r2_fit_and_passes_a_clean_one():
    poor, good = _profile(POOR), _profile(GOOD)
    assert poor["lambda_z_r2_adj"] < 0.8 and poor["AUC_inf"] is not None   # the fit IS accepted by the picker
    assert poor["lambda_z_acceptable"] is False
    assert good["lambda_z_acceptable"] is True
    assert _profile([0, 10, 5, 0, 0, 0])["lambda_z_acceptable"] is None    # no terminal fit at all: not evaluated


def _rows():
    return [{**_profile(c), "subject": i, "dose": 100} for i, c in enumerate([GOOD, GOOD, POOR, GOOD], 1)]


def test_summarize_by_dose_excludes_unacceptable_fits_and_says_so():
    from app.compute.nca import summarize_by_dose
    rows = _rows()
    s = summarize_by_dose(rows)
    entry = s["by_dose"][0]
    assert entry["n"] == 4 and entry["n_lambda_z_excluded"] == 1
    assert s["lambda_z_excluded"]["subjects"] == [3] and s["lambda_z_excluded"]["include_unacceptable"] is False
    inf = next(p for p in s["descriptive"][0]["parameters"] if p["parameter"] == "AUC_inf")
    assert inf["n"] == 3
    # Cmax / AUC_last do not depend on lambda_z: all four stay
    assert next(p for p in s["descriptive"][0]["parameters"] if p["parameter"] == "Cmax")["n"] == 4
    kept = summarize_by_dose(rows, include_unacceptable_lambda_z=True)
    assert kept["by_dose"][0]["n_lambda_z_excluded"] == 0
    assert next(p for p in kept["descriptive"][0]["parameters"] if p["parameter"] == "AUC_inf")["n"] == 4
    assert rows[2]["AUC_inf"] is not None            # the stored per-subject result is untouched


def test_dose_proportionality_excludes_unacceptable_fits(orch):
    sid = orch.create_session().id
    rows = []
    for dose, scale in ((50, 0.5), (100, 1.0), (200, 2.0)):
        for i, c in enumerate([GOOD, GOOD, POOR]):
            r = _profile([x * scale for x in c])
            rows.append({**r, "subject": f"{dose}-{i}", "dose": dose})
    orch.get_session(sid).state.nca_parameters = rows
    res = orch.run_tool(sid, "run_dose_proportionality", "dose_prop", {})["result"]
    assert res["lambda_z_excluded"]["n_subjects"] == 3 and res["lambda_z_excluded"]["include_unacceptable"] is False
    res_all = orch.run_tool(sid, "run_dose_proportionality", "dose_prop",
                            {"include_unacceptable_lambda_z": True})["result"]
    assert res_all["lambda_z_excluded"]["n_subjects"] == 0


def test_bioequivalence_excludes_unacceptable_auc_inf_and_reports_it(orch):
    lines = ["ID,TIME,DV,AMT,EVID,TRT,PERIOD,SEQUENCE"]
    for sid in (1, 2, 3, 4, 5):
        seq = "TR" if sid % 2 else "RT"
        for trt in ("T", "R"):
            conc = POOR if (sid == 5 and trt == "T") else GOOD
            d = f"{trt},{seq.index(trt) + 1},{seq}"
            lines.append(f"{sid},0,0,100,1,{d}")
            lines += [f"{sid},{t},{c * (1.1 if trt == 'T' else 1.0) * (1 + 0.05 * sid)},0,0,{d}"
                      for t, c in zip(LZ_T, conc, strict=True) if t > 0]
    s = _load(orch, "\n".join(lines) + "\n")
    res = orch.run_tool(s, "run_bioequivalence", "be", BE_X)["result"]
    assert res["lambda_z_excluded"]["subjects"] == {"T": [5], "R": []}
    assert res["lambda_z_excluded"]["include_unacceptable"] is False
    assert res["parameters"]["AUC_inf"]["n_test"] == 4 and res["parameters"]["AUC_last"]["n_test"] == 5
    res2 = orch.run_tool(s, "run_bioequivalence", "be", {**BE_X, "include_unacceptable_lambda_z": True})["result"]
    assert res2["parameters"]["AUC_inf"]["n_test"] == 5


# ── F-04: 2x2 crossover model, no silent downgrade to parallel on dropout ────────────────────────────────
def _crossover(seq_sizes: dict[str, int], theta=math.log(1.10), period2=math.log(0.8), noise=None):
    """Noise-free log-scale 2x2 data: per-subject effect + treatment effect + a constant period-2 shift."""
    rows = []
    n = 0
    for seq, size in seq_sizes.items():
        for _ in range(size):
            n += 1
            subj = 0.3 * math.sin(n * 1.7) + 4.0 + (0.0 if noise is None else noise[n % len(noise)])
            order = ("T", "R") if seq == "TR" else ("R", "T")
            for period, trt in enumerate(order, 1):
                y = subj + (theta if trt == "T" else 0.0) + (period2 if period == 2 else 0.0)
                rows.append({"subject": n, "sequence": seq, "period": period, "trt": trt, "y": y})
    return rows


def _split(rows):
    subj = sorted({r["subject"] for r in rows})
    by = {(r["subject"], r["trt"]): r for r in rows}
    return ([math.exp(by[s, "T"]["y"]) for s in subj], [math.exp(by[s, "R"]["y"]) for s in subj],
            [by[s, "T"]["sequence"] for s in subj], [by[s, "T"]["period"] for s in subj])


def test_balanced_2x2_with_a_constant_period_shift_has_zero_intra_cv_and_df_n1_plus_n2_minus_2():
    from app.compute.bioequivalence import be_one_parameter
    t, r, seq, per = _split(_crossover({"TR": 6, "RT": 6}))
    res = be_one_parameter(t, r, paired=True, sequences=seq, test_periods=per)
    assert res["cv_intra_pct"] == 0.0 and res["df"] == 6 + 6 - 2
    assert res["gmr_pct"] == pytest.approx(110.0, abs=1e-6)
    assert res["model"] == "2x2 crossover: sequence, subject(sequence), period, treatment"


def test_a_period_effect_biases_the_plain_paired_analysis_but_not_the_2x2_model():
    from app.compute.bioequivalence import be_one_parameter
    t, r, seq, per = _split(_crossover({"TR": 3, "RT": 7}))
    adj = be_one_parameter(t, r, paired=True, sequences=seq, test_periods=per)
    plain = be_one_parameter(t, r, paired=True)
    assert adj["gmr_pct"] == pytest.approx(110.0, abs=1e-6)
    assert abs(plain["gmr_pct"] - 110.0) > 1.0                       # unbalanced sequences + period shift
    assert adj["df"] == 3 + 7 - 2


def test_2x2_matches_the_closed_form_sequence_difference_estimator_with_noise():
    from app.compute.bioequivalence import be_one_parameter
    rng = np.random.default_rng(3)
    rows = _crossover({"TR": 5, "RT": 6}, noise=None)
    for rw in rows:
        rw["y"] += rng.normal(0, 0.05)
    t, r, seq, per = _split(rows)
    res = be_one_parameter(t, r, paired=True, sequences=seq, test_periods=per)
    # classical: d = (y2 - y1)/2 per subject; F = mean(d | TR... ) difference of sequence means
    by = {(x["subject"], x["period"]): x["y"] for x in rows}
    seqs = {x["subject"]: x["sequence"] for x in rows}
    d = {s: (by[s, 2] - by[s, 1]) / 2 for s in seqs}
    d_tr = [d[s] for s in seqs if seqs[s] == "TR"]            # period 2 minus period 1 = R - T (+ period)
    d_rt = [d[s] for s in seqs if seqs[s] == "RT"]
    theta_hat = np.mean(d_rt) - np.mean(d_tr)
    assert math.log(res["gmr_pct"] / 100) == pytest.approx(theta_hat, abs=1e-4)


def _be_tool_csv(drop_subject_period: tuple[int, str] | None = None, with_sequence=True) -> str:
    head = "ID,TIME,DV,AMT,EVID,TRT,PERIOD" + (",SEQUENCE" if with_sequence else "")
    lines = [head]
    n = 0
    for seq, size in (("TR", 4), ("RT", 4)):
        for _ in range(size):
            n += 1
            order = ("T", "R") if seq == "TR" else ("R", "T")
            for period, trt in enumerate(order, 1):
                if drop_subject_period == (n, trt):
                    continue
                f = (1.1 if trt == "T" else 1.0) * (0.8 if period == 2 else 1.0) * (1 + 0.1 * n)
                extra = f",{period},{seq}" if with_sequence else f",{period}"
                lines.append(f"{n},0,0,100,1,{trt}{extra}")
                lines += [f"{n},{t},{c * f},0,0,{trt}{extra}" for t, c in zip(TIMES, CONC_T, strict=True)]
    return "\n".join(lines) + "\n"


def test_tool_fits_the_2x2_model_when_period_and_sequence_columns_exist(orch):
    res = orch.run_tool(_load(orch, _be_tool_csv()), "run_bioequivalence", "be", BE_X)["result"]
    cmax = res["parameters"]["Cmax"]
    assert "2x2 crossover" in res["model"] and cmax["df"] == 4 + 4 - 2
    assert cmax["gmr_pct"] == pytest.approx(110.0, abs=1e-3)       # profile shapes are identical: only the factor
    assert res["excluded_subjects"]["n"] == 0


def test_a_dropout_does_not_switch_the_design_to_parallel(orch):
    res = orch.run_tool(_load(orch, _be_tool_csv(drop_subject_period=(3, "R"))), "run_bioequivalence", "be",
                        BE_X)["result"]
    assert res["design"] == "crossover" and "2x2 crossover" in res["model"]
    assert res["excluded_subjects"]["n"] == 1 and res["excluded_subjects"]["subject_ids"] == [3]
    assert res["n_completers"] == 7
    assert res["parameters"]["Cmax"]["df"] == 4 + 3 - 2            # n1 + n2 - 2 on the completers (3 TR, 4 RT)


def test_without_design_columns_the_paired_analysis_is_labelled_and_warned(orch):
    res = orch.run_tool(_load(orch, _be_tool_csv(drop_subject_period=(3, "R"), with_sequence=False)),
                        "run_bioequivalence", "be", BE_X)["result"]
    assert "no period or sequence terms" in res["model"]
    assert res["design"] == "crossover" and "paired log differences (crossover)" in res["model"]
    assert res["excluded_subjects"]["subject_ids"] == [3] and res["n_completers"] == 7
    assert any("period" in w and "sequence" in w for w in res["warnings"])


def test_disjoint_subject_groups_are_still_analysed_as_parallel(orch):
    lines = ["ID,TIME,DV,AMT,EVID,TRT"]
    for sid in range(1, 9):
        trt = "T" if sid <= 4 else "R"
        lines.append(f"{sid},0,0,100,1,{trt}")
        lines += [f"{sid},{t},{c * (1 + 0.1 * sid)},0,0,{trt}" for t, c in zip(TIMES, CONC_T, strict=True)]
    res = orch.run_tool(_load(orch, "\n".join(lines) + "\n"), "run_bioequivalence", "be", BE_PARALLEL)["result"]
    assert res["design"] == "parallel"


# ── F-18: ADPC exports observation rows only, carries the censoring flag, units from metadata ───────────
def _adpc(df, roles, meta=None):
    from app.core.cdisc import build_adpc
    from app.core.pharmstate import PharmState
    return build_adpc(df, roles, PharmState(dataset_metadata={"dataset_id": "S", **(meta or {})}))


def test_adpc_skips_dosing_rows():
    import pandas as pd
    df = pd.DataFrame({"ID": [1, 1], "TIME": [0.0, 1.0], "DV": [0.0, 25.0], "AMT": [100, 0], "EVID": [1, 0]})
    rows, _ = _adpc(df, {"ID": "ID", "TIME": "TIME", "DV": "DV", "AMT": "AMT", "EVID": "EVID"})
    assert len(rows) == 1 and rows[0]["AVAL"] == 25.0 and rows[0]["ATPTN"] == 1.0


def test_adpc_carries_the_censoring_flag_and_does_not_report_the_lloq_as_a_concentration():
    import pandas as pd
    df = pd.DataFrame({"ID": [1] * 5, "TIME": [0.0, 1.0, 2.0, 4.0, 8.0], "DV": [0.0, 5.0, 3.0, 0.5, 0.5],
                       "AMT": [0] * 5, "EVID": [0] * 5, "CENS": [1, 0, 0, 1, 1]})
    roles = {"ID": "ID", "TIME": "TIME", "DV": "DV", "AMT": "AMT", "EVID": "EVID", "CENS": "CENS"}
    rows, cols = _adpc(df, roles)
    assert "BLQFL" in cols and "ANL01FL" in cols
    assert [r["BLQFL"] for r in rows] == ["Y", "", "", "Y", "Y"]
    assert [r["AVAL"] for r in rows if r["BLQFL"] == "Y"] == [None, None, None]
    assert [r["LLOQ"] for r in rows if r["BLQFL"] == "Y"][1:] == [0.5, 0.5]
    # M1: the leading BLQ is a zero baseline (analysed), later BLQ rows are not
    assert [r["ANL01FL"] for r in rows] == ["Y", "Y", "Y", "N", "N"]


def test_adpc_and_adpp_units_come_from_dataset_metadata_else_stay_blank():
    import pandas as pd

    from app.core.cdisc import build_adpp
    from app.core.pharmstate import PharmState
    df = pd.DataFrame({"ID": [1], "TIME": [1.0], "DV": [5.0]})
    roles = {"ID": "ID", "TIME": "TIME", "DV": "DV"}
    blank, _ = _adpc(df, roles)
    assert blank[0]["AVALU"] == ""
    rows, _ = _adpc(df, roles, {"units": {"conc": "ng/mL", "time": "h", "dose": "mg"}})
    assert rows[0]["AVALU"] == "ng/mL"
    nca = [{"subject": 1, "dose": 100, "Cmax": 10.0, "Tmax": 1.0, "AUC_last": 40.0, "t_half": 7.0, "CL_F": 2.0}]
    pp = build_adpp(PharmState(nca_parameters=nca, dataset_metadata={
        "dataset_id": "S", "units": {"conc": "ng/mL", "time": "h", "dose": "mg"}}))[0]
    units = {r["PARAMCD"]: r["PPSTRESU"] for r in pp}
    assert units == {"CMAX": "ng/mL", "TMAX": "h", "AUCLST": "h*ng/mL", "LAMZHL": "h", "CLFO": "mg/(h*ng/mL)"}
    bare = build_adpp(PharmState(nca_parameters=nca, dataset_metadata={"dataset_id": "S"}))[0]
    assert {r["PPSTRESU"] for r in bare} == {""}


# ── PR #22 safety contract x 2x2 crossover model (merge of main into this branch) ───────────────────────
def _many_subject_csv(n_per_seq: int, drop: tuple[int, str] | None = None) -> str:
    """n_per_seq subjects in each of TR and RT, shared profile shape, design columns present."""
    lines = ["ID,TIME,DV,AMT,EVID,TRT,PERIOD,SEQUENCE"]
    n = 0
    for seq in ("TR", "RT"):
        for _ in range(n_per_seq):
            n += 1
            for period, trt in enumerate(seq, 1):
                if drop == (n, trt):
                    continue
                f = (1.0 if trt == "T" else 1.0) * (1 + 0.03 * math.sin(n * 2.3 + (0.2 if trt == "T" else 0.0)))
                lines.append(f"{n},0,0,100,1,{trt},{period},{seq}")
                lines += [f"{n},{t},{c * f},0,0,{trt},{period},{seq}" for t, c in zip(TIMES, CONC_T, strict=True)]
    return "\n".join(lines) + "\n"


def test_contract_refusals_still_apply_on_the_2x2_path(orch):
    # missing required arguments: nothing is inferred, even though PERIOD and SEQUENCE are present
    s = _load(orch, _many_subject_csv(6))
    res = orch.run_tool(s, "run_bioequivalence", "be", {"design": "crossover", "period_col": "PERIOD"})["result"]
    assert res["status"] == "missing_required_args" and "profile" in res["missing"]
    # a replicate (same treatment in two periods) is refused, not fitted
    lines = _many_subject_csv(6).splitlines()
    last_r = max(i for i, ln in enumerate(lines) if ln.startswith("1,") and ",R,2,TR" in ln)
    lines[last_r] = lines[last_r].replace(",R,2,TR", ",R,3,TR")          # subject 1 gets R in two periods
    bad = "\n".join(lines) + "\n"
    res = orch.run_tool(_load(orch, bad, "rep.csv"), "run_bioequivalence", "be", BE_X)["result"]
    assert res["status"] == "unsupported_design"
    # the evaluable-subject floor still blocks a pass: 12 per sequence is required by the profile, 6 is not enough
    res = orch.run_tool(s, "run_bioequivalence", "be", {**BE_X, "profile": "ich_m13a"})["result"]
    assert res["status"] == "ok" and "2x2 crossover" in res["model"]
    assert res["parameters"]["Cmax"]["n_evaluable"] == 12
    res = orch.run_tool(_load(orch, _many_subject_csv(3), "small.csv"), "run_bioequivalence", "be", BE_X)["result"]
    assert res["parameters"]["Cmax"]["acceptance"] == "insufficient_evaluable_subjects"
    assert res["bioequivalent"] is False


def test_valid_2x2_crossover_is_analysed_with_excluded_subjects_reported(orch):
    s = _load(orch, _many_subject_csv(7, drop=(2, "R")), "dropout.csv")
    res = orch.run_tool(s, "run_bioequivalence", "be", {**BE_X, "profile": "ich_m13a"})["result"]
    assert res["status"] == "ok" and "2x2 crossover" in res["model"] and res["design"] == "crossover"
    cmax = res["parameters"]["Cmax"]
    assert cmax["n_evaluable"] == 13 and cmax["df"] == 13 * 2 - (13 + 1 + 1)      # N_obs - rank(subject, period, trt)
    assert res["excluded_subjects"] == {**res["excluded_subjects"], "n": 1, "subject_ids": [2]}
    assert any(e["subject"] == 2 and e["reason"] == "no_evaluable_counterpart" for e in res["exclusions"])
    assert res["n_completers"] == 13
