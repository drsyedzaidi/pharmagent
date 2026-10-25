"""Statistics / BLQ review findings F-04, F-05, F-15, F-16, F-18 (external review of main fd119a9)."""
from __future__ import annotations

import math

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
def _be_csv(cens_dv: float) -> str:
    lines = ["ID,TIME,DV,AMT,EVID,TRT,CENS"]
    scale = {1: 1.0, 2: 1.2, 3: 0.9, 4: 1.1}
    for sid in (1, 2, 3, 4):
        for trt, conc in (("T", CONC_T), ("R", CONC_R)):
            lines.append(f"{sid},0,0,100,1,{trt},0")
            for t, c in zip(TIMES, conc, strict=True):
                lines.append(f"{sid},{t},{c * scale[sid] * (1.05 if trt == 'T' else 1.0)},0,0,{trt},0")
            for t in (24, 48):
                lines.append(f"{sid},{t},{cens_dv},0,0,{trt},1")
    return "\n".join(lines) + "\n"


def test_be_censored_rows_are_not_integrated_as_concentrations(orch):
    a = orch.run_tool(_load(orch, _be_csv(0.5), "a.csv"), "run_bioequivalence", "be", {})["result"]
    b = orch.run_tool(_load(orch, _be_csv(0.0), "b.csv"), "run_bioequivalence", "be", {})["result"]
    for p in ("Cmax", "AUC_last", "AUC_inf"):
        assert a["parameters"][p] == b["parameters"][p], p
    # the censored tail must equal the dataset with those rows absent
    no_tail = "\n".join(ln for ln in _be_csv(0.5).splitlines() if not ln.endswith(",1")) + "\n"
    c = orch.run_tool(_load(orch, no_tail, "c.csv"), "run_bioequivalence", "be", {})["result"]
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
    lines = ["ID,TIME,DV,AMT,EVID,TRT"]
    for sid in (1, 2, 3, 4, 5):
        for trt in ("T", "R"):
            conc = POOR if (sid == 5 and trt == "T") else GOOD
            lines.append(f"{sid},0,0,100,1,{trt}")
            lines += [f"{sid},{t},{c * (1.1 if trt == 'T' else 1.0) * (1 + 0.05 * sid)},0,0,{trt}"
                      for t, c in zip(LZ_T, conc, strict=True) if t > 0]
    s = _load(orch, "\n".join(lines) + "\n")
    res = orch.run_tool(s, "run_bioequivalence", "be", {})["result"]
    assert res["lambda_z_excluded"]["subjects"] == {"T": [5], "R": []}
    assert res["lambda_z_excluded"]["include_unacceptable"] is False
    assert res["parameters"]["AUC_inf"]["n_test"] == 4 and res["parameters"]["AUC_last"]["n_test"] == 5
    res2 = orch.run_tool(s, "run_bioequivalence", "be", {"include_unacceptable_lambda_z": True})["result"]
    assert res2["parameters"]["AUC_inf"]["n_test"] == 5
