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
