"""Monolix adapter: model generation, result mapping, process handling. No
Monolix is run — ``subprocess.run`` is faked to write the files the R script
would, so this passes on any machine and never needs a GUI session."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

from app.engines import monolix as mx
from app.engines.base import CandidateSpec, EngineResult
from app.engines.monolix import (
    MonolixAdapter,
    header_types,
    mlxtran_model,
    omega_sd_to_cv_pct,
    parse_results,
)


def _subjects(n=6):
    import random
    rng = random.Random(1)
    subs = []
    for i in range(n):
        cl, v, ka = 4.0 * math.exp(rng.gauss(0, 0.2)), 40.0 * math.exp(rng.gauss(0, 0.15)), 1.0
        ts = [0.5, 1, 2, 4, 8, 12, 24]
        cs = [100 * ka / (v * (ka - cl / v)) * (math.exp(-cl / v * t) - math.exp(-ka * t)) * (1 + rng.gauss(0, 0.05))
              for t in ts]
        subs.append({"subject": f"S{i+1}", "doses": [{"time": 0.0, "amt": 100.0}],
                     "obs_t": ts, "obs_c": [max(c, 0.01) for c in cs], "wt": 70.0})
    return subs


# ── pure pieces ─────────────────────────────────────────────────────────────

def test_mlxtran_uses_pkmodel_macro_not_library_ids():
    txt = mlxtran_model("oral_1cmt")
    assert "input = {ka, V, Cl}" in txt and "Cc = pkmodel(ka, V, Cl)" in txt and "output = Cc" in txt


def test_header_types_align_with_dataset_columns():
    assert header_types() == ["id", "time", "observation", "amount", "evid", "ignore", "contcov"]


def test_omega_sd_to_cv():
    assert abs(omega_sd_to_cv_pct(0.3) - 100 * math.sqrt(math.exp(0.09) - 1)) < 1e-9


def test_parse_results_maps_monolix_names_to_app_names():
    rows = [{"parameter": "ka_pop", "estimate": "1.1", "RSE_pct": "12"},
            {"parameter": "V_pop", "estimate": "41", "RSE_pct": "5"},
            {"parameter": "Cl_pop", "estimate": "4.2", "RSE_pct": "6.5"},
            {"parameter": "omega_ka", "estimate": "0.4", "RSE_pct": "30"},
            {"parameter": "omega_V", "estimate": "0.2", "RSE_pct": "NA"},
            {"parameter": "omega_Cl", "estimate": "0.25", "RSE_pct": ""},
            {"parameter": "b", "estimate": "0.11", "RSE_pct": "9"}]
    p = parse_results(rows, "oral_1cmt")
    assert p["theta"] == {"KA": 1.1, "V": 41.0, "CL": 4.2} and p["missing"] == []
    assert p["rse_pct"]["CL"] == 6.5 and p["rse_pct"]["KA"] == 12
    assert set(p["omega_cv_pct"]) == {"KA", "V", "CL"}
    assert abs(p["omega_cv_pct"]["CL"] - omega_sd_to_cv_pct(0.25)) < 1e-9
    assert p["sigma"] == {"prop": 0.11, "add": None}


def test_parse_results_reports_missing_parameters():
    p = parse_results([{"parameter": "V_pop", "estimate": "40", "RSE_pct": "5"}], "oral_1cmt")
    assert p["missing"] == ["CL", "KA"]


# ── availability ────────────────────────────────────────────────────────────

def test_unavailable_when_isolated_r_is_missing(monkeypatch, tmp_path):
    mx._probe_cache.clear()
    monkeypatch.setenv("PHARMAGENT_MONOLIX_RSCRIPT", str(tmp_path / "nope"))
    a = MonolixAdapter()
    assert a.available() is False and "not installed" in a.unavailable_reason


def _installed(monkeypatch, tmp_path):
    mx._probe_cache.clear()
    rs = tmp_path / "Rscript"; rs.write_text("#!/bin/sh\n")
    suite = tmp_path / "suite"; suite.mkdir()
    monkeypatch.setenv("PHARMAGENT_MONOLIX_RSCRIPT", str(rs))
    monkeypatch.setenv("PHARMAGENT_MONOLIX_SUITE", str(suite))


def test_available_only_when_the_engine_initialises(monkeypatch, tmp_path):
    _installed(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(mx.subprocess, "run", lambda cmd, **k: (calls.append(cmd), _Proc(out="PHARMAGENT_MONOLIX_OK"))[1])
    a = MonolixAdapter()
    assert a.available() is True and a.available() is True
    assert len(calls) == 1                                   # probe is cached per process
    assert "initializeLixoftConnectors" in calls[0][-1]


def test_expired_licence_makes_the_engine_absent_with_a_reason(monkeypatch, tmp_path):
    _installed(monkeypatch, tmp_path)
    monkeypatch.setattr(mx.subprocess, "run",
                        lambda cmd, **k: _Proc(out="PHARMAGENT_MONOLIX_FAIL", err='[ERROR] Could not initialize the software "monolix".'))
    a = MonolixAdapter()
    assert a.available() is False and "licence" in a.unavailable_reason


def test_headless_probe_timeout_is_absent_not_a_hang(monkeypatch, tmp_path):
    _installed(monkeypatch, tmp_path)
    def run(*a, **k):
        raise mx.subprocess.TimeoutExpired(cmd="x", timeout=1)
    monkeypatch.setattr(mx.subprocess, "run", run)
    a = MonolixAdapter()
    assert a.available() is False and "GUI session" in a.unavailable_reason


def test_unsupported_model_fails_cleanly():
    r = MonolixAdapter().fit(CandidateSpec(model_key="oral_1cmt_transit"), _subjects())
    assert r.status == "failed" and "supports" in r.message


# ── fit with a faked R process ──────────────────────────────────────────────

class _Proc:
    def __init__(self, out="", err=""):
        self.stdout, self.stderr, self.returncode = out, err, 0


def _fake_run(writer):
    """subprocess.run stand-in: reads config.json, calls writer(cfg) to emit files."""
    calls: list[dict] = []

    def run(cmd, capture_output=True, text=True, timeout=None, env=None):
        cfg = json.loads(Path(cmd[-1]).read_text())
        calls.append({"cmd": cmd, "cfg": cfg, "env": env, "timeout": timeout})
        writer(cfg)
        return _Proc()
    return run, calls


def _write_success(cfg, params=None):
    out = Path(cfg["outDir"]); out.mkdir(parents=True, exist_ok=True)
    params = params or [("ka_pop", 1.05, 10), ("V_pop", 39.5, 4), ("Cl_pop", 4.1, 5),
                        ("omega_ka", 0.3, 40), ("omega_V", 0.15, 30), ("omega_Cl", 0.2, 25),
                        ("b", 0.08, 12)]
    (out / "results.csv").write_text("parameter,estimate,RSE_pct\n" +
                                     "".join(f"{p},{e},{r}\n" for p, e, r in params))
    (out / "loglik.json").write_text(json.dumps({"-2LL": 512.3, "AIC": 526.3, "BIC": 530.0}))
    (out / "status.json").write_text(json.dumps({"ok": True, "message": "ok"}))


def test_fit_success_maps_scores_and_records_inputs(monkeypatch):
    run, calls = _fake_run(_write_success)
    monkeypatch.setattr(mx.subprocess, "run", run)
    monkeypatch.setenv("PHARMAGENT_MONOLIX_RSCRIPT", "/tmp/fake-Rscript")
    monkeypatch.setenv("DYLD_LIBRARY_PATH", "/Applications/anaconda3/lib")
    r = MonolixAdapter().fit(CandidateSpec(model_key="oral_1cmt", iiv_params=["CL", "V"]), _subjects())

    assert isinstance(r, EngineResult) and r.status == "ok" and r.engine == "monolix_saem"
    assert r.params == {"KA": 1.05, "V": 39.5, "CL": 4.1} and r.ofv == 512.3
    assert r.iiv_params == ["CL", "V"] and r.error_model == "proportional"
    assert r.sigma["prop"] == 0.08 and r.rse_pct["CL"] == 5.0
    assert r.pred_rmse is not None and r.n_subjects == 6          # scored like every engine
    assert r.converged and r.runtime_s is not None
    c = calls[0]
    assert c["cfg"]["headerTypes"] == header_types() and c["cfg"]["errorModel"] == "proportional"
    assert c["cfg"]["distribution"] == "lognormal"
    assert "DYLD_LIBRARY_PATH" not in c["env"]                     # stripped, as run-monolix.sh does
    assert c["cmd"][-2].endswith("monolix_fit.R") and c["timeout"] == mx._FIT_TIMEOUT_S
    # the generated model + data really were on disk for the R side
    assert Path(c["cfg"]["modelFile"]).name == "model_oral_1cmt.txt"


def test_error_model_translation(monkeypatch):
    run, calls = _fake_run(lambda cfg: _write_success(cfg, params=[
        ("ka_pop", 1, 1), ("V_pop", 40, 1), ("Cl_pop", 4, 1), ("a", 0.5, 1), ("b", 0.1, 1)]))
    monkeypatch.setattr(mx.subprocess, "run", run)
    r = MonolixAdapter().fit(CandidateSpec(model_key="oral_1cmt", error_model="combined"), _subjects())
    assert calls[0]["cfg"]["errorModel"] == "combined1"
    assert r.error_model == "combined" and r.sigma == {"prop": 0.1, "add": 0.5}


def test_engine_error_in_status_is_a_failed_row(monkeypatch):
    def writer(cfg):
        out = Path(cfg["outDir"]); out.mkdir(parents=True, exist_ok=True)
        (out / "status.json").write_text(json.dumps({"ok": False, "message": "ERROR: license not found"}))
    run, _ = _fake_run(writer)
    monkeypatch.setattr(mx.subprocess, "run", run)
    r = MonolixAdapter().fit(CandidateSpec(model_key="iv_1cmt"), _subjects())
    assert r.status == "failed" and "license" in r.message


def test_timeout_names_the_gui_session_requirement(monkeypatch):
    def run(*a, **k):
        raise mx.subprocess.TimeoutExpired(cmd="x", timeout=1)
    monkeypatch.setattr(mx.subprocess, "run", run)
    r = MonolixAdapter().fit(CandidateSpec(model_key="iv_1cmt"), _subjects())
    assert r.status == "failed" and "GUI session" in r.message


def test_no_status_file_reports_process_output(monkeypatch):
    def run(cmd, **k):
        return _Proc(err="Rscript: cannot open shared object liblixoftConnectors.so")
    monkeypatch.setattr(mx.subprocess, "run", run)
    r = MonolixAdapter().fit(CandidateSpec(model_key="iv_1cmt"), _subjects())
    assert r.status == "failed" and "liblixoftConnectors" in r.message


def test_missing_parameter_in_results_is_failed(monkeypatch):
    run, _ = _fake_run(lambda cfg: _write_success(cfg, params=[("V_pop", 40, 1), ("b", 0.1, 1)]))
    monkeypatch.setattr(mx.subprocess, "run", run)
    r = MonolixAdapter().fit(CandidateSpec(model_key="oral_1cmt"), _subjects())
    assert r.status == "failed" and "missing" in r.message


# ── registry ────────────────────────────────────────────────────────────────

def test_monolix_is_a_selectable_engine():
    from app.engines import MonolixAdapter as Exported
    from app.tools.engine_tools import _resolve_adapters
    names = [a.name for a in _resolve_adapters(["pharmagent_focei", "monolix"])]
    assert names == ["pharmagent_focei", "monolix"] and Exported is MonolixAdapter


def test_runner_records_absent_when_unavailable(monkeypatch, tmp_path):
    from app.engines.runner import run_matrix_subjects
    mx._probe_cache.clear()
    monkeypatch.setenv("PHARMAGENT_MONOLIX_RSCRIPT", str(tmp_path / "nope"))
    out = run_matrix_subjects(_subjects(), [CandidateSpec(model_key="oral_1cmt")], [MonolixAdapter()])
    rows = out["results"] if isinstance(out, dict) and "results" in out else out
    statuses = [r["status"] if isinstance(r, dict) else r.status for r in rows]
    assert statuses == ["absent"]


def test_r_script_is_packaged():
    assert mx.R_SCRIPT.is_file() and "initializeLixoftConnectors" in mx.R_SCRIPT.read_text()
    assert os.access(mx.R_SCRIPT, os.R_OK)


@pytest.mark.skipif(not mx._rscript().is_file(), reason="isolated Monolix R not installed")
def test_real_monolix_binary_is_wired_when_present():
    """On this Mac the isolated R exists; we only check the command shape (no run)."""
    cmd = mx._command(mx._rscript(), "cfg.json")
    assert cmd[-1] == "cfg.json" and cmd[-2].endswith("monolix_fit.R")
