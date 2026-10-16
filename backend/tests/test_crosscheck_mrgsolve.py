"""The mrgsolve cross-check, tested without R: scenario construction, PharmAgent's simulator against CLOSED-FORM
solutions (no other engine involved), the comparer, and static properties of the R script. Nothing here executes R;
the mrgsolve run itself is done by a person (scripts/crosscheck_mrgsolve/README.md)."""
from __future__ import annotations

import csv
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.special import lambertw

DIR = Path(__file__).resolve().parents[1] / "scripts" / "crosscheck_mrgsolve"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


make, run_pa, cmp = (_load(n) for n in ("make_scenarios", "run_pharmagent_sim", "compare_sims"))


def _read(p: Path) -> list[dict[str, str]]:
    with p.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


@pytest.fixture(scope="module")
def sim(tmp_path_factory):
    out = tmp_path_factory.mktemp("mx")
    assert run_pa.main(["x", str(DIR), str(out)]) == 0
    return cmp.load(out / "pharmagent_sim.csv")


def _params(sid: str) -> dict[str, float]:
    s = next(r for r in _read(DIR / "scenarios.csv") if r["id"] == sid)
    return {k: float(v) for k, v in (kv.split("=") for kv in s["params"].split(";"))}


def _wt(sid: str) -> float:
    return float(next(r for r in _read(DIR / "scenarios.csv") if r["id"] == sid)["wt"])


def _doses(sid: str) -> list[tuple[float, float, float]]:
    return [(float(r["time"]), float(r["amt"]), float(r["rate"])) for r in _read(DIR / "doses.csv") if r["id"] == sid]


def _nmax(t_c: tuple[np.ndarray, np.ndarray], truth: np.ndarray) -> float:
    return float(np.max(np.abs(t_c[1] - truth)) / np.max(np.abs(truth)))


# ── the scenario tables ─────────────────────────────────────────────────────────────────────────
def test_committed_tables_are_exactly_what_the_builder_produces(tmp_path):
    make.main(["x", str(tmp_path)])
    for name in ("scenarios.csv", "doses.csv", "obs.csv"):
        assert (tmp_path / name).read_bytes() == (DIR / name).read_bytes(), name


def test_design_covers_every_model_and_situation():
    scen = _read(DIR / "scenarios.csv")
    assert len(scen) == 38 and {s["model"] for s in scen} == set(make.MODELS)
    kinds = {s["id"].split(":")[1] for s in scen}
    assert kinds == {"single", "multiple", "weight", "infusion", "saturate"}
    assert all(s["model"].startswith("iv") for s in scen if s["id"].endswith("infusion"))     # IV models only
    assert {s["wt"] for s in scen} == {"70", "100"}


def test_every_scenario_has_doses_and_a_shared_grid_starting_at_zero():
    scen, doses, obs = _read(DIR / "scenarios.csv"), _read(DIR / "doses.csv"), _read(DIR / "obs.csv")
    for s in scen:
        d = [r for r in doses if r["id"] == s["id"]]
        o = [float(r["time"]) for r in obs if r["id"] == s["id"]]
        assert d and o and o[0] == 0.0 and o == sorted(o) and len(o) >= 160


# ── PharmAgent's simulator versus closed-form solutions ─────────────────────────────────────────
def test_iv_1cmt_bolus_equals_the_exponential(sim):
    sid = "iv_1cmt:single"
    p = _params(sid)
    t, _ = sim[sid]
    truth = (100 / p["V"]) * np.exp(-(p["CL"] / p["V"]) * t)
    assert _nmax(sim[sid], truth) < 1e-9


def test_iv_1cmt_multiple_dosing_equals_superposition(sim):
    sid = "iv_1cmt:multiple"
    p = _params(sid)
    t, _ = sim[sid]
    k = p["CL"] / p["V"]
    truth = sum(np.where(t >= td, (amt / p["V"]) * np.exp(-k * (t - td)), 0.0) for td, amt, _ in _doses(sid))
    assert _nmax(sim[sid], truth) < 1e-9


def test_allometric_weight_scaling_matches_the_stated_convention(sim):
    sid = "iv_1cmt:weight"
    p, f = _params(sid), _wt(sid) / 70
    cl, v = p["CL"] * f ** 0.75, p["V"] * f
    t, _ = sim[sid]
    assert _nmax(sim[sid], (100 / v) * np.exp(-(cl / v) * t)) < 1e-9


def test_oral_1cmt_equals_the_bateman_function(sim):
    sid = "oral_1cmt:single"
    p = _params(sid)
    t, _ = sim[sid]
    k, ka = p["CL"] / p["V"], p["KA"]
    truth = 100 * ka / (p["V"] * (ka - k)) * (np.exp(-k * t) - np.exp(-ka * t))
    assert _nmax(sim[sid], truth) < 1e-9


def test_oral_1cmt_lag_shifts_the_whole_curve(sim):
    sid = "oral_1cmt_lag:single"
    p = _params(sid)
    t, _ = sim[sid]
    k, ka, lag = p["CL"] / p["V"], p["KA"], p["ALAG"]
    tt = np.clip(t - lag, 0, None)
    truth = 100 * ka / (p["V"] * (ka - k)) * (np.exp(-k * tt) - np.exp(-ka * tt))
    assert _nmax(sim[sid], truth) < 1e-9


def test_iv_2cmt_equals_the_biexponential(sim):
    sid = "iv_2cmt:single"
    p = _params(sid)
    k10, k12, k21 = p["CL"] / p["VC"], p["Q"] / p["VC"], p["Q"] / p["VP"]
    s = k10 + k12 + k21
    disc = math.sqrt(s * s - 4 * k10 * k21)
    alpha, beta = (s + disc) / 2, (s - disc) / 2
    t, _ = sim[sid]
    truth = (100 / p["VC"]) * ((k21 - alpha) / (beta - alpha) * np.exp(-alpha * t)
                               + (k21 - beta) / (alpha - beta) * np.exp(-beta * t))
    assert _nmax(sim[sid], truth) < 1e-9


def test_iv_infusion_equals_the_closed_form(sim):
    sid = "iv_1cmt:infusion"
    p = _params(sid)
    (_, amt, rate), = _doses(sid)
    k, tinf = p["CL"] / p["V"], amt / rate
    t, _ = sim[sid]
    during = (rate / p["CL"]) * (1 - np.exp(-k * np.minimum(t, tinf)))
    truth = np.where(t <= tinf, during, (rate / p["CL"]) * (1 - np.exp(-k * tinf)) * np.exp(-k * (t - tinf)))
    # an infusion forces the numerical (LSODA, rtol 1e-7) path, so ~1e-6 is the solver tolerance, not exactness
    assert _nmax(sim[sid], truth) < 1e-5


def test_saturable_elimination_equals_the_lambert_w_solution(sim):
    """V dC/dt = -VMAX C/(KM + C) has the exact solution C = KM W((C0/KM) exp((C0 - VMAX t/V)/KM))."""
    for sid, dose in (("iv_1cmt_mm:single", 100.0), ("iv_1cmt_mm:saturate", 2000.0)):
        p = _params(sid)
        t, _ = sim[sid]
        c0 = dose / p["V"]
        truth = p["KM"] * np.real(lambertw((c0 / p["KM"]) * np.exp((c0 - p["VMAX"] * t / p["V"]) / p["KM"])))
        assert _nmax(sim[sid], truth) < 1e-5, sid


# ── the comparer ────────────────────────────────────────────────────────────────────────────────
def _toy(scale=1.0):
    t = np.linspace(0, 10, 11)
    return {"S": (t, scale * np.exp(-0.3 * t))}


def test_comparer_matches_identical_and_flags_each_kind_of_gap():
    ref = _toy()
    assert all(r["status"] == "match" for r in cmp.compare(_toy(), ref)["rows"])
    assert cmp.compare(_toy(1.001), ref)["rows"][0]["status"] == "OUT"
    bent = {"S": (ref["S"][0] + 0.5, ref["S"][1])}
    assert cmp.compare(bent, ref)["rows"][0]["status"] == "GRID_MISMATCH"
    res = cmp.compare(_toy(), {"T": ref["S"]})
    assert res["only_mine"] == ["S"] and res["only_ref"] == ["T"]


def test_comparer_exit_codes(sim, tmp_path, capsys):
    mine = tmp_path / "a.csv"
    with mine.open("w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["id", "time", "cp"])
        for sid, (t, c) in sim.items():
            w.writerows(zip([sid] * len(t), (repr(float(x)) for x in t), (repr(float(x)) for x in c), strict=True))
    assert cmp.main(["x", str(mine), str(mine)]) == 0
    assert cmp.main(["x", str(mine), str(tmp_path / "missing.csv")]) == 2
    assert "NOT PRODUCED YET" in capsys.readouterr().out
    rows = _read(mine)
    rows[10]["cp"] = repr(float(rows[10]["cp"]) * 1.5 + 1.0)
    other = tmp_path / "b.csv"
    with other.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "time", "cp"], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    assert cmp.main(["x", str(mine), str(other)]) == 1


# ── the R script is a reviewed artefact: static properties only (nothing is executed) ───────────
def test_r_script_installs_nothing_and_has_every_model_and_convention():
    src = (DIR / "run_mrgsolve.R").read_text(encoding="utf-8")
    for forbidden in ("install.packages", "download.file", "system(", "system2(", "url(", "Sys.setenv", "source(",
                      "eval(", "remotes::", "devtools"):
        assert forbidden not in src, forbidden
    for key in make.MODELS:
        assert f"  {key} = '" in src, key
    for needed in ("atol = 1e-10", "rtol = 1e-10", "obsonly = TRUE", "ALAG_DEPOT = ALAG", "KTR = 4.0/MTT",
                   "allo_exponent", "0.75", "dose first", "packageVersion", "requireNamespace"):
        assert needed in src, needed


def test_r_model_equations_name_the_same_parameters_as_the_registry():
    """The R model text must declare exactly the registry's parameters for each model (names and defaults)."""
    import re
    src = (DIR / "run_mrgsolve.R").read_text(encoding="utf-8")
    from app.compute.pk_models import REGISTRY
    for key in make.MODELS:
        block = src.split(f"  {key} = '")[1].split("'")[0]
        declared = dict(re.findall(r"(\w+)\s*=\s*([0-9.]+)", block.split("$PARAM")[1].split("$")[0]))
        want = {k: v for k, v in REGISTRY[key].defaults.items() if k in REGISTRY[key].params}
        assert {k: float(v) for k, v in declared.items()} == {k: float(v) for k, v in want.items()}, key


# ── offline regression against the committed mrgsolve output (no R needed) ──────────────────────
REF = DIR / "reference" / "mrgsolve_sim.csv"


def test_pharmagent_still_agrees_with_the_committed_mrgsolve_reference(sim):
    ref = cmp.load(REF)
    res = cmp.compare(sim, ref, 1e-5)
    assert not res["only_mine"] and not res["only_ref"]
    bad = [r for r in res["rows"] if r["status"] != "match"]
    assert not bad, bad
    assert len(res["rows"]) == 38


def test_reference_grid_is_the_scenario_grid():
    obs = _read(DIR / "obs.csv")
    ref = _read(REF)
    assert [(r["id"], round(float(r["time"]), 5)) for r in ref] == [(r["id"], round(float(r["time"]), 5)) for r in obs]
    assert "mrgsolve" in (DIR / "reference" / "mrgsolve_versions.txt").read_text()
