"""The PKNCA cross-check pipeline, tested without R: input construction, PharmAgent's NCA against an analytic
solution and against a textbook re-implementation, and the comparer's behaviour. The PKNCA run itself is done by a
person (scripts/crosscheck_pknca/README.md); nothing here executes R."""
from __future__ import annotations

import csv
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from app.compute.nca import Profile, nca_subject

DIR = Path(__file__).resolve().parents[1] / "scripts" / "crosscheck_pknca"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


make_input, ref_np, compare, run_pa = (_load(n) for n in ("make_nca_input", "reference_numpy", "compare", "run_pharmagent"))


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


# ── the shared input table ──────────────────────────────────────────────────────────────────────
def test_committed_input_is_exactly_what_the_builder_produces(tmp_path):
    make_input.main(["x", str(make_input.DEFAULT_IN), str(tmp_path / "in.csv")])
    assert (tmp_path / "in.csv").read_bytes() == (DIR / "nca_input.csv").read_bytes()


def test_input_rules_hold():
    rows = _read(DIR / "nca_input.csv")
    ids = {r["ID"] for r in rows}
    assert len(ids) == 6
    for sid in ids:
        mine = [r for r in rows if r["ID"] == sid]
        doses = [r for r in mine if r["EVID"] == "1"]
        obs = [r for r in mine if r["EVID"] == "0"]
        assert len(doses) == 1 and doses[0]["TIME"] == "0" and doses[0]["AMT"] == "54"      # first dose only
        assert all(float(r["TIME"]) <= 24 for r in obs)                                     # first interval only
        assert all(r["DV"] != "" for r in obs)                                              # missing is dropped
        assert [r["TIME"] for r in obs if r["BLQ"] == "1"] == ["0"]                        # only the pre-dose zero


# ── PharmAgent's NCA versus an ANALYTIC truth (no engine involved) ──────────────────────────────
def test_pharmagent_nca_equals_the_analytic_solution_for_a_mono_exponential():
    c0, k = 10.0, 0.2
    t = np.array([0.0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 36, 48])
    out = nca_subject(Profile("S", t, c0 * np.exp(-k * t), dose=100.0))
    assert out["Cmax"] == pytest.approx(c0, rel=1e-6) and out["Tmax"] == 0.0
    assert out["lambda_z"] == pytest.approx(k, rel=1e-5)                      # fit of exact exponential data
    assert out["t_half"] == pytest.approx(math.log(2) / k, rel=1e-5)
    assert out["AUC_inf"] == pytest.approx(c0 / k, rel=1e-5)                  # log-down is exact for exponentials
    assert out["CL_F"] == pytest.approx(100.0 / (c0 / k), rel=1e-5)


def test_the_textbook_reference_matches_the_analytic_solution_too():
    c0, k = 10.0, 0.2
    t = np.array([0.0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 36, 48])
    out = ref_np.subject("S", t, c0 * np.exp(-k * t))
    assert out["AUCinf.obs"] == pytest.approx(c0 / k, rel=1e-9) and out["lambda.z"] == pytest.approx(k, rel=1e-9)


# ── PharmAgent end to end on the fixture, against the textbook reference ────────────────────────
@pytest.fixture(scope="module")
def outputs(tmp_path_factory):
    out = tmp_path_factory.mktemp("cc")
    assert run_pa.main(["x", str(DIR / "nca_input.csv"), str(out)]) == 0
    assert ref_np.main(["x", str(DIR / "nca_input.csv"), str(out)]) == 0
    return out


def test_pharmagent_detects_every_role_by_name_and_returns_all_subjects(outputs):
    import json
    roles = json.loads((outputs / "pharmagent_roles.json").read_text())
    assert roles == {"ID": "ID", "TIME": "TIME", "DV": "DV", "AMT": "AMT", "EVID": "EVID", "BLQ": "CENS"}
    assert len(_read(outputs / "pharmagent.csv")) == 6


def test_pharmagent_agrees_with_the_textbook_reference_on_the_fixture(outputs):
    res = compare.compare(compare.load(outputs / "pharmagent.csv"), compare.load(outputs / "numpy_reference.csv"))
    bad = [r for r in res["rows"] if r["status"] != "match"]
    assert not bad, bad
    assert len(res["rows"]) == 6 * len(compare.PARAMS)


# ── the comparer ────────────────────────────────────────────────────────────────────────────────
def _tbl(**kw):
    base = {"Cmax": 2.0, "Tmax": 8.0, "AUClast": 18.0, "AUCinf.obs": 18.5, "lambda.z": 0.3, "half.life": 2.31,
            "lambda.z.n.points": 3.0, "adj.r.squared": 0.99}
    base.update(kw)
    return {"S1": base}


def test_compare_flags_each_kind_of_difference():
    mine = _tbl()
    assert all(r["status"] == "match" for r in compare.compare(mine, _tbl())["rows"])
    for param, val in (("AUClast", 18.5), ("lambda.z.n.points", 4.0), ("adj.r.squared", 0.9), ("Cmax", 2.1)):
        rows = compare.compare(mine, _tbl(**{param: val}))["rows"]
        assert [r["param"] for r in rows if r["status"] == "DIFFERS"] == [param]


def test_compare_reports_missing_values_and_subjects():
    res = compare.compare(_tbl(**{"lambda.z": None}), _tbl())
    assert {r["param"] for r in res["rows"] if r["status"] == "one_missing"} == {"lambda.z"}
    other = {"S2": _tbl()["S1"]}
    res = compare.compare(_tbl(), other)
    assert res["only_mine"] == ["S1"] and res["only_ref"] == ["S2"]


def test_compare_exit_codes(outputs, tmp_path, capsys):
    pa, npr = outputs / "pharmagent.csv", outputs / "numpy_reference.csv"
    assert compare.main(["x", str(pa), str(npr)]) == 0
    assert compare.main(["x", str(pa), str(npr), str(tmp_path / "pknca_default.csv")]) == 2     # not produced yet
    assert "NOT PRODUCED YET" in capsys.readouterr().out
    bent = tmp_path / "bent.csv"
    rows = _read(npr)
    rows[0]["AUClast"] = str(float(rows[0]["AUClast"]) * 1.1)
    with bent.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    assert compare.main(["x", str(pa), str(bent)]) == 1
    assert compare.main(["x", str(tmp_path / "missing.csv"), str(npr)]) == 2


# ── the R script is a reviewed artefact: static properties only (nothing is executed) ───────────
def test_r_script_installs_nothing_and_pins_the_conventions():
    src = (DIR / "run_pknca.R").read_text(encoding="utf-8")
    for forbidden in ("install.packages", "download.file", "system(", "system2(", "url(", "Sys.setenv", "source(",
                      "eval(", "library(devtools", "remotes::"):
        assert forbidden not in src, forbidden
    for needed in ("lin up/log down", "adj.r.squared.factor = 0", "min.hl.points = 3", "allow.tmax.in.half.life = FALSE",
                   'route = "extravascular"', "pknca_default", "pknca_matched", "packageVersion"):
        assert needed in src, needed
