"""The mrgsolve POPULATION cross-check, tested without R.

Layer A  shared-draw determinism: PharmAgent's real simulate_replicate_subject fed a scripted RNG, checked against closed
         forms re-derived here (no other engine), plus the comparer and static properties of the R script.
Layer B  PharmAgent's OWN random-number population simulation checked statistically against exact distributions
         (eta variance, residual scale, vpc_band percentiles vs the analytic quantile of a monotone model).
The mrgsolve run itself is done by a person (scripts/crosscheck_mrgsolve/README.md); nothing here executes R."""
from __future__ import annotations

import csv
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from app.compute.nlme import _PopSpec, _predict, _Subject, simulate_replicate_subject
from app.compute.pk_models import get_model
from app.compute.vpc import vpc_band

DIR = Path(__file__).resolve().parents[1] / "scripts" / "crosscheck_mrgsolve" / "pop"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"pop_{name}", DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


make, run_pa, cmp = (_load(n) for n in ("make_pop", "run_pharmagent_pop", "compare_pop"))


def _read(p: Path) -> list[dict[str, str]]:
    with p.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


@pytest.fixture(scope="module")
def pa(tmp_path_factory):
    out = tmp_path_factory.mktemp("mxpop")
    assert run_pa.main(["x", str(DIR), str(out)]) == 0
    return _read(out / "pharmagent_pop.csv"), _read(out / "pharmagent_eta.csv")


# ───────────────────────────── Layer A: tables and runner ─────────────────────────────

def test_committed_tables_are_what_the_generator_makes(tmp_path):
    assert make.main(["x", str(tmp_path)]) == 0
    for name in ("design", "doses", "obs", "draws"):
        assert (tmp_path / f"{name}.csv").read_bytes() == (DIR / f"{name}.csv").read_bytes(), name


def test_draws_are_clipped_and_complete():
    design, draws = _read(DIR / "design.csv"), _read(DIR / "draws.csv")
    obs = _read(DIR / "obs.csv")
    assert max(abs(float(r["z"])) for r in draws) <= make.CLIP
    for d in design:
        n_eta = len([p for p in d["iiv"].split(",") if p])
        n_obs = len([r for r in obs if r["id"] == d["id"]])
        got = [r for r in draws if r["id"] == d["id"]]
        assert len(got) == int(d["n_subjects"]) * (n_eta + n_obs)


def test_scripted_rng_refuses_extra_or_misshaped_requests():
    rng = run_pa.ScriptedRng([0.5, -0.5], [1.0])
    with pytest.raises(RuntimeError):
        rng.normal(0.0, np.array([1.0]))                  # wrong number of etas
    rng = run_pa.ScriptedRng([0.5], [1.0])
    rng.normal(0.0, np.array([2.0]))
    with pytest.raises(RuntimeError):
        rng.normal(0.0, np.array([2.0]))                  # etas asked for twice
    rng.standard_normal()
    with pytest.raises(RuntimeError):
        rng.standard_normal()                             # more residual draws than observations


def test_runner_uses_every_draw_and_never_resamples(pa):
    pop, eta = pa
    design = _read(DIR / "design.csv")
    n_obs = {r["id"]: len([o for o in _read(DIR / "obs.csv") if o["id"] == r["id"]]) for r in design}
    assert len(pop) == sum(int(d["n_subjects"]) * n_obs[d["id"]] for d in design)
    assert len(eta) == sum(int(d["n_subjects"]) * len([p for p in d["iiv"].split(",") if p]) for d in design)


def _iv1_closed_form(theta, eta, wt, doses, times):
    """iv_1cmt bolus: C = D/V exp(-CL/V t), eta on CL and V BEFORE the 0.75 / 1 allometric scaling."""
    f = wt / 70.0
    cl = theta["CL"] * math.exp(eta["CL"]) * f ** 0.75
    v = theta["V"] * math.exp(eta["V"]) * f
    return [sum(a / v * math.exp(-cl / v * (t - t0)) for t0, a in doses if t >= t0) for t in times]


def test_pharmagent_population_matches_closed_form_for_iv1(pa):
    pop, eta = pa
    d = next(r for r in _read(DIR / "design.csv") if r["id"] == "pop_iv1")
    theta = cmp_kv(d["theta"])
    om2 = cmp_kv(d["omega2"])
    draws = [r for r in _read(DIR / "draws.csv") if r["id"] == "pop_iv1"]
    times = [float(r["time"]) for r in _read(DIR / "obs.csv") if r["id"] == "pop_iv1"]
    for s in range(1, int(d["n_subjects"]) + 1):
        z = sorted((r for r in draws if int(r["subject"]) == s), key=lambda r: (r["kind"], int(r["idx"])))
        z_eta = [float(r["z"]) for r in z if r["kind"] == "eta"]
        z_eps = [float(r["z"]) for r in z if r["kind"] == "eps"]
        e = {p: math.sqrt(om2[p]) * zz for p, zz in zip(["CL", "V"], z_eta, strict=True)}
        ip = _iv1_closed_form(theta, e, float(d["wt"]), [(0.0, 100.0)], times)
        sig_p, sig_a = float(d["sigma_prop"]), float(d["sigma_add"])
        rows = [r for r in pop if r["id"] == "pop_iv1" and int(r["subject"]) == s]
        for r, f, zz in zip(rows, ip, z_eps, strict=True):
            sd = math.sqrt(sig_a ** 2 + (sig_p * f) ** 2)
            assert float(r["ipred"]) == pytest.approx(f, rel=1e-6)
            assert float(r["dv"]) == pytest.approx(f + sd * zz, rel=1e-6)
        got = {r["param"]: float(r["eta"]) for r in eta if r["id"] == "pop_iv1" and int(r["subject"]) == s}
        assert got == pytest.approx(e, abs=1e-12)


def cmp_kv(s: str) -> dict[str, float]:
    return run_pa.kv(s)


def test_weight_scaling_is_applied_after_the_eta(pa):
    """The 100 kg infusion scenario: two-compartment, so check the invariant instead of a closed form -- a subject whose
    eta is zero would equal the scaled typical value; with eta != 0 the realised CL must be theta*exp(eta)*(wt/70)^0.75."""
    pop, eta = pa
    d = next(r for r in _read(DIR / "design.csv") if r["id"] == "pop_iv2_infusion_100kg")
    theta, om2 = cmp_kv(d["theta"]), cmp_kv(d["omega2"])
    spec = _PopSpec(get_model(d["model"]), ["CL", "VC"], "additive")
    doses = [{"time": 0.0, "amt": 100.0, "rate": 25.0}]
    times = [float(r["time"]) for r in _read(DIR / "obs.csv") if r["id"] == "pop_iv2_infusion_100kg"]
    subj = _Subject({"subject": "x", "doses": doses, "obs_t": times, "obs_c": [1.0] * len(times), "wt": 100.0})
    e = {r["param"]: float(r["eta"]) for r in eta if r["id"] == "pop_iv2_infusion_100kg" and r["subject"] == "1"}
    assert set(e) == {"CL", "VC"} and om2["CL"] > 0
    f1 = _predict(spec, subj, {**get_model(d["model"]).defaults, **theta}, np.array([e["CL"], e["VC"]]))
    f0 = _predict(spec, subj, {**get_model(d["model"]).defaults, **theta}, np.zeros(2))
    got = [float(r["ipred"]) for r in pop if r["id"] == "pop_iv2_infusion_100kg" and r["subject"] == "1"]
    assert got == pytest.approx(list(f1), rel=1e-9)
    assert not np.allclose(f1, f0)                        # the eta really moves the prediction


def test_blq_flag_is_dv_below_lloq(pa):
    pop, _ = pa
    rows = [r for r in pop if r["id"] == "pop_lag_blq"]
    assert any(r["blq"] == "1" for r in rows) and any(r["blq"] == "0" for r in rows)   # the scenario exercises both
    assert all((float(r["dv"]) < 0.3) == (r["blq"] == "1") for r in rows)


# ───────────────────────────── comparer ─────────────────────────────

def test_comparer_passes_identical_and_fails_a_perturbation(pa):
    pop, eta = pa
    r = cmp.compare(pop, pop, eta, eta)
    assert cmp.passed(r) and r["max_dv_rel"] == 0.0
    bad = [dict(x) for x in pop]
    bad[5]["dv"] = repr(float(bad[5]["dv"]) * 1.001)
    assert not cmp.passed(cmp.compare(pop, bad, eta, eta))
    bad_eta = [dict(x) for x in eta]
    bad_eta[0]["eta"] = repr(float(bad_eta[0]["eta"]) + 1e-9)
    assert not cmp.passed(cmp.compare(pop, pop, eta, bad_eta))


def test_comparer_counts_a_blq_flip_unless_on_the_threshold(pa):
    pop, eta = pa
    i = next(k for k, x in enumerate(pop) if x["id"] == "pop_lag_blq" and x["blq"] == "0")
    flipped = [dict(x) for x in pop]
    flipped[i]["blq"] = "1"
    assert cmp.compare(pop, flipped, eta, eta, lloq={"pop_lag_blq": 0.3})["blq_flips"] == 1
    on_edge = [dict(x) for x in flipped]
    on_edge[i]["dv"] = repr(0.3 * (1 + 1e-7))
    pop_edge = [dict(x) for x in pop]
    pop_edge[i]["dv"] = on_edge[i]["dv"]
    assert cmp.compare(pop_edge, on_edge, eta, eta, lloq={"pop_lag_blq": 0.3})["blq_edge"] == 1


def test_comparer_rejects_misaligned_rows(pa):
    pop, eta = pa
    with pytest.raises(ValueError):
        cmp.compare(pop[:-1], pop, eta, eta)
    shuffled = pop[1:] + pop[:1]
    with pytest.raises(ValueError):
        cmp.compare(pop, shuffled, eta, eta)


def test_comparer_exit_2_when_reference_missing(tmp_path, capsys):
    assert cmp.main(["x", str(tmp_path), str(tmp_path)]) == 2
    assert "reference not produced" in capsys.readouterr().out


# ───────────────────────────── R script, static ─────────────────────────────

def test_r_script_draws_nothing_and_stays_self_contained():
    src = (DIR / "run_mrgsolve_pop.R").read_text(encoding="utf-8")
    code = "\n".join(line.split("#", 1)[0] for line in src.splitlines())
    for forbidden in ("rnorm", "runif", "set.seed", "rmvnorm", "mvrnorm", "source(", "install.packages", "download", "url("):
        assert forbidden not in code, forbidden
    assert "draws.csv" in src and "sqrt(om2[iiv]) * z_eta" in src and "dv <- ipred + sd * z_eps" in src
    assert (src.index("exp(eta)") < src.index("allo_exponent(n)"))             # eta first, THEN weight scaling


# ───────────────────────────── Layer B: own-RNG statistics ─────────────────────────────

N_STAT = 4000


def _simulate_population(model, theta, om2, sp, sa, iiv, err, doses, times, n, seed, wt=70.0):
    rng = np.random.default_rng(seed)
    spec = _PopSpec(get_model(model), iiv, err)
    subj = _Subject({"subject": "x", "doses": doses, "obs_t": times, "obs_c": [1.0] * len(times), "wt": wt})
    etas, z = [], []
    for _ in range(n):
        res = simulate_replicate_subject(model, theta, om2, sp, sa, iiv, err, doses, times, wt, rng)
        assert res["ok"]
        e = np.array([res["eta"][p] for p in iiv])
        f = _predict(spec, subj, {**get_model(model).defaults, **theta}, e)
        etas.append(e)
        z.append((np.array(res["obs_c"]) - f) / np.sqrt(sa ** 2 + (sp * f) ** 2))
    return np.array(etas), np.array(z)


def test_own_rng_eta_has_the_requested_variance_and_zero_mean():
    etas, _ = _simulate_population("iv_1cmt", {"CL": 5.0, "V": 50.0}, {"CL": 0.09, "V": 0.04}, 0.05, 0.0,
                                   ["CL", "V"], "proportional", [{"time": 0.0, "amt": 100.0, "rate": 0.0}],
                                   [1.0, 4.0], N_STAT, 11)
    assert etas.std(axis=0, ddof=1) == pytest.approx([0.3, 0.2], rel=0.04)       # se of an sd ~ 1.1 %
    assert abs(etas.mean(axis=0)[0]) < 4 * 0.3 / math.sqrt(N_STAT)
    assert abs(np.corrcoef(etas.T)[0, 1]) < 0.05                                  # independent effects


@pytest.mark.parametrize("err,sp,sa", [("proportional", 0.15, 0.0), ("additive", 0.0, 0.02), ("combined", 0.1, 0.01)])
def test_own_rng_residual_is_standard_normal_on_the_modelled_scale(err, sp, sa):
    _, z = _simulate_population("iv_1cmt", {"CL": 5.0, "V": 50.0}, {"CL": 0.04}, sp, sa, ["CL"], err,
                                [{"time": 0.0, "amt": 100.0, "rate": 0.0}], [0.5, 2.0, 6.0], N_STAT, 3)
    assert z.shape == (N_STAT, 3)
    assert abs(z.mean()) < 4 / math.sqrt(z.size)
    assert z.std(ddof=1) == pytest.approx(1.0, rel=0.04)


def test_vpc_band_matches_the_exact_percentiles_for_a_monotone_model():
    """CL-only lognormal IIV on an iv_1cmt bolus: C(t) falls monotonically in eta, so the p-th percentile of C is C at the
    (1-p)-th percentile of eta -- an exact reference with no simulation in it."""
    sd = 0.3
    cv = 100.0 * math.sqrt(math.exp(sd ** 2) - 1.0)
    out = vpc_band("iv_1cmt", {"CL": 5.0, "V": 50.0}, {"CL": cv}, [{"time": 0.0, "amt": 100.0, "rate": 0.0}],
                   tmax=12.0, n_grid=25, n_sim=N_STAT, seed=7)
    t = np.array(out["times"])
    z95 = 1.6448536269514722

    def exact(z):
        return 100.0 / 50.0 * np.exp(-(5.0 * math.exp(sd * z) / 50.0) * t)

    assert np.allclose(out["p50"], exact(0.0), rtol=0.02)
    assert np.allclose(out["p05"], exact(+z95), rtol=0.04)                         # high eta -> low concentration
    assert np.allclose(out["p95"], exact(-z95), rtol=0.04)
    assert np.all(np.array(out["p05"]) <= np.array(out["p50"])) and np.all(np.array(out["p50"]) <= np.array(out["p95"]))


# ───────────────────────────── offline regression vs the committed mrgsolve run ─────────────────────────────

REF = DIR.parent / "reference" / "pop"


def test_pharmagent_still_agrees_with_the_committed_mrgsolve_population_run(pa):
    pop, eta = pa
    r = cmp.compare(pop, _read(REF / "mrgsolve_pop.csv"), eta, _read(REF / "mrgsolve_eta.csv"),
                    lloq={"pop_lag_blq": 0.3})
    assert cmp.passed(r), r
    assert r["max_ipred_rel"] < 5e-6 and r["max_eta_abs"] < 1e-12


def test_comparer_main_passes_on_the_committed_reference(pa, tmp_path):
    out = tmp_path / "pa"
    assert run_pa.main(["x", str(DIR), str(out)]) == 0
    assert cmp.main(["x", str(out), str(REF)]) == 0
