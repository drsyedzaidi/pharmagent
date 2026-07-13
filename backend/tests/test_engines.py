"""Cross-engine orchestration tests (v0) — keyless, no Monolix/R/license.

Two kinds of test:
* fit-based (module-scoped, reused) — exercise the real native fitter through the
  adapter and the mock oracle;
* selection-invariant (hand-constructed EngineResults, fast) — lock down THE rule:
  the winner is never chosen by cross-engine OFV/AIC/BIC.
"""
import numpy as np
import pytest

from app.compute.pk_models import get_model
from app.compute.pk_simulate import simulate
from app.engines import (
    CandidateSpec,
    EngineResult,
    MockEngineAdapter,
    Nlmixr2Adapter,
    PharmAgentAdapter,
    aic_bic,
    k_from_result,
    run_matrix_subjects,
    select_winner,
    vpc_coverage,
)

_TIMES = np.array([0.5, 1, 2, 3, 4, 6, 8, 12, 24], dtype=float)
_SPEC = CandidateSpec(model_key="oral_1cmt", iiv_params=["CL", "V"])


@pytest.fixture(scope="module")
def subjects():
    """Five synthetic oral_1cmt subjects with lognormal IIV + small residual noise."""
    rng = np.random.default_rng(42)
    model = get_model("oral_1cmt")
    base = dict(model.defaults)
    out = []
    for sid in range(1, 6):
        p = {k: base[k] * float(np.exp(rng.normal(0, 0.2))) for k in base}
        cp = simulate(model, p, [{"time": 0.0, "amt": 100.0}], _TIMES)["cp"]
        obs = cp * np.exp(rng.normal(0, 0.05, size=cp.size))
        out.append({"subject": f"S{sid}", "doses": [{"time": 0.0, "amt": 100.0}],
                    "obs_t": _TIMES.tolist(), "obs_c": obs.tolist(), "wt": 70.0})
    return out


@pytest.fixture(scope="module")
def native(subjects):
    return PharmAgentAdapter().fit(_SPEC, subjects)


# ── fit-based ──────────────────────────────────────────────────────────────
def test_native_field_map_matches_native_keys(native):
    r = native.raw
    assert native.params == r["theta"]
    assert native.ofv == r["ofv"]
    assert native.converged == bool(r["converged"])
    assert native.n_obs == r["n_obs"]
    assert native.engine == "pharmagent_focei"
    if native.ofv is not None:
        assert native.aic == pytest.approx(r["ofv"] + 2 * k_from_result(r))


def test_aic_bic_absent_natively_but_derived(native):
    # The native NLME result carries no aic/bic key; the adapter derives them.
    assert "aic" not in native.raw and "bic" not in native.raw
    if native.ofv is not None:
        assert native.aic is not None and native.bic is not None


def test_oracle_mock_scores_best(subjects):
    adapters = [MockEngineAdapter("oracle", bias=0.0),
                MockEngineAdapter("worse", bias=0.4)]
    matrix = run_matrix_subjects(subjects, [_SPEC], adapters)
    sel = select_winner(matrix["results"])
    assert sel["winner"].engine == "oracle"
    worse = next(r for r in matrix["results"] if r.engine == "worse")
    assert sel["winner"].pred_rmse <= worse.pred_rmse


def test_runner_determinism_fixed_seed(subjects):
    a = run_matrix_subjects(subjects, [_SPEC], [MockEngineAdapter("m")], seed=20250614)
    b = run_matrix_subjects(subjects, [_SPEC], [MockEngineAdapter("m")], seed=20250614)
    assert a["results"][0].pred_rmse == b["results"][0].pred_rmse
    assert a["results"][0].vpc_coverage90 == b["results"][0].vpc_coverage90


# ── selection invariants (hand-constructed, fast) ──────────────────────────
def _res(engine, *, ofv=None, aic=None, rmse=None, cov=None, r2=None, bias=0.0,
         converged=True, status="ok", model="oral_1cmt"):
    return EngineResult(engine=engine, model_name=model, converged=converged,
                        status=status, ofv=ofv, aic=aic, pred_rmse=rmse,
                        vpc_coverage90=cov, pred_r2=r2, pred_bias=bias)


def test_selection_never_ranks_across_engines_by_ofv():
    # A has the far lower OFV but WORSE predictions; B must win.
    a = _res("A", ofv=10.0, aic=20.0, rmse=0.90, cov=0.30)
    b = _res("B", ofv=1000.0, aic=2000.0, rmse=0.20, cov=0.95)
    sel = select_winner([a, b])
    assert sel["winner"].engine == "B"


def test_within_engine_likelihood_is_bucketed_by_engine():
    rs = [_res("A", ofv=5, aic=9, rmse=0.5, model="m1"),
          _res("A", ofv=4, aic=7, rmse=0.6, model="m2"),
          _res("B", ofv=100, aic=110, rmse=0.4, model="m1")]
    within = select_winner(rs)["within_engine_likelihood"]
    assert set(within) == {"A", "B"}
    # each bucket AIC-ascending; no bucket mixes engines
    assert [x["aic"] for x in within["A"]] == [7, 9]
    assert len(within["B"]) == 1


def test_tiebreak_uses_vpc_coverage():
    lo = _res("lo", rmse=0.50, cov=0.60)
    hi = _res("hi", rmse=0.50, cov=0.90)
    assert select_winner([lo, hi])["winner"].engine == "hi"


def test_graceful_skip_absent_engine(subjects):
    class Absent:
        name = "monolix"
        def available(self):  # noqa: E301,E704
            return False
        def fit(self, spec, subs, *, seed=0):  # pragma: no cover
            raise AssertionError("absent engine must not be called")

    matrix = run_matrix_subjects(subjects, [_SPEC],
                                 [Absent(), MockEngineAdapter("m")])
    statuses = {r.engine: r.status for r in matrix["results"]}
    assert statuses["monolix"] == "absent"
    assert select_winner(matrix["results"])["winner"] is not None


def test_failed_engine_becomes_data_not_exception(subjects):
    class Boom:
        name = "boom"
        def available(self):  # noqa: E301,E704
            return True
        def fit(self, spec, subs, *, seed=0):
            raise RuntimeError("engine exploded")

    matrix = run_matrix_subjects(subjects, [_SPEC],
                                 [Boom(), MockEngineAdapter("m")])
    boom = next(r for r in matrix["results"] if r.engine == "boom")
    assert boom.status == "failed" and "exploded" in boom.message
    # matrix still completed with the good engine
    assert any(r.engine == "m" and r.status == "ok" for r in matrix["results"])


def test_nlmixr2_adapter_honours_availability(subjects):
    # Environment-dependent: absent without R/nlmixr2, a real result when present.
    ad = Nlmixr2Adapter()
    matrix = run_matrix_subjects(subjects, [_SPEC], [ad])
    r = matrix["results"][0]
    if ad.available():
        assert r.status in ("ok", "failed")
    else:
        assert r.status == "absent" and matrix["n_available"] == 0


def test_score_from_population_tolerates_partial_omega(subjects):
    # iiv_params is a superset of omega_cv_pct keys (V has no reported IIV).
    # Must NOT silently fall back to typical for every subject (the map_estimate
    # floor fix); n_map_fallback stays 0 and predictions are real.
    from app.compute.pk_models import get_model
    from app.engines.scoring import score_from_population
    theta = dict(get_model("oral_1cmt").defaults)
    sc = score_from_population(
        "oral_1cmt", subjects, theta=theta, omega_cv_pct={"CL": 30.0},
        sigma_prop=0.1, sigma_add=None, iiv_params=["CL", "V"])
    assert sc["n_map_fallback"] == 0
    assert sc["pred_rmse"] is not None


def test_nlmixr2_r_model_injects_allometric_wt():
    from app.compute.pk_models import get_model
    from app.engines.nlmixr2 import _build_r_script
    m = get_model("oral_1cmt")
    script = _build_r_script("oral_1cmt", ["CL", "V"], dict(m.defaults), "focei",
                             allometric=dict(m.allometric or {}))
    assert "(WT/70.0)^0.75" in script   # CL allometric exponent
    assert "(WT/70.0)^1.0" in script    # V allometric exponent
    assert "converged=conv" in script   # real convergence signal, not hardcoded TRUE


def test_dataset_io_writes_nonmem_layout():
    from app.engines.dataset_io import subjects_to_records
    subs = [{"subject": "S1", "doses": [{"time": 0.0, "amt": 100.0}],
             "obs_t": [1.0, 2.0], "obs_c": [5.0, 3.0], "wt": 70.0}]
    rows = subjects_to_records(subs, is_iv=False)
    dose = [r for r in rows if r["EVID"] == 1]
    obs = [r for r in rows if r["EVID"] == 0]
    assert len(dose) == 1 and dose[0]["AMT"] == 100.0 and dose[0]["CMT"] == 1
    assert len(obs) == 2 and obs[0]["CMT"] == 2 and obs[0]["DV"] == 5.0
    # IV: observation compartment is the central (1), not depot (2)
    assert subjects_to_records(subs, is_iv=True)[1]["CMT"] == 1


def test_vpc_coverage_unit_interval_and_none():
    ok = {"status": "ok", "bins": [
        {"obs_p50": 5.0, "sim_med_lo": 4.0, "sim_med_hi": 6.0},   # hit
        {"obs_p50": 9.0, "sim_med_lo": 4.0, "sim_med_hi": 6.0},   # miss
    ]}
    assert vpc_coverage(ok) == 0.5
    assert vpc_coverage({"status": "empty", "bins": []}) is None


def test_vpc_coverage_ignores_none_bins_without_crashing():
    # A status='ok' pcVPC can carry degenerate bins with None fields.
    pc = {"status": "ok", "bins": [
        {"obs_p50": 5.0, "sim_med_lo": 4.0, "sim_med_hi": 6.0},   # hit (assessable)
        {"obs_p50": None, "sim_med_lo": None, "sim_med_hi": 6.0},  # un-assessable
    ]}
    assert vpc_coverage(pc) == 1.0  # 1 hit / 1 assessable, no TypeError
    all_none = {"status": "ok", "bins": [
        {"obs_p50": None, "sim_med_lo": None, "sim_med_hi": None}]}
    assert vpc_coverage(all_none) is None


def test_selection_metric_documents_no_ofv():
    sel = select_winner([_res("A", rmse=0.3)])
    assert "within-engine only" in sel["note"]


def test_to_audit_dict_strips_raw():
    r = EngineResult(engine="A", raw={"huge": [1, 2, 3]}, pred_rmse=0.3)
    d = r.to_audit_dict()
    assert "raw" not in d and d["engine"] == "A" and d["pred_rmse"] == 0.3


def test_k_from_result_counts_categorical_levels():
    base = {"theta": {"CL": 5, "V": 50, "KA": 1}, "omega_cv_pct": {"CL": 30, "V": 20},
            "sigma": {"prop": 0.1, "add": None}, "n_obs": 45, "ofv": 100.0}
    assert k_from_result(base) == 3 + 2 + 1  # theta + omega + sigma_prop
    # nlme.py emits `levels` as the NON-reference set: a 3-level covariate -> 2
    # levels -> 2 estimated coefficients.
    base_cat = {**base, "covariate_effects": [
        {"param": "CL", "covariate": "SEX", "kind": "categorical", "levels": ["b", "c"]}]}
    assert k_from_result(base_cat) == 6 + 2  # +2 coefficients (2 non-ref levels)
    aic, bic = aic_bic(base_cat)
    assert aic == pytest.approx(100.0 + 2 * 8)


def test_k_from_result_binary_categorical_single_level():
    # A binary covariate has ONE non-reference level -> one coefficient.
    r = {"theta": {"CL": 5}, "omega_cv_pct": {"CL": 30}, "sigma": {"prop": 0.1, "add": None},
         "n_obs": 30, "ofv": 50.0,
         "covariate_effects": [{"kind": "categorical", "levels": ["M"]}]}
    assert k_from_result(r) == 1 + 1 + 1 + 1  # theta + omega + sigma_prop + 1 cat coef


def test_ensemble_consensus_of_converged_engines(subjects):
    # Ensemble = geometric mean of the converged fits' params, scored on the same
    # footing; competes in the ranking. Needs >=2 usable engines.
    from app.engines import build_ensemble
    from app.engines.runner import run_matrix_subjects
    matrix = run_matrix_subjects(subjects, [_SPEC],
                                 [MockEngineAdapter("a", bias=0.0),
                                  MockEngineAdapter("b", bias=0.3)])
    ens = build_ensemble(matrix["results"], "oral_1cmt", subjects)
    assert ens is not None and ens.engine == "ensemble" and ens.converged
    # geometric mean of the two members' CL lies between them
    cls = sorted(r.params["CL"] for r in matrix["results"])
    assert cls[0] <= ens.params["CL"] <= cls[1]
    assert ens.pred_rmse is not None and set(ens.raw["members"]) == {"a", "b"}


def test_ensemble_needs_at_least_two_engines(subjects):
    from app.engines import build_ensemble
    from app.engines.runner import run_matrix_subjects
    one = run_matrix_subjects(subjects, [_SPEC], [MockEngineAdapter("solo")])
    assert build_ensemble(one["results"], "oral_1cmt", subjects) is None


def test_ensemble_aggregates_covariate_effects():
    # Consensus covariate effects: continuous coefficients averaged (arithmetic,
    # they are exponents/slopes), categorical {level: coef} averaged per level.
    from app.engines.ensemble import _aggregate_cov_effects
    a = EngineResult(engine="a", covariate_effects=[
        {"param": "CL", "covariate": "WT", "kind": "power", "coefficient": 0.6},
        {"param": "V", "covariate": "SEX", "kind": "categorical", "coefficient": {"F": 0.2}}])
    b = EngineResult(engine="b", covariate_effects=[
        {"param": "CL", "covariate": "WT", "kind": "power", "coefficient": 1.0},
        {"param": "V", "covariate": "SEX", "kind": "categorical", "coefficient": {"F": 0.4}}])
    agg = {(e["param"], e["covariate"]): e["coefficient"]
           for e in _aggregate_cov_effects([a, b])}
    assert agg[("CL", "WT")] == pytest.approx(0.8)       # mean(0.6, 1.0)
    assert agg[("V", "SEX")]["F"] == pytest.approx(0.3)  # per-level mean(0.2, 0.4)
    # no covariate effects -> empty aggregate
    assert _aggregate_cov_effects([EngineResult(engine="c")]) == []
