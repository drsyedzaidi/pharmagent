"""Censoring (CENS / BLQ) handling is explicit and consistent across every
`_build_subjects` caller (PMxbench gap G5).

No real NLME/SCM fit runs here: the estimators are monkeypatched with fakes
that wrap the subjects in the real `nlme._Subject`, so `n_blq` is exactly what
a real fit would count on the subjects the tool hands over.
"""
import numpy as np
import pandas as pd
import pytest

import app.compute.nlme as nlme
from app.core.pharmstate import PharmState
from app.tools import pkmodel_tools
from app.tools.base import ToolContext
from app.tools.data_tools import profile_pk_dataset
from app.tools.pkmodel_tools import (
    fit_pk_model,
    run_covariate_forest,
    run_exposure_forest,
    run_individual_exposures,
    run_nlme,
    run_prior_check,
    run_scm,
    run_vpc,
)

MODEL_KEY = "oral_1cmt"
LLOQ = 0.05
N_POSTDOSE_CENS = 2


def _dataset() -> pd.DataFrame:
    """NONMEM convention: CENS=1 rows carry the LLOQ in DV. Two post-dose
    censored records plus one predose censored record (t=0, never fitted)."""
    rows = []
    for sid in range(1, 6):
        rows.append({"ID": sid, "TIME": 0.0, "DV": np.nan, "AMT": 100.0, "CENS": 0,
                     "AGE": 30 + 5 * sid})
        if sid == 3:
            rows.append({"ID": sid, "TIME": 0.0, "DV": LLOQ, "AMT": np.nan, "CENS": 1,
                         "AGE": 30 + 5 * sid})
        for t, c in zip([0.5, 1, 2, 4, 8, 12, 24], [0.8, 1.4, 1.2, 0.9, 0.5, 0.2, 0.1]):
            dv, cens = c * (1 + 0.03 * sid), 0
            if sid in (1, 2) and t == 24:
                dv, cens = LLOQ, 1
            rows.append({"ID": sid, "TIME": t, "DV": dv, "AMT": np.nan, "CENS": cens,
                         "AGE": 30 + 5 * sid})
    return pd.DataFrame(rows)


ROLES = {"ID": "ID", "TIME": "TIME", "DV": "DV", "AMT": "AMT", "CENS": "CENS"}


def _loaded():
    ctx = ToolContext(dataset_store={"d1": _dataset()})
    state = PharmState(dataset_id="d1", dataset_metadata={"detected_roles": ROLES})
    return state, ctx


def _n_blq(subjects: list[dict]) -> int:
    return int(sum(int(s.blq.sum()) for s in nlme._prepare_subjects(subjects)))


def test_run_scm_uses_m3_like_the_base_nlme_fit(monkeypatch):
    seen: dict = {}

    def fake_population_fit(model_key, subjects, **kw):
        return {"method": "FOCE-I", "label": "1-cmt oral", "ofv": 1.0,
                "n_subjects": len(subjects), "iiv_params": ["CL", "V"],
                "converged": True, "theta": {}, "omega_cv_pct": {}, "sigma": {},
                "n_blq": _n_blq(subjects)}

    def fake_scm(model_key, subjects, **kw):
        seen["subjects"] = subjects
        return {"status": "ok", "model_key": model_key, "label": "1-cmt oral",
                "base_ofv": 1.0, "final_ofv": 1.0, "selected": [], "steps": [],
                "n_candidates": 1, "final": {"n_blq": _n_blq(subjects)}}

    monkeypatch.setattr(nlme, "population_fit", fake_population_fit)
    monkeypatch.setattr(nlme, "scm", fake_scm)
    state, ctx = _loaded()
    base = run_nlme(state, ctx, {"model_key": MODEL_KEY}).writes["nlme_results"]
    res = run_scm(state, ctx, {"model_key": MODEL_KEY,
                               "candidates": [{"param": "CL", "covariate": "AGE"}]})
    final = res.writes["scm_results"]["final"]
    assert base["n_blq"] == N_POSTDOSE_CENS
    assert final["n_blq"] == base["n_blq"]
    # the LLOQ is carried as the censoring limit, not as a quantified value
    assert {s.get("lloq") for s in seen["subjects"] if s.get("obs_blq")
            and any(s["obs_blq"])} == {LLOQ}


# --- M1 for the estimators without M3 ----------------------------------------

N_QUANTIFIED = 5 * 7 - N_POSTDOSE_CENS


def _dv0_dataset() -> pd.DataFrame:
    """Same records, censored rows coded DV=0 instead of DV=LLOQ."""
    df = _dataset()
    return df.assign(DV=df["DV"].where(df["CENS"] != 1, 0.0))


def _obs(subjects: list[dict]) -> list[float]:
    return [float(c) for s in subjects for c in s["obs_c"]]


def test_m1_drops_censored_records_and_counts_them():
    subjects, _m, _p, n = pkmodel_tools._build_subjects_counted(_dataset(), ROLES)
    assert n == N_POSTDOSE_CENS                   # the t=0 predose record is never counted
    assert len(_obs(subjects)) == N_QUANTIFIED
    assert LLOQ not in _obs(subjects)             # never scored at the LLOQ
    # the 3-tuple wrapper applies the same M1 exclusion
    assert _obs(pkmodel_tools._build_subjects(_dataset(), ROLES)[0]) == _obs(subjects)


def test_m1_count_matches_the_m3_flags():
    m3, _m, _p = pkmodel_tools._build_subjects(_dataset(), ROLES, with_blq=True)
    _s, _m2, _p2, n = pkmodel_tools._build_subjects_counted(_dataset(), ROLES)
    assert n == sum(sum(s["obs_blq"]) for s in m3)


def test_m1_does_not_depend_on_the_dv_carried_by_censored_rows():
    a, _m, _p, na = pkmodel_tools._build_subjects_counted(_dataset(), ROLES)
    b, _m2, _p2, nb = pkmodel_tools._build_subjects_counted(_dv0_dataset(), ROLES)
    assert na == nb == N_POSTDOSE_CENS
    assert _obs(a) == _obs(b)


def test_no_cens_column_reports_none():
    roles = {k: v for k, v in ROLES.items() if v != "CENS"}
    df = _dataset().drop(columns=["CENS"])
    subjects, _m, _p, n = pkmodel_tools._build_subjects_counted(df, roles)
    assert n is None
    assert len(_obs(subjects)) == 5 * 7           # no censoring information to act on


def _fake_fit(captured: dict):
    def fake(subjects, model_key):
        captured["subjects"] = subjects
        return {"model_key": model_key, "label": "1-cmt oral", "group": "oral",
                "n_subjects": len(subjects), "n_converged": len(subjects),
                "mean_aic": 1.0, "total_aic": 5.0, "population": {"parameters": {}},
                "individual_fits": []}
    return fake


def test_fit_pk_model_fit_mode_uses_m1(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(pkmodel_tools, "fit_pk_dataset", _fake_fit(captured))
    state, ctx = _loaded()
    res = fit_pk_model(state, ctx, {"model_key": MODEL_KEY})
    assert res.writes["pk_model_results"]["n_blq_dropped"] == N_POSTDOSE_CENS
    assert res.result["n_blq_dropped"] == N_POSTDOSE_CENS
    assert "M1" in res.summary
    assert LLOQ not in _obs(captured["subjects"])


def test_fit_pk_model_compare_mode_uses_m1(monkeypatch):
    captured: dict = {}

    def fake_compare(subjects, candidates):
        captured["subjects"] = subjects
        return {"ranking": [], "best_model": None, "best": None}

    monkeypatch.setattr(pkmodel_tools, "compare_models", fake_compare)
    state, ctx = _loaded()
    res = fit_pk_model(state, ctx, {"compare": True, "models": [MODEL_KEY]})
    assert res.writes["pk_model_results"]["n_blq_dropped"] == N_POSTDOSE_CENS
    assert LLOQ not in _obs(captured["subjects"])


def _pk_model_results() -> dict:
    return {
        "status": "ok", "mode": "fit", "model_key": MODEL_KEY,
        "individual_fits": [{"subject": sid, "converged": True,
                             "params": {"CL": 4.0 + 0.1 * sid, "V": 40.0, "KA": 1.2}}
                            for sid in range(1, 6)],
        "population": {"parameters": {
            "CL": {"typical_value": 4.2, "iiv_cv_pct": 25.0},
            "V": {"typical_value": 40.0, "iiv_cv_pct": 20.0},
            "KA": {"typical_value": 1.2, "iiv_cv_pct": 15.0}}},
    }


def test_run_vpc_overlay_uses_m1():
    state, ctx = _loaded()
    state.pk_model_results = _pk_model_results()
    res = run_vpc(state, ctx, {})
    payload = res.writes["vpc_results"]
    assert payload["n_blq_dropped"] == N_POSTDOSE_CENS
    assert res.result["n_blq_dropped"] == N_POSTDOSE_CENS
    assert LLOQ not in payload["obs_c"]
    assert payload["gof"]["n"] == N_QUANTIFIED


def test_run_vpc_blq_check_still_sees_the_censored_records():
    state, ctx = _loaded()
    state.pk_model_results = _pk_model_results()
    blq = run_vpc(state, ctx, {"blq_check": True}).writes["vpc_results"]["blq_vpc"]
    assert blq["n_blq"] == N_POSTDOSE_CENS


def _nlme_fit() -> dict:
    return {
        "status": "ok", "model_key": MODEL_KEY, "label": "1-compartment oral",
        "theta": {"CL": 4.2, "V": 40.0, "KA": 1.2}, "theta_rse_pct": {"CL": 10.0},
        "theta_prior": {"names": ["CL"], "mean_log": [1.4], "cov_log": [[0.04]]},
        "omega_cv_pct": {"CL": 25.0, "V": 20.0}, "iiv_params": ["CL", "V"],
        "sigma": {"prop": 0.15}, "error_model": "proportional",
        "covariate_effects": [{"param": "CL", "covariate": "AGE", "kind": "power",
                               "center": 45.0, "coefficient": 0.5, "rse_pct": 20.0}],
        "individual": [{"subject": sid, "eta": {"CL": 0.01 * sid, "V": 0.0}}
                       for sid in range(1, 6)],
    }


def test_run_prior_check_uses_m1(monkeypatch):
    import app.compute.bayes as bayes
    captured: dict = {}

    def fake_ppc(model_key, **kw):
        captured.update(kw)
        return {"status": "ok", "coverage_pct": 90.0}

    monkeypatch.setattr(bayes, "prior_predictive_check", fake_ppc)
    monkeypatch.setattr(bayes, "prior_posterior_diagnostic", lambda nl: {"mean_shrinkage": 0.2})
    state, ctx = _loaded()
    state.nlme_results = _nlme_fit()
    res = run_prior_check(state, ctx, {})
    assert res.writes["prior_check_results"]["n_blq_dropped"] == N_POSTDOSE_CENS
    assert res.result["n_blq_dropped"] == N_POSTDOSE_CENS
    assert len(captured["obs_conc"]) == N_QUANTIFIED
    assert LLOQ not in captured["obs_conc"]


def test_forest_and_exposure_helpers_report_m1():
    state, ctx = _loaded()
    state.nlme_results = _nlme_fit()
    forest = run_covariate_forest(state, ctx, {"source": "nlme"})
    assert forest.writes["forest_results"]["n_blq_dropped"] == N_POSTDOSE_CENS
    expo = run_exposure_forest(state, ctx, {"n_draws": 20})
    assert expo.writes["exposure_forest_results"]["n_blq_dropped"] == N_POSTDOSE_CENS
    ind = run_individual_exposures(state, ctx, {"dose": 100, "tau": 12, "n_doses": 7})
    assert ind.writes["individual_exposures"]["n_blq_dropped"] == N_POSTDOSE_CENS


# --- profile_pk_dataset BLQ % (G8) --------------------------------------------

def test_profile_blq_pct_comes_from_cens():
    state, ctx = _loaded()
    q = profile_pk_dataset(state, ctx, {}).writes["data_quality"]
    n_obs = int(_dataset()["DV"].notna().sum())
    assert q["n_observations"] == n_obs
    # 3 censored records (2 post-dose + 1 predose); DV carries the LLOQ, so
    # the DV <= 0 rule would report 0 %
    assert q["blq_pct"] == round(100.0 * 3 / n_obs, 2)
    assert q["blq_basis"] == "CENS"


def test_profile_without_cens_keeps_the_dv_rule():
    df = _dv0_dataset().drop(columns=["CENS"])
    roles = {k: v for k, v in ROLES.items() if v != "CENS"}
    ctx = ToolContext(dataset_store={"d1": df})
    state = PharmState(dataset_id="d1", dataset_metadata={"detected_roles": roles})
    q = profile_pk_dataset(state, ctx, {}).writes["data_quality"]
    n_obs = int(df["DV"].notna().sum())
    assert q["blq_pct"] == round(100.0 * 3 / n_obs, 2)
    assert q["blq_basis"] == "DV<=0"


# --- review follow-ups: flag codings, DVID, edge subjects ---------------------

def test_cdisc_y_blank_flags_mean_the_same_as_1_and_0():
    df = _dataset()
    df = df.assign(CENS=df["CENS"].map({1: "Y", 0: ""}))
    _s, _m, _p, n = pkmodel_tools._build_subjects_counted(df, ROLES)
    assert n == N_POSTDOSE_CENS
    m3, _m2, _p2 = pkmodel_tools._build_subjects(df, ROLES, with_blq=True)
    assert sum(sum(s["obs_blq"]) for s in m3) == N_POSTDOSE_CENS
    ctx = ToolContext(dataset_store={"d1": df})
    state = PharmState(dataset_id="d1", dataset_metadata={"detected_roles": ROLES})
    q = profile_pk_dataset(state, ctx, {}).writes["data_quality"]
    assert q["blq_pct"] == round(100.0 * 3 / int(df["DV"].notna().sum()), 2)


def test_an_unknown_cens_code_fails_instead_of_meaning_not_censored():
    df = _dataset()
    df = df.assign(CENS=df["CENS"].astype(object).where(df["CENS"] != 1, "Q"))
    with pytest.raises(ValueError, match="below-LOQ flag"):
        pkmodel_tools._build_subjects_counted(df, ROLES)


def test_fully_censored_subject_is_counted_and_dose_less_subject_is_not():
    df = _dataset()
    extra = [{"ID": 6, "TIME": 0.0, "DV": np.nan, "AMT": 100.0, "CENS": 0, "AGE": 60}]
    extra += [{"ID": 6, "TIME": t, "DV": LLOQ, "AMT": np.nan, "CENS": 1, "AGE": 60}
              for t in (1.0, 2.0)]                                  # every sample censored
    extra += [{"ID": 7, "TIME": t, "DV": LLOQ, "AMT": np.nan, "CENS": 1, "AGE": 65}
              for t in (1.0, 2.0)]                                  # no dose record
    df = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)
    subjects, _m, _p, n = pkmodel_tools._build_subjects_counted(df, ROLES)
    assert n == N_POSTDOSE_CENS + 2
    assert {int(s["subject"]) for s in subjects} == {1, 2, 3, 4, 5}


def _dvid_dataset() -> pd.DataFrame:
    """PK (DVID=1) + PD (DVID=2) records; one censored record per endpoint."""
    rows = []
    for sid in range(1, 4):
        rows.append({"ID": sid, "TIME": 0.0, "DV": np.nan, "AMT": 100.0, "DVID": 1, "CENS": 0})
        for t, c, e in zip([1, 2, 4, 8], [1.4, 1.2, 0.9, 0.5], [10.0, 9.0, 7.0, 5.0]):
            pk_cens = int(sid == 1 and t == 8)
            pd_cens = int(sid == 2 and t == 8)
            rows.append({"ID": sid, "TIME": t, "DV": LLOQ if pk_cens else c, "AMT": np.nan,
                         "DVID": 1, "CENS": pk_cens})
            rows.append({"ID": sid, "TIME": t, "DV": 1.0 if pd_cens else e, "AMT": np.nan,
                         "DVID": 2, "CENS": pd_cens})
    return pd.DataFrame(rows)


DVID_ROLES = {**ROLES, "DVID": "DVID"}


def test_dvid_path_applies_m1_to_both_endpoints():
    for with_blq in (False, True):                     # no M3 on DVID data either way
        subjects, _m, has_pd, n = pkmodel_tools._build_subjects_counted(
            _dvid_dataset(), DVID_ROLES, with_blq=with_blq)
        assert has_pd and n == 2
        assert LLOQ not in _obs(subjects)
        assert 1.0 not in [float(e) for s in subjects for e in s.get("pd_e", [])]
        assert sum(len(s["obs_c"]) for s in subjects) == 3 * 4 - 1
        assert sum(len(s.get("pd_e", [])) for s in subjects) == 3 * 4 - 1


def test_run_nlme_reports_m1_on_dvid_data(monkeypatch):
    monkeypatch.setattr(nlme, "population_fit", lambda model_key, subjects, **kw: {
        "method": "FOCE-I", "label": "1-cmt oral", "ofv": 1.0, "n_subjects": len(subjects),
        "iiv_params": ["CL", "V"], "converged": True, "theta": {}, "omega_cv_pct": {},
        "sigma": {}, "n_blq": _n_blq(subjects)})
    ctx = ToolContext(dataset_store={"d1": _dvid_dataset()})
    state = PharmState(dataset_id="d1", dataset_metadata={"detected_roles": DVID_ROLES})
    res = run_nlme(state, ctx, {"model_key": MODEL_KEY})
    assert res.writes["nlme_results"]["n_blq_dropped"] == 2
    assert "M1" in res.summary


def test_run_nlme_m3_payload_has_no_m1_key(monkeypatch):
    monkeypatch.setattr(nlme, "population_fit", lambda model_key, subjects, **kw: {
        "method": "FOCE-I", "label": "1-cmt oral", "ofv": 1.0, "n_subjects": len(subjects),
        "iiv_params": ["CL", "V"], "converged": True, "theta": {}, "omega_cv_pct": {},
        "sigma": {}, "n_blq": _n_blq(subjects)})
    state, ctx = _loaded()
    assert "n_blq_dropped" not in run_nlme(state, ctx, {"model_key": MODEL_KEY}).writes["nlme_results"]


def test_fit_pk_model_names_m1_when_it_removes_every_observation():
    df = _dataset()
    df = df.assign(CENS=(df["TIME"] > 0).astype(int))
    ctx = ToolContext(dataset_store={"d1": df})
    state = PharmState(dataset_id="d1", dataset_metadata={"detected_roles": ROLES})
    res = fit_pk_model(state, ctx, {"model_key": MODEL_KEY})
    assert res.writes["pk_model_results"]["status"] == "no_subjects"
    assert res.writes["pk_model_results"]["n_blq_dropped"] == 5 * 7
    assert "M1" in res.summary
