"""Adversarial reviewer: independent refutation engine + loop driver.

Covers:
  - clean NCA state  → goal met, no CRITICAL/HIGH
  - AUCinf < AUClast → HIGH (internal consistency)
  - CL/F <= 0        → CRITICAL
  - CL/F vs dose/AUCinf inconsistency → HIGH
  - raw-data Cmax disagreement (units flip) → CRITICAL recompute finding
  - AUClast outside Cmax*Tlast band → CRITICAL recompute finding
  - NLME over-param (high condition number / shrinkage) → findings
  - loop driver converges + reports goal status
  - HTTP endpoint POST /review returns findings + goal verdict
"""
from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.compute import adversarial
from app.core.pharmstate import PharmState

# ── fixtures ────────────────────────────────────────────────────────────────────

_ROLES = {"ID": "ID", "TIME": "TIME", "DV": "DV", "AMT": "AMT", "EVID": "EVID"}


def _clean_nca() -> list[dict]:
    # CL/F == dose/AUCinf == 100/3200 ≈ 0.03125 (use consistent numbers)
    return [{"subject": 1, "dose": 100.0, "Cmax": 4.5, "Tmax": 1.0, "Tlast": 24.0,
             "AUC_last": 30.0, "AUC_inf": 32.0, "t_half": 8.0,
             "CL_F": 100.0 / 32.0, "Vz_F": 36.0,
             "lambda_z_r2_adj": 0.97, "lambda_z_n_points": 5, "pct_AUC_extrap": 6.0}]


def _raw_df() -> pd.DataFrame:
    # subject 1: peak conc 4.5 at t=1, matches Cmax above
    rows = [
        {"ID": 1, "TIME": 0.0, "DV": 0.0, "AMT": 100.0, "EVID": 1},
        {"ID": 1, "TIME": 0.5, "DV": 3.0, "AMT": 0, "EVID": 0},
        {"ID": 1, "TIME": 1.0, "DV": 4.5, "AMT": 0, "EVID": 0},
        {"ID": 1, "TIME": 4.0, "DV": 3.2, "AMT": 0, "EVID": 0},
        {"ID": 1, "TIME": 12.0, "DV": 1.4, "AMT": 0, "EVID": 0},
        {"ID": 1, "TIME": 24.0, "DV": 0.5, "AMT": 0, "EVID": 0},
    ]
    return pd.DataFrame(rows)


def _ids(findings) -> set[str]:
    return {f["id"] for f in findings}


def _sev(findings, sev) -> list[dict]:
    return [f for f in findings if f["severity"] == sev]


# ── engine: clean ───────────────────────────────────────────────────────────────

def test_clean_state_meets_goal():
    st = PharmState(nca_parameters=_clean_nca()).model_dump()
    r = adversarial.review(st, _raw_df(), _ROLES)
    assert r["goal_met"] is True
    assert r["counts"]["CRITICAL"] == 0 and r["counts"]["HIGH"] == 0


# ── engine: state-internal refutations ──────────────────────────────────────────

def test_aucinf_below_auclast_is_high():
    p = _clean_nca()
    p[0]["AUC_inf"] = 25.0  # below AUC_last 30 → impossible
    r = adversarial.review(PharmState(nca_parameters=p).model_dump(), None, None)
    assert any(f["id"].startswith("nca-aucorder") and f["severity"] == "HIGH"
               for f in r["findings"])
    assert r["goal_met"] is False


def test_nonpositive_clf_is_critical():
    p = _clean_nca()
    p[0]["CL_F"] = -1.0
    r = adversarial.review(PharmState(nca_parameters=p).model_dump(), None, None)
    assert _sev(r["findings"], "CRITICAL")
    assert r["goal_met"] is False


def test_clf_inconsistent_with_dose_over_aucinf_is_high():
    p = _clean_nca()
    p[0]["CL_F"] = 0.5  # but dose/AUCinf = 100/32 ≈ 3.125 → big mismatch
    r = adversarial.review(PharmState(nca_parameters=p).model_dump(), None, None)
    assert any(f["id"].startswith("nca-clf-consist") for f in r["findings"])


def test_extrap_out_of_range_is_high():
    p = _clean_nca()
    p[0]["pct_AUC_extrap"] = 140.0
    r = adversarial.review(PharmState(nca_parameters=p).model_dump(), None, None)
    assert any(f["id"].startswith("nca-extrap") for f in r["findings"])


# ── engine: raw-data recomputation (the real independence) ──────────────────────

def test_cmax_disagreement_is_critical_recompute():
    p = _clean_nca()
    p[0]["Cmax"] = 4500.0  # units flip: reported ng/mL but data is µg/mL → 1000x
    r = adversarial.review(PharmState(nca_parameters=p).model_dump(), _raw_df(), _ROLES)
    crit = _sev(r["findings"], "CRITICAL")
    assert any(f["id"].startswith("recompute-cmax") for f in crit)
    assert r["goal_met"] is False


def test_auc_band_violation_is_critical_recompute():
    p = _clean_nca()
    p[0]["AUC_last"] = 30000.0  # absurd vs Cmax*Tlast = 4.5*24 = 108
    r = adversarial.review(PharmState(nca_parameters=p).model_dump(), _raw_df(), _ROLES)
    assert any(f["id"].startswith("recompute-aucband") for f in r["findings"])


def test_recompute_skipped_without_raw_data_but_internal_still_runs():
    p = _clean_nca()
    p[0]["Cmax"] = 4500.0          # would be caught by recompute…
    p[0]["CL_F"] = -1.0            # …but this internal one fires without raw data
    r = adversarial.review(PharmState(nca_parameters=p).model_dump(), None, None)
    assert r["checked"]["nca_recompute"] is False
    assert _sev(r["findings"], "CRITICAL")  # the CL/F sign finding


# ── engine: NLME refutations ────────────────────────────────────────────────────

def test_nlme_overparam_flags():
    nl = {"status": "ok", "converged": True, "condition_number": 5000.0,
          "omega_cv_pct": {"CL": 220.0}, "shrinkage_pct": {"CL": 45.0},
          "theta_rse_pct": {"CL": 80.0}}
    r = adversarial.review(PharmState(nlme_results=nl).model_dump(), None, None)
    ids = _ids(r["findings"])
    assert "nlme-cond" in ids
    assert any(i.startswith("nlme-iiv") for i in ids)
    assert any(i.startswith("nlme-shrink") for i in ids)
    assert r["goal_met"] is False     # condition number is HIGH


def test_nlme_not_converged_is_high():
    nl = {"status": "ok", "converged": False}
    r = adversarial.review(PharmState(nlme_results=nl).model_dump(), None, None)
    assert any(f["id"] == "nlme-converge" and f["severity"] == "HIGH"
               for f in r["findings"])


# ── engine: dose monotonicity ───────────────────────────────────────────────────

def test_dose_exposure_non_monotonic_is_medium():
    summary = {"by_dose": [
        {"dose": 100.0, "AUC_inf_geomean": 50.0},
        {"dose": 300.0, "AUC_inf_geomean": 40.0},   # falls as dose rises
    ]}
    r = adversarial.review(PharmState(nca_summary=summary).model_dump(), None, None)
    assert any(f["id"].startswith("nca-monotonic") for f in r["findings"])
    assert r["counts"]["MEDIUM"] >= 1


def test_findings_sorted_by_severity():
    p = _clean_nca()
    p[0]["CL_F"] = -1.0            # CRITICAL
    p[0]["AUC_inf"] = 25.0         # HIGH (below AUClast 30)
    r = adversarial.review(PharmState(nca_parameters=p).model_dump(), None, None)
    sevs = [f["severity"] for f in r["findings"]]
    assert sevs == sorted(sevs, key=lambda s: adversarial._RANK[s])


# ── HTTP endpoint + loop ────────────────────────────────────────────────────────

@pytest.fixture()
def client():
    from app.main import app
    return TestClient(app)


def _seed_session_with_nca(client, params, summary=None):
    sid = client.post("/api/sessions").json()["id"]
    # write NCA params straight into state via a direct tool isn't exposed; use the
    # orchestrator the app holds.
    from app.core.pharmstate import apply_writes
    from app.main import orch
    sess = orch.get_session(sid)
    writes = {"nca_parameters": params}
    if summary is not None:
        writes["nca_summary"] = summary
    sess.state = apply_writes(sess.state, "nca", writes)
    orch._persist(sess)
    return sid


def test_http_review_clean_meets_goal(client):
    sid = _seed_session_with_nca(client, _clean_nca())
    resp = client.post(f"/api/sessions/{sid}/review", json={})
    assert resp.status_code == 200
    body = resp.json()
    assert body["goal_met"] is True
    assert body["counts"]["CRITICAL"] == 0 and body["counts"]["HIGH"] == 0
    assert body["iterations"] >= 1


def test_http_review_flags_and_blocks_goal(client):
    bad = _clean_nca()
    bad[0]["CL_F"] = -5.0
    sid = _seed_session_with_nca(client, bad)
    body = client.post(f"/api/sessions/{sid}/review", json={"goal": "no blockers"}).json()
    assert body["goal_met"] is False
    assert any(f["severity"] == "CRITICAL" for f in body["findings"])
    assert body["goal"] == "no blockers"


def test_http_review_writes_state(client):
    sid = _seed_session_with_nca(client, _clean_nca())
    client.post(f"/api/sessions/{sid}/review", json={})
    state = client.get(f"/api/sessions/{sid}/state").json()
    assert state["review_results"] is not None
    assert "findings" in state["review_results"]


def test_loop_stops_at_max_iter_without_progress(client):
    bad = _clean_nca()
    bad[0]["CL_F"] = -5.0          # deterministic, never auto-resolves
    sid = _seed_session_with_nca(client, bad)
    body = client.post(f"/api/sessions/{sid}/review", json={"max_iter": 5}).json()
    # no remediation → no progress → loop stops early (<= 2 passes), not all 5
    assert body["iterations"] <= 2
    assert body["goal_met"] is False


# ── modeling-state refutations (pk_model_results + engine_comparison_results) ──
def test_modeling_review_inspects_pkmodel_and_engine():
    st = PharmState(
        pk_model_results={"status": "ok", "mode": "compare",
                          "ranking": [{"aic": 100.0}, {"aic": 130.0}],
                          "best": {"label": "2-cmt oral", "n_subjects": 5, "n_converged": 5}},
        engine_comparison_results={"status": "ok",
            "winner": {"engine": "pharmagent_focei"}, "n_available": 2,
            "prediction_ranking": [{"engine": "pharmagent_focei"}]},  # only 1 engine fit
    ).model_dump()
    r = adversarial.review(st, None, None)
    assert r["checked"]["pkmodel"] and r["checked"]["engine"]
    assert "engine-single" in {f["id"] for f in r["findings"]}   # single engine fit the winner
    assert r["goal_met"] is True                                 # MEDIUM is non-blocking


def test_modeling_review_flags_nonconverged_best_model():
    st = PharmState(pk_model_results={"status": "ok", "mode": "compare",
        "ranking": [{"aic": 100.0}, {"aic": 130.0}],
        "best": {"label": "2-cmt oral", "n_subjects": 6, "n_converged": 3}}).model_dump()
    r = adversarial.review(st, None, None)
    assert any(f["id"] == "pkmodel-converge" and f["severity"] == "HIGH" for f in r["findings"])
    assert r["goal_met"] is False                                # HIGH is blocking


def test_empty_review_fails_closed():
    r = adversarial.review(PharmState().model_dump(), None, None)
    assert r["nothing_checked"] is True
    assert r["goal_met"] is False                                # nothing inspected -> not a pass


def test_scm_stepwise_selection_flags_postselection_inference():
    st = PharmState(scm_results={"status": "ok", "label": "1-cmt oral",
        "base_ofv": 100.0, "final_ofv": 88.0,
        "selected": [{"param": "CL", "covariate": "WT", "kind": "power"}]}).model_dump()
    r = adversarial.review(st, None, None)
    assert r["checked"]["scm"] is True
    f = next(x for x in r["findings"] if x["id"] == "scm-postselection")
    assert f["severity"] == "MEDIUM"
    assert r["goal_met"] is True   # informational, non-blocking


def test_scm_with_nothing_selected_raises_no_postselection_finding():
    st = PharmState(scm_results={"status": "ok", "label": "1-cmt oral",
        "base_ofv": 100.0, "final_ofv": 100.0, "selected": []}).model_dump()
    r = adversarial.review(st, None, None)
    assert not any(x["id"] == "scm-postselection" for x in r["findings"])
