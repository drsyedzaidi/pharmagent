"""Tool-wiring tests for run_covariate_forest.

Covers the single-provenance source selection (auto/nlme/scm), the SCM
`final`-has-no-`status`-key wrapping fix (an SCM-sourced forest must not
silently report `no_fit`), categorical reference-level recovery from the
dataset, continuous percentile computation, and JSON safety — without running
a real fit.
"""
import json

import numpy as np
import pandas as pd
import pytest

from app.core.pharmstate import AGENT_WRITE_FIELDS, PharmState, apply_writes
from app.tools.base import ToolContext
from app.tools.builtins import default_registry
from app.tools.pkmodel_tools import run_covariate_forest

MODEL_KEY = "oral_1cmt"


def _dataset(n: int = 20) -> pd.DataFrame:
    rows = []
    for sid in range(1, n + 1):
        wt = 50.0 + 2.0 * sid
        sex = "F" if sid % 3 == 0 else "M"  # M is modal (majority) level
        rows.append({"ID": sid, "TIME": 0.0, "DV": np.nan, "AMT": 100.0, "WT": wt, "SEX": sex})
        for t, c in zip([0.5, 1, 2, 4, 8, 12], [0.8, 1.4, 1.2, 0.9, 0.5, 0.2]):
            rows.append({"ID": sid, "TIME": t, "DV": c * (1 + 0.01 * sid), "AMT": np.nan,
                        "WT": wt, "SEX": sex})
    return pd.DataFrame(rows)


def _roles() -> dict:
    return {"ID": "ID", "TIME": "TIME", "DV": "DV", "AMT": "AMT"}


def _nlme_with_covariates() -> dict:
    return {
        "status": "ok", "model_key": MODEL_KEY, "label": "1-compartment oral",
        "theta": {"CL": 4.2, "V": 40.0, "KA": 1.2},
        "omega_cv_pct": {"CL": 25.0, "V": 20.0},
        "sigma": {"prop": 0.15, "add": 0.3},
        "covariate_effects": [
            {"param": "CL", "covariate": "WT", "kind": "power", "center": 70.0,
             "coefficient": 0.75, "rse_pct": 12.0, "levels": None, "description": "..."},
        ],
    }


def _scm_outer(*, with_selection: bool = True) -> dict:
    # `final` deliberately carries NO "status" key -- app.compute.nlme.scm()'s
    # `final = fit_one(...)` is a raw `_assemble()` result, which never emits
    # one; only the OUTER scm() dict below does. This is the exact shape the
    # SCM-source wrapping fix in run_covariate_forest must handle.
    final = {
        "model_key": MODEL_KEY, "label": "1-compartment oral",
        "theta": {"CL": 4.0, "V": 42.0, "KA": 1.1},
        "omega_cv_pct": {"CL": 22.0, "V": 18.0},
        "sigma": {"prop": 0.12, "add": 0.25},
        "covariate_effects": ([{"param": "CL", "covariate": "SEX", "kind": "categorical",
                                "levels": ["F"], "coefficient": {"F": -0.3},
                                "rse_pct": {"F": 22.0}, "description": "..."}]
                              if with_selection else []),
    }
    return {"status": "ok", "model_key": MODEL_KEY, "label": "1-compartment oral",
           "base_ofv": 120.0, "final_ofv": 110.0, "selected": (["CL~SEX"] if with_selection else []),
           "final": final,
           "selection_caveat": ("Stepwise selection: the retained effects' standard errors and "
                                "p-values are optimistic (post-selection inference); confirm on a "
                                "pre-specified covariate set or validate by resampling."
                                ) if with_selection else None}


@pytest.fixture
def loaded():
    ctx = ToolContext(dataset_store={"d1": _dataset()})
    state = PharmState(dataset_id="d1", dataset_metadata={"detected_roles": _roles()})
    return state, ctx


def test_tool_is_registered():
    tool = default_registry().get("run_covariate_forest")
    assert tool.agent == "modeler"


def test_state_write_access_includes_field():
    assert "forest_results" in AGENT_WRITE_FIELDS["modeler"]
    st = apply_writes(PharmState(), "modeler", {"forest_results": {"status": "ok"}})
    assert st.forest_results == {"status": "ok"}


def test_no_fit_is_graceful(loaded):
    state, ctx = loaded
    res = run_covariate_forest(state, ctx, {})
    assert res.writes["forest_results"]["status"] == "no_fit"


def test_nlme_only_produces_forest(loaded):
    state, ctx = loaded
    state.nlme_results = _nlme_with_covariates()
    res = run_covariate_forest(state, ctx, {})
    payload = res.writes["forest_results"]
    assert payload["status"] == "ok"
    assert payload["source"] == "nlme"
    assert payload["summary"]["n_rows"] > 0


def test_scm_source_does_not_silently_report_no_fit(loaded):
    # THE critical bug this wiring exists to fix: scm()'s `final` carries no
    # `status` key (only the outer scm dict does) -- an unwrapped `final`
    # passed straight to `covariate_forest`'s status gate would report
    # no_fit on every SCM-sourced request.
    state, ctx = loaded
    state.scm_results = _scm_outer(with_selection=True)
    res = run_covariate_forest(state, ctx, {"source": "scm"})
    payload = res.writes["forest_results"]
    assert payload["status"] == "ok"
    assert payload["source"] == "scm"
    assert payload["summary"]["n_rows"] > 0


def test_scm_selection_caveat_is_surfaced_in_notes(loaded):
    state, ctx = loaded
    state.scm_results = _scm_outer(with_selection=True)
    res = run_covariate_forest(state, ctx, {"source": "scm"})
    payload = res.writes["forest_results"]
    assert any("post-selection inference" in n for n in payload["notes"])


def test_scm_with_no_selected_effects_falls_back_to_nlme_in_auto_mode(loaded):
    # auto mode: SCM ran but selected nothing -> must not be chosen (it has no
    # covariate_effects, so covariate_forest would produce zero rows); falls
    # back to a plain NLME fit if one exists.
    state, ctx = loaded
    state.scm_results = _scm_outer(with_selection=False)
    state.nlme_results = _nlme_with_covariates()
    res = run_covariate_forest(state, ctx, {})  # source defaults to "auto"
    payload = res.writes["forest_results"]
    assert payload["source"] == "nlme"


def test_auto_prefers_scm_when_it_has_selected_effects(loaded):
    state, ctx = loaded
    state.scm_results = _scm_outer(with_selection=True)
    state.nlme_results = _nlme_with_covariates()
    res = run_covariate_forest(state, ctx, {})
    payload = res.writes["forest_results"]
    assert payload["source"] == "scm"


def test_categorical_reference_level_is_recovered_from_dataset(loaded):
    # SEX is 2/3 "M" in the fixture dataset -> "M" is the majority/reference
    # level, matching nlme.py's own max(uniq, key=vals.count) logic.
    state, ctx = loaded
    state.scm_results = _scm_outer(with_selection=True)
    res = run_covariate_forest(state, ctx, {"source": "scm"})
    rows = res.writes["forest_results"]["rows"]
    ref_row = next(r for r in rows if r["eval_value"] == "M")
    assert "(reference)" in ref_row["eval_label"]
    f_row = next(r for r in rows if r["eval_value"] == "F")
    assert "vs M" in f_row["eval_label"]


def test_continuous_percentiles_computed_from_dataset(loaded):
    state, ctx = loaded
    state.nlme_results = _nlme_with_covariates()
    res = run_covariate_forest(state, ctx, {"percentiles": [10.0, 90.0]})
    payload = res.writes["forest_results"]
    assert payload["percentiles"] == [10.0, 90.0]
    wt_rows = [r for r in payload["rows"] if r["covariate"] == "WT"]
    assert len(wt_rows) == 2
    assert payload["cov_stats"]["WT"]["n_cov"] == 20


def test_invalid_ci_level_is_graceful_not_a_crash(loaded):
    state, ctx = loaded
    state.nlme_results = _nlme_with_covariates()
    res = run_covariate_forest(state, ctx, {"ci_level": 1.5})
    assert res.writes["forest_results"]["status"] == "invalid_args"


def test_bounds_threaded_through(loaded):
    state, ctx = loaded
    state.nlme_results = _nlme_with_covariates()
    res = run_covariate_forest(state, ctx, {"bounds": [0.8, 1.25]})
    payload = res.writes["forest_results"]
    assert payload["bounds"] == [0.8, 1.25]


def test_payload_is_json_safe(loaded):
    state, ctx = loaded
    state.nlme_results = _nlme_with_covariates()
    res = run_covariate_forest(state, ctx, {})
    json.dumps(res.writes["forest_results"])
    json.dumps(res.result)


# ── SCM candidate screening (collinearity + never-tested reporting) ──────────
# Lecture rule (IU PopPK Week 9): avoid simultaneous inclusion of covariates
# correlated beyond |r| > 0.3 -- two near-collinear covariates cannot both be
# identified, and a stepwise search would arbitrarily keep whichever happened to
# be tested first. Covariates that are screened out must be reported, because a
# silently shortened candidate list reads as "tested and rejected".

def _subjects_with_cov(covs: list[dict]) -> list[dict]:
    """Minimal subject dicts carrying only the covariate payload under test."""
    return [{"subject": i, "doses": [{"time": 0.0, "amt": 100.0}],
             "obs_t": [1.0, 4.0], "obs_c": [5.0, 2.0], "wt": 70.0, "cov": c}
            for i, c in enumerate(covs)]


def test_collinear_covariate_is_screened_out_and_reported():
    from app.tools.pkmodel_tools import _covariate_candidates
    # AST and ALT are near-perfectly correlated (two measures of hepatic
    # function); EGFR is independent of both. Only one of AST/ALT may be offered.
    rng = np.random.default_rng(11)
    egfr = rng.normal(90.0, 20.0, 12)
    ast = np.linspace(20.0, 40.0, 12)
    covs = [{"EGFR": float(g), "AST": float(a), "ALT": float(2.0 * a + 1.0)}
            for g, a in zip(egfr, ast)]
    cands, dropped = _covariate_candidates(_subjects_with_cov(covs), ["CL"])
    names = [c["covariate"] for c in cands]
    assert "EGFR" in names, "independent covariate must survive the screen"
    assert not ("AST" in names and "ALT" in names), "collinear pair both offered"
    collinear = [d for d in dropped if d["reason"] == "collinear"]
    assert collinear, "screened-out covariate was not reported"
    assert collinear[0]["covariate"] in {"AST", "ALT"}
    assert "|r|" in collinear[0]["detail"]


def test_independent_covariates_all_survive_the_screen():
    from app.tools.pkmodel_tools import _covariate_candidates
    rng = np.random.default_rng(4)
    covs = [{"EGFR": float(v), "AGE": float(a), "ALB": float(b)}
            for v, a, b in zip(rng.normal(90, 20, 30), rng.normal(45, 15, 30),
                               rng.normal(4, 0.5, 30))]
    cands, dropped = _covariate_candidates(_subjects_with_cov(covs), ["CL"])
    names = {c["covariate"] for c in cands}
    assert names == {"EGFR", "AGE", "ALB"}
    assert not [d for d in dropped if d["reason"] == "collinear"]


def test_weight_and_constant_covariates_are_reported_not_silently_dropped():
    from app.tools.pkmodel_tools import _covariate_candidates
    covs = [{"WT": 70.0 + i, "EGFR": 90.0 + i, "SITE": 1.0} for i in range(10)]
    cands, dropped = _covariate_candidates(_subjects_with_cov(covs), ["CL"])
    names = {c["covariate"] for c in cands}
    assert "WT" not in names        # built-in allometry would double-count
    assert "SITE" not in names      # no variation
    reasons = {d["covariate"]: d["reason"] for d in dropped}
    assert reasons.get("WT") == "weight_builtin_allometry"
    assert reasons.get("SITE") == "no_variation"


def test_screening_is_deterministic():
    from app.tools.pkmodel_tools import _covariate_candidates
    covs = [{"EGFR": 90.0 + i, "AST": 20.0 + i, "ALT": 40.0 + 2.0 * i}
            for i in range(12)]
    subs = _subjects_with_cov(covs)
    a_c, a_d = _covariate_candidates(subs, ["CL"])
    b_c, b_d = _covariate_candidates(subs, ["CL"])
    assert a_c == b_c and a_d == b_d
