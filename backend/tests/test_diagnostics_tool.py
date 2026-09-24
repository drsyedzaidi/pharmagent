"""Tool-wiring tests for run_diagnostics (single-provenance NLME sourcing,
needs_nlme / blq_unsupported branches, BLQ handling, JSON safety).

The npd (Comets normalized prediction discrepancy) block requires a converged
NLME fit of the SAME structural model to supply a residual-error model: it is
never built from a mix of the two-stage typical/IIV and an unrelated sigma
(see app.compute.diagnostics.npde and the single-provenance rule in
run_diagnostics). These tests exercise that selection logic directly, without
running a real fit.
"""
import json

import numpy as np
import pandas as pd
import pytest

from app.core.pharmstate import PharmState
from app.tools.base import ToolContext
from app.tools.builtins import default_registry
from app.tools.pkmodel_tools import run_diagnostics

MODEL_KEY = "oral_1cmt"


def _dataset(with_cens: bool = False) -> pd.DataFrame:
    rows = []
    for sid in range(1, 5):
        rows.append({"ID": sid, "TIME": 0.0, "DV": np.nan, "AMT": 100.0, "CENS": 0})
        for t, c in zip([0.5, 1, 2, 4, 8, 12], [0.8, 1.4, 1.2, 0.9, 0.5, 0.2]):
            dv = c * (1 + 0.03 * sid)
            cens = 0
            if with_cens and sid == 1 and t == 12:
                dv, cens = 0.05, 1  # one BLQ row, carrying the LLOQ in DV
            rows.append({"ID": sid, "TIME": t, "DV": dv, "AMT": np.nan, "CENS": cens})
    return pd.DataFrame(rows)


def _roles(with_cens: bool = False) -> dict:
    roles = {"ID": "ID", "TIME": "TIME", "DV": "DV", "AMT": "AMT"}
    if with_cens:
        roles["CENS"] = "CENS"
    return roles


def _pk_model_results() -> dict:
    return {
        "status": "ok", "mode": "fit", "model_key": MODEL_KEY,
        "individual_fits": [
            {"subject": sid, "converged": True,
             "params": {"CL": 4.0 + 0.1 * sid, "V": 40.0, "KA": 1.2}}
            for sid in range(1, 5)
        ],
        "population": {"parameters": {
            "CL": {"typical_value": 4.2, "iiv_cv_pct": 25.0},
            "V": {"typical_value": 40.0, "iiv_cv_pct": 20.0},
            "KA": {"typical_value": 1.2, "iiv_cv_pct": 15.0},
        }},
    }


def _nlme_results(*, model_key: str = MODEL_KEY, prop: float = 0.15, add: float = 0.3) -> dict:
    return {
        "status": "ok", "model_key": model_key, "label": "1-compartment oral",
        "theta": {"CL": 4.2, "V": 40.0, "KA": 1.2},
        "omega_cv_pct": {"CL": 25.0, "V": 20.0},
        "sigma": {"prop": prop, "add": add},
        "iiv_params": ["CL", "V"], "error_model": "proportional",
        "covariate_effects": [],
        "individual": [{"subject": sid, "eta": {"CL": 0.02 * sid, "V": 0.0}}
                       for sid in range(1, 5)],
        "n_obs": 24, "n_blq": 0,
    }


@pytest.fixture
def loaded():
    ctx = ToolContext(dataset_store={"d1": _dataset()})
    state = PharmState(dataset_id="d1", pk_model_results=_pk_model_results(),
                       dataset_metadata={"detected_roles": _roles()})
    return state, ctx


def test_tool_is_registered():
    tool = default_registry().get("run_diagnostics")
    assert tool.agent == "modeler"


def test_no_fit_is_graceful():
    res = run_diagnostics(PharmState(), ToolContext(), {})
    assert res.writes["diagnostics_results"]["status"] == "no_fit"


def test_without_nlme_npd_reports_needs_nlme(loaded):
    # Only the two-stage fit is present; no run_nlme has ever succeeded.
    state, ctx = loaded
    res = run_diagnostics(state, ctx, {})
    payload = res.writes["diagnostics_results"]
    assert payload["status"] == "ok"
    assert payload["residuals"]["summary"]["n"] > 0  # legacy two-stage IWRES unaffected
    assert payload["npde"]["status"] == "needs_nlme"
    assert payload["cwres"]["status"] == "needs_nlme"
    assert "npde" not in payload["npde"]  # no fabricated predictive-distribution output
    assert "cwres" not in payload["cwres"]
    assert res.result["npde_status"] == "needs_nlme"
    assert res.result["cwres_status"] == "needs_nlme"
    assert payload["nlme_provenance"] is None


def test_nlme_from_a_different_model_is_not_used(loaded):
    # A converged NLME fit exists, but for a DIFFERENT structural model —
    # single-provenance rule: theta/Omega/sigma must never be borrowed across
    # models, for CWRES or npd.
    state, ctx = loaded
    state.nlme_results = _nlme_results(model_key="oral_2cmt")
    res = run_diagnostics(state, ctx, {})
    payload = res.writes["diagnostics_results"]
    assert payload["npde"]["status"] == "needs_nlme"
    assert payload["cwres"]["status"] == "needs_nlme"


def test_matching_nlme_enables_npd(loaded):
    state, ctx = loaded
    state.nlme_results = _nlme_results()
    res = run_diagnostics(state, ctx, {})
    payload = res.writes["diagnostics_results"]
    assert "status" not in payload["npde"]  # ok path carries no status key
    assert payload["npde"]["metric"] == "npd"
    assert payload["npde"]["summary"]["sigma_prop"] == 0.15
    assert payload["npde"]["summary"]["sigma_add"] == 0.3
    assert res.result["npde_status"] == "ok"
    assert payload["nlme_provenance"] == MODEL_KEY


def test_matching_nlme_enables_cwres_and_reuses_stored_etas(loaded):
    state, ctx = loaded
    state.nlme_results = _nlme_results()
    res = run_diagnostics(state, ctx, {})
    payload = res.writes["diagnostics_results"]
    cw = payload["cwres"]
    assert "status" not in cw  # ok path carries no status key
    assert cw["summary"]["n"] > 0
    assert cw["summary"]["cwres_mean"] is not None
    assert cw["summary"]["n_etas_reused"] == 4  # all 4 subjects had a stored EBE
    assert cw["summary"]["n_etas_resolved"] == 0
    assert cw["summary"]["interaction"] is True  # default
    assert res.result["cwres_status"] == "ok"
    assert res.result["cwres_summary"]["cwres_mean"] is not None


def test_cwres_interaction_flag_is_threaded_through(loaded):
    state, ctx = loaded
    state.nlme_results = _nlme_results()
    res = run_diagnostics(state, ctx, {"cwres_interaction": False})
    cw = res.writes["diagnostics_results"]["cwres"]
    assert cw["summary"]["interaction"] is False
    assert cw["summary"]["cwres_variant"] == "foce"


def test_blq_rows_disable_npd_but_not_cwres(loaded):
    # BLQ present + converged matching NLME: npd is withheld (the widened
    # simulated cloud vs BLQ-excluded observed side creates a spurious trend),
    # NOT silently computed on a biased subset. CWRES has no such pathology
    # (it drops BLQ rows internally via `_cwres_subject`) so it still runs.
    ctx = ToolContext(dataset_store={"d1": _dataset(with_cens=True)})
    state = PharmState(dataset_id="d1", pk_model_results=_pk_model_results(),
                       dataset_metadata={"detected_roles": _roles(with_cens=True)},
                       nlme_results=_nlme_results())
    res = run_diagnostics(state, ctx, {})
    payload = res.writes["diagnostics_results"]
    assert payload["npde"]["status"] == "blq_unsupported"
    assert payload["npde"]["n_blq"] == 1
    # legacy two-stage IWRES still drops the BLQ row rather than treating it
    # as quantified.
    assert payload["residuals"]["summary"]["n"] == 4 * 6 - 1
    # CWRES is unaffected by the npd-specific gate.
    assert "status" not in payload["cwres"]
    assert payload["cwres"]["summary"]["n_blq_dropped"] == 1


def test_payload_is_json_safe(loaded):
    state, ctx = loaded
    state.nlme_results = _nlme_results()
    res = run_diagnostics(state, ctx, {})
    json.dumps(res.writes["diagnostics_results"])
    json.dumps(res.result)
