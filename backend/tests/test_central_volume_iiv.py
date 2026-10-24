"""G3: default IIV and SCM parameter lists must name each model's real central
volume, and a covariate effect on a parameter the model lacks must be rejected.

The 2- and 3-compartment models call the central volume ``VC``, while the
defaults assumed ``V``. The NLME default silently became CL-only IIV, and the
automatic SCM candidates on "V" were coefficients the fit ignored: delta-OFV 0,
still counted as tested, and VC itself never screened. No test here runs a fit.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from app.compute import nlme
from app.compute.pk_models import get_model
from app.core.pharmstate import PharmState
from app.tools.base import ToolContext
from app.tools.pkmodel_tools import run_scm


@pytest.mark.parametrize("model_key, expected", [
    ("iv_1cmt", ["CL", "V"]),
    ("oral_1cmt", ["CL", "V"]),
    ("iv_2cmt", ["CL", "VC"]),
    ("oral_2cmt", ["CL", "VC"]),
    ("iv_3cmt", ["CL", "VC"]),
    ("iv_1cmt_mm", ["V"]),            # no CL parameter: unchanged
])
def test_default_iiv_names_the_models_central_volume(model_key: str, expected: list[str]) -> None:
    assert nlme._resolve_iiv(get_model(model_key), None) == expected


def test_requested_v_means_the_central_volume() -> None:
    assert nlme._resolve_iiv(get_model("iv_2cmt"), ["CL", "V"]) == ["CL", "VC"]
    assert nlme._resolve_iiv(get_model("iv_1cmt"), ["CL", "V"]) == ["CL", "V"]
    assert nlme._resolve_iiv(get_model("iv_2cmt"), ["Q"]) == ["Q"]
    # V and VC both requested on a VC model collapse to one entry, order kept
    assert nlme._resolve_iiv(get_model("iv_2cmt"), ["V", "VC", "CL"]) == ["VC", "CL"]
    # nothing usable requested falls back to the model default, as before
    assert nlme._resolve_iiv(get_model("iv_2cmt"), ["NOPE"]) == ["CL", "VC"]


def _dataset(n: int = 12) -> pd.DataFrame:
    rows = []
    for sid in range(1, n + 1):
        age = 20.0 + 3.0 * sid
        sex = "F" if sid % 3 == 0 else "M"
        rows.append({"ID": sid, "TIME": 0.0, "DV": np.nan, "AMT": 100.0, "AGE": age, "SEX": sex})
        for t, c in zip([0.25, 1, 2, 4, 8, 12], [3.0, 2.2, 1.6, 1.0, 0.5, 0.25], strict=True):
            rows.append({"ID": sid, "TIME": t, "DV": c * (1 + 0.01 * sid), "AMT": np.nan,
                         "AGE": age, "SEX": sex})
    return pd.DataFrame(rows)


def _scm_state() -> tuple[PharmState, ToolContext]:
    ctx = ToolContext(dataset_store={"d1": _dataset()})
    roles = {"ID": "ID", "TIME": "TIME", "DV": "DV", "AMT": "AMT"}
    return PharmState(dataset_id="d1", dataset_metadata={"detected_roles": roles}), ctx


def _capture_scm(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace the SCM engine so the tool's arguments are recorded and no fit runs."""
    seen: dict[str, Any] = {}

    def fake_scm(model_key: str, subjects: list[dict], *, candidates: list[dict],
                 iiv_params: list[str], **_kw: Any) -> dict[str, Any]:
        seen.update(model_key=model_key, candidates=candidates, iiv_params=iiv_params)
        return {"status": "ok", "label": model_key, "selected": [], "steps": [],
                "n_candidates": len(candidates), "base_ofv": 0.0, "final_ofv": 0.0}

    monkeypatch.setattr(nlme, "scm", fake_scm)
    return seen


@pytest.mark.parametrize("model_key", ["iv_2cmt", "oral_2cmt", "iv_3cmt"])
def test_auto_scm_candidates_screen_vc_on_multi_compartment_models(
        monkeypatch: pytest.MonkeyPatch, model_key: str) -> None:
    seen = _capture_scm(monkeypatch)
    state, ctx = _scm_state()
    run_scm(state, ctx, {"model_key": model_key})

    assert seen["iiv_params"] == ["CL", "VC"]
    params = {c["param"] for c in seen["candidates"]}
    assert params == {"CL", "VC"}, params          # VC is screened, phantom V is not
    covs_on_vc = {c["covariate"] for c in seen["candidates"] if c["param"] == "VC"}
    assert covs_on_vc == {"AGE", "SEX"}


def test_auto_scm_candidates_unchanged_on_one_compartment_models(
        monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _capture_scm(monkeypatch)
    state, ctx = _scm_state()
    run_scm(state, ctx, {"model_key": "oral_1cmt"})
    assert seen["iiv_params"] == ["CL", "V"]
    assert {c["param"] for c in seen["candidates"]} == {"CL", "V"}


def _no_fit(*_a: Any, **_k: Any) -> Any:
    raise AssertionError("a fit was started: validation must reject the spec first")


_V_ON_2CMT = [{"param": "V", "covariate": "AGE", "kind": "power"}]


@pytest.mark.parametrize("fitter", ["focei_fit", "saem_fit"])
def test_covariate_effect_on_a_missing_parameter_is_rejected_before_fitting(
        monkeypatch: pytest.MonkeyPatch, fitter: str) -> None:
    monkeypatch.setattr(nlme, "_prepare_subjects", _no_fit)
    with pytest.raises(ValueError, match=r"'V'.*iv_2cmt.*VC"):
        getattr(nlme, fitter)("iv_2cmt", [], iiv_params=["CL"], error_model="proportional",
                              covariate_model=_V_ON_2CMT)


def test_scm_rejects_a_candidate_on_a_missing_parameter_before_fitting(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nlme, "_prepare_subjects", _no_fit)
    monkeypatch.setattr(nlme, "focei_fit", _no_fit)
    with pytest.raises(ValueError, match=r"'V'.*iv_2cmt.*VC"):
        nlme.scm("iv_2cmt", [], candidates=_V_ON_2CMT, iiv_params=["CL", "VC"])


def test_valid_covariate_effect_still_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    """The guard must not reject a real parameter: VC on iv_2cmt gets past it."""
    reached: list[bool] = []

    def stop(*_a: Any, **_k: Any) -> Any:
        reached.append(True)
        raise RuntimeError("stop after validation")

    monkeypatch.setattr(nlme, "_prepare_subjects", stop)
    with pytest.raises(RuntimeError, match="stop after validation"):
        nlme.focei_fit("iv_2cmt", [], iiv_params=["CL", "VC"], error_model="proportional",
                       covariate_model=[{"param": "VC", "covariate": "AGE", "kind": "power"}])
    assert reached == [True]
