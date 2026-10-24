"""G1: compare mode must choose its default candidate set from the data's route.

``fit_pk_model`` compare mode used the oral set whenever ``models`` was absent,
so the poppk_full "Compare structural models" step ranked oral absorption
models on IV infusion data (``_DEFAULT_IV`` was never used). The default set now
follows the dosing route: IV when every dose event is an infusion (RATE > 0) or
a ROUTE column marks every dosing record IV, else oral. Which set was used, and
why, is recorded in ``pk_model_results``. No test here runs a fit.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from app.core.pharmstate import PharmState
from app.tools import pkmodel_tools
from app.tools.base import ToolContext
from app.tools.pkmodel_tools import _DEFAULT_IV, _DEFAULT_ORAL, fit_pk_model

_TIMES = [0.05, 0.25, 1.0, 3.0, 7.0, 14.0, 21.0]


def _dataset(*, rates: list[float] | None = None, routes: list[Any] | None = None,
             cmt: int | None = None, n: int = 4) -> pd.DataFrame:
    """One 100 mg dose per subject at t=0, then a declining profile.

    ``rates`` / ``routes`` give each subject's dose-row RATE / ROUTE (cycled),
    as a 1-hour infusion dataset would after PMxbench ``prepare_dataset``.
    """
    rows = []
    for i, sid in enumerate(range(1, n + 1)):
        dose: dict[str, Any] = {"ID": sid, "TIME": 0.0, "DV": np.nan, "AMT": 100.0, "WT": 60.0 + sid}
        obs_extra: dict[str, Any] = {}
        if rates is not None:
            dose["RATE"] = rates[i % len(rates)]
            obs_extra["RATE"] = 0.0
        if routes is not None:
            dose["ROUTE"] = routes[i % len(routes)]
            obs_extra["ROUTE"] = routes[i % len(routes)]
        if cmt is not None:
            dose["CMT"] = cmt
            obs_extra["CMT"] = cmt
        rows.append(dose)
        for t in _TIMES:
            rows.append({"ID": sid, "TIME": t, "DV": 40.0 * np.exp(-0.1 * t), "AMT": 0.0,
                         "WT": 60.0 + sid, **obs_extra})
    return pd.DataFrame(rows)


def _capture_compare(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace the two-stage comparison so the candidate list is recorded, no fit runs."""
    seen: dict[str, Any] = {}

    def fake_compare(subjects: list[dict], candidates: list[str]) -> dict[str, Any]:
        seen["candidates"] = list(candidates)
        seen["subjects"] = subjects
        return {"ranking": [{"model_key": k} for k in candidates],
                "best_model": candidates[0] if candidates else None, "best": None}

    monkeypatch.setattr(pkmodel_tools, "compare_models", fake_compare)
    return seen


def _compare(monkeypatch: pytest.MonkeyPatch, df: pd.DataFrame,
             args: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    seen = _capture_compare(monkeypatch)
    ctx = ToolContext(dataset_store={"d1": df})
    res = fit_pk_model(PharmState(dataset_id="d1"), ctx, {"compare": True, **(args or {})})
    return seen, res.writes["pk_model_results"]


# ── IV evidence → IV candidate set ────────────────────────────────────────────

def test_iv_infusion_dataset_gets_iv_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every dose has RATE > 0 (a 1-hour infusion, days time scale) -> IV set."""
    seen, out = _compare(monkeypatch, _dataset(rates=[100.0 * 24]))
    assert seen["candidates"] == _DEFAULT_IV
    cs = out["candidate_set"]
    assert cs["route"] == "iv" and cs["source"] == "default"
    assert cs["models"] == _DEFAULT_IV
    assert "RATE" in cs["reason"]


@pytest.mark.parametrize("route", ["IV", "iv", " Intravenous ", "INFUSION", "bolus", "i.v."])
def test_route_column_marking_every_dose_iv_gets_iv_candidates(
        monkeypatch: pytest.MonkeyPatch, route: str) -> None:
    seen, out = _compare(monkeypatch, _dataset(routes=[route]))
    assert seen["candidates"] == _DEFAULT_IV
    assert out["candidate_set"]["route"] == "iv"
    assert "ROUTE" in out["candidate_set"]["reason"]


# ── no / partial IV evidence → oral set (the previous default, unchanged) ─────

def test_plain_dataset_still_gets_oral_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    seen, out = _compare(monkeypatch, _dataset())
    assert seen["candidates"] == _DEFAULT_ORAL
    cs = out["candidate_set"]
    assert cs["route"] == "oral" and cs["source"] == "default"
    assert cs["models"] == _DEFAULT_ORAL
    assert cs["reason"]


def test_some_doses_without_rate_is_not_iv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only SOME doses are infusions: not every dose event has RATE > 0."""
    seen, out = _compare(monkeypatch, _dataset(rates=[2400.0, 0.0]))
    assert seen["candidates"] == _DEFAULT_ORAL
    assert out["candidate_set"]["route"] == "oral"


def test_mixed_route_column_is_not_iv(monkeypatch: pytest.MonkeyPatch) -> None:
    seen, out = _compare(monkeypatch, _dataset(routes=["IV", "ORAL"]))
    assert seen["candidates"] == _DEFAULT_ORAL
    assert out["candidate_set"]["route"] == "oral"


def test_dose_record_with_blank_route_is_not_iv(monkeypatch: pytest.MonkeyPatch) -> None:
    seen, _out = _compare(monkeypatch, _dataset(routes=["IV", None]))
    assert seen["candidates"] == _DEFAULT_ORAL


def test_explicit_extravascular_route_wins_over_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    """RATE on a depot dose is zero-order absorption, not IV: an explicit oral ROUTE decides."""
    seen, out = _compare(monkeypatch, _dataset(rates=[50.0], routes=["PO"]))
    assert seen["candidates"] == _DEFAULT_ORAL
    assert out["candidate_set"]["route"] == "oral"


def test_cmt_alone_is_not_route_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    """CMT=1 is the dosing compartment of IV (central) AND oral (depot) models alike."""
    seen, _out = _compare(monkeypatch, _dataset(cmt=1))
    assert seen["candidates"] == _DEFAULT_ORAL


# ── caller-specified models are used as given ─────────────────────────────────

def test_requested_models_are_used_and_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    seen, out = _compare(monkeypatch, _dataset(rates=[2400.0]),
                         {"models": ["iv_1cmt", "oral_1cmt"]})
    assert seen["candidates"] == ["iv_1cmt", "oral_1cmt"]
    cs = out["candidate_set"]
    assert cs["source"] == "requested"
    assert cs["models"] == ["iv_1cmt", "oral_1cmt"]
    assert cs["route"] == "iv"          # detected route still recorded for the reviewer


def test_poppk_full_compare_step_args_pick_iv_on_infusion_data(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """The workflow step passes only {"compare": True}; that alone must give IV models."""
    from app.workflows import WORKFLOWS

    step = next(s for s in WORKFLOWS["poppk_full"]["steps"] if s["tool"] == "fit_pk_model")
    seen = _capture_compare(monkeypatch)
    ctx = ToolContext(dataset_store={"d1": _dataset(rates=[2400.0])})
    fit_pk_model(PharmState(dataset_id="d1"), ctx, dict(step["args"]))
    assert seen["candidates"] == _DEFAULT_IV
