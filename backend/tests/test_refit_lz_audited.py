"""F-12: a manual lambda_z refit is an audited analyst intervention that updates session state."""
from __future__ import annotations

import itertools
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.llm import MockLLM
from app.core.orchestrator import Orchestrator
from app.core.store import SessionStore

SAMPLE = str(Path(__file__).parent.parent / "sample_data" / "oral_pk.csv")


@pytest.fixture
def setup(monkeypatch):
    import app.main as main
    orch = Orchestrator(llm=MockLLM(), clock=lambda c=itertools.count(): f"t{next(c)}",
                        store=SessionStore(":memory:"))
    monkeypatch.setattr(main, "orch", orch)
    client = TestClient(main.app)
    sid = client.post("/api/sessions").json()["id"]
    orch.run_tool(sid, "load_dataset", "data_manager", {"path": SAMPLE})
    orch.run_tool(sid, "compute_nca", "nca", {})
    return client, orch, sid


def _subject(orch, sid):
    sess = orch.get_session(sid)
    plot = sess.state.nca_plot_data["subjects"][0]
    row = next(p for p in sess.state.nca_parameters if str(p["subject"]) == plot["id"])
    return plot, row


def _pick_other_points(plot, row):
    """Last 3 positive points: a deliberate analyst override of the auto window when it differs."""
    pts = list(zip(plot["x"], plot["y"], strict=True))[-3:]
    return [p[0] for p in pts], [p[1] for p in pts]


def test_refit_updates_state_and_leaves_an_attributed_audit_entry(setup):
    client, orch, sid = setup
    plot, before = _subject(orch, sid)
    times, concs = _pick_other_points(plot, before)
    n_before = len(orch.get_session(sid).audit.entries)

    r = client.post(f"/api/sessions/{sid}/nca/refit_lz",
                    json={"subject": plot["id"], "selected_times": times, "selected_concs": concs,
                          "reason": "terminal phase starts later"})
    assert r.status_code == 200
    body = r.json()
    assert set(body) >= {"lambda_z", "lambda_z_intercept", "t_half", "r2_adj", "n_pts",
                         "lz_x", "lz_y", "fit_x", "fit_y"}

    _, after = _subject(orch, sid)
    assert after["lambda_z"] == body["lambda_z"] and after["lambda_z_n_points"] == 3
    assert after["lambda_z"] != before["lambda_z"] or after["lambda_z_pts_t"] != before["lambda_z_pts_t"]
    assert after["t_half"] == pytest.approx(body["t_half"], rel=1e-4)
    assert after["lambda_z_manual"] is True

    entries = orch.get_session(sid).audit.entries
    assert len(entries) == n_before + 1
    e = entries[-1]
    assert e.tool == "refit_lambda_z" and e.actor == "anonymous"
    assert e.reason == "terminal phase starts later"
    assert plot["id"] in e.action
    assert orch.get_session(sid).audit.verify()


def test_refit_rejects_points_that_are_not_on_the_subjects_profile(setup):
    client, orch, sid = setup
    plot, row = _subject(orch, sid)
    times, concs = _pick_other_points(plot, row)
    n_before = len(orch.get_session(sid).audit.entries)
    _, snapshot = _subject(orch, sid)

    forged = client.post(f"/api/sessions/{sid}/nca/refit_lz",
                         json={"subject": plot["id"], "selected_times": times,
                               "selected_concs": [c * 2 for c in concs]})
    assert forged.status_code == 400
    unknown = client.post(f"/api/sessions/{sid}/nca/refit_lz",
                          json={"subject": "no-such-subject", "selected_times": times,
                                "selected_concs": concs})
    assert unknown.status_code == 400
    dup = client.post(f"/api/sessions/{sid}/nca/refit_lz",
                      json={"subject": plot["id"], "selected_times": [times[0]] * 3,
                            "selected_concs": [concs[0]] * 3})
    assert dup.status_code == 400
    assert len(orch.get_session(sid).audit.entries) == n_before
    assert _subject(orch, sid)[1] == snapshot
