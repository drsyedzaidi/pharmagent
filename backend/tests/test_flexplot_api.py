"""HTTP + tool-wiring tests for the flexplot visualization feature.

Exercises the registered tool, the ``POST /flexplot`` endpoint (audited, writes
``flexplot_data`` to state), and the ``GET /variables`` picker endpoint. Keyless
via MockLLM, mirroring ``test_api.py``.
"""
import itertools
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.config import settings
from app.core.llm import MockLLM
from app.core.orchestrator import Orchestrator
from app.core.store import SessionStore
from app.tools.builtins import default_registry

SAMPLE = str(Path(__file__).parent.parent / "sample_data" / "oral_pk.csv")


@pytest.fixture
def client():
    main.orch = Orchestrator(llm=MockLLM(), clock=lambda c=itertools.count(): f"t{next(c)}",
                             store=SessionStore(":memory:"))
    token = settings.api_token
    yield TestClient(main.app)
    settings.api_token = token


def _loaded_session(client) -> str:
    sid = client.post("/api/sessions").json()["id"]
    client.post(f"/api/sessions/{sid}/chat", json={"message": f"load dataset {SAMPLE}"})
    return sid


def test_tool_is_registered_and_owned_by_data_manager():
    tool = default_registry().get("generate_flexplot")
    assert tool.agent == "data_manager"


def test_flexplot_endpoint_writes_state_and_audits(client):
    sid = _loaded_session(client)
    r = client.post(f"/api/sessions/{sid}/flexplot", json={"y": "DV", "x": "TIME", "fit": "loess"})
    assert r.status_code == 200
    body = r.json()
    assert body["audit_ok"] is True
    fp = body["state"]["flexplot_data"]
    assert fp["kind"] == "scatter"
    assert fp["summary"]["n"] > 0
    assert fp["cells"][0]["fit"] is not None
    # audit chain still verifies after the tool write
    audit = client.get(f"/api/sessions/{sid}/audit").json()
    assert audit["verified"] is True


def test_flexplot_dotplot_and_color(client):
    sid = _loaded_session(client)
    r = client.post(f"/api/sessions/{sid}/flexplot",
                    json={"y": "DV", "x": "DOSE", "center": "mean_se"})
    assert r.status_code == 200
    fp = r.json()["state"]["flexplot_data"]
    assert fp["kind"] == "dotplot"
    assert fp["cells"][0]["crossbars"]


def test_flexplot_bad_outcome_returns_400(client):
    sid = _loaded_session(client)
    r = client.post(f"/api/sessions/{sid}/flexplot", json={"y": "NOPE", "x": "TIME"})
    assert r.status_code == 400


def test_variables_endpoint_lists_typed_columns(client):
    sid = _loaded_session(client)
    r = client.get(f"/api/sessions/{sid}/variables")
    assert r.status_code == 200
    body = r.json()
    names = {v["name"]: v for v in body["variables"]}
    assert "DV" in names and names["DV"]["type"] == "continuous"
    assert "detected_roles" in body


def test_variables_without_dataset_returns_400(client):
    sid = client.post("/api/sessions").json()["id"]
    assert client.get(f"/api/sessions/{sid}/variables").status_code == 400


def test_flexplot_rejects_out_of_range_params(client):
    sid = _loaded_session(client)
    # ci must be in (0,1); n_bins bounded -> 422 from the request model, not a 500
    assert client.post(f"/api/sessions/{sid}/flexplot",
                       json={"y": "DV", "x": "TIME", "ci": 5.0}).status_code == 422
    assert client.post(f"/api/sessions/{sid}/flexplot",
                       json={"y": "DV", "n_bins": 10_000_000}).status_code == 422
