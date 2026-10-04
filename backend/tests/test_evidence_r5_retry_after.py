"""Every evidence 429 / 503 tells the client when to retry: the server-wide upload saturation (503) and the
per-session upload rate limit (429) carry Retry-After like the parse queue, the render limits and the layout
rebuild limits do, over HTTP through the global handler; and no HTTPException(429|503) is raised in
app/evidence without a headers argument."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from app.evidence import upload_limits
from tests import evidence_fixtures as fx
from tests.test_evidence_upload_api import _sid, _upload
from tests.test_evidence_upload_api import client as client  # noqa: F401  (the module's fixture)


@pytest.fixture(scope="module")
def papers(tmp_path_factory):
    return fx.build_all(tmp_path_factory.mktemp("papers"))


def _retry_after(response) -> int:
    value = response.headers.get("retry-after")
    assert value is not None, dict(response.headers)
    assert value.isdigit() and int(value) >= 1, value
    return int(value)


def test_upload_server_saturation_503_carries_retry_after(client, papers, monkeypatch):
    sid = _sid(client)
    monkeypatch.setattr(upload_limits, "MAX_UPLOADS_IN_FLIGHT", 0)
    r = _upload(client, sid, papers["paper_a"])
    assert r.status_code == 503, r.text
    assert _retry_after(r) == upload_limits.RETRY_AFTER_S
    assert "uploads" in r.json()["error"]["message"]


def test_upload_rate_limit_429_carries_retry_after_for_the_rest_of_the_window(client, papers, monkeypatch):
    sid = _sid(client)
    monkeypatch.setattr(upload_limits, "UPLOADS_PER_WINDOW", 1)
    assert _upload(client, sid, papers["paper_a"]).status_code == 200
    r = _upload(client, sid, papers["paper_b"])
    assert r.status_code == 429, r.text
    assert 1 <= _retry_after(r) <= upload_limits.WINDOW_S + 1


def test_a_refused_upload_leaves_no_slot_held(client, papers, monkeypatch):
    sid = _sid(client)
    monkeypatch.setattr(upload_limits, "MAX_UPLOADS_IN_FLIGHT", 0)
    assert _upload(client, sid, papers["paper_a"]).status_code == 503
    monkeypatch.setattr(upload_limits, "MAX_UPLOADS_IN_FLIGHT", 32)
    assert upload_limits._in_flight[0] == 0
    assert _upload(client, sid, papers["paper_a"]).status_code == 200


def _status_of(call: ast.Call) -> int | None:
    func = call.func
    name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
    if name not in {"HTTPException", "_refuse", "refuse"} or not call.args:
        return None
    first = call.args[0]
    return first.value if isinstance(first, ast.Constant) and isinstance(first.value, int) else None


def test_no_429_or_503_in_app_evidence_is_raised_without_headers():
    """Static guard: a literal HTTPException(429|503, ...) must pass headers= (the shared helpers always do)."""
    root = Path(__file__).resolve().parents[1] / "app" / "evidence"
    bare = []
    for path in sorted(root.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text("utf-8"))):
            if isinstance(node, ast.Call) and _status_of(node) in (429, 503):
                is_exception = getattr(node.func, "id", None) == "HTTPException"
                if is_exception and not any(k.arg == "headers" for k in node.keywords):
                    bare.append(f"{path.name}:{node.lineno}")
    assert bare == [], bare
