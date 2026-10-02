"""Shared helpers of the page-image API tests: client fixture, sessions with uploaded PDFs, records
inserted straight into the ledger, PNG probes."""
from __future__ import annotations

import io
import itertools
from pathlib import Path
from typing import Any

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

import app.main as main
from app.config import settings
from app.core.llm import MockLLM
from app.core.orchestrator import Orchestrator
from app.core.store import SessionStore
from app.evidence import ingest, llm_access, render_cache, render_limits
from app.evidence import store as evstore
from app.evidence.extract_tools import check_proposals
from tests import evidence_fixtures as fx
from tests.test_evidence_extract import INCLUDE_ALL, TEMPLATE

EV = "/api/sessions/{sid}/evidence"


def build_papers(tmp_path_factory) -> dict[str, Path]:
    return fx.build_all(tmp_path_factory.mktemp("papers"))


def make_client(tmp_path, monkeypatch):
    """Generator behind each test module's ``client`` fixture (fresh ledger, MockLLM, open auth)."""
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    evstore.configure(":memory:")
    render_limits.reset()
    main.orch = Orchestrator(llm=MockLLM(), clock=lambda c=itertools.count(): f"t{next(c)}",
                             store=SessionStore(":memory:"))
    token = settings.api_token
    yield TestClient(main.app)
    settings.api_token = token
    llm_access.set_llm(None)
    render_limits.reset()


def session_with(client: TestClient, files: dict[str, bytes], h: dict | None = None, extract: bool = True) -> str:
    """A session holding *files* (name -> PDF bytes), screened in, with the template; records extracted."""
    h = h or {}
    sid = client.post("/api/sessions", headers=h).json()["id"]
    for name, data in files.items():
        r = client.post(EV.format(sid=sid) + "/upload", headers=h,
                        files={"file": (f"{name}.pdf", data, "application/pdf")})
        assert r.status_code == 200, r.text
    assert client.post(EV.format(sid=sid) + "/criteria", json={"criteria": INCLUDE_ALL}, headers=h).status_code == 200
    r = client.post(EV.format(sid=sid) + "/screen", json={}, headers=h)
    decisions = [{"screen_id": s["id"], "decision": "approve", "reason": "ok"} for s in r.json()["result"]["screens"]]
    assert client.post(EV.format(sid=sid) + "/screen/decide", json={"decisions": decisions}, headers=h).status_code == 200
    assert client.post(EV.format(sid=sid) + "/template", json=TEMPLATE, headers=h).status_code == 200
    if extract:
        assert client.post(EV.format(sid=sid) + "/extract", json={}, headers=h).status_code == 200
    return sid


def records(client: TestClient, sid: str, h: dict | None = None) -> list[dict]:
    return client.get(EV.format(sid=sid), headers=h or {}).json()["records"]


def source_id(client: TestClient, sid: str, name: str, h: dict | None = None) -> str:
    rows = client.get(EV.format(sid=sid), headers=h or {}).json()["sources"]
    return next(s["id"] for s in rows if s["filename"] == f"{name}.pdf")


def record_of(client: TestClient, sid: str, name: str, field: str, value: str | None = None,
              h: dict | None = None) -> dict:
    src = source_id(client, sid, name, h)
    return next(r for r in records(client, sid, h)
                if r["source_id"] == src and r["field"] == field and (value is None or r["value"] == value))


def meta(client: TestClient, sid: str, rid: str, h: dict | None = None, **params: Any):
    return client.get(EV.format(sid=sid) + f"/records/{rid}/image/meta", params=params, headers=h or {})


def image(client: TestClient, sid: str, rid: str, sha: str | None, h: dict | None = None, **params: Any):
    q = {**params, **({"sha": sha} if sha is not None else {})}
    return client.get(EV.format(sid=sid) + f"/records/{rid}/image", params=q, headers=h or {})


def pixels(png: bytes) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(png)).convert("RGB"))


def ink(png: bytes, rect: dict | list, pad: int = 0) -> float:
    """Share of dark pixels inside a meta rect ({x,y,w,h} or [x,y,w,h])."""
    x, y, w, h = (rect["x"], rect["y"], rect["w"], rect["h"]) if isinstance(rect, dict) else rect
    arr = pixels(png).min(axis=2) < 140
    reg = arr[max(0, int(y) - pad):int(y + h) + 1 + pad, max(0, int(x) - pad):int(x + w) + 1 + pad]
    return float(reg.mean()) if reg.size else 0.0


def add_record(client: TestClient, sid: str, src_id: str, locator: dict, *, value: str = "3,412",
               field: str = "cmax", verify_status: str = "pass", flags: list | None = None) -> str:
    """Insert a proposed record with exactly *locator* (no verification) and return its id."""
    store = evstore.get_store()
    src = store.get_source(src_id)
    tmpl = store.get_config(sid, "template")["body_hash"]
    prop = {"field": field, "value": value, "value_num": None, "unit": None, "group_label": "", "locator": locator,
            "extractor": "manual", "verify_status": verify_status, "verify_reason": "", "flags": flags or []}
    return store.insert_records(sid, src, tmpl, [prop])[0].id


def verified_text_record(client: TestClient, sid: str, name: str, *, value: str, snippet: str, page: int = 1,
                         field: str = "cmax", field_type: str = "number", column: int | None = None) -> str:
    """A text-locator record verified by the real verifier + QC flags, inserted into the ledger."""
    store = evstore.get_store()
    src = store.get_source(source_id(client, sid, name))
    doc = ingest.load_pages(src)
    tmpl = store.get_config(sid, "template")["body"]
    prop = {"field": field, "value": value, "unit": None, "group": "", "extractor": "manual",
            "locator": {"page": page, "kind": "text", "snippet": snippet, "column": column}}
    checked = check_proposals([prop], doc, tmpl)[0]
    assert checked["verify_status"] == "pass", checked["verify_reason"]
    return add_record(client, sid, src.id, checked["locator"], value=value, field=field, flags=checked["flags"])


def drop_layout(client: TestClient, sid: str, name: str) -> None:
    src = evstore.get_store().get_source(source_id(client, sid, name))
    Path(src.parse_path).with_name(f"{src.sha256}.layout.json").unlink()


def clear_render_cache() -> None:
    d = render_cache.render_dir(settings.data_dir)
    for p in d.iterdir():
        p.unlink()
