"""Owner decision: exports are provenance-bearing outputs, without altering standardized payloads.
CSV and define.xml bytes are unchanged; a companion .provenance.json manifest (status, assurance / audit-integrity mode,
catalog version, canonical-JSON digest, run identity, export checksum) is served for every download; HTTP downloads also
carry provenance headers; the CDISC zip embeds the manifest; control streams get a comment line."""
from __future__ import annotations

import hashlib
import io
import itertools
import json
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.config import settings
from app.core import exporters
from app.core.llm import MockLLM
from app.core.orchestrator import Orchestrator
from app.core.store import SessionStore

SAMPLE = str(Path(__file__).parent.parent / "sample_data" / "oral_pk.csv")
DIGEST = "conventions_catalog_canonical_json_sha256"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    main.orch = Orchestrator(llm=MockLLM(), clock=lambda c=itertools.count(): f"t{next(c)}",
                             store=SessionStore(":memory:"))
    return TestClient(main.app)


def _nca_session(client) -> str:
    sid = client.post("/api/sessions").json()["id"]
    main.orch.run_tool(sid, "load_dataset", "data_manager", {"path": SAMPLE})
    main.orch.run_tool(sid, "compute_nca", "nca", {})
    return sid


def _record(sid) -> dict:
    return dict(main.orch.get_session(sid).state.run_provenance)


def _check_manifest(m: dict, sid: str, payload: bytes, status="verified"):
    assert m["session_id"] == sid and m["run_provenance_status"] == status
    assert m["audit_integrity_mode"] == "hash_only" and m["run_provenance_assurance"] == "hash_only"
    assert m["export"]["sha256"] == _sha(payload)
    if status == "verified":
        rec = _record(sid)
        assert m["conventions_catalog_version"] == rec["conventions_catalog_version"]
        assert m[DIGEST] == rec[DIGEST] and m["run_identity"]["git_sha"] == rec["git_sha"]


def test_csv_is_unchanged_and_has_provenance_headers_and_a_manifest(client):
    sid = _nca_session(client)
    r = client.get(f"/api/sessions/{sid}/export/nca")
    assert r.status_code == 200
    assert r.content == exporters.export_csv(main.orch.get_session(sid).state, "nca").encode("utf-8")
    assert r.headers["X-PharmAgent-Run-Provenance-Status"] == "verified"
    assert r.headers["X-PharmAgent-Conventions-Catalog-Digest"] == _record(sid)[DIGEST]
    assert r.headers["X-PharmAgent-Conventions-Catalog-Version"] == _record(sid)["conventions_catalog_version"]
    assert r.headers["X-PharmAgent-Run-Provenance-Assurance"] == "hash_only"
    m = client.get(f"/api/sessions/{sid}/provenance/export/nca").json()
    _check_manifest(m, sid, r.content)
    assert m["export"]["kind"] == "nca"


def test_cdisc_zip_embeds_the_manifest_and_leaves_define_xml_unchanged(client):
    from app.core import cdisc

    sid = _nca_session(client)
    r = client.get(f"/api/sessions/{sid}/cdisc")
    assert r.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    names = zf.namelist()
    assert "provenance.json" in names
    sess = main.orch.get_session(sid)
    roles = (sess.state.dataset_metadata or {}).get("detected_roles", {})
    original = zipfile.ZipFile(io.BytesIO(cdisc.build_package(sess.state, sess.ctx.dataset_store[sess.state.dataset_id],
                                                              roles)))
    for name in original.namelist():
        assert zf.read(name) == original.read(name), name              # standard members untouched
    inner = json.loads(zf.read("provenance.json"))
    assert inner["run_provenance_status"] == "verified" and inner[DIGEST] == _record(sid)[DIGEST]
    assert inner["members"] == {n: _sha(original.read(n)) for n in original.namelist()}
    assert r.headers["X-PharmAgent-Run-Provenance-Status"] == "verified"
    companion = client.get(f"/api/sessions/{sid}/provenance/cdisc").json()
    _check_manifest(companion, sid, r.content)


def test_control_stream_gets_a_comment_and_a_manifest(client, monkeypatch):
    sid = client.post("/api/sessions").json()["id"]
    monkeypatch.setattr(main.exporters, "export_control", lambda state, kind: ("$PROBLEM test\n", "ctl"))
    r = client.get(f"/api/sessions/{sid}/export/control/nonmem")
    assert r.status_code == 200
    first, rest = r.text.split("\n", 1)
    assert first.startswith(";") and "verified" in first and _record(sid)[DIGEST] in first
    assert rest == "$PROBLEM test\n"
    m = client.get(f"/api/sessions/{sid}/provenance/export/control/nonmem").json()
    _check_manifest(m, sid, r.content)


def test_report_download_has_headers_and_a_manifest(client):
    sid = _nca_session(client)
    main.orch.run_tool(sid, "generate_report", "report", {})
    path = main.orch.get_session(sid).state.report_path
    r = client.get(f"/api/sessions/{sid}/report/{Path(path).name}")
    assert r.status_code == 200 and r.headers["X-PharmAgent-Run-Provenance-Status"] == "verified"
    _check_manifest(client.get(f"/api/sessions/{sid}/provenance/report").json(), sid, Path(path).read_bytes())


def test_a_legacy_session_export_says_legacy_unverified(client, tmp_path):
    sid = _nca_session(client)
    sess = main.orch.get_session(sid)
    sess.state = sess.state.model_copy(update={"run_provenance": None})
    from app.core.audit import hash_payload
    g = sess.audit.entries[0]
    g.outputs_hash = hash_payload({"session_id": sid})
    from app.core.audit import AuditChain
    chain = AuditChain.from_list(sess.audit.to_list())
    for i, e in enumerate(chain.entries):                              # a consistent pre-marker chain
        if i:
            e.prev_hash = chain.entries[i - 1].entry_hash
        e.entry_hash = e.compute_hash()
    sess.audit = chain
    r = client.get(f"/api/sessions/{sid}/export/nca")
    assert r.status_code == 200 and r.headers["X-PharmAgent-Run-Provenance-Status"] == "legacy_unverified"
    m = client.get(f"/api/sessions/{sid}/provenance/export/nca").json()
    assert m["run_provenance_status"] == "legacy_unverified"
    assert m["conventions_catalog_version"] is None and m[DIGEST] is None and m["run_identity"] is None
    assert m["export"]["sha256"] == _sha(r.content)


def test_manifests_are_refused_for_a_quarantined_session(client):
    from app.core.provenance import collect_provenance

    sid = _nca_session(client)
    sess = main.orch.sessions[sid]
    forged = tuple({**collect_provenance(), DIGEST: "0" * 64}.items())
    sess.state = sess.state.model_copy(update={"run_provenance": forged})
    for path in ("/provenance/export/nca", "/provenance/cdisc", "/provenance/report",
                 "/provenance/report/272", "/provenance/memo", "/provenance/export/control/nonmem"):
        assert client.get(f"/api/sessions/{sid}{path}").status_code == 423, path
