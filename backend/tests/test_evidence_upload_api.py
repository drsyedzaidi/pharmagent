"""Evidence upload / criteria / template / snapshot endpoints over HTTP (keyless MockLLM).

Fixture pattern from test_api: a fresh in-memory Orchestrator per test, settings.data_dir
pointed at tmp_path and the evidence ledger configured ':memory:'.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.config import settings
from app.core.llm import MockLLM
from app.core.orchestrator import Orchestrator
from app.core.store import SessionStore
from app.evidence import ingest
from app.evidence import store as evstore
from tests import evidence_fixtures as fx

EV = "/api/sessions/{sid}/evidence"


@pytest.fixture(scope="module")
def papers(tmp_path_factory) -> dict[str, Path]:
    d = tmp_path_factory.mktemp("papers")
    paths = fx.build_all(d)
    paths["encrypted"] = fx.encrypted_pdf(d)
    paths["many"] = fx.many_pages_pdf(d, 301)
    return paths


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    evstore.configure(":memory:")
    main.orch = Orchestrator(llm=MockLLM(), clock=lambda c=itertools.count(): f"t{next(c)}",
                             store=SessionStore(":memory:"))
    token = settings.api_token
    yield TestClient(main.app)
    settings.api_token = token


def _sid(client) -> str:
    return client.post("/api/sessions").json()["id"]


def _upload(client, sid, blob: bytes | Path, name: str = "paper.pdf", **form):
    data = blob.read_bytes() if isinstance(blob, Path) else blob
    return client.post(EV.format(sid=sid) + "/upload",
                       files={"file": (name, data, "application/pdf")}, data=form)


def _ingest_entries(client, sid) -> list[dict]:
    audit = client.get(f"/api/sessions/{sid}/audit").json()
    return [e for e in audit["entries"] if e["tool"] == "ingest_evidence_source"]


def _err(r) -> str:
    return r.json()["error"]["message"]


def _files_under(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file()]


# ------------------------------------------------------------------ upload ---

def test_upload_paper_a(client, tmp_path, papers):
    sid = _sid(client)
    r = _upload(client, sid, papers["paper_a"], name="paper_a.pdf")
    assert r.status_code == 200, r.text
    body = r.json()
    src = body["source"]
    assert src["id"].startswith("src_") and src["pages"] == 1 and src["parse_status"] == "parsed"
    assert body["duplicate"] is False and src["filename"] == "paper_a.pdf"
    assert src["uploaded_by"] == "anonymous" and src["title"].startswith("Pharmacokinetics")
    assert "pdf_path" not in src and "parse_path" not in src      # server paths stay server-side
    # content-addressed storage + parse cache under data_dir/evidence
    assert (tmp_path / "evidence" / f"{src['sha256']}.pdf").read_bytes() == papers["paper_a"].read_bytes()
    assert (tmp_path / "evidence" / f"{src['sha256']}.pages.json").is_file()
    assert not any((tmp_path / "evidence" / "incoming").iterdir())  # transient upload removed
    entries = _ingest_entries(client, sid)
    assert len(entries) == 1 and entries[0]["actor"] == "anonymous" and entries[0]["agent"] == "evidence"
    summary = client.get(f"/api/sessions/{sid}/state").json()["evidence_summary"]
    assert summary["sources"] == 1 and summary["parsed"] == 1


def test_reupload_is_a_duplicate_without_new_source_or_audit_entry(client, papers):
    sid = _sid(client)
    first = _upload(client, sid, papers["paper_a"], name="a.pdf").json()
    again = _upload(client, sid, papers["paper_a"], name="renamed.pdf")
    assert again.status_code == 200
    assert again.json()["duplicate"] is True and again.json()["source"]["id"] == first["source"]["id"]
    assert again.json()["source"]["filename"] == "a.pdf"
    snap = client.get(EV.format(sid=sid)).json()
    assert len(snap["sources"]) == 1 and len(_ingest_entries(client, sid)) == 1


def test_same_pdf_in_two_sessions_is_two_sources_one_file(client, tmp_path, papers):
    s1, s2 = _sid(client), _sid(client)
    a = _upload(client, s1, papers["paper_a"]).json()["source"]
    b = _upload(client, s2, papers["paper_a"]).json()["source"]
    assert a["id"] != b["id"] and a["sha256"] == b["sha256"]
    assert len([p for p in (tmp_path / "evidence").glob("*.pdf")]) == 1


def test_rejects_non_pdf_suffix(client, papers):
    sid = _sid(client)
    r = _upload(client, sid, b"a,b\n1,2\n", name="data.csv")
    assert r.status_code == 400 and "PDF only" in _err(r)


def test_rejects_csv_renamed_as_pdf_by_magic(client, tmp_path, papers):
    sid = _sid(client)
    r = _upload(client, sid, b"a,b\n1,2\n", name="data.pdf")
    assert r.status_code == 400 and "not a PDF" in _err(r)
    assert _files_under(tmp_path) == []


def test_rejects_oversize_before_storing(client, tmp_path):
    sid = _sid(client)
    blob = b"%PDF-1.4\n" + b"\0" * (26 * 1024 * 1024)
    r = _upload(client, sid, blob, name="big.pdf")
    assert r.status_code == 413
    assert _files_under(tmp_path) == []


def test_rejects_encrypted(client, tmp_path, papers):
    sid = _sid(client)
    r = _upload(client, sid, papers["encrypted"], name="locked.pdf")
    assert r.status_code == 422 and "encrypted PDF" in _err(r)
    assert _files_under(tmp_path) == [] and _ingest_entries(client, sid) == []


def test_rejects_page_cap(client, tmp_path, papers):
    sid = _sid(client)
    r = _upload(client, sid, papers["many"], name="long.pdf")
    assert r.status_code == 422 and "page cap" in _err(r)
    assert _files_under(tmp_path) == []


def test_filename_is_sanitised_and_stored_by_hash(client, tmp_path, papers):
    sid = _sid(client)
    src = _upload(client, sid, papers["paper_b"], name="../../x.pdf").json()["source"]
    assert src["filename"] == "x.pdf"
    assert (tmp_path / "evidence" / f"{src['sha256']}.pdf").is_file()
    assert not (tmp_path / "x.pdf").exists() and not (tmp_path.parent / "x.pdf").exists()


def test_licence_note_is_recorded(client, papers):
    sid = _sid(client)
    src = _upload(client, sid, papers["paper_c"], licence_note="CC BY 4.0").json()["source"]
    assert src["licence_note"] == "CC BY 4.0" and src["decimal_style"] == ","


def test_other_sessions_sources_are_invisible(client, papers):
    s1, s2 = _sid(client), _sid(client)
    src = _upload(client, s1, papers["paper_a"]).json()["source"]
    assert client.get(EV.format(sid=s2) + f"/sources/{src['id']}/pdf").status_code == 404
    assert client.get(EV.format(sid=s2) + f"/sources/{src['id']}/pages/1").status_code == 404
    assert client.get(EV.format(sid=s1) + "/sources/src_nope/pdf").status_code == 404
    ok = client.get(EV.format(sid=s1) + f"/sources/{src['id']}/pdf")
    assert ok.status_code == 200 and ok.headers["content-type"].startswith("application/pdf")
    assert ok.content == papers["paper_a"].read_bytes()


def test_pages_endpoint(client, papers):
    sid = _sid(client)
    src = _upload(client, sid, papers["paper_c"]).json()["source"]
    base = EV.format(sid=sid) + f"/sources/{src['id']}/pages"
    assert client.get(base + "/0").status_code == 404
    assert client.get(base + "/3").status_code == 404
    page = client.get(base + "/1").json()
    assert page["page"] == 1 and page["pages"] == 2 and page["source_id"] == src["id"]
    assert "compound Y" in page["text"] and page["decimal_style"] == ","
    page2 = client.get(base + "/2").json()
    assert page2["page"] == 2 and any(t["caption"].startswith("Table 1") for t in page2["tables"])


def test_pages_fail_closed_when_the_pdf_changes_on_disk(client, tmp_path, papers):
    sid = _sid(client)
    src = _upload(client, sid, papers["paper_a"]).json()["source"]
    (tmp_path / "evidence" / f"{src['sha256']}.pdf").write_bytes(papers["paper_b"].read_bytes())
    r = client.get(EV.format(sid=sid) + f"/sources/{src['id']}/pages/1")
    assert r.status_code == 400 and "source file changed" in _err(r)


# ----------------------------------------------------------------- configs ---

CRITERIA = [
    {"id": "c1", "text": "human pharmacokinetic study", "kind": "inclusion", "required": True,
     "keywords": ["pharmacokinetic", "healthy", "subjects"]},
    {"id": "c2", "text": "in vitro or animal only", "kind": "exclusion", "required": False,
     "keywords": ["in vitro", "rat"]},
]
TEMPLATE = {"fields": [
    {"name": "cmax", "type": "number", "unit": "ng/mL", "required": True, "aliases": ["Cmax"]},
    {"name": "n", "type": "integer", "min": 1, "max": 10000},
    {"name": "formulation", "type": "text"},
]}


def test_criteria_saved_hashed_and_audited(client, papers):
    sid = _sid(client)
    r = client.post(EV.format(sid=sid) + "/criteria", json={"criteria": CRITERIA})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["kind"] == "criteria" and len(body["body_hash"]) == 64 and body["saved_by"] == "anonymous"
    assert [c["id"] for c in body["body"]] == ["c1", "c2"]
    snap = client.get(EV.format(sid=sid)).json()
    assert snap["criteria"]["body_hash"] == body["body_hash"] and snap["template"] is None
    assert snap["summary"]["criteria_hash"] == body["body_hash"]
    audit = client.get(f"/api/sessions/{sid}/audit").json()
    assert [e for e in audit["entries"] if e["tool"] == "save_evidence_config"]
    # re-save replaces the row; a different body gives a different hash
    r2 = client.post(EV.format(sid=sid) + "/criteria", json={"criteria": CRITERIA[:1]})
    assert r2.json()["body_hash"] != body["body_hash"]
    assert len(client.get(EV.format(sid=sid)).json()["criteria"]["body"]) == 1


@pytest.mark.parametrize("bad", [
    [],                                                                     # empty
    [{"id": "c1", "text": "x", "kind": "inclusion"}, {"id": "c1", "text": "y", "kind": "exclusion"}],
    [{"id": "c1", "text": "", "kind": "inclusion"}],                        # text 1..500
    [{"id": "c1", "text": "x" * 501, "kind": "inclusion"}],
    [{"id": "c1", "text": "x", "kind": "maybe"}],                           # kind enum
    [{"id": "c1", "text": "x", "kind": "inclusion", "required": "yes"}],    # required bool
    [{"id": "c1", "text": "x", "kind": "inclusion", "keywords": "pk"}],     # keywords list
    [{"id": "", "text": "x", "kind": "inclusion"}],
    "not a list",
])
def test_criteria_validation(client, bad):
    sid = _sid(client)
    r = client.post(EV.format(sid=sid) + "/criteria", json={"criteria": bad})
    assert r.status_code == 400, r.text
    assert client.get(EV.format(sid=sid)).json()["criteria"] is None


def test_template_saved_and_normalised(client):
    sid = _sid(client)
    r = client.post(EV.format(sid=sid) + "/template", json=TEMPLATE)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["kind"] == "template" and len(body["body_hash"]) == 64
    fields = {f["name"]: f for f in body["body"]["fields"]}
    assert fields["n"]["decimal"] == "auto" and fields["n"]["required"] is False
    assert fields["cmax"]["aliases"] == ["Cmax"] and fields["formulation"]["unit"] is None
    snap = client.get(EV.format(sid=sid)).json()
    assert snap["template"]["body_hash"] == body["body_hash"]
    assert snap["summary"]["template_hash"] == body["body_hash"]


@pytest.mark.parametrize("bad", [
    {"fields": []},
    {"fields": [{"name": "a", "type": "number"}, {"name": "a", "type": "text"}]},   # unique names
    {"fields": [{"name": "1abc", "type": "number"}]},                                # name regex
    {"fields": [{"name": "bad name", "type": "number"}]},
    {"fields": [{"name": "a", "type": "float"}]},                                    # type enum
    {"fields": [{"name": "a", "type": "number", "min": 5, "max": 1}]},               # min<=max
    {"fields": [{"name": "a", "type": "number", "decimal": ";"}]},
    {"fields": [{"name": "a", "type": "number", "aliases": "A"}]},
    {"fields": [{"name": f"f{i}", "type": "number"} for i in range(61)]},            # <= 60 fields
    {"nope": []},
    [],
])
def test_template_validation(client, bad):
    sid = _sid(client)
    r = client.post(EV.format(sid=sid) + "/template", json=bad)
    assert r.status_code == 400, r.text
    assert client.get(EV.format(sid=sid)).json()["template"] is None


# ------------------------------------------------------------- invariants ---

def test_legacy_csv_upload_is_untouched(client):
    sid = _sid(client)
    r = client.post(f"/api/sessions/{sid}/upload", files={"file": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 400 and "CSV only" in _err(r)


def test_audit_chain_verifies_after_evidence_actions(client, papers):
    sid = _sid(client)
    _upload(client, sid, papers["paper_a"])
    _upload(client, sid, papers["paper_d"])
    client.post(EV.format(sid=sid) + "/criteria", json={"criteria": CRITERIA})
    client.post(EV.format(sid=sid) + "/template", json=TEMPLATE)
    audit = client.get(f"/api/sessions/{sid}/audit").json()
    assert audit["integrity"]["chain_ok"] is True and len(_ingest_entries(client, sid)) == 2
    snap = client.get(EV.format(sid=sid)).json()
    assert snap["counts"]["sources"] == 2 and snap["summary"]["sources"] == 2


def test_evidence_routes_require_an_owned_session(client, papers):
    settings.api_token = "secret-token"
    h = {"Authorization": "Bearer secret-token"}
    sid = client.post("/api/sessions", headers=h).json()["id"]
    assert client.get(EV.format(sid=sid)).status_code == 401
    assert client.get(EV.format(sid=sid), headers=h).status_code == 200
    assert client.get(EV.format(sid="sess_nope"), headers=h).status_code == 404
    r = client.post(EV.format(sid=sid) + "/upload", headers=h,
                    files={"file": ("a.pdf", papers["paper_a"].read_bytes(), "application/pdf")})
    assert r.json()["source"]["uploaded_by"].startswith("token:")


def test_ingest_tool_refuses_paths_outside_the_evidence_dir(client, tmp_path, papers):
    """The tool is reachable through the registry: it must not ingest arbitrary files."""
    sid = _sid(client)
    stray = tmp_path / "stray.pdf"
    stray.write_bytes(papers["paper_a"].read_bytes())
    with pytest.raises(ValueError, match="evidence directory"):
        main.orch.run_tool(sid, "ingest_evidence_source", "evidence",
                           {"path": str(stray), "sha256": "0" * 64, "filename": "stray.pdf"})
    moved = ingest.evidence_dir(tmp_path) / "wrong.pdf"
    moved.write_bytes(papers["paper_a"].read_bytes())
    with pytest.raises(ValueError, match="sha256"):
        main.orch.run_tool(sid, "ingest_evidence_source", "evidence",
                           {"path": str(moved), "sha256": "0" * 64, "filename": "wrong.pdf"})
