"""A stored pdf_path / parse_path is validated against the evidence data directory BEFORE the file is hashed,
read or served: a ledger row that points outside it (a temp file, or a symlink inside the directory that leads
out) is refused with 409 and the file is never opened.

The ledger is internal, so this is defence in depth: the sha256 check would also catch a wrong file, but only
after reading it, and reading is what a forged path must not get.
"""
from __future__ import annotations

import builtins
import io
import os

import pytest

from app.config import settings
from app.evidence import ingest, relayout
from app.evidence import store as evstore
from app.evidence.path_guard import OutsideDataDirError, confine
from tests.evidence_image_support import EV, build_papers, make_client, meta, record_of, session_with, source_id


@pytest.fixture(scope="module")
def papers(tmp_path_factory):
    return build_papers(tmp_path_factory)


@pytest.fixture
def client(tmp_path, monkeypatch):
    yield from make_client(tmp_path, monkeypatch)


class OpenSpy:
    """Records every attempt to open a file whose resolved path is in *watched* (open, io.open, os.open,
    the sha256 helper)."""

    def __init__(self, monkeypatch, watched: list) -> None:
        self.hits: list[str] = []
        self.watched = {os.path.realpath(str(p)) for p in watched}
        self._real = {"open": builtins.open, "io": io.open, "os": os.open}
        monkeypatch.setattr(builtins, "open", self._wrap(self._real["open"]))
        monkeypatch.setattr(io, "open", self._wrap(self._real["io"]))
        monkeypatch.setattr(os, "open", self._wrap(self._real["os"]))
        real_sha = ingest.file_sha256
        monkeypatch.setattr(ingest, "file_sha256", lambda p, *a, **k: (self._note(p), real_sha(p, *a, **k))[1])

    def _note(self, path) -> None:
        try:
            real = os.path.realpath(os.fspath(path))
        except TypeError:                       # a file descriptor
            return
        if real in self.watched:
            self.hits.append(real)

    def _wrap(self, real):
        def opener(file, *a, **k):
            self._note(file)
            return real(file, *a, **k)
        return opener


def _setup(client, papers):
    sid = session_with(client, {"paper_a": papers["paper_a"].read_bytes()})
    rid = record_of(client, sid, "paper_a", "cmax", "3,412")["id"]
    src = evstore.get_store().get_source(source_id(client, sid, "paper_a"))
    return sid, rid, src


def _point(src, *, pdf: str | None = None, parse: str | None = None) -> None:
    con = evstore.get_store()._conn
    con.execute("UPDATE evidence_source SET pdf_path=?, parse_path=? WHERE id=?",
                (pdf or src.pdf_path, parse or src.parse_path, src.id))
    con.commit()


def _outside_copy(tmp_path_factory, src):
    d = tmp_path_factory.mktemp("elsewhere")
    pdf = d / f"{src.sha256}.pdf"                         # a perfectly valid copy: the sha256 would match
    pdf.write_bytes(open(src.pdf_path, "rb").read())
    pages = d / f"{src.sha256}.pages.json"
    pages.write_bytes(open(src.parse_path, "rb").read())
    return pdf, pages


# ---------------------------------------------------------------- the guard ---

def test_confine_returns_the_resolved_path_inside_the_root(tmp_path):
    (tmp_path / "a").mkdir()
    f = tmp_path / "a" / "x.pdf"
    f.write_bytes(b"x")
    assert confine(f, tmp_path) == f.resolve()
    assert confine(tmp_path / "a" / ".." / "a" / "x.pdf", tmp_path) == f.resolve()
    assert confine(tmp_path / "missing.pdf", tmp_path) == (tmp_path / "missing.pdf").resolve()   # absent is not outside


def test_confine_refuses_outside_dotdot_symlink_and_nul(tmp_path, caplog):
    root = tmp_path / "root"
    root.mkdir()
    out = tmp_path / "out.pdf"
    out.write_bytes(b"x")
    (root / "link.pdf").symlink_to(out)
    (root / "dirlink").symlink_to(tmp_path)
    for bad in (out, root / ".." / "out.pdf", root / "link.pdf", root / "dirlink" / "out.pdf", "/etc/hosts",
                str(root / "x\x00.pdf"), ""):
        with caplog.at_level("WARNING"), pytest.raises(OutsideDataDirError):
            confine(bad, root)
    assert "outside the evidence directory" in caplog.text


def test_a_sibling_directory_with_the_same_prefix_is_outside(tmp_path):
    root = tmp_path / "evidence"
    sibling = tmp_path / "evidence-other"
    root.mkdir()
    sibling.mkdir()
    (sibling / "x.pdf").write_bytes(b"x")
    with pytest.raises(OutsideDataDirError):
        confine(sibling / "x.pdf", root)


# ------------------------------------------------------------ image routes ---

def test_image_meta_refuses_a_pdf_path_outside_without_opening_it(client, papers, tmp_path_factory, monkeypatch):
    sid, rid, src = _setup(client, papers)
    pdf, _ = _outside_copy(tmp_path_factory, src)
    _point(src, pdf=str(pdf))
    spy = OpenSpy(monkeypatch, [pdf])
    r = meta(client, sid, rid)
    assert r.status_code == 409, r.text
    assert "outside the evidence directory" in r.json()["error"]["message"] and str(pdf) not in r.text
    assert spy.hits == []


def test_image_meta_refuses_a_symlink_inside_the_directory_that_leads_out(client, papers, tmp_path_factory,
                                                                       monkeypatch, tmp_path):
    sid, rid, src = _setup(client, papers)
    pdf, _ = _outside_copy(tmp_path_factory, src)
    link = tmp_path / "evidence" / "innocent.pdf"
    link.symlink_to(pdf)
    _point(src, pdf=str(link))
    spy = OpenSpy(monkeypatch, [pdf])
    r = meta(client, sid, rid)
    assert r.status_code == 409 and spy.hits == []


def test_image_route_refuses_a_parse_path_outside_without_opening_it(client, papers, tmp_path_factory, monkeypatch):
    sid, rid, src = _setup(client, papers)
    _, pages = _outside_copy(tmp_path_factory, src)
    _point(src, parse=str(pages))
    spy = OpenSpy(monkeypatch, [pages])
    r = client.get(EV.format(sid=sid) + f"/records/{rid}/image", params={"sha": "x"})
    assert r.status_code == 409 and spy.hits == []


def test_a_legitimate_source_is_still_served(client, papers):
    sid, rid, _ = _setup(client, papers)
    assert meta(client, sid, rid).status_code == 200


# ------------------------------------------------------------ other routes ---

def test_source_pdf_download_never_serves_a_path_outside(client, papers, tmp_path_factory, monkeypatch):
    sid, _, src = _setup(client, papers)
    pdf, _ = _outside_copy(tmp_path_factory, src)
    _point(src, pdf=str(pdf))
    spy = OpenSpy(monkeypatch, [pdf])
    r = client.get(EV.format(sid=sid) + f"/sources/{src.id}/pdf")
    assert r.status_code == 409 and spy.hits == [] and r.headers["content-type"].startswith("application/json")


def test_source_page_text_refuses_a_path_outside(client, papers, tmp_path_factory, monkeypatch):
    sid, _, src = _setup(client, papers)
    pdf, _ = _outside_copy(tmp_path_factory, src)
    _point(src, pdf=str(pdf))
    spy = OpenSpy(monkeypatch, [pdf])
    r = client.get(EV.format(sid=sid) + f"/sources/{src.id}/pages/1")
    assert r.status_code == 409 and spy.hits == []


def test_layout_rebuild_refuses_a_path_outside_and_never_parses_it(client, papers, tmp_path_factory, monkeypatch):
    sid, _, src = _setup(client, papers)
    pdf, _ = _outside_copy(tmp_path_factory, src)
    _point(src, pdf=str(pdf))
    spy = OpenSpy(monkeypatch, [pdf])
    out = relayout.rebuild_layout(evstore.get_store().get_source(src.id), sid, data_dir=settings.data_dir)
    assert out["outcome"] == "unreadable" and spy.hits == []


# ----------------------------------------------------------- ingest helpers ---

def test_load_pages_and_source_file_intact_confine_when_given_the_data_dir(client, papers, tmp_path_factory, tmp_path):
    sid, _, src = _setup(client, papers)
    assert ingest.load_pages(src, tmp_path)["sha256"] == src.sha256 and ingest.source_file_intact(src, tmp_path)
    pdf, _ = _outside_copy(tmp_path_factory, src)
    _point(src, pdf=str(pdf))
    moved = evstore.get_store().get_source(src.id)
    with pytest.raises(OutsideDataDirError):
        ingest.load_pages(moved, tmp_path)
    assert ingest.source_file_intact(moved, tmp_path) is False
    assert ingest.source_file_intact(moved) is True       # unconfined callers (no data_dir) behave as before


def test_relayout_endpoint_refuses_a_path_outside_and_records_no_marker(client, papers, tmp_path_factory, monkeypatch):
    sid, _, src = _setup(client, papers)
    pdf, pages = _outside_copy(tmp_path_factory, src)
    _point(src, pdf=str(pdf), parse=str(pages))
    before = sorted(p.name for p in pages.parent.iterdir())
    spy = OpenSpy(monkeypatch, [pdf, pages])
    r = client.post(EV.format(sid=sid) + f"/sources/{src.id}/relayout")
    assert r.status_code == 200, r.text
    assert r.json()["outcome"] == "unreadable" and "outside the evidence directory" in r.json()["reason"]
    assert spy.hits == [] and sorted(p.name for p in pages.parent.iterdir()) == before   # nothing written beside it
