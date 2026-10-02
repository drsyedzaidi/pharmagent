"""Ingest: validation (magic, size, encryption, page cap), content-addressed storage, cache, fail-closed reload."""
from __future__ import annotations

import hashlib
import json
import os

import pytest
from fpdf import FPDF

from app.evidence import ingest
from app.evidence import store as evstore
from app.evidence.models import IngestError
from tests import evidence_fixtures as fx


@pytest.fixture(scope="module")
def papers(tmp_path_factory) -> dict:
    d = tmp_path_factory.mktemp("papers")
    paths = fx.build_all(d)
    paths["encrypted"] = fx.encrypted_pdf(d)
    paths["no_text"] = fx.no_text_pdf(d)
    paths["many"] = fx.many_pages_pdf(d, 301)
    return paths


@pytest.fixture
def st():
    return evstore.configure(":memory:")


# ------------------------------------------------------------- validation ---

def test_rejects_non_pdf_bytes():
    with pytest.raises(IngestError, match="not a PDF"):
        ingest.validate_pdf_bytes(b"hello world, definitely not a pdf")
    with pytest.raises(IngestError, match="not a PDF"):
        ingest.validate_pdf_bytes(b"")


def test_rejects_encrypted(papers):
    with pytest.raises(IngestError, match="encrypted PDF"):
        ingest.validate_pdf_file(papers["encrypted"])


def test_rejects_page_cap(papers):
    with pytest.raises(IngestError, match="page cap"):
        ingest.validate_pdf_file(papers["many"])


def test_rejects_too_large():
    blob = b"%PDF-1.4\n" + b"\0" * (ingest.MAX_BYTES + 1)
    with pytest.raises(IngestError, match="too large"):
        ingest.validate_pdf_bytes(blob)


def test_rejects_garbage_with_pdf_magic():
    with pytest.raises(IngestError, match="unreadable PDF"):
        ingest.validate_pdf_bytes(b"%PDF-1.7\n garbage garbage garbage")


def test_validate_returns_page_count(papers):
    assert ingest.validate_pdf_file(papers["paper_c"]) == 2


# ---------------------------------------------------------------- storage ---

def test_write_by_sha256_is_content_addressed_and_idempotent(tmp_path, papers):
    data = papers["paper_a"].read_bytes()
    sha, path = ingest.write_by_sha256(data, tmp_path)
    assert sha == hashlib.sha256(data).hexdigest()
    assert path == tmp_path / "evidence" / f"{sha}.pdf" and path.read_bytes() == data
    os.utime(path, (1_000_000, 1_000_000))
    sha2, path2 = ingest.write_by_sha256(data, tmp_path)
    assert (sha2, path2) == (sha, path) and path.stat().st_mtime == 1_000_000   # untouched


def test_parse_and_cache_writes_pages_json_and_reuses_it(tmp_path, papers):
    sha, pdf_path = ingest.write_by_sha256(papers["paper_a"].read_bytes(), tmp_path)
    doc, parse_path = ingest.parse_and_cache(pdf_path, tmp_path)
    assert parse_path == tmp_path / "evidence" / f"{sha}.pages.json"
    assert json.loads(parse_path.read_text())["sha256"] == sha == doc["sha256"]
    os.utime(parse_path, (1_000_000, 1_000_000))
    doc2, _ = ingest.parse_and_cache(pdf_path, tmp_path)
    assert doc2 == doc and parse_path.stat().st_mtime == 1_000_000


# ------------------------------------------------------------- filenames ---

@pytest.mark.parametrize("raw,clean", [
    ("../../x.pdf", "x.pdf"),
    ("..\\..\\win.pdf", "win.pdf"),
    ("my paper (1).pdf", "my_paper__1_.pdf"),
    ("Étude clinique.PDF", "_tude_clinique.PDF"),
    ("", "upload.pdf"),
    ("....", "upload.pdf"),
    ("a" * 300 + ".pdf", "a" * 196 + ".pdf"),
])
def test_sanitise_filename(raw, clean):
    assert ingest.sanitise_filename(raw) == clean


# ----------------------------------------------------------------- ingest ---

def test_ingest_pdf_end_to_end_and_idempotent(tmp_path, st, papers):
    data = papers["paper_a"].read_bytes()
    src = ingest.ingest_pdf(data, "../paper a.pdf", "s1", "tester", data_dir=tmp_path, store=st)
    assert src.parse_status == "parsed" and src.pages == 1 and src.text_chars > 0
    assert src.filename == "paper_a.pdf" and src.title.startswith("Pharmacokinetics of drug X")
    assert src.decimal_style == "." and src.metadata["author"] == "Evidence Fixtures"
    assert src.pdf_path.endswith(f"{src.sha256}.pdf") and src.parse_path.endswith(".pages.json")
    again = ingest.ingest_pdf(data, "renamed.pdf", "s1", "tester", data_dir=tmp_path, store=st)
    assert again.id == src.id and again.filename == "paper_a.pdf"
    assert len(st.list_sources("s1")) == 1
    doc = ingest.load_pages(src)
    assert doc["sha256"] == src.sha256 and doc["pages"][0]["n"] == 1


def test_ingest_no_text_pdf_is_recorded_as_no_text(tmp_path, st, papers):
    src = ingest.ingest_pdf(papers["no_text"].read_bytes(), "scan.pdf", "s1", "tester",
                            data_dir=tmp_path, store=st)
    assert src.parse_status == "no_text" and src.text_chars == 0


def test_ingest_decimal_comma_source(tmp_path, st, papers):
    src = ingest.ingest_pdf(papers["paper_c"].read_bytes(), "c.pdf", "s1", "tester",
                            data_dir=tmp_path, store=st)
    assert src.decimal_style == "," and src.pages == 2


def test_ingest_rejects_invalid_before_writing(tmp_path, st):
    with pytest.raises(IngestError):
        ingest.ingest_pdf(b"nope", "x.pdf", "s1", "tester", data_dir=tmp_path, store=st)
    assert not (tmp_path / "evidence").exists() or not any((tmp_path / "evidence").iterdir())
    assert st.list_sources("s1") == []


def test_title_falls_back_to_first_text_line_and_is_capped(tmp_path, st):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=11)
    pdf.cell(0, 8, "First line title " + "x" * 400, new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, "second line")
    data = pdf.output()
    src = ingest.ingest_pdf(bytes(data), "untitled.pdf", "s1", "tester", data_dir=tmp_path, store=st)
    assert src.title.startswith("First line title") and len(src.title) <= 200


def test_load_pages_fails_closed_when_pdf_bytes_change(tmp_path, st, papers):
    src = ingest.ingest_pdf(papers["paper_d"].read_bytes(), "d.pdf", "s1", "tester",
                            data_dir=tmp_path, store=st)
    assert ingest.load_pages(src)["sha256"] == src.sha256
    with open(src.pdf_path, "ab") as fh:
        fh.write(b"\n% tampered\n")
    with pytest.raises(ValueError, match="source file changed"):
        ingest.load_pages(src)


def test_load_pages_fails_closed_when_pages_json_sha_mismatch(tmp_path, st, papers):
    src = ingest.ingest_pdf(papers["paper_d"].read_bytes(), "d.pdf", "s1", "tester",
                            data_dir=tmp_path, store=st)
    doc = json.loads(open(src.parse_path).read())
    doc["sha256"] = "0" * 64
    open(src.parse_path, "w").write(json.dumps(doc))
    with pytest.raises(ValueError, match="parse cache mismatch"):
        ingest.load_pages(src)
