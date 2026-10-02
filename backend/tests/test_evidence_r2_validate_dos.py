"""Upload validation is hostile parsing: counting pages of a tiny PDF whose catalog sits in a huge
FlateDecode object stream must not run the server process for minutes. Validation runs in the same
killable child as the parse (wall clock + RSS caps), so the bomb is refused within the cap and the
main process never inflates it."""
from __future__ import annotations

import time

import pytest

from app.evidence import ingest
from app.evidence.models import IngestError
from tests import evidence_fixtures as fx
from tests import evidence_trickpdf as T

PAD = 24 * 1024 * 1024               # ~24 KB on disk; ~20 s of pdfminer work in-process before the fix


@pytest.fixture
def bomb(tmp_path):
    path = tmp_path / "objstm_bomb.pdf"
    path.write_bytes(T.objstm_bomb_pdf(PAD))
    return path


def test_objstm_bomb_validation_is_bounded(bomb, monkeypatch):
    monkeypatch.setattr(ingest, "VALIDATE_WALL_S", 2.0)
    start = time.monotonic()
    with pytest.raises(IngestError):
        ingest.validate_pdf_file(bomb)
    assert time.monotonic() - start < 12.0


def test_objstm_bomb_ingest_is_bounded(bomb, tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "VALIDATE_WALL_S", 2.0)
    start = time.monotonic()
    with pytest.raises(IngestError):
        ingest.validate_pdf_bytes(bomb.read_bytes())
    assert time.monotonic() - start < 12.0


def test_ordinary_pdfs_still_validate(tmp_path):
    paths = fx.build_all(tmp_path)
    assert ingest.validate_pdf_file(paths["paper_c"]) == 2
    assert ingest.validate_pdf_bytes(paths["paper_a"].read_bytes()) == 1


def test_not_a_pdf_and_oversize_are_refused_before_any_child(tmp_path):
    with pytest.raises(IngestError, match="not a PDF"):
        ingest.validate_pdf_bytes(b"hello")
    with pytest.raises(IngestError, match="too large"):
        ingest.validate_pdf_bytes(ingest.PDF_MAGIC + b"0" * (ingest.MAX_BYTES + 1))
