"""Plain-text extraction for corpus documents — the lightweight path only.

``.txt`` / ``.md`` are read as UTF-8. ``.pdf`` goes through pypdf (pure Python,
imported lazily so the test suite and CI never need it). There is deliberately
no MinerU / OCR dependency here: an image-only PDF yields little or no text and
is reported as such rather than silently indexed as empty.
"""
from __future__ import annotations

from pathlib import Path

TEXT_SUFFIXES: frozenset[str] = frozenset({".txt", ".md"})
PDF_SUFFIXES: frozenset[str] = frozenset({".pdf"})
SUPPORTED_SUFFIXES: frozenset[str] = TEXT_SUFFIXES | PDF_SUFFIXES
PAGE_BREAK = "\n\n"
MIN_EXTRACTED_CHARS = 200   # below this a "document" is almost certainly image-only or empty


class ExtractionError(ValueError):
    """The document could not be turned into usable text."""


def _extract_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:   # pragma: no cover - depends on the environment
        raise ExtractionError("PDF support needs pypdf (`pip install pypdf`)") from exc
    try:
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:  # noqa: BLE001 - pypdf raises a zoo of exception types
        raise ExtractionError(f"cannot read PDF {path.name}: {exc}") from exc
    return PAGE_BREAK.join(pages)


def extract_text(path: Path) -> str:
    """UTF-8 text of ``path``. Raises ExtractionError for unsupported types, unreadable
    files and documents that yield too little text to index."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise ExtractionError(f"cannot read {path.name}: {exc}") from exc
    elif suffix in PDF_SUFFIXES:
        text = _extract_pdf(path)
    else:
        raise ExtractionError(
            f"unsupported document type {suffix or '(none)'}; supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}")
    if len(text.strip()) < MIN_EXTRACTED_CHARS:
        raise ExtractionError(
            f"{path.name} yielded only {len(text.strip())} characters of text (image-only PDF? empty file?); "
            "not indexed")
    return text
