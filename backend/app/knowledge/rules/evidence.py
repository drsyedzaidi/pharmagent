"""Bind a rule to the passages that support it, by hash, and check that binding later.

``evidence_for_chunks`` turns chunk ids (from ``python -m app.knowledge search``) into
``EvidenceRef`` pointers: document hash, character span, and the SHA-256 of the span's text.
Nothing from the passage is copied. ``verify_ref`` re-derives the span from the stored
document text and compares hashes, so a replaced or edited document is detected.
"""
from __future__ import annotations

from pathlib import Path

from app.knowledge.index import load_index
from app.knowledge.manifest import ManifestError, load_manifest, load_text, sha256_text
from app.knowledge.rules.schema import EvidenceRef

PRESENT = "present"
STATUSES: tuple[str, ...] = (PRESENT, "corpus_empty", "missing_document", "missing_text", "span_out_of_range",
                             "text_changed")


class EvidenceError(ValueError):
    """A chunk id that cannot be turned into evidence."""


def evidence_for_chunks(root: Path, chunk_ids: list[str]) -> list[EvidenceRef]:
    index = load_index(root)
    docs = {d.sha256: d for d in index.manifest.documents}
    refs: list[EvidenceRef] = []
    seen: set[str] = set()
    for cid in chunk_ids:
        if cid in seen:
            continue
        seen.add(cid)
        chunk = index.chunk(cid)
        if chunk is None:
            raise EvidenceError(f"unknown chunk id {cid!r}; find ids with `python -m app.knowledge search`")
        doc = docs[chunk.doc_sha]
        refs.append(EvidenceRef(
            doc_sha=chunk.doc_sha, document=doc.title[:300], year=doc.year, source_type=doc.source_type,
            section=chunk.section[:160], start=chunk.start, end=chunk.end, text_sha256=chunk.chunk_hash))
    return refs


def verify_ref(root: Path, ref: EvidenceRef) -> str:
    """One of ``STATUSES``; ``present`` means the recorded span still hashes to ``text_sha256``."""
    manifest = load_manifest(root)
    if manifest.is_empty:
        return "corpus_empty"
    if manifest.get(ref.doc_sha) is None:
        return "missing_document"
    try:
        text = load_text(root, ref.doc_sha)
    except ManifestError:
        return "missing_text"
    if ref.end > len(text):
        return "span_out_of_range"
    return PRESENT if sha256_text(text[ref.start:ref.end]) == ref.text_sha256 else "text_changed"
