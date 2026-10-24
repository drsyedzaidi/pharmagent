"""Bind a rule to the passages that support it, by hash, and check that binding later.

``evidence_for_chunks`` turns chunk ids (from ``python -m app.knowledge search``) into
``EvidenceRef`` pointers: document hash, character span, and the SHA-256 of the span's text.
Nothing from the passage is copied (not even its section heading). ``verify_ref`` re-derives the span from the stored
document text and compares hashes, so a replaced or edited document is detected. It also re-checks the stored
attribution (title, year, source type) against the manifest, so a rule cannot keep citing a document whose
bibliographic record has since been corrected or swapped.
"""
from __future__ import annotations

import re
from pathlib import Path

from app.knowledge.index import load_index
from app.knowledge.manifest import ManifestError, load_manifest, load_text, sha256_text
from app.knowledge.rules.schema import EvidenceRef

PRESENT = "present"
STATUSES: tuple[str, ...] = (PRESENT, "corpus_empty", "missing_document", "missing_text", "span_out_of_range",
                             "text_changed", "metadata_changed")


COPIED_RUN_CHARS = 60      # a verbatim run this long from the cited passage is a quotation, not the owner's words


class EvidenceError(ValueError):
    """A chunk id that cannot be turned into evidence, or text that copies the cited passage."""


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def copied_run(own_words: dict[str, str], passages: list[str]) -> str | None:
    """The name of the first ``own_words`` field that repeats ``COPIED_RUN_CHARS`` or more consecutive
    characters of any passage (whitespace and case folded), else None."""
    squashed = [_squash(p) for p in passages]
    for name, text in own_words.items():
        t = _squash(text)
        for i in range(0, max(0, len(t) - COPIED_RUN_CHARS) + 1):
            window = t[i:i + COPIED_RUN_CHARS]
            if len(window) == COPIED_RUN_CHARS and any(window in p for p in squashed):
                return name
    return None


def evidence_for_chunks(root: Path, chunk_ids: list[str],
                        own_words: dict[str, str] | None = None) -> list[EvidenceRef]:
    """Hash pointers for ``chunk_ids``. With ``own_words`` (the rule's free-text fields) it refuses text that
    repeats a long run of a cited passage: the catalog is committed, so it must hold the owner's words only."""
    index = load_index(root)
    docs = {d.sha256: d for d in index.manifest.documents}
    refs: list[EvidenceRef] = []
    passages: list[str] = []
    seen: set[str] = set()
    for cid in chunk_ids:
        if cid in seen:
            continue
        seen.add(cid)
        chunk = index.chunk(cid)
        if chunk is None:
            raise EvidenceError(f"unknown chunk id {cid!r}; find ids with `python -m app.knowledge search`")
        doc = docs[chunk.doc_sha]
        passages.append(chunk.text)
        refs.append(EvidenceRef(
            doc_sha=chunk.doc_sha, document=doc.title[:300], year=doc.year, source_type=doc.source_type,
            start=chunk.start, end=chunk.end, text_sha256=chunk.chunk_hash))
    if own_words:
        offender = copied_run(own_words, passages)
        if offender:
            raise EvidenceError(f"--{offender} repeats {COPIED_RUN_CHARS}+ characters of the cited passage verbatim; "
                                "the catalog is committed, so state the rule in your own words")
    return refs


def verify_ref(root: Path, ref: EvidenceRef) -> str:
    """One of ``STATUSES``; ``present`` means the recorded span still hashes to ``text_sha256``."""
    manifest = load_manifest(root)
    if manifest.is_empty:
        return "corpus_empty"
    record = manifest.get(ref.doc_sha)
    if record is None:
        return "missing_document"
    try:
        text = load_text(root, ref.doc_sha)
    except ManifestError:
        return "missing_text"
    if ref.end > len(text):
        return "span_out_of_range"
    if sha256_text(text[ref.start:ref.end]) != ref.text_sha256:
        return "text_changed"                  # the more serious fact wins over a metadata mismatch
    stored = (record.title[:300], record.year, record.source_type)     # same truncation as evidence_for_chunks
    return PRESENT if (ref.document, ref.year, ref.source_type) == stored else "metadata_changed"
