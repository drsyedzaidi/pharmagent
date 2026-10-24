"""Section-aware chunking with exact character offsets.

A document is first split into sections at heading-like lines (markdown ``#``
headings, numbered headings such as ``3.2 Shrinkage``, short ALL-CAPS lines).
Each section is then cut into overlapping word windows of roughly 500 tokens
(``CHUNK_WORDS`` words ≈ 500 BPE tokens for English prose). Every chunk keeps
its ``[start, end)`` offsets into the document text, so a chunk can always be
re-derived from the stored text and a quote can be located exactly.

Chunking is deterministic: the same text always yields the same chunks, ids and
hashes, which is what lets an audit entry name the evidence it saw.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.knowledge.manifest import sha256_text

CHUNK_WORDS = 380          # ≈ 500 tokens of English prose
CHUNK_OVERLAP_WORDS = 60   # carried over between adjacent windows of one section
MAX_CHUNK_CHARS = 3600     # hard cap so a run of very long "words" cannot blow up a chunk
MIN_SECTION_WORDS = 40     # a shorter section is merged into the following one
MAX_HEADING_CHARS = 90

_WORD_RE = re.compile(r"\S+")
_MD_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+\S")
_NUM_HEADING_RE = re.compile(r"^\s*(?:\d+(?:\.\d+){0,3}|[IVXLC]+\.|[A-Z]\.)\s+[A-Z][^.]{2,}$")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str      # "<doc_sha[:12]>:<ordinal>"
    doc_sha: str
    ordinal: int
    start: int         # char offset into the document text (inclusive)
    end: int           # char offset (exclusive)
    section: str       # heading of the section the chunk came from ("" when none)
    text: str
    chunk_hash: str    # sha256 of ``text``


def is_heading(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > MAX_HEADING_CHARS:
        return False
    if _MD_HEADING_RE.match(line):
        return True
    if _NUM_HEADING_RE.match(s):
        return True
    letters = [c for c in s if c.isalpha()]
    return len(letters) >= 4 and all(c.isupper() for c in letters) and len(s.split()) <= 10


def _clean_heading(line: str) -> str:
    return line.strip().lstrip("#").strip()


def split_sections(text: str) -> list[tuple[str, int, int]]:
    """``(heading, start, end)`` spans covering the text in order. Tiny sections are
    merged forward so headings alone never become chunks."""
    spans: list[tuple[str, int, int]] = []
    heading, start = "", 0
    pos = 0
    for line in text.splitlines(keepends=True):
        if is_heading(line) and pos > start:
            spans.append((heading, start, pos))
            heading, start = _clean_heading(line), pos
        elif is_heading(line):
            heading = _clean_heading(line)
        pos += len(line)
    if pos > start or not spans:
        spans.append((heading, start, len(text)))

    def n_words(a: int, b: int) -> int:
        return len(_WORD_RE.findall(text[a:b]))

    merged: list[tuple[str, int, int]] = []
    for h, s, e in spans:
        if merged:
            ph, ps, pe = merged[-1]
            if n_words(ps, pe) < MIN_SECTION_WORDS:      # previous too small: absorb this one
                merged[-1] = (ph or h, ps, e)
                continue
            if n_words(s, e) < MIN_SECTION_WORDS:        # this one too small: fold into previous
                merged[-1] = (ph, ps, e)
                continue
        merged.append((h, s, e))
    return merged


def _windows(n_words: int) -> list[tuple[int, int]]:
    """Word-index windows ``[i, j)`` with overlap; the last window is never a tiny tail."""
    if n_words <= 0:
        return []
    if n_words <= CHUNK_WORDS:
        return [(0, n_words)]
    step = CHUNK_WORDS - CHUNK_OVERLAP_WORDS
    out: list[tuple[int, int]] = []
    i = 0
    while True:
        j = min(i + CHUNK_WORDS, n_words)
        out.append((i, j))
        if j >= n_words:
            break
        i += step
    # a final window shorter than the overlap is already covered by the previous one
    if len(out) >= 2 and out[-1][1] - out[-1][0] <= CHUNK_OVERLAP_WORDS:
        out.pop()
        out[-1] = (out[-1][0], n_words)
    return out


def chunk_document(doc_sha: str, text: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    for heading, s, e in split_sections(text):
        words = list(_WORD_RE.finditer(text, s, e))
        for i, j in _windows(len(words)):
            c_start, c_end = words[i].start(), words[j - 1].end()
            if c_end - c_start > MAX_CHUNK_CHARS:
                c_end = c_start + MAX_CHUNK_CHARS
            body = text[c_start:c_end]
            if not body.strip():
                continue
            ordinal = len(chunks)
            chunks.append(Chunk(
                chunk_id=f"{doc_sha[:12]}:{ordinal}", doc_sha=doc_sha, ordinal=ordinal,
                start=c_start, end=c_end, section=heading, text=body, chunk_hash=sha256_text(body)))
    return chunks
