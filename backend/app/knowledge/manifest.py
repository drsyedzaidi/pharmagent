"""Content-addressed corpus manifest.

``<root>/manifest.json`` lists every document the owner has added: its original
path, SHA-256, title, year, source type and who/when it was added. The corpus
version is the SHA-256 over the sorted document hashes, so adding or removing
any document changes it and every mikebrain result records exactly which corpus
it was answered from. The extracted text of each document is kept beside the
manifest as ``<root>/text/<sha256>.txt`` (never committed — the data dir is
git-ignored).

The stored ``corpus_version`` is informational only: it is recomputed on every
load and never trusted.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

MANIFEST_FILE = "manifest.json"
TEXT_DIR = "text"
SCHEMA_VERSION = 1
SOURCE_TYPES: tuple[str, ...] = ("book", "paper", "guideline")


class ManifestError(ValueError):
    """A manifest operation that cannot proceed (duplicate, unknown document, bad field)."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class DocumentRecord:
    path: str                 # original path as supplied by the owner (provenance)
    sha256: str               # of the original file bytes
    title: str
    year: int | None
    source_type: str          # book | paper | guideline
    added_by: str
    added_at: str             # ISO-8601 UTC

    def validate(self) -> None:
        if self.source_type not in SOURCE_TYPES:
            raise ManifestError(f"source_type must be one of {', '.join(SOURCE_TYPES)}; got {self.source_type!r}")
        if not self.sha256 or len(self.sha256) != 64:
            raise ManifestError("sha256 must be a 64-hex digest")
        if not self.title.strip():
            raise ManifestError("title is required")
        if self.year is not None and not (1800 <= int(self.year) <= 2200):
            raise ManifestError(f"year out of range: {self.year!r}")


def corpus_version_of(doc_hashes: list[str] | tuple[str, ...]) -> str:
    """SHA-256 over the sorted document hashes, one per line. Order-independent, so
    the same set of documents always yields the same version."""
    return sha256_text("\n".join(sorted(doc_hashes)))


@dataclass(frozen=True)
class Manifest:
    documents: tuple[DocumentRecord, ...] = ()

    @property
    def corpus_version(self) -> str:
        return corpus_version_of([d.sha256 for d in self.documents])

    @property
    def is_empty(self) -> bool:
        return not self.documents

    def get(self, sha256: str) -> DocumentRecord | None:
        return next((d for d in self.documents if d.sha256 == sha256), None)

    def add(self, record: DocumentRecord) -> Manifest:
        record.validate()
        if self.get(record.sha256) is not None:
            raise ManifestError(f"document already in corpus: {record.sha256[:12]} ({record.title})")
        return replace(self, documents=(*self.documents, record))

    def remove(self, sha256: str) -> Manifest:
        if self.get(sha256) is None:
            raise ManifestError(f"unknown document: {sha256[:12]}")
        return replace(self, documents=tuple(d for d in self.documents if d.sha256 != sha256))

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": SCHEMA_VERSION,
                "corpus_version": self.corpus_version,
                "documents": [asdict(d) for d in self.documents]}


def _record_from_dict(raw: dict[str, Any]) -> DocumentRecord:
    year = raw.get("year")
    rec = DocumentRecord(
        path=str(raw.get("path") or ""),
        sha256=str(raw.get("sha256") or ""),
        title=str(raw.get("title") or ""),
        year=int(year) if year not in (None, "") else None,
        source_type=str(raw.get("source_type") or ""),
        added_by=str(raw.get("added_by") or ""),
        added_at=str(raw.get("added_at") or ""),
    )
    rec.validate()
    return rec


def manifest_path(root: Path) -> Path:
    return Path(root) / MANIFEST_FILE


def text_path(root: Path, sha256: str) -> Path:
    return Path(root) / TEXT_DIR / f"{sha256}.txt"


def load_manifest(root: Path) -> Manifest:
    """The manifest under ``root`` (empty when none exists). A malformed file fails
    closed with ManifestError rather than being read as an empty corpus."""
    p = manifest_path(root)
    if not p.is_file():
        return Manifest()
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ManifestError(f"cannot read {p}: {exc}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("documents"), list):
        raise ManifestError(f"malformed manifest at {p}")
    docs = tuple(_record_from_dict(d) for d in raw["documents"] if isinstance(d, dict))
    seen: set[str] = set()
    for d in docs:
        if d.sha256 in seen:
            raise ManifestError(f"duplicate document in manifest: {d.sha256[:12]}")
        seen.add(d.sha256)
    return Manifest(docs)


def save_manifest(root: Path, manifest: Manifest) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    p = manifest_path(root)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest.to_dict(), indent=1, sort_keys=True), encoding="utf-8")
    tmp.replace(p)
    return p


def store_text(root: Path, sha256: str, text: str) -> Path:
    p = text_path(root, sha256)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def load_text(root: Path, sha256: str) -> str:
    p = text_path(root, sha256)
    if not p.is_file():
        raise ManifestError(f"extracted text missing for {sha256[:12]} (expected {p})")
    return p.read_text(encoding="utf-8")


def add_document(root: Path, record: DocumentRecord, text: str) -> Manifest:
    """Record ``record`` and keep its extracted ``text``; returns the new manifest."""
    manifest = load_manifest(root).add(record)
    store_text(root, record.sha256, text)
    save_manifest(root, manifest)
    return manifest


def remove_document(root: Path, sha256: str) -> Manifest:
    manifest = load_manifest(root).remove(sha256)
    save_manifest(root, manifest)
    p = text_path(root, sha256)
    if p.is_file():
        p.unlink()
    return manifest
