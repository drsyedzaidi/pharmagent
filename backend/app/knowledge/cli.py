"""Owner CLI for the mikebrain corpus (run from ``backend/``)::

    python -m app.knowledge add  <file.pdf|.txt|.md> --title "..." [--year 2011] [--type book|paper|guideline]
    python -m app.knowledge list
    python -m app.knowledge search "eta shrinkage EBE diagnostics" [-k 6] [--no-embed]
    python -m app.knowledge remove <sha256 or unique prefix>

The corpus root defaults to ``<PHARMAGENT_DATA_DIR>/knowledge`` (git-ignored).
Documents are the owner's own copies; only extracted text and hashes are stored.
"""
from __future__ import annotations

import argparse
import getpass
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from app.knowledge import manifest as mf
from app.knowledge.extract import ExtractionError, extract_text
from app.knowledge.index import DEFAULT_K, CorpusIndex, NoEmbedder, default_embedder

KNOWLEDGE_SUBDIR = "knowledge"


def default_root() -> Path:
    from app.config import settings
    return Path(settings.data_dir) / KNOWLEDGE_SUBDIR


def cmd_add(root: Path, path: Path, title: str | None, year: int | None, source_type: str, added_by: str) -> int:
    if not path.is_file():
        print(f"not a file: {path}", file=sys.stderr)
        return 2
    try:
        text = extract_text(path)
    except ExtractionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    rec = mf.DocumentRecord(
        path=str(path.resolve()), sha256=mf.file_sha256(path), title=(title or path.stem).strip(),
        year=year, source_type=source_type, added_by=added_by,
        added_at=datetime.now(UTC).isoformat(timespec="seconds"))
    try:
        manifest = mf.add_document(root, rec, text)
    except mf.ManifestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    n_chunks = len(CorpusIndex.load(root).chunks)
    print(json.dumps({"added": rec.sha256, "title": rec.title, "chars": len(text),
                      "documents": len(manifest.documents), "corpus_chunks": n_chunks,
                      "corpus_version": manifest.corpus_version}, indent=1))
    return 0


def cmd_list(root: Path) -> int:
    manifest = mf.load_manifest(root)
    print(json.dumps({"root": str(root), "corpus_version": manifest.corpus_version,
                      "documents": [{"sha256": d.sha256, "title": d.title, "year": d.year,
                                     "source_type": d.source_type, "added_by": d.added_by,
                                     "added_at": d.added_at, "path": d.path,
                                     "text_present": mf.text_path(root, d.sha256).is_file()}
                                    for d in manifest.documents]}, indent=1))
    return 0


def cmd_search(root: Path, query: str, k: int, no_embed: bool) -> int:
    index = CorpusIndex.load(root)
    res = index.search(query, k=k, embedder=NoEmbedder() if no_embed else default_embedder())
    out = res.to_dict()
    for h in out["hits"]:
        h["text"] = h["text"][:300] + ("…" if len(h["text"]) > 300 else "")
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return 0


def cmd_remove(root: Path, sha_prefix: str) -> int:
    manifest = mf.load_manifest(root)
    matches = [d for d in manifest.documents if d.sha256.startswith(sha_prefix)]
    if len(matches) != 1:
        print(f"error: {len(matches)} documents match {sha_prefix!r}; give a unique prefix", file=sys.stderr)
        return 2
    new = mf.remove_document(root, matches[0].sha256)
    print(json.dumps({"removed": matches[0].sha256, "documents": len(new.documents),
                      "corpus_version": new.corpus_version}, indent=1))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m app.knowledge", description="mikebrain corpus management")
    p.add_argument("--root", type=Path, default=None, help="corpus root (default <data_dir>/knowledge)")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add", help="extract, hash and record a document")
    a.add_argument("path", type=Path)
    a.add_argument("--title")
    a.add_argument("--year", type=int)
    a.add_argument("--type", dest="source_type", choices=mf.SOURCE_TYPES, default="paper")
    a.add_argument("--by", dest="added_by", default=getpass.getuser())
    sub.add_parser("list", help="list documents and the corpus version")
    s = sub.add_parser("search", help="hybrid search (BM25 + Ollama embeddings when reachable)")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=DEFAULT_K)
    s.add_argument("--no-embed", action="store_true", help="BM25 only")
    r = sub.add_parser("remove", help="remove a document by sha256 (prefix)")
    r.add_argument("sha")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root or default_root()
    if args.cmd == "add":
        return cmd_add(root, args.path, args.title, args.year, args.source_type, args.added_by)
    if args.cmd == "list":
        return cmd_list(root)
    if args.cmd == "search":
        return cmd_search(root, args.query, args.k, args.no_embed)
    if args.cmd == "remove":
        return cmd_remove(root, args.sha)
    return 2   # pragma: no cover - argparse rejects unknown commands first
