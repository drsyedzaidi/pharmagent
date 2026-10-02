"""On-disk cache of rendered page images: ``<data_dir>/evidence/render/<key>.png`` + ``<key>.json``.

The key is computed from INPUTS (PDF sha256, page, the validated render arguments with boxes rounded
to 0.1 pt, RENDER_VERSION), so a hit never starts a child. Entries are content-addressed: sessions
sharing a PDF share pixels, which is safe because the route authorises per record, not per file (and
every request re-verifies the PDF hash before it looks here). The highlight is not part of the pixels.

Writes are atomic (PNG first, then JSON: an entry without both files, or whose PNG disagrees with its
JSON in size or sha256, is a miss and is deleted). The directory is capped at MAX_BYTES / MAX_ENTRIES;
past either, least-recently-used entries (mtime, touched on every hit) go until 80 % of both caps
remain. Single flight: two requests for one key run one render, the second finds the entry.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from app.core.store import _json as canonical_json
from app.evidence.ingest import evidence_dir

MAX_BYTES = 256 * 2**20
MAX_ENTRIES = 4000
EVICT_TO = 0.8
META_KEYS = ("width", "height", "scale", "dpi", "crop_px", "crop_pt", "rects_px", "snippet_px", "page_px",
             "png_sha256", "png_bytes")

_lock = threading.Lock()
_flights: dict[str, list[Any]] = {}


def render_dir(data_dir: str | Path | None = None) -> Path:
    d = evidence_dir(data_dir) / "render"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _round(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 1)
    if isinstance(value, list):
        return [_round(v) for v in value]
    if isinstance(value, dict):
        return {k: _round(v) for k, v in value.items()}
    return value


def cache_key(pdf_sha256: str, args: dict[str, Any], version: str) -> str:
    """sha256 of the canonical JSON of everything that decides the pixels."""
    return hashlib.sha256(canonical_json({"pdf": pdf_sha256, "args": _round(args), "v": version}).encode()).hexdigest()


def _atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}.{threading.get_ident()}")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _drop(directory: Path, key: str) -> None:
    for ext in ("png", "json"):
        (directory / f"{key}.{ext}").unlink(missing_ok=True)


def get(key: str, data_dir: str | Path | None = None) -> tuple[bytes, dict[str, Any]] | None:
    """(png, meta) of a whole, intact entry (touched as recently used), else None."""
    directory = render_dir(data_dir)
    png_path, meta_path = directory / f"{key}.png", directory / f"{key}.json"
    try:
        meta = json.loads(meta_path.read_text("utf-8"))
        png = png_path.read_bytes()
        ok = (isinstance(meta, dict) and meta.get("png_bytes") == len(png)
              and meta.get("png_sha256") == hashlib.sha256(png).hexdigest())
    except (OSError, ValueError):
        ok = False
    if not ok:
        if png_path.exists() or meta_path.exists():
            _drop(directory, key)
        return None
    now = time.time()
    for p in (png_path, meta_path):
        try:
            os.utime(p, (now, now))
        except OSError:
            pass
    return png, {k: meta[k] for k in META_KEYS if k in meta}


def put(key: str, png: bytes, meta: dict[str, Any], data_dir: str | Path | None = None) -> None:
    directory = render_dir(data_dir)
    _atomic(directory / f"{key}.png", png)
    _atomic(directory / f"{key}.json", json.dumps({k: meta[k] for k in META_KEYS if k in meta}).encode("utf-8"))
    evict(data_dir)


def evict(data_dir: str | Path | None = None, *, max_bytes: int | None = None,
          max_entries: int | None = None) -> int:
    """Delete least-recently-used entries while over a cap, down to EVICT_TO of both; returns the number
    deleted. Orphans (a half-written entry) count as entries and go first when old."""
    max_bytes, max_entries = max_bytes or MAX_BYTES, max_entries or MAX_ENTRIES
    directory = render_dir(data_dir)
    with _lock:
        entries: dict[str, list[float]] = {}
        for p in directory.iterdir():
            if p.suffix not in (".png", ".json"):
                continue
            try:
                st = p.stat()
            except OSError:
                continue
            e = entries.setdefault(p.stem, [0.0, 0.0])
            e[0] += st.st_size
            e[1] = max(e[1], st.st_mtime)
        total, count = sum(e[0] for e in entries.values()), len(entries)
        if total <= max_bytes and count <= max_entries:
            return 0
        deleted = 0
        for key, (size, _) in sorted(entries.items(), key=lambda kv: kv[1][1]):
            if total <= EVICT_TO * max_bytes and count <= EVICT_TO * max_entries:
                break
            _drop(directory, key)
            total, count, deleted = total - size, count - 1, deleted + 1
        return deleted


@contextmanager
def single_flight(key: str) -> Iterator[None]:
    """Serialise work on one key: the second caller waits for the first, then finds its cache entry."""
    with _lock:
        slot = _flights.setdefault(key, [threading.Lock(), 0])
        slot[1] += 1
    try:
        with slot[0]:
            yield
    finally:
        with _lock:
            slot[1] -= 1
            if slot[1] == 0:
                _flights.pop(key, None)
