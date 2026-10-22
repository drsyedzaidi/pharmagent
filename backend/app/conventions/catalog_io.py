"""Load and validate the catalog file. One cached load per process serves both applied conventions and run provenance,
so the recorded catalog identity always describes the catalog actually used."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .schema import Catalog

DEFAULT_CATALOG = Path(__file__).with_name("catalog.json")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key in conventions catalog: {key!r}")
        out[key] = value
    return out


def read_catalog_json(path: Path | str) -> dict[str, Any]:
    """Parse the catalog file, rejecting duplicate keys at any depth (json would silently keep the last one)."""
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys)


def canonical_json_sha256(raw: dict[str, Any]) -> str:
    """SHA-256 of the parsed catalog as canonical JSON: sorted keys, no whitespace, UTF-8, non-ASCII kept as is."""
    canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_catalog(path: Path | str | None = None) -> Catalog:
    return Catalog.model_validate(read_catalog_json(path if path is not None else DEFAULT_CATALOG))


@dataclass(frozen=True)
class CatalogRecord:
    catalog: Catalog
    canonical_json_sha256: str


@lru_cache(maxsize=1)
def cached_catalog() -> CatalogRecord:
    """The process-wide catalog and its digest, from a single read of DEFAULT_CATALOG (constant until restart)."""
    raw = read_catalog_json(DEFAULT_CATALOG)
    return CatalogRecord(catalog=Catalog.model_validate(raw), canonical_json_sha256=canonical_json_sha256(raw))
