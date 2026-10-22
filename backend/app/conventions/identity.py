"""Immutable identity of the conventions catalog, for run provenance: its version and a canonical SHA-256.

The digest is over the file's parsed content serialised as sorted-key, compact JSON, so reformatting the file does not
change it but any content change does. The file is validated against the schema before it is fingerprinted."""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

from .catalog_io import DEFAULT_CATALOG
from .schema import Catalog


@lru_cache(maxsize=8)
def _identity(path: str) -> tuple[tuple[str, str], ...]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    Catalog.model_validate(raw)
    canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return (("conventions_catalog_version", str(raw["catalog_version"])),
            ("conventions_catalog_sha256", hashlib.sha256(canonical.encode("utf-8")).hexdigest()))


def catalog_identity(path: Path | str | None = None) -> dict[str, str]:
    """A fresh dict each call, so a caller cannot alter the cached identity."""
    return dict(_identity(str(Path(path or DEFAULT_CATALOG).resolve())))
