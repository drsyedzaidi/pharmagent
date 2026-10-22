"""Identity of the conventions catalog for run provenance: its version and a canonical-JSON digest.

``conventions_catalog_canonical_json_sha256`` is the SHA-256 of the catalog's parsed content serialised as canonical
JSON. It is not the SHA-256 of the file bytes, so reformatting the file does not change it but any content change does.
Duplicate keys are rejected when the file is read. To reproduce it:

    import hashlib, json
    raw = json.load(open("backend/app/conventions/catalog.json", encoding="utf-8"))
    text = json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    print(hashlib.sha256(text.encode("utf-8")).hexdigest())
"""
from __future__ import annotations

from pathlib import Path

from .catalog_io import cached_catalog, canonical_json_sha256, read_catalog_json
from .schema import Catalog

VERSION_KEY = "conventions_catalog_version"
DIGEST_KEY = "conventions_catalog_canonical_json_sha256"


def catalog_identity(path: Path | str | None = None) -> dict[str, str]:
    """Identity of the process-wide catalog (the one applied conventions use), or of the file at ``path``.
    A fresh dict each call."""
    if path is None:
        rec = cached_catalog()
        return {VERSION_KEY: rec.catalog.catalog_version, DIGEST_KEY: rec.canonical_json_sha256}
    raw = read_catalog_json(path)
    return {VERSION_KEY: Catalog.model_validate(raw).catalog_version, DIGEST_KEY: canonical_json_sha256(raw)}
