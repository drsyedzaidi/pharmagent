"""Runtime use of the conventions catalog (slice 2). Read-only: results name the conventions they applied; nothing here
changes a computed number. Run provenance is guarded code and is deliberately not touched."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .catalog_io import cached_catalog, load_catalog
from .render import APPROVAL_MEANING, rule_text
from .schema import Catalog


def _catalog() -> Catalog:
    """The same cached catalog that run provenance fingerprints (one load per process)."""
    return cached_catalog().catalog


def validate_catalog_at_startup(path: Path | str | None = None) -> str:
    """Load and validate the catalog; raise ValueError so the application fails fast on a broken file."""
    try:
        return load_catalog(path).catalog_version
    except (ValueError, OSError) as exc:           # pydantic ValidationError and JSON errors are ValueErrors
        raise ValueError(f"conventions catalog is invalid: {exc}") from exc


def applied_conventions(ids: list[str]) -> dict[str, Any]:
    """JSON-safe block naming the catalog version and each applied convention. An unknown id is a programming error."""
    catalog = _catalog()
    by_id = {c.id: c for c in catalog.conventions}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise KeyError(f"unknown convention ids: {missing}")
    return {
        "catalog_version": catalog.catalog_version,
        "approval_meaning": APPROVAL_MEANING,
        "conventions": [{
            "id": c.id, "version": c.version, "approval": c.approval.status,
            "approved_by": c.approval.by, "approved_on": c.approval.date, "rule_text": rule_text(c),
        } for c in (by_id[i] for i in ids)],
    }
