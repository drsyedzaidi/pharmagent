"""Load and validate the catalog file. Not wired into any tool or into app startup in slice 1."""
from __future__ import annotations

import json
from pathlib import Path

from .schema import Catalog

DEFAULT_CATALOG = Path(__file__).with_name("catalog.json")


def load_catalog(path: Path | str | None = None) -> Catalog:
    p = Path(path) if path is not None else DEFAULT_CATALOG
    return Catalog.model_validate(json.loads(p.read_text(encoding="utf-8")))
