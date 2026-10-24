"""Load, fingerprint and append to the rule catalog.

The catalog file is committed to the repository (it holds only rule logic and hashes, no
passage text), so owner approval is a reviewed pull request, exactly as for the conventions
catalog. ``catalog_sha256`` is the SHA-256 of the parsed content as canonical JSON (the same
recipe as ``app.conventions.catalog_io.canonical_json_sha256``), so reformatting the file does
not change it and any content change does. Every rule-derived finding carries it.

One cached load per process: restart the server after changing the catalog.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.conventions.catalog_io import canonical_json_sha256
from app.knowledge.rules.schema import Rule, RuleCatalog

DEFAULT_CATALOG = Path(__file__).with_name("catalog.json")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key in rule catalog: {key!r}")
        out[key] = value
    return out


def read_rule_json(path: Path | str) -> dict[str, Any]:
    """Parse the catalog file, rejecting duplicate keys at any depth (json would keep the last one silently)."""
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys)


@dataclass(frozen=True)
class RuleCatalogRecord:
    catalog: RuleCatalog
    catalog_sha256: str


def load_rule_catalog(path: Path | str | None = None) -> RuleCatalogRecord:
    """Parse and validate ``path`` (default: the shipped catalog). Raises ValueError/OSError."""
    raw = read_rule_json(path if path is not None else DEFAULT_CATALOG)
    return RuleCatalogRecord(catalog=RuleCatalog.model_validate(raw), catalog_sha256=canonical_json_sha256(raw))


@lru_cache(maxsize=1)
def cached_rule_catalog() -> RuleCatalogRecord:
    """The process-wide catalog read once from ``DEFAULT_CATALOG`` (constant until restart)."""
    return load_rule_catalog(DEFAULT_CATALOG)


def _bump_patch(version: str) -> str:
    major, minor, patch = (int(p) for p in version.split("."))
    return f"{major}.{minor}.{patch + 1}"


def append_pending_rule(rule: Rule, path: Path | str | None = None) -> RuleCatalogRecord:
    """Append ``rule`` to the catalog file and return the new record.

    Refuses anything but a ``pending_owner_review`` rule with no approver: the only writer of
    the catalog cannot approve. Approval is the owner editing the entry in a reviewed change.
    The patch version is bumped, the whole file is re-validated before it is written, and the
    write is atomic.
    """
    if rule.approval.status != "pending_owner_review" or rule.approval.by or rule.approval.date:
        raise ValueError("rules enter the catalog as pending_owner_review with no approver; "
                         "the owner approves in review")
    target = Path(path if path is not None else DEFAULT_CATALOG)
    raw = read_rule_json(target)
    current = RuleCatalog.model_validate(raw)
    if any(r.id == rule.id for r in current.rules):
        raise ValueError(f"rule id already in the catalog: {rule.id}")
    new_raw = {**raw, "catalog_version": _bump_patch(current.catalog_version),
               "rules": [*raw.get("rules", []), rule.model_dump(mode="json")]}
    validated = RuleCatalog.model_validate(new_raw)          # whole file re-validated before it is written
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(new_raw, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, target)
    return RuleCatalogRecord(catalog=validated, catalog_sha256=canonical_json_sha256(new_raw))
