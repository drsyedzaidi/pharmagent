"""Load, fingerprint and append to the rule catalog.

The catalog file is committed to the repository (it holds only rule logic and hashes, no
passage text), so owner approval is a reviewed pull request, exactly as for the conventions
catalog. ``catalog_sha256`` is the SHA-256 of the parsed content as canonical JSON (sorted keys,
no whitespace, UTF-8, non-ASCII kept: the same recipe the conventions catalog documents, defined
here rather than imported), so reformatting the file does not change it and any content change
does. Every rule-derived finding carries it.

One cached load per process: restart the server after changing the catalog.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.knowledge.rules.schema import Rule, RuleCatalog

SHIPPED_CATALOG = Path(__file__).with_name("catalog.json")
DEFAULT_CATALOG = SHIPPED_CATALOG       # what the app reads; a name tests may repoint, SHIPPED_CATALOG never moves


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key in rule catalog: {key!r}")
        out[key] = value
    return out


def canonical_json_sha256(raw: dict[str, Any]) -> str:
    canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


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
    write is atomic: a private temporary file created exclusively in the same directory, synced,
    then renamed over the catalog. A symlinked catalog path is refused, never followed.
    """
    if rule.approval.status != "pending_owner_review" or rule.approval.by or rule.approval.date:
        raise ValueError("rules enter the catalog as pending_owner_review with no approver; "
                         "the owner approves in review")
    target = Path(path if path is not None else DEFAULT_CATALOG)
    if target.is_symlink():
        raise ValueError("the rule catalog path is a symlink; refusing to write through it or replace it")
    before = target.read_bytes()
    raw = json.loads(before.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
    current = RuleCatalog.model_validate(raw)
    if any(r.id == rule.id for r in current.rules):
        raise ValueError(f"rule id already in the catalog: {rule.id}")
    new_raw = {**raw, "catalog_version": _bump_patch(current.catalog_version),
               "rules": [*raw.get("rules", []), rule.model_dump(mode="json")]}
    validated = RuleCatalog.model_validate(new_raw)          # whole file re-validated before it is written
    payload = (json.dumps(new_raw, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        if hashlib.sha256(target.read_bytes()).digest() != hashlib.sha256(before).digest():
            raise ValueError("the catalog changed while the rule was being added; nothing was written, retry")
        os.replace(tmp_name, target)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_name)
    return RuleCatalogRecord(catalog=validated, catalog_sha256=canonical_json_sha256(new_raw))
