"""Deterministic canonical metadata normalization, newline normalization, and SHA-256 digest generation."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from .schema import (
    PlaybookCompatibility,
    PlaybookSummary,
    ScientificPlaybook,
)


def _normalize_newlines(text: str) -> str:
    """Normalize CRLF and CR line endings to LF (`\\n`)."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _sha256_text(text: str) -> str:
    """Compute hex SHA-256 of newline-normalized UTF-8 text."""
    return hashlib.sha256(_normalize_newlines(text).encode("utf-8")).hexdigest()


def _canonical_metadata_dict(meta: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize frontmatter or model metadata into a canonical JSON-serializable dict."""
    compat = meta.get("compatibility")
    if isinstance(compat, PlaybookCompatibility):
        compat_dump = compat.model_dump()
        compat_val: Any = {**compat_dump, "engines": list(compat_dump["engines"])}
    elif isinstance(compat, Mapping):
        coerced_compat = PlaybookCompatibility.model_validate(dict(compat)).model_dump()
        compat_val = {**coerced_compat, "engines": list(coerced_compat["engines"])}
    else:
        compat_val = str(compat).strip()

    raw_source = meta.get("source")
    if isinstance(raw_source, Mapping):
        kind = str(raw_source.get("kind", "")).strip()
        ref = str(raw_source.get("reference", "")).strip()
        source_val = f"{kind}: {ref}" if (kind and ref) else str(raw_source)
    else:
        source_val = str(raw_source).strip()

    raw_by = meta.get("approved_by")
    approved_by = str(raw_by).strip() if isinstance(raw_by, str) and raw_by.strip() else raw_by
    raw_dt = meta.get("approved_date")
    approved_date = str(raw_dt).strip() if isinstance(raw_dt, str) and raw_dt.strip() else raw_dt

    return {
        "playbook_id": str(meta["playbook_id"]).strip(),
        "name": str(meta["name"]).strip(),
        "description": str(meta["description"]).strip(),
        "version": str(meta["version"]).strip(),
        "domain": str(meta["domain"]).strip(),
        "status": str(meta["status"]).strip(),
        "approval_status": str(meta["approval_status"]).strip(),
        "approved_by": approved_by,
        "approved_date": approved_date,
        "license": str(meta["license"]).strip(),
        "source": source_val,
        "allowed_tools": [str(x).strip() for x in (meta.get("allowed_tools") or ())],
        "required_inputs": [str(x).strip() for x in (meta.get("required_inputs") or ())],
        "produced_artifacts": [str(x).strip() for x in (meta.get("produced_artifacts") or ())],
        "limitations": [str(x).strip() for x in (meta.get("limitations") or ())],
        "requires_human_review": bool(meta["requires_human_review"]),
        "compatibility": compat_val,
        "runtime_available": bool(meta.get("runtime_available", True)),
    }


def compute_canonical_digest(
    playbook_or_metadata: ScientificPlaybook | PlaybookSummary | Mapping[str, Any],
    *,
    instructions: str | None = None,
    resource_digests: tuple[tuple[str, str], ...] | Mapping[str, str] | None = None,
) -> str:
    """Compute the deterministic canonical SHA-256 digest of a Scientific Playbook.

    Reproduction specification:
    1. Normalize all text newlines (`\\r\\n` and `\\r` -> `\\n`).
    2. Construct the canonical payload dictionary:
       `{"digest_version": 1, "instructions": <normalized_instructions>, "metadata": <normalized_metadata>,
         "resources": [{"path": <posix_rel_path>, "sha256": <hex_digest>}, ...]}`
       with `resources` sorted ascending by `path`.
    3. Serialize with `json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)`.
    4. Compute `hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()`.
    """
    if isinstance(playbook_or_metadata, ScientificPlaybook):
        meta_map: Mapping[str, Any] = playbook_or_metadata.model_dump()
        body_text = instructions if instructions is not None else playbook_or_metadata.instructions
        res_pairs = (
            playbook_or_metadata.resource_digests
            if resource_digests is None
            else (
                tuple(sorted(resource_digests.items()))
                if isinstance(resource_digests, Mapping)
                else tuple(sorted(resource_digests))
            )
        )
    elif isinstance(playbook_or_metadata, PlaybookSummary):
        if instructions is None:
            raise ValueError("`instructions` is required when computing digest from a PlaybookSummary")
        meta_map = playbook_or_metadata.model_dump()
        body_text = instructions
        res_pairs = (
            playbook_or_metadata.resource_digests
            if resource_digests is None
            else (
                tuple(sorted(resource_digests.items()))
                if isinstance(resource_digests, Mapping)
                else tuple(sorted(resource_digests))
            )
        )
    elif isinstance(playbook_or_metadata, Mapping):
        if instructions is None:
            raise ValueError("`instructions` is required when computing digest from a metadata mapping")
        meta_map = playbook_or_metadata
        body_text = instructions
        if resource_digests is None:
            res_pairs = ()
        elif isinstance(resource_digests, Mapping):
            res_pairs = tuple(sorted((str(k), str(v)) for k, v in resource_digests.items()))
        else:
            res_pairs = tuple(sorted((str(k), str(v)) for k, v in resource_digests))
    else:
        raise TypeError("Expected ScientificPlaybook, PlaybookSummary, or metadata Mapping")

    payload = {
        "digest_version": 1,
        "metadata": _canonical_metadata_dict(meta_map),
        "instructions": _normalize_newlines(body_text).strip(),
        "resources": [
            {"path": str(rel_path), "sha256": str(digest)}
            for rel_path, digest in sorted(res_pairs, key=lambda item: item[0])
        ],
    }
    canonical_json = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
