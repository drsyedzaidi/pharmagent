"""Deterministic catalog discovery, progressive-disclosure loader, resource reader, and provenance API."""
from __future__ import annotations

import re
from pathlib import Path
from types import MappingProxyType
from typing import Any

from app.tools.base import ToolRegistry
from app.tools.builtins import default_registry

from .digest import _canonical_metadata_dict, _normalize_newlines, _sha256_text, compute_canonical_digest
from .filesystem import (
    DEFAULT_SKILLS_ROOT,
    _inspect_playbook_resources,
    _read_utf8_bounded,
    _resolve_confined_resource_path,
    _validate_and_normalize_resource_arg,
    _validate_identifier_arg,
)
from .schema import (
    MAX_RESOURCE_BYTES,
    PlaybookDisabledError,
    PlaybookLookupError,
    PlaybookProvenance,
    PlaybookResource,
    PlaybookSecurityError,
    PlaybookSummary,
    PlaybookValidationError,
    PlaybookVersionError,
    ScientificPlaybook,
    freeze_value,
)
from .validation import (
    _classify_frontmatter_issues,
    _scan_markdown_links,
    validate_all_playbooks,
    validate_playbook,
)
from .yaml_loader import (
    _ZERO_WIDTH_RE,
    MAX_YAML_DEPTH,
    _DuplicateYamlKeyError,
    _parse_frontmatter,
    _StrictSafeLoader,
    _UnsafeYamlError,
)

__all__ = [
    "DEFAULT_SKILLS_ROOT",
    "MAX_YAML_DEPTH",
    "_DuplicateYamlKeyError",
    "_StrictSafeLoader",
    "_UnsafeYamlError",
    "_ZERO_WIDTH_RE",
    "_canonical_metadata_dict",
    "_classify_frontmatter_issues",
    "_inspect_playbook_resources",
    "_normalize_newlines",
    "_parse_frontmatter",
    "_read_utf8_bounded",
    "_scan_markdown_links",
    "catalog_provenance_snapshot",
    "check_tool_allowed",
    "compute_canonical_digest",
    "list_playbooks",
    "load_playbook",
    "playbook_provenance",
    "read_playbook_resource",
    "validate_all_playbooks",
    "validate_playbook",
]


def _require_valid_catalog(
    skills_root: Path | str | None = None,
    *,
    registry: ToolRegistry | None = None,
) -> tuple[Path, tuple[ScientificPlaybook, ...]]:
    root = Path(skills_root) if skills_root is not None else DEFAULT_SKILLS_ROOT
    report = validate_all_playbooks(skills_root=root, registry=registry)
    if not report.valid:
        sec = report.security_violations
        all_issues = [i for r in report.results for i in r.issues] + list(report.catalog_issues)
        summary = "; ".join(f"{i.code} ({i.field_or_path}): {i.message}" for i in all_issues[:5])
        if sec:
            raise PlaybookSecurityError(f"Playbook catalog failed security validation: {summary}")
        raise PlaybookValidationError(f"Playbook catalog failed validation: {summary}")
    playbooks = tuple(
        sorted(
            (r.playbook for r in report.results if r.playbook is not None),
            key=lambda p: (p.playbook_id, p.version),
        )
    )
    return root.resolve(strict=True), playbooks


def list_playbooks(
    skills_root: Path | str | None = None,
    *,
    registry: ToolRegistry | None = None,
    include_disabled: bool = True,
) -> tuple[PlaybookSummary, ...]:
    """Return deterministic discovery summaries for all playbooks (progressive disclosure tier 1)."""
    _root, playbooks = _require_valid_catalog(skills_root=skills_root, registry=registry)
    return tuple(
        pb.to_summary()
        for pb in playbooks
        if include_disabled or pb.runtime_available
    )


def load_playbook(
    playbook_id: str,
    version: str | None = None,
    *,
    skills_root: Path | str | None = None,
    registry: ToolRegistry | None = None,
    allow_disabled: bool = False,
) -> ScientificPlaybook:
    """Load and validate full instructions for a single playbook (progressive disclosure tier 2)."""
    clean_id = _validate_identifier_arg(playbook_id, "playbook_id")
    if version is not None:
        if not isinstance(version, str) or not re.fullmatch(
            r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$", version.strip()
        ):
            raise PlaybookVersionError(f"Invalid semantic version requested: {version!r}")
        version = version.strip()

    _root, playbooks = _require_valid_catalog(skills_root=skills_root, registry=registry)
    matched = next((pb for pb in playbooks if pb.playbook_id == clean_id or pb.name == clean_id), None)
    if matched is None:
        raise PlaybookLookupError(f"Unknown Scientific Playbook: {clean_id!r}")
    if version is not None and matched.version != version:
        raise PlaybookVersionError(
            f"Playbook {matched.playbook_id!r} has version {matched.version!r}, not requested {version!r}"
        )
    if (
        not matched.runtime_available
        or matched.approval_status == "rejected"
        or matched.status == "deprecated"
    ) and not allow_disabled:
        raise PlaybookDisabledError(
            f"Playbook {matched.playbook_id!r} is disabled for runtime execution "
            f"(status={matched.status!r}, approval_status={matched.approval_status!r}, "
            f"runtime_available={matched.runtime_available})"
        )
    return matched


def read_playbook_resource(
    playbook_id: str,
    resource: str,
    *,
    version: str | None = None,
    skills_root: Path | str | None = None,
    registry: ToolRegistry | None = None,
    allow_disabled: bool = False,
) -> PlaybookResource:
    """Read a single validated, confined resource file from a playbook (progressive disclosure tier 3)."""
    posix, norm_rel = _validate_and_normalize_resource_arg(resource)

    raw_root = Path(skills_root) if skills_root is not None else DEFAULT_SKILLS_ROOT
    pb = load_playbook(
        playbook_id,
        version=version,
        skills_root=raw_root,
        registry=registry,
        allow_disabled=allow_disabled,
    )
    real_root = raw_root.resolve(strict=True)
    pb_dir = (real_root / pb.name).resolve(strict=True)
    if not pb_dir.is_relative_to(real_root):
        raise PlaybookSecurityError("Playbook directory escapes skills_root")

    real_candidate = _resolve_confined_resource_path(pb_dir, posix, norm_rel, pb.playbook_id)

    if norm_rel not in pb.resource_paths:
        raise PlaybookLookupError(f"Resource {norm_rel!r} is not registered in playbook {pb.playbook_id!r}")

    text, read_issues = _read_utf8_bounded(real_candidate, MAX_RESOURCE_BYTES, norm_rel)
    if read_issues:
        raise PlaybookSecurityError("; ".join(i.message for i in read_issues))
    assert text is not None
    sha = _sha256_text(text)
    expected_sha = dict(pb.resource_digests).get(norm_rel)
    if not expected_sha or sha != expected_sha:
        raise PlaybookSecurityError(f"Resource {norm_rel!r} digest mismatch against playbook canonical digest")

    return PlaybookResource(
        playbook_id=pb.playbook_id,
        name=pb.name,
        version=pb.version,
        canonical_digest=pb.canonical_digest,
        approval_status=pb.approval_status,
        source=pb.source,
        license=pb.license,
        allowed_tools=pb.allowed_tools,
        required_inputs=pb.required_inputs,
        limitations=pb.limitations,
        resource_path=norm_rel,
        content=text,
        sha256=sha,
        size_bytes=len(text.encode("utf-8")),
    )


def check_tool_allowed(
    playbook: ScientificPlaybook | PlaybookSummary,
    tool_name: str,
    *,
    registry: ToolRegistry | None = None,
) -> bool:
    """Enforce that `tool_name` exists in `ToolRegistry` and is allowlisted by `playbook`."""
    if not isinstance(tool_name, str) or not tool_name.strip():
        raise PlaybookSecurityError("non_existent_tool: tool_name must be a non-empty string")
    clean_tool = tool_name.strip()
    reg = registry if registry is not None else default_registry()
    if clean_tool not in set(reg.names()):
        raise PlaybookSecurityError(
            f"non_existent_tool: Tool {clean_tool!r} does not exist in the current ToolRegistry"
        )
    if (
        not playbook.runtime_available
        or playbook.approval_status == "rejected"
        or playbook.status == "deprecated"
    ):
        raise PlaybookDisabledError(
            f"Playbook {playbook.playbook_id!r} is not available for runtime execution and cannot authorize tools"
        )
    if clean_tool not in playbook.allowed_tools:
        raise PlaybookSecurityError(
            f"undeclared_tool: Tool {clean_tool!r} is not in allowed_tools for playbook {playbook.playbook_id!r}"
        )
    return True


def playbook_provenance(
    playbook_id: str,
    *,
    version: str | None = None,
    skills_root: Path | str | None = None,
    registry: ToolRegistry | None = None,
    allow_disabled: bool = False,
) -> PlaybookProvenance:
    """Return an immutable provenance identity record for a loaded Scientific Playbook."""
    pb = load_playbook(
        playbook_id,
        version=version,
        skills_root=skills_root,
        registry=registry,
        allow_disabled=allow_disabled,
    )
    return PlaybookProvenance(
        playbook_id=pb.playbook_id,
        name=pb.name,
        version=pb.version,
        canonical_digest=pb.canonical_digest,
        status=pb.status,
        approval_status=pb.approval_status,
        approved_by=pb.approved_by,
        approved_date=pb.approved_date,
        source=pb.source,
        license=pb.license,
        allowed_tools=pb.allowed_tools,
        required_inputs=pb.required_inputs,
        limitations=pb.limitations,
        requires_human_review=pb.requires_human_review,
    )


def catalog_provenance_snapshot(
    skills_root: Path | str | None = None,
    *,
    registry: ToolRegistry | None = None,
) -> MappingProxyType[str, Any]:
    """Return a deeply immutable snapshot of all catalog playbooks for future provenance hooks."""
    summaries = list_playbooks(skills_root=skills_root, registry=registry, include_disabled=True)
    return freeze_value({
        "playbooks": [
            {
                "playbook_id": s.playbook_id,
                "name": s.name,
                "version": s.version,
                "canonical_digest": s.canonical_digest,
                "status": s.status,
                "approval_status": s.approval_status,
                "runtime_available": s.runtime_available,
            }
            for s in summaries
        ]
    })
