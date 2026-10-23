"""Deterministic catalog discovery, review vs runtime loaders, tool authorization, and provenance API."""

from __future__ import annotations

import re
from datetime import date as _date
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
    PlaybookApprovalError,
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
    "authorize_playbook_tool",
    "catalog_provenance_snapshot",
    "check_tool_allowed",
    "compute_canonical_digest",
    "list_playbooks",
    "load_approved_playbook",
    "load_playbook",
    "load_playbook_for_review",
    "playbook_provenance",
    "read_playbook_resource",
    "read_playbook_resource_for_review",
    "validate_all_playbooks",
    "validate_playbook",
]

_ISO_DATE_FULL_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_SEMVER_FULL_RE = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")


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


def _verify_runtime_approved(
    pb: ScientificPlaybook | PlaybookSummary,
    *,
    expected_digest: str | None = None,
) -> None:
    """Enforce that a playbook is enabled, owner-approved, version-stable, and digest-valid for runtime use."""
    if not pb.runtime_available or pb.approval_status == "rejected" or pb.status == "deprecated":
        raise PlaybookDisabledError(
            f"Playbook {pb.playbook_id!r} is disabled for runtime execution "
            f"(status={pb.status!r}, approval_status={pb.approval_status!r}, "
            f"runtime_available={pb.runtime_available})"
        )
    if (
        pb.approval_status != "approved"
        or pb.status != "approved"
        or not (isinstance(pb.approved_by, str) and pb.approved_by.strip())
        or not (isinstance(pb.approved_date, str) and _ISO_DATE_FULL_RE.fullmatch(pb.approved_date))
    ):
        raise PlaybookApprovalError(
            f"Playbook {pb.playbook_id!r} requires owner approval before runtime execution or tool authorization "
            f"(status={pb.status!r}, approval_status={pb.approval_status!r})"
        )
    try:
        _date.fromisoformat(pb.approved_date)
    except ValueError as exc:
        raise PlaybookApprovalError(
            f"Playbook {pb.playbook_id!r} has invalid approved_date {pb.approved_date!r}"
        ) from exc
    if pb.version.startswith("0."):
        raise PlaybookApprovalError(
            f"Playbook {pb.playbook_id!r} version {pb.version!r} is a 0.x.y pre-release and cannot execute at runtime"
        )
    if isinstance(pb, ScientificPlaybook):
        actual_digest = compute_canonical_digest(pb)
        if actual_digest != pb.canonical_digest:
            raise PlaybookSecurityError(
                f"digest_mismatch: Playbook {pb.playbook_id!r} canonical digest does not match content"
            )
    if expected_digest is not None and pb.canonical_digest != expected_digest:
        raise PlaybookSecurityError(
            f"digest_mismatch: Playbook {pb.playbook_id!r} digest {pb.canonical_digest!r} "
            f"!= expected {expected_digest!r}"
        )


def list_playbooks(
    skills_root: Path | str | None = None,
    *,
    registry: ToolRegistry | None = None,
    include_disabled: bool = True,
) -> tuple[PlaybookSummary, ...]:
    """Return deterministic discovery summaries (progressive disclosure tier 1).

    With `include_disabled=False` (runtime discovery), only playbooks that pass the same
    approval gate as `load_playbook` are returned, so every discovered playbook is loadable.
    """
    _root, playbooks = _require_valid_catalog(skills_root=skills_root, registry=registry)
    return tuple(pb.to_summary() for pb in playbooks if include_disabled or _is_runtime_loadable(pb))


def _is_runtime_loadable(pb: ScientificPlaybook) -> bool:
    try:
        _verify_runtime_approved(pb)
    except PlaybookDisabledError:
        return False
    return True


def load_playbook_for_review(
    playbook_id: str,
    version: str | None = None,
    *,
    skills_root: Path | str | None = None,
    registry: ToolRegistry | None = None,
    allow_disabled: bool = False,
    expected_digest: str | None = None,
) -> ScientificPlaybook:
    """Explicit human-review API: load a validated playbook's full text (including `pending_owner_review`)."""
    clean_id = _validate_identifier_arg(playbook_id, "playbook_id")
    if version is not None:
        if not isinstance(version, str) or not _SEMVER_FULL_RE.fullmatch(version.strip()):
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
        not matched.runtime_available or matched.approval_status == "rejected" or matched.status == "deprecated"
    ) and not allow_disabled:
        raise PlaybookDisabledError(
            f"Playbook {matched.playbook_id!r} is disabled for runtime execution "
            f"(status={matched.status!r}, approval_status={matched.approval_status!r}, "
            f"runtime_available={matched.runtime_available})"
        )
    if compute_canonical_digest(matched) != matched.canonical_digest:
        raise PlaybookSecurityError(f"Digest mismatch when loading {matched.playbook_id!r}")
    if expected_digest is not None and matched.canonical_digest != expected_digest:
        raise PlaybookSecurityError(
            f"digest_mismatch: Playbook {matched.playbook_id!r} digest {matched.canonical_digest!r} "
            f"!= expected {expected_digest!r}"
        )
    return matched


def load_approved_playbook(
    playbook_id: str,
    version: str | None = None,
    *,
    skills_root: Path | str | None = None,
    registry: ToolRegistry | None = None,
    expected_digest: str | None = None,
) -> ScientificPlaybook:
    """Runtime loader: load a playbook ONLY if it is owner-approved, runtime-enabled, and digest-valid."""
    pb = load_playbook_for_review(
        playbook_id,
        version=version,
        skills_root=skills_root,
        registry=registry,
        allow_disabled=False,
    )
    _verify_runtime_approved(pb, expected_digest=expected_digest)
    return pb


def load_playbook(
    playbook_id: str,
    version: str | None = None,
    *,
    skills_root: Path | str | None = None,
    registry: ToolRegistry | None = None,
    expected_digest: str | None = None,
) -> ScientificPlaybook:
    """Load a validated playbook for runtime execution (`approval_status == 'approved'` unconditionally required)."""
    return load_approved_playbook(
        playbook_id,
        version=version,
        skills_root=skills_root,
        registry=registry,
        expected_digest=expected_digest,
    )


def _read_resource_from_loaded_playbook(
    pb: ScientificPlaybook,
    raw_root: Path,
    posix: Any,
    norm_rel: str,
) -> PlaybookResource:
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


def read_playbook_resource_for_review(
    playbook_id: str,
    resource: str,
    *,
    version: str | None = None,
    skills_root: Path | str | None = None,
    registry: ToolRegistry | None = None,
    allow_disabled: bool = False,
    expected_digest: str | None = None,
) -> PlaybookResource:
    """Explicit human-review API: read a confined resource from a playbook (including `pending_owner_review`)."""
    posix, norm_rel = _validate_and_normalize_resource_arg(resource)
    raw_root = Path(skills_root) if skills_root is not None else DEFAULT_SKILLS_ROOT
    pb = load_playbook_for_review(
        playbook_id,
        version=version,
        skills_root=raw_root,
        registry=registry,
        allow_disabled=allow_disabled,
        expected_digest=expected_digest,
    )
    return _read_resource_from_loaded_playbook(pb, raw_root, posix, norm_rel)


def read_playbook_resource(
    playbook_id: str,
    resource: str,
    *,
    version: str | None = None,
    skills_root: Path | str | None = None,
    registry: ToolRegistry | None = None,
    expected_digest: str | None = None,
) -> PlaybookResource:
    """Read a confined resource from an owner-approved playbook (progressive disclosure tier 3)."""
    posix, norm_rel = _validate_and_normalize_resource_arg(resource)
    raw_root = Path(skills_root) if skills_root is not None else DEFAULT_SKILLS_ROOT
    pb = load_playbook(
        playbook_id,
        version=version,
        skills_root=raw_root,
        registry=registry,
        expected_digest=expected_digest,
    )
    return _read_resource_from_loaded_playbook(pb, raw_root, posix, norm_rel)


def authorize_playbook_tool(
    playbook_or_id: ScientificPlaybook | PlaybookSummary | str,
    tool_name: str,
    *,
    version: str | None = None,
    skills_root: Path | str | None = None,
    registry: ToolRegistry | None = None,
    expected_digest: str | None = None,
) -> bool:
    """Authorize `tool_name` only if the playbook is owner-approved on disk, digest-valid, and allowlists the tool."""
    if not isinstance(tool_name, str) or not tool_name.strip():
        raise PlaybookSecurityError("non_existent_tool: tool_name must be a non-empty string")
    clean_tool = tool_name.strip()
    reg = registry if registry is not None else default_registry()
    if clean_tool not in set(reg.names()):
        raise PlaybookSecurityError(
            f"non_existent_tool: Tool {clean_tool!r} does not exist in the current ToolRegistry"
        )

    if isinstance(playbook_or_id, (ScientificPlaybook, PlaybookSummary)):
        _verify_runtime_approved(playbook_or_id, expected_digest=expected_digest)
        target_root = Path(skills_root) if skills_root is not None else DEFAULT_SKILLS_ROOT
        on_disk = load_approved_playbook(
            playbook_or_id.playbook_id,
            version=version or playbook_or_id.version,
            skills_root=target_root,
            registry=reg,
            expected_digest=expected_digest or playbook_or_id.canonical_digest,
        )
        expected_summary = (
            playbook_or_id.to_summary() if isinstance(playbook_or_id, ScientificPlaybook) else playbook_or_id
        )
        if on_disk.to_summary() != expected_summary:
            raise PlaybookSecurityError("digest_mismatch: in-memory playbook metadata differs from on-disk playbook")
    elif isinstance(playbook_or_id, str):
        on_disk = load_approved_playbook(
            playbook_or_id,
            version=version,
            skills_root=skills_root,
            registry=reg,
            expected_digest=expected_digest,
        )
    else:
        raise PlaybookSecurityError(f"Invalid playbook reference type: {type(playbook_or_id).__name__}")

    if clean_tool not in on_disk.allowed_tools:
        raise PlaybookSecurityError(
            f"undeclared_tool: Tool {clean_tool!r} is not in allowed_tools for playbook {on_disk.playbook_id!r}"
        )
    return True


def check_tool_allowed(
    playbook: ScientificPlaybook | PlaybookSummary,
    tool_name: str,
    *,
    skills_root: Path | str | None = None,
    registry: ToolRegistry | None = None,
    expected_digest: str | None = None,
) -> bool:
    """Enforce that `playbook` is owner-approved, digest-valid on disk, and allowlists `tool_name` in `ToolRegistry`."""
    return authorize_playbook_tool(
        playbook,
        tool_name,
        skills_root=skills_root,
        registry=registry,
        expected_digest=expected_digest,
    )


def playbook_provenance(
    playbook_id: str,
    *,
    version: str | None = None,
    skills_root: Path | str | None = None,
    registry: ToolRegistry | None = None,
    allow_disabled: bool = False,
) -> PlaybookProvenance:
    """Return an immutable provenance identity record for a Scientific Playbook."""
    pb = load_playbook_for_review(
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
    return freeze_value(
        {
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
        }
    )
