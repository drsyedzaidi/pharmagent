"""Schema/governance checks, tool allowlist verification, Markdown link scanning, and catalog validation."""

from __future__ import annotations

import re
import stat
from collections.abc import Mapping
from datetime import date as _date
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from urllib.parse import unquote

from pydantic import ValidationError

from app.tools.base import ToolRegistry
from app.tools.builtins import default_registry

from .digest import compute_canonical_digest
from .filesystem import (
    _URI_SCHEME_RE,
    DEFAULT_SKILLS_ROOT,
    _inspect_playbook_resources,
    _read_utf8_bounded,
)
from .schema import (
    ALLOWED_APPROVAL_STATUSES,
    ALLOWED_DOMAINS,
    ALLOWED_STATUSES,
    MAX_SKILL_MD_BYTES,
    CatalogValidationReport,
    PlaybookValidationIssue,
    PlaybookValidationResult,
    ScientificPlaybook,
)
from .yaml_loader import _ZERO_WIDTH_RE, _parse_frontmatter

_MD_LINK_RE = re.compile(r"!?\[(?:[^\[\]]|\[[^\[\]]*\])*\]\(([^)]*)\)")
_MD_REF_DEF_RE = re.compile(r"(?m)^\s*\[[^\]]+\]:\s*(\S+)")
_MD_AUTOLINK_RE = re.compile(r"<([a-zA-Z][a-zA-Z0-9+.-]*:[^>\s]+)>")
_SEMVER_FULL_RE = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")

_REQUIRED_FRONTMATTER_FIELDS = (
    "name",
    "description",
    "playbook_id",
    "version",
    "domain",
    "status",
    "approval_status",
    "license",
    "source",
    "allowed_tools",
    "required_inputs",
    "produced_artifacts",
    "limitations",
    "requires_human_review",
    "compatibility",
)
_KNOWN_FRONTMATTER_FIELDS = frozenset(
    {
        *_REQUIRED_FRONTMATTER_FIELDS,
        "approved_by",
        "approved_date",
        "runtime_available",
    }
)
_FRONTMATTER_SCAN_KEYS = (
    "description",
    "source",
    "license",
    "approved_by",
    "required_inputs",
    "produced_artifacts",
    "limitations",
    "compatibility",
)


def _issue(
    code: str,
    field_or_path: str,
    message: str,
    severity: Literal["error", "security"] = "error",
) -> PlaybookValidationIssue:
    return PlaybookValidationIssue(code=code, field_or_path=field_or_path, message=message, severity=severity)


def _check_single_markdown_target(
    raw_target: str,
    source_label: str,
    valid_resources: set[str],
) -> list[PlaybookValidationIssue]:
    cleaned = raw_target.strip()
    if not cleaned:
        return []
    target = cleaned.split()[0].strip("<>")
    if target.startswith("#"):
        return []
    lower = target.lower()
    if lower.startswith(("http://", "https://", "mailto:")):
        return []
    if _URI_SCHEME_RE.match(lower) and not re.match(r"^[A-Za-z]:[/\\]", target):
        return [
            _issue(
                "absolute_path", source_label, f"Forbidden URI scheme in markdown reference: {target!r}", "security"
            ),
            _issue(
                "external_reference",
                source_label,
                f"External URI reference is outside playbook directory: {target!r}",
                "security",
            ),
        ]

    target_path_only = target.split("#", 1)[0].split("?", 1)[0]
    if not target_path_only:
        return []
    decoded_path = unquote(target_path_only)
    if "\\" in target or "\\" in decoded_path:
        return [
            _issue(
                "path_traversal",
                source_label,
                f"Backslashes are forbidden in markdown resource links: {target!r}",
                "security",
            ),
            _issue(
                "external_reference",
                source_label,
                f"Markdown link with backslash is forbidden: {target!r}",
                "security",
            ),
        ]
    if decoded_path.startswith("/") or re.match(r"^[A-Za-z]:", decoded_path):
        return [
            _issue(
                "absolute_path", source_label, f"Absolute path in markdown link is forbidden: {target!r}", "security"
            ),
            _issue(
                "external_reference",
                source_label,
                f"Markdown link points outside playbook directory: {target!r}",
                "security",
            ),
        ]
    raw_segments = decoded_path.split("/")
    if ".." in raw_segments or "." in raw_segments or "" in raw_segments:
        return [
            _issue(
                "path_traversal",
                source_label,
                f"Path traversal or dot-segment in markdown link is forbidden: {target!r}",
                "security",
            ),
            _issue(
                "external_reference",
                source_label,
                f"Markdown link escapes playbook directory: {target!r}",
                "security",
            ),
        ]
    norm_rel = PurePosixPath(decoded_path).as_posix()
    if norm_rel != "SKILL.md" and norm_rel not in valid_resources:
        return [
            _issue(
                "external_reference",
                source_label,
                f"Referenced resource {norm_rel!r} does not exist in confined playbook directory",
                "security",
            )
        ]
    return []


def _scan_markdown_links(
    text: str,
    source_label: str,
    valid_resources: set[str],
) -> list[PlaybookValidationIssue]:
    """Scan Markdown text for path traversal, absolute paths, forbidden URI schemes, or missing resources."""
    raw_targets: list[str] = [m.group(1) for m in _MD_LINK_RE.finditer(text)]
    raw_targets.extend(m.group(1) for m in _MD_REF_DEF_RE.finditer(text))
    raw_targets.extend(m.group(1) for m in _MD_AUTOLINK_RE.finditer(text))
    issues: list[PlaybookValidationIssue] = []
    for raw_target in raw_targets:
        issues.extend(_check_single_markdown_target(raw_target, source_label, valid_resources))
    return issues


def _validate_approval_fields(raw_meta: dict[str, Any]) -> list[PlaybookValidationIssue]:
    issues: list[PlaybookValidationIssue] = []
    raw_status = raw_meta.get("status")
    raw_approval = raw_meta.get("approval_status")
    if raw_approval is None:
        return issues
    if raw_approval not in ALLOWED_APPROVAL_STATUSES:
        return [
            _issue(
                "invalid_approval_status",
                "approval_status",
                f"Invalid approval_status {raw_approval!r}; allowed: {ALLOWED_APPROVAL_STATUSES}",
            )
        ]
    by = raw_meta.get("approved_by")
    dt = raw_meta.get("approved_date")
    if (raw_status == "approved") != (raw_approval == "approved"):
        issues.append(
            _issue(
                "invalid_approval_status",
                "approval_status",
                "status='approved' and approval_status='approved' must both be set together",
            )
        )
    if raw_approval in ("pending_owner_review", "rejected") and (by is not None or dt is not None):
        issues.append(
            _issue(
                "invalid_approval_status",
                "approval_status",
                f"{raw_approval} playbook cannot specify approved_by or approved_date",
            )
        )
    if raw_approval == "approved":
        if not (isinstance(by, str) and by.strip() and isinstance(dt, str) and dt.strip()):
            issues.append(
                _issue(
                    "invalid_approval_status",
                    "approval_status",
                    "approved playbook requires non-empty approved_by and approved_date",
                )
            )
        elif not re.fullmatch(r"^\d{4}-\d{2}-\d{2}$", dt):
            issues.append(
                _issue("invalid_approval_status", "approved_date", f"approved_date must be YYYY-MM-DD, got {dt!r}")
            )
        else:
            try:
                _date.fromisoformat(dt)
            except ValueError:
                issues.append(
                    _issue(
                        "invalid_approval_status",
                        "approved_date",
                        f"approved_date is not a valid calendar date: {dt!r}",
                    )
                )
    return issues


def _classify_frontmatter_issues(
    raw_meta: dict[str, Any],
    pb_dir_name: str,
) -> list[PlaybookValidationIssue]:
    """Validate frontmatter fields, semantic versioning, status/approval governance, and tool lists."""
    issues: list[PlaybookValidationIssue] = [
        _issue("unknown_field", k, f"Unknown frontmatter field {k!r}")
        for k in sorted(set(raw_meta) - _KNOWN_FRONTMATTER_FIELDS)
    ]
    issues.extend(
        _issue("missing_required_field", k, f"Missing required frontmatter field {k!r}")
        for k in _REQUIRED_FRONTMATTER_FIELDS
        if k not in raw_meta or raw_meta[k] is None
    )

    raw_version = raw_meta.get("version")
    if raw_version is not None and (not isinstance(raw_version, str) or not _SEMVER_FULL_RE.fullmatch(raw_version)):
        issues.append(
            _issue(
                "invalid_version", "version", f"Invalid semantic version {raw_version!r}; expected MAJOR.MINOR.PATCH"
            )
        )

    raw_status = raw_meta.get("status")
    if raw_status is not None and raw_status not in ALLOWED_STATUSES:
        issues.append(
            _issue("invalid_status", "status", f"Invalid status {raw_status!r}; allowed: {ALLOWED_STATUSES}")
        )

    issues.extend(_validate_approval_fields(raw_meta))

    raw_domain = raw_meta.get("domain")
    if raw_domain is not None and raw_domain not in ALLOWED_DOMAINS:
        issues.append(
            _issue("invalid_domain", "domain", f"Invalid domain {raw_domain!r}; allowed: {ALLOWED_DOMAINS}")
        )

    if raw_meta.get("requires_human_review") is not True:
        issues.append(
            _issue(
                "policy_violation",
                "requires_human_review",
                "Scientific Playbooks must set requires_human_review: true",
                "security",
            )
        )

    runtime_avail = raw_meta.get("runtime_available", True)
    if not isinstance(runtime_avail, bool):
        issues.append(_issue("invalid_field", "runtime_available", "runtime_available must be a boolean"))
        runtime_avail = True
    elif runtime_avail and (raw_meta.get("approval_status") == "rejected" or raw_status == "deprecated"):
        issues.append(
            _issue(
                "invalid_approval_status",
                "runtime_available",
                "rejected or deprecated playbooks must set runtime_available: false",
            )
        )

    raw_tools = raw_meta.get("allowed_tools")
    if not isinstance(raw_tools, list):
        issues.append(_issue("invalid_allowed_tools", "allowed_tools", "allowed_tools must be a list of tool names"))
    else:
        if runtime_avail and len(raw_tools) == 0:
            issues.append(
                _issue(
                    "empty_allowed_tools",
                    "allowed_tools",
                    "Runtime-available playbooks must declare at least one tool in allowed_tools",
                    "security",
                )
            )
        if len(set(map(str, raw_tools))) != len(raw_tools):
            issues.append(
                _issue("duplicate_allowed_tool", "allowed_tools", "allowed_tools contains duplicate tool entries")
            )

    raw_name = raw_meta.get("name")
    if isinstance(raw_name, str) and pb_dir_name and raw_name != pb_dir_name:
        issues.append(
            _issue(
                "directory_name_mismatch",
                "name",
                f"Frontmatter name {raw_name!r} does not match directory name {pb_dir_name!r}",
            )
        )
    return issues


def _collect_texts_to_scan(
    raw_meta: dict[str, Any],
    body_text: str,
    resources_map: dict[str, tuple[str, str, int]],
) -> list[tuple[str, str]]:
    texts: list[tuple[str, str]] = [("SKILL.md", body_text)]
    for fm_key in _FRONTMATTER_SCAN_KEYS:
        fm_val = raw_meta.get(fm_key)
        if isinstance(fm_val, str):
            texts.append((f"frontmatter.{fm_key}", fm_val))
        elif isinstance(fm_val, list):
            joined = "\n".join(str(x) for x in fm_val if isinstance(x, str))
            if joined:
                texts.append((f"frontmatter.{fm_key}", joined))
        elif isinstance(fm_val, Mapping):
            joined = "\n".join(str(v) for v in fm_val.values() if isinstance(v, (str, list)))
            if joined:
                texts.append((f"frontmatter.{fm_key}", joined))
    for rel_path, (res_text, _sha, _sz) in sorted(resources_map.items()):
        texts.append((rel_path, res_text))
    return texts


def _scan_tools_and_links(
    raw_meta: dict[str, Any],
    body_text: str,
    resources_map: dict[str, tuple[str, str, int]],
    registered_tools: set[str],
) -> list[PlaybookValidationIssue]:
    issues: list[PlaybookValidationIssue] = []
    declared_tools = [
        item.strip()
        for item in (raw_meta.get("allowed_tools") or [])
        if isinstance(raw_meta.get("allowed_tools"), list) and isinstance(item, str) and item.strip()
    ]
    declared_set = set(declared_tools)

    for tool_name in declared_tools:
        if tool_name not in registered_tools:
            issues.append(
                _issue(
                    "non_existent_tool",
                    "allowed_tools",
                    f"Declared tool {tool_name!r} does not exist in the current ToolRegistry",
                    "security",
                )
            )

    valid_res_set = set(resources_map)
    for source_label, text_content in _collect_texts_to_scan(raw_meta, body_text, resources_map):
        cleaned = _ZERO_WIDTH_RE.sub("", text_content).replace("\ufeff", "")
        for reg_tool in sorted(registered_tools):
            if reg_tool not in declared_set and re.search(rf"\b{re.escape(reg_tool)}\b", cleaned):
                issues.append(
                    _issue(
                        "undeclared_tool",
                        source_label,
                        f"Playbook content in {source_label} references registered tool "
                        f"{reg_tool!r} which is not declared in allowed_tools",
                        "security",
                    )
                )
        issues.extend(_scan_markdown_links(text_content, source_label, valid_res_set))
    return issues


def _resolve_playbook_dir_for_validation(
    pb_dir: Path,
    skills_root: Path | str | None,
) -> tuple[Path | None, PlaybookValidationIssue | None]:
    try:
        dir_lstat = pb_dir.lstat()
    except OSError as exc:
        return None, _issue("missing_playbook_dir", str(pb_dir), f"Playbook directory does not exist: {exc}")
    if stat.S_ISLNK(dir_lstat.st_mode):
        return None, _issue(
            "symlink_escape", str(pb_dir.name), "Playbook directory must not be a symlink", "security"
        )
    try:
        real_pb_dir = pb_dir.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        return None, _issue(
            "symlink_escape", str(pb_dir.name), f"Cannot resolve playbook directory: {exc}", "security"
        )
    if skills_root is not None:
        try:
            real_skills_root = Path(skills_root).resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            return None, _issue("invalid_skills_root", str(skills_root), f"Invalid skills_root: {exc}")
        if not real_pb_dir.is_relative_to(real_skills_root):
            return None, _issue(
                "symlink_escape",
                str(pb_dir.name),
                "Playbook directory resolves outside skills_root",
                "security",
            )
    return real_pb_dir, None


def validate_playbook(
    playbook_dir_or_file: Path | str,
    *,
    skills_root: Path | str | None = None,
    registry: ToolRegistry | None = None,
) -> PlaybookValidationResult:
    """Deterministically validate a single playbook directory or SKILL.md path."""
    reg = registry if registry is not None else default_registry()
    registered_tools = set(reg.names())

    target = Path(playbook_dir_or_file)
    pb_dir = target.parent if target.name == "SKILL.md" else target
    skill_file = target if target.name == "SKILL.md" else target / "SKILL.md"

    real_pb_dir, dir_issue = _resolve_playbook_dir_for_validation(pb_dir, skills_root)
    if dir_issue is not None or real_pb_dir is None:
        return PlaybookValidationResult(valid=False, issues=(dir_issue,) if dir_issue else ())

    actual_dir_name = real_pb_dir.name or pb_dir.name
    try:
        dir_entry_names = {p.name for p in real_pb_dir.iterdir()}
    except OSError:
        dir_entry_names = set()

    if ("SKILL.md" not in dir_entry_names and not skill_file.is_symlink()) or (
        not skill_file.exists() and not skill_file.is_symlink()
    ):
        return PlaybookValidationResult(
            valid=False,
            name=actual_dir_name,
            issues=(_issue("missing_skill_md", "SKILL.md", "Playbook directory is missing SKILL.md"),),
        )

    raw_skill_text, read_issues = _read_utf8_bounded(skill_file, MAX_SKILL_MD_BYTES, "SKILL.md")
    resources_map, resource_issues = _inspect_playbook_resources(pb_dir, real_pb_dir)
    issues: list[PlaybookValidationIssue] = [*read_issues, *resource_issues]
    if raw_skill_text is None:
        return PlaybookValidationResult(valid=False, name=actual_dir_name, issues=tuple(issues))

    raw_meta, body_text, parse_issues = _parse_frontmatter(raw_skill_text)
    issues.extend(parse_issues)
    if raw_meta is None:
        return PlaybookValidationResult(valid=False, name=actual_dir_name, issues=tuple(issues))

    if "allowed_tools" in raw_meta and raw_meta["allowed_tools"] is None:
        raw_meta = {**raw_meta, "allowed_tools": []}
    if not body_text:
        issues.append(
            _issue(
                "empty_instructions",
                "SKILL.md",
                "SKILL.md must contain non-empty markdown instructions after frontmatter",
            )
        )

    issues.extend(_classify_frontmatter_issues(raw_meta, actual_dir_name))
    issues.extend(_scan_tools_and_links(raw_meta, body_text, resources_map, registered_tools))

    pb_id = raw_meta.get("playbook_id") if isinstance(raw_meta.get("playbook_id"), str) else None
    pb_name = raw_meta.get("name") if isinstance(raw_meta.get("name"), str) else actual_dir_name
    pb_ver = raw_meta.get("version") if isinstance(raw_meta.get("version"), str) else None
    pb_status = raw_meta.get("status") if isinstance(raw_meta.get("status"), str) else None
    pb_approval = raw_meta.get("approval_status") if isinstance(raw_meta.get("approval_status"), str) else None
    raw_rt = raw_meta.get("runtime_available", True)
    pb_runtime = raw_rt if isinstance(raw_rt, bool) else None

    if issues:
        return PlaybookValidationResult(
            valid=False,
            playbook_id=pb_id,
            name=pb_name,
            version=pb_ver,
            status=pb_status,
            approval_status=pb_approval,
            runtime_available=pb_runtime,
            issues=tuple(issues),
        )

    resource_paths = tuple(sorted(resources_map))
    resource_digests = tuple((rel, resources_map[rel][1]) for rel in resource_paths)
    try:
        digest = compute_canonical_digest(raw_meta, instructions=body_text, resource_digests=resource_digests)
        playbook = ScientificPlaybook(
            **raw_meta,
            canonical_digest=digest,
            instructions=body_text,
            resource_paths=resource_paths,
            resource_digests=resource_digests,
        )
    except (ValidationError, ValueError, TypeError) as exc:
        return PlaybookValidationResult(
            valid=False,
            playbook_id=pb_id,
            name=pb_name,
            version=pb_ver,
            status=pb_status,
            approval_status=pb_approval,
            runtime_available=pb_runtime,
            issues=(_issue("schema_validation_error", "frontmatter", str(exc)),),
        )

    return PlaybookValidationResult(
        valid=True,
        playbook_id=playbook.playbook_id,
        name=playbook.name,
        version=playbook.version,
        canonical_digest=playbook.canonical_digest,
        status=playbook.status,
        approval_status=playbook.approval_status,
        runtime_available=playbook.runtime_available,
        issues=(),
        playbook=playbook,
    )


def _check_catalog_duplicates(
    results: list[PlaybookValidationResult],
) -> tuple[list[PlaybookValidationIssue], set[str]]:
    seen_ids: dict[str, str] = {}
    seen_names: dict[str, str] = {}
    seen_versions: dict[str, set[str]] = {}
    duplicate_identities: set[str] = set()
    issues: list[PlaybookValidationIssue] = []

    for res in results:
        if res.playbook_id:
            seen_versions.setdefault(res.playbook_id, set()).add(res.version or "")
            if res.playbook_id in seen_ids:
                duplicate_identities.add(res.playbook_id)
                issues.append(
                    _issue(
                        "duplicate_playbook_id",
                        res.playbook_id,
                        f"Duplicate playbook_id {res.playbook_id!r} in {res.name!r} and {seen_ids[res.playbook_id]!r}",
                        "security",
                    )
                )
                if len(seen_versions[res.playbook_id]) > 1:
                    issues.append(
                        _issue(
                            "conflicting_version",
                            res.playbook_id,
                            f"Conflicting versions {sorted(seen_versions[res.playbook_id])} "
                            f"for playbook_id {res.playbook_id!r}",
                        )
                    )
            else:
                seen_ids[res.playbook_id] = res.name or ""

        if res.name:
            if res.name in seen_names:
                duplicate_identities.add(res.name)
                issues.append(_issue("duplicate_name", res.name, f"Duplicate playbook name {res.name!r}", "security"))
            else:
                seen_names[res.name] = res.playbook_id or ""

    for pid, owner_name in seen_ids.items():
        if pid in seen_names and seen_names[pid] != pid:
            duplicate_identities.add(pid)
            issues.append(
                _issue(
                    "duplicate_playbook_id",
                    pid,
                    f"playbook_id {pid!r} (in {owner_name!r}) collides with another playbook's name",
                    "security",
                )
            )
    return issues, duplicate_identities


def _empty_catalog_report(root: Path, issue: PlaybookValidationIssue) -> CatalogValidationReport:
    return CatalogValidationReport(
        valid=False,
        skills_root=str(root),
        total_playbooks=0,
        approved_count=0,
        pending_approval_count=0,
        draft_count=0,
        reviewed_count=0,
        deprecated_count=0,
        runtime_available_count=0,
        disabled_count=0,
        results=(),
        catalog_issues=(issue,),
    )


def validate_all_playbooks(
    skills_root: Path | str | None = None,
    *,
    registry: ToolRegistry | None = None,
) -> CatalogValidationReport:
    """Validate all playbooks under `skills_root` and check cross-playbook uniqueness."""
    root = Path(skills_root) if skills_root is not None else DEFAULT_SKILLS_ROOT
    reg = registry if registry is not None else default_registry()

    if not root.exists():
        return _empty_catalog_report(
            root,
            _issue("missing_skills_root", str(root), f"Skills catalog directory does not exist: {root}"),
        )
    try:
        real_root = root.resolve(strict=True)
        entries = sorted(root.iterdir(), key=lambda p: p.name)
    except OSError as exc:
        return _empty_catalog_report(
            root,
            _issue("unreadable_skills_root", str(root), f"Cannot read skills root: {exc}"),
        )

    catalog_issues: list[PlaybookValidationIssue] = []
    results: list[PlaybookValidationResult] = []
    for entry in entries:
        try:
            lst = entry.lstat()
        except OSError as exc:
            catalog_issues.append(_issue("unreadable_file", entry.name, str(exc)))
            continue
        if stat.S_ISLNK(lst.st_mode):
            catalog_issues.append(
                _issue(
                    "symlink_escape",
                    entry.name,
                    f"Top-level symlink {entry.name!r} is forbidden in skills_root",
                    "security",
                )
            )
        elif entry.name.startswith("."):
            catalog_issues.append(
                _issue(
                    "hidden_file", entry.name, f"Hidden entry {entry.name!r} is forbidden in skills_root", "security"
                )
            )
        elif not stat.S_ISDIR(lst.st_mode):
            catalog_issues.append(
                _issue(
                    "unexpected_root_file",
                    entry.name,
                    f"Unexpected non-directory entry {entry.name!r} in skills_root",
                    "security",
                )
            )
        else:
            results.append(validate_playbook(entry, skills_root=real_root, registry=reg))

    dup_issues, duplicate_identities = _check_catalog_duplicates(results)
    catalog_issues.extend(dup_issues)

    all_issues = [issue for r in results for issue in r.issues] + catalog_issues
    missing_tools = sorted({i.message for i in all_issues if i.code in {"non_existent_tool", "undeclared_tool"}})
    security_violations = tuple(i for i in all_issues if i.severity == "security")

    return CatalogValidationReport(
        valid=bool(results) and all(r.valid for r in results) and not catalog_issues,
        skills_root=str(root),
        total_playbooks=len(results),
        approved_count=sum(1 for r in results if r.valid and r.approval_status == "approved"),
        pending_approval_count=sum(1 for r in results if r.approval_status == "pending_owner_review"),
        draft_count=sum(1 for r in results if r.status == "draft"),
        reviewed_count=sum(1 for r in results if r.status == "reviewed"),
        deprecated_count=sum(1 for r in results if r.status == "deprecated"),
        runtime_available_count=sum(1 for r in results if r.valid and r.runtime_available is True),
        disabled_count=sum(1 for r in results if r.runtime_available is False),
        results=tuple(results),
        catalog_issues=tuple(catalog_issues),
        missing_tool_references=tuple(missing_tools),
        duplicate_identities=tuple(sorted(duplicate_identities)),
        security_violations=security_violations,
    )
