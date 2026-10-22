"""Deterministic catalog discovery, validator, progressive-disclosure loader, and digest calculation."""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any
from urllib.parse import unquote

import yaml
from pydantic import ValidationError

from app.tools.base import ToolRegistry
from app.tools.builtins import default_registry

from .schema import (
    ALLOWED_APPROVAL_STATUSES,
    ALLOWED_DOMAINS,
    ALLOWED_RESOURCE_EXTENSIONS,
    ALLOWED_STATUSES,
    FORBIDDEN_SCRIPT_EXTENSIONS,
    MAX_RESOURCE_BYTES,
    MAX_SKILL_MD_BYTES,
    CatalogValidationReport,
    PlaybookCompatibility,
    PlaybookDisabledError,
    PlaybookLookupError,
    PlaybookProvenance,
    PlaybookResource,
    PlaybookSecurityError,
    PlaybookSummary,
    PlaybookValidationError,
    PlaybookValidationIssue,
    PlaybookValidationResult,
    PlaybookVersionError,
    ScientificPlaybook,
    freeze_value,
)

DEFAULT_SKILLS_ROOT = Path(__file__).resolve().parents[3] / "skills"

_FRONTMATTER_RE = re.compile(r"\A---[ \t]*\n(?P<yaml>.*?)\n---[ \t]*(?:\n(?P<body>.*))?\Z", re.DOTALL)
_MD_LINK_RE = re.compile(r"!?\[(?:[^\[\]]|\[[^\[\]]*\])*\]\(([^)]*)\)")
_MD_REF_DEF_RE = re.compile(r"(?m)^\s*\[[^\]]+\]:\s*(\S+)")
_MD_AUTOLINK_RE = re.compile(r"<([a-zA-Z][a-zA-Z0-9+.-]*:[^>\s]+)>")
_URI_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")
_ZERO_WIDTH_RE = re.compile(r"[\u00ad\u200b-\u200f\u202a-\u202e\u2060-\u206f]")
MAX_YAML_DEPTH = 16
_SAFE_YAML_TAGS = frozenset({
    "tag:yaml.org,2002:str",
    "tag:yaml.org,2002:int",
    "tag:yaml.org,2002:float",
    "tag:yaml.org,2002:bool",
    "tag:yaml.org,2002:null",
    "tag:yaml.org,2002:seq",
    "tag:yaml.org,2002:map",
})
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
_KNOWN_FRONTMATTER_FIELDS = frozenset({
    *_REQUIRED_FRONTMATTER_FIELDS,
    "approved_by",
    "approved_date",
    "runtime_available",
})


class _DuplicateYamlKeyError(ValueError):
    """Raised when YAML frontmatter contains duplicate mapping keys."""


class _UnsafeYamlError(ValueError):
    """Raised when YAML frontmatter uses unsafe tags, constructors, anchors, or aliases."""


class _StrictSafeLoader(yaml.SafeLoader):
    """YAML SafeLoader subclass that forbids duplicate keys, custom tags, anchors, aliases, and implicit dates."""

    # Copy yaml_implicit_resolvers into _StrictSafeLoader's own class dict without mutating yaml.SafeLoader
    yaml_implicit_resolvers = {
        ch: [(tag, regexp) for tag, regexp in resolvers if tag != "tag:yaml.org,2002:timestamp"]
        for ch, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
    }


def _strict_compose_node(self: _StrictSafeLoader, parent: Any, index: Any) -> Any:
    depth = getattr(self, "_yaml_depth", 0) + 1
    if depth > MAX_YAML_DEPTH:
        raise _UnsafeYamlError(f"YAML nesting depth exceeds maximum allowed ({MAX_YAML_DEPTH})")
    self._yaml_depth = depth  # type: ignore[attr-defined]
    try:
        if self.check_event(yaml.AliasEvent):
            raise _UnsafeYamlError("YAML aliases and anchors are forbidden in playbook frontmatter")
        event = self.peek_event()
        if getattr(event, "anchor", None) is not None:
            raise _UnsafeYamlError("YAML anchors are forbidden in playbook frontmatter")
        if getattr(event, "tag", None) not in (None, "!", "?") and event.tag not in _SAFE_YAML_TAGS:
            raise _UnsafeYamlError(f"Unsafe or unsupported YAML tag: {event.tag!r}")
        node = yaml.SafeLoader.compose_node(self, parent, index)
        if node.tag not in _SAFE_YAML_TAGS:
            raise _UnsafeYamlError(f"Unsafe or unsupported YAML tag: {node.tag!r}")
        return node
    finally:
        self._yaml_depth = depth - 1  # type: ignore[attr-defined]


def _strict_construct_mapping(self: _StrictSafeLoader, node: yaml.MappingNode, deep: bool = False) -> dict[str, Any]:
    if not isinstance(node, yaml.MappingNode):
        raise _UnsafeYamlError("Expected a YAML mapping node")
    mapping: dict[str, Any] = {}
    for key_node, value_node in node.value:
        if key_node.tag != "tag:yaml.org,2002:str":
            raise _UnsafeYamlError(f"YAML mapping keys must be strings, got tag {key_node.tag!r}")
        key = self.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise _UnsafeYamlError("YAML mapping keys must be strings")
        if key in mapping:
            raise _DuplicateYamlKeyError(f"Duplicate YAML key {key!r} in frontmatter")
        mapping[key] = self.construct_object(value_node, deep=deep)
    return mapping


_StrictSafeLoader.compose_node = _strict_compose_node  # type: ignore[assignment]
_StrictSafeLoader.add_constructor("tag:yaml.org,2002:map", _strict_construct_mapping)


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(_normalize_newlines(text).encode("utf-8")).hexdigest()


def _canonical_metadata_dict(meta: Mapping[str, Any]) -> dict[str, Any]:
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


def _parse_frontmatter(raw_text: str) -> tuple[dict[str, Any] | None, str, list[PlaybookValidationIssue]]:
    normalized = _normalize_newlines(raw_text)
    m = _FRONTMATTER_RE.match(normalized)
    if not m:
        return None, "", [
            PlaybookValidationIssue(
                code="invalid_frontmatter",
                field_or_path="SKILL.md",
                message="SKILL.md must begin with YAML frontmatter delimited by `---` lines",
                severity="error",
            )
        ]
    yaml_block = m.group("yaml")
    body = (m.group("body") or "").strip()
    try:
        parsed = yaml.load(yaml_block, Loader=_StrictSafeLoader)
    except _DuplicateYamlKeyError as exc:
        return None, body, [
            PlaybookValidationIssue(
                code="duplicate_yaml_key",
                field_or_path="frontmatter",
                message=str(exc),
                severity="security",
            )
        ]
    except (_UnsafeYamlError, RecursionError) as exc:
        msg = str(exc) if isinstance(exc, _UnsafeYamlError) else "YAML nesting depth exceeded recursion limit"
        return None, body, [
            PlaybookValidationIssue(
                code="unsafe_yaml",
                field_or_path="frontmatter",
                message=msg,
                severity="security",
            )
        ]
    except yaml.constructor.ConstructorError as exc:
        return None, body, [
            PlaybookValidationIssue(
                code="unsafe_yaml",
                field_or_path="frontmatter",
                message=f"Unsafe YAML constructor rejected: {exc}",
                severity="security",
            )
        ]
    except yaml.YAMLError as exc:
        return None, body, [
            PlaybookValidationIssue(
                code="invalid_frontmatter",
                field_or_path="frontmatter",
                message=f"Invalid YAML syntax in frontmatter: {exc}",
                severity="error",
            )
        ]

    if not isinstance(parsed, dict):
        return None, body, [
            PlaybookValidationIssue(
                code="invalid_frontmatter",
                field_or_path="frontmatter",
                message="YAML frontmatter must be a key-value mapping",
                severity="error",
            )
        ]
    return parsed, body, []


def _read_utf8_bounded(path: Path, max_bytes: int, rel_label: str) -> tuple[str | None, list[PlaybookValidationIssue]]:
    try:
        st = path.lstat()
    except OSError as exc:
        return None, [
            PlaybookValidationIssue(
                code="unreadable_file",
                field_or_path=rel_label,
                message=f"Cannot stat file: {exc}",
                severity="error",
            )
        ]
    if stat.S_ISLNK(st.st_mode):
        return None, [
            PlaybookValidationIssue(
                code="symlink_escape",
                field_or_path=rel_label,
                message="Symlinks are forbidden inside playbook directories",
                severity="security",
            )
        ]
    if not stat.S_ISREG(st.st_mode):
        return None, [
            PlaybookValidationIssue(
                code="invalid_file_type",
                field_or_path=rel_label,
                message="Playbook entries must be regular files",
                severity="security",
            )
        ]
    if st.st_size > max_bytes:
        return None, [
            PlaybookValidationIssue(
                code="oversized_file",
                field_or_path=rel_label,
                message=f"File size {st.st_size} bytes exceeds limit of {max_bytes} bytes",
                severity="security",
            )
        ]
    open_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(path, open_flags)
        try:
            fst = os.fstat(fd)
            if not stat.S_ISREG(fst.st_mode):
                return None, [
                    PlaybookValidationIssue(
                        code="invalid_file_type",
                        field_or_path=rel_label,
                        message="Opened file descriptor is not a regular file",
                        severity="security",
                    )
                ]
            if fst.st_size > max_bytes:
                return None, [
                    PlaybookValidationIssue(
                        code="oversized_file",
                        field_or_path=rel_label,
                        message=f"File size {fst.st_size} bytes exceeds limit of {max_bytes} bytes",
                        severity="security",
                    )
                ]
            with os.fdopen(fd, "rb") as fh:
                fd = -1
                raw = fh.read(max_bytes + 1)
        finally:
            if fd >= 0:
                os.close(fd)
    except OSError as exc:
        return None, [
            PlaybookValidationIssue(
                code="unreadable_file",
                field_or_path=rel_label,
                message=f"Cannot read file: {exc}",
                severity="error",
            )
        ]
    if len(raw) > max_bytes:
        return None, [
            PlaybookValidationIssue(
                code="oversized_file",
                field_or_path=rel_label,
                message=f"File exceeds limit of {max_bytes} bytes",
                severity="security",
            )
        ]
    if b"\x00" in raw:
        return None, [
            PlaybookValidationIssue(
                code="invalid_encoding",
                field_or_path=rel_label,
                message="Null bytes are forbidden in playbook files",
                severity="security",
            )
        ]
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        return None, [
            PlaybookValidationIssue(
                code="invalid_encoding",
                field_or_path=rel_label,
                message=f"File is not valid UTF-8: {exc}",
                severity="security",
            )
        ]
    if _ZERO_WIDTH_RE.search(text):
        return None, [
            PlaybookValidationIssue(
                code="invalid_encoding",
                field_or_path=rel_label,
                message="Zero-width or bidirectional control characters are forbidden in playbook files",
                severity="security",
            )
        ]
    return text, []


def _scan_markdown_links(
    text: str,
    source_label: str,
    valid_resources: set[str],
) -> list[PlaybookValidationIssue]:
    issues: list[PlaybookValidationIssue] = []
    raw_targets: list[str] = [m.group(1) for m in _MD_LINK_RE.finditer(text)]
    raw_targets.extend(m.group(1) for m in _MD_REF_DEF_RE.finditer(text))
    raw_targets.extend(m.group(1) for m in _MD_AUTOLINK_RE.finditer(text))

    for raw_target in raw_targets:
        cleaned = raw_target.strip()
        if not cleaned:
            continue
        target = cleaned.split()[0].strip("<>")
        if target.startswith("#"):
            continue
        lower = target.lower()
        if lower.startswith(("http://", "https://", "mailto:")):
            continue
        if _URI_SCHEME_RE.match(lower) and not re.match(r"^[A-Za-z]:[/\\]", target):
            issues.append(
                PlaybookValidationIssue(
                    code="absolute_path",
                    field_or_path=source_label,
                    message=f"Forbidden URI scheme in markdown reference: {target!r}",
                    severity="security",
                )
            )
            issues.append(
                PlaybookValidationIssue(
                    code="external_reference",
                    field_or_path=source_label,
                    message=f"External URI reference is outside playbook directory: {target!r}",
                    severity="security",
                )
            )
            continue

        target_path_only = target.split("#", 1)[0].split("?", 1)[0]
        if not target_path_only:
            continue
        decoded_path = unquote(target_path_only)
        if "\\" in target or "\\" in decoded_path:
            issues.append(
                PlaybookValidationIssue(
                    code="path_traversal",
                    field_or_path=source_label,
                    message=f"Backslashes are forbidden in markdown resource links: {target!r}",
                    severity="security",
                )
            )
            issues.append(
                PlaybookValidationIssue(
                    code="external_reference",
                    field_or_path=source_label,
                    message=f"Markdown link with backslash is forbidden: {target!r}",
                    severity="security",
                )
            )
            continue

        if decoded_path.startswith("/") or re.match(r"^[A-Za-z]:", decoded_path):
            issues.append(
                PlaybookValidationIssue(
                    code="absolute_path",
                    field_or_path=source_label,
                    message=f"Absolute path in markdown link is forbidden: {target!r}",
                    severity="security",
                )
            )
            issues.append(
                PlaybookValidationIssue(
                    code="external_reference",
                    field_or_path=source_label,
                    message=f"Markdown link points outside playbook directory: {target!r}",
                    severity="security",
                )
            )
            continue

        raw_segments = decoded_path.split("/")
        if ".." in raw_segments or "." in raw_segments or "" in raw_segments:
            issues.append(
                PlaybookValidationIssue(
                    code="path_traversal",
                    field_or_path=source_label,
                    message=f"Path traversal or dot-segment in markdown link is forbidden: {target!r}",
                    severity="security",
                )
            )
            issues.append(
                PlaybookValidationIssue(
                    code="external_reference",
                    field_or_path=source_label,
                    message=f"Markdown link escapes playbook directory: {target!r}",
                    severity="security",
                )
            )
            continue

        norm_rel = PurePosixPath(decoded_path).as_posix()
        if norm_rel != "SKILL.md" and norm_rel not in valid_resources:
            issues.append(
                PlaybookValidationIssue(
                    code="external_reference",
                    field_or_path=source_label,
                    message=f"Referenced resource {norm_rel!r} does not exist in confined playbook directory",
                    severity="security",
                )
            )
    return issues


def _inspect_playbook_resources(
    pb_dir: Path,
    real_pb_dir: Path,
) -> tuple[dict[str, tuple[str, str, int]], list[PlaybookValidationIssue]]:
    """Walk playbook directory deterministically and validate all non-SKILL.md files."""
    resources: dict[str, tuple[str, str, int]] = {}
    issues: list[PlaybookValidationIssue] = []

    stack: list[Path] = [pb_dir]
    while stack:
        current = stack.pop()
        try:
            entries = sorted(current.iterdir(), key=lambda p: p.name)
        except OSError as exc:
            issues.append(
                PlaybookValidationIssue(
                    code="unreadable_file",
                    field_or_path=str(current.name),
                    message=f"Cannot list directory: {exc}",
                    severity="error",
                )
            )
            continue
        for entry in entries:
            rel_posix = entry.relative_to(pb_dir).as_posix()
            try:
                lst = entry.lstat()
            except OSError as exc:
                issues.append(
                    PlaybookValidationIssue(
                        code="unreadable_file",
                        field_or_path=rel_posix,
                        message=f"Cannot stat entry: {exc}",
                        severity="error",
                    )
                )
                continue

            if stat.S_ISLNK(lst.st_mode):
                issues.append(
                    PlaybookValidationIssue(
                        code="symlink_escape",
                        field_or_path=rel_posix,
                        message=f"Symlink {rel_posix!r} is forbidden inside playbook directory",
                        severity="security",
                    )
                )
                continue

            try:
                resolved_entry = entry.resolve(strict=True)
            except (OSError, RuntimeError):
                issues.append(
                    PlaybookValidationIssue(
                        code="symlink_escape",
                        field_or_path=rel_posix,
                        message=f"Entry {rel_posix!r} could not be safely resolved",
                        severity="security",
                    )
                )
                continue

            if not resolved_entry.is_relative_to(real_pb_dir):
                issues.append(
                    PlaybookValidationIssue(
                        code="symlink_escape",
                        field_or_path=rel_posix,
                        message=f"Entry {rel_posix!r} resolves outside playbook root",
                        severity="security",
                    )
                )
                continue

            if entry.name.startswith("."):
                issues.append(
                    PlaybookValidationIssue(
                        code="hidden_file",
                        field_or_path=rel_posix,
                        message=f"Hidden file or directory {rel_posix!r} is forbidden in playbooks",
                        severity="security",
                    )
                )
                if stat.S_ISREG(lst.st_mode):
                    if (lst.st_mode & 0o111) or any(
                        s.lower() in FORBIDDEN_SCRIPT_EXTENSIONS for s in entry.suffixes
                    ):
                        issues.append(
                            PlaybookValidationIssue(
                                code="executable_or_script_file",
                                field_or_path=rel_posix,
                                message=f"Hidden executable or script {rel_posix!r} is forbidden",
                                severity="security",
                            )
                        )
                continue

            if stat.S_ISDIR(lst.st_mode):
                if entry.name.lower() in {"scripts", "bin", "__pycache__", "hooks", "tools", "libexec"}:
                    issues.append(
                        PlaybookValidationIssue(
                            code="executable_or_script_file",
                            field_or_path=rel_posix,
                            message=f"Bundled script directory {rel_posix!r} is forbidden in the MVP",
                            severity="security",
                        )
                    )
                elif rel_posix != "references" and not rel_posix.startswith("references/"):
                    issues.append(
                        PlaybookValidationIssue(
                            code="disallowed_resource_type",
                            field_or_path=rel_posix,
                            message=(
                                f"Only `references/` subdirectory is permitted inside a playbook, got {rel_posix!r}"
                            ),
                            severity="security",
                        )
                    )
                stack.append(entry)
                continue

            if rel_posix.lower() == "skill.md":
                if rel_posix != "SKILL.md":
                    issues.append(
                        PlaybookValidationIssue(
                            code="missing_skill_md",
                            field_or_path=rel_posix,
                            message=f"Playbook file must be named uppercase 'SKILL.md', got {rel_posix!r}",
                            severity="error",
                        )
                    )
                if lst.st_mode & 0o111:
                    issues.append(
                        PlaybookValidationIssue(
                            code="executable_or_script_file",
                            field_or_path=rel_posix,
                            message="SKILL.md must not have executable permission bits set",
                            severity="security",
                        )
                    )
                continue

            suffix = entry.suffix.lower()
            if any(s.lower() in FORBIDDEN_SCRIPT_EXTENSIONS for s in entry.suffixes) or (lst.st_mode & 0o111):
                issues.append(
                    PlaybookValidationIssue(
                        code="executable_or_script_file",
                        field_or_path=rel_posix,
                        message=f"Executable or script file {rel_posix!r} is forbidden in the MVP",
                        severity="security",
                    )
                )
                continue

            if suffix not in ALLOWED_RESOURCE_EXTENSIONS:
                issues.append(
                    PlaybookValidationIssue(
                        code="disallowed_resource_type",
                        field_or_path=rel_posix,
                        message=(
                            f"Resource {rel_posix!r} has disallowed extension {suffix!r}; "
                            f"allowed: {ALLOWED_RESOURCE_EXTENSIONS}"
                        ),
                        severity="security",
                    )
                )
                continue

            text, read_issues = _read_utf8_bounded(entry, MAX_RESOURCE_BYTES, rel_posix)
            if read_issues:
                issues.extend(read_issues)
                continue
            assert text is not None
            if text.lstrip("\ufeff \t\r\n").startswith("#!"):
                issues.append(
                    PlaybookValidationIssue(
                        code="executable_or_script_file",
                        field_or_path=rel_posix,
                        message=f"Resource {rel_posix!r} begins with a shebang (`#!`) and is rejected as a script",
                        severity="security",
                    )
                )
                continue

            resources[rel_posix] = (text, _sha256_text(text), len(text.encode("utf-8")))

    return resources, issues


def _classify_frontmatter_issues(
    raw_meta: dict[str, Any],
    pb_dir_name: str,
) -> list[PlaybookValidationIssue]:
    issues: list[PlaybookValidationIssue] = []

    unknown = sorted(set(raw_meta) - _KNOWN_FRONTMATTER_FIELDS)
    for k in unknown:
        issues.append(
            PlaybookValidationIssue(
                code="unknown_field",
                field_or_path=k,
                message=f"Unknown frontmatter field {k!r}",
                severity="error",
            )
        )

    missing = [k for k in _REQUIRED_FRONTMATTER_FIELDS if k not in raw_meta or raw_meta[k] is None]
    for k in missing:
        issues.append(
            PlaybookValidationIssue(
                code="missing_required_field",
                field_or_path=k,
                message=f"Missing required frontmatter field {k!r}",
                severity="error",
            )
        )

    raw_version = raw_meta.get("version")
    if raw_version is not None and (
        not isinstance(raw_version, str)
        or not re.fullmatch(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$", raw_version)
    ):
        issues.append(
            PlaybookValidationIssue(
                code="invalid_version",
                field_or_path="version",
                message=f"Invalid semantic version {raw_version!r}; expected MAJOR.MINOR.PATCH",
                severity="error",
            )
        )

    raw_status = raw_meta.get("status")
    if raw_status is not None and raw_status not in ALLOWED_STATUSES:
        issues.append(
            PlaybookValidationIssue(
                code="invalid_status",
                field_or_path="status",
                message=f"Invalid status {raw_status!r}; allowed: {ALLOWED_STATUSES}",
                severity="error",
            )
        )

    raw_approval = raw_meta.get("approval_status")
    if raw_approval is not None and raw_approval not in ALLOWED_APPROVAL_STATUSES:
        issues.append(
            PlaybookValidationIssue(
                code="invalid_approval_status",
                field_or_path="approval_status",
                message=f"Invalid approval_status {raw_approval!r}; allowed: {ALLOWED_APPROVAL_STATUSES}",
                severity="error",
            )
        )
    elif raw_approval in ALLOWED_APPROVAL_STATUSES:
        by = raw_meta.get("approved_by")
        dt = raw_meta.get("approved_date")
        if (raw_status == "approved") != (raw_approval == "approved"):
            issues.append(
                PlaybookValidationIssue(
                    code="invalid_approval_status",
                    field_or_path="approval_status",
                    message="status='approved' and approval_status='approved' must both be set together",
                    severity="error",
                )
            )
        if raw_approval in ("pending_owner_review", "rejected") and (by is not None or dt is not None):
            issues.append(
                PlaybookValidationIssue(
                    code="invalid_approval_status",
                    field_or_path="approval_status",
                    message=f"{raw_approval} playbook cannot specify approved_by or approved_date",
                    severity="error",
                )
            )
        if raw_approval == "approved":
            if not (isinstance(by, str) and by.strip() and isinstance(dt, str) and dt.strip()):
                issues.append(
                    PlaybookValidationIssue(
                        code="invalid_approval_status",
                        field_or_path="approval_status",
                        message="approved playbook requires non-empty approved_by and approved_date",
                        severity="error",
                    )
                )
            else:
                from datetime import date as _date

                if not re.fullmatch(r"^\d{4}-\d{2}-\d{2}$", dt):
                    issues.append(
                        PlaybookValidationIssue(
                            code="invalid_approval_status",
                            field_or_path="approved_date",
                            message=f"approved_date must be YYYY-MM-DD, got {dt!r}",
                            severity="error",
                        )
                    )
                else:
                    try:
                        _date.fromisoformat(dt)
                    except ValueError:
                        issues.append(
                            PlaybookValidationIssue(
                                code="invalid_approval_status",
                                field_or_path="approved_date",
                                message=f"approved_date is not a valid calendar date: {dt!r}",
                                severity="error",
                            )
                        )

    raw_domain = raw_meta.get("domain")
    if raw_domain is not None and raw_domain not in ALLOWED_DOMAINS:
        issues.append(
            PlaybookValidationIssue(
                code="invalid_domain",
                field_or_path="domain",
                message=f"Invalid domain {raw_domain!r}; allowed: {ALLOWED_DOMAINS}",
                severity="error",
            )
        )

    raw_human = raw_meta.get("requires_human_review")
    if raw_human is not True:
        issues.append(
            PlaybookValidationIssue(
                code="policy_violation",
                field_or_path="requires_human_review",
                message="Scientific Playbooks must set requires_human_review: true",
                severity="security",
            )
        )

    runtime_avail = raw_meta.get("runtime_available", True)
    if not isinstance(runtime_avail, bool):
        issues.append(
            PlaybookValidationIssue(
                code="invalid_field",
                field_or_path="runtime_available",
                message="runtime_available must be a boolean",
                severity="error",
            )
        )
        runtime_avail = True
    elif runtime_avail and (raw_approval == "rejected" or raw_status == "deprecated"):
        issues.append(
            PlaybookValidationIssue(
                code="invalid_approval_status",
                field_or_path="runtime_available",
                message="rejected or deprecated playbooks must set runtime_available: false",
                severity="error",
            )
        )

    raw_tools = raw_meta.get("allowed_tools")
    if not isinstance(raw_tools, list):
        issues.append(
            PlaybookValidationIssue(
                code="invalid_allowed_tools",
                field_or_path="allowed_tools",
                message="allowed_tools must be a list of tool names",
                severity="error",
            )
        )
    else:
        if runtime_avail and len(raw_tools) == 0:
            issues.append(
                PlaybookValidationIssue(
                    code="empty_allowed_tools",
                    field_or_path="allowed_tools",
                    message="Runtime-available playbooks must declare at least one tool in allowed_tools",
                    severity="security",
                )
            )
        if len(set(map(str, raw_tools))) != len(raw_tools):
            issues.append(
                PlaybookValidationIssue(
                    code="duplicate_allowed_tool",
                    field_or_path="allowed_tools",
                    message="allowed_tools contains duplicate tool entries",
                    severity="error",
                )
            )

    raw_name = raw_meta.get("name")
    if isinstance(raw_name, str) and pb_dir_name and raw_name != pb_dir_name:
        issues.append(
            PlaybookValidationIssue(
                code="directory_name_mismatch",
                field_or_path="name",
                message=f"Frontmatter name {raw_name!r} does not match directory name {pb_dir_name!r}",
                severity="error",
            )
        )

    return issues


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
    if target.name == "SKILL.md":
        pb_dir = target.parent
        skill_file = target
    else:
        pb_dir = target
        skill_file = target / "SKILL.md"

    issues: list[PlaybookValidationIssue] = []

    try:
        dir_lstat = pb_dir.lstat()
    except OSError as exc:
        return PlaybookValidationResult(
            valid=False,
            issues=(
                PlaybookValidationIssue(
                    code="missing_playbook_dir",
                    field_or_path=str(pb_dir),
                    message=f"Playbook directory does not exist: {exc}",
                    severity="error",
                ),
            ),
        )

    if stat.S_ISLNK(dir_lstat.st_mode):
        return PlaybookValidationResult(
            valid=False,
            issues=(
                PlaybookValidationIssue(
                    code="symlink_escape",
                    field_or_path=str(pb_dir.name),
                    message="Playbook directory must not be a symlink",
                    severity="security",
                ),
            ),
        )

    try:
        real_pb_dir = pb_dir.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        return PlaybookValidationResult(
            valid=False,
            issues=(
                PlaybookValidationIssue(
                    code="symlink_escape",
                    field_or_path=str(pb_dir.name),
                    message=f"Cannot resolve playbook directory: {exc}",
                    severity="security",
                ),
            ),
        )

    if skills_root is not None:
        try:
            real_skills_root = Path(skills_root).resolve(strict=True)
            if not real_pb_dir.is_relative_to(real_skills_root):
                return PlaybookValidationResult(
                    valid=False,
                    issues=(
                        PlaybookValidationIssue(
                            code="symlink_escape",
                            field_or_path=str(pb_dir.name),
                            message="Playbook directory resolves outside skills_root",
                            severity="security",
                        ),
                    ),
                )
        except (OSError, RuntimeError) as exc:
            return PlaybookValidationResult(
                valid=False,
                issues=(
                    PlaybookValidationIssue(
                        code="invalid_skills_root",
                        field_or_path=str(skills_root),
                        message=f"Invalid skills_root: {exc}",
                        severity="error",
                    ),
                ),
            )

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
            issues=(
                PlaybookValidationIssue(
                    code="missing_skill_md",
                    field_or_path="SKILL.md",
                    message="Playbook directory is missing SKILL.md",
                    severity="error",
                ),
            ),
        )

    raw_skill_text, read_issues = _read_utf8_bounded(skill_file, MAX_SKILL_MD_BYTES, "SKILL.md")
    issues.extend(read_issues)

    resources_map, resource_issues = _inspect_playbook_resources(pb_dir, real_pb_dir)
    issues.extend(resource_issues)

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
            PlaybookValidationIssue(
                code="empty_instructions",
                field_or_path="SKILL.md",
                message="SKILL.md must contain non-empty markdown instructions after frontmatter",
                severity="error",
            )
        )

    issues.extend(_classify_frontmatter_issues(raw_meta, actual_dir_name))

    # Validate tool references against the active ToolRegistry
    declared_tools: list[str] = []
    if isinstance(raw_meta.get("allowed_tools"), list):
        for item in raw_meta["allowed_tools"]:
            if isinstance(item, str) and item.strip():
                declared_tools.append(item.strip())
    declared_set = set(declared_tools)

    for tool_name in declared_tools:
        if tool_name not in registered_tools:
            issues.append(
                PlaybookValidationIssue(
                    code="non_existent_tool",
                    field_or_path="allowed_tools",
                    message=f"Declared tool {tool_name!r} does not exist in the current ToolRegistry",
                    severity="security",
                )
            )

    # Scan frontmatter text fields, SKILL.md body, and confined resources for undeclared tools & external links
    texts_to_scan: list[tuple[str, str]] = [("SKILL.md", body_text)]
    for fm_key in (
        "description",
        "source",
        "license",
        "approved_by",
        "required_inputs",
        "produced_artifacts",
        "limitations",
        "compatibility",
    ):
        fm_val = raw_meta.get(fm_key)
        if isinstance(fm_val, str):
            texts_to_scan.append((f"frontmatter.{fm_key}", fm_val))
        elif isinstance(fm_val, list):
            joined = "\n".join(str(x) for x in fm_val if isinstance(x, str))
            if joined:
                texts_to_scan.append((f"frontmatter.{fm_key}", joined))
        elif isinstance(fm_val, Mapping):
            joined = "\n".join(
                str(v)
                for v in fm_val.values()
                if isinstance(v, (str, list))
            )
            if joined:
                texts_to_scan.append((f"frontmatter.{fm_key}", joined))

    for rel_path, (res_text, _sha, _sz) in sorted(resources_map.items()):
        texts_to_scan.append((rel_path, res_text))

    for source_label, text_content in texts_to_scan:
        cleaned_for_tools = _ZERO_WIDTH_RE.sub("", text_content).replace("\ufeff", "")
        for reg_tool in sorted(registered_tools):
            if reg_tool in declared_set:
                continue
            if re.search(rf"\b{re.escape(reg_tool)}\b", cleaned_for_tools):
                issues.append(
                    PlaybookValidationIssue(
                        code="undeclared_tool",
                        field_or_path=source_label,
                        message=(
                            f"Playbook content in {source_label} references registered tool "
                            f"{reg_tool!r} which is not declared in allowed_tools"
                        ),
                        severity="security",
                    )
                )
        issues.extend(_scan_markdown_links(text_content, source_label, set(resources_map)))

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
            playbook=None,
        )

    resource_paths = tuple(sorted(resources_map))
    resource_digests = tuple((rel, resources_map[rel][1]) for rel in resource_paths)

    try:
        digest = compute_canonical_digest(
            raw_meta,
            instructions=body_text,
            resource_digests=resource_digests,
        )
        playbook = ScientificPlaybook(
            **raw_meta,
            canonical_digest=digest,
            instructions=body_text,
            resource_paths=resource_paths,
            resource_digests=resource_digests,
        )
    except (ValidationError, ValueError, TypeError) as exc:
        pydantic_issues = [
            PlaybookValidationIssue(
                code="schema_validation_error",
                field_or_path="frontmatter",
                message=str(exc),
                severity="error",
            )
        ]
        return PlaybookValidationResult(
            valid=False,
            playbook_id=pb_id,
            name=pb_name,
            version=pb_ver,
            status=pb_status,
            approval_status=pb_approval,
            runtime_available=pb_runtime,
            issues=tuple(pydantic_issues),
            playbook=None,
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


def validate_all_playbooks(
    skills_root: Path | str | None = None,
    *,
    registry: ToolRegistry | None = None,
) -> CatalogValidationReport:
    """Validate all playbooks under `skills_root` and check cross-playbook uniqueness."""
    root = Path(skills_root) if skills_root is not None else DEFAULT_SKILLS_ROOT
    reg = registry if registry is not None else default_registry()

    catalog_issues: list[PlaybookValidationIssue] = []
    if not root.exists():
        catalog_issues.append(
            PlaybookValidationIssue(
                code="missing_skills_root",
                field_or_path=str(root),
                message=f"Skills catalog directory does not exist: {root}",
                severity="error",
            )
        )
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
            catalog_issues=tuple(catalog_issues),
        )

    try:
        real_root = root.resolve(strict=True)
        entries = sorted(root.iterdir(), key=lambda p: p.name)
    except OSError as exc:
        catalog_issues.append(
            PlaybookValidationIssue(
                code="unreadable_skills_root",
                field_or_path=str(root),
                message=f"Cannot read skills root: {exc}",
                severity="error",
            )
        )
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
            catalog_issues=tuple(catalog_issues),
        )

    results: list[PlaybookValidationResult] = []
    for entry in entries:
        try:
            lst = entry.lstat()
        except OSError as exc:
            catalog_issues.append(
                PlaybookValidationIssue(
                    code="unreadable_file",
                    field_or_path=entry.name,
                    message=str(exc),
                    severity="error",
                )
            )
            continue
        if stat.S_ISLNK(lst.st_mode):
            catalog_issues.append(
                PlaybookValidationIssue(
                    code="symlink_escape",
                    field_or_path=entry.name,
                    message=f"Top-level symlink {entry.name!r} is forbidden in skills_root",
                    severity="security",
                )
            )
            continue
        if entry.name.startswith("."):
            catalog_issues.append(
                PlaybookValidationIssue(
                    code="hidden_file",
                    field_or_path=entry.name,
                    message=f"Hidden entry {entry.name!r} is forbidden in skills_root",
                    severity="security",
                )
            )
            continue
        if not stat.S_ISDIR(lst.st_mode):
            catalog_issues.append(
                PlaybookValidationIssue(
                    code="unexpected_root_file",
                    field_or_path=entry.name,
                    message=f"Unexpected non-directory entry {entry.name!r} in skills_root",
                    severity="security",
                )
            )
            continue
        results.append(validate_playbook(entry, skills_root=real_root, registry=reg))

    # Cross-playbook uniqueness checks (playbook_id, name, conflicting versions)
    seen_ids: dict[str, str] = {}
    seen_names: dict[str, str] = {}
    seen_versions: dict[str, set[str]] = {}
    duplicate_identities: set[str] = set()

    for res in results:
        if res.playbook_id:
            seen_versions.setdefault(res.playbook_id, set()).add(res.version or "")
            if res.playbook_id in seen_ids:
                duplicate_identities.add(res.playbook_id)
                catalog_issues.append(
                    PlaybookValidationIssue(
                        code="duplicate_playbook_id",
                        field_or_path=res.playbook_id,
                        message=(
                            f"Duplicate playbook_id {res.playbook_id!r} in {res.name!r} "
                            f"and {seen_ids[res.playbook_id]!r}"
                        ),
                        severity="security",
                    )
                )
                if len(seen_versions[res.playbook_id]) > 1:
                    catalog_issues.append(
                        PlaybookValidationIssue(
                            code="conflicting_version",
                            field_or_path=res.playbook_id,
                            message=(
                                f"Conflicting versions {sorted(seen_versions[res.playbook_id])} "
                                f"for playbook_id {res.playbook_id!r}"
                            ),
                            severity="error",
                        )
                    )
            else:
                seen_ids[res.playbook_id] = res.name or ""

        if res.name:
            if res.name in seen_names:
                duplicate_identities.add(res.name)
                catalog_issues.append(
                    PlaybookValidationIssue(
                        code="duplicate_name",
                        field_or_path=res.name,
                        message=f"Duplicate playbook name {res.name!r}",
                        severity="security",
                    )
                )
            else:
                seen_names[res.name] = res.playbook_id or ""

    # Check cross-collision between playbook_id and another playbook's name
    for pid, owner_name in seen_ids.items():
        if pid in seen_names and seen_names[pid] != pid:
            duplicate_identities.add(pid)
            catalog_issues.append(
                PlaybookValidationIssue(
                    code="duplicate_playbook_id",
                    field_or_path=pid,
                    message=f"playbook_id {pid!r} (in {owner_name!r}) collides with another playbook's name",
                    severity="security",
                )
            )

    all_issues = [issue for r in results for issue in r.issues] + catalog_issues
    missing_tools = sorted({
        issue.message
        for issue in all_issues
        if issue.code in {"non_existent_tool", "undeclared_tool"}
    })
    security_violations = tuple(issue for issue in all_issues if issue.severity == "security")

    approved_count = sum(1 for r in results if r.valid and r.approval_status == "approved")
    pending_count = sum(1 for r in results if r.approval_status == "pending_owner_review")
    draft_count = sum(1 for r in results if r.status == "draft")
    reviewed_count = sum(1 for r in results if r.status == "reviewed")
    deprecated_count = sum(1 for r in results if r.status == "deprecated")
    runtime_count = sum(1 for r in results if r.valid and r.runtime_available is True)
    disabled_count = sum(1 for r in results if r.runtime_available is False)

    valid = bool(results) and all(r.valid for r in results) and not catalog_issues
    return CatalogValidationReport(
        valid=valid,
        skills_root=str(root),
        total_playbooks=len(results),
        approved_count=approved_count,
        pending_approval_count=pending_count,
        draft_count=draft_count,
        reviewed_count=reviewed_count,
        deprecated_count=deprecated_count,
        runtime_available_count=runtime_count,
        disabled_count=disabled_count,
        results=tuple(results),
        catalog_issues=tuple(catalog_issues),
        missing_tool_references=tuple(missing_tools),
        duplicate_identities=tuple(sorted(duplicate_identities)),
        security_violations=security_violations,
    )


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


def _validate_identifier_arg(identifier: str, label: str = "playbook_id") -> str:
    if not isinstance(identifier, str) or not identifier.strip():
        raise PlaybookSecurityError(f"{label} must be a non-empty string")
    if "\x00" in identifier or "/" in identifier or "\\" in identifier or ".." in identifier:
        raise PlaybookSecurityError(f"Invalid characters or path traversal in {label}: {identifier!r}")
    return identifier.strip()


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
    if not isinstance(resource, str) or not resource.strip():
        raise PlaybookSecurityError("Resource path must be a non-empty relative path")
    raw_res = resource.strip()
    decoded_res = unquote(raw_res)
    if "\x00" in raw_res or "\x00" in decoded_res or "\\" in raw_res or "\\" in decoded_res:
        raise PlaybookSecurityError(f"Forbidden characters in resource path: {resource!r}")
    lower = decoded_res.lower()
    if _URI_SCHEME_RE.match(lower) and not re.match(r"^[A-Za-z]:[/\\]", decoded_res):
        raise PlaybookSecurityError(f"Remote or URI resource retrieval is forbidden: {resource!r}")
    if re.match(r"^[A-Za-z]:", decoded_res) or decoded_res.startswith("/"):
        raise PlaybookSecurityError(f"Absolute resource paths are forbidden: {resource!r}")

    raw_segments = decoded_res.split("/")
    if ".." in raw_segments or "." in raw_segments or "" in raw_segments:
        raise PlaybookSecurityError(f"Path traversal or dot-segment in resource path is forbidden: {resource!r}")

    posix = PurePosixPath(decoded_res)
    if posix.is_absolute():
        raise PlaybookSecurityError(f"Absolute resource paths are forbidden: {resource!r}")
    if any(part.startswith(".") for part in posix.parts):
        raise PlaybookSecurityError(f"Hidden resource paths are forbidden: {resource!r}")

    norm_rel = posix.as_posix()
    if norm_rel.lower() == "skill.md":
        raise PlaybookSecurityError("Use load_playbook() to read SKILL.md, not read_playbook_resource()")

    if any(s.lower() in FORBIDDEN_SCRIPT_EXTENSIONS for s in posix.suffixes) or (
        posix.suffix.lower() not in ALLOWED_RESOURCE_EXTENSIONS
    ):
        raise PlaybookSecurityError(
            f"Resource {norm_rel!r} has disallowed or executable extension {posix.suffix.lower()!r}"
        )

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

    candidate = pb_dir.joinpath(*posix.parts)
    # Check every ancestor component for symlinks
    cur = pb_dir
    for part in posix.parts:
        cur = cur / part
        if cur.is_symlink():
            raise PlaybookSecurityError(f"Symlink escape detected in resource path: {norm_rel!r}")
    try:
        real_candidate = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise PlaybookLookupError(f"Resource {norm_rel!r} does not exist in playbook {pb.playbook_id!r}") from exc

    if not real_candidate.is_relative_to(pb_dir):
        raise PlaybookSecurityError(f"Resource {norm_rel!r} resolves outside playbook root")

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
