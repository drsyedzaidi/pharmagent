"""Filesystem confinement, path normalization, O_NOFOLLOW symlink checks, and resource inspection."""

from __future__ import annotations

import os
import re
import stat
from pathlib import Path, PurePosixPath
from typing import Literal
from urllib.parse import unquote

from .digest import _sha256_text
from .schema import (
    ALLOWED_RESOURCE_EXTENSIONS,
    FORBIDDEN_SCRIPT_EXTENSIONS,
    MAX_RESOURCE_BYTES,
    PlaybookLookupError,
    PlaybookSecurityError,
    PlaybookValidationIssue,
)
from .yaml_loader import _ZERO_WIDTH_RE

DEFAULT_SKILLS_ROOT = Path(__file__).resolve().parents[3] / "skills"
_URI_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")
_FORBIDDEN_RESOURCE_DIRS = frozenset({"scripts", "bin", "__pycache__", "hooks", "tools", "libexec"})


def _fs_issue(
    code: str,
    field_or_path: str,
    message: str,
    severity: Literal["error", "security"] = "security",
) -> PlaybookValidationIssue:
    return PlaybookValidationIssue(code=code, field_or_path=field_or_path, message=message, severity=severity)


def _read_utf8_bounded(
    path: Path, max_bytes: int, rel_label: str
) -> tuple[str | None, list[PlaybookValidationIssue]]:
    """Safely read a regular non-symlink UTF-8 file using O_NOFOLLOW and fstat size bounds."""
    try:
        st = path.lstat()
    except OSError as exc:
        return None, [_fs_issue("unreadable_file", rel_label, f"Cannot stat file: {exc}", "error")]
    if stat.S_ISLNK(st.st_mode):
        return None, [_fs_issue("symlink_escape", rel_label, "Symlinks are forbidden inside playbook directories")]
    if not stat.S_ISREG(st.st_mode):
        return None, [_fs_issue("invalid_file_type", rel_label, "Playbook entries must be regular files")]
    if st.st_size > max_bytes:
        return None, [
            _fs_issue("oversized_file", rel_label, f"File size {st.st_size} bytes exceeds limit of {max_bytes} bytes")
        ]

    open_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(path, open_flags)
        try:
            fst = os.fstat(fd)
            if not stat.S_ISREG(fst.st_mode):
                return None, [
                    _fs_issue("invalid_file_type", rel_label, "Opened file descriptor is not a regular file")
                ]
            if fst.st_size > max_bytes:
                return None, [
                    _fs_issue(
                        "oversized_file",
                        rel_label,
                        f"File size {fst.st_size} bytes exceeds limit of {max_bytes} bytes",
                    )
                ]
            with os.fdopen(fd, "rb") as fh:
                fd = -1
                raw = fh.read(max_bytes + 1)
        finally:
            if fd >= 0:
                os.close(fd)
    except OSError as exc:
        return None, [_fs_issue("unreadable_file", rel_label, f"Cannot read file: {exc}", "error")]

    if len(raw) > max_bytes:
        return None, [_fs_issue("oversized_file", rel_label, f"File exceeds limit of {max_bytes} bytes")]
    if b"\x00" in raw:
        return None, [_fs_issue("invalid_encoding", rel_label, "Null bytes are forbidden in playbook files")]
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        return None, [_fs_issue("invalid_encoding", rel_label, f"File is not valid UTF-8: {exc}")]
    if _ZERO_WIDTH_RE.search(text):
        return None, [
            _fs_issue(
                "invalid_encoding",
                rel_label,
                "Zero-width or bidirectional control characters are forbidden in playbook files",
            )
        ]
    return text, []


def _inspect_single_resource_entry(
    entry: Path,
    rel_posix: str,
    real_pb_dir: Path,
    stack: list[Path],
    resources: dict[str, tuple[str, str, int]],
) -> list[PlaybookValidationIssue]:
    try:
        lst = entry.lstat()
    except OSError as exc:
        return [_fs_issue("unreadable_file", rel_posix, f"Cannot stat entry: {exc}", "error")]

    if stat.S_ISLNK(lst.st_mode):
        return [
            _fs_issue("symlink_escape", rel_posix, f"Symlink {rel_posix!r} is forbidden inside playbook directory")
        ]

    try:
        resolved_entry = entry.resolve(strict=True)
    except (OSError, RuntimeError):
        return [_fs_issue("symlink_escape", rel_posix, f"Entry {rel_posix!r} could not be safely resolved")]

    if not resolved_entry.is_relative_to(real_pb_dir):
        return [_fs_issue("symlink_escape", rel_posix, f"Entry {rel_posix!r} resolves outside playbook root")]

    if entry.name.startswith("."):
        issues = [
            _fs_issue("hidden_file", rel_posix, f"Hidden file or directory {rel_posix!r} is forbidden in playbooks")
        ]
        if stat.S_ISREG(lst.st_mode) and (
            (lst.st_mode & 0o111) or any(s.lower() in FORBIDDEN_SCRIPT_EXTENSIONS for s in entry.suffixes)
        ):
            issues.append(
                _fs_issue(
                    "executable_or_script_file", rel_posix, f"Hidden executable or script {rel_posix!r} is forbidden"
                )
            )
        return issues

    if stat.S_ISDIR(lst.st_mode):
        dir_issues: list[PlaybookValidationIssue] = []
        if entry.name.lower() in _FORBIDDEN_RESOURCE_DIRS:
            dir_issues.append(
                _fs_issue(
                    "executable_or_script_file",
                    rel_posix,
                    f"Bundled script directory {rel_posix!r} is forbidden in the MVP",
                )
            )
        elif rel_posix != "references" and not rel_posix.startswith("references/"):
            dir_issues.append(
                _fs_issue(
                    "disallowed_resource_type",
                    rel_posix,
                    f"Only `references/` subdirectory is permitted inside a playbook, got {rel_posix!r}",
                )
            )
        stack.append(entry)
        return dir_issues

    if rel_posix.lower() == "skill.md":
        sk_issues: list[PlaybookValidationIssue] = []
        if rel_posix != "SKILL.md":
            sk_issues.append(
                _fs_issue(
                    "missing_skill_md",
                    rel_posix,
                    f"Playbook file must be named uppercase 'SKILL.md', got {rel_posix!r}",
                    "error",
                )
            )
        if lst.st_mode & 0o111:
            sk_issues.append(
                _fs_issue(
                    "executable_or_script_file", rel_posix, "SKILL.md must not have executable permission bits set"
                )
            )
        return sk_issues

    suffix = entry.suffix.lower()
    if any(s.lower() in FORBIDDEN_SCRIPT_EXTENSIONS for s in entry.suffixes) or (lst.st_mode & 0o111):
        return [
            _fs_issue(
                "executable_or_script_file",
                rel_posix,
                f"Executable or script file {rel_posix!r} is forbidden in the MVP",
            )
        ]

    if suffix not in ALLOWED_RESOURCE_EXTENSIONS:
        return [
            _fs_issue(
                "disallowed_resource_type",
                rel_posix,
                f"Resource {rel_posix!r} has disallowed extension {suffix!r}; allowed: {ALLOWED_RESOURCE_EXTENSIONS}",
            )
        ]

    text, read_issues = _read_utf8_bounded(entry, MAX_RESOURCE_BYTES, rel_posix)
    if read_issues:
        return read_issues
    assert text is not None
    if text.lstrip("\ufeff \t\r\n").startswith("#!"):
        return [
            _fs_issue(
                "executable_or_script_file",
                rel_posix,
                f"Resource {rel_posix!r} begins with a shebang (`#!`) and is rejected as a script",
            )
        ]

    resources[rel_posix] = (text, _sha256_text(text), len(text.encode("utf-8")))
    return []


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
            issues.append(_fs_issue("unreadable_file", str(current.name), f"Cannot list directory: {exc}", "error"))
            continue
        for entry in entries:
            rel_posix = entry.relative_to(pb_dir).as_posix()
            issues.extend(_inspect_single_resource_entry(entry, rel_posix, real_pb_dir, stack, resources))

    return resources, issues


def _validate_identifier_arg(identifier: str, label: str = "playbook_id") -> str:
    """Validate playbook_id or name lookup argument against traversal or null bytes."""
    if not isinstance(identifier, str) or not identifier.strip():
        raise PlaybookSecurityError(f"{label} must be a non-empty string")
    if "\x00" in identifier or "/" in identifier or "\\" in identifier or ".." in identifier:
        raise PlaybookSecurityError(f"Invalid characters or path traversal in {label}: {identifier!r}")
    return identifier.strip()


def _validate_and_normalize_resource_arg(resource: str) -> tuple[PurePosixPath, str]:
    """Validate and normalize a relative resource path argument against traversal, URIs, and script types."""
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
    return posix, norm_rel


def _resolve_confined_resource_path(
    pb_dir: Path,
    posix: PurePosixPath,
    norm_rel: str,
    playbook_id: str,
) -> Path:
    """Resolve a resource path inside `pb_dir`, rejecting any symlink or escape along the chain."""
    candidate = pb_dir.joinpath(*posix.parts)
    cur = pb_dir
    for part in posix.parts:
        cur = cur / part
        if cur.is_symlink():
            raise PlaybookSecurityError(f"Symlink escape detected in resource path: {norm_rel!r}")
    try:
        real_candidate = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise PlaybookLookupError(f"Resource {norm_rel!r} does not exist in playbook {playbook_id!r}") from exc

    if not real_candidate.is_relative_to(pb_dir):
        raise PlaybookSecurityError(f"Resource {norm_rel!r} resolves outside playbook root")
    return real_candidate
