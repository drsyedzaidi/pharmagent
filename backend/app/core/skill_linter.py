"""Skill Integrity Linter — deterministic validation of authored SKILL.md folders.

Modeled on ``AI-assisted-MIDD``'s ``.posit/assistant/skills/manage-skills/scripts/check_skills.R``,
adapted for PharmAgent's tool registry (``default_registry``) and shared state
contract (``PharmState.model_fields``).

Security & Containment:
  * Symlinks are strictly forbidden for ``skills_dir``, skill subdirectories,
    ``SKILL.md``, and any referenced files/directories.
  * Containment and ``lstat`` regular-file + byte-size bounds are verified
    BEFORE reading any file (using ``os.O_NOFOLLOW`` where supported), preventing
    arbitrary file reads or FIFO/large-file denial-of-service from a
    repository-controlled ``SKILL.md``.
"""
from __future__ import annotations

import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from app.core.pharmstate import PharmState
from app.tools.builtins import default_registry

_NAME_RE = re.compile(r"^[a-z0-9-]{1,64}$")
_FORBIDDEN_NAME_RE = re.compile(r"anthropic|claude")
_XML_TAG_RE = re.compile(r"[<>]")
_MD_LINK_RE = re.compile(r"\]\((?!https?:|#|mailto:|file://)([^)#]+)\)")
_SKILL_DIR_URI_RE = re.compile(r"file:///\{skill_dir\}/([^)# `'\"]+)")
_OTHER_FILE_URI_RE = re.compile(r"file://(?!/\{skill_dir\}/|\{skill_dir\}/)([^)# `'\"]+)")

_MAX_SKILL_BYTES = 256 * 1024  # 256 KiB hard ceiling before reading SKILL.md

DEFAULT_SKILLS_DIR = Path(__file__).resolve().parents[3] / "skills"


@dataclass(frozen=True)
class SkillFinding:
    skill: str
    check: str
    status: str  # "pass" | "warn" | "fail"
    detail: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "skill": self.skill,
            "check": self.check,
            "status": self.status,
            "detail": self.detail,
        }


def _parse_frontmatter(lines: list[str]) -> tuple[dict[str, Any] | None, int | None, str]:
    """Return ``(frontmatter_dict, end_line_index, error_detail)``."""
    if not lines or lines[0].strip() != "---":
        return None, None, "SKILL.md must start with a --- block"
    end_idx: int | None = None
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            end_idx = idx
            break
    if end_idx is None:
        return None, None, "unclosed --- frontmatter block"
    raw_yaml = "\n".join(lines[1:end_idx])
    try:
        parsed = yaml.safe_load(raw_yaml)
    except yaml.YAMLError as exc:
        return None, end_idx, f"invalid YAML frontmatter: {exc}"
    if not isinstance(parsed, dict):
        return None, end_idx, "frontmatter must parse to a YAML mapping"
    return parsed, end_idx, ""


def _read_contained_skill_md(
    base_root: Path,
    skill_dir: Path,
    skill_md: Path,
) -> tuple[str | None, str]:
    """Validate containment, symlink-freedom, regular-file type, and size BEFORE reading."""
    try:
        dir_st = os.lstat(skill_dir)
    except OSError as exc:
        return None, f"cannot stat skill directory: {exc}"
    if stat.S_ISLNK(dir_st.st_mode):
        return None, "skill directory must not be a symlink"
    if not stat.S_ISDIR(dir_st.st_mode):
        return None, "skill path is not a regular directory"

    resolved_dir = skill_dir.resolve()
    if base_root not in resolved_dir.parents:
        return None, "skill directory escapes skills_dir"

    try:
        md_st = os.lstat(skill_md)
    except OSError as exc:
        return None, f"cannot stat SKILL.md: {exc}"
    if stat.S_ISLNK(md_st.st_mode):
        return None, "SKILL.md must not be a symlink"
    if not stat.S_ISREG(md_st.st_mode):
        return None, "SKILL.md must be a regular file"
    if md_st.st_size > _MAX_SKILL_BYTES:
        return None, f"SKILL.md exceeds {_MAX_SKILL_BYTES} byte limit ({md_st.st_size} bytes)"

    resolved_md = skill_md.resolve()
    if resolved_dir not in resolved_md.parents:
        return None, "SKILL.md escapes skill directory"

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(skill_md, flags)
    except OSError as exc:
        return None, f"cannot safely open SKILL.md: {exc}"
    try:
        f_st = os.fstat(fd)
        if not stat.S_ISREG(f_st.st_mode):
            return None, "SKILL.md is not a regular file"
        if f_st.st_size > _MAX_SKILL_BYTES:
            return None, f"SKILL.md exceeds {_MAX_SKILL_BYTES} byte limit"
        raw = os.read(fd, _MAX_SKILL_BYTES + 1)
    finally:
        os.close(fd)

    if len(raw) > _MAX_SKILL_BYTES:
        return None, f"SKILL.md exceeds {_MAX_SKILL_BYTES} byte limit"
    try:
        return raw.decode("utf-8"), ""
    except UnicodeDecodeError as exc:
        return None, f"SKILL.md is not valid UTF-8: {exc}"


def _resolve_within_skill(skill_dir: Path, rel_path: str) -> tuple[bool, str]:
    """Verify ``rel_path`` stays within ``skill_dir``, contains no symlinks, and exists."""
    clean = rel_path.strip()
    if not clean:
        return True, ""
    if clean.startswith("file:///"):
        prefix = "file:///{skill_dir}/"
        if clean.startswith(prefix):
            clean = clean[len(prefix) :].strip()
            if not clean:
                return True, ""
        else:
            return False, f"{clean} (escapes skill directory)"
    elif clean.startswith("file://"):
        return False, f"{clean} (escapes skill directory)"
    # Strip optional markdown link title, e.g. 'references/foo.md "Title"'
    clean = clean.split()[0]
    if Path(clean).is_absolute():
        return False, f"{clean} (escapes skill directory)"

    root = skill_dir.resolve()
    candidate = skill_dir / clean
    # Check every intermediate component from skill_dir to candidate for symlinks
    cur = skill_dir
    for part in Path(clean).parts:
        cur = cur / part
        try:
            st = os.lstat(cur)
        except OSError:
            break
        if stat.S_ISLNK(st.st_mode):
            return False, f"{clean} (symlink not permitted)"

    try:
        target = candidate.resolve()
    except OSError:
        return False, f"{clean} (invalid path)"
    if root != target and root not in target.parents:
        return False, f"{clean} (escapes skill directory)"
    try:
        st_final = os.lstat(candidate)
    except OSError:
        return False, clean
    if stat.S_ISLNK(st_final.st_mode):
        return False, f"{clean} (symlink not permitted)"
    if not (stat.S_ISREG(st_final.st_mode) or stat.S_ISDIR(st_final.st_mode)):
        return False, f"{clean} (not a regular file or directory)"
    return True, ""


def _extract_skill_links(body: str) -> list[str]:
    """Extract skill-local paths from ``file:///{skill_dir}/...`` and relative markdown links."""
    skill_uri_paths = list(_SKILL_DIR_URI_RE.findall(body))
    other_file_uris = [f"file://{m}" for m in _OTHER_FILE_URI_RE.findall(body)]
    md_paths = list(_MD_LINK_RE.findall(body))
    return skill_uri_paths + other_file_uris + md_paths


def check_skills(
    skills_dir: str | Path | None = None,
    *,
    registered_tools: set[str] | None = None,
    valid_state_fields: set[str] | None = None,
) -> list[SkillFinding]:
    """Validate all ``<skills_dir>/<slug>/SKILL.md`` skill packages."""
    base_dir = Path(skills_dir) if skills_dir is not None else DEFAULT_SKILLS_DIR
    if not base_dir.exists() or not base_dir.is_dir():
        raise FileNotFoundError(f"skills directory does not exist: {base_dir}")
    if base_dir.is_symlink():
        raise ValueError(f"skills directory must not be a symlink: {base_dir}")
    base_root = base_dir.resolve()

    known_tools = registered_tools if registered_tools is not None else set(default_registry().names())
    known_fields = valid_state_fields if valid_state_fields is not None else set(PharmState.model_fields.keys())

    findings: list[SkillFinding] = []
    names_seen: list[str] = []

    def add(skill: str, check: str, ok: bool, detail: str = "", level: str = "fail") -> None:
        findings.append(
            SkillFinding(
                skill=skill,
                check=check,
                status="pass" if ok else level,
                detail="" if ok else detail,
            )
        )

    # Discover top-level entries and nested SKILL.md files without following symlinks
    skill_candidates: list[tuple[Path, Path]] = []
    for entry in sorted(base_dir.iterdir()):
        try:
            st = os.lstat(entry)
        except OSError:
            continue
        if stat.S_ISLNK(st.st_mode):
            add(entry.name, "skill_containment", False, "skill directory must not be a symlink")
            continue
        if stat.S_ISDIR(st.st_mode):
            md_candidate = entry / "SKILL.md"
            try:
                os.lstat(md_candidate)
                skill_candidates.append((entry, md_candidate))
            except OSError:
                pass

    shallow_dirs = {d.resolve() for d, _ in skill_candidates if not d.is_symlink()}
    deep_skills: list[str] = []
    for dirpath, dirnames, filenames in os.walk(base_dir, followlinks=False):
        dp = Path(dirpath)
        # Prune any symlinked directories so os.walk never traverses them
        safe_dirs: list[str] = []
        for d in sorted(dirnames):
            sub = dp / d
            try:
                if stat.S_ISLNK(os.lstat(sub).st_mode):
                    continue
            except OSError:
                continue
            safe_dirs.append(d)
        dirnames[:] = safe_dirs
        if "SKILL.md" in filenames and dp != base_dir and dp.resolve() not in shallow_dirs:
            deep_skills.append(str((dp / "SKILL.md").relative_to(base_dir)))

    for skill_dir, skill_md in skill_candidates:
        slug = skill_dir.name
        text, contain_err = _read_contained_skill_md(base_root, skill_dir, skill_md)
        add(slug, "skill_containment", text is not None, contain_err)
        if text is None:
            continue
        lines = text.splitlines()

        # 1. Frontmatter
        fm, fm_end, fm_err = _parse_frontmatter(lines)
        add(slug, "frontmatter_present", fm is not None, fm_err)
        if fm is None or fm_end is None:
            continue

        name = fm.get("name")
        name_str = str(name).strip() if isinstance(name, str) else ""
        if name_str:
            names_seen.append(name_str)

        add(
            slug,
            "name_matches_folder",
            name_str == slug,
            f"name {name_str!r} vs folder {slug!r}",
        )
        name_ok = bool(
            name_str
            and _NAME_RE.match(name_str)
            and not _FORBIDDEN_NAME_RE.search(name_str)
        )
        add(
            slug,
            "name_format",
            name_ok,
            "lowercase letters, digits and hyphens, <= 64 chars, no 'anthropic'/'claude'",
        )

        desc = fm.get("description")
        desc_str = str(desc).strip() if isinstance(desc, str) else ""
        add(
            slug,
            "description_present",
            bool(desc_str),
            "description: one line, what the skill does and when to use it",
        )
        if desc_str:
            add(
                slug,
                "description_length",
                len(desc_str) <= 1024,
                f"{len(desc_str)} chars (max 1024)",
            )
            add(
                slug,
                "description_no_xml",
                not bool(_XML_TAG_RE.search(desc_str)),
                "remove < and >",
            )

        # 2. PharmAgent contract: allowed_tools & produced_artifacts
        allowed_tools = fm.get("allowed_tools")
        if allowed_tools is not None:
            if not isinstance(allowed_tools, list) or not all(isinstance(t, str) for t in allowed_tools):
                add(slug, "allowed_tools_registered", False, "allowed_tools must be a list of tool names")
            else:
                unknown_tools = [t for t in allowed_tools if t not in known_tools]
                add(
                    slug,
                    "allowed_tools_registered",
                    not unknown_tools,
                    f"unknown tool(s): {', '.join(unknown_tools)}",
                )

        produced_artifacts = fm.get("produced_artifacts")
        if produced_artifacts is not None:
            if not isinstance(produced_artifacts, list) or not all(isinstance(a, str) for a in produced_artifacts):
                add(
                    slug,
                    "produced_artifacts_valid",
                    False,
                    "produced_artifacts must be a list of PharmState field names",
                )
            elif fm.get("runtime_available") is False:
                # A disabled specification (runtime_available: false) describes outputs
                # that do not exist yet, so its artifact names are not live PharmState fields.
                add(slug, "produced_artifacts_valid", True, "disabled specification: artifact names not checked")
            else:
                unknown_fields = [a for a in produced_artifacts if a not in known_fields]
                add(
                    slug,
                    "produced_artifacts_valid",
                    not unknown_fields,
                    f"unknown PharmState field(s): {', '.join(unknown_fields)}",
                )

        # 3. Body size & links
        add(
            slug,
            "skill_under_500_lines",
            len(lines) <= 500,
            f"{len(lines)} lines: move detail to references/",
            level="warn",
        )

        body = "\n".join(lines[fm_end + 1 :])
        linked_paths = _extract_skill_links(body)
        missing_or_escaped: list[str] = []
        for rel in dict.fromkeys(linked_paths):
            ok_path, err_msg = _resolve_within_skill(skill_dir, rel)
            if not ok_path:
                missing_or_escaped.append(err_msg)

        # 4. Reference files linked from SKILL.md (one level deep, no symlinks)
        refs_dir = skill_dir / "references"
        try:
            refs_st = os.lstat(refs_dir)
        except OSError:
            refs_st = None
        if refs_st is not None:
            if stat.S_ISLNK(refs_st.st_mode):
                missing_or_escaped.append("references (symlink not permitted)")
            elif stat.S_ISDIR(refs_st.st_mode):
                orphan_refs: list[str] = []
                for r in sorted(refs_dir.iterdir()):
                    if r.suffix != ".md":
                        continue
                    try:
                        r_st = os.lstat(r)
                    except OSError:
                        continue
                    if stat.S_ISLNK(r_st.st_mode):
                        missing_or_escaped.append(f"references/{r.name} (symlink not permitted)")
                        continue
                    if r.name not in body:
                        orphan_refs.append(r.name)
                add(
                    slug,
                    "references_linked",
                    not orphan_refs,
                    ", ".join(orphan_refs),
                    level="warn",
                )

        add(
            slug,
            "own_paths_exist",
            not missing_or_escaped,
            ", ".join(missing_or_escaped),
        )

    # 5. Cross-skill checks
    add(
        "(all)",
        "skills_one_folder_deep",
        not deep_skills,
        f"not discovered: {', '.join(sorted(deep_skills))}",
    )
    seen: set[str] = set()
    dups: list[str] = []
    for n in names_seen:
        if n in seen and n not in dups:
            dups.append(n)
        seen.add(n)
    add("(all)", "names_unique", not dups, ", ".join(dups))

    return findings
