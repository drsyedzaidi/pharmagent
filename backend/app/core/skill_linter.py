"""Skill Integrity Linter — deterministic validation of authored SKILL.md folders.

Modeled on ``AI-assisted-MIDD``'s ``.posit/assistant/skills/manage-skills/scripts/check_skills.R``,
adapted for PharmAgent's tool registry (``default_registry``) and shared state
contract (``PharmState.model_fields``).

Checks per skill folder (``<skills_dir>/<slug>/SKILL.md``):
  1. ``frontmatter_present``: starts with ``---`` YAML frontmatter block and parses as a dict.
  2. ``name_matches_folder``: frontmatter ``name`` equals ``<slug>``.
  3. ``name_format``: lowercase alphanumeric and hyphens (1–64 chars), no ``anthropic``/``claude``.
  4. ``description_present``: non-empty string description, <= 1024 chars, no raw ``<`` or ``>`` XML tags.
  5. ``skill_under_500_lines``: ``SKILL.md`` has <= 500 lines (warn if exceeded).
  6. ``allowed_tools_registered``: every tool in optional ``allowed_tools`` exists in ``default_registry()``.
  7. ``produced_artifacts_valid``: every field in optional ``produced_artifacts`` exists in ``PharmState.model_fields``.
  8. ``own_paths_exist``: relative markdown links ``[...](path)`` resolve inside the skill folder without escaping it.
  9. ``references_linked``: every ``references/*.md`` file is linked from ``SKILL.md`` (warn if orphaned).
  10. ``skills_one_folder_deep`` & ``names_unique`` across the entire ``skills_dir``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from app.core.pharmstate import PharmState
from app.tools.builtins import default_registry

_NAME_RE = re.compile(r"^[a-z0-9-]{1,64}$")
_FORBIDDEN_NAME_RE = re.compile(r"anthropic|claude")
_XML_TAG_RE = re.compile(r"[<>]")
_MD_LINK_RE = re.compile(r"\]\((?!https?:|#|mailto:)([^)#]+)\)")
_SKILL_DIR_URI_RE = re.compile(r"file:///\{skill_dir\}/([^) `'\"]+)")

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


def _resolve_within_skill(skill_dir: Path, rel_path: str) -> tuple[bool, str]:
    """Verify ``rel_path`` stays within ``skill_dir`` and exists on disk."""
    clean = rel_path.strip()
    if not clean:
        return True, ""
    root = skill_dir.resolve()
    try:
        target = (skill_dir / clean).resolve()
    except OSError:
        return False, f"{clean} (invalid path)"
    if root != target and root not in target.parents:
        return False, f"{clean} (escapes skill directory)"
    if not target.exists():
        return False, clean
    return True, ""


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

    known_tools = registered_tools if registered_tools is not None else set(default_registry().names())
    known_fields = valid_state_fields if valid_state_fields is not None else set(PharmState.model_fields.keys())

    skill_files = sorted(base_dir.glob("*/SKILL.md"))
    all_skill_files = sorted(base_dir.rglob("SKILL.md"))
    shallow_set = {p.resolve() for p in skill_files}
    deep_skills = [
        str(p.relative_to(base_dir))
        for p in all_skill_files
        if p.resolve() not in shallow_set
    ]

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

    for skill_md in skill_files:
        skill_dir = skill_md.parent
        slug = skill_dir.name
        text = skill_md.read_text(encoding="utf-8")
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
        linked_paths = list(_SKILL_DIR_URI_RE.findall(body)) + list(_MD_LINK_RE.findall(body))
        missing_or_escaped: list[str] = []
        for rel in dict.fromkeys(linked_paths):
            ok_path, err_msg = _resolve_within_skill(skill_dir, rel)
            if not ok_path:
                missing_or_escaped.append(err_msg)
        add(
            slug,
            "own_paths_exist",
            not missing_or_escaped,
            ", ".join(missing_or_escaped),
        )

        # 4. Reference files linked from SKILL.md (one level deep)
        refs_dir = skill_dir / "references"
        if refs_dir.exists() and refs_dir.is_dir():
            ref_files = sorted(refs_dir.glob("*.md"))
            orphan_refs = [r.name for r in ref_files if r.name not in body]
            add(
                slug,
                "references_linked",
                not orphan_refs,
                ", ".join(orphan_refs),
                level="warn",
            )

    # 5. Cross-skill checks
    add(
        "(all)",
        "skills_one_folder_deep",
        not deep_skills,
        f"not discovered: {', '.join(deep_skills)}",
    )
    seen: set[str] = set()
    dups: list[str] = []
    for n in names_seen:
        if n in seen and n not in dups:
            dups.append(n)
        seen.add(n)
    add("(all)", "names_unique", not dups, ", ".join(dups))

    return findings
