"""Contract and security boundary tests for Scientific Playbooks (`app.playbooks`).

Verifies:
- Schema and frontmatter contracts (required fields, SemVer, status/approval rules, deep immutability)
- Deterministic canonical SHA-256 digest over metadata, instructions, and confined resources
- Progressive disclosure (`list_playbooks`, `load_playbook`, `read_playbook_resource`)
- Validation (`validate_playbook`, `validate_all_playbooks`) and CLI (`python -m app.playbooks --check`)
- Security boundaries (path traversal, symlink escape, duplicate YAML keys, unsafe YAML tags,
  oversized files, invalid UTF-8, hidden/executable/script files, undeclared/non-existent tools)
- Isolated, unregistered read-only tool adapter (`writes={}`)
- Non-interference with existing captured PharmAgent skills (`app.core.skills`)
"""
from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

import pytest

from app.core.pharmstate import PharmState
from app.core.skills import Skill, SkillStore, distill_steps
from app.playbooks import (
    MAX_ADAPTER_OUTPUT_BYTES,
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
    PlaybookValidationResult,
    PlaybookVersionError,
    ScientificPlaybook,
    check_tool_allowed,
    compute_canonical_digest,
    list_playbooks,
    load_playbook,
    playbook_provenance,
    read_playbook_resource,
    validate_all_playbooks,
    validate_playbook,
)
from app.playbooks.__main__ import main as cli_main
from app.playbooks.adapter import build_playbook_tools, run_playbook_adapter
from app.tools.base import ToolContext
from app.tools.builtins import default_registry

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent
SKILLS_ROOT = REPO_ROOT / "skills"


def _render_skill_md(
    *,
    name: str = "demo-protocol-playbook",
    description: str = "Demo protocol simulation playbook using reviewed tools.",
    playbook_id: str = "pharmagent.demo_protocol",
    version: str = "1.0.0",
    domain: str = "clinical_trial_simulation",
    status: str = "reviewed",
    approval_status: str = "pending_owner_review",
    approved_by: str | None = None,
    approved_date: str | None = None,
    license_id: str = "LicenseRef-PharmAgent-Internal",
    source: str = "PharmAgent authored; inspired by agentskills.io and Nova In Silico ACoP 2026 M-019",
    allowed_tools: list[str] | None = None,
    required_inputs: list[str] | None = None,
    produced_artifacts: list[str] | None = None,
    limitations: list[str] | None = None,
    requires_human_review: bool = True,
    compatibility: str = "PharmAgent >=0.1.0",
    runtime_available: bool = True,
    extra_frontmatter: str = "",
    body: str | None = None,
) -> str:
    tools = allowed_tools if allowed_tools is not None else ["load_dataset", "compute_nca", "simulate_pk_profile"]
    inputs = required_inputs if required_inputs is not None else ["dataset_id", "dose_regimen"]
    artifacts = produced_artifacts if produced_artifacts is not None else ["simulation_results", "nca_summary"]
    limits = limitations if limitations is not None else [
        "Does not execute PBPK or QSP models.",
        "Requires human pharmacometrician review before protocol recommendation.",
    ]
    lines = [
        "---",
        f"name: {name}",
        f"description: {description}",
        f"playbook_id: {playbook_id}",
        f"version: {version}",
        f"domain: {domain}",
        f"status: {status}",
        f"approval_status: {approval_status}",
    ]
    if approved_by is not None:
        lines.append(f"approved_by: {approved_by}")
    if approved_date is not None:
        lines.append(f"approved_date: {approved_date}")
    lines.extend([
        f"license: {license_id}",
        f"source: {source}",
        "allowed_tools:",
        *[f"  - {t}" for t in tools],
        "required_inputs:",
        *[f"  - {i}" for i in inputs],
        "produced_artifacts:",
        *[f"  - {a}" for a in artifacts],
        "limitations:",
        *[f"  - {lim}" for lim in limits],
        f"requires_human_review: {'true' if requires_human_review else 'false'}",
        f"compatibility: {compatibility}",
        f"runtime_available: {'true' if runtime_available else 'false'}",
    ])
    if extra_frontmatter:
        lines.append(extra_frontmatter.strip("\n"))
    lines.append("---")
    if body is None:
        body = (
            "# Demo Protocol Playbook\n\n"
            "Use `load_dataset`, `compute_nca`, and `simulate_pk_profile` in sequence.\n"
            "All numerical claims must originate from deterministic PharmAgent tools.\n"
        )
    return "\n".join(lines) + "\n\n" + body


def _write_playbook(
    root: Path,
    folder_name: str = "demo-protocol-playbook",
    *,
    skill_md: str | None = None,
    resources: dict[str, str | bytes] | None = None,
) -> Path:
    pb_dir = root / folder_name
    pb_dir.mkdir(parents=True, exist_ok=True)
    content = skill_md if skill_md is not None else _render_skill_md(name=folder_name)
    (pb_dir / "SKILL.md").write_text(content, encoding="utf-8")
    for rel, data in (resources or {}).items():
        target = pb_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, bytes):
            target.write_bytes(data)
        else:
            target.write_text(data, encoding="utf-8")
    return pb_dir


# ── 1. Schema, Frontmatter & Deep Immutability Contracts ─────────────────────


def test_valid_playbook_validates_and_returns_deeply_immutable_objects(tmp_path: Path):
    pb_dir = _write_playbook(
        tmp_path,
        resources={"references/checklist.md": "# Checklist\n\nVerify units with `load_dataset`.\n"},
    )
    res = validate_playbook(pb_dir, skills_root=tmp_path)
    assert isinstance(res, PlaybookValidationResult)
    assert res.valid is True
    assert res.issues == ()
    assert res.playbook is not None

    pb = res.playbook
    assert isinstance(pb, ScientificPlaybook)
    assert pb.playbook_id == "pharmagent.demo_protocol"
    assert pb.name == "demo-protocol-playbook"
    assert pb.version == "1.0.0"
    assert pb.status == "reviewed"
    assert pb.approval_status == "pending_owner_review"
    assert pb.allowed_tools == ("load_dataset", "compute_nca", "simulate_pk_profile")
    assert pb.required_inputs == ("dataset_id", "dose_regimen")
    assert pb.produced_artifacts == ("simulation_results", "nca_summary")
    assert pb.resource_paths == ("references/checklist.md",)
    assert len(pb.canonical_digest) == 64

    # Deep immutability: top-level attributes and nested collections cannot be mutated
    with pytest.raises((TypeError, ValueError)):
        pb.version = "2.0.0"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        pb.allowed_tools.append("run_nlme")  # type: ignore[attr-defined]
    meta = pb.to_metadata_mapping()
    assert isinstance(meta, MappingProxyType)
    with pytest.raises(TypeError):
        meta["version"] = "9.9.9"  # type: ignore[index]


@pytest.mark.parametrize(
    "override,expected_code",
    [
        ({"version": "1.0"}, "invalid_version"),
        ({"version": "v1.0.0"}, "invalid_version"),
        ({"version": "01.0.0"}, "invalid_version"),
        ({"status": "published"}, "invalid_status"),
        ({"approval_status": "auto_approved"}, "invalid_approval_status"),
        ({"status": "approved", "approval_status": "pending_owner_review"}, "invalid_approval_status"),
        ({"approval_status": "approved"}, "invalid_approval_status"),
        (
            {"approval_status": "pending_owner_review", "approved_by": "owner", "approved_date": "2026-10-22"},
            "invalid_approval_status",
        ),
        (
            {"approval_status": "approved", "approved_by": "owner", "approved_date": "2026-02-30"},
            "invalid_approval_status",
        ),
        ({"requires_human_review": False}, "policy_violation"),
        ({"allowed_tools": []}, "empty_allowed_tools"),
        ({"extra_frontmatter": "unknown_field: 123"}, "unknown_field"),
    ],
)
def test_invalid_frontmatter_fields_are_rejected(tmp_path: Path, override: dict, expected_code: str):
    md = _render_skill_md(**override)
    pb_dir = _write_playbook(tmp_path, skill_md=md)
    res = validate_playbook(pb_dir, skills_root=tmp_path)
    assert res.valid is False
    codes = {issue.code for issue in res.issues}
    assert expected_code in codes, f"expected {expected_code} in {codes}; issues={res.issues}"


def test_approved_playbook_requires_valid_approver_and_iso_date(tmp_path: Path):
    md = _render_skill_md(
        status="approved",
        approval_status="approved",
        approved_by="Syed Saoud Zaidi",
        approved_date="2026-10-22",
    )
    pb_dir = _write_playbook(tmp_path, skill_md=md)
    res = validate_playbook(pb_dir, skills_root=tmp_path)
    assert res.valid is True
    assert res.playbook is not None
    assert res.playbook.approval_status == "approved"
    assert res.playbook.approved_by == "Syed Saoud Zaidi"
    assert res.playbook.approved_date == "2026-10-22"


def test_missing_required_frontmatter_field_is_rejected(tmp_path: Path):
    md_no_domain = "\n".join(
        line for line in _render_skill_md().splitlines() if not line.startswith("domain:")
    ) + "\n"
    pb_dir = _write_playbook(tmp_path / "no_domain", skill_md=md_no_domain)
    res = validate_playbook(pb_dir, skills_root=tmp_path / "no_domain")
    assert res.valid is False
    assert any(i.code == "missing_required_field" and i.field_or_path == "domain" for i in res.issues)


def test_structured_compatibility_mapping_is_supported_and_immutable(tmp_path: Path):
    md = _render_skill_md(
        compatibility="\n  product: PharmAgent\n  min_version: 0.1.0\n  engines:\n    - pharmagent_focei",
    )
    pb_dir = _write_playbook(tmp_path, skill_md=md)
    res = validate_playbook(pb_dir, skills_root=tmp_path)
    assert res.valid is True
    assert res.playbook is not None
    assert isinstance(res.playbook.compatibility, PlaybookCompatibility)
    assert res.playbook.compatibility.product == "PharmAgent"
    assert res.playbook.compatibility.engines == ("pharmagent_focei",)
    assert compute_canonical_digest(res.playbook) == res.playbook.canonical_digest
    summary = res.playbook.to_summary()
    assert isinstance(summary.compatibility, PlaybookCompatibility)


# ── 2. Canonical Digest Determinism & 1-Byte Sensitivity ─────────────────────


def test_canonical_digest_changes_on_single_byte_change_in_metadata_body_or_resource(tmp_path: Path):
    pb_dir = _write_playbook(
        tmp_path,
        resources={"references/note.md": "Reference value tolerance: 0.01\n"},
    )
    base_pb = load_playbook("pharmagent.demo_protocol", skills_root=tmp_path)
    d0 = base_pb.canonical_digest

    # 1-byte change in instruction body
    (pb_dir / "SKILL.md").write_text(
        _render_skill_md(
            body="# Demo Protocol Playbook\n\nUse `load_dataset`, `compute_nca`, and `simulate_pk_profile` in sequence!\n"
        ),
        encoding="utf-8",
    )
    d_body = load_playbook("pharmagent.demo_protocol", skills_root=tmp_path).canonical_digest
    assert d_body != d0

    # 1-byte change in metadata description
    (pb_dir / "SKILL.md").write_text(
        _render_skill_md(
            description="Demo protocol simulation playbook using reviewed tools!",
            body="# Demo Protocol Playbook\n\nUse `load_dataset`, `compute_nca`, and `simulate_pk_profile` in sequence!\n",
        ),
        encoding="utf-8",
    )
    d_meta = load_playbook("pharmagent.demo_protocol", skills_root=tmp_path).canonical_digest
    assert d_meta not in {d0, d_body}

    # 1-byte change in confined resource file
    (pb_dir / "references" / "note.md").write_text("Reference value tolerance: 0.02\n", encoding="utf-8")
    d_res = load_playbook("pharmagent.demo_protocol", skills_root=tmp_path).canonical_digest
    assert d_res not in {d0, d_body, d_meta}


def test_canonical_digest_is_reproducible_and_newline_normalized(tmp_path: Path):
    lf_dir = tmp_path / "lf"
    crlf_dir = tmp_path / "crlf"
    reordered_dir = tmp_path / "reordered"
    md_lf = _render_skill_md()
    md_crlf = md_lf.replace("\n", "\r\n")
    # Permute frontmatter key order (reverse key blocks while preserving identical values)
    md_reordered = (
        "---\n"
        "runtime_available: true\n"
        "compatibility: PharmAgent >=0.1.0\n"
        "requires_human_review: true\n"
        "limitations:\n"
        "  - Does not execute PBPK or QSP models.\n"
        "  - Requires human pharmacometrician review before protocol recommendation.\n"
        "produced_artifacts:\n"
        "  - simulation_results\n"
        "  - nca_summary\n"
        "required_inputs:\n"
        "  - dataset_id\n"
        "  - dose_regimen\n"
        "allowed_tools:\n"
        "  - load_dataset\n"
        "  - compute_nca\n"
        "  - simulate_pk_profile\n"
        "source: PharmAgent authored; inspired by agentskills.io and Nova In Silico ACoP 2026 M-019\n"
        "license: LicenseRef-PharmAgent-Internal\n"
        "approval_status: pending_owner_review\n"
        "status: reviewed\n"
        "domain: clinical_trial_simulation\n"
        "version: 1.0.0\n"
        "playbook_id: pharmagent.demo_protocol\n"
        "description: Demo protocol simulation playbook using reviewed tools.\n"
        "name: demo-protocol-playbook\n"
        "---\n\n"
        "# Demo Protocol Playbook\n\n"
        "Use `load_dataset`, `compute_nca`, and `simulate_pk_profile` in sequence.\n"
        "All numerical claims must originate from deterministic PharmAgent tools.\n"
    )
    _write_playbook(lf_dir, skill_md=md_lf, resources={"references/a.md": "line1\nline2\n"})
    _write_playbook(crlf_dir, skill_md=md_crlf, resources={"references/a.md": "line1\r\nline2\r\n"})
    _write_playbook(reordered_dir, skill_md=md_reordered, resources={"references/a.md": "line1\rline2\r"})

    pb_lf = load_playbook("pharmagent.demo_protocol", skills_root=lf_dir)
    pb_crlf = load_playbook("pharmagent.demo_protocol", skills_root=crlf_dir)
    pb_reordered = load_playbook("pharmagent.demo_protocol", skills_root=reordered_dir)
    assert pb_lf.canonical_digest == pb_crlf.canonical_digest == pb_reordered.canonical_digest
    assert pb_lf.canonical_digest == compute_canonical_digest(pb_lf)


# ── 3. Progressive Disclosure & Provenance Contracts ─────────────────────────


def test_progressive_disclosure_separates_discovery_from_instructions_and_resources(tmp_path: Path):
    _write_playbook(
        tmp_path,
        folder_name="zeta-playbook",
        skill_md=_render_skill_md(
            name="zeta-playbook",
            playbook_id="pharmagent.zeta",
            body="# Zeta Secret Marker\n\nCall `load_dataset`, `compute_nca`, `simulate_pk_profile`.\n",
        ),
        resources={"references/deep.md": "DEEP_RESOURCE_MARKER via `load_dataset`.\n"},
    )
    _write_playbook(
        tmp_path,
        folder_name="alpha-playbook",
        skill_md=_render_skill_md(
            name="alpha-playbook",
            playbook_id="pharmagent.alpha",
            body="# Alpha Secret Marker\n\nCall `load_dataset`, `compute_nca`, `simulate_pk_profile`.\n",
        ),
    )

    # 1. list_playbooks returns deterministic sorted PlaybookSummary objects without instructions or resource content
    summaries = list_playbooks(skills_root=tmp_path)
    assert [s.playbook_id for s in summaries] == ["pharmagent.alpha", "pharmagent.zeta"]
    for s in summaries:
        assert isinstance(s, PlaybookSummary)
        assert not hasattr(s, "instructions")
        dumped = str(s.model_dump())
        assert "Secret Marker" not in dumped
        assert "DEEP_RESOURCE_MARKER" not in dumped
        assert s.playbook_id and s.version and s.canonical_digest
        assert s.approval_status == "pending_owner_review"
        assert s.source and s.license
        assert s.allowed_tools and s.required_inputs and s.limitations

    # 2. load_playbook loads full instructions for one playbook (without loading resource bodies)
    pb = load_playbook("pharmagent.zeta", version="1.0.0", skills_root=tmp_path)
    assert "Zeta Secret Marker" in pb.instructions
    assert "DEEP_RESOURCE_MARKER" not in pb.instructions
    assert pb.resource_paths == ("references/deep.md",)

    # Lookup by version mismatch fails closed
    with pytest.raises(PlaybookVersionError):
        load_playbook("pharmagent.zeta", version="9.9.9", skills_root=tmp_path)

    # Lookup of unknown playbook fails closed
    with pytest.raises(PlaybookLookupError):
        load_playbook("pharmagent.does_not_exist", skills_root=tmp_path)

    # 3. read_playbook_resource loads one confined resource with full identity/digest metadata
    res = read_playbook_resource("pharmagent.zeta", "references/deep.md", skills_root=tmp_path)
    assert isinstance(res, PlaybookResource)
    assert "DEEP_RESOURCE_MARKER" in res.content
    assert res.playbook_id == "pharmagent.zeta"
    assert res.version == "1.0.0"
    assert res.canonical_digest == pb.canonical_digest
    assert res.approval_status == "pending_owner_review"
    assert res.allowed_tools == pb.allowed_tools
    assert res.required_inputs == pb.required_inputs
    assert res.limitations == pb.limitations
    assert res.untrusted_data is True

    # Provenance hook returns immutable identity metadata
    prov = playbook_provenance("pharmagent.zeta", skills_root=tmp_path)
    assert isinstance(prov, PlaybookProvenance)
    assert prov.canonical_digest == pb.canonical_digest
    assert prov.playbook_id == "pharmagent.zeta"


def test_disabled_playbook_is_blocked_at_runtime_unless_explicitly_inspected(tmp_path: Path):
    _write_playbook(
        tmp_path,
        folder_name="draft-spec",
        skill_md=_render_skill_md(
            name="draft-spec",
            playbook_id="pharmagent.draft_spec",
            status="draft",
            runtime_available=False,
            allowed_tools=[],
            body="# Draft Spec\n\nSpecification only; no runtime execution.\n",
        ),
        resources={"references/spec.md": "# Spec Resource\n"},
    )
    with pytest.raises(PlaybookDisabledError):
        load_playbook("pharmagent.draft_spec", skills_root=tmp_path)
    with pytest.raises(PlaybookDisabledError):
        read_playbook_resource("pharmagent.draft_spec", "references/spec.md", skills_root=tmp_path)

    pb = load_playbook("pharmagent.draft_spec", skills_root=tmp_path, allow_disabled=True)
    assert pb.runtime_available is False
    assert pb.allowed_tools == ()


# ── 4. Security Boundaries ───────────────────────────────────────────────────


def test_rejects_duplicate_yaml_keys_and_unsafe_yaml_tags(tmp_path: Path):
    dup_md = _render_skill_md(extra_frontmatter="version: 2.0.0")
    pb_dup = _write_playbook(tmp_path / "dup", skill_md=dup_md)
    res_dup = validate_playbook(pb_dup, skills_root=tmp_path / "dup")
    assert res_dup.valid is False
    assert any(i.code == "duplicate_yaml_key" for i in res_dup.issues)

    unsafe_md = _render_skill_md(extra_frontmatter="exploit: !!python/object/apply:os.system ['echo pwned']")
    pb_unsafe = _write_playbook(tmp_path / "unsafe", skill_md=unsafe_md)
    res_unsafe = validate_playbook(pb_unsafe, skills_root=tmp_path / "unsafe")
    assert res_unsafe.valid is False
    assert any(i.code == "unsafe_yaml" for i in res_unsafe.issues)


@pytest.mark.parametrize(
    "bad_resource",
    [
        "../secret.md",
        "references/../../secret.md",
        "/etc/passwd",
        "\\windows\\system32\\drivers\\etc\\hosts",
        "references/.hidden.md",
        "SKILL.md",
    ],
)
def test_read_playbook_resource_rejects_traversal_absolute_and_hidden_paths(
    tmp_path: Path, bad_resource: str
):
    (tmp_path / "secret.md").write_text("TOP_SECRET", encoding="utf-8")
    _write_playbook(
        tmp_path,
        resources={"references/ok.md": "safe content\n"},
    )
    with pytest.raises(PlaybookSecurityError):
        read_playbook_resource("pharmagent.demo_protocol", bad_resource, skills_root=tmp_path)


def test_rejects_symlink_escape_in_playbook_and_resources(tmp_path: Path):
    outside = tmp_path / "outside.md"
    outside.write_text("OUTSIDE_SECRET", encoding="utf-8")
    pb_dir = _write_playbook(tmp_path / "catalog")
    (pb_dir / "references").mkdir(exist_ok=True)
    (pb_dir / "references" / "leak.md").symlink_to(outside)

    res = validate_playbook(pb_dir, skills_root=tmp_path / "catalog")
    assert res.valid is False
    assert any(i.code == "symlink_escape" for i in res.issues)

    with pytest.raises(PlaybookSecurityError):
        read_playbook_resource(
            "pharmagent.demo_protocol", "references/leak.md", skills_root=tmp_path / "catalog"
        )


def test_rejects_bundled_scripts_executables_and_hidden_files(tmp_path: Path):
    pb_script = _write_playbook(
        tmp_path / "script_case",
        resources={"scripts/run.py": "import os\nos.system('id')\n"},
    )
    res_script = validate_playbook(pb_script, skills_root=tmp_path / "script_case")
    assert res_script.valid is False
    assert any(i.code == "executable_or_script_file" for i in res_script.issues)

    pb_hidden = _write_playbook(
        tmp_path / "hidden_case",
        resources={".hidden_exec": "#!/bin/sh\necho pwn\n"},
    )
    res_hidden = validate_playbook(pb_hidden, skills_root=tmp_path / "hidden_case")
    assert res_hidden.valid is False
    codes = {i.code for i in res_hidden.issues}
    assert "hidden_file" in codes

    pb_chmod = _write_playbook(
        tmp_path / "chmod_case",
        resources={"references/exec_note.md": "Safe text with executable bit.\n"},
    )
    (pb_chmod / "references" / "exec_note.md").chmod(0o755)
    res_chmod = validate_playbook(pb_chmod, skills_root=tmp_path / "chmod_case")
    assert res_chmod.valid is False
    assert any(i.code == "executable_or_script_file" for i in res_chmod.issues)


def test_rejects_oversized_skill_md_and_oversized_resources(tmp_path: Path):
    huge_body = "# Huge\n\n" + ("A" * (MAX_SKILL_MD_BYTES + 1024))
    pb_huge = _write_playbook(tmp_path / "huge_skill", skill_md=_render_skill_md(body=huge_body))
    res_huge = validate_playbook(pb_huge, skills_root=tmp_path / "huge_skill")
    assert res_huge.valid is False
    assert any(i.code == "oversized_file" for i in res_huge.issues)

    # Exact MAX_RESOURCE_BYTES boundary is accepted; MAX_RESOURCE_BYTES + 1 is rejected
    pb_exact_res = _write_playbook(
        tmp_path / "exact_res",
        resources={"references/exact.md": "B" * MAX_RESOURCE_BYTES},
    )
    assert validate_playbook(pb_exact_res, skills_root=tmp_path / "exact_res").valid is True

    pb_huge_res = _write_playbook(
        tmp_path / "huge_res",
        resources={"references/big.md": "B" * (MAX_RESOURCE_BYTES + 1)},
    )
    res_huge_res = validate_playbook(pb_huge_res, skills_root=tmp_path / "huge_res")
    assert res_huge_res.valid is False
    assert any(i.code == "oversized_file" for i in res_huge_res.issues)


def test_rejects_invalid_utf8_in_skill_md_or_resource(tmp_path: Path):
    pb_bad_utf8 = _write_playbook(tmp_path / "bad_utf8")
    (pb_bad_utf8 / "SKILL.md").write_bytes(b"---\nname: x\n---\n\xff\xfeinvalid")
    res = validate_playbook(pb_bad_utf8, skills_root=tmp_path / "bad_utf8")
    assert res.valid is False
    assert any(i.code == "invalid_encoding" for i in res.issues)

    pb_bad_res_utf8 = _write_playbook(tmp_path / "bad_res_utf8")
    (pb_bad_res_utf8 / "references").mkdir(exist_ok=True)
    (pb_bad_res_utf8 / "references" / "bad.md").write_bytes(b"\xff\xfeinvalid_utf8_resource")
    res_bad_res = validate_playbook(pb_bad_res_utf8, skills_root=tmp_path / "bad_res_utf8")
    assert res_bad_res.valid is False
    assert any(
        i.code == "invalid_encoding" and i.field_or_path == "references/bad.md"
        for i in res_bad_res.issues
    )


def test_rejects_nonexistent_and_undeclared_tools(tmp_path: Path):
    reg = default_registry()

    # Non-existent tool in allowed_tools
    pb_nonexistent = _write_playbook(
        tmp_path / "nonexistent",
        skill_md=_render_skill_md(allowed_tools=["load_dataset", "run_pbpk_organ_model"]),
    )
    res_ne = validate_playbook(pb_nonexistent, skills_root=tmp_path / "nonexistent", registry=reg)
    assert res_ne.valid is False
    assert any(i.code == "non_existent_tool" for i in res_ne.issues)

    # Undeclared real tool referenced in SKILL.md instructions
    pb_undeclared = _write_playbook(
        tmp_path / "undeclared",
        skill_md=_render_skill_md(
            allowed_tools=["load_dataset", "compute_nca"],
            body="# Sneaky Playbook\n\nFirst run `load_dataset`, then invoke `run_nlme` without declaring it.\n",
        ),
    )
    res_un = validate_playbook(pb_undeclared, skills_root=tmp_path / "undeclared", registry=reg)
    assert res_un.valid is False
    assert any(i.code == "undeclared_tool" for i in res_un.issues)

    # Undeclared real tool referenced inside a confined resource file
    pb_res_undeclared = _write_playbook(
        tmp_path / "res_undeclared",
        skill_md=_render_skill_md(allowed_tools=["load_dataset", "compute_nca", "simulate_pk_profile"]),
        resources={"references/sneaky_ref.md": "When needed, also run `run_nlme`.\n"},
    )
    res_ref_un = validate_playbook(pb_res_undeclared, skills_root=tmp_path / "res_undeclared", registry=reg)
    assert res_ref_un.valid is False
    assert any(
        i.code == "undeclared_tool" and i.field_or_path == "references/sneaky_ref.md"
        for i in res_ref_un.issues
    )
    rep_ref_un = validate_all_playbooks(skills_root=tmp_path / "res_undeclared", registry=reg)
    assert len(rep_ref_un.missing_tool_references) == 1

    # Runtime allowlist check helper also enforces both
    _write_playbook(tmp_path / "ok")
    pb_ok = load_playbook("pharmagent.demo_protocol", skills_root=tmp_path / "ok", registry=reg)
    assert check_tool_allowed(pb_ok, "compute_nca", registry=reg) is True
    with pytest.raises(PlaybookSecurityError, match="undeclared_tool"):
        check_tool_allowed(pb_ok, "run_nlme", registry=reg)
    with pytest.raises(PlaybookSecurityError, match="non_existent_tool"):
        check_tool_allowed(pb_ok, "invented_pbpk_tool", registry=reg)


def test_rejects_external_path_references_in_markdown(tmp_path: Path):
    pb_ext = _write_playbook(
        tmp_path / "ext_ref",
        skill_md=_render_skill_md(
            body="# External Ref\n\nSee [secret](../outside.md) and `load_dataset`, `compute_nca`, `simulate_pk_profile`.\n"
        ),
    )
    res = validate_playbook(pb_ext, skills_root=tmp_path / "ext_ref")
    assert res.valid is False
    assert any(i.code in {"path_traversal", "external_reference"} for i in res.issues)


def test_validate_all_playbooks_detects_duplicate_ids_and_names(tmp_path: Path):
    _write_playbook(
        tmp_path,
        folder_name="playbook-one",
        skill_md=_render_skill_md(name="playbook-one", playbook_id="pharmagent.dup_id"),
    )
    _write_playbook(
        tmp_path,
        folder_name="playbook-two",
        skill_md=_render_skill_md(name="playbook-two", playbook_id="pharmagent.dup_id"),
    )
    report = validate_all_playbooks(skills_root=tmp_path)
    assert isinstance(report, CatalogValidationReport)
    assert report.valid is False
    assert "pharmagent.dup_id" in report.duplicate_identities


# ── 5. Isolated Unregistered Tool Adapter & CLI Contracts ────────────────────


def test_isolated_adapter_is_read_only_capped_and_unregistered(tmp_path: Path):
    _write_playbook(
        tmp_path,
        resources={"references/guide.md": "Guide text referencing `load_dataset`.\n"},
    )
    reg = default_registry()
    # Adapter tools must NOT be registered in default_registry()
    for tool in build_playbook_tools(skills_root=tmp_path, registry=reg):
        assert tool.name not in reg.names()

    state = PharmState(session_id="s_test")
    ctx = ToolContext()

    # 1. list_playbooks operation
    r_list = run_playbook_adapter(
        state, ctx, {"operation": "list_playbooks"}, skills_root=tmp_path, registry=reg
    )
    assert r_list.writes == {}
    assert r_list.result["status"] == "ok"
    assert len(r_list.result["playbooks"]) == 1
    item = r_list.result["playbooks"][0]
    assert item["playbook_id"] == "pharmagent.demo_protocol"
    assert len(item["canonical_digest"]) == 64

    # 2. load_playbook operation
    r_load = run_playbook_adapter(
        state,
        ctx,
        {"operation": "load_playbook", "playbook_id": "pharmagent.demo_protocol"},
        skills_root=tmp_path,
        registry=reg,
    )
    assert r_load.writes == {}
    assert r_load.result["status"] == "ok"
    assert r_load.result["playbook"]["playbook_id"] == "pharmagent.demo_protocol"
    assert r_load.result["playbook"]["canonical_digest"] == item["canonical_digest"]
    assert r_load.result["playbook"]["untrusted_data"] is True

    # 3. read_resource operation
    r_res = run_playbook_adapter(
        state,
        ctx,
        {
            "operation": "read_resource",
            "playbook_id": "pharmagent.demo_protocol",
            "resource": "references/guide.md",
        },
        skills_root=tmp_path,
        registry=reg,
    )
    assert r_res.writes == {}
    assert r_res.result["status"] == "ok"
    assert r_res.result["resource"]["canonical_digest"] == item["canonical_digest"]
    assert len(r_res.result["resource"]["content"]) <= MAX_ADAPTER_OUTPUT_BYTES

    # 4. Rejects arbitrary path or shell/script arguments
    r_bad = run_playbook_adapter(
        state,
        ctx,
        {"operation": "load_playbook", "playbook_id": "pharmagent.demo_protocol", "skills_root": "/etc"},
        skills_root=tmp_path,
        registry=reg,
    )
    assert r_bad.writes == {}
    assert r_bad.result["status"] == "error"


def test_cli_check_reports_counts_and_exits_nonzero_on_invalid_catalog(tmp_path: Path, capsys):
    _write_playbook(tmp_path / "valid_cat")
    rc = cli_main(["--check", "--skills-root", str(tmp_path / "valid_cat")])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Playbooks: 1" in out
    assert "Pending owner review: 1" in out
    assert "Validation failures: 0" in out
    assert "Missing tool references: 0" in out
    assert "Duplicate identities: 0" in out
    assert "Security violations: 0" in out

    # Invalid catalog in both text and JSON formats exits with rc == 1 and reports issues
    _write_playbook(
        tmp_path / "invalid_cat",
        skill_md=_render_skill_md(version="bad_semver", allowed_tools=["load_dataset", "no_such_tool"]),
    )
    rc_bad_text = cli_main(["--check", "--skills-root", str(tmp_path / "invalid_cat")])
    out_bad_text = capsys.readouterr().out
    assert rc_bad_text == 1
    assert "valid=False" in out_bad_text
    assert "! [" in out_bad_text
    assert "invalid_version" in out_bad_text

    rc_bad_json = cli_main(["--check", "--format", "json", "--skills-root", str(tmp_path / "invalid_cat")])
    out_bad_json = capsys.readouterr().out
    assert rc_bad_json == 1
    assert '"valid": false' in out_bad_json
    assert '"non_existent_tool"' in out_bad_json


def test_no_runtime_module_imports_playbooks_package():
    importers = sorted(
        str(p.relative_to(BACKEND_ROOT))
        for p in (BACKEND_ROOT / "app").rglob("*.py")
        if "playbooks" not in p.parts and "app.playbooks" in p.read_text(encoding="utf-8")
    )
    assert importers == []


def test_existing_captured_skills_remain_unchanged():
    store = SkillStore(":memory:")
    steps = distill_steps([
        {"agent": "data_manager", "tool": "load_dataset", "args": {"path": "x.csv"}},
        {"agent": "nca", "tool": "compute_nca", "args": {}},
    ])
    sk = Skill(
        name="captured-nca",
        description="captured session workflow",
        goal="reproduce NCA",
        steps=steps,
        source_session="sess_1",
        owner="alice",
        created_at="2026-10-22T00:00:00Z",
        version=1,
    )
    store.save(sk)
    loaded = store.get("captured-nca")
    assert loaded is not None
    assert loaded.steps == [{"agent": "nca", "tool": "compute_nca", "args": {}}]
    assert "# Skill: captured-nca" in loaded.to_markdown()
