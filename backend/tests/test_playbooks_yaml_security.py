"""Strict YAML frontmatter security, depth bounds, resolver isolation, and schema governance tests."""
from __future__ import annotations

import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest
import yaml

from app.core.pharmstate import PharmState
from app.playbooks import (
    PlaybookCompatibility,
    PlaybookDisabledError,
    PlaybookSummary,
    PlaybookValidationError,
    check_tool_allowed,
    freeze_value,
    list_playbooks,
    load_playbook,
    run_playbook_adapter,
    validate_all_playbooks,
    validate_playbook,
)
from app.tools.base import ToolContext
from tests.test_playbooks_catalog import _render_skill_md, _write_playbook


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


def test_yaml_anchors_aliases_numeric_keys_and_malformed_frontmatter_are_rejected(tmp_path: Path):
    for idx, (yaml_text, expected_code) in enumerate(
        [
            ("---\nname: &a test\nalias: *a\n---\nbody\n", "unsafe_yaml"),
            ("---\n123: numeric_key\n---\nbody\n", "unsafe_yaml"),
            ("---\nname: [unclosed\n---\nbody\n", "invalid_frontmatter"),
            ("---\n- just\n- a\n- list\n---\nbody\n", "invalid_frontmatter"),
            ("No frontmatter delimiter at all\n", "invalid_frontmatter"),
        ]
    ):
        d = tmp_path / f"yaml-case-{idx}"
        d.mkdir()
        (d / "SKILL.md").write_text(yaml_text, encoding="utf-8")
        res = validate_playbook(d / "SKILL.md")
        assert res.valid is False
        assert any(i.code == expected_code for i in res.issues)

    # Non-list allowed_tools branch
    nonlist_dir = tmp_path / "nonlist-tools"
    nonlist_dir.mkdir()
    (nonlist_dir / "SKILL.md").write_text(
        "---\n"
        "name: nonlist-tools\n"
        "description: Non-list allowed_tools.\n"
        "playbook_id: pharmagent.nonlist_tools\n"
        "version: 1.0.0\n"
        "domain: nca\n"
        "status: draft\n"
        "approval_status: pending_owner_review\n"
        "license: MIT\n"
        "source: PharmAgent\n"
        "allowed_tools: compute_nca\n"
        "required_inputs:\n"
        "  - dataset_id\n"
        "produced_artifacts:\n"
        "  - nca_summary\n"
        "limitations:\n"
        "  - None.\n"
        "requires_human_review: true\n"
        "compatibility: PharmAgent >=0.1.0\n"
        "---\n\n"
        "Run `compute_nca`.\n",
        encoding="utf-8",
    )
    assert any(i.code == "invalid_allowed_tools" for i in validate_playbook(nonlist_dir).issues)

    # Non-security catalog failure raises PlaybookValidationError on list_playbooks
    non_sec_root = tmp_path / "non-sec-root"
    non_sec_pb = non_sec_root / "broken-ver"
    non_sec_pb.mkdir(parents=True)
    (non_sec_pb / "SKILL.md").write_text(
        "---\n"
        "name: broken-ver\n"
        "description: Invalid version only.\n"
        "playbook_id: pharmagent.broken_ver\n"
        "version: not_semver\n"
        "domain: nca\n"
        "status: draft\n"
        "approval_status: pending_owner_review\n"
        "license: MIT\n"
        "source: PharmAgent\n"
        "allowed_tools:\n"
        "  - compute_nca\n"
        "required_inputs:\n"
        "  - dataset_id\n"
        "produced_artifacts:\n"
        "  - nca_summary\n"
        "limitations:\n"
        "  - None.\n"
        "requires_human_review: true\n"
        "compatibility: PharmAgent >=0.1.0\n"
        "---\n\n"
        "Run `compute_nca`.\n",
        encoding="utf-8",
    )
    with pytest.raises(PlaybookValidationError):
        list_playbooks(skills_root=non_sec_root)
    adapter_val_err = run_playbook_adapter(
        PharmState(), ToolContext(), {"operation": "list_playbooks"}, skills_root=non_sec_root
    )
    assert adapter_val_err.result["error_code"] == "validation_error"


def test_global_yaml_safe_loader_is_not_mutated_and_deep_nesting_is_bounded(tmp_path: Path):
    # Global yaml.SafeLoader.yaml_implicit_resolvers must NOT be mutated
    parsed_date = yaml.safe_load("d: 2026-10-22")
    assert isinstance(parsed_date["d"], datetime.date)

    # Deeply nested YAML does not raise RecursionError
    deep_dir = tmp_path / "deep-yaml"
    deep_dir.mkdir()
    (deep_dir / "SKILL.md").write_text(
        "---\nx: " + ("[" * 1200) + ("]" * 1200) + "\n---\nbody\n",
        encoding="utf-8",
    )
    res_deep = validate_playbook(deep_dir)
    assert res_deep.valid is False
    assert any(i.code in {"unsafe_yaml", "invalid_frontmatter"} for i in res_deep.issues)

    # Governance invariants for rejected / deprecated / draft+approved
    gov_dir = tmp_path / "draft-approved"
    gov_dir.mkdir()
    (gov_dir / "SKILL.md").write_text(
        "---\n"
        "name: draft-approved\n"
        "description: Draft claiming approved.\n"
        "playbook_id: pharmagent.draft_approved\n"
        "version: 1.0.0\n"
        "domain: nca\n"
        "status: draft\n"
        "approval_status: approved\n"
        "approved_by: Owner\n"
        "approved_date: 2026-10-22\n"
        "license: MIT\n"
        "source: PharmAgent\n"
        "allowed_tools:\n"
        "  - compute_nca\n"
        "required_inputs:\n"
        "  - dataset_id\n"
        "produced_artifacts:\n"
        "  - nca_summary\n"
        "limitations:\n"
        "  - None.\n"
        "requires_human_review: true\n"
        "compatibility: PharmAgent >=0.1.0\n"
        "---\n\n"
        "Run `compute_nca`.\n",
        encoding="utf-8",
    )
    res_gov = validate_playbook(gov_dir, skills_root=tmp_path)
    assert res_gov.valid is False
    assert any(i.code == "invalid_approval_status" for i in res_gov.issues)

    dep_root = tmp_path / "dep-root"
    dep_dir = dep_root / "dep-pb"
    dep_dir.mkdir(parents=True)
    (dep_dir / "SKILL.md").write_text(
        "---\n"
        "name: dep-pb\n"
        "description: Deprecated playbook.\n"
        "playbook_id: pharmagent.dep_pb\n"
        "version: 1.0.0\n"
        "domain: nca\n"
        "status: deprecated\n"
        "approval_status: pending_owner_review\n"
        "license: MIT\n"
        "source: PharmAgent\n"
        "allowed_tools:\n"
        "  - compute_nca\n"
        "required_inputs:\n"
        "  - dataset_id\n"
        "produced_artifacts:\n"
        "  - nca_summary\n"
        "limitations:\n"
        "  - Deprecated.\n"
        "requires_human_review: true\n"
        "compatibility: PharmAgent >=0.1.0\n"
        "runtime_available: false\n"
        "---\n\n"
        "Run `compute_nca`.\n",
        encoding="utf-8",
    )
    with pytest.raises(PlaybookDisabledError):
        load_playbook("pharmagent.dep_pb", skills_root=dep_root)
    pb_dep = load_playbook("pharmagent.dep_pb", skills_root=dep_root, allow_disabled=True)
    with pytest.raises(PlaybookDisabledError):
        check_tool_allowed(pb_dep, "compute_nca")


def test_schema_validators_and_governance_invariants_directly():
    comp = PlaybookCompatibility(product="PharmAgent", min_version="0.1.0", engines=None)  # type: ignore[arg-type]
    assert comp.engines == ()
    frozen_comp = freeze_value(comp)
    assert isinstance(frozen_comp, MappingProxyType)
    assert freeze_value(frozen_comp) == frozen_comp

    with pytest.raises(ValueError, match="non-empty strings"):
        PlaybookCompatibility(engines=["mrgsolve", "  "])
    with pytest.raises(ValueError, match="list or tuple"):
        PlaybookCompatibility(engines=123)  # type: ignore[arg-type]

    base_kwargs: dict[str, Any] = {
        "playbook_id": "pharmagent.schema_unit",
        "name": "pharmagent-schema-unit",
        "version": "1.0.0",
        "domain": "nca",
        "description": "Direct schema validator test.",
        "status": "reviewed",
        "approval_status": "pending_owner_review",
        "canonical_digest": "a" * 64,
        "license": "MIT",
        "source": "PharmAgent internal",
        "allowed_tools": ("compute_nca",),
        "required_inputs": ("dataset_id",),
        "produced_artifacts": ("nca_summary",),
        "limitations": ("Unit test only.",),
        "requires_human_review": True,
        "compatibility": comp,
        "resource_paths": None,  # type: ignore[arg-type]
        "resource_digests": None,  # type: ignore[arg-type]
    }
    s_ok = PlaybookSummary(**base_kwargs)
    assert s_ok.resource_paths == ()
    assert s_ok.resource_digests == ()

    s_dict_dig = PlaybookSummary(**{**base_kwargs, "resource_digests": {"references/a.md": "b" * 64}})
    assert s_dict_dig.resource_digests == (("references/a.md", "b" * 64),)

    with pytest.raises(ValueError, match="kind"):
        PlaybookSummary(**{**base_kwargs, "source": {"kind": "", "reference": ""}})
    with pytest.raises(ValueError, match="non-empty string"):
        PlaybookSummary(**{**base_kwargs, "source": "   "})
    with pytest.raises(ValueError, match="compatibility"):
        PlaybookSummary(**{**base_kwargs, "compatibility": "   "})
    with pytest.raises(ValueError, match="duplicate"):
        PlaybookSummary(**{**base_kwargs, "allowed_tools": ["compute_nca", "compute_nca"]})
    with pytest.raises(ValueError, match="sequence"):
        PlaybookSummary(**{**base_kwargs, "required_inputs": "not-a-list"})
    with pytest.raises(ValueError, match="non-empty strings"):
        PlaybookSummary(**{**base_kwargs, "required_inputs": [""]})
    with pytest.raises(ValueError, match="at least one"):
        PlaybookSummary(**{**base_kwargs, "required_inputs": []})
    with pytest.raises(ValueError, match="resource_digests"):
        PlaybookSummary(**{**base_kwargs, "resource_digests": 123})
    with pytest.raises(ValueError, match="invalid resource digest"):
        PlaybookSummary(**{**base_kwargs, "resource_digests": [("ref.md", "not-a-sha256")]})
    with pytest.raises(ValueError, match="resource_digests"):
        PlaybookSummary(**{**base_kwargs, "resource_digests": [()]})

    with pytest.raises(ValueError, match="requires_human_review"):
        PlaybookSummary(**{**base_kwargs, "requires_human_review": False})
    with pytest.raises(ValueError, match="allowed_tools"):
        PlaybookSummary(**{**base_kwargs, "runtime_available": True, "allowed_tools": ()})
    with pytest.raises(ValueError, match="status: approved"):
        PlaybookSummary(**{**base_kwargs, "status": "approved", "approval_status": "pending_owner_review"})
    with pytest.raises(ValueError, match="pending_owner_review"):
        PlaybookSummary(**{**base_kwargs, "approved_by": "owner"})
    with pytest.raises(ValueError, match="approved_by"):
        PlaybookSummary(**{**base_kwargs, "status": "approved", "approval_status": "approved", "approved_by": None})
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        PlaybookSummary(
            **{
                **base_kwargs,
                "status": "approved",
                "approval_status": "approved",
                "approved_by": "owner",
                "approved_date": "2026/05/01",
            }
        )
    with pytest.raises(ValueError, match="valid calendar date"):
        PlaybookSummary(
            **{
                **base_kwargs,
                "status": "approved",
                "approval_status": "approved",
                "approved_by": "owner",
                "approved_date": "2026-02-30",
            }
        )


def test_catalog_duplicate_name_conflicting_version_and_cross_collision(tmp_path: Path):
    empty_pb = tmp_path / "empty-pb"
    empty_pb.mkdir()
    cat_root = tmp_path / "cat-hazards"
    cat_root.mkdir()
    (cat_root / ".DS_Store").write_text("junk", encoding="utf-8")
    (cat_root / "stray.txt").write_text("stray", encoding="utf-8")
    (cat_root / "top-sym").symlink_to(empty_pb)

    for sub_name, pid, ver, pb_name in (
        ("pb-one", "pharmagent.dup_id", "1.0.0", "pb-one"),
        ("pb-two", "pharmagent.dup_id", "2.0.0", "pb-one"),
        ("pb-three", "pb-one", "1.0.0", "pb-three"),
    ):
        sd = cat_root / sub_name
        sd.mkdir()
        (sd / "SKILL.md").write_text(
            "---\n"
            f"name: {pb_name}\n"
            "description: Duplicate test.\n"
            f"playbook_id: {pid}\n"
            f"version: {ver}\n"
            "domain: nca\n"
            "status: draft\n"
            "approval_status: pending_owner_review\n"
            "license: MIT\n"
            "source: PharmAgent\n"
            "allowed_tools:\n"
            "  - compute_nca\n"
            "required_inputs:\n"
            "  - dataset_id\n"
            "produced_artifacts:\n"
            "  - nca_summary\n"
            "limitations:\n"
            "  - None.\n"
            "requires_human_review: true\n"
            "compatibility: PharmAgent >=0.1.0\n"
            "---\n\n"
            "Run `compute_nca`.\n",
            encoding="utf-8",
        )

    cat_rep = validate_all_playbooks(skills_root=cat_root)
    assert cat_rep.valid is False
    cat_codes = {i.code for i in cat_rep.catalog_issues}
    assert {
        "symlink_escape",
        "hidden_file",
        "unexpected_root_file",
        "duplicate_playbook_id",
        "conflicting_version",
        "duplicate_name",
    } <= cat_codes
    assert any("collides with another playbook's name" in i.message for i in cat_rep.catalog_issues)
