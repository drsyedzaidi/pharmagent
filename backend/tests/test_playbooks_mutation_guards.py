"""Explicit mutation guards verifying security, digest binding, AST execution bans, and deep immutability."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import app.playbooks.catalog as catalog_mod
from app.core.pharmstate import PharmState
from app.playbooks import (
    PlaybookApprovalError,
    PlaybookSecurityError,
    check_tool_allowed,
    compute_canonical_digest,
    freeze_value,
    load_playbook,
    load_playbook_for_review,
    read_playbook_resource,
    run_playbook_adapter,
    validate_playbook,
)
from app.tools.base import ToolContext
from app.tools.builtins import default_registry
from tests.test_playbooks_catalog import _render_skill_md, _write_playbook
from tests.test_playbooks_digest_and_immutability import _assert_deeply_immutable


def test_mutation_guard_1_bypass_tool_allowlist_is_caught(tmp_path: Path):
    """Guard 1: If a mutant bypasses tool-allowlist enforcement, our security check fails."""
    reg = default_registry()
    pending_pb = load_playbook_for_review("pharmagent.protocol_simulation", registry=reg)
    with pytest.raises(PlaybookApprovalError):
        check_tool_allowed(pending_pb, "load_dataset", registry=reg)

    _write_playbook(
        tmp_path,
        skill_md=_render_skill_md(
            status="approved",
            approval_status="approved",
            approved_by="Syed Saoud Zaidi",
            approved_date="2026-10-22",
        ),
    )
    pb = load_playbook("pharmagent.demo_protocol", skills_root=tmp_path, registry=reg)
    assert "run_bioequivalence" in reg.names()
    assert "run_bioequivalence" not in pb.allowed_tools

    with pytest.raises(PlaybookSecurityError, match="undeclared_tool"):
        check_tool_allowed(pb, "run_bioequivalence", skills_root=tmp_path, registry=reg)
    with pytest.raises(PlaybookSecurityError, match="non_existent_tool"):
        check_tool_allowed(pb, "non_existent_pbpk_tool", skills_root=tmp_path, registry=reg)


def test_mutation_guard_2_disable_traversal_rejection_is_caught(tmp_path: Path):
    """Guard 2: Path traversal, file:// URIs, absolute paths, and symlink escapes must always fail closed."""
    reg = default_registry()
    for attack_path in (
        "../pharmagent-model-review/SKILL.md",
        "references/../../pharmagent-model-review/SKILL.md",
        "./references/scenario-audit-template.md",
        "references/./scenario-audit-template.md",
        "/etc/passwd",
        "file:///etc/passwd",
        "https://example.com/remote.md",
        "references\\..\\secret.md",
        "references/.hidden.md",
        "skill.md",
    ):
        with pytest.raises(PlaybookSecurityError):
            read_playbook_resource("pharmagent.protocol_simulation", attack_path, registry=reg)

    trav_dir = tmp_path / "md-trav"
    trav_dir.mkdir()
    (trav_dir / "SKILL.md").write_text(
        "---\n"
        "name: md-trav\n"
        "description: Traversal link test.\n"
        "playbook_id: pharmagent.md_trav\n"
        "version: 1.0.0\n"
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
        "  - Test.\n"
        "requires_human_review: true\n"
        "compatibility: PharmAgent >=0.1.0\n"
        "---\n\n"
        "Run `compute_nca`. See [leak](../secret.md), [abs](/etc/passwd), [uri](file:///etc/passwd), "
        "and [win](references\\..\\secret.md).\n",
        encoding="utf-8",
    )
    res = validate_playbook(trav_dir, skills_root=tmp_path)
    assert res.valid is False
    codes = {i.code for i in res.issues}
    assert {"path_traversal", "absolute_path", "external_reference"} <= codes


def test_mutation_guard_3_remove_digest_binding_is_caught(tmp_path: Path):
    """Guard 3: Digest must bind metadata, instructions, AND resource contents; post-load resource tampering is caught."""
    pb = load_playbook_for_review("pharmagent.protocol_simulation")
    base_digest = pb.canonical_digest

    d_meta = compute_canonical_digest(
        {**pb.model_dump(), "version": "1.0.1"},
        instructions=pb.instructions,
        resource_digests=pb.resource_digests,
    )
    d_inst = compute_canonical_digest(
        pb.model_dump(),
        instructions=pb.instructions + "\nExtra line.",
        resource_digests=pb.resource_digests,
    )
    d_res = compute_canonical_digest(
        pb.model_dump(),
        instructions=pb.instructions,
        resource_digests=(("references/scenario-audit-template.md", "0" * 64),),
    )
    assert len({base_digest, d_meta, d_inst, d_res}) == 4

    pb_dir = tmp_path / "tamper-test"
    (pb_dir / "references").mkdir(parents=True)
    (pb_dir / "SKILL.md").write_text(
        "---\n"
        "name: tamper-test\n"
        "description: Digest binding guard.\n"
        "playbook_id: pharmagent.tamper_test\n"
        "version: 1.0.0\n"
        "domain: nca\n"
        "status: approved\n"
        "approval_status: approved\n"
        "approved_by: Syed Saoud Zaidi\n"
        "approved_date: 2026-10-22\n"
        "license: MIT\n"
        "source: PharmAgent internal\n"
        "allowed_tools:\n"
        "  - compute_nca\n"
        "required_inputs:\n"
        "  - dataset_id\n"
        "produced_artifacts:\n"
        "  - nca_summary\n"
        "limitations:\n"
        "  - Guard test.\n"
        "requires_human_review: true\n"
        "compatibility: PharmAgent >=0.1.0\n"
        "---\n\n"
        "Run `compute_nca` per [ref](references/ref.md).\n",
        encoding="utf-8",
    )
    (pb_dir / "references" / "ref.md").write_text("original reference\n", encoding="utf-8")
    loaded = load_playbook("pharmagent.tamper_test", skills_root=tmp_path)
    stale_pb = loaded.model_copy(update={"resource_digests": (("references/ref.md", "f" * 64),)})
    stripped_pb = loaded.model_copy(update={"resource_digests": ()})

    orig_load = catalog_mod.load_playbook
    try:
        catalog_mod.load_playbook = lambda *a, **kw: stale_pb  # type: ignore[assignment]
        with pytest.raises(PlaybookSecurityError, match="digest mismatch"):
            read_playbook_resource("pharmagent.tamper_test", "references/ref.md", skills_root=tmp_path)

        catalog_mod.load_playbook = lambda *a, **kw: stripped_pb  # type: ignore[assignment]
        with pytest.raises(PlaybookSecurityError, match="digest"):
            read_playbook_resource("pharmagent.tamper_test", "references/ref.md", skills_root=tmp_path)
    finally:
        catalog_mod.load_playbook = orig_load  # type: ignore[assignment]


def _check_ast_for_forbidden_execution(tree: ast.AST, label: str) -> None:
    forbidden_name_calls = {"eval", "exec", "compile", "__import__", "system", "popen", "Popen"}
    forbidden_attr_calls = {
        "system",
        "popen",
        "Popen",
        "check_output",
        "check_call",
        "execv",
        "execve",
        "execl",
        "execle",
        "execlp",
        "execvp",
        "spawnl",
        "spawnle",
        "spawnv",
        "spawnve",
    }
    forbidden_imports = {"subprocess", "pty", "shlex", "ctypes"}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in forbidden_imports, f"{label} imports {alias.name}"
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] not in forbidden_imports, f"{label} imports {node.module}"
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                assert node.func.id not in forbidden_name_calls, f"{label} calls {node.func.id}"
            elif isinstance(node.func, ast.Attribute):
                assert node.func.attr not in forbidden_attr_calls, f"{label} calls .{node.func.attr}()"


def test_mutation_guard_4_allow_script_execution_is_caught(tmp_path: Path):
    """Guard 4: `app.playbooks` must never import or invoke subprocess/os.system/eval/exec, and must reject scripts."""
    playbooks_pkg = Path(catalog_mod.__file__).resolve().parent

    with pytest.raises(AssertionError, match="calls .system"):
        _check_ast_for_forbidden_execution(ast.parse("import os\nos.system('id')\n"), "synthetic")

    for py_file in sorted(playbooks_pkg.glob("*.py")):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        _check_ast_for_forbidden_execution(tree, py_file.name)

    pb_dir = tmp_path / "shebang-md"
    (pb_dir / "references").mkdir(parents=True)
    (pb_dir / "SKILL.md").write_text(
        "---\n"
        "name: shebang-md\n"
        "description: Shebang in markdown resource.\n"
        "playbook_id: pharmagent.shebang_md\n"
        "version: 1.0.0\n"
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
        "  - Test.\n"
        "requires_human_review: true\n"
        "compatibility: PharmAgent >=0.1.0\n"
        "---\n\n"
        "Run `compute_nca`.\n",
        encoding="utf-8",
    )
    (pb_dir / "references" / "sneaky.md").write_text("\ufeff#!/usr/bin/env python3\nprint('pwn')\n", encoding="utf-8")
    (pb_dir / "references" / "double.py.md").write_text("print('double ext')\n", encoding="utf-8")
    res = validate_playbook(pb_dir, skills_root=tmp_path)
    assert res.valid is False
    exec_paths = {i.field_or_path for i in res.issues if i.code == "executable_or_script_file"}
    assert {"references/sneaky.md", "references/double.py.md"} <= exec_paths


def test_mutation_guard_5_make_returned_metadata_mutable_is_caught(tmp_path: Path):
    """Guard 5: Every returned model and metadata mapping must be deeply immutable; `_assert_deeply_immutable` catches any mutable dict/list."""
    with pytest.raises(AssertionError, match="Mutable container"):
        _assert_deeply_immutable({"mutable": [1, 2, 3]})

    frozen = freeze_value(
        {"nested": {"items": [1, 2, {"k": "v"}]}, "set_items": {"b", "a"}, "bytes_buf": bytearray(b"xyz")}
    )
    _assert_deeply_immutable(frozen)
    assert frozen["set_items"] == ("a", "b")
    assert frozen["bytes_buf"] == b"xyz"
    with pytest.raises(TypeError):
        frozen["nested"]["items"][2]["k"] = "mutated"  # type: ignore[index]

    _write_playbook(
        tmp_path,
        skill_md=_render_skill_md(
            status="approved",
            approval_status="approved",
            approved_by="Syed Saoud Zaidi",
            approved_date="2026-10-22",
        ),
    )
    out1 = run_playbook_adapter(
        PharmState(),
        ToolContext(),
        {"operation": "load_playbook", "playbook_id": "pharmagent.demo_protocol"},
        skills_root=tmp_path,
    )
    out1.result["playbook"]["allowed_tools"].append("mutated_tool")
    pb_fresh = load_playbook("pharmagent.demo_protocol", skills_root=tmp_path)
    assert "mutated_tool" not in pb_fresh.allowed_tools
    _assert_deeply_immutable(pb_fresh)
    _assert_deeply_immutable(pb_fresh.to_summary())


def test_mutation_guard_6_remove_owner_approval_gate_is_caught():
    """Guard 6 (Blocker 3 item 9): Removing the owner-approval gate in runtime loader or tool auth must fail."""
    import app.playbooks as pb_pkg

    reg = default_registry()
    pending_pb = pb_pkg.load_playbook_for_review("pharmagent.protocol_simulation", registry=reg)
    assert pending_pb.approval_status == "pending_owner_review"

    with pytest.raises(pb_pkg.PlaybookApprovalError):
        pb_pkg.load_approved_playbook("pharmagent.protocol_simulation", registry=reg)
    with pytest.raises(pb_pkg.PlaybookApprovalError):
        pb_pkg.load_playbook("pharmagent.protocol_simulation", registry=reg)
    with pytest.raises(pb_pkg.PlaybookApprovalError):
        check_tool_allowed(pending_pb, "load_dataset", registry=reg)
    with pytest.raises(pb_pkg.PlaybookApprovalError):
        pb_pkg.authorize_playbook_tool(pending_pb, "load_dataset", registry=reg)
