"""Isolated, unregistered read-only tool adapter (`app.playbooks.adapter`) and CLI (`__main__`) tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.core.pharmstate import PharmState
from app.playbooks import (
    MAX_ADAPTER_OUTPUT_BYTES,
    PlaybookVersionError,
    build_playbook_tools,
    read_playbook_resource,
    run_playbook_adapter,
)
from app.playbooks.__main__ import main as cli_main
from app.tools.base import ToolContext
from app.tools.builtins import default_registry
from tests.test_playbooks_catalog import _render_skill_md, _write_playbook

BACKEND_ROOT = Path(__file__).resolve().parents[1]


def test_isolated_adapter_is_read_only_capped_and_unregistered(tmp_path: Path):
    _write_playbook(
        tmp_path,
        skill_md=_render_skill_md(
            status="approved",
            approval_status="approved",
            approved_by="Syed Saoud Zaidi",
            approved_date="2026-10-22",
        ),
        resources={"references/guide.md": "Guide text referencing `load_dataset`.\n"},
    )
    reg = default_registry()
    for tool in build_playbook_tools(skills_root=tmp_path, registry=reg):
        assert tool.name not in reg.names()

    state = PharmState(session_id="s_test")
    ctx = ToolContext()

    r_list = run_playbook_adapter(state, ctx, {"operation": "list_playbooks"}, skills_root=tmp_path, registry=reg)
    assert r_list.writes == {}
    assert r_list.result["status"] == "ok"
    assert len(r_list.result["playbooks"]) == 1
    item = r_list.result["playbooks"][0]
    assert item["playbook_id"] == "pharmagent.demo_protocol"
    assert len(item["canonical_digest"]) == 64

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


def test_adapter_validate_catalog_and_error_paths_and_truncation(tmp_path: Path):
    reg = default_registry()
    state = PharmState()
    ctx = ToolContext()

    r_val = run_playbook_adapter(state, ctx, {"operation": "validate_catalog"}, registry=reg)
    assert r_val.writes == {}
    assert r_val.result["status"] == "ok"
    assert r_val.result["valid"] is True
    assert r_val.result["total_playbooks"] == 3

    r_op = run_playbook_adapter(state, ctx, {"operation": "execute_script"}, registry=reg)
    assert r_op.writes == {}
    assert r_op.result["error_code"] == "invalid_operation"

    assert (
        run_playbook_adapter(state, ctx, {"operation": "load_playbook"}, registry=reg).result["error_code"]
        == "missing_playbook_id"
    )
    assert (
        run_playbook_adapter(state, ctx, {"operation": "read_resource"}, registry=reg).result["error_code"]
        == "missing_playbook_id"
    )
    assert (
        run_playbook_adapter(
            state, ctx, {"operation": "read_resource", "playbook_id": "pharmagent.protocol_simulation"}, registry=reg
        ).result["error_code"]
        == "missing_resource"
    )

    assert (
        run_playbook_adapter(
            state, ctx, {"operation": "load_playbook", "playbook_id": "pharmagent.no_such"}, registry=reg
        ).result["error_code"]
        == "not_found"
    )
    assert (
        run_playbook_adapter(
            state,
            ctx,
            {"operation": "load_playbook", "playbook_id": "pharmagent.protocol_simulation", "version": "9.9.9"},
            registry=reg,
        ).result["error_code"]
        == "version_mismatch"
    )
    assert (
        run_playbook_adapter(
            state,
            ctx,
            {
                "operation": "read_resource",
                "playbook_id": "pharmagent.protocol_simulation",
                "resource": "../outside.md",
            },
            registry=reg,
        ).result["error_code"]
        == "security_violation"
    )

    (tool,) = build_playbook_tools(registry=reg)
    t_res = tool.run(state, ctx, {"operation": "list_playbooks", "include_disabled": False})
    assert t_res.writes == {}
    # Runtime discovery lists only loadable playbooks; all shipped playbooks await owner approval.
    assert t_res.result["playbooks"] == []

    rc_ok = cli_main(["--check", "--format", "json"])
    assert rc_ok == 0
    rc_missing = cli_main(["--check", "--format", "json", "--skills-root", str(tmp_path / "missing_dir")])
    assert rc_missing == 1


def test_adapter_output_truncation_when_instructions_exceed_cap(tmp_path: Path):
    pb_dir = tmp_path / "large-valid-pb"
    pb_dir.mkdir()
    large_body = "Run `compute_nca`.\n\n" + ("Detailed procedural step paragraph. " * 2000)
    (pb_dir / "SKILL.md").write_text(
        "---\n"
        "name: large-valid-pb\n"
        "description: Large instructions truncation test.\n"
        "playbook_id: pharmagent.large_valid\n"
        "version: 1.0.0\n"
        "domain: nca\n"
        "status: approved\n"
        "approval_status: approved\n"
        "approved_by: Syed Saoud Zaidi\n"
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
        "---\n\n" + large_body,
        encoding="utf-8",
    )
    out = run_playbook_adapter(
        PharmState(),
        ToolContext(),
        {"operation": "load_playbook", "playbook_id": "pharmagent.large_valid"},
        skills_root=tmp_path,
    )
    assert out.writes == {}
    assert out.result["status"] == "ok"
    assert out.result["playbook"]["truncated"] is True
    assert len(json.dumps(out.result).encode("utf-8")) <= 64 * 1024


def test_adapter_version_and_argument_validation_hardening():
    with pytest.raises(PlaybookVersionError):
        read_playbook_resource(
            "pharmagent.protocol_simulation",
            "references/scenario-audit-template.md",
            version="9.9.9",
        )
    r_ver_err = run_playbook_adapter(
        PharmState(),
        ToolContext(),
        {
            "operation": "read_resource",
            "playbook_id": "pharmagent.protocol_simulation",
            "version": "9.9.9",
            "resource": "references/scenario-audit-template.md",
        },
    )
    assert r_ver_err.result["error_code"] == "version_mismatch"

    r_nonmap = run_playbook_adapter(PharmState(), ToolContext(), ["not", "a", "dict"])  # type: ignore[arg-type]
    assert r_nonmap.result["error_code"] == "invalid_arguments"

    r_nonbool = run_playbook_adapter(
        PharmState(),
        ToolContext(),
        {"operation": "list_playbooks", "include_disabled": "false"},
    )
    assert r_nonbool.result["error_code"] == "invalid_arguments"
