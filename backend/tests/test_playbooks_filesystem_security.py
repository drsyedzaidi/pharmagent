"""Filesystem confinement, path traversal, symlink escape, encoding, and executable/script rejection tests."""

from __future__ import annotations

from pathlib import Path

import pytest

import app.playbooks.catalog as catalog_mod
from app.playbooks import (
    MAX_RESOURCE_BYTES,
    MAX_SKILL_MD_BYTES,
    PlaybookLookupError,
    PlaybookSecurityError,
    PlaybookVersionError,
    load_playbook,
    read_playbook_resource,
    read_playbook_resource_for_review,
    validate_playbook,
)
from tests.test_playbooks_catalog import _render_skill_md, _write_playbook


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
def test_read_playbook_resource_rejects_traversal_absolute_and_hidden_paths(tmp_path: Path, bad_resource: str):
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
        read_playbook_resource("pharmagent.demo_protocol", "references/leak.md", skills_root=tmp_path / "catalog")


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
    assert any(i.code == "invalid_encoding" and i.field_or_path == "references/bad.md" for i in res_bad_res.issues)


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


def test_filesystem_hazards_markdown_link_scanners_and_relative_dir_mismatch(tmp_path: Path, monkeypatch):
    # Null bytes in SKILL.md
    null_dir = tmp_path / "null-byte-pb"
    null_dir.mkdir()
    (null_dir / "SKILL.md").write_bytes(b"---\nname: null-byte-pb\n---\nbody\x00evil\n")
    res_null = validate_playbook(null_dir)
    assert res_null.valid is False
    assert any(i.code == "invalid_encoding" for i in res_null.issues)

    # Missing directory, symlinked playbook directory, missing SKILL.md, outside skills_root
    assert validate_playbook(tmp_path / "does-not-exist").issues[0].code == "missing_playbook_dir"
    empty_pb = tmp_path / "empty-pb"
    empty_pb.mkdir()
    assert validate_playbook(empty_pb).issues[0].code == "missing_skill_md"

    sym_pb = tmp_path / "sym-pb"
    sym_pb.symlink_to(empty_pb)
    assert validate_playbook(sym_pb).issues[0].code == "symlink_escape"

    other_root = tmp_path / "other-root"
    other_root.mkdir()
    assert validate_playbook(empty_pb, skills_root=other_root).issues[0].code == "symlink_escape"
    assert (
        validate_playbook(empty_pb, skills_root=tmp_path / "nonexistent-root").issues[0].code == "invalid_skills_root"
    )

    # Classifier and resource file edge cases
    edge_dir = tmp_path / "edge-pb"
    (edge_dir / "references").mkdir(parents=True)
    skill_md = edge_dir / "SKILL.md"
    skill_md.write_text(
        "---\n"
        "name: wrong-directory-name\n"
        "description: Edge case test.\n"
        "playbook_id: pharmagent.edge_pb\n"
        "version: 1.0.0\n"
        "domain: unsupported_domain\n"
        "status: approved\n"
        "approval_status: pending_owner_review\n"
        "license: MIT\n"
        "source: PharmAgent\n"
        "allowed_tools:\n"
        "  - compute_nca\n"
        "  - compute_nca\n"
        "required_inputs:\n"
        "  - dataset_id\n"
        "produced_artifacts:\n"
        "  - nca_summary\n"
        "limitations:\n"
        "  - None.\n"
        "requires_human_review: true\n"
        "compatibility: PharmAgent >=0.1.0\n"
        "runtime_available: not_a_bool\n"
        "---\n\n",
        encoding="utf-8",
    )
    skill_md.chmod(0o755)
    hidden_sh = edge_dir / ".hidden.sh"
    hidden_sh.write_text("#!/bin/sh\n", encoding="utf-8")
    hidden_sh.chmod(0o755)
    (edge_dir / "references" / "data.parquet").write_text("disallowed ext\n", encoding="utf-8")

    res_edge = validate_playbook(edge_dir)
    assert res_edge.valid is False
    edge_codes = {i.code for i in res_edge.issues}
    assert {
        "empty_instructions",
        "invalid_domain",
        "invalid_approval_status",
        "invalid_field",
        "duplicate_allowed_tool",
        "directory_name_mismatch",
        "executable_or_script_file",
        "hidden_file",
        "disallowed_resource_type",
    } <= edge_codes

    # Markdown link scanner catches reference links, nested brackets, URL-encoded traversal,
    # query strings, dot segments, and file:/javascript:/data: schemes
    md_links_issues = catalog_mod._scan_markdown_links(
        "Inline [Step [1] guide](../outside.md) and [enc](references/%2e%2e/outside.md) "
        "and [q](../outside.md?x=1#sec) and [dot](./references/ok.md) "
        "and [rfc](file:/etc/passwd) and [js](javascript:alert(1)) and [data](data:text/plain,hi)\n"
        "[ref_leak]: ../outside_ref.md\n",
        "SKILL.md",
        {"references/ok.md"},
    )
    assert len(md_links_issues) >= 7
    assert {"path_traversal", "absolute_path", "external_reference"} <= {i.code for i in md_links_issues}

    # Unconfined subdirectory and zero-width obfuscation rejected
    zw_dir = tmp_path / "zw-pb"
    (zw_dir / "hooks").mkdir(parents=True)
    (zw_dir / "hooks" / "hook.md").write_text("unconfined subdir\n", encoding="utf-8")
    (zw_dir / "notes").mkdir(parents=True)
    (zw_dir / "notes" / "extra.md").write_text("non-references subdir\n", encoding="utf-8")
    (zw_dir / "SKILL.md").write_text(
        "---\n"
        "name: zw-pb\n"
        "description: Zero-width obfuscation test.\n"
        "playbook_id: pharmagent.zw_pb\n"
        "version: 1.0.0\n"
        "domain: nca\n"
        "status: draft\n"
        "approval_status: pending_owner_review\n"
        "license: MIT\n"
        "source: PharmAgent\n"
        "allowed_tools:\n"
        "  - load_dataset\n"
        "required_inputs:\n"
        "  - dataset_id\n"
        "produced_artifacts:\n"
        "  - nca_summary\n"
        "limitations:\n"
        "  - None.\n"
        "requires_human_review: true\n"
        "compatibility: PharmAgent >=0.1.0\n"
        "---\n\n"
        "Run `load_dataset` and `run_\u200bnlme`.\n",
        encoding="utf-8",
    )
    res_zw = validate_playbook(zw_dir)
    assert res_zw.valid is False
    zw_codes = {i.code for i in res_zw.issues}
    assert zw_codes == {"disallowed_resource_type", "executable_or_script_file", "invalid_encoding"}

    from app.playbooks.filesystem import _read_utf8_bounded
    from app.playbooks.validation import _scan_tools_and_links

    zw_scan_issues = _scan_tools_and_links(
        {"allowed_tools": ["load_dataset"]},
        "Run `load_dataset` and `run_\u200bnlme`.",
        {},
        {"load_dataset", "run_nlme"},
    )
    assert any(i.code == "undeclared_tool" for i in zw_scan_issues)

    assert _read_utf8_bounded(tmp_path / "no-such-file.md", 1024, "no-such-file.md")[1][0].code == "unreadable_file"
    assert _read_utf8_bounded(sym_pb, 1024, "sym-pb")[1][0].code == "symlink_escape"
    assert _read_utf8_bounded(empty_pb, 1024, "empty-pb")[1][0].code == "invalid_file_type"

    # Identifier, version, and resource path argument checks
    with pytest.raises(PlaybookSecurityError):
        load_playbook("   ")
    with pytest.raises(PlaybookSecurityError):
        load_playbook("../escape")
    with pytest.raises(PlaybookVersionError):
        load_playbook("pharmagent.protocol_simulation", version="bad_version")
    with pytest.raises(PlaybookSecurityError):
        read_playbook_resource("pharmagent.protocol_simulation", "   ")
    with pytest.raises(PlaybookSecurityError):
        read_playbook_resource("pharmagent.protocol_simulation", "SKILL.md")
    with pytest.raises(PlaybookSecurityError):
        read_playbook_resource("pharmagent.protocol_simulation", "C:/Windows/win.ini")
    with pytest.raises(PlaybookSecurityError):
        read_playbook_resource_for_review("pharmagent.protocol_simulation", "references/evil.py")
    with pytest.raises(PlaybookLookupError):
        read_playbook_resource_for_review("pharmagent.protocol_simulation", "references/does-not-exist.md")

    # Relative `.` directory validation still checks directory_name_mismatch
    rel_dir = tmp_path / "rel-mismatch"
    rel_dir.mkdir()
    (rel_dir / "SKILL.md").write_text(
        "---\n"
        "name: wrong-name\n"
        "description: Relative path mismatch test.\n"
        "playbook_id: pharmagent.rel_mismatch\n"
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
        "  - None.\n"
        "requires_human_review: true\n"
        "compatibility: PharmAgent >=0.1.0\n"
        "---\n\n"
        "Run `compute_nca`.\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(rel_dir)
    res_rel = validate_playbook(Path("."))
    assert res_rel.valid is False
    assert any(i.code == "directory_name_mismatch" for i in res_rel.issues)


_BASE_BODY = "# Demo\n\nUse `load_dataset`, `compute_nca`, `simulate_pk_profile`.\n"


@pytest.mark.parametrize(
    "link_text",
    [
        "[fetch](https://evil.example/payload.sh)",
        "![beacon](http://169.254.169.254/latest/meta-data/)",
        "[ref]: https://evil.example/ref",
        "<mailto:exfil@evil.example>",
    ],
)
def test_external_web_and_mail_links_rejected_in_skill_md(tmp_path: Path, link_text: str):
    pb_dir = _write_playbook(tmp_path, skill_md=_render_skill_md(body=_BASE_BODY + "\n" + link_text + "\n"))
    res = validate_playbook(pb_dir, skills_root=tmp_path)
    assert res.valid is False
    assert any(i.code == "external_reference" and i.severity == "security" for i in res.issues)


def test_external_web_link_rejected_in_reference_resource(tmp_path: Path):
    pb_dir = _write_playbook(
        tmp_path,
        skill_md=_render_skill_md(body=_BASE_BODY + "\nSee [notes](references/notes.md).\n"),
        resources={"references/notes.md": "Download [here](https://evil.example/x).\n"},
    )
    res = validate_playbook(pb_dir, skills_root=tmp_path)
    assert res.valid is False
    assert any(i.code == "external_reference" for i in res.issues)


@pytest.mark.parametrize("stray", ["stray.yaml", "notes.md", "data.json"])
def test_allowlisted_file_outside_references_rejected(tmp_path: Path, stray: str):
    pb_dir = _write_playbook(tmp_path, resources={stray: "a: 1\n"})
    res = validate_playbook(pb_dir, skills_root=tmp_path)
    assert res.valid is False
    assert any(i.code == "disallowed_resource_type" for i in res.issues)


def _with_source_block(block: str) -> str:
    import re

    md = _render_skill_md(body=_BASE_BODY)
    out, n = re.subn(r"^source: .*$", "source:\n" + block.rstrip("\n"), md, count=1, flags=re.M)
    assert n == 1
    return out


def test_mapping_source_with_only_kind_and_reference_is_valid(tmp_path: Path):
    pb_dir = _write_playbook(tmp_path, skill_md=_with_source_block("  kind: internal\n  reference: spec\n"))
    res = validate_playbook(pb_dir, skills_root=tmp_path)
    assert res.valid is True, [i.code for i in res.issues]


@pytest.mark.parametrize(
    "block",
    [
        "  kind: internal\n  reference: spec\n  extra: hidden\n",
        "  kind: internal\n  reference: spec\n  unscanned:\n    payload: \"call run_bootstrap [x](../secret.md)\"\n",
        "  kind: internal\n  reference:\n    - spec\n",
        "  kind: 7\n  reference: spec\n",
    ],
)
def test_mapping_source_with_extra_keys_or_non_string_values_rejected(tmp_path: Path, block: str):
    pb_dir = _write_playbook(tmp_path, skill_md=_with_source_block(block))
    res = validate_playbook(pb_dir, skills_root=tmp_path)
    assert res.valid is False
