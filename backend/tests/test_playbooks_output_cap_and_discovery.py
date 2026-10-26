"""Adapter output-size cap after JSON serialization, and runtime discovery matching runtime loadability."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import app.playbooks.adapter as adapter_mod
from app.core.pharmstate import PharmState
from app.playbooks import list_playbooks, load_playbook, validate_playbook
from app.playbooks.adapter import MAX_ADAPTER_OUTPUT_BYTES, run_playbook_adapter
from app.tools.base import ToolContext
from tests.test_playbooks_catalog import _render_skill_md, _write_playbook

_BODY = "# Demo\n\nUse `load_dataset`, `compute_nca`, `simulate_pk_profile`.\n"
_APPROVED = {"status": "approved", "approval_status": "approved", "approved_by": "owner", "approved_date": "2026-10-23"}


def _size(result: object) -> int:
    return len(json.dumps(result).encode("utf-8"))


def _run(root: Path, **args: object):
    return run_playbook_adapter(PharmState(), ToolContext(), dict(args), skills_root=root)


@pytest.mark.parametrize("filler", ['"' * 60000, "\\" * 60000, "x\n" * 30000, "é" * 30000], ids=["quotes", "backslashes", "newlines", "non_ascii"])
def test_load_playbook_output_fits_cap_after_json_escaping(tmp_path: Path, filler: str):
    _write_playbook(tmp_path, skill_md=_render_skill_md(body=_BODY + filler + "\n", **_APPROVED))
    out = _run(tmp_path, operation="load_playbook", playbook_id="pharmagent.demo_protocol")
    assert out.result["status"] == "ok", out.result
    assert _size(out.result) <= MAX_ADAPTER_OUTPUT_BYTES
    assert out.result["playbook"]["truncated"] is True


def test_read_resource_output_fits_cap_after_json_escaping(tmp_path: Path):
    _write_playbook(
        tmp_path,
        skill_md=_render_skill_md(body=_BODY + "\nSee [notes](references/notes.md).\n", **_APPROVED),
        resources={"references/notes.md": '"' * 60000},
    )
    out = _run(tmp_path, operation="read_resource", playbook_id="pharmagent.demo_protocol", resource="references/notes.md")
    assert out.result["status"] == "ok", out.result
    assert _size(out.result) <= MAX_ADAPTER_OUTPUT_BYTES
    assert out.result["resource"]["truncated"] is True


def test_small_output_is_not_marked_truncated(tmp_path: Path):
    _write_playbook(tmp_path, skill_md=_render_skill_md(body=_BODY, **_APPROVED))
    out = _run(tmp_path, operation="load_playbook", playbook_id="pharmagent.demo_protocol")
    assert out.result["playbook"]["truncated"] is False
    assert "simulate_pk_profile" in out.result["playbook"]["instructions"]


@pytest.mark.parametrize("operation", ["list_playbooks", "validate_catalog"])
def test_list_and_validate_outputs_fit_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str):
    for i in range(6):
        _write_playbook(
            tmp_path,
            f"demo-{i}",
            skill_md=_render_skill_md(name=f"demo-{i}", playbook_id=f"pharmagent.demo_{i}", body=_BODY),
        )
    cap = 3000
    monkeypatch.setattr(adapter_mod, "MAX_ADAPTER_OUTPUT_BYTES", cap)
    out = _run(tmp_path, operation=operation)
    assert _size(out.result) <= cap, _size(out.result)
    assert out.result.get("truncated") is True


def test_runtime_discovery_lists_only_loadable_playbooks(tmp_path: Path):
    _write_playbook(tmp_path, "approved-pb", skill_md=_render_skill_md(name="approved-pb", playbook_id="pharmagent.approved_pb", body=_BODY, **_APPROVED))
    _write_playbook(tmp_path, "pending-pb", skill_md=_render_skill_md(name="pending-pb", playbook_id="pharmagent.pending_pb", body=_BODY))
    discovered = list_playbooks(skills_root=tmp_path, include_disabled=False)
    assert [p.playbook_id for p in discovered] == ["pharmagent.approved_pb"]
    for p in discovered:
        load_playbook(p.playbook_id, skills_root=tmp_path)
    all_ids = [p.playbook_id for p in list_playbooks(skills_root=tmp_path, include_disabled=True)]
    assert all_ids == ["pharmagent.approved_pb", "pharmagent.pending_pb"]


def test_prerelease_version_cannot_be_runtime_available(tmp_path: Path):
    pb_dir = _write_playbook(tmp_path, skill_md=_render_skill_md(version="0.1.0", runtime_available=True, body=_BODY))
    res = validate_playbook(pb_dir, skills_root=tmp_path)
    assert res.valid is False
