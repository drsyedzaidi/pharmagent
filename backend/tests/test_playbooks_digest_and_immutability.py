"""Canonical SHA-256 digest determinism, 1-byte sensitivity, and deep immutability tests."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest
from pydantic import BaseModel

from app.playbooks import (
    catalog_provenance_snapshot,
    compute_canonical_digest,
    list_playbooks,
    load_playbook_for_review,
    playbook_provenance,
    read_playbook_resource_for_review,
    validate_all_playbooks,
    validate_playbook,
)
from app.tools.builtins import default_registry
from tests.test_playbooks_catalog import _render_skill_md, _write_playbook


def _assert_deeply_immutable(obj: Any, path: str = "root") -> None:
    """Assert that no mutable container (list, dict, set, bytearray) exists anywhere in `obj`."""
    if isinstance(obj, (str, int, float, bool, bytes, type(None))):
        return
    if isinstance(obj, (list, dict, set, bytearray)):
        raise AssertionError(f"Mutable container {type(obj).__name__} found at {path}")
    if isinstance(obj, tuple):
        for idx, item in enumerate(obj):
            _assert_deeply_immutable(item, f"{path}[{idx}]")
        return
    if isinstance(obj, MappingProxyType):
        for k, v in obj.items():
            _assert_deeply_immutable(k, f"{path}.key({k!r})")
            _assert_deeply_immutable(v, f"{path}[{k!r}]")
        return
    if isinstance(obj, BaseModel):
        if not obj.model_config.get("frozen"):
            raise AssertionError(f"Unfrozen Pydantic model {type(obj).__name__} at {path}")
        for field_name in type(obj).model_fields:
            _assert_deeply_immutable(getattr(obj, field_name), f"{path}.{field_name}")
        return
    raise AssertionError(f"Unexpected type {type(obj).__name__} at {path}")


def test_canonical_digest_changes_on_single_byte_change_in_metadata_body_or_resource(tmp_path: Path):
    pb_dir = _write_playbook(
        tmp_path,
        resources={"references/note.md": "Reference value tolerance: 0.01\n"},
    )
    base_pb = load_playbook_for_review("pharmagent.demo_protocol", skills_root=tmp_path)
    d0 = base_pb.canonical_digest

    (pb_dir / "SKILL.md").write_text(
        _render_skill_md(
            body="# Demo Protocol Playbook\n\nUse `load_dataset`, `compute_nca`, and `simulate_pk_profile` in sequence!\n"
        ),
        encoding="utf-8",
    )
    d_body = load_playbook_for_review("pharmagent.demo_protocol", skills_root=tmp_path).canonical_digest
    assert d_body != d0

    (pb_dir / "SKILL.md").write_text(
        _render_skill_md(
            description="Demo protocol simulation playbook using reviewed tools!",
            body="# Demo Protocol Playbook\n\nUse `load_dataset`, `compute_nca`, and `simulate_pk_profile` in sequence!\n",
        ),
        encoding="utf-8",
    )
    d_meta = load_playbook_for_review("pharmagent.demo_protocol", skills_root=tmp_path).canonical_digest
    assert d_meta not in {d0, d_body}

    (pb_dir / "references" / "note.md").write_text("Reference value tolerance: 0.02\n", encoding="utf-8")
    d_res = load_playbook_for_review("pharmagent.demo_protocol", skills_root=tmp_path).canonical_digest
    assert d_res not in {d0, d_body, d_meta}


def test_canonical_digest_is_reproducible_and_newline_normalized(tmp_path: Path):
    lf_dir = tmp_path / "lf"
    crlf_dir = tmp_path / "crlf"
    reordered_dir = tmp_path / "reordered"
    md_lf = _render_skill_md()
    md_crlf = md_lf.replace("\n", "\r\n")
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

    pb_lf = load_playbook_for_review("pharmagent.demo_protocol", skills_root=lf_dir)
    pb_crlf = load_playbook_for_review("pharmagent.demo_protocol", skills_root=crlf_dir)
    pb_reordered = load_playbook_for_review("pharmagent.demo_protocol", skills_root=reordered_dir)
    assert pb_lf.canonical_digest == pb_crlf.canonical_digest == pb_reordered.canonical_digest
    assert pb_lf.canonical_digest == compute_canonical_digest(pb_lf)


def test_caller_isolation_and_deep_immutability_across_all_returned_types():
    reg = default_registry()
    summaries_a = list_playbooks(registry=reg)
    pb_a = load_playbook_for_review("pharmagent.protocol_simulation", registry=reg)
    res_a = read_playbook_resource_for_review(
        "pharmagent.protocol_simulation",
        "references/scenario-audit-template.md",
        registry=reg,
    )
    prov_a = playbook_provenance("pharmagent.protocol_simulation", registry=reg)
    snap_a = catalog_provenance_snapshot(registry=reg)
    report_a = validate_all_playbooks(registry=reg)

    for obj in (*summaries_a, pb_a, res_a, prov_a, snap_a, report_a):
        _assert_deeply_immutable(obj)
    for obj in (summaries_a[0], pb_a, res_a, prov_a, report_a):
        _assert_deeply_immutable(obj.to_metadata_mapping())

    with pytest.raises(TypeError):
        snap_a["playbooks"][0]["status"] = "approved"  # type: ignore[index]

    pb_b = load_playbook_for_review("pharmagent.protocol_simulation", registry=reg)
    assert pb_b == pb_a
    assert pb_b.canonical_digest == pb_a.canonical_digest


def test_deterministic_output_regardless_of_filesystem_enumeration_order(monkeypatch):
    orig_iterdir = Path.iterdir

    monkeypatch.setattr(Path, "iterdir", lambda self: iter(sorted(orig_iterdir(self), key=lambda p: p.name)))
    list_fwd = list_playbooks()
    rep_fwd = validate_all_playbooks()
    pb_fwd = load_playbook_for_review("pharmagent.protocol_simulation")

    monkeypatch.setattr(
        Path,
        "iterdir",
        lambda self: iter(sorted(orig_iterdir(self), key=lambda p: p.name, reverse=True)),
    )
    list_rev = list_playbooks()
    rep_rev = validate_all_playbooks()
    pb_rev = load_playbook_for_review("pharmagent.protocol_simulation")

    assert list_fwd == list_rev
    assert rep_fwd.model_dump() == rep_rev.model_dump()
    assert pb_fwd.canonical_digest == pb_rev.canonical_digest
    assert pb_fwd.resource_paths == pb_rev.resource_paths


def test_digest_computation_from_summary_and_mapping_and_source_mapping(tmp_path: Path):
    pb = load_playbook_for_review("pharmagent.protocol_simulation")
    summary = pb.to_summary()
    assert compute_canonical_digest(summary, instructions=pb.instructions) == pb.canonical_digest
    assert (
        compute_canonical_digest(
            summary.model_dump(),
            instructions=pb.instructions,
            resource_digests=dict(pb.resource_digests),
        )
        == pb.canonical_digest
    )
    with pytest.raises(ValueError):
        compute_canonical_digest(summary)
    with pytest.raises(ValueError):
        compute_canonical_digest(summary.model_dump())
    with pytest.raises(TypeError):
        compute_canonical_digest(12345)  # type: ignore[arg-type]

    pb_dir = tmp_path / "map-source"
    pb_dir.mkdir()
    (pb_dir / "SKILL.md").write_text(
        "---\n"
        "name: map-source\n"
        "description: Mapping source test.\n"
        "playbook_id: pharmagent.map_source\n"
        "version: 1.0.0\n"
        "domain: nca\n"
        "status: draft\n"
        "approval_status: pending_owner_review\n"
        "license: MIT\n"
        "source:\n"
        "  kind: owner_decision\n"
        "  reference: PharmAgent internal\n"
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
    v_res = validate_playbook(pb_dir, skills_root=tmp_path)
    assert v_res.valid is True
    assert v_res.playbook is not None
    assert v_res.playbook.source == "owner_decision: PharmAgent internal"
    assert compute_canonical_digest(v_res.playbook) == v_res.playbook.canonical_digest
