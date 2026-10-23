"""Catalog discovery, progressive disclosure, shipped playbook contracts, and session-skill separation tests."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

import pytest

from app.core.pharmstate import PharmState
from app.core.skills import Skill, SkillStore, distill_steps
from app.playbooks import (
    DEFAULT_SKILLS_ROOT,
    CatalogValidationReport,
    PlaybookApprovalError,
    PlaybookCompatibility,
    PlaybookDisabledError,
    PlaybookLookupError,
    PlaybookProvenance,
    PlaybookResource,
    PlaybookSummary,
    PlaybookValidationResult,
    PlaybookVersionError,
    ScientificPlaybook,
    compute_canonical_digest,
    list_playbooks,
    load_playbook,
    load_playbook_for_review,
    playbook_provenance,
    read_playbook_resource,
    read_playbook_resource_for_review,
    run_playbook_adapter,
    validate_all_playbooks,
    validate_playbook,
)
from app.tools.base import ToolContext
from app.tools.builtins import default_registry


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
    limits = (
        limitations
        if limitations is not None
        else [
            "Does not execute PBPK or QSP models.",
            "Requires human pharmacometrician review before protocol recommendation.",
        ]
    )
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
    lines.extend(
        [
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
        ]
    )
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
        (
            {
                "status": "approved",
                "approval_status": "approved",
                "approved_by": "owner",
                "approved_date": "2026-10-22",
                "runtime_available": False,
            },
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
    md_no_domain = "\n".join(line for line in _render_skill_md().splitlines() if not line.startswith("domain:")) + "\n"
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

    with pytest.raises(PlaybookApprovalError):
        load_playbook("pharmagent.zeta", version="1.0.0", skills_root=tmp_path)

    pb = load_playbook_for_review("pharmagent.zeta", version="1.0.0", skills_root=tmp_path)
    assert "Zeta Secret Marker" in pb.instructions
    assert "DEEP_RESOURCE_MARKER" not in pb.instructions
    assert pb.resource_paths == ("references/deep.md",)

    with pytest.raises(PlaybookVersionError):
        load_playbook_for_review("pharmagent.zeta", version="9.9.9", skills_root=tmp_path)

    with pytest.raises(PlaybookLookupError):
        load_playbook_for_review("pharmagent.does_not_exist", skills_root=tmp_path)

    with pytest.raises(PlaybookApprovalError):
        read_playbook_resource("pharmagent.zeta", "references/deep.md", skills_root=tmp_path)

    res = read_playbook_resource_for_review("pharmagent.zeta", "references/deep.md", skills_root=tmp_path)
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

    pb = load_playbook_for_review("pharmagent.draft_spec", skills_root=tmp_path, allow_disabled=True)
    assert pb.runtime_available is False
    assert pb.allowed_tools == ()


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


def test_shipped_playbooks_validate_against_real_default_registry():
    reg = default_registry()
    report = validate_all_playbooks(skills_root=DEFAULT_SKILLS_ROOT, registry=reg)
    assert report.valid is True
    assert report.total_playbooks == 3
    assert report.approved_count == 0
    assert report.pending_approval_count == 3
    assert report.draft_count == 1
    assert report.reviewed_count == 2
    assert report.deprecated_count == 0
    assert report.runtime_available_count == 2
    assert report.disabled_count == 1
    assert report.missing_tool_references == ()
    assert report.duplicate_identities == ()
    assert report.security_violations == ()

    summaries = list_playbooks(skills_root=DEFAULT_SKILLS_ROOT, registry=reg)
    assert [s.playbook_id for s in summaries] == [
        "pharmagent.literature_extraction",
        "pharmagent.model_review",
        "pharmagent.protocol_simulation",
    ]
    runtime_only = list_playbooks(skills_root=DEFAULT_SKILLS_ROOT, registry=reg, include_disabled=False)
    # Runtime discovery lists only loadable playbooks; all shipped playbooks await owner approval.
    assert [s.playbook_id for s in runtime_only] == []


def test_protocol_simulation_playbook_content_contract():
    reg = default_registry()
    registered = set(reg.names())
    with pytest.raises(PlaybookApprovalError):
        load_playbook("pharmagent.protocol_simulation", registry=reg)
    pb = load_playbook_for_review("pharmagent.protocol_simulation", registry=reg)

    assert pb.name == "pharmagent-protocol-simulation"
    assert pb.version == "1.0.0"
    assert pb.domain == "clinical_trial_simulation"
    assert pb.status == "reviewed"
    assert pb.approval_status == "pending_owner_review"
    assert pb.requires_human_review is True
    assert pb.runtime_available is True
    assert set(pb.allowed_tools) <= registered

    expected_tools = {
        "load_dataset",
        "profile_pk_dataset",
        "validate_cdisc",
        "compute_nca",
        "simulate_pk_profile",
        "run_dose_sweep",
        "run_clinsim",
        "run_special_population",
        "run_pediatric_simulation",
        "fit_exposure_response",
        "select_optimal_dose",
        "run_qc",
        "adversarial_review",
    }
    assert expected_tools <= set(pb.allowed_tools)

    text = pb.instructions.lower()
    for required_concept in (
        "scientific question",
        "population",
        "route",
        "formulation",
        "dose",
        "treatment duration",
        "covariates",
        "endpoints",
        "comparator",
        "uncertainty",
        "scenario matrix",
        "scenario_id",
        "exposure-response",
        "unresolved",
        "human",
        "canonical_digest",
    ):
        assert required_concept in text, f"Missing concept {required_concept!r} in protocol playbook"

    assert "no invented simulation outputs" in text
    assert "no silent unit conversion" in text
    assert "no silent imputation" in text
    assert "no claim of regulatory endorsement" in text
    assert "no automatic dose recommendation without human review" in text
    assert "allowlist" in text

    res = read_playbook_resource_for_review(
        "pharmagent.protocol_simulation",
        "references/scenario-audit-template.md",
        registry=reg,
    )
    assert res.canonical_digest == pb.canonical_digest
    assert "SCEN-01" in res.content


def test_model_review_playbook_content_contract():
    reg = default_registry()
    registered = set(reg.names())
    with pytest.raises(PlaybookApprovalError):
        load_playbook("pharmagent.model_review", registry=reg)
    pb = load_playbook_for_review("pharmagent.model_review", registry=reg)

    assert pb.name == "pharmagent-model-review"
    assert pb.version == "1.0.0"
    assert pb.domain == "pharmacometrics_model_review"
    assert pb.status == "reviewed"
    assert pb.approval_status == "pending_owner_review"
    assert pb.requires_human_review is True
    assert pb.runtime_available is True
    assert set(pb.allowed_tools) <= registered

    expected_tools = {
        "list_pk_models",
        "fit_compartmental",
        "run_poppk",
        "fit_pk_model",
        "run_nlme",
        "run_scm",
        "run_diagnostics",
        "run_vpc",
        "run_covariate_forest",
        "run_engine_comparison",
        "simulate_pk_profile",
        "adversarial_review",
    }
    assert expected_tools <= set(pb.allowed_tools)

    text = pb.instructions.lower()
    for phrase in (
        "model structure inventory",
        "parameter and unit extraction",
        "random-effects",
        "residual-error",
        "covariate",
        "translation checklist",
        "run_engine_comparison",
        "simulate_pk_profile",
        "no pbpk or qsp",
        "cannot execute nonmem or mrgsolve at runtime",
        "pr #21",
        "unmerged",
        "never compare native `ofv`, `aic`, or `bic` across different engines",
    ):
        assert phrase in text, f"Missing required phrase {phrase!r} in model-review playbook"

    res = read_playbook_resource_for_review(
        "pharmagent.model_review",
        "references/translation-verification-checklist.md",
        registry=reg,
    )
    assert res.canonical_digest == pb.canonical_digest
    assert "ADVAN" in res.content and "mrgsolve" in res.content


def test_literature_extraction_playbook_disabled_state_and_specification_contract():
    reg = default_registry()

    with pytest.raises(PlaybookDisabledError):
        load_playbook("pharmagent.literature_extraction", registry=reg)
    with pytest.raises(PlaybookDisabledError):
        read_playbook_resource(
            "pharmagent.literature_extraction",
            "references/evidence-record-schema-spec.md",
            registry=reg,
        )

    adapter_res = run_playbook_adapter(
        PharmState(),
        ToolContext(),
        {"operation": "load_playbook", "playbook_id": "pharmagent.literature_extraction"},
        registry=reg,
    )
    assert adapter_res.writes == {}
    assert adapter_res.result["status"] == "error"
    assert adapter_res.result["error_code"] == "playbook_disabled"

    pb = load_playbook_for_review("pharmagent.literature_extraction", registry=reg, allow_disabled=True)
    assert pb.status == "draft"
    assert pb.approval_status == "pending_owner_review"
    assert pb.runtime_available is False
    assert pb.requires_human_review is True
    assert pb.allowed_tools == ()

    text = pb.instructions.lower()
    for required_spec_topic in (
        "source_sha256",
        "license",
        "source_locator",
        "verbatim",
        "derived",
        "raw_unit",
        "canonical_unit",
        "ocr_confidence",
        "dispersion",
        "table_provenance",
        "prompt-injection",
        "private-document",
        "no external upload",
        "human verification",
        "immutable",
        "conflicting sources",
        "retraction",
    ):
        assert required_spec_topic in text, f"Missing spec topic {required_spec_topic!r}"

    res = read_playbook_resource_for_review(
        "pharmagent.literature_extraction",
        "references/evidence-record-schema-spec.md",
        registry=reg,
        allow_disabled=True,
    )
    assert res.canonical_digest == pb.canonical_digest
    assert "proposed_unverified" in res.content


def test_existing_captured_skills_remain_unchanged():
    store = SkillStore(":memory:")
    steps = distill_steps(
        [
            {"agent": "data_manager", "tool": "load_dataset", "args": {"path": "x.csv"}},
            {"agent": "nca", "tool": "compute_nca", "args": {}},
        ]
    )
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
