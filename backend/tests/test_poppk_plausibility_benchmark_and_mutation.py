"""Offline benchmark verification, mutation guards, and branch coverage tests."""
from __future__ import annotations

import pytest

from app.compute.poppk_plausibility import (
    IIVSpec,
    JustificationRecord,
    ParameterSpec,
    PlausibilityFinding,
    PopPKContext,
    PopPKModelInput,
    PredictionPoint,
    ResidualErrorSpec,
    UnitCompatibilityError,
    build_benchmark_cases,
    evaluate_poppk_plausibility,
    load_rule_catalog,
    normalize_parameter_unit,
    run_plausibility_benchmark,
)
from app.compute.poppk_plausibility import evaluator as evaluator_mod
from app.compute.poppk_plausibility import rules as rules_mod
from app.compute.poppk_plausibility.rules_structural import make_finding

# ---------------------------------------------------------------------------
# 1. Offline Benchmark Suite Verification
# ---------------------------------------------------------------------------

def test_offline_benchmark_suite_achieves_full_recall_and_zero_false_flags() -> None:
    summary = run_plausibility_benchmark()
    assert summary.clean_model_count == 3
    assert summary.corrupted_model_count == 18
    assert summary.published_reference_count == 2
    assert summary.planted_violation_recall == 1.0
    assert summary.false_flag_rate_on_clean_models == 0.0
    assert summary.unresolved_rate_on_incomplete_context == 1.0
    assert summary.rules_exercised_count == 20
    assert list(summary.rules_exercised) == [f"PL-{i:03d}" for i in range(1, 21)]
    assert summary.severity_distribution["CRITICAL"] > 0
    assert summary.severity_distribution["HIGH"] > 0
    assert summary.severity_distribution["WARN"] > 0


# ---------------------------------------------------------------------------
# 2. Explicit Mutation / False-Green Guards (3 deliberate mutations)
# ---------------------------------------------------------------------------

def test_mutation_guard_1_removed_rule_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutating the rule engine by silencing PL-004 must cause benchmark recall to drop below 1.0."""
    def _mutated_pl004(catalog, params, phys):
        rule = catalog.by_id("PL-004")
        return [make_finding(rule, parameter="CL", status="PASS")]

    monkeypatch.setattr(rules_mod, "eval_pl004_systemic_cl", _mutated_pl004)
    summary = run_plausibility_benchmark()
    assert summary.planted_violation_recall < 1.0


def test_mutation_guard_2_reversed_inequality_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutating a threshold inequality (flipping < to > on Vc plasma floor) must trigger false flags on clean models."""
    def _mutated_pl006(catalog, params, phys):
        rule = catalog.by_id("PL-006")
        # Reversed inequality: flags when vc_l_kg > min_vc_l_kg instead of < min_vc_l_kg
        return [make_finding(rule, parameter="Vc", status="FLAGGED", severity="HIGH")]

    monkeypatch.setattr(rules_mod, "eval_pl006_central_volume", _mutated_pl006)
    summary = run_plausibility_benchmark()
    assert summary.false_flag_rate_on_clean_models > 0.0


def test_mutation_guard_3_stripped_override_provenance_or_critical_override_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutating apply_justifications to strip justification provenance or allow CRITICAL override is caught."""
    def _mutated_apply_justifications(
        findings: tuple[PlausibilityFinding, ...],
        justifications: tuple[JustificationRecord, ...],
    ) -> tuple[PlausibilityFinding, ...]:
        # Defect: drops justification object and allows CRITICAL findings to be accepted
        return tuple(
            f.model_copy(update={"override_status": "accepted_with_justification", "justification": None})
            if f.status == "FLAGGED"
            else f
            for f in findings
        )

    monkeypatch.setattr(evaluator_mod, "apply_justifications", _mutated_apply_justifications)

    crit_model = PopPKModelInput(
        model_id="mut_crit",
        context=PopPKContext(population="adult", reference_weight_kg=70.0, route="iv", modality="small_molecule"),
        parameters=(ParameterSpec(name="CL", value=10.0, unit="L/h"), ParameterSpec(name="Vc", value=25.0, unit="L")),
        iiv=(IIVSpec(parameter="CL", omega_var=-0.1),),
        justifications=(
            JustificationRecord(
                rule_id="PL-013",
                parameter="CL",
                rationale="Attempted override",
                scientific_basis_or_citation="None",
                reviewer="Reviewer",
                timestamp="2026-10-22T00:00:00Z",
            ),
        ),
    )
    rep = evaluate_poppk_plausibility(crit_model)
    f13 = [f for f in rep.findings if f.rule_id == "PL-013"][0]
    with pytest.raises(AssertionError):
        assert f13.override_status == "rejected"
        assert f13.justification is not None


# ---------------------------------------------------------------------------
# 3. Additional Branch & Edge-Case Coverage Tests
# ---------------------------------------------------------------------------

def test_non_human_species_and_unknown_population_and_renal_impairment_unresolved() -> None:
    for ctx in (
        PopPKContext(species="rat", population="adult", reference_weight_kg=0.25),
        PopPKContext(species="human", population="unknown", reference_weight_kg=70.0),
        PopPKContext(species="human", population="hepatic_impairment", child_pugh=None, reference_weight_kg=70.0),
        PopPKContext(species="human", population="adult", reference_weight_kg=-5.0),
    ):
        m = PopPKModelInput(
            model_id="ctx_edge",
            context=ctx,
            parameters=(ParameterSpec(name="CL", value=5.0, unit="L/h"), ParameterSpec(name="Vc", value=10.0, unit="L")),
        )
        rep = evaluate_poppk_plausibility(m)
        assert rep.screening_status == "INCOMPLETE_CONTEXT"

    # Renal impairment without eGFR leaves CL_R unresolved
    m_renal = PopPKModelInput(
        model_id="renal_no_egfr",
        context=PopPKContext(population="renal_impairment", reference_weight_kg=70.0, egfr_ml_min_1_73m2=None),
        parameters=(
            ParameterSpec(name="CL", value=5.0, unit="L/h"),
            ParameterSpec(name="CL_R", value=2.0, unit="L/h"),
            ParameterSpec(name="Vc", value=20.0, unit="L"),
        ),
    )
    rep_renal = evaluate_poppk_plausibility(m_renal)
    f05 = [f for f in rep_renal.findings if f.rule_id == "PL-005"][0]
    assert f05.status == "UNRESOLVED"

    # Pregnancy WITH gestational_age_weeks resolves and passes plausible parameters
    m_preg_ok = PopPKModelInput(
        model_id="preg_ok",
        context=PopPKContext(population="pregnancy", reference_weight_kg=75.0, gestational_age_weeks=32.0),
        parameters=(ParameterSpec(name="CL", value=15.0, unit="L/h"), ParameterSpec(name="Vc", value=35.0, unit="L")),
    )
    assert evaluate_poppk_plausibility(m_preg_ok).screening_status == "SCREENING_PASS"


def test_unit_compatibility_errors_and_weight_normalized_flows() -> None:
    with pytest.raises(UnitCompatibilityError) as exc_missing:
        normalize_parameter_unit("CL", 10.0, None)
    assert exc_missing.value.kind == "missing"

    with pytest.raises(UnitCompatibilityError) as exc_unk:
        normalize_parameter_unit("CL", 10.0, "lightyears")
    assert exc_unk.value.kind == "unknown"

    with pytest.raises(UnitCompatibilityError) as exc_incompat:
        normalize_parameter_unit("CL", 10.0, "mg/L")
    assert exc_incompat.value.kind == "incompatible"

    # Weight-normalized clearance L/h/kg and mL/min/kg
    m_wt_cl = PopPKModelInput(
        model_id="wt_cl",
        context=PopPKContext(population="adult", reference_weight_kg=70.0, route="iv", modality="small_molecule"),
        parameters=(
            ParameterSpec(name="CL", value=0.1, unit="L/h/kg"),
            ParameterSpec(name="Vc", value=0.5, unit="L/kg"),
            ParameterSpec(name="Vp", value=0.4, unit="L/kg"),
        ),
    )
    rep = evaluate_poppk_plausibility(m_wt_cl)
    assert rep.screening_status == "SCREENING_PASS"


def test_stochastic_and_structural_edge_branches() -> None:
    # 1. Inconsistent omega_var vs cv_pct in IIV, non-square omega_matrix, near-collinear r=0.97, zero RUV, huge prediction
    m_edges = PopPKModelInput(
        model_id="edges",
        context=PopPKContext(population="adult", reference_weight_kg=70.0, route="iv", modality="small_molecule"),
        parameters=(
            ParameterSpec(name="CL", value=10.0, unit="L/h"),
            ParameterSpec(name="Vc", value=2500.0, unit="L"),  # > 20 L/kg warn, <= 50 L/kg high
            ParameterSpec(name="Vp", value=6000.0, unit="L"),  # derived Vss > 100 L/kg warn
            ParameterSpec(name="Q", value=350.0, unit="L/h"),  # > cardiac output 300 L/h
            ParameterSpec(name="ka", value=15.0, unit="1/h"),  # > 10 1/h warn
        ),
        iiv=(
            IIVSpec(parameter="CL", omega_var=0.09, cv_pct=90.0),  # inconsistent cv_pct vs omega_var
            IIVSpec(parameter="Vc", omega_var=2.0),  # > 200% CV -> HIGH
        ),
        omega_matrix=((0.09, 0.01),),  # non-square matrix
        correlations={"CL:Vc": 0.97},  # near-collinear WARN
        residual_error=(ResidualErrorSpec(kind="additive", sigma_var=0.0, unit="mg/L"),),  # zero RUV
        predictions=(PredictionPoint(time=1.0, pred=2_000_000.0, ipred=10.0, unit="mg/L"),),  # > 1e6 mg/L
    )
    rep = evaluate_poppk_plausibility(m_edges)
    flagged_ids = {f.rule_id for f in rep.findings if f.status == "FLAGGED"}
    for expected in ("PL-006", "PL-007", "PL-009", "PL-012", "PL-014", "PL-015", "PL-016", "PL-019", "PL-020"):
        assert expected in flagged_ids

    # 2. Asymmetric omega_matrix and implied correlation from symmetric omega_matrix
    m_asym = PopPKModelInput(
        model_id="asym",
        context=PopPKContext(population="adult", reference_weight_kg=70.0, route="iv", modality="small_molecule"),
        parameters=(ParameterSpec(name="CL", value=10.0, unit="L/h"), ParameterSpec(name="Vc", value=25.0, unit="L")),
        omega_matrix=((0.09, 0.02), (0.05, 0.04)),
    )
    rep_asym = evaluate_poppk_plausibility(m_asym)
    assert [f for f in rep_asym.findings if f.rule_id == "PL-014"][0].status == "FLAGGED"

    # 3. Blank justification field rejected by Pydantic validator
    with pytest.raises(ValueError):
        JustificationRecord(
            rule_id="PL-004",
            parameter="CL",
            rationale="   ",
            scientific_basis_or_citation="Ref",
            reviewer="Dr. X",
            timestamp="2026-10-22",
        )

    # 4. Unknown rule_id in catalog.by_id raises KeyError
    with pytest.raises(KeyError):
        load_rule_catalog().by_id("PL-999")
    assert len(build_benchmark_cases()) == 23


def test_isolated_read_only_tool_adapter_executes_without_mutating_state() -> None:
    from app.core.audit import AuditChain
    from app.core.pharmstate import PharmState
    from app.tools.base import ToolContext, ToolRegistry
    from app.tools.poppk_plausibility_tools import TOOLS

    reg = ToolRegistry()
    for tool in TOOLS:
        reg.register(tool)

    initial_state = PharmState(session_id="test_sess")
    ctx = ToolContext()
    audit = AuditChain()
    new_state, res = reg.execute(
        "screen_poppk_plausibility",
        state=initial_state,
        ctx=ctx,
        args={
            "model_id": "adapter_test",
            "context": {"population": "adult", "reference_weight_kg": 70.0, "route": "iv", "modality": "small_molecule"},
            "parameters": [
                {"name": "CL", "value": 10.0, "unit": "L/h"},
                {"name": "Vc", "value": 30.0, "unit": "L"},
            ],
        },
        audit=audit,
        timestamp="2026-10-22T12:00:00Z",
        actor="unit_test",
        caller_agent="poppk",
    )
    assert new_state == initial_state
    assert res.writes == {}
    assert res.result["screening_status"] == "SCREENING_PASS"
    assert "does not constitute regulatory validation" in res.summary
    assert len(audit.entries) == 1
    assert audit.verify() is True

