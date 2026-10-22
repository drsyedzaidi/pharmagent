"""Offline benchmark verification, mutation guards, and reviewer regression tests."""
from __future__ import annotations

import json

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
from app.compute.poppk_plausibility.rules_structural import by_canonical, make_finding, to_l_kg

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

    by_case = {r["case_id"]: r for r in summary.case_results}
    assert by_case["ref_dirks_meibohm_2010_typical_igg_mab"]["screening_status"] == "SCREENING_PASS"
    assert by_case["ref_dirks_meibohm_2010_typical_igg_mab"]["flagged_rules"] == []
    assert (
        by_case["placeholder_justified_extrahepatic_cl"]["screening_status"]
        == "REVIEW_ACCEPTED_WITH_JUSTIFICATION"
    )
    assert by_case["placeholder_justified_extrahepatic_cl"]["flagged_rules"] == ["PL-004"]


# ---------------------------------------------------------------------------
# 2. Explicit Mutation / False-Green Guards (independent mutations)
# ---------------------------------------------------------------------------

def test_mutation_guard_1_removed_rule_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutating the rule engine by silencing PL-004 must cause benchmark recall to drop below 1.0."""
    def _mutated_pl004(catalog, model, params, phys):
        rule = catalog.by_id("PL-004")
        return [make_finding(rule, parameter="CL", status="PASS")]

    monkeypatch.setattr(rules_mod, "eval_pl004_systemic_cl", _mutated_pl004)
    summary = run_plausibility_benchmark()
    assert summary.planted_violation_recall < 1.0


def test_mutation_guard_2_reversed_inequality_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    """Flipping < 0.03 L/kg to > 0.03 L/kg on Vc plasma floor triggers false flags on clean models."""
    def _mutated_pl006(catalog, params, phys):
        rule = catalog.by_id("PL-006")
        vc = by_canonical(params, "Vc")
        vc_l_kg = to_l_kg(vc, phys) if vc else None
        if vc_l_kg is not None and vc_l_kg > 0.03:  # Reversed inequality: > instead of <
            return [make_finding(rule, parameter="Vc", status="FLAGGED", severity="HIGH")]
        return [make_finding(rule, parameter="Vc", status="PASS")]

    monkeypatch.setattr(rules_mod, "eval_pl006_central_volume", _mutated_pl006)
    summary = run_plausibility_benchmark()
    assert summary.false_flag_rate_on_clean_models > 0.0


def test_mutation_guard_3a_critical_override_acceptance_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutating apply_justifications to allow CRITICAL overrides is caught independently."""
    def _mutated_apply_allow_critical(
        findings: tuple[PlausibilityFinding, ...],
        justifications: tuple[JustificationRecord, ...],
    ) -> tuple[PlausibilityFinding, ...]:
        j = justifications[0] if justifications else None
        return tuple(
            f.model_copy(update={"override_status": "accepted_with_justification", "justification": j})
            if f.status == "FLAGGED"
            else f
            for f in findings
        )

    monkeypatch.setattr(evaluator_mod, "apply_justifications", _mutated_apply_allow_critical)
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


def test_mutation_guard_3b_stripped_override_provenance_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutating apply_justifications to drop justification provenance on an allowed override is caught."""
    def _mutated_apply_strip_provenance(
        findings: tuple[PlausibilityFinding, ...],
        justifications: tuple[JustificationRecord, ...],
    ) -> tuple[PlausibilityFinding, ...]:
        return tuple(
            f.model_copy(update={"override_status": "accepted_with_justification", "justification": None})
            if f.status == "FLAGGED"
            else f
            for f in findings
        )

    monkeypatch.setattr(evaluator_mod, "apply_justifications", _mutated_apply_strip_provenance)
    warn_model = PopPKModelInput(
        model_id="mut_prov",
        context=PopPKContext(population="adult", reference_weight_kg=70.0, route="iv", modality="small_molecule"),
        parameters=(ParameterSpec(name="CL", value=140.0, unit="L/h"), ParameterSpec(name="Vc", value=25.0, unit="L")),
        justifications=(
            JustificationRecord(
                rule_id="PL-004",
                parameter="CL",
                rationale="Extrahepatic esterase metabolism",
                scientific_basis_or_citation="Assay ref",
                reviewer="Reviewer",
                timestamp="2026-10-22T00:00:00Z",
            ),
        ),
    )
    rep = evaluate_poppk_plausibility(warn_model)
    f04 = [f for f in rep.findings if f.rule_id == "PL-004"][0]
    assert f04.override_status == "accepted_with_justification"
    with pytest.raises(AssertionError):
        assert f04.justification is not None


# ---------------------------------------------------------------------------
# 3. Additional Branch & Edge-Case Coverage Tests
# ---------------------------------------------------------------------------

def test_non_human_species_and_unknown_population_and_renal_impairment_unresolved() -> None:
    for ctx in (
        PopPKContext(species="rat", population="adult", reference_weight_kg=0.25),
        PopPKContext(species="human", population="unknown", reference_weight_kg=70.0),
        PopPKContext(species="human", population="hepatic_impairment", child_pugh=None, reference_weight_kg=70.0),
        PopPKContext(species="human", population="adult", reference_weight_kg=-5.0),
        PopPKContext(species="human", population="pregnancy", reference_weight_kg=70.0, gestational_age_weeks=50.0),
        PopPKContext(species="human", population="elderly", reference_weight_kg=70.0, age_years=150.0),
        PopPKContext(species="human", population="adult", reference_weight_kg=70.0, blood_to_plasma_ratio=-1.0),
    ):
        m = PopPKModelInput(
            model_id="ctx_edge",
            context=ctx,
            parameters=(ParameterSpec(name="CL", value=5.0, unit="L/h"), ParameterSpec(name="Vc", value=10.0, unit="L")),
        )
        rep = evaluate_poppk_plausibility(m)
        assert rep.screening_status == "INCOMPLETE_CONTEXT"

    # Negative or non-finite eGFR leaves CL_R unresolved even in adult population
    m_neg_egfr = PopPKModelInput(
        model_id="neg_egfr",
        context=PopPKContext(population="adult", reference_weight_kg=70.0, egfr_ml_min_1_73m2=-10.0),
        parameters=(
            ParameterSpec(name="CL", value=5.0, unit="L/h"),
            ParameterSpec(name="CL_R", value=2.0, unit="L/h"),
            ParameterSpec(name="Vc", value=20.0, unit="L"),
        ),
    )
    assert [f for f in evaluate_poppk_plausibility(m_neg_egfr).findings if f.rule_id == "PL-005"][0].status == "UNRESOLVED"

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

    # Special populations with pending_owner_review rules remain UNRESOLVED even with valid covariates
    m_preg_pending = PopPKModelInput(
        model_id="preg_pending",
        context=PopPKContext(population="pregnancy", reference_weight_kg=75.0, gestational_age_weeks=32.0),
        parameters=(ParameterSpec(name="CL", value=15.0, unit="L/h"), ParameterSpec(name="Vc", value=35.0, unit="L")),
    )
    assert evaluate_poppk_plausibility(m_preg_pending).screening_status == "INCOMPLETE_CONTEXT"


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

    # Weight-normalized clearance L/h/kg, mL/h/kg, L/day/kg
    m_wt_cl = PopPKModelInput(
        model_id="wt_cl",
        context=PopPKContext(population="adult", reference_weight_kg=70.0, route="iv", modality="small_molecule"),
        parameters=(
            ParameterSpec(name="CL", value=0.1, unit="L/h/kg"),
            ParameterSpec(name="Vc", value=0.5, unit="L/kg"),
            ParameterSpec(name="Vp", value=0.4, unit="L/kg"),
            ParameterSpec(name="Q", value=10.0, unit="mL/h/kg"),
        ),
    )
    rep = evaluate_poppk_plausibility(m_wt_cl)
    assert rep.screening_status == "SCREENING_PASS"


def test_stochastic_and_structural_edge_branches() -> None:
    m_edges = PopPKModelInput(
        model_id="edges",
        context=PopPKContext(population="adult", reference_weight_kg=70.0, route="iv", modality="small_molecule"),
        parameters=(
            ParameterSpec(name="CL", value=10.0, unit="L/h"),
            ParameterSpec(name="Vc", value=2500.0, unit="L"),
            ParameterSpec(name="Vp", value=6000.0, unit="L"),
            ParameterSpec(name="Q", value=350.0, unit="L/h"),
            ParameterSpec(name="ka", value=15.0, unit="1/h"),
        ),
        iiv=(
            IIVSpec(parameter="CL", omega_var=0.09, cv_pct=90.0),
            IIVSpec(parameter="Vc", omega_var=2.0),
        ),
        omega_matrix=((0.09, 0.01),),
        correlations={"CL:Vc": 0.97},
        residual_error=(ResidualErrorSpec(kind="additive", sigma_var=0.0, unit="mg/L"),),
        predictions=(PredictionPoint(time=1.0, pred=2_000_000.0, ipred=10.0, unit="mg/L"),),
    )
    rep = evaluate_poppk_plausibility(m_edges)
    flagged_ids = {f.rule_id for f in rep.findings if f.status == "FLAGGED"}
    for expected in ("PL-006", "PL-007", "PL-009", "PL-012", "PL-014", "PL-015", "PL-016", "PL-019", "PL-020"):
        assert expected in flagged_ids

    # Asymmetric omega_matrix
    m_asym = PopPKModelInput(
        model_id="asym",
        context=PopPKContext(population="adult", reference_weight_kg=70.0, route="iv", modality="small_molecule"),
        parameters=(ParameterSpec(name="CL", value=10.0, unit="L/h"), ParameterSpec(name="Vc", value=25.0, unit="L")),
        omega_matrix=((0.09, 0.02), (0.05, 0.04)),
    )
    rep_asym = evaluate_poppk_plausibility(m_asym)
    assert [f for f in rep_asym.findings if f.rule_id == "PL-014"][0].status == "FLAGGED"

    with pytest.raises(ValueError):
        JustificationRecord(
            rule_id="PL-004",
            parameter="CL",
            rationale="   ",
            scientific_basis_or_citation="Ref",
            reviewer="Dr. X",
            timestamp="2026-10-22",
        )

    with pytest.raises(KeyError):
        load_rule_catalog().by_id("PL-999")
    assert len(build_benchmark_cases()) == 23


def test_reviewer_scientific_and_security_regressions() -> None:
    base_ctx = PopPKContext(population="adult", reference_weight_kg=70.0, route="iv", modality="small_molecule")

    # 1. Duplicate canonical parameters ("Vc" and "V1") flagged under PL-002 with override_allowed=False
    m_dup = PopPKModelInput(
        model_id="dup_vc",
        context=base_ctx,
        parameters=(
            ParameterSpec(name="CL", value=10.0, unit="L/h"),
            ParameterSpec(name="Vc", value=30.0, unit="L"),
            ParameterSpec(name="V1", value=0.5, unit="L"),
        ),
    )
    rep_dup = evaluate_poppk_plausibility(m_dup)
    f02_dup = [f for f in rep_dup.findings if f.rule_id == "PL-002" and f.status == "FLAGGED"]
    assert len(f02_dup) == 1
    assert f02_dup[0].override_allowed is False

    # 2. Apparent CL/F and Vc/F combined with valid F = 0.5 -> systemic CL = 120 L/h > Q_H (90 L/h) flagged!
    m_oral_f = PopPKModelInput(
        model_id="oral_f",
        context=PopPKContext(population="adult", reference_weight_kg=70.0, route="oral", modality="small_molecule"),
        parameters=(
            ParameterSpec(name="F", value=0.5, unit="fraction"),
            ParameterSpec(name="CL/F", value=240.0, unit="L/h"),
            ParameterSpec(name="Vc/F", value=2.0, unit="L"),
        ),
    )
    rep_oral_f = evaluate_poppk_plausibility(m_oral_f)
    assert [f for f in rep_oral_f.findings if f.rule_id == "PL-004"][0].status == "FLAGGED"
    assert [f for f in rep_oral_f.findings if f.rule_id == "PL-006"][0].status == "FLAGGED"

    # 3. PL-005: CL_R (15 L/h) > systemic CL (10 L/h) flagged HIGH with override_allowed=False
    m_clr_gt_cl = PopPKModelInput(
        model_id="clr_gt_cl",
        context=base_ctx,
        parameters=(
            ParameterSpec(name="CL", value=10.0, unit="L/h"),
            ParameterSpec(name="CL_R", value=15.0, unit="L/h"),
            ParameterSpec(name="Vc", value=25.0, unit="L"),
        ),
    )
    f05_gt = [f for f in evaluate_poppk_plausibility(m_clr_gt_cl).findings if f.rule_id == "PL-005"][0]
    assert f05_gt.status == "FLAGGED"
    assert f05_gt.override_allowed is False

    # 4. PL-008 & PL-009: 2-compartment terminal beta half-life and SC mAb slow ka pass cleanly
    m_2cmt_thalf = PopPKModelInput(
        model_id="2cmt_thalf",
        context=base_ctx,
        parameters=(
            ParameterSpec(name="CL", value=10.0, unit="L/h"),
            ParameterSpec(name="Vc", value=20.0, unit="L"),
            ParameterSpec(name="Vp", value=80.0, unit="L"),
            ParameterSpec(name="Q", value=20.0, unit="L/h"),
            ParameterSpec(name="t_half", value=9.30, unit="h"),  # matches ln(2)/beta (~9.30 h), not ln(2)*Vc/CL (1.39 h)
        ),
    )
    assert [f for f in evaluate_poppk_plausibility(m_2cmt_thalf).findings if f.rule_id == "PL-008"][0].status == "PASS"

    m_sc_mab = PopPKModelInput(
        model_id="sc_mab",
        context=PopPKContext(population="adult", reference_weight_kg=70.0, route="sc", modality="monoclonal_antibody"),
        parameters=(
            ParameterSpec(name="F", value=0.65, unit="fraction"),
            ParameterSpec(name="CL/F", value=0.015, unit="L/h"),
            ParameterSpec(name="Vc/F", value=5.0, unit="L"),
            ParameterSpec(name="ka", value=0.012, unit="1/h"),
        ),
    )
    rep_sc_mab = evaluate_poppk_plausibility(m_sc_mab)
    assert rep_sc_mab.screening_status == "SCREENING_PASS"

    # 5. PL-020: overridable HIGH concentration at idx 0 cannot mask CRITICAL negative concentration at idx 1
    m_pred_mask = PopPKModelInput(
        model_id="pred_mask",
        context=base_ctx,
        parameters=(ParameterSpec(name="CL", value=10.0, unit="L/h"), ParameterSpec(name="Vc", value=25.0, unit="L")),
        predictions=(
            PredictionPoint(time=1.0, pred=2_000_000.0, unit="mg/L"),
            PredictionPoint(time=2.0, pred=-1.0, unit="mg/L"),
        ),
    )
    rep_pred = evaluate_poppk_plausibility(m_pred_mask)
    assert rep_pred.screening_status == "CRITICAL_IMPLAUSIBILITY"
    f20 = [f for f in rep_pred.findings if f.rule_id == "PL-020"]
    assert len(f20) == 2

    # 6. RFC-8259 strict JSON safety when observed_value is NaN, and catalog immutability
    m_nan = PopPKModelInput(
        model_id="nan_json",
        context=base_ctx,
        parameters=(ParameterSpec(name="CL", value=float("nan"), unit="L/h"), ParameterSpec(name="Vc", value=25.0, unit="L")),
    )
    rep_nan = evaluate_poppk_plausibility(m_nan)
    dumped = json.dumps(rep_nan.to_dict(), allow_nan=False)
    assert "NaN" in dumped

    cat1 = load_rule_catalog()
    with pytest.raises(TypeError):
        cat1.by_id("PL-004").thresholds["mutated_key"] = 999999.0  # type: ignore[index]

    # 7. Stochastic & micro-rate reviewer edge cases (omega-only PL-012/015, normal IIV without cv_pct, inconsistent RUV)
    m_stoch_rev = PopPKModelInput(
        model_id="stoch_rev",
        context=base_ctx,
        parameters=(
            ParameterSpec(name="CL", value=10.0, unit="L/h"),
            ParameterSpec(name="Vc", value=25.0, unit="L"),
            ParameterSpec(name="k12", value=0.4, unit="mg"),  # invalid micro-rate unit -> PL-019 UNRESOLVED
        ),
        allometric_exponents={"ka": 0.5},  # unrecognized allometric key -> PL-010 UNRESOLVED
        omega_matrix=((0.0, 0.05), (0.05, 0.04)),  # zero variance with non-zero covariance -> PL-015 CRITICAL
        residual_error=(
            ResidualErrorSpec(kind="proportional", sigma_var=0.04, sd_or_cv=50.0, unit="%"),  # inconsistent -> PL-016 HIGH
            ResidualErrorSpec(kind="additive", sigma_var=-0.01, unit="mg/L"),  # negative -> PL-013 & PL-016 CRITICAL
        ),
        predictions=(
            PredictionPoint(time=float("nan"), pred=1.0, unit="mg/L"),
            PredictionPoint(time=1.0, pred=5.0, unit="unknown_conc_unit"),
        ),
    )
    rep_stoch = evaluate_poppk_plausibility(m_stoch_rev)
    by_rule = {f.rule_id: f for f in rep_stoch.findings}
    assert by_rule["PL-010"].status == "UNRESOLVED"
    assert by_rule["PL-012"].status == "PASS"
    assert by_rule["PL-013"].status == "FLAGGED"
    assert by_rule["PL-015"].status == "FLAGGED"
    assert by_rule["PL-019"].status == "UNRESOLVED"

    m_norm_iiv = PopPKModelInput(
        model_id="norm_iiv",
        context=base_ctx,
        parameters=(ParameterSpec(name="CL", value=10.0, unit="L/h"), ParameterSpec(name="Vc", value=25.0, unit="L")),
        iiv=(
            IIVSpec(parameter="CL", omega_var=0.09, distribution="normal"),
            IIVSpec(parameter="Vc", omega_var=0.04, distribution="proportional"),
        ),
    )
    rep_norm = evaluate_poppk_plausibility(m_norm_iiv)
    f12_map = {f.parameter: f.status for f in rep_norm.findings if f.rule_id == "PL-012"}
    assert f12_map["CL"] == "UNRESOLVED"
    assert f12_map["Vc"] == "PASS"


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


def test_correction_cycle_scientific_applicability_provenance_and_deep_immutability() -> None:
    from pathlib import Path

    from pydantic import ValidationError

    from app.compute.poppk_plausibility.context import resolve_physiology_context

    # 0. Canonical skill path check
    repo_root = Path(__file__).resolve().parents[2]
    assert (repo_root / "skills" / "poppk-plausibility-checker" / "SKILL.md").is_file()
    assert not (repo_root / ".claude" / "skills" / "poppk-plausibility-checker").exists()

    cat = load_rule_catalog()
    approved_all = cat.reference_physiology.model_copy(
        update={
            "special_population_rules_status": {
                "pediatric": "approved",
                "neonate": "approved",
                "elderly": "approved",
                "pregnancy": "approved",
                "renal_impairment": "approved",
                "hepatic_impairment": "approved",
            }
        }
    )
    for sp_ctx in (
        PopPKContext(population="neonate", reference_weight_kg=3.5, age_years=0.05),
        PopPKContext(population="elderly", reference_weight_kg=68.0, age_years=80.0),
        PopPKContext(population="pregnancy", reference_weight_kg=75.0, gestational_age_weeks=32.0),
        PopPKContext(population="renal_impairment", reference_weight_kg=70.0, egfr_ml_min_1_73m2=35.0),
    ):
        assert resolve_physiology_context(sp_ctx, approved_all).is_resolvable is True

    # 1. PL-004: elimination_pathway, clearance_matrix, blood_to_plasma_ratio (CL_blood = CL_plasma / Rb),
    # and ambiguous extravascular scale
    phys_rb = resolve_physiology_context(
        PopPKContext(
            population="adult",
            reference_weight_kg=70.0,
            route="iv",
             clearance_matrix="plasma",
            blood_to_plasma_ratio=0.5,
        ),
        cat.reference_physiology,
    )
    # Whole-blood reference flows Q_H and CO are never multiplied or divided by Rb in context.py
    assert phys_rb.hepatic_blood_flow_l_h == 90.0
    assert phys_rb.cardiac_output_l_h == 300.0
    assert phys_rb.blood_to_plasma_ratio == 0.5

    # Plasma CL = 60 L/h with Rb = 0.5 -> CL_blood = 60 / 0.5 = 120 L/h > Q_H (90 L/h) -> FLAGGED WARN
    m_plasma_high = PopPKModelInput(
        model_id="plasma_rb_05",
        context=PopPKContext(
            population="adult",
            reference_weight_kg=70.0,
            route="iv",
            clearance_matrix="plasma",
            blood_to_plasma_ratio=0.5,
        ),
        parameters=(ParameterSpec(name="CL", value=60.0, unit="L/h"), ParameterSpec(name="Vc", value=30.0, unit="L")),
    )
    f04_rb05 = [f for f in evaluate_poppk_plausibility(m_plasma_high).findings if f.rule_id == "PL-004"][0]
    assert f04_rb05.status == "FLAGGED"
    assert f04_rb05.normalized_value == 120.0

    # Plasma CL = 120 L/h with Rb = 2.0 -> CL_blood = 120 / 2.0 = 60 L/h <= Q_H (90 L/h) -> PASS
    m_plasma_pass = PopPKModelInput(
        model_id="plasma_rb_20",
        context=PopPKContext(
            population="adult",
            reference_weight_kg=70.0,
            route="iv",
            clearance_matrix="plasma",
            blood_to_plasma_ratio=2.0,
        ),
        parameters=(ParameterSpec(name="CL", value=120.0, unit="L/h"), ParameterSpec(name="Vc", value=30.0, unit="L")),
    )
    f04_rb20 = [f for f in evaluate_poppk_plausibility(m_plasma_pass).findings if f.rule_id == "PL-004"][0]
    assert f04_rb20.status == "PASS"
    assert f04_rb20.normalized_value == 60.0

    # Plasma matrix without blood_to_plasma_ratio -> UNRESOLVED
    m_plasma_no_rb = PopPKModelInput(
        model_id="plasma_no_rb",
        context=PopPKContext(population="adult", reference_weight_kg=70.0, route="iv", clearance_matrix="plasma"),
        parameters=(ParameterSpec(name="CL", value=20.0, unit="L/h"), ParameterSpec(name="Vc", value=30.0, unit="L")),
    )
    assert [f for f in evaluate_poppk_plausibility(m_plasma_no_rb).findings if f.rule_id == "PL-004"][0].status == "UNRESOLVED"

    # Renal or extrahepatic elimination pathway -> PL-004 NOT_APPLICABLE; unknown pathway -> UNRESOLVED
    m_renal_elim = PopPKModelInput(
        model_id="renal_elim",
        context=PopPKContext(population="adult", reference_weight_kg=70.0, route="iv", elimination_pathway="renal"),
        parameters=(ParameterSpec(name="CL", value=120.0, unit="L/h"), ParameterSpec(name="Vc", value=30.0, unit="L")),
    )
    assert [f for f in evaluate_poppk_plausibility(m_renal_elim).findings if f.rule_id == "PL-004"][0].status == "NOT_APPLICABLE"

    m_unk_elim = PopPKModelInput(
        model_id="unk_elim",
        context=PopPKContext(population="adult", reference_weight_kg=70.0, route="iv", elimination_pathway="unknown"),
        parameters=(ParameterSpec(name="CL", value=20.0, unit="L/h"), ParameterSpec(name="Vc", value=30.0, unit="L")),
    )
    assert [f for f in evaluate_poppk_plausibility(m_unk_elim).findings if f.rule_id == "PL-004"][0].status == "UNRESOLVED"

    # Oral route with generic "CL" and "Vc" (scale="auto") and no F -> resolved_scale="ambiguous" -> UNRESOLVED
    m_oral_ambig = PopPKModelInput(
        model_id="oral_ambig",
        context=PopPKContext(population="adult", reference_weight_kg=70.0, route="oral"),
        parameters=(ParameterSpec(name="CL", value=20.0, unit="L/h"), ParameterSpec(name="Vc", value=30.0, unit="L")),
    )
    rep_ambig = evaluate_poppk_plausibility(m_oral_ambig)
    assert [f for f in rep_ambig.findings if f.rule_id == "PL-004"][0].status == "UNRESOLVED"
    assert [f for f in rep_ambig.findings if f.rule_id == "PL-006"][0].status == "UNRESOLVED"

    # 2. PL-006: framed as modeling_heuristic, never "physically impossible"
    r06 = cat.by_id("PL-006")
    assert r06.rule_type == "modeling_heuristic"
    assert "physically impossible" not in r06.scientific_rationale.lower()

    # 3. PL-007: equality with Vc + sum(Vp) requires compartmental_structure == "mammillary_central_elimination"
    m_non_mam = PopPKModelInput(
        model_id="non_mam",
        context=PopPKContext(
            population="adult",
            reference_weight_kg=70.0,
            route="iv",
            compartmental_structure="peripheral_elimination",
        ),
        parameters=(
            ParameterSpec(name="CL", value=10.0, unit="L/h"),
            ParameterSpec(name="Vc", value=20.0, unit="L"),
            ParameterSpec(name="Vp", value=30.0, unit="L"),
            ParameterSpec(name="Vss", value=80.0, unit="L"),
        ),
    )
    assert [f for f in evaluate_poppk_plausibility(m_non_mam).findings if f.rule_id == "PL-007"][0].status == "PASS"

    m_unk_struct = PopPKModelInput(
        model_id="unk_struct",
        context=PopPKContext(
            population="adult",
            reference_weight_kg=70.0,
            route="iv",
            compartmental_structure="unknown",
        ),
        parameters=(
            ParameterSpec(name="CL", value=10.0, unit="L/h"),
            ParameterSpec(name="Vc", value=20.0, unit="L"),
            ParameterSpec(name="Vp", value=30.0, unit="L"),
        ),
    )
    assert [f for f in evaluate_poppk_plausibility(m_unk_struct).findings if f.rule_id == "PL-007"][0].status == "UNRESOLVED"

    # 4. PL-008: terminal half-life (t_half = ln(2)/beta) vs effective half-life (t_half_eff = ln(2)*Vss/CL)
    # For CL=10, Vc=20, Vp=80, Q=20: t_half_term = 9.301 h, whereas t_half_eff = ln(2)*100/10 = 6.931 h.
    # Supplying t_half = 6.931 h (confusing effective half-life for terminal half-life) MUST BE FLAGGED!
    m_thalf_confused = PopPKModelInput(
        model_id="thalf_confused",
        context=PopPKContext(population="adult", reference_weight_kg=70.0, route="iv"),
        parameters=(
            ParameterSpec(name="CL", value=10.0, unit="L/h"),
            ParameterSpec(name="Vc", value=20.0, unit="L"),
            ParameterSpec(name="Vp", value=80.0, unit="L"),
            ParameterSpec(name="Q", value=20.0, unit="L/h"),
            ParameterSpec(name="t_half", value=6.931, unit="h"),
        ),
    )
    f08_conf = [f for f in evaluate_poppk_plausibility(m_thalf_confused).findings if f.rule_id == "PL-008"][0]
    assert f08_conf.status == "FLAGGED"
    assert f08_conf.override_allowed is False
    assert "not interchangeable" in f08_conf.scientific_rationale

    # Supplying t_half_eff = 6.931 h AND t_half = 9.301 h both PASS!
    m_both_thalf = PopPKModelInput(
        model_id="both_thalf",
        context=PopPKContext(population="adult", reference_weight_kg=70.0, route="iv"),
        parameters=(
            ParameterSpec(name="CL", value=10.0, unit="L/h"),
            ParameterSpec(name="Vc", value=20.0, unit="L"),
            ParameterSpec(name="Vp", value=80.0, unit="L"),
            ParameterSpec(name="Q", value=20.0, unit="L/h"),
            ParameterSpec(name="t_half", value=9.301, unit="h"),
            ParameterSpec(name="t_half_eff", value=6.931, unit="h"),
        ),
    )
    f08_both = [f for f in evaluate_poppk_plausibility(m_both_thalf).findings if f.rule_id == "PL-008"]
    assert len(f08_both) == 2
    assert all(f.status == "PASS" for f in f08_both)

    # 2-compartment model with Vp but missing Q -> terminal half-life is UNRESOLVED
    m_vp_no_q = PopPKModelInput(
        model_id="vp_no_q",
        context=PopPKContext(population="adult", reference_weight_kg=70.0, route="iv"),
        parameters=(
            ParameterSpec(name="CL", value=10.0, unit="L/h"),
            ParameterSpec(name="Vc", value=20.0, unit="L"),
            ParameterSpec(name="Vp", value=80.0, unit="L"),
        ),
    )
    assert [f for f in evaluate_poppk_plausibility(m_vp_no_q).findings if f.rule_id == "PL-008"][0].status == "UNRESOLVED"

    # 5. PL-009: route/formulation dependence for ka and flip-flop kinetics
    m_depot_ff = PopPKModelInput(
        model_id="depot_ff",
        context=PopPKContext(
            population="adult",
            reference_weight_kg=70.0,
            route="im",
            formulation_release="depot_or_long_acting",
        ),
        parameters=(
            ParameterSpec(name="CL", value=15.0, unit="L/h", scale="systemic"),
            ParameterSpec(name="Vc", value=30.0, unit="L", scale="systemic"),
            ParameterSpec(name="ka", value=0.02, unit="1/h"),
        ),
    )
    assert [f for f in evaluate_poppk_plausibility(m_depot_ff).findings if f.rule_id == "PL-009"][0].status == "PASS"

    m_unk_form = PopPKModelInput(
        model_id="unk_form",
        context=PopPKContext(
            population="adult",
            reference_weight_kg=70.0,
            route="oral",
            formulation_release="unknown",
        ),
        parameters=(
            ParameterSpec(name="CL", value=15.0, unit="L/h", scale="systemic"),
            ParameterSpec(name="Vc", value=30.0, unit="L", scale="systemic"),
            ParameterSpec(name="ka", value=1.0, unit="1/h"),
        ),
    )
    assert [f for f in evaluate_poppk_plausibility(m_unk_form).findings if f.rule_id == "PL-009"][0].status == "UNRESOLVED"

    # 6. Deep immutability across input models, catalog, report, and benchmark summary
    m_immut = PopPKModelInput(
        model_id="immut",
        context=PopPKContext(population="adult", reference_weight_kg=70.0, route="iv"),
        parameters=(ParameterSpec(name="CL", value=10.0, unit="L/h"), ParameterSpec(name="Vc", value=25.0, unit="L")),
        allometric_exponents={"CL": 0.75},
        correlations={"CL:Vc": 0.2},
    )
    with pytest.raises(TypeError):
        m_immut.allometric_exponents["CL"] = 99.0  # type: ignore[index]
    with pytest.raises(TypeError):
        m_immut.correlations["CL:Vc"] = 0.99  # type: ignore[index]
    with pytest.raises(TypeError):
        cat.reference_physiology.special_population_rules_status["pediatric"] = "approved"  # type: ignore[index]
    with pytest.raises(TypeError):
        cat.reference_physiology.population_scaling.child_pugh_flow_factors["A"] = 1.0  # type: ignore[index]

    rep_immut = evaluate_poppk_plausibility(m_immut)
    with pytest.raises(TypeError):
        rep_immut.context_summary["species"] = "rat"  # type: ignore[index]
    with pytest.raises(TypeError):
        rep_immut.summary_counts["PASS"] = 999  # type: ignore[index]

    bench_sum = run_plausibility_benchmark()
    with pytest.raises(TypeError):
        bench_sum.severity_distribution["INFO"] = 0  # type: ignore[index]
    with pytest.raises(ValidationError):
        bench_sum.case_results[0].case_id = "mutated"  # type: ignore[misc]
