"""Offline software regression harness for the PopPK Physiological Plausibility Checker.

NOTE: This synthetic/reference benchmark verifies deterministic rule-engine execution,
planted-violation recall, and false-flag rate on synthetic fixtures. It does NOT
constitute scientific, clinical, or regulatory validation of any pharmacometric model.
"""
from __future__ import annotations

import math
from typing import Any, Literal

from app.compute.poppk_plausibility.evaluator import evaluate_poppk_plausibility
from app.compute.poppk_plausibility.schema import (
    FrozenIntMap,
    IIVSpec,
    JustificationRecord,
    ParameterSpec,
    PopPKContext,
    PopPKModelInput,
    PredictionPoint,
    ResidualErrorSpec,
    ScreeningStatus,
    _FrozenModel,
)


class BenchmarkCase(_FrozenModel):
    case_id: str
    group: Literal[
        "known_plausible_synthetic",
        "corrupted_planted_violation",
        "published_model_reference",
    ]
    source_status: Literal["synthetic", "literature_reference", "synthetic_placeholder"]
    citation_note: str
    model: PopPKModelInput
    expected_flagged_rules: tuple[str, ...] = ()
    expected_unresolved_rules: tuple[str, ...] = ()


class BenchmarkCaseResult(_FrozenModel):
    case_id: str
    group: str
    source_status: str
    screening_status: ScreeningStatus
    flagged_rules: tuple[str, ...]
    unresolved_rules: tuple[str, ...]

    def __getitem__(self, key: str) -> Any:
        val = getattr(self, key)
        return list(val) if isinstance(val, tuple) else val


class BenchmarkSummary(_FrozenModel):
    harness_note: str = "Synthetic/reference software regression harness only; not scientific validation."
    total_cases: int
    clean_model_count: int
    corrupted_model_count: int
    published_reference_count: int
    planted_violation_recall: float
    false_flag_rate_on_clean_models: float
    unresolved_rate_on_incomplete_context: float
    rules_exercised_count: int
    rules_exercised: tuple[str, ...]
    severity_distribution: FrozenIntMap
    case_results: tuple[BenchmarkCaseResult, ...]


def _p(name: str, value: float, unit: str, scale: str = "auto") -> ParameterSpec:
    return ParameterSpec(name=name, value=value, unit=unit, scale=scale)  # type: ignore[arg-type]


def build_benchmark_cases() -> tuple[BenchmarkCase, ...]:
    """Construct the offline software regression suite covering clean, corrupted, and reference cases."""
    base_ctx = PopPKContext(
        species="human", population="adult",
        reference_weight_kg=70.0, route="iv",
        modality="small_molecule", fraction_unbound_plasma=0.2,
        egfr_ml_min_1_73m2=100.0,
    )
    base_params = (
        _p("CL", 10.0, "L/h", scale="systemic"), _p("Vc", 25.0, "L", scale="systemic"),
    )

    clean_cases = (
        BenchmarkCase(
            case_id="clean_adult_iv_2cmt", group="known_plausible_synthetic", source_status="synthetic",
            citation_note="Synthetic reference 70 kg adult IV 2-compartment small-molecule model.",
            model=PopPKModelInput(
                model_id="clean_adult_iv_2cmt", context=base_ctx,
                parameters=(
                    _p("CL", 12.0, "L/h", scale="systemic"), _p("CL_R", 0.8, "L/h", scale="systemic"),
                    _p("Vc", 25.0, "L", scale="systemic"), _p("Vp", 35.0, "L", scale="systemic"),
                    _p("Vss", 60.0, "L", scale="systemic"), _p("Q", 15.0, "L/h", scale="systemic"),
                ),
                allometric_exponents={"CL": 0.75, "Vc": 1.0},
                iiv=(IIVSpec(parameter="CL", omega_var=0.09), IIVSpec(parameter="Vc", omega_var=0.04)),
                omega_matrix=((0.09, 0.018), (0.018, 0.04)), correlations={"CL:Vc": 0.30},
                residual_error=(ResidualErrorSpec(kind="proportional", sigma_var=0.0225, unit="fraction"),),
                predictions=(PredictionPoint(time=1.0, pred=5.0, ipred=4.8, unit="mg/L"),),
            ),
        ),
        BenchmarkCase(
            case_id="clean_adult_oral_1cmt", group="known_plausible_synthetic", source_status="synthetic",
            citation_note="Synthetic reference 70 kg adult oral 1-compartment small-molecule model.",
            model=PopPKModelInput(
                model_id="clean_adult_oral_1cmt", context=PopPKContext(population="adult", route="oral"),
                parameters=(
                    _p("F", 0.80, "fraction"), _p("CL", 8.0, "L/h", scale="systemic"),
                    _p("Vc", 40.0, "L", scale="systemic"), _p("ka", 1.2, "1/h"),
                    _p("t_half", math.log(2.0) * 5.0, "h"),
                ),
            ),
        ),
        BenchmarkCase(
            case_id="clean_adult_er_oral_1cmt", group="known_plausible_synthetic", source_status="synthetic",
            citation_note="Synthetic reference 70 kg adult extended-release oral 1-compartment model.",
            model=PopPKModelInput(
                model_id="clean_adult_er_oral_1cmt",
                context=PopPKContext(
                    population="adult", route="oral", formulation_release="extended_release"
                ),
                parameters=(
                    _p("F", 0.75, "fraction"), _p("CL", 15.0, "L/h", scale="systemic"),
                    _p("Vc", 30.0, "L", scale="systemic"), _p("ka", 0.12, "1/h"),
                ),
            ),
        ),
    )

    corrupted_cases = (
        BenchmarkCase(
            case_id="corrupt_pl001_negative_cl", group="corrupted_planted_violation",
            source_status="synthetic", citation_note="Planted violation: negative structural clearance.",
            model=PopPKModelInput(
                model_id="corrupt_pl001", context=base_ctx,
                parameters=(_p("CL", -3.0, "L/h"), _p("Vc", 25.0, "L")),
            ),
            expected_flagged_rules=("PL-001",),
        ),
        BenchmarkCase(
            case_id="corrupt_pl002_bad_unit", group="corrupted_planted_violation",
            source_status="synthetic", citation_note="Planted violation: mass unit mg supplied for clearance CL.",
            model=PopPKModelInput(
                model_id="corrupt_pl002", context=base_ctx, parameters=(_p("CL", 10.0, "mg"), _p("Vc", 25.0, "L")),
            ),
            expected_flagged_rules=("PL-002",),
        ),
        BenchmarkCase(
            case_id="corrupt_pl003_f_above_one", group="corrupted_planted_violation",
            source_status="synthetic", citation_note="Planted violation: bioavailability F = 1.4 fraction.",
            model=PopPKModelInput(
                model_id="corrupt_pl003", context=base_ctx, parameters=(*base_params, _p("F", 1.4, "fraction")),
            ),
            expected_flagged_rules=("PL-003",),
        ),
        BenchmarkCase(
            case_id="corrupt_pl004_cl_above_qh", group="corrupted_planted_violation", source_status="synthetic",
            citation_note="Planted violation: systemic CL = 140 L/h > adult Q_H (90 L/h).",
            model=PopPKModelInput(
                model_id="corrupt_pl004", context=base_ctx,
                parameters=(_p("CL", 140.0, "L/h"), _p("Vc", 30.0, "L")),
            ),
            expected_flagged_rules=("PL-004",),
        ),
        BenchmarkCase(
            case_id="corrupt_pl005_clr_above_qr", group="corrupted_planted_violation", source_status="synthetic",
            citation_note="Planted violation: renal clearance CL_R = 42 L/h > adult Q_R,plasma (36 L/h).",
            model=PopPKModelInput(
                model_id="corrupt_pl005", context=base_ctx, parameters=(*base_params, _p("CL_R", 42.0, "L/h")),
            ),
            expected_flagged_rules=("PL-005",),
        ),
        BenchmarkCase(
            case_id="corrupt_pl006_vc_below_plasma", group="corrupted_planted_violation",
            source_status="synthetic",
            citation_note="Planted violation: central volume Vc = 0.8 L < adult typical lower bound (2.1 L).",
            model=PopPKModelInput(
                model_id="corrupt_pl006", context=base_ctx, parameters=(_p("CL", 10.0, "L/h"), _p("Vc", 0.8, "L")),
            ),
            expected_flagged_rules=("PL-006",),
        ),
        BenchmarkCase(
            case_id="corrupt_pl007_vss_sum_mismatch", group="corrupted_planted_violation",
            source_status="synthetic", citation_note="Planted violation: Vss (95 L) != Vc (25 L) + Vp (35 L).",
            model=PopPKModelInput(
                model_id="corrupt_pl007", context=base_ctx,
                parameters=(*base_params, _p("Vp", 35.0, "L"), _p("Vss", 95.0, "L")),
            ),
            expected_flagged_rules=("PL-007",),
        ),
        BenchmarkCase(
            case_id="corrupt_pl008_thalf_inconsistent", group="corrupted_planted_violation",
            source_status="synthetic",
            citation_note="Planted violation: reported t_half = 10 h vs ln(2)*25/10 = 1.73 h.",
            model=PopPKModelInput(
                model_id="corrupt_pl008", context=base_ctx, parameters=(*base_params, _p("t_half", 10.0, "h")),
            ),
            expected_flagged_rules=("PL-008",),
        ),
        BenchmarkCase(
            case_id="corrupt_pl009_flip_flop", group="corrupted_planted_violation",
            source_status="synthetic",
            citation_note="Planted violation: immediate-release oral ka (0.2 1/h) <= ke (10/25 = 0.4 1/h).",
            model=PopPKModelInput(
                model_id="corrupt_pl009",
                context=PopPKContext(
                    population="adult", route="oral", formulation_release="immediate_release"
                ),
                parameters=(*base_params, _p("ka", 0.2, "1/h")),
            ),
            expected_flagged_rules=("PL-009",),
        ),
        BenchmarkCase(
            case_id="corrupt_pl010_pl011_allometry", group="corrupted_planted_violation",
            source_status="synthetic", citation_note="Planted violation: allometric exponents CL=0.15, Vc=1.75.",
            model=PopPKModelInput(
                model_id="corrupt_pl010_011", context=base_ctx, parameters=base_params,
                allometric_exponents={"CL": 0.15, "Vc": 1.75},
            ),
            expected_flagged_rules=("PL-010", "PL-011"),
        ),
        BenchmarkCase(
            case_id="corrupt_pl012_high_iiv", group="corrupted_planted_violation",
            source_status="synthetic", citation_note="Planted violation: IIV omega_var = 1.2 (> 100% CV).",
            model=PopPKModelInput(
                model_id="corrupt_pl012", context=base_ctx, parameters=base_params,
                iiv=(IIVSpec(parameter="CL", omega_var=1.2),),
            ),
            expected_flagged_rules=("PL-012",),
        ),
        BenchmarkCase(
            case_id="corrupt_pl013_negative_variance", group="corrupted_planted_violation",
            source_status="synthetic", citation_note="Planted violation: negative IIV variance omega_var = -0.02.",
            model=PopPKModelInput(
                model_id="corrupt_pl013", context=base_ctx, parameters=base_params,
                iiv=(IIVSpec(parameter="CL", omega_var=-0.02),),
            ),
            expected_flagged_rules=("PL-013",),
        ),
        BenchmarkCase(
            case_id="corrupt_pl014_pl015_non_psd_and_corr", group="corrupted_planted_violation",
            source_status="synthetic", citation_note="Planted violation: non-PSD omega_matrix and |r| > 1.",
            model=PopPKModelInput(
                model_id="corrupt_pl014_015", context=base_ctx, parameters=base_params,
                omega_matrix=((0.09, 0.25), (0.25, 0.04)), correlations={"CL:Vc": 1.38},
            ),
            expected_flagged_rules=("PL-014", "PL-015"),
        ),
        BenchmarkCase(
            case_id="corrupt_pl016_excessive_ruv", group="corrupted_planted_violation", source_status="synthetic",
            citation_note="Planted violation: proportional RUV sigma_var = 0.64 (80% CV > 50% CV).",
            model=PopPKModelInput(
                model_id="corrupt_pl016", context=base_ctx, parameters=base_params,
                residual_error=(ResidualErrorSpec(kind="proportional", sigma_var=0.64, unit="fraction"),),
            ),
            expected_flagged_rules=("PL-016",),
        ),
        BenchmarkCase(
            case_id="corrupt_pl017_pl018_mab_bounds", group="corrupted_planted_violation",
            source_status="synthetic", citation_note="Planted violation: mAb with Vc=35 L and CL=5 L/h.",
            model=PopPKModelInput(
                model_id="corrupt_pl017_018",
                context=PopPKContext(population="adult", route="iv", modality="monoclonal_antibody"),
                parameters=(_p("CL", 5.0, "L/h", scale="systemic"), _p("Vc", 35.0, "L", scale="systemic")),
            ),
            expected_flagged_rules=("PL-017", "PL-018"),
        ),
        BenchmarkCase(
            case_id="corrupt_pl019_micro_rate_mismatch", group="corrupted_planted_violation",
            source_status="synthetic", citation_note="Planted violation: k12 = 1.5 1/h != Q/Vc (10/25 = 0.4 1/h).",
            model=PopPKModelInput(
                model_id="corrupt_pl019", context=base_ctx,
                parameters=(*base_params, _p("Q", 10.0, "L/h"), _p("k12", 1.5, "1/h")),
            ),
            expected_flagged_rules=("PL-019",),
        ),
        BenchmarkCase(
            case_id="corrupt_pl020_negative_prediction", group="corrupted_planted_violation",
            source_status="synthetic",
            citation_note="Planted violation: negative predicted concentration (-0.25 mg/L).",
            model=PopPKModelInput(
                model_id="corrupt_pl020", context=base_ctx, parameters=base_params,
                predictions=(PredictionPoint(time=2.0, pred=-0.25, ipred=0.5, unit="mg/L"),),
            ),
            expected_flagged_rules=("PL-020",),
        ),
        BenchmarkCase(
            case_id="incomplete_pediatric_missing_weight", group="corrupted_planted_violation",
            source_status="synthetic",
            citation_note="Incomplete context: pediatric model missing reference_weight_kg and approved rule.",
            model=PopPKModelInput(
                model_id="incomplete_ped", context=PopPKContext(population="pediatric", reference_weight_kg=None),
                parameters=base_params,
            ),
            expected_unresolved_rules=("PL-004", "PL-006"),
        ),
    )

    published_cases = (
        BenchmarkCase(
            case_id="ref_dirks_meibohm_2010_typical_igg_mab", group="published_model_reference",
            source_status="literature_reference",
            citation_note=(
                "Dirks & Meibohm 2010 (Clin Pharmacokinet 49(10):633-659): typical human IgG1 mAb "
                "parameters (Vc ~ 3.1 L, Vp ~ 2.6 L, CL ~ 0.25 L/day = 0.0104 L/h, Q ~ 0.6 L/day)."
            ),
            model=PopPKModelInput(
                model_id="dirks_meibohm_2010_igg",
                context=PopPKContext(population="adult", route="iv", modality="monoclonal_antibody"),
                parameters=(
                    _p("CL", 0.25, "L/day", scale="systemic"), _p("Vc", 3.1, "L", scale="systemic"),
                    _p("Vp", 2.6, "L", scale="systemic"), _p("Q", 0.60, "L/day", scale="systemic"),
                ),
                allometric_exponents={"CL": 0.75, "Vc": 1.0},
            ),
        ),
        BenchmarkCase(
            case_id="placeholder_justified_extrahepatic_cl", group="published_model_reference",
            source_status="synthetic_placeholder",
            citation_note=(
                "Synthetic placeholder representing an ultra-short-acting blood-esterase-cleared "
                "opioid with documented justification for CL > Q_H."
            ),
            model=PopPKModelInput(
                model_id="placeholder_esterase_cl", context=base_ctx,
                parameters=(_p("CL", 160.0, "L/h"), _p("Vc", 12.0, "L")),
                justifications=(
                    JustificationRecord(
                        rule_id="PL-004", parameter="CL",
                        rationale="Rapid nonspecific blood and tissue esterase hydrolysis.",
                        scientific_basis_or_citation="Synthetic placeholder for esterase-cleared compound.",
                        reviewer="Benchmark Harness", timestamp="2026-10-22T00:00:00Z",
                    ),
                ),
            ),
            expected_flagged_rules=("PL-004",),
        ),
    )

    return (*clean_cases, *corrupted_cases, *published_cases)


def run_plausibility_benchmark() -> BenchmarkSummary:
    """Run all offline benchmark cases and compute recall, false-flag, and coverage metrics."""
    cases = build_benchmark_cases()
    clean_total, clean_false_flags = 0, 0
    planted_expected, planted_recalled = 0, 0
    unresolved_expected, unresolved_hit = 0, 0
    exercised_rules: set[str] = set()
    severity_dist: dict[str, int] = {"INFO": 0, "WARN": 0, "HIGH": 0, "CRITICAL": 0}
    case_results: list[BenchmarkCaseResult] = []

    for c in cases:
        rep = evaluate_poppk_plausibility(c.model)
        flagged_rules = tuple(sorted({f.rule_id for f in rep.findings if f.status == "FLAGGED"}))
        unresolved_rules = tuple(sorted({f.rule_id for f in rep.findings if f.status == "UNRESOLVED"}))
        for f in rep.findings:
            if f.status in {"PASS", "FLAGGED", "UNRESOLVED"}:
                exercised_rules.add(f.rule_id)
            severity_dist[f.severity] = severity_dist.get(f.severity, 0) + 1

        if c.group == "known_plausible_synthetic":
            clean_total += 1
            if flagged_rules:
                clean_false_flags += 1

        for exp_r in c.expected_flagged_rules:
            planted_expected += 1
            if exp_r in flagged_rules:
                planted_recalled += 1

        for exp_u in c.expected_unresolved_rules:
            unresolved_expected += 1
            if exp_u in unresolved_rules:
                unresolved_hit += 1

        case_results.append(
            BenchmarkCaseResult(
                case_id=c.case_id,
                group=c.group,
                source_status=c.source_status,
                screening_status=rep.screening_status,
                flagged_rules=flagged_rules,
                unresolved_rules=unresolved_rules,
            )
        )

    return BenchmarkSummary(
        total_cases=len(cases), clean_model_count=clean_total,
        corrupted_model_count=sum(1 for c in cases if c.group == "corrupted_planted_violation"),
        published_reference_count=sum(1 for c in cases if c.group == "published_model_reference"),
        planted_violation_recall=planted_recalled / max(planted_expected, 1),
        false_flag_rate_on_clean_models=clean_false_flags / max(clean_total, 1),
        unresolved_rate_on_incomplete_context=unresolved_hit / max(unresolved_expected, 1),
        rules_exercised_count=len(exercised_rules), rules_exercised=tuple(sorted(exercised_rules)),
        severity_distribution=severity_dist, case_results=tuple(case_results),
    )
