"""Contract, unit, boundary, context, override, and determinism tests for PopPK Plausibility Checker.

Covers all 20 deterministic plausibility rules (PL-001 through PL-020) and the
non-destructive human justification workflow.
"""
from __future__ import annotations

import copy
import math

import pytest

from app.compute.poppk_plausibility import (
    IIVSpec,
    JustificationRecord,
    ParameterSpec,
    PlausibilityFinding,
    PlausibilityReport,
    PopPKContext,
    PopPKModelInput,
    PredictionPoint,
    ResidualErrorSpec,
    evaluate_poppk_plausibility,
    load_rule_catalog,
    normalize_parameter_unit,
)


def _baseline_adult_iv_small_molecule() -> PopPKModelInput:
    """Reference plausible 2-compartment IV small-molecule PopPK model in 70 kg adults."""
    return PopPKModelInput(
        model_id="ref_sm_iv_2cmt",
        context=PopPKContext(
            species="human",
            population="adult",
            reference_weight_kg=70.0,
            route="iv",
            modality="small_molecule",
            fraction_unbound_plasma=0.2,
            egfr_ml_min_1_73m2=100.0,
        ),
        parameters=(
            ParameterSpec(name="CL", value=12.0, unit="L/h", scale="systemic"),
            ParameterSpec(name="CL_R", value=1.0, unit="L/h", scale="systemic"),
            ParameterSpec(name="Vc", value=25.0, unit="L", scale="systemic"),
            ParameterSpec(name="Vp", value=35.0, unit="L", scale="systemic"),
            ParameterSpec(name="Vss", value=60.0, unit="L", scale="systemic"),
            ParameterSpec(name="Q", value=15.0, unit="L/h", scale="systemic"),
        ),
        allometric_exponents={"CL": 0.75, "Vc": 1.0, "Q": 0.75, "Vp": 1.0},
        iiv=(
            IIVSpec(parameter="CL", omega_var=0.09),
            IIVSpec(parameter="Vc", omega_var=0.04),
        ),
        omega_matrix=((0.09, 0.018), (0.018, 0.04)),
        correlations={"CL:Vc": 0.30},
        residual_error=(
            ResidualErrorSpec(kind="proportional", sigma_var=0.0225, unit="fraction"),
            ResidualErrorSpec(kind="additive", sigma_var=0.01, unit="mg/L"),
        ),
        predictions=(
            PredictionPoint(time=0.0, pred=10.0, ipred=10.5, unit="mg/L"),
            PredictionPoint(time=4.0, pred=4.2, ipred=4.0, unit="mg/L"),
            PredictionPoint(time=12.0, pred=1.1, ipred=1.0, unit="mg/L"),
        ),
    )


def _find(report: PlausibilityReport, rule_id: str, parameter: str | None = None) -> list[PlausibilityFinding]:
    return [
        f for f in report.findings
        if f.rule_id == rule_id and (parameter is None or f.parameter == parameter)
    ]


# ---------------------------------------------------------------------------
# 1. Catalog & Baseline Plausible Model Contract
# ---------------------------------------------------------------------------

def test_catalog_contains_all_twenty_rules_with_metadata() -> None:
    catalog = load_rule_catalog()
    assert catalog.catalog_version == "1.0.0"
    rule_ids = [r.rule_id for r in catalog.rules]
    expected_ids = [f"PL-{i:03d}" for i in range(1, 21)]
    assert rule_ids == expected_ids
    for rule in catalog.rules:
        assert rule.version
        assert rule.title
        assert rule.category
        assert rule.scientific_rationale
        assert rule.reference_basis
        assert rule.approval.status in {"pending_owner_review", "approved"}


def test_baseline_small_molecule_iv_model_passes_all_applicable_rules() -> None:
    model = _baseline_adult_iv_small_molecule()
    report = evaluate_poppk_plausibility(model)
    assert report.screening_status == "SCREENING_PASS"
    assert "does not constitute regulatory validation" in report.disclaimer
    assert "valid" not in report.screening_status.lower()
    flagged = [f for f in report.findings if f.status == "FLAGGED"]
    assert flagged == []
    # mAb rules PL-017 and PL-018 should be NOT_APPLICABLE for small molecules
    assert _find(report, "PL-017")[0].status == "NOT_APPLICABLE"
    assert _find(report, "PL-018")[0].status == "NOT_APPLICABLE"


# ---------------------------------------------------------------------------
# 2. Unit Conversion & Normalization Tests (PL-002)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("param_name", "value", "unit", "expected_val", "expected_unit"),
    [
        ("CL", 12.0, "L/h", 12.0, "L/h"),
        ("CL", 200.0, "mL/min", 12.0, "L/h"),
        ("CL", 24.0, "L/day", 1.0, "L/h"),
        ("CL", 24.0, "L/d", 1.0, "L/h"),
        ("Vc", 30.0, "L", 30.0, "L"),
        ("Vc", 3000.0, "mL", 3.0, "L"),
        ("Vc", 0.5, "L/kg", 0.5, "L/kg"),
        ("Vc", 50.0, "mL/kg", 0.05, "L/kg"),
        ("ka", 1.5, "1/h", 1.5, "1/h"),
        ("ka", 1.5, "h^-1", 1.5, "1/h"),
        ("F", 0.8, "fraction", 0.8, "fraction"),
        ("F", 0.8, "unitless", 0.8, "fraction"),
        ("F", 80.0, "%", 0.8, "fraction"),
        ("t_half", 10.0, "h", 10.0, "h"),
        ("t_half", 2.0, "d", 48.0, "h"),
        ("t_half", 120.0, "min", 2.0, "h"),
    ],
)
def test_deterministic_unit_conversions(
    param_name: str, value: float, unit: str, expected_val: float, expected_unit: str
) -> None:
    norm_val, norm_unit = normalize_parameter_unit(param_name, value, unit)
    assert norm_unit == expected_unit
    assert math.isclose(norm_val, expected_val, rel_tol=1e-9)


def test_pl002_incompatible_or_unknown_units_flagged_critical() -> None:
    model = PopPKModelInput(
        model_id="bad_units",
        context=PopPKContext(population="adult", reference_weight_kg=70.0, route="iv", modality="small_molecule"),
        parameters=(
            ParameterSpec(name="CL", value=10.0, unit="mg"),  # mass instead of flow
            ParameterSpec(name="Vc", value=20.0, unit="parsec"),  # unknown unit
        ),
    )
    report = evaluate_poppk_plausibility(model)
    cl_u = _find(report, "PL-002", "CL")[0]
    vc_u = _find(report, "PL-002", "Vc")[0]
    assert cl_u.status == "FLAGGED"
    assert cl_u.severity == "CRITICAL"
    assert cl_u.override_allowed is False
    assert vc_u.status == "FLAGGED"
    assert vc_u.severity == "CRITICAL"


def test_pl002_missing_unit_on_dimensional_parameter_returns_unresolved() -> None:
    model = PopPKModelInput(
        model_id="missing_unit",
        context=PopPKContext(population="adult", reference_weight_kg=70.0, route="iv", modality="small_molecule"),
        parameters=(ParameterSpec(name="CL", value=10.0, unit=None),),
    )
    report = evaluate_poppk_plausibility(model)
    finding = _find(report, "PL-002", "CL")[0]
    assert finding.status == "UNRESOLVED"
    # Dependent clearance rule must also be UNRESOLVED rather than false PASS
    assert _find(report, "PL-004")[0].status == "UNRESOLVED"


# ---------------------------------------------------------------------------
# 3. Rule-by-Rule Unit & Boundary Tests (PL-001 through PL-020)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_val", [0.0, -5.0, float("nan"), float("inf"), float("-inf")])
def test_pl001_nonpositive_or_nonfinite_parameter_is_critical(bad_val: float) -> None:
    model = PopPKModelInput(
        model_id="pl001_bad",
        context=PopPKContext(population="adult", reference_weight_kg=70.0, route="iv", modality="small_molecule"),
        parameters=(
            ParameterSpec(name="CL", value=bad_val, unit="L/h"),
            ParameterSpec(name="Vc", value=25.0, unit="L"),
        ),
    )
    report = evaluate_poppk_plausibility(model)
    f = _find(report, "PL-001", "CL")[0]
    assert f.status == "FLAGGED"
    assert f.severity == "CRITICAL"
    assert f.override_allowed is False
    assert report.screening_status == "CRITICAL_IMPLAUSIBILITY"


@pytest.mark.parametrize(
    ("f_val", "f_unit", "expected_status", "expected_severity", "override_ok"),
    [
        (0.75, "fraction", "PASS", "INFO", True),
        (1.0, "fraction", "PASS", "INFO", True),
        (100.0, "%", "PASS", "INFO", True),
        (1.25, "fraction", "FLAGGED", "HIGH", True),  # > 100%, could be relative F if justified
        (80.0, "fraction", "FLAGGED", "HIGH", True),  # 80 given as fraction instead of %
        (0.0, "fraction", "FLAGGED", "CRITICAL", False),
        (-0.2, "fraction", "FLAGGED", "CRITICAL", False),
    ],
)
def test_pl003_bioavailability_bounds_and_percent_handling(
    f_val: float, f_unit: str, expected_status: str, expected_severity: str, override_ok: bool
) -> None:
    model = PopPKModelInput(
        model_id="pl003_test",
        context=PopPKContext(population="adult", reference_weight_kg=70.0, route="oral", modality="small_molecule"),
        parameters=(
            ParameterSpec(name="F", value=f_val, unit=f_unit),
            ParameterSpec(name="CL", value=10.0, unit="L/h", scale="systemic"),
            ParameterSpec(name="Vc", value=30.0, unit="L", scale="systemic"),
            ParameterSpec(name="ka", value=1.2, unit="1/h"),
        ),
    )
    report = evaluate_poppk_plausibility(model)
    finding = _find(report, "PL-003", "F")[0]
    assert finding.status == expected_status
    assert finding.severity == expected_severity
    assert finding.override_allowed is override_ok


def test_pl004_systemic_cl_vs_hepatic_blood_flow_and_clf_disambiguation() -> None:
    # 1. Systemic CL = 90.0 L/h (exact reference adult Q_H boundary) -> PASS
    m_boundary = PopPKModelInput(
        model_id="cl_at_qh",
        context=PopPKContext(population="adult", reference_weight_kg=70.0, route="iv", modality="small_molecule"),
        parameters=(
            ParameterSpec(name="CL", value=90.0, unit="L/h", scale="systemic"),
            ParameterSpec(name="Vc", value=40.0, unit="L", scale="systemic"),
        ),
    )
    assert _find(evaluate_poppk_plausibility(m_boundary), "PL-004")[0].status == "PASS"

    # 2. Systemic CL = 110.0 L/h (> Q_H = 90 L/h, <= cardiac output 300 L/h) -> FLAGGED WARN, override_allowed=True
    m_above_qh = PopPKModelInput(
        model_id="cl_above_qh",
        context=PopPKContext(population="adult", reference_weight_kg=70.0, route="iv", modality="small_molecule"),
        parameters=(
            ParameterSpec(name="CL", value=110.0, unit="L/h", scale="systemic"),
            ParameterSpec(name="Vc", value=40.0, unit="L", scale="systemic"),
        ),
    )
    f_above = _find(evaluate_poppk_plausibility(m_above_qh), "PL-004")[0]
    assert f_above.status == "FLAGGED"
    assert f_above.severity == "WARN"
    assert f_above.override_allowed is True

    # 3. Systemic CL = 350.0 L/h (> cardiac output 300 L/h) -> FLAGGED HIGH
    m_above_co = PopPKModelInput(
        model_id="cl_above_co",
        context=PopPKContext(population="adult", reference_weight_kg=70.0, route="iv", modality="small_molecule"),
        parameters=(
            ParameterSpec(name="CL", value=350.0, unit="L/h", scale="systemic"),
            ParameterSpec(name="Vc", value=40.0, unit="L", scale="systemic"),
        ),
    )
    f_co = _find(evaluate_poppk_plausibility(m_above_co), "PL-004")[0]
    assert f_co.status == "FLAGGED"
    assert f_co.severity == "HIGH"

    # 4. Apparent oral clearance CL/F = 150.0 L/h without F -> NOT_APPLICABLE (or INFO, not falsely flagged as systemic CL > Q_H)
    m_clf = PopPKModelInput(
        model_id="clf_oral",
        context=PopPKContext(population="adult", reference_weight_kg=70.0, route="oral", modality="small_molecule"),
        parameters=(
            ParameterSpec(name="CL/F", value=150.0, unit="L/h", scale="apparent"),
            ParameterSpec(name="Vc/F", value=200.0, unit="L", scale="apparent"),
            ParameterSpec(name="ka", value=1.5, unit="1/h"),
        ),
    )
    f_clf = _find(evaluate_poppk_plausibility(m_clf), "PL-004")[0]
    assert f_clf.status == "NOT_APPLICABLE"


def test_pl005_renal_clearance_filtration_and_renal_plasma_flow() -> None:
    # Adult GFR = 120 mL/min = 7.2 L/h; fu = 0.5 -> fu*GFR = 3.6 L/h; Q_R,plasma = 36 L/h
    # 1. CL_R = 3.0 L/h <= fu*GFR -> PASS
    m_pass = PopPKModelInput(
        model_id="clr_pass",
        context=PopPKContext(
            population="adult",
            reference_weight_kg=70.0,
            route="iv",
            modality="small_molecule",
            fraction_unbound_plasma=0.5,
            egfr_ml_min_1_73m2=120.0,
        ),
        parameters=(
            ParameterSpec(name="CL", value=10.0, unit="L/h"),
            ParameterSpec(name="CL_R", value=3.0, unit="L/h"),
            ParameterSpec(name="Vc", value=30.0, unit="L"),
        ),
    )
    assert _find(evaluate_poppk_plausibility(m_pass), "PL-005")[0].status == "PASS"

    # 2. CL_R = 10.0 L/h > fu*GFR (3.6 L/h) without documented secretion -> FLAGGED WARN
    m_sec = PopPKModelInput(
        model_id="clr_sec",
        context=PopPKContext(
            population="adult",
            reference_weight_kg=70.0,
            route="iv",
            modality="small_molecule",
            fraction_unbound_plasma=0.5,
            egfr_ml_min_1_73m2=120.0,
            renal_secretion_known=False,
        ),
        parameters=(
            ParameterSpec(name="CL", value=15.0, unit="L/h"),
            ParameterSpec(name="CL_R", value=10.0, unit="L/h"),
            ParameterSpec(name="Vc", value=30.0, unit="L"),
        ),
    )
    f_sec = _find(evaluate_poppk_plausibility(m_sec), "PL-005")[0]
    assert f_sec.status == "FLAGGED"
    assert f_sec.severity == "WARN"

    # 3. CL_R = 45.0 L/h > Q_R,plasma (36.0 L/h) -> FLAGGED HIGH even if secretion is known
    m_qr = PopPKModelInput(
        model_id="clr_qr",
        context=PopPKContext(
            population="adult",
            reference_weight_kg=70.0,
            route="iv",
            modality="small_molecule",
            fraction_unbound_plasma=0.5,
            egfr_ml_min_1_73m2=120.0,
            renal_secretion_known=True,
        ),
        parameters=(
            ParameterSpec(name="CL", value=60.0, unit="L/h"),
            ParameterSpec(name="CL_R", value=45.0, unit="L/h"),
            ParameterSpec(name="Vc", value=30.0, unit="L"),
        ),
    )
    f_qr = _find(evaluate_poppk_plausibility(m_qr), "PL-005")[0]
    assert f_qr.status == "FLAGGED"
    assert f_qr.severity == "HIGH"


def test_pl006_central_volume_physiological_bounds_and_weight_normalization() -> None:
    # 70 kg adult: plasma floor 0.03 L/kg = 2.1 L.
    # 1. Vc = 1.0 L (< 2.1 L) -> FLAGGED HIGH
    m_low = PopPKModelInput(
        model_id="vc_low",
        context=PopPKContext(population="adult", reference_weight_kg=70.0, route="iv", modality="small_molecule"),
        parameters=(
            ParameterSpec(name="CL", value=5.0, unit="L/h"),
            ParameterSpec(name="Vc", value=1.0, unit="L"),
        ),
    )
    f_low = _find(evaluate_poppk_plausibility(m_low), "PL-006")[0]
    assert f_low.status == "FLAGGED"
    assert f_low.severity == "HIGH"

    # 2. Weight-normalized Vc = 0.02 L/kg (< 0.03 L/kg) -> FLAGGED HIGH
    m_low_wt = PopPKModelInput(
        model_id="vc_low_wt",
        context=PopPKContext(population="adult", reference_weight_kg=70.0, route="iv", modality="small_molecule"),
        parameters=(
            ParameterSpec(name="CL", value=5.0, unit="L/h"),
            ParameterSpec(name="Vc", value=0.02, unit="L/kg"),
        ),
    )
    assert _find(evaluate_poppk_plausibility(m_low_wt), "PL-006")[0].status == "FLAGGED"

    # 3. Weight-normalized Vc = 0.5 L/kg (35 L in 70 kg adult) -> PASS
    m_ok_wt = PopPKModelInput(
        model_id="vc_ok_wt",
        context=PopPKContext(population="adult", reference_weight_kg=70.0, route="iv", modality="small_molecule"),
        parameters=(
            ParameterSpec(name="CL", value=5.0, unit="L/h"),
            ParameterSpec(name="Vc", value=0.5, unit="L/kg"),
        ),
    )
    assert _find(evaluate_poppk_plausibility(m_ok_wt), "PL-006")[0].status == "PASS"


def test_pl007_vss_physiological_spaces_and_compartment_sum_consistency() -> None:
    # Mismatched Vss != Vc + Vp (Vc=20 L, Vp=30 L, Vss=80 L) -> FLAGGED HIGH
    m_mismatch = PopPKModelInput(
        model_id="vss_mismatch",
        context=PopPKContext(population="adult", reference_weight_kg=70.0, route="iv", modality="small_molecule"),
        parameters=(
            ParameterSpec(name="CL", value=10.0, unit="L/h"),
            ParameterSpec(name="Vc", value=20.0, unit="L"),
            ParameterSpec(name="Vp", value=30.0, unit="L"),
            ParameterSpec(name="Vss", value=80.0, unit="L"),
            ParameterSpec(name="Q", value=10.0, unit="L/h"),
        ),
    )
    f_mis = _find(evaluate_poppk_plausibility(m_mismatch), "PL-007")[0]
    assert f_mis.status == "FLAGGED"
    assert f_mis.severity == "HIGH"


def test_pl008_half_life_consistency_with_cl_and_v() -> None:
    # 1-cmt: CL = 10 L/h, Vc = 50 L -> expected t_half = ln(2) * 5 = 3.4657 h.
    # Supplying t_half = 12.0 h -> FLAGGED HIGH
    m_bad_thalf = PopPKModelInput(
        model_id="bad_thalf",
        context=PopPKContext(population="adult", reference_weight_kg=70.0, route="iv", modality="small_molecule"),
        parameters=(
            ParameterSpec(name="CL", value=10.0, unit="L/h"),
            ParameterSpec(name="Vc", value=50.0, unit="L"),
            ParameterSpec(name="t_half", value=12.0, unit="h"),
        ),
    )
    f_bad = _find(evaluate_poppk_plausibility(m_bad_thalf), "PL-008")[0]
    assert f_bad.status == "FLAGGED"
    assert f_bad.severity == "HIGH"

    # Supplying consistent t_half = 3.465736 h -> PASS
    m_good_thalf = PopPKModelInput(
        model_id="good_thalf",
        context=PopPKContext(population="adult", reference_weight_kg=70.0, route="iv", modality="small_molecule"),
        parameters=(
            ParameterSpec(name="CL", value=10.0, unit="L/h"),
            ParameterSpec(name="Vc", value=50.0, unit="L"),
            ParameterSpec(name="t_half", value=math.log(2.0) * 5.0, unit="h"),
        ),
    )
    assert _find(evaluate_poppk_plausibility(m_good_thalf), "PL-008")[0].status == "PASS"


def test_pl009_ka_plausibility_and_flip_flop_detection() -> None:
    # ke = CL / Vc = 20 / 10 = 2.0 1/h; ka = 0.5 1/h <= ke -> flip-flop kinetics WARN
    m_flipflop = PopPKModelInput(
        model_id="flip_flop",
        context=PopPKContext(population="adult", reference_weight_kg=70.0, route="oral", modality="small_molecule"),
        parameters=(
            ParameterSpec(name="CL", value=20.0, unit="L/h"),
            ParameterSpec(name="Vc", value=10.0, unit="L"),
            ParameterSpec(name="ka", value=0.5, unit="1/h"),
        ),
    )
    f_ff = _find(evaluate_poppk_plausibility(m_flipflop), "PL-009", "ka")[0]
    assert f_ff.status == "FLAGGED"
    assert f_ff.severity == "WARN"
    assert "flip-flop" in f_ff.scientific_rationale.lower()


@pytest.mark.parametrize(
    ("cl_exp", "expected_status", "expected_severity"),
    [
        (0.75, "PASS", "INFO"),
        (0.60, "PASS", "INFO"),
        (0.90, "PASS", "INFO"),
        (0.50, "FLAGGED", "WARN"),
        (1.05, "FLAGGED", "WARN"),
        (0.10, "FLAGGED", "HIGH"),
        (-0.25, "FLAGGED", "HIGH"),
        (1.50, "FLAGGED", "HIGH"),
    ],
)
def test_pl010_allometric_clearance_exponent_boundaries(
    cl_exp: float, expected_status: str, expected_severity: str
) -> None:
    model = PopPKModelInput(
        model_id="allo_cl",
        context=PopPKContext(population="adult", reference_weight_kg=70.0, route="iv", modality="small_molecule"),
        parameters=(
            ParameterSpec(name="CL", value=10.0, unit="L/h"),
            ParameterSpec(name="Vc", value=30.0, unit="L"),
        ),
        allometric_exponents={"CL": cl_exp},
    )
    f = _find(evaluate_poppk_plausibility(model), "PL-010", "CL")[0]
    assert f.status == expected_status
    assert f.severity == expected_severity


@pytest.mark.parametrize(
    ("v_exp", "expected_status", "expected_severity"),
    [
        (1.00, "PASS", "INFO"),
        (0.80, "PASS", "INFO"),
        (1.20, "PASS", "INFO"),
        (0.70, "FLAGGED", "WARN"),
        (1.35, "FLAGGED", "WARN"),
        (0.30, "FLAGGED", "HIGH"),
        (1.80, "FLAGGED", "HIGH"),
    ],
)
def test_pl011_allometric_volume_exponent_boundaries(
    v_exp: float, expected_status: str, expected_severity: str
) -> None:
    model = PopPKModelInput(
        model_id="allo_v",
        context=PopPKContext(population="adult", reference_weight_kg=70.0, route="iv", modality="small_molecule"),
        parameters=(
            ParameterSpec(name="CL", value=10.0, unit="L/h"),
            ParameterSpec(name="Vc", value=30.0, unit="L"),
        ),
        allometric_exponents={"Vc": v_exp},
    )
    f = _find(evaluate_poppk_plausibility(model), "PL-011", "Vc")[0]
    assert f.status == expected_status
    assert f.severity == expected_severity


def test_pl012_and_pl013_iiv_magnitude_and_negative_variance() -> None:
    # High IIV (omega_var = 1.0 -> CV% = 131.1% > 100%) -> PL-012 WARN, override_allowed=True
    m_high_iiv = PopPKModelInput(
        model_id="high_iiv",
        context=PopPKContext(population="adult", reference_weight_kg=70.0, route="iv", modality="small_molecule"),
        parameters=(
            ParameterSpec(name="CL", value=10.0, unit="L/h"),
            ParameterSpec(name="Vc", value=30.0, unit="L"),
        ),
        iiv=(IIVSpec(parameter="CL", omega_var=1.0),),
    )
    r_high = evaluate_poppk_plausibility(m_high_iiv)
    f12 = _find(r_high, "PL-012", "CL")[0]
    assert f12.status == "FLAGGED"
    assert f12.severity == "WARN"
    assert f12.override_allowed is True

    # Negative variance (omega_var = -0.05) -> PL-013 CRITICAL, override_allowed=False
    m_neg_var = PopPKModelInput(
        model_id="neg_var",
        context=PopPKContext(population="adult", reference_weight_kg=70.0, route="iv", modality="small_molecule"),
        parameters=(
            ParameterSpec(name="CL", value=10.0, unit="L/h"),
            ParameterSpec(name="Vc", value=30.0, unit="L"),
        ),
        iiv=(IIVSpec(parameter="CL", omega_var=-0.05),),
    )
    r_neg = evaluate_poppk_plausibility(m_neg_var)
    f13 = _find(r_neg, "PL-013", "CL")[0]
    assert f13.status == "FLAGGED"
    assert f13.severity == "CRITICAL"
    assert f13.override_allowed is False


def test_pl014_and_pl015_omega_psd_and_correlation_bounds() -> None:
    # Non-PSD Omega matrix: [[0.09, 0.20], [0.20, 0.04]] -> r = 0.20 / 0.06 = 3.33 > 1, det < 0
    m_non_psd = PopPKModelInput(
        model_id="non_psd",
        context=PopPKContext(population="adult", reference_weight_kg=70.0, route="iv", modality="small_molecule"),
        parameters=(
            ParameterSpec(name="CL", value=10.0, unit="L/h"),
            ParameterSpec(name="Vc", value=30.0, unit="L"),
        ),
        omega_matrix=((0.09, 0.20), (0.20, 0.04)),
        correlations={"CL:Vc": 1.25},
    )
    report = evaluate_poppk_plausibility(m_non_psd)
    f14 = _find(report, "PL-014")[0]
    f15 = _find(report, "PL-015", "CL:Vc")[0]
    assert f14.status == "FLAGGED"
    assert f14.severity == "CRITICAL"
    assert f14.override_allowed is False
    assert f15.status == "FLAGGED"
    assert f15.severity == "CRITICAL"
    assert f15.override_allowed is False


def test_pl016_residual_error_checks() -> None:
    # Negative residual variance -> CRITICAL
    m_neg_res = PopPKModelInput(
        model_id="neg_res",
        context=PopPKContext(population="adult", reference_weight_kg=70.0, route="iv", modality="small_molecule"),
        parameters=(
            ParameterSpec(name="CL", value=10.0, unit="L/h"),
            ParameterSpec(name="Vc", value=30.0, unit="L"),
        ),
        residual_error=(ResidualErrorSpec(kind="proportional", sigma_var=-0.01, unit="fraction"),),
    )
    f16 = _find(evaluate_poppk_plausibility(m_neg_res), "PL-016")[0]
    assert f16.status == "FLAGGED"
    assert f16.severity == "CRITICAL"
    assert f16.override_allowed is False


def test_pl017_and_pl018_monoclonal_antibody_specific_rules() -> None:
    # 1. Plausible mAb: Vc = 3.5 L (0.05 L/kg), CL = 0.012 L/h (0.288 L/d) -> PASS
    m_mab_ok = PopPKModelInput(
        model_id="mab_ok",
        context=PopPKContext(
            population="adult",
            reference_weight_kg=70.0,
            route="iv",
            modality="monoclonal_antibody",
        ),
        parameters=(
            ParameterSpec(name="CL", value=0.012, unit="L/h", scale="systemic"),
            ParameterSpec(name="Vc", value=3.5, unit="L", scale="systemic"),
            ParameterSpec(name="Vp", value=2.8, unit="L", scale="systemic"),
            ParameterSpec(name="Q", value=0.03, unit="L/h", scale="systemic"),
        ),
    )
    r_ok = evaluate_poppk_plausibility(m_mab_ok)
    assert _find(r_ok, "PL-017")[0].status == "PASS"
    assert _find(r_ok, "PL-018")[0].status == "PASS"

    # 2. Implausible mAb with small-molecule-like Vc = 45 L and CL = 8.0 L/h -> FLAGGED HIGH on both PL-017 and PL-018
    m_mab_bad = PopPKModelInput(
        model_id="mab_bad",
        context=PopPKContext(
            population="adult",
            reference_weight_kg=70.0,
            route="iv",
            modality="monoclonal_antibody",
        ),
        parameters=(
            ParameterSpec(name="CL", value=8.0, unit="L/h", scale="systemic"),
            ParameterSpec(name="Vc", value=45.0, unit="L", scale="systemic"),
        ),
    )
    r_bad = evaluate_poppk_plausibility(m_mab_bad)
    f17 = _find(r_bad, "PL-017")[0]
    f18 = _find(r_bad, "PL-018")[0]
    assert f17.status == "FLAGGED"
    assert f17.severity == "HIGH"
    assert f18.status == "FLAGGED"
    assert f18.severity == "HIGH"


def test_pl019_intercompartmental_micro_rate_consistency() -> None:
    # Q = 15 L/h, Vc = 25 L -> k12 should be 0.6 1/h; Vp = 30 L -> k21 should be 0.5 1/h.
    # Supplying k12 = 1.2 1/h -> FLAGGED HIGH
    m_micro = PopPKModelInput(
        model_id="micro_mismatch",
        context=PopPKContext(population="adult", reference_weight_kg=70.0, route="iv", modality="small_molecule"),
        parameters=(
            ParameterSpec(name="CL", value=10.0, unit="L/h"),
            ParameterSpec(name="Vc", value=25.0, unit="L"),
            ParameterSpec(name="Vp", value=30.0, unit="L"),
            ParameterSpec(name="Q", value=15.0, unit="L/h"),
            ParameterSpec(name="k12", value=1.2, unit="1/h"),
            ParameterSpec(name="k21", value=0.5, unit="1/h"),
        ),
    )
    f19 = _find(evaluate_poppk_plausibility(m_micro), "PL-019")[0]
    assert f19.status == "FLAGGED"
    assert f19.severity == "HIGH"


def test_pl020_implausible_or_nonfinite_predictions() -> None:
    m_pred = PopPKModelInput(
        model_id="bad_pred",
        context=PopPKContext(population="adult", reference_weight_kg=70.0, route="iv", modality="small_molecule"),
        parameters=(
            ParameterSpec(name="CL", value=10.0, unit="L/h"),
            ParameterSpec(name="Vc", value=25.0, unit="L"),
        ),
        predictions=(
            PredictionPoint(time=1.0, pred=-0.5, ipred=1.2, unit="mg/L"),
            PredictionPoint(time=2.0, pred=float("nan"), ipred=0.8, unit="mg/L"),
        ),
    )
    f20 = _find(evaluate_poppk_plausibility(m_pred), "PL-020")[0]
    assert f20.status == "FLAGGED"
    assert f20.severity == "CRITICAL"
    assert f20.override_allowed is False


# ---------------------------------------------------------------------------
# 4. Population & Missing-Context Tests (Pediatric, Renal, Hepatic, Pregnancy, Unknown)
# ---------------------------------------------------------------------------

def test_pediatric_and_missing_context_return_unresolved_not_false_pass() -> None:
    # Pediatric population WITHOUT reference_weight_kg must return UNRESOLVED for weight-scaled rules
    m_ped_no_wt = PopPKModelInput(
        model_id="ped_no_wt",
        context=PopPKContext(
            population="pediatric",
            reference_weight_kg=None,
            route="iv",
            modality="small_molecule",
        ),
        parameters=(
            ParameterSpec(name="CL", value=8.0, unit="L/h", scale="systemic"),
            ParameterSpec(name="Vc", value=12.0, unit="L", scale="systemic"),
        ),
    )
    r_no_wt = evaluate_poppk_plausibility(m_ped_no_wt)
    assert _find(r_no_wt, "PL-004")[0].status == "UNRESOLVED"
    assert _find(r_no_wt, "PL-006")[0].status == "UNRESOLVED"
    assert r_no_wt.screening_status == "INCOMPLETE_CONTEXT"

    # Pediatric population WITH weight=10 kg: scaled Q_H = 90 * (10/70)^0.75 = 20.91 L/h
    # CL = 30.0 L/h exceeds pediatric Q_H (even though < 90 L/h adult Q_H!) -> FLAGGED
    m_ped_10kg = PopPKModelInput(
        model_id="ped_10kg",
        context=PopPKContext(
            population="pediatric",
            reference_weight_kg=10.0,
            route="iv",
            modality="small_molecule",
        ),
        parameters=(
            ParameterSpec(name="CL", value=30.0, unit="L/h", scale="systemic"),
            ParameterSpec(name="Vc", value=4.0, unit="L", scale="systemic"),
        ),
    )
    r_10kg = evaluate_poppk_plausibility(m_ped_10kg)
    assert _find(r_10kg, "PL-004")[0].status == "FLAGGED"


def test_hepatic_impairment_and_pregnancy_context_handling() -> None:
    # Child-Pugh C reduces effective hepatic flow threshold (0.5 * 90 = 45 L/h), so CL = 60 L/h is flagged
    m_hepatic = PopPKModelInput(
        model_id="hepatic_c",
        context=PopPKContext(
            population="hepatic_impairment",
            reference_weight_kg=70.0,
            child_pugh="C",
            route="iv",
            modality="small_molecule",
        ),
        parameters=(
            ParameterSpec(name="CL", value=60.0, unit="L/h", scale="systemic"),
            ParameterSpec(name="Vc", value=30.0, unit="L", scale="systemic"),
        ),
    )
    assert _find(evaluate_poppk_plausibility(m_hepatic), "PL-004")[0].status == "FLAGGED"

    # Pregnancy without gestational_age_weeks returns UNRESOLVED on physiological flow/volume bounds
    m_preg_unknown = PopPKModelInput(
        model_id="preg_unknown",
        context=PopPKContext(
            population="pregnancy",
            reference_weight_kg=70.0,
            gestational_age_weeks=None,
            route="iv",
            modality="small_molecule",
        ),
        parameters=(
            ParameterSpec(name="CL", value=15.0, unit="L/h", scale="systemic"),
            ParameterSpec(name="Vc", value=30.0, unit="L", scale="systemic"),
        ),
    )
    assert _find(evaluate_poppk_plausibility(m_preg_unknown), "PL-004")[0].status == "UNRESOLVED"


def test_unknown_modality_returns_unresolved_for_mab_rules() -> None:
    m_unk_mod = PopPKModelInput(
        model_id="unk_modality",
        context=PopPKContext(
            population="adult",
            reference_weight_kg=70.0,
            route="iv",
            modality="unknown",
        ),
        parameters=(
            ParameterSpec(name="CL", value=5.0, unit="L/h", scale="systemic"),
            ParameterSpec(name="Vc", value=25.0, unit="L", scale="systemic"),
        ),
    )
    r_unk = evaluate_poppk_plausibility(m_unk_mod)
    assert _find(r_unk, "PL-017")[0].status == "UNRESOLVED"
    assert _find(r_unk, "PL-018")[0].status == "UNRESOLVED"


# ---------------------------------------------------------------------------
# 5. Human Override / Justification Governance Tests
# ---------------------------------------------------------------------------

def test_valid_justification_transitions_override_status_and_preserves_finding() -> None:
    justification = JustificationRecord(
        rule_id="PL-004",
        parameter="CL",
        rationale="Extrahepatic blood esterase hydrolysis accounts for clearance exceeding hepatic blood flow.",
        scientific_basis_or_citation="Remifentanil clinical pharmacology package insert; Egan 1995.",
        reviewer="Dr. Jane Doe",
        timestamp="2026-10-22T12:00:00Z",
    )
    model = PopPKModelInput(
        model_id="remifentanil_like",
        context=PopPKContext(population="adult", reference_weight_kg=70.0, route="iv", modality="small_molecule"),
        parameters=(
            ParameterSpec(name="CL", value=150.0, unit="L/h", scale="systemic"),
            ParameterSpec(name="Vc", value=10.0, unit="L", scale="systemic"),
        ),
        justifications=(justification,),
    )
    report = evaluate_poppk_plausibility(model)
    finding = _find(report, "PL-004", "CL")[0]
    # Finding must remain visible as FLAGGED (never silently erased)
    assert finding.status == "FLAGGED"
    assert finding.override_status == "accepted_with_justification"
    assert finding.justification is not None
    assert finding.justification.reviewer == "Dr. Jane Doe"
    assert report.screening_status == "REVIEW_ACCEPTED_WITH_JUSTIFICATION"
    assert report.summary_counts["accepted_with_justification"] == 1
    assert report.summary_counts["justification_required"] == 0


def test_hard_mathematical_impossibility_cannot_be_overridden() -> None:
    justification = JustificationRecord(
        rule_id="PL-013",
        parameter="CL",
        rationale="Attempted override of negative variance.",
        scientific_basis_or_citation="None",
        reviewer="Dr. Jane Doe",
        timestamp="2026-10-22T12:00:00Z",
    )
    model = PopPKModelInput(
        model_id="illegal_override",
        context=PopPKContext(population="adult", reference_weight_kg=70.0, route="iv", modality="small_molecule"),
        parameters=(
            ParameterSpec(name="CL", value=10.0, unit="L/h"),
            ParameterSpec(name="Vc", value=25.0, unit="L"),
        ),
        iiv=(IIVSpec(parameter="CL", omega_var=-0.1),),
        justifications=(justification,),
    )
    report = evaluate_poppk_plausibility(model)
    finding = _find(report, "PL-013", "CL")[0]
    assert finding.override_allowed is False
    assert finding.override_status == "rejected"
    assert finding.status == "FLAGGED"
    assert report.screening_status == "CRITICAL_IMPLAUSIBILITY"


# ---------------------------------------------------------------------------
# 6. Determinism, Order Invariance & Non-Mutation Property Tests
# ---------------------------------------------------------------------------

def test_determinism_order_invariance_and_input_immutability() -> None:
    raw_dict = {
        "model_id": "order_test",
        "context": {
            "species": "human",
            "population": "adult",
            "reference_weight_kg": 70.0,
            "route": "iv",
            "modality": "small_molecule",
        },
        "parameters": [
            {"name": "Vc", "value": 30000.0, "unit": "mL", "scale": "systemic"},
            {"name": "CL", "value": 200.0, "unit": "mL/min", "scale": "systemic"},
            {"name": "Vp", "value": 40.0, "unit": "L", "scale": "systemic"},
            {"name": "Q", "value": 10.0, "unit": "L/h", "scale": "systemic"},
        ],
        "allometric_exponents": {"Vc": 1.0, "CL": 0.75},
        "iiv": [
            {"parameter": "Vc", "omega_var": 0.04},
            {"parameter": "CL", "omega_var": 0.09},
        ],
    }
    snapshot = copy.deepcopy(raw_dict)
    reordered = copy.deepcopy(raw_dict)
    reordered["parameters"] = list(reversed(reordered["parameters"]))
    reordered["iiv"] = list(reversed(reordered["iiv"]))

    r1 = evaluate_poppk_plausibility(raw_dict)
    r2 = evaluate_poppk_plausibility(reordered)

    # Caller's raw dictionary was never mutated
    assert raw_dict == snapshot
    # Normalized values were not written back into raw_dict
    assert raw_dict["parameters"][0]["value"] == 30000.0

    # Output reports and SHA-256 digests are identical regardless of input ordering
    assert r1.report_sha256 == r2.report_sha256
    assert r1.to_dict() == r2.to_dict()
    assert r1.to_markdown() == r2.to_markdown()
