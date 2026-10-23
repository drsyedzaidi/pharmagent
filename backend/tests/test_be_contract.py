"""Average-BE safety contract (slice 1): explicit design/reference/test/profile, fail-closed validation, regulatory
rounding with decimal arithmetic, an exclusion ledger, and the minimum number of evaluable subjects.

Sources (read from the downloaded texts, 2026-10-23):
- FDA, Statistical Approaches to Establishing Bioequivalence (May 2026), sec. II.A footnote 6: "the rounded confidence
  interval value should be at least 80.00 percent and not more than 125.00 percent".
- EMA CPMP/EWP/QWP/1401/98 Rev. 1/Corr**, sec. 4.1.8: bounds compared "when rounded to two decimal places".
- ICH M13A (EMA/CHMP/ICH/953493/2022), sec. 2.2.3.1: fewer than 12 evaluable subjects (crossover, or per arm for a
  parallel design) is not acceptable; sec. 2.2.3.2: crossover analysis includes subjects with evaluable data for both
  products. M13A states the 80.00-125.00% range but no rounding rule.
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from app.compute.be_contract import (
    PROFILES,
    BEContractError,
    assess_average_be,
    regulatory_round,
)


def _rec(subject, treatment, cmax):
    return {"subject": subject, "treatment": treatment, "params": {"Cmax": cmax}}


def _crossover(n=12, ratio=1.0, ref=100.0):
    recs = []
    for i in range(n):
        r = ref + i
        recs += [_rec(f"S{i}", "R", r), _rec(f"S{i}", "T", r * ratio * (1.0 + (0.01 if i % 2 else -0.01)))]
    return recs


def _parallel(n=12):
    return ([_rec(f"T{i}", "T", 100.0 + i) for i in range(n)]
            + [_rec(f"R{i}", "R", 100.0 + i) for i in range(n)])


def _run(recs, **kw):
    args = {"design": "crossover", "profile": "fda_2026", "test": "T", "reference": "R"}
    args.update(kw)
    return assess_average_be(recs, **args)


# ── profiles ───────────────────────────────────────────────────────────────────────────────────
def test_the_three_profiles_exist_and_ema_2010_is_separate():
    assert set(PROFILES) == {"fda_2026", "ich_m13a", "ema_2010"}
    for p in PROFILES.values():
        assert (p.lower, p.upper) == (Decimal("80.00"), Decimal("125.00"))
        assert p.ci_percent == 90 and p.min_evaluable == 12 and p.source


def test_only_profiles_whose_text_states_rounding_claim_it():
    assert PROFILES["fda_2026"].rounding_source and PROFILES["ema_2010"].rounding_source
    assert PROFILES["ich_m13a"].rounding_source is None            # M13A states no rounding rule


# ── fail closed on missing or contradictory arguments ──────────────────────────────────────────
@pytest.mark.parametrize("missing", ["design", "profile", "test", "reference"])
def test_each_required_argument_is_required(missing):
    with pytest.raises(BEContractError, match=missing):
        _run(_crossover(), **{missing: None})


@pytest.mark.parametrize("bad", [{"design": "replicate"}, {"profile": "who_2017"}, {"test": "R"},
                                 {"reference": "X"}, {"test": "X"}])
def test_contradictory_or_unknown_arguments_fail_closed(bad):
    with pytest.raises(BEContractError):
        _run(_crossover(), **bad)


def test_custom_limits_need_a_written_justification():
    with pytest.raises(BEContractError, match="justification"):
        _run(_crossover(), limits=(90.0, 111.11))
    out = _run(_crossover(), limits=(90.0, 111.11), limits_justification="NTI per protocol section 9.2")
    assert out["limits"] == ["90.00", "111.11"] and out["limits_justification"] == "NTI per protocol section 9.2"


def test_a_replicate_crossover_is_refused_not_averaged():
    recs = _crossover() + [_rec("S0", "R", 101.0)]                 # S0 received R twice
    with pytest.raises(BEContractError, match="replicate"):
        _run(recs)


def test_a_parallel_subject_in_both_arms_is_refused():
    recs = _parallel() + [_rec("T0", "R", 100.0)]
    with pytest.raises(BEContractError, match="both"):
        _run(recs, design="parallel")


# ── never switch design ────────────────────────────────────────────────────────────────────────
def test_one_incomplete_crossover_subject_is_excluded_and_the_design_stays_crossover():
    recs = _crossover(13) + [_rec("S99", "T", 100.0)]              # S99 has no reference period
    out = _run(recs)
    assert out["design"] == "crossover"
    p = out["parameters"]["Cmax"]
    assert p["n_evaluable"] == 13
    assert {"subject": "S99", "treatment": "T", "parameter": "Cmax", "reason": "no_evaluable_counterpart",
            "original_value": 100.0, "prespecified": False} in out["exclusions"]


# ── exclusion ledger ───────────────────────────────────────────────────────────────────────────
def test_every_dropped_value_is_in_the_ledger_with_its_original_value():
    recs = _crossover(14)
    recs[0]["params"]["Cmax"] = 0.0                                # S0 R non-positive
    recs[3]["params"]["Cmax"] = None                               # S1 T missing
    out = _run(recs)
    led = {(e["subject"], e["treatment"], e["reason"]): e for e in out["exclusions"]}
    assert led[("S0", "R", "non_positive_value")]["original_value"] == 0.0
    assert led[("S1", "T", "missing_value")]["original_value"] is None
    # the partner of each is excluded too, with its own reason and value
    assert ("S0", "T", "no_evaluable_counterpart") in led and ("S1", "R", "no_evaluable_counterpart") in led
    assert out["parameters"]["Cmax"]["n_evaluable"] == 12


def test_prespecified_exclusions_are_applied_and_marked_prespecified():
    out = _run(_crossover(13), prespecified_exclusions=[{"subject": "S4", "reason": "vomiting (protocol 7.3)"}])
    s4 = [e for e in out["exclusions"] if e["subject"] == "S4"]
    assert len(s4) == 2 and all(e["prespecified"] and e["reason"] == "vomiting (protocol 7.3)" for e in s4)
    assert out["parameters"]["Cmax"]["n_evaluable"] == 12


def test_a_prespecified_exclusion_must_name_a_known_subject_and_a_reason():
    with pytest.raises(BEContractError):
        _run(_crossover(), prespecified_exclusions=[{"subject": "NOPE", "reason": "x"}])
    with pytest.raises(BEContractError):
        _run(_crossover(), prespecified_exclusions=[{"subject": "S1"}])


def test_other_treatment_arms_are_left_out_and_reported():
    recs = _crossover() + [_rec(f"S{i}", "R2", 100.0) for i in range(12)]
    out = _run(recs)
    assert out["ignored_treatments"] == ["R2"] and out["parameters"]["Cmax"]["n_evaluable"] == 12


# ── minimum evaluable subjects ─────────────────────────────────────────────────────────────────
def test_eleven_evaluable_crossover_subjects_cannot_pass():
    p = _run(_crossover(11))["parameters"]["Cmax"]
    assert p["within_limits"] is False and p["acceptance"] == "insufficient_evaluable_subjects"


def test_twelve_evaluable_crossover_subjects_can_pass():
    out = _run(_crossover(12))
    assert out["parameters"]["Cmax"]["acceptance"] == "within_limits" and out["bioequivalent"] is True


def test_parallel_needs_twelve_per_arm():
    recs = _parallel(12)
    recs = [r for r in recs if r["subject"] != "R11"]
    p = _run(recs, design="parallel")["parameters"]["Cmax"]
    assert p["acceptance"] == "insufficient_evaluable_subjects" and p["n_reference"] == 11


# ── decimal rounding at the boundary ───────────────────────────────────────────────────────────
@pytest.mark.parametrize("value, expected", [
    (79.995, "80.00"), (79.994999, "79.99"), (125.004999, "125.00"), (125.005, "125.01"),
    (80.0, "80.00"), (124.999999999, "125.00"), (1.005, "1.01"),     # 1.005 is 1.00499.. in binary
])
def test_regulatory_rounding_is_decimal_half_up_on_the_shortest_repr(value, expected):
    assert regulatory_round(value) == Decimal(expected)


@pytest.mark.parametrize("lo, hi, passes", [
    (79.995, 110.0, True), (79.994999, 110.0, False), (90.0, 125.004999, True), (90.0, 125.005, False),
])
def test_the_verdict_uses_the_rounded_bounds(monkeypatch, lo, hi, passes):
    import app.compute.be_contract as bc

    monkeypatch.setattr(bc, "_estimate", lambda *a, **k: {"gmr_pct": 100.0, "ci_lower_pct": lo,
                                                            "ci_upper_pct": hi, "cv_intra_pct": 10.0})
    p = _run(_crossover())["parameters"]["Cmax"]
    assert p["within_limits"] is passes
    assert p["ci_lower_rounded"] == str(regulatory_round(lo)) and p["ci_upper_rounded"] == str(regulatory_round(hi))


def test_the_m13a_profile_compares_unrounded_bounds_and_says_so(monkeypatch):
    import app.compute.be_contract as bc

    monkeypatch.setattr(bc, "_estimate", lambda *a, **k: {"gmr_pct": 100.0, "ci_lower_pct": 79.995,
                                                            "ci_upper_pct": 110.0, "cv_intra_pct": 10.0})
    out = _run(_crossover(), profile="ich_m13a")
    assert out["parameters"]["Cmax"]["within_limits"] is False
    assert out["rounding"] == "none (the profile's source states no rounding rule)"


# ── the result says what is still interim ──────────────────────────────────────────────────────
def test_the_result_names_the_interim_model():
    out = _run(_crossover())
    assert "interim" in out["model"] and "period" in out["model"]
