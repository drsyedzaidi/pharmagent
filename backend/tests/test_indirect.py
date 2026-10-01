"""Bucher adjusted indirect comparison: closed-form maths, checked by hand.

Method (Bucher et al. 1997): given effects of A vs B and of C vs B, the indirect
effect of A vs C is the DIFFERENCE of the two effects on the analysis scale
(log for ratio measures, identity for differences) and the variances ADD:

    d_AC = d_AB - d_CB            var(d_AC) = se_AB^2 + se_CB^2

A 100(1-alpha)% CI is d_AC +/- z * se, z = 1.959964 at 95%, and the two-sided
p-value tests d_AC = 0 (ratio 1 on the natural scale): p = erfc(|z_stat| / sqrt(2)).

Every expected number below is derived by hand in the comment above its test;
the assertions use closed forms (math.sqrt / math.erfc / math.exp), never the
function under test.
"""
import json
import math

import pytest

from app.compute.indirect import (
    bucher_indirect,
    ci_from_se,
    normalize_scale,
    se_from_ci,
)

Z95 = 1.959964          # standard-normal 97.5th percentile
Z90 = 1.644854          # 95th percentile


def _p_two_sided(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2.0))


# ── hand-derived worked example, ratio scale ───────────────────────────────────
# A vs B: HR 0.50, SE(log HR) 0.20.   C vs B: HR 0.80, SE(log HR) 0.30.
#   d_AC  = ln 0.50 - ln 0.80 = -0.693147 + 0.223144 = -0.470004
#   HR_AC = exp(d_AC) = 0.50 / 0.80 = 0.625   (exactly)
#   SE    = sqrt(0.20^2 + 0.30^2) = sqrt(0.04 + 0.09) = sqrt(0.13) = 0.360555
#   CI    = 0.625 * exp(-/+ 1.959964 * 0.360555) = 0.625 * exp(-/+ 0.706680)
#         = 0.625 / 2.02728  ..  0.625 * 2.02728 = (0.3083, 1.2670)
#   z     = -0.470004 / 0.360555 = -1.30356   p = 2 * (1 - Phi(1.30356)) = 0.1924
def test_ratio_scale_worked_example_by_hand():
    r = bucher_indirect(
        {"estimate": 0.50, "se": 0.20},
        {"estimate": 0.80, "se": 0.30},
        scale="HR")
    assert r["status"] == "ok"
    assert r["scale"] == "HR" and r["scale_type"] == "ratio"
    assert r["estimate"] == pytest.approx(0.625, abs=1e-12)
    assert r["se"] == pytest.approx(math.sqrt(0.13), abs=1e-12)
    assert r["analysis_estimate"] == pytest.approx(math.log(0.5) - math.log(0.8), abs=1e-12)
    assert r["ci_level"] == 0.95
    assert r["ci_lower"] == pytest.approx(0.625 * math.exp(-Z95 * math.sqrt(0.13)), abs=1e-5)
    assert r["ci_upper"] == pytest.approx(0.625 * math.exp(Z95 * math.sqrt(0.13)), abs=1e-5)
    assert r["ci_lower"] == pytest.approx(0.3083, abs=2e-4)     # the hand figure
    assert r["ci_upper"] == pytest.approx(1.2670, abs=2e-4)
    assert r["z"] == pytest.approx(-0.470004 / 0.360555, abs=1e-5)
    assert r["z"] == pytest.approx(-1.30356, abs=1e-4)
    assert r["p_value"] == pytest.approx(_p_two_sided(r["z"]), abs=1e-12)
    assert r["p_value"] == pytest.approx(0.1924, abs=5e-4)


# ── hand-derived worked example, difference scale ──────────────────────────────
# A vs B: MD 5.0, SE 0.3.   C vs B: MD 2.0, SE 0.4.
#   MD_AC = 5.0 - 2.0 = 3.0         SE = sqrt(0.09 + 0.16) = sqrt(0.25) = 0.5 (exact)
#   CI    = 3.0 +/- 1.959964 * 0.5 = (2.020018, 3.979982)
#   z     = 3.0 / 0.5 = 6            p = 2 * (1 - Phi(6)) = 1.973e-9
def test_difference_scale_worked_example_by_hand():
    r = bucher_indirect(
        {"estimate": 5.0, "se": 0.3},
        {"estimate": 2.0, "se": 0.4},
        scale="MD")
    assert r["scale_type"] == "difference"
    assert r["estimate"] == pytest.approx(3.0, abs=1e-12)
    assert r["se"] == pytest.approx(0.5, abs=1e-12)
    assert r["ci_lower"] == pytest.approx(2.020018, abs=1e-5)
    assert r["ci_upper"] == pytest.approx(3.979982, abs=1e-5)
    assert r["z"] == pytest.approx(6.0, abs=1e-12)
    assert r["p_value"] == pytest.approx(1.9733e-9, rel=1e-3)
    # difference scale: the analysis estimate IS the estimate (identity link)
    assert r["analysis_estimate"] == r["estimate"]


# ── published numbers: Bucher et al., J Clin Epidemiol 1997;50(6):683-91 ───────
# PubMed record 9250266 (doi 10.1016/s0895-4356(97)00049-8), fetched: the indirect
# comparison of sulphamethoxazole-trimethoprim vs dapsone/pyrimethamine gave
# OR 0.37 (95% CI 0.21 to 0.65). By hand:
#   SE(log OR) = (ln 0.65 - ln 0.21) / (2 * 1.959964)
#              = (-0.430783 + 1.560648) / 3.919928 = 1.129865 / 3.919928 = 0.288236
#   rebuilt CI = 0.37 * exp(-/+ 1.959964 * 0.288236) = 0.37 * exp(-/+ 0.564937)
#              = 0.37 / 1.75935 .. 0.37 * 1.75935 = (0.2103, 0.6510) ~ (0.21, 0.65)
# (the ~0.0005 gap is the rounding of the published ratio to two decimals).
def test_published_bucher_interval_round_trip():
    se = se_from_ci(0.21, 0.65, ci_level=0.95, scale="OR")
    assert se == pytest.approx(0.288236, abs=1e-6)
    lo, hi = ci_from_se(0.37, se, ci_level=0.95, scale="OR")
    assert lo == pytest.approx(0.2103, abs=5e-4) and hi == pytest.approx(0.6510, abs=5e-4)
    assert lo == pytest.approx(0.21, abs=2e-3) and hi == pytest.approx(0.65, abs=2e-3)


def test_variances_add_reproduces_the_published_width():
    """Split the published indirect variance equally between the two direct
    comparisons (each SE = 0.288236 / sqrt 2 = 0.203809); the Bucher SE must
    recover 0.288236 and the estimate 0.296 / 0.80 = 0.37 (exact)."""
    half = 0.288236 / math.sqrt(2.0)
    r = bucher_indirect({"estimate": 0.296, "se": half}, {"estimate": 0.80, "se": half},
                        scale="OR")
    assert r["estimate"] == pytest.approx(0.37, abs=1e-12)
    assert r["se"] == pytest.approx(0.288236, abs=1e-9)
    assert r["ci_lower"] == pytest.approx(0.2103, abs=5e-4)
    assert r["ci_upper"] == pytest.approx(0.6510, abs=5e-4)


# ── CI <-> SE conversion ───────────────────────────────────────────────────────
# SE = (ln U - ln L) / (2 z) on ratio scales, (U - L) / (2 z) on difference scales.
@pytest.mark.parametrize("level,z", [(0.95, Z95), (0.90, Z90)])
def test_ratio_se_ci_round_trip(level, z):
    est, se = 0.5, 0.2
    lo, hi = ci_from_se(est, se, ci_level=level, scale="HR")
    assert lo == pytest.approx(est * math.exp(-z * se), abs=1e-5)
    assert hi == pytest.approx(est * math.exp(z * se), abs=1e-5)
    assert se_from_ci(lo, hi, ci_level=level, scale="HR") == pytest.approx(se, abs=1e-5)


def test_difference_se_ci_round_trip():
    # MD 5.0, SE 0.3 -> half-width 1.959964 * 0.3 = 0.587989 -> (4.412011, 5.587989)
    lo, hi = ci_from_se(5.0, 0.3, ci_level=0.95, scale="MD")
    assert (lo, hi) == (pytest.approx(4.412011, abs=1e-5), pytest.approx(5.587989, abs=1e-5))
    assert se_from_ci(lo, hi, ci_level=0.95, scale="MD") == pytest.approx(0.3, abs=1e-9)


def test_ci_input_is_converted_to_the_same_answer_as_se_input():
    lo, hi = ci_from_se(0.5, 0.2, ci_level=0.95, scale="HR")
    by_ci = bucher_indirect({"estimate": 0.5, "ci_lower": lo, "ci_upper": hi},
                            {"estimate": 0.8, "se": 0.3}, scale="HR")
    by_se = bucher_indirect({"estimate": 0.5, "se": 0.2},
                            {"estimate": 0.8, "se": 0.3}, scale="HR")
    assert by_ci["se"] == pytest.approx(by_se["se"], abs=1e-9)
    assert by_ci["estimate"] == pytest.approx(by_se["estimate"], abs=1e-12)
    assert by_ci["inputs"]["ab"]["se_source"] == "from_ci"
    assert by_se["inputs"]["ab"]["se_source"] == "supplied"


def test_input_ci_level_is_respected():
    lo, hi = ci_from_se(0.5, 0.2, ci_level=0.90, scale="HR")
    r = bucher_indirect({"estimate": 0.5, "ci_lower": lo, "ci_upper": hi, "ci_level": 0.90},
                        {"estimate": 0.8, "se": 0.3}, scale="HR")
    assert r["inputs"]["ab"]["se"] == pytest.approx(0.2, abs=1e-5)
    assert r["ci_level"] == 0.95          # output level is independent of the inputs'


def test_output_ci_level_option():
    r = bucher_indirect({"estimate": 5.0, "se": 0.3}, {"estimate": 2.0, "se": 0.4},
                        scale="MD", ci_level=0.90)
    assert r["ci_level"] == 0.90
    assert r["ci_lower"] == pytest.approx(3.0 - Z90 * 0.5, abs=1e-5)
    assert r["ci_upper"] == pytest.approx(3.0 + Z90 * 0.5, abs=1e-5)


# ── structural properties ──────────────────────────────────────────────────────
def test_swapping_the_two_comparisons_inverts_a_ratio_and_keeps_se_and_p():
    ab, cb = {"estimate": 0.5, "se": 0.2}, {"estimate": 0.8, "se": 0.3}
    fwd = bucher_indirect(ab, cb, scale="RR")
    rev = bucher_indirect(cb, ab, scale="RR")
    assert rev["estimate"] == pytest.approx(1.0 / fwd["estimate"], abs=1e-12)
    assert rev["se"] == pytest.approx(fwd["se"], abs=1e-12)
    assert rev["p_value"] == pytest.approx(fwd["p_value"], abs=1e-12)
    assert rev["ci_lower"] == pytest.approx(1.0 / fwd["ci_upper"], abs=1e-9)


def test_identical_effects_give_null_result():
    r = bucher_indirect({"estimate": 0.7, "se": 0.1}, {"estimate": 0.7, "se": 0.1}, scale="HR")
    assert r["estimate"] == pytest.approx(1.0, abs=1e-12)
    assert r["z"] == pytest.approx(0.0, abs=1e-12)
    assert r["p_value"] == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("name,canon,kind", [
    ("hr", "HR", "ratio"), ("Or", "OR", "ratio"), (" RR ", "RR", "ratio"),
    ("md", "MD", "difference"), ("SMD", "SMD", "difference"), ("rd", "RD", "difference"),
])
def test_scale_names_are_case_insensitive_and_typed(name, canon, kind):
    assert normalize_scale(name) == (canon, kind)


def test_labels_and_inputs_are_echoed():
    r = bucher_indirect({"estimate": 0.5, "se": 0.2, "label": "Trial one"},
                        {"estimate": 0.8, "se": 0.3}, scale="HR",
                        treatments=("Drug A", "Placebo", "Drug C"))
    assert r["contrast"] == "Drug A vs Drug C via Placebo"
    assert r["treatments"] == {"A": "Drug A", "B": "Placebo", "C": "Drug C"}
    ab, cb = r["inputs"]["ab"], r["inputs"]["cb"]
    assert ab["estimate"] == 0.5 and ab["se"] == 0.2 and ab["label"] == "Trial one"
    assert cb["estimate"] == 0.8 and cb["se"] == 0.3
    assert ab["analysis_estimate"] == pytest.approx(math.log(0.5), abs=1e-12)
    assert r["inputs"]["cb"]["analysis_estimate"] == pytest.approx(math.log(0.8), abs=1e-12)


def test_result_is_json_serialisable_and_inputs_not_mutated():
    ab, cb = {"estimate": 0.5, "se": 0.2}, {"estimate": 0.8, "se": 0.3}
    r = bucher_indirect(ab, cb, scale="HR")
    json.dumps(r)
    assert ab == {"estimate": 0.5, "se": 0.2} and cb == {"estimate": 0.8, "se": 0.3}


# ── strict validation ──────────────────────────────────────────────────────────
OK_AB = {"estimate": 0.5, "se": 0.2}
OK_CB = {"estimate": 0.8, "se": 0.3}


@pytest.mark.parametrize("ab,cb,scale,msg", [
    ({"estimate": -0.5, "se": 0.2}, OK_CB, "HR", "must be positive"),
    ({"estimate": 0.0, "se": 0.2}, OK_CB, "OR", "must be positive"),
    (OK_AB, {"estimate": 0.8, "se": 0.0}, "HR", "se must be > 0"),
    (OK_AB, {"estimate": 0.8, "se": -1.0}, "HR", "se must be > 0"),
    ({"estimate": 0.5, "se": float("nan")}, OK_CB, "HR", "finite"),
    ({"estimate": float("inf"), "se": 0.2}, OK_CB, "MD", "finite"),
    ({"estimate": True, "se": 0.2}, OK_CB, "MD", "number"),
    ({"estimate": "0.5", "se": 0.2}, OK_CB, "HR", "number"),
    ({"se": 0.2}, OK_CB, "HR", "estimate is required"),
    ({"estimate": 0.5}, OK_CB, "HR", "se or a CI"),
    ({"estimate": 0.5, "se": 0.2, "ci_lower": 0.3, "ci_upper": 0.8}, OK_CB, "HR",
     "not both"),
    ({"estimate": 0.5, "ci_lower": 0.3}, OK_CB, "HR", "both ci_lower and ci_upper"),
    ({"estimate": 0.5, "ci_lower": 0.8, "ci_upper": 0.3}, OK_CB, "HR", "ci_lower must be below"),
    ({"estimate": 0.5, "ci_lower": 0.5, "ci_upper": 0.5}, OK_CB, "HR", "ci_lower must be below"),
    ({"estimate": 0.9, "ci_lower": 0.3, "ci_upper": 0.8}, OK_CB, "HR", "inside its CI"),
    ({"estimate": 0.5, "ci_lower": -0.1, "ci_upper": 0.8}, OK_CB, "HR", "must be positive"),
    ({"estimate": 0.5, "ci_lower": 0.3, "ci_upper": 0.8, "ci_level": 1.0}, OK_CB, "HR",
     "ci_level"),
    ({"estimate": 0.5, "ci_lower": 0.3, "ci_upper": 0.8, "ci_level": 0.0}, OK_CB, "HR",
     "ci_level"),
    (OK_AB, OK_CB, "banana", "unknown scale"),
    (OK_AB, OK_CB, "", "unknown scale"),
    ([0.5, 0.2], OK_CB, "HR", "must be an object"),
    (OK_AB, None, "HR", "must be an object"),
])
def test_invalid_inputs_raise_clear_value_errors(ab, cb, scale, msg):
    with pytest.raises(ValueError, match=msg):
        bucher_indirect(ab, cb, scale=scale)


def test_negative_estimate_is_fine_on_a_difference_scale():
    r = bucher_indirect({"estimate": -1.5, "se": 0.2}, {"estimate": -0.5, "se": 0.2}, scale="MD")
    assert r["estimate"] == pytest.approx(-1.0, abs=1e-12)


@pytest.mark.parametrize("level", [0.0, 1.0, -0.2, 1.5, float("nan")])
def test_output_ci_level_must_be_a_proper_fraction(level):
    with pytest.raises(ValueError, match="ci_level"):
        bucher_indirect(OK_AB, OK_CB, scale="HR", ci_level=level)


# ── numeric edge cases must be ValueErrors (a 400), never ZeroDivision/Overflow (a 500) ─────
def test_extreme_but_finite_standard_errors_still_work():
    r = bucher_indirect({"estimate": 0.5, "se": 1e-200}, {"estimate": 0.8, "se": 1e-200}, scale="HR")
    assert r["se"] == pytest.approx(math.sqrt(2.0) * 1e-200, rel=1e-12)     # hypot: no underflow to 0
    assert math.isfinite(r["z"]) and r["p_value"] == 0.0
    json.dumps(r)


@pytest.mark.parametrize("ab,cb,scale,kw,msg", [
    ({"estimate": 0.5, "se": 1e200}, {"estimate": 0.8, "se": 1e200}, "HR", {}, "out of numeric range"),
    ({"estimate": 1e300, "se": 0.1}, {"estimate": 1e-300, "se": 0.1}, "OR", {}, "out of numeric range"),
    ({"estimate": 1e308, "se": 0.1}, {"estimate": -1e308, "se": 0.1}, "MD", {}, "not finite"),
    ({"estimate": 0.5, "ci_lower": 0.3, "ci_upper": 0.8, "ci_level": 1e-20}, OK_CB, "HR", {}, "ci_level"),
    (OK_AB, OK_CB, "HR", {"ci_level": 1 - 1e-16}, "ci_level"),
    (OK_AB, OK_CB, "HR", {"ci_level": 1e-20}, "ci_level"),
])
def test_numeric_extremes_raise_value_error(ab, cb, scale, kw, msg):
    with pytest.raises(ValueError, match=msg):
        bucher_indirect(ab, cb, scale=scale, **kw)


def test_ci_derived_se_must_be_positive_and_finite():
    with pytest.raises(ValueError, match="se must be > 0"):
        bucher_indirect({"estimate": 0.0, "ci_lower": 0.0, "ci_upper": 5e-324},    # SE underflows to 0
                        OK_CB, scale="MD")


@pytest.mark.parametrize("bad", ["Drug A cuts mortality 87 pct", "Drug \u2461", "Dose \u00bd", "x\uff19"])
def test_treatment_labels_may_not_carry_numerals(bad):
    with pytest.raises(ValueError, match="digits"):
        bucher_indirect(OK_AB, OK_CB, scale="HR", treatments=(bad, "B", "C"))
    with pytest.raises(ValueError, match="digits"):
        bucher_indirect(OK_AB, OK_CB, scale="HR", treatments=("A", bad, "C"))
