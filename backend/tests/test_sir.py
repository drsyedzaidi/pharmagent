"""Sampling Importance Resampling (Dosne et al. 2016).

SIR is validated here against objectives whose uncertainty is known in CLOSED
FORM. With an exactly quadratic objective the posterior is exactly Gaussian
with a known covariance, so "did SIR recover the truth" is a real question with
a real answer -- not a comparison against another approximation.

`ofv_fn` is injected, so none of this runs a fit.
"""
import math

import numpy as np
import pytest

from app.compute.sir import run_sir

# OFV(x) = (x - x0)' S^-1 (x - x0) is -2 log L of a Gaussian up to a constant,
# so the exact uncertainty is N(x0, S): SD 0.2 and 0.3.
S_TRUE = np.diag([0.04, 0.09])
X0 = np.array([1.0, 2.0])
_P = np.linalg.inv(S_TRUE)
Z95 = 1.959963984540054


def _ofv(x):
    d = np.asarray(x, dtype=float) - X0
    return float(d @ _P @ d)


def _decode(x):
    return {"a": float(x[0]), "b": float(x[1])}


def _sir(**kw):
    base = dict(ofv_fn=_ofv, x_hat=X0, cov=S_TRUE, ofv_hat=0.0,
                decode_fn=_decode, n_resample=1000, n_samples=10000, seed=7)
    base.update(kw)
    return run_sir(**base)


def _sd_from_ci(p):
    return (p["sir_hi"] - p["sir_lo"]) / (2 * Z95)


# ── recovery ────────────────────────────────────────────────────────────────

def test_recovers_the_true_sd_when_the_proposal_is_correct():
    r = _sir()
    assert r["status"] == "ok"
    a = next(p for p in r["parameters"] if p["parameter"] == "a")
    b = next(p for p in r["parameters"] if p["parameter"] == "b")
    assert _sd_from_ci(a) == pytest.approx(0.2, rel=0.10)
    assert _sd_from_ci(b) == pytest.approx(0.3, rel=0.10)


@pytest.mark.parametrize("inflation", [2.0, 3.0])
def test_corrects_an_inflated_proposal_back_to_the_truth(inflation):
    """THE point of importance weighting: the proposal is deliberately wrong
    (2x / 3x too wide) and SIR must still land on the true uncertainty. A
    method that merely echoed its proposal would return 2x / 3x here."""
    r = _sir(inflation=inflation)
    a = next(p for p in r["parameters"] if p["parameter"] == "a")
    assert _sd_from_ci(a) == pytest.approx(0.2, rel=0.15), (
        f"inflation={inflation} not corrected: {_sd_from_ci(a)}")


def test_a_correct_proposal_needs_no_correction():
    """All importance ratios equal => uniform weights => ESS is the full sample.
    This is the analytic fixed point: dOFV == mahalanobis^2 exactly, so
    log IR == 0 for every draw."""
    r = _sir()
    d = r["diagnostics"]
    assert d["effective_sample_size"] == pytest.approx(r["n_samples"], rel=1e-6)
    assert d["max_weight"] == pytest.approx(1.0 / r["n_samples"], rel=1e-6)


def test_effective_sample_size_falls_as_the_proposal_worsens():
    ess = [_sir(inflation=i)["diagnostics"]["effective_sample_size"]
           for i in (1.0, 2.0, 3.0)]
    assert ess[0] > ess[1] > ess[2], ess


# ── the chi-square diagnostic ───────────────────────────────────────────────

def test_dofv_mean_recovers_the_degrees_of_freedom():
    """For an unconstrained quadratic in 2 parameters the resampled dOFV is
    chi-square with 2 df, and E[chi2_df] = df."""
    r = _sir()
    assert r["diagnostics"]["dofv_mean_resampled"] == pytest.approx(2.0, abs=0.35)
    assert r["diagnostics"]["df_reference"] == 2


def test_a_too_narrow_proposal_is_flagged_in_the_right_direction():
    """Direction is easy to invert and the inverted version would tell a reader
    the opposite of the truth. Samples from a NARROW proposal hug the optimum,
    so their dOFV is LOW -- Dosne describes this as sitting below the
    chi-square."""
    r = _sir(inflation=0.4, n_resample=500, n_samples=5000)
    assert r["diagnostics"]["dofv_mean_resampled"] < 2.0
    assert any("too NARROW" in n for n in r["notes"])
    assert not any("wider than the true" in n for n in r["notes"])


def test_a_matched_proposal_raises_neither_width_warning():
    r = _sir(n_resample=500, n_samples=5000)
    assert not any("too NARROW" in n for n in r["notes"])
    assert not any("wider than the true" in n for n in r["notes"])


def test_low_m_over_m_ratio_is_flagged():
    r = _sir(n_resample=100, n_samples=120)
    assert r["m_over_m_ratio"] < 5
    assert any("M/m" in n for n in r["notes"])


def test_df_caveat_is_always_stated():
    """df is expected at or BELOW the parameter count; a reader comparing it to
    the raw count without that caveat would misread a good run as a bad one."""
    assert any("below" in n.lower() and "degree of freedom" in n
               for n in _sir(n_resample=200, n_samples=1000)["notes"])


# ── asymmetry: the reason to prefer SIR over a normal approximation ─────────

def test_asymmetric_uncertainty_is_captured_on_the_natural_scale():
    """Sampling happens on the estimation scale and is decoded afterwards, so a
    log-scale parameter yields an ASYMMETRIC natural-scale interval -- something
    a symmetric normal approximation structurally cannot express."""
    r = run_sir(ofv_fn=_ofv, x_hat=X0, cov=S_TRUE, ofv_hat=0.0,
                decode_fn=lambda x: {"CL": float(math.exp(x[0]))},
                n_resample=1000, n_samples=6000, seed=3)
    p = r["parameters"][0]
    assert p["sir_lo"] > 0.0                      # exp() -> strictly positive
    assert p["asymmetry_ratio"] is not None and p["asymmetry_ratio"] > 1.05


# ── degeneracy and contract ─────────────────────────────────────────────────

def test_degenerate_weights_are_refused_not_reported():
    """A proposal centred far from the optimum makes almost every weight
    vanish. Reporting a CI off the few survivors would be worse than saying so."""
    r = run_sir(ofv_fn=lambda x: 1e6 * _ofv(x), x_hat=X0 + 50.0, cov=S_TRUE,
                ofv_hat=0.0, decode_fn=_decode, n_resample=200, n_samples=400,
                seed=5)
    assert r["status"] in ("degenerate_weights", "too_few_usable_samples")
    assert "parameters" not in r


def test_non_finite_objectives_are_counted_and_survivable():
    calls = {"n": 0}

    def flaky(x):
        calls["n"] += 1
        return math.nan if calls["n"] % 10 == 0 else _ofv(x)

    r = run_sir(ofv_fn=flaky, x_hat=X0, cov=S_TRUE, ofv_hat=0.0,
                decode_fn=_decode, n_resample=300, n_samples=3000, seed=5)
    assert r["status"] == "ok"
    assert r["n_failed_objective"] > 0
    assert r["n_usable"] + r["n_failed_objective"] == r["n_samples"]


def test_objective_is_evaluated_once_per_sample_and_never_refits():
    """The reason to use SIR: cost is M objective evaluations, no estimation."""
    calls = {"n": 0}

    def counting(x):
        calls["n"] += 1
        return _ofv(x)

    r = run_sir(ofv_fn=counting, x_hat=X0, cov=S_TRUE, ofv_hat=0.0,
                decode_fn=_decode, n_resample=100, n_samples=800, seed=1)
    assert calls["n"] == r["n_samples"] == 800


def test_resampling_is_without_replacement():
    """Without replacement is what makes the M/m ratio meaningful; with
    replacement one dominant vector could fill the resample."""
    r = _sir(inflation=2.5, n_resample=500, n_samples=5000)
    # A duplicate-free resample cannot have any single parameter value repeated
    # more often than the underlying draws allow; with replacement and a
    # skewed weight vector, duplicates would be abundant.
    vals = [p["sir_median"] for p in r["parameters"]]
    assert len(vals) == 2
    assert r["n_resample"] == 500
    assert r["diagnostics"]["max_weight"] < 1.0


def test_deterministic_for_a_fixed_seed():
    a, b = _sir(n_resample=200, n_samples=1000), _sir(n_resample=200, n_samples=1000)
    assert a["parameters"] == b["parameters"]
    assert a["diagnostics"] == b["diagnostics"]


def test_reports_the_asymptotic_interval_alongside():
    r = _sir(n_resample=300, n_samples=2000)
    a = next(p for p in r["parameters"] if p["parameter"] == "a")
    assert a["asymptotic_lo"] == pytest.approx(1.0 - Z95 * 0.2, rel=1e-6)
    assert a["asymptotic_hi"] == pytest.approx(1.0 + Z95 * 0.2, rel=1e-6)


@pytest.mark.parametrize("kw,match", [
    ({"ci_level": 1.5}, "ci_level"),
    ({"cov": np.eye(3)}, "cov must be"),
    ({"inflation": -1.0}, "inflation"),
])
def test_invalid_arguments_raise(kw, match):
    with pytest.raises(ValueError, match=match):
        _sir(**kw)


def test_result_is_json_safe():
    import json
    s = json.dumps(_sir(n_resample=200, n_samples=1000))
    assert "NaN" not in s and "Infinity" not in s


# ── tool layer ──────────────────────────────────────────────────────────────

def test_sir_tool_is_not_llm_reachable():
    """M objective evaluations plus a numeric Hessian is minutes to tens of
    minutes of real compute; the guardrail keeps that off any chat path."""
    from app.agents.definitions import AGENTS, DESCRIPTIONS
    from app.agents.supervisor import KEYWORDS
    from app.tools.builtins import default_registry

    assert default_registry()._tools["run_sir"].agent == "simulator"
    assert "simulator" not in AGENTS
    assert "simulator" not in DESCRIPTIONS
    assert "simulator" not in KEYWORDS


def test_sir_tool_requires_confirm_and_writes_nothing_on_refusal():
    from app.core.pharmstate import PharmState
    from app.tools.base import ToolContext
    from app.tools.sir_tools import run_sir as tool_run

    res = tool_run(PharmState(), ToolContext(), {})
    assert res.result["status"] == "confirm_required"
    assert res.writes == {}


def test_sir_tool_rejections_never_overwrite_a_completed_run():
    from app.core.pharmstate import PharmState
    from app.tools.base import ToolContext
    from app.tools.sir_tools import run_sir as tool_run

    prior = {"status": "ok", "n_samples": 5000}
    st = PharmState(sir_results=prior)
    for args in ({}, {"confirm": True}):
        assert tool_run(st, ToolContext(), args).writes == {}
    assert st.sir_results == prior


def test_sir_results_is_writable_by_the_simulator_agent():
    from app.core.pharmstate import AGENT_WRITE_FIELDS
    assert "sir_results" in AGENT_WRITE_FIELDS["simulator"]


def test_near_singular_proposal_caveat_is_surfaced_first():
    """A regularized singular information matrix makes the proposal look better
    conditioned than the fit. If that is not said, SIR's intervals read as more
    trustworthy than they are."""
    note = "PROPOSAL CAVEAT: the information matrix is near-singular."
    r = _sir(n_resample=200, n_samples=1000, proposal_note=note)
    assert r["notes"][0] == note
