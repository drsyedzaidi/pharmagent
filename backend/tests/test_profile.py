"""Log-likelihood profiling.

Validated against objectives whose profile is known in CLOSED FORM. The
sharpest check is that a quadratic objective must reproduce the Wald interval
EXACTLY -- profiling out the other parameters of a quadratic leaves
dOFV(v) = ((v - est)/SE)^2, so the 3.84 crossing is est +/- 1.96*SE. Anything
else means the search or the interpolation is wrong.

`profile_ofv_fn` is injected; nothing here runs an estimator.
"""
import math

import pytest

from app.compute.profile import _chi2_1df, run_profile

Z95 = 1.959963984540054
SE = {"CL": 0.25, "V": 3.0}
EST = {"CL": 4.0, "V": 40.0}


def _quadratic(name, v):
    """Profile of a quadratic objective: ((v - est)/SE)^2."""
    return ((v - EST[name]) / SE[name]) ** 2


def _run(fn=_quadratic, estimates=None, **kw):
    est = estimates if estimates is not None else EST
    base = dict(profile_ofv_fn=fn, estimates=est, ofv_hat=0.0,
                initial_step={k: SE.get(k, abs(v) * 0.2) for k, v in est.items()})
    base.update(kw)
    return run_profile(**base)


# ── the analytic anchor ─────────────────────────────────────────────────────

@pytest.mark.parametrize("name", ["CL", "V"])
def test_quadratic_objective_reproduces_the_wald_interval_exactly(name):
    r = _run()
    p = next(x for x in r["parameters"] if x["parameter"] == name)
    est, se = EST[name], SE[name]
    assert p["profile_lo"] == pytest.approx(est - Z95 * se, rel=1e-3)
    assert p["profile_hi"] == pytest.approx(est + Z95 * se, rel=1e-3)


def test_quadratic_profile_is_symmetric():
    p = next(x for x in _run()["parameters"] if x["parameter"] == "CL")
    assert p["asymmetry_ratio"] == pytest.approx(1.0, abs=0.01)


def test_cutoff_is_the_chi_square_one_df_quantile():
    assert _run()["dofv_cutoff"] == pytest.approx(3.841458820694124, abs=1e-6)
    assert _chi2_1df(0.95) == pytest.approx(3.841458820694124, abs=1e-9)
    assert _chi2_1df(0.90) == pytest.approx(2.705543454095404, abs=1e-6)


def test_a_tighter_level_gives_a_narrower_interval():
    wide = _run(ci_level=0.95)["parameters"][0]
    narrow = _run(ci_level=0.80)["parameters"][0]
    assert narrow["profile_lo"] > wide["profile_lo"]
    assert narrow["profile_hi"] < wide["profile_hi"]


# ── asymmetry: the reason to profile at all ─────────────────────────────────

def test_non_quadratic_objective_gives_the_exact_asymmetric_interval():
    """A log-scale quadratic has closed-form limits est*exp(+/-1.96*s), which
    are asymmetric on the natural scale -- something a Wald interval cannot
    represent."""
    s = 0.25

    def logscale(_name, v):
        return (math.log(max(v, 1e-9) / 4.0) / s) ** 2

    p = _run(logscale, {"CL": 4.0}, initial_step={"CL": 1.0})["parameters"][0]
    assert p["profile_lo"] == pytest.approx(4.0 * math.exp(-Z95 * s), rel=1e-3)
    assert p["profile_hi"] == pytest.approx(4.0 * math.exp(+Z95 * s), rel=1e-3)
    assert p["asymmetry_ratio"] > 1.5


# ── diagnostics that make a wrong answer visible ────────────────────────────

def test_detects_that_the_fit_was_not_at_an_optimum():
    """A constrained fit beating the reported optimum means dOFV < 0: the
    original fit had not converged, so every limit measured against it is
    referenced to the wrong point. That must be said loudly, not folded into
    a plausible-looking interval."""
    r = _run(lambda _n, v: (v - 5.0) ** 2 - 1.0, {"CL": 4.0},
             initial_step={"CL": 0.5})
    flagged = r["diagnostics"]["fit_not_at_optimum"]
    assert flagged and flagged[0]["parameter"] == "CL"
    assert any("not at a minimum" in n for n in r["notes"])
    assert any("Re-fit" in n for n in r["notes"])


def test_unbounded_parameter_reports_none_rather_than_extrapolating():
    """An objective that asymptotes below the cut-off never bounds the
    parameter. Inventing a limit past the evaluated region would be
    fabrication; None plus a reason is the honest answer."""
    r = _run(lambda _n, v: 3.0 * (1.0 - math.exp(-((v - 4.0) ** 2))), {"CL": 4.0},
             initial_step={"CL": 0.5})
    p = r["parameters"][0]
    assert p["profile_lo"] is None and p["profile_hi"] is None
    assert r["diagnostics"]["unbounded_parameters"] == ["CL"]
    assert "never reached the cut-off" in (p["upper_reason"] or "")
    assert any("do not bound that parameter" in n for n in r["notes"])


def test_wide_but_bounded_objective_is_not_called_unbounded():
    """Guard against over-eager 'unbounded': 0.001*(v-4)^2 crosses 3.84 at
    |v-4| = 62, which the outward doubling must actually reach."""
    r = _run(lambda _n, v: 0.001 * (v - 4.0) ** 2, {"CL": 4.0},
             initial_step={"CL": 0.1})
    p = r["parameters"][0]
    assert r["diagnostics"]["unbounded_parameters"] == []
    half = math.sqrt(3.841458820694124 / 0.001)
    assert p["profile_lo"] == pytest.approx(4.0 - half, rel=1e-2)
    assert p["profile_hi"] == pytest.approx(4.0 + half, rel=1e-2)


def test_non_monotone_profile_is_flagged():
    """A second optimum means the reported limit is only the FIRST crossing,
    not the edge of the supported region."""
    def bumpy(_n, v):
        d = v - 4.0
        return d * d * (1.0 - 0.9 * math.exp(-((abs(d) - 3.0) ** 2)))

    r = _run(bumpy, {"CL": 4.0}, initial_step={"CL": 0.5})
    assert r["diagnostics"]["non_monotone_parameters"] == ["CL"]
    assert any("second optimum" in n for n in r["notes"])


def test_the_bounds_only_limitation_is_always_stated():
    """Dosne's stated drawback: no joint distribution, nothing to simulate
    from, one parameter at a time. A reader must not mistake this for
    bootstrap/SIR output."""
    assert any("BOUNDS only" in n for n in _run()["notes"])


# ── contract ────────────────────────────────────────────────────────────────

def test_profiles_only_the_requested_parameters():
    r = _run(params=("CL",))
    assert [p["parameter"] for p in r["parameters"]] == ["CL"]


def test_records_the_evaluated_profile_points():
    p = _run(params=("CL",))["parameters"][0]
    assert p["n_evaluations"] == len(p["profile"]) > 4
    vals = [q["value"] for q in p["profile"]]
    assert vals == sorted(vals)
    assert min(vals) < EST["CL"] < max(vals)


def test_a_poor_initial_step_still_finds_the_same_limits():
    """The step only affects how many evaluations it takes to bracket, never
    the answer."""
    tight = _run(params=("CL",), initial_step={"CL": 1e-4})["parameters"][0]
    loose = _run(params=("CL",), initial_step={"CL": 5.0})["parameters"][0]
    assert tight["profile_lo"] == pytest.approx(loose["profile_lo"], rel=1e-2)
    assert tight["profile_hi"] == pytest.approx(loose["profile_hi"], rel=1e-2)


def test_deterministic():
    assert _run()["parameters"] == _run()["parameters"]


def test_rejects_unknown_parameters_and_bad_levels():
    with pytest.raises(ValueError, match="absent from estimates"):
        _run(params=("NOPE",))
    with pytest.raises(ValueError, match="ci_level"):
        _run(ci_level=1.5)


def test_degenerate_inputs():
    assert run_profile(profile_ofv_fn=_quadratic, estimates={},
                       ofv_hat=0.0)["status"] == "no_parameters"
    assert run_profile(profile_ofv_fn=_quadratic, estimates=EST,
                       ofv_hat=float("nan"))["status"] == "no_fit"


def test_non_finite_objective_values_do_not_crash_the_search():
    def flaky(_n, v):
        return math.nan if v < 3.7 else ((v - 4.0) / 0.25) ** 2

    r = _run(flaky, {"CL": 4.0}, initial_step={"CL": 0.25})
    assert r["status"] == "ok"
    p = r["parameters"][0]
    assert p["profile_hi"] == pytest.approx(4.0 + Z95 * 0.25, rel=1e-2)


def test_result_is_json_safe():
    import json
    s = json.dumps(_run())
    assert "NaN" not in s and "Infinity" not in s


# ── tool layer ──────────────────────────────────────────────────────────────

def test_profile_tool_is_not_llm_reachable():
    from app.agents.definitions import AGENTS, DESCRIPTIONS
    from app.agents.supervisor import KEYWORDS
    from app.tools.builtins import default_registry

    assert default_registry()._tools["run_profile"].agent == "simulator"
    assert "simulator" not in AGENTS
    assert "simulator" not in DESCRIPTIONS
    assert "simulator" not in KEYWORDS


def test_profile_tool_requires_confirm():
    from app.core.pharmstate import PharmState
    from app.tools.base import ToolContext
    from app.tools.profile_tools import run_profile as tool_run

    res = tool_run(PharmState(), ToolContext(), {})
    assert res.result["status"] == "confirm_required"
    assert res.writes == {}


def test_profile_tool_rejections_never_overwrite_a_completed_run():
    from app.core.pharmstate import PharmState
    from app.tools.base import ToolContext
    from app.tools.profile_tools import run_profile as tool_run

    prior = {"status": "ok", "n_parameters": 2}
    st = PharmState(profile_results=prior)
    for args in ({}, {"confirm": True}):
        assert tool_run(st, ToolContext(), args).writes == {}
    assert st.profile_results == prior


def test_profile_results_is_writable_by_the_simulator_agent():
    from app.core.pharmstate import AGENT_WRITE_FIELDS
    assert "profile_results" in AGENT_WRITE_FIELDS["simulator"]
