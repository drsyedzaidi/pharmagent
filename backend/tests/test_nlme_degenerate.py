"""Collapsed FOCE-I fits are detected, reported, and escalated.

theta is optimised as log(theta) with nothing stopping the search at -inf, so a
flat/improving direction ends in underflow and the parameter is reported as an
exact 0.0 — a clearance of zero, with plausible RSE% on its neighbours and
converged=True. These tests pin the detection, the honest reporting, and the
escalate-only-when-needed behaviour.
"""
from __future__ import annotations

import pytest

from app.compute import nlme
from app.compute.pk_models import REGISTRY


class _Spec:
    """Minimal stand-in for _PopSpec: only the fields the detector reads."""
    def __init__(self, model_key="oral_2cmt"):
        self.model = REGISTRY[model_key]
        self.param_names = list(self.model.params)


# ── detector ─────────────────────────────────────────────────────────────────

def test_healthy_estimates_are_not_flagged():
    spec = _Spec()
    healthy = {"CL": 0.357, "VC": 3.66, "Q": 0.46, "VP": 3.67, "KA": 0.24}
    assert nlme._degenerate_thetas(spec, healthy) == []


def test_underflowed_parameter_is_flagged():
    """The observed failure: exp(-745) -> 0.0 for CL."""
    spec = _Spec()
    assert nlme._degenerate_thetas(
        spec, {"CL": 0.0, "VC": 0.398, "Q": 0.356, "VP": 121.5, "KA": 0.038}) == ["CL"]


def test_near_underflow_is_flagged_even_though_it_is_not_exactly_zero():
    """The real raw value was 3.9e-08 — an absolute floor near 1e-9 misses it,
    which is why the test is relative to the model's own default (CL=5)."""
    spec = _Spec()
    assert "CL" in nlme._degenerate_thetas(
        spec, {"CL": 3.8872834405503134e-08, "VC": 0.398, "Q": 0.356,
               "VP": 121.5, "KA": 0.038})


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0, 0.0])
def test_non_numeric_or_non_positive_is_flagged(bad):
    spec = _Spec()
    assert "CL" in nlme._degenerate_thetas(
        spec, {"CL": bad, "VC": 3.0, "Q": 0.4, "VP": 3.0, "KA": 0.3})


def test_unit_changes_do_not_trip_the_detector():
    """A monoclonal antibody's CL is ~0.25 L/day and ~0.0104 L/h; both are real
    estimates, so neither may be called degenerate."""
    spec = _Spec()
    for cl in (0.25, 0.0104, 1.7e-4):          # per day, per hour, per minute
        assert nlme._degenerate_thetas(
            spec, {"CL": cl, "VC": 3.5, "Q": 0.4, "VP": 2.8, "KA": 0.3}) == []


# ── reporting ────────────────────────────────────────────────────────────────

def _subjects():
    """Two trivially-fittable subjects (one-compartment-ish decay)."""
    return [{"subject": i, "doses": [{"time": 0.0, "amt": 100.0}],
             "obs_t": [1.0, 4.0, 8.0, 24.0],
             "obs_c": [9.0, 7.0, 5.0, 2.0], "wt": 70.0, "cov": {}}
            for i in (1, 2, 3, 4)]


def test_assemble_forces_not_converged_and_names_the_parameter():
    """Even if the optimiser reports success, _assemble must overrule it — the
    automatic method arbitration in `auto` rejects non-converged candidates, so
    this is what stops a collapsed fit from winning a model comparison."""
    from app.compute.pk_models import REGISTRY as R
    spec = nlme._PopSpec(R["oral_1cmt"], ["CL"], "proportional", [])
    subs = nlme._prepare_subjects(_subjects())
    theta = {"CL": 0.0, "V": 50.0, "KA": 1.0}          # CL underflowed
    out = nlme._assemble(spec, "FOCE-I", "oral_1cmt", theta, nlme.np.zeros(0),
                         {"CL": 0.1}, 0.1, 0.0, 123.0,
                         [nlme.np.zeros(1) for _ in subs], subs, 16,
                         True, 10)                      # optimiser said success
    assert out["converged"] is False
    assert out["degenerate_params"] == ["CL"]


def test_healthy_fit_has_no_degenerate_key():
    """The key set of a good fit is unchanged (nothing new to break consumers)."""
    res = nlme.population_fit("oral_1cmt", _subjects(), method="focei",
                              iiv_params=["CL"], compute_uncertainty=False)
    assert "degenerate_params" not in res
    assert "escalated_from" not in res


# ── escalation policy ────────────────────────────────────────────────────────

_GOOD = {"theta": {"CL": 5.0, "V": 50.0, "KA": 1.0}, "converged": True,
         "method": "FOCE-I (SAEM-seeded)", "ofv": 100.0}
_BAD = {"theta": {"CL": 0.0, "V": 50.0, "KA": 1.0}, "converged": False,
        "method": "FOCE-I", "ofv": 90.0, "degenerate_params": ["CL"]}


def test_collapse_escalates_once_and_is_labelled(monkeypatch):
    calls = []
    monkeypatch.setattr(nlme, "focei_fit", lambda *a, **k: dict(_BAD))
    monkeypatch.setattr(nlme, "_focei_saem_fit",
                        lambda *a, **k: (calls.append(1), dict(_GOOD))[1])
    res = nlme.population_fit("oral_1cmt", _subjects(), method="focei")
    assert len(calls) == 1                                    # escalated exactly once
    assert "auto-escalated" in res["method"]
    assert res["escalated_from"]["degenerate_params"] == ["CL"]
    assert res["theta"]["CL"] == 5.0


def test_healthy_cold_fit_does_not_pay_for_escalation(monkeypatch):
    """Escalation is cost-proportional: a well-behaved fit must not trigger the
    seeded search, which costs several times more."""
    calls = []
    monkeypatch.setattr(nlme, "focei_fit", lambda *a, **k: dict(_GOOD))
    monkeypatch.setattr(nlme, "_focei_saem_fit",
                        lambda *a, **k: (calls.append(1), dict(_GOOD))[1])
    res = nlme.population_fit("oral_1cmt", _subjects(), method="focei")
    assert calls == []
    assert "auto-escalated" not in res["method"]


def test_escalation_that_also_collapses_reports_the_original_failure(monkeypatch):
    """Never hide a collapse behind a second collapse."""
    monkeypatch.setattr(nlme, "focei_fit", lambda *a, **k: dict(_BAD))
    monkeypatch.setattr(nlme, "_focei_saem_fit", lambda *a, **k: dict(_BAD))
    res = nlme.population_fit("oral_1cmt", _subjects(), method="focei")
    assert res["degenerate_params"] == ["CL"]
    assert "escalated_from" not in res
    assert res["converged"] is False


def test_map_fit_is_not_escalated(monkeypatch):
    """The prior lives in the FOCE-I objective; the SAEM seed stage does not
    carry it, so escalating a MAP fit would silently drop the prior."""
    calls = []
    monkeypatch.setattr(nlme, "focei_fit", lambda *a, **k: dict(_BAD))
    monkeypatch.setattr(nlme, "_focei_saem_fit",
                        lambda *a, **k: (calls.append(1), dict(_GOOD))[1])
    res = nlme.population_fit("oral_1cmt", _subjects(), method="focei",
                              theta_prior={"CL": {"mean": 5.0, "var": 0.1}})
    assert calls == []
    assert res["degenerate_params"] == ["CL"]


def test_other_methods_are_not_escalated(monkeypatch):
    """Only the cold-start default self-heals; an explicitly chosen method is
    reported as it ran."""
    calls = []
    monkeypatch.setattr(nlme, "saem_fit", lambda *a, **k: dict(_BAD))
    monkeypatch.setattr(nlme, "_focei_saem_fit",
                        lambda *a, **k: (calls.append(1), dict(_GOOD))[1])
    res = nlme.population_fit("oral_1cmt", _subjects(), method="saem")
    assert calls == []
    assert res["degenerate_params"] == ["CL"]


def test_escalation_can_be_opted_out(monkeypatch):
    """Bootstrap discards collapsed replicates rather than re-fitting each one,
    so it must be able to skip the escalation cost."""
    calls = []
    monkeypatch.setattr(nlme, "focei_fit", lambda *a, **k: dict(_BAD))
    monkeypatch.setattr(nlme, "_focei_saem_fit",
                        lambda *a, **k: (calls.append(1), dict(_GOOD))[1])
    res = nlme.population_fit("oral_1cmt", _subjects(), method="focei",
                              escalate_on_collapse=False)
    assert calls == []
    assert res["degenerate_params"] == ["CL"]      # still reported, just not re-fitted


def test_bootstrap_opts_out_of_escalation():
    """Pin the call site: a collapsed replicate must not trigger a seeded re-fit."""
    import inspect

    from app.tools import bootstrap_tools
    src = inspect.getsource(bootstrap_tools)
    assert "escalate_on_collapse=False" in src


# ── prevention: the search cannot run theta off the log scale ────────────────

def test_log_clamp_stops_the_underflow_that_caused_the_collapse():
    """exp(-745) is exactly 0.0. Powell reached that point; the decode clamp is
    what stops a clearance of zero being decodable at all."""
    import math

    import numpy as np
    spec = nlme._PopSpec(REGISTRY["oral_2cmt"], ["CL", "VC"], "proportional", [])
    x = nlme._pack(spec, {"CL": 0.25, "VC": 3.5, "Q": 0.4, "VP": 2.8, "KA": 0.3},
                   np.zeros(0), {"CL": 0.1, "VC": 0.1}, 0.3, 0.0)
    x[0] = -745.0                                   # what the optimiser actually did
    cl = nlme._unpack(spec, x)[0]["CL"]
    assert cl > 0.0 and math.isfinite(cl)
    assert cl == pytest.approx(5.0 * 1e-7 / 10)     # default 5 x REL_MIN / margin


def test_clamped_theta_is_still_reported_as_degenerate():
    """The clamp sits one decade OUTSIDE the detection window on purpose: hitting
    the wall must not look like a healthy small estimate."""
    import numpy as np
    spec = nlme._PopSpec(REGISTRY["oral_2cmt"], ["CL", "VC"], "proportional", [])
    x = nlme._pack(spec, {"CL": 0.25, "VC": 3.5, "Q": 0.4, "VP": 2.8, "KA": 0.3},
                   np.zeros(0), {"CL": 0.1, "VC": 0.1}, 0.3, 0.0)
    x[0] = -745.0
    theta = nlme._unpack(spec, x)[0]
    assert nlme._degenerate_thetas(spec, theta) == ["CL"]


def test_values_inside_the_box_decode_bit_identically():
    """Inertness guarantee: min/max is a no-op inside the box, so an already
    validated fit decodes to the exact same float it did before the clamp."""
    import math

    import numpy as np
    spec = nlme._PopSpec(REGISTRY["oral_2cmt"], ["CL", "VC"], "proportional", [])
    base = nlme._pack(spec, {"CL": 0.25, "VC": 3.5, "Q": 0.4, "VP": 2.8, "KA": 0.3},
                      np.zeros(0), {"CL": 0.1, "VC": 0.1}, 0.3, 0.0)
    for cl in (0.25, 0.0104, 1.7e-4, 5.0, 120.0, 1e4):
        x = base.copy()
        x[0] = math.log(cl)
        assert nlme._unpack(spec, x)[0]["CL"] == math.exp(math.log(cl))


def test_box_is_strictly_wider_than_the_detection_window():
    import math
    spec = nlme._PopSpec(REGISTRY["oral_2cmt"], ["CL", "VC"], "proportional", [])
    lo, hi = nlme._theta_log_bounds(spec, "CL")
    ref = float(REGISTRY["oral_2cmt"].defaults["CL"])
    assert math.exp(lo) < ref * nlme._THETA_REL_MIN
    assert math.exp(hi) > ref * nlme._THETA_REL_MAX


def test_model_without_a_default_is_left_unbounded():
    """No default means no reference magnitude; bounding on a guess would be
    worse than not bounding, so that parameter behaves exactly as before."""
    import math
    spec = nlme._PopSpec(REGISTRY["oral_2cmt"], ["CL"], "proportional", [])
    spec.model.defaults.pop("__probe__", None)
    lo, hi = nlme._theta_log_bounds(spec, "__probe__")   # unknown name -> no default
    assert lo == -math.inf and hi == math.inf
