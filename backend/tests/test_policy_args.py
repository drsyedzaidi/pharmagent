"""F-13 / F-24: planner-proposed arguments are validated against the tool schema.

* F-13: empty / wrongly typed model arguments no longer slip past the model-key
  check (they used to fall back to the default candidate set unreviewed).
* F-24: ``policy.decide`` and ``ToolRegistry.execute`` share one validator, so
  unknown keys, wrong JSON types and out-of-range numerics are refused.
"""
from __future__ import annotations

import pytest

from app.core import policy
from app.core.audit import AuditChain
from app.core.pharmstate import PharmState
from app.tools.base import ToolContext
from app.tools.builtins import default_registry

_REG = default_registry()
_READY = PharmState(pk_model_results={"status": "ok"}, nlme_results={"status": "ok"})


def _decide(tool: str, args: dict, state: PharmState = _READY):
    roster = {n: _REG.get(n) for n in _REG.names()}
    return policy.decide(tool, args, state, roster=roster, history=[], budget=policy.Budget())


# ---- F-13 ------------------------------------------------------------------
@pytest.mark.parametrize("args", [
    {"models": []}, {"models": ()}, {"model_key": ""}, {"models": ""},
    {"models": 5}, {"models": {"a": 1}}, {"model_key": ["oral_1cmt"]},
    {"model_key": 7}, {"models": [None]},
])
def test_fit_pk_model_rejects_empty_and_wrong_typed_model_args(args):
    d = _decide("fit_pk_model", args)
    assert d.allow is False and d.mode == policy.MODE_REJECTED, d


@pytest.mark.parametrize("tool", ["run_scm"])
def test_scm_candidates_wrong_type_rejected(tool):
    assert _decide(tool, {"candidates": "WT"}).allow is False
    assert _decide(tool, {"candidates": 3}).allow is False


def test_argument_less_comparison_stays_autonomous_by_design():
    # Documented design (workflows.poppk_goal): the structural comparison with no
    # model pinned is autonomous; the template-level gate belongs to poppk_full.
    for args in ({"compare": True}, {}):
        d = _decide("fit_pk_model", args)
        assert d.allow and d.mode == policy.MODE_AUTONOMOUS, (args, d)


# ---- F-24 ------------------------------------------------------------------
@pytest.mark.parametrize("tool,args", [
    ("run_vpc", {"n_rep": 1000000}),                       # unknown key
    ("run_vpc", {"stratify_by": 5}),                       # wrong type
    ("run_vpc", {"dose_normalize": "yes"}),
    ("compute_nca", {"lloq": -50}),
    ("compute_nca", {"dataset_id": 3}),
    ("generate_spaghetti_plot", {"log_scale": "true"}),
    ("run_diagnostics", {"cwres_interaction": 1}),
    ("run_covariate_forest", {"ci_level": 5}),
    ("run_covariate_forest", {"ci_level": "0.9"}),
    ("run_covariate_forest", {"source": "bogus"}),
    ("run_bioequivalence", {"ci_lower": 1, "ci_upper": 999}),
    ("calc_be_sample_size", {"cv_intra": 0.2, "alpha": 3}),
    ("calc_be_sample_size", {"cv_intra": 0.2, "alpha": 0}),
    ("generate_flexplot", {"y": "DV", "n_bins": 10**9}),
    ("run_clinsim", {"n_subjects": 10**9}),
    ("run_clinsim", {"n_subjects": True}),
    ("run_simest", {"n_rep": 10**6}),
    ("run_sir", {"n_resample": 10**9}),
    ("indirect_comparison", {"ab": {"ci_lower": 5, "ci_upper": 1}, "cb": {}, "scale": "log"}),
])
def test_policy_rejects_invalid_arguments(tool, args):
    d = _decide(tool, args)
    assert d.allow is False and d.mode == policy.MODE_REJECTED, (tool, args, d)
    assert d.reason


def test_policy_reason_names_the_offending_argument():
    d = _decide("run_vpc", {"n_rep": 5})
    assert "n_rep" in d.reason


@pytest.mark.parametrize("tool,args", [
    ("run_vpc", {}),
    ("run_vpc", {"stratify_by": None, "dose_normalize": False, "x_by": "time",
                 "exposure_check": False, "blq_check": False}),
    ("compute_nca", {"is_iv": True}),
    ("run_covariate_forest", {"source": "scm", "ci_level": 0.9}),
    ("generate_flexplot", {"y": "DV", "n_bins": 10}),
    ("fit_pk_model", {"compare": True, "models": ["oral_1cmt", "oral_2cmt"]}),
])
def test_policy_still_admits_valid_arguments(tool, args):
    assert _decide(tool, args).allow is True


def test_registry_execute_enforces_the_same_validator():
    audit = AuditChain()
    with pytest.raises(ValueError, match="n_rep"):
        _REG.execute("run_vpc", state=PharmState(), ctx=ToolContext(), args={"n_rep": 5},
                     audit=audit, timestamp="t")
    with pytest.raises(ValueError, match="n_bins"):
        _REG.execute("generate_flexplot", state=PharmState(), ctx=ToolContext(),
                     args={"y": "DV", "n_bins": 10**9}, audit=audit, timestamp="t")
    assert len(audit.entries) == 0       # refused before anything is computed/audited


def test_framework_confirm_key_is_accepted_by_the_registry_but_not_from_a_planner():
    """take_pending_tool re-runs an approved tool with confirm=True even when its schema
    (run_nlme, run_scm, run_engine_comparison) does not declare it."""
    from app.tools.arg_validation import validate_args
    from app.tools.base import FRAMEWORK_ARG_KEYS
    schema = _REG.get("run_nlme").input_schema
    assert validate_args(schema, {"confirm": True}) is not None
    assert validate_args(schema, {"confirm": True}, allow_extra=FRAMEWORK_ARG_KEYS) is None
    assert _decide("run_nlme", {"confirm": True}).allow is False
