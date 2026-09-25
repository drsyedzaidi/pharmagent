"""Statistician agent: parametric vs non-parametric advice.

Compute-level rules are checked on synthetic samples with known distributions
(log-normal -> log-scale parametric; heavy right tail on the log scale ->
non-parametric; Tmax -> always rank-based). Integration checks route a
statistics question to the agent through the keyless MockLLM and verify the
write scope and the derived-only content of ``stats_advice``.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest

from app.agents.definitions import AGENTS, DESCRIPTIONS
from app.agents.supervisor import KEYWORDS, Supervisor
from app.compute.stats_advice import (
    advise,
    choose_scale_and_family,
    distribution_profile,
    recommend_tests,
)
from app.core.llm import MockLLM
from app.core.orchestrator import Orchestrator
from app.core.pharmstate import AGENT_WRITE_FIELDS, PharmState, PharmStateError, apply_writes
from app.core.store import SessionStore

SAMPLES = Path(__file__).parent.parent / "sample_data"
RNG = np.random.default_rng(20260925)


def _lognormal(n: int, cv: float = 0.3) -> list[float]:
    sigma = np.sqrt(np.log(1 + cv**2))
    return [float(v) for v in RNG.lognormal(mean=np.log(100.0), sigma=sigma, size=n)]


def _orch() -> Orchestrator:
    counter = itertools.count()
    return Orchestrator(llm=MockLLM(), clock=lambda: f"t{next(counter)}",
                        store=SessionStore(":memory:"))


# ── compute rules ─────────────────────────────────────────────────────────

def test_lognormal_exposure_is_parametric_on_log_scale():
    # Arrange: 40 log-normal values, CV 30% (typical Cmax)
    prof = distribution_profile(_lognormal(40))

    # Act
    decision = choose_scale_and_family("Cmax", prof)

    # Assert
    assert prof["n"] == 40 and prof["geometric_mean"] == pytest.approx(100.0, rel=0.15)
    assert prof["shapiro_log_p"] >= 0.05
    assert decision == {"scale": "log", "family": "parametric", "rationale": decision["rationale"]}


def test_heavy_tail_after_log_is_non_parametric():
    # Arrange: log-scale mixture with a far right tail -> Shapiro rejects on log scale
    base = _lognormal(30, cv=0.2)
    values = base + [float(v) for v in np.array(base[:6]) * 40.0]
    prof = distribution_profile(values)

    # Act
    decision = choose_scale_and_family("AUC_last", prof)

    # Assert
    assert prof["shapiro_log_p"] < 0.05
    assert decision["family"] == "non-parametric" and decision["scale"] == "rank"


def test_tmax_is_always_rank_based():
    prof = distribution_profile(_lognormal(40))            # even if it looked normal
    decision = choose_scale_and_family("Tmax", prof)
    assert decision["family"] == "non-parametric" and decision["scale"] == "rank"


def test_small_group_keeps_log_parametric_convention():
    prof = distribution_profile([1.0, 2.0, 40.0, 3.0, 2.5])   # n=5, wild but under-powered
    decision = choose_scale_and_family("Cmax", prof)
    assert decision["family"] == "parametric" and "too small" in decision["rationale"]


def test_nonpositive_values_block_log_transform():
    prof = distribution_profile([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    decision = choose_scale_and_family("Cmax", prof)
    assert prof["n_nonpositive"] == 1 and decision["family"] == "non-parametric"


@pytest.mark.parametrize("n_groups,paired,family,expect", [
    (1, False, "parametric", "geometric mean"),
    (2, True, "parametric", "paired t-test"),
    (2, True, "non-parametric", "Wilcoxon signed-rank"),
    (2, False, "parametric", "two-sample t-test"),
    (2, False, "non-parametric", "Mann-Whitney"),
    (3, False, "parametric", "one-way ANOVA"),
    (3, False, "non-parametric", "Kruskal-Wallis"),
    (3, True, "non-parametric", "Friedman"),
])
def test_design_maps_to_test(n_groups, paired, family, expect):
    out = recommend_tests(family=family, n_groups=n_groups, paired=paired, equal_variance=True)
    assert expect in out["primary"]


def test_unequal_variance_switches_to_welch():
    out = recommend_tests(family="parametric", n_groups=2, paired=False, equal_variance=False)
    assert out["primary"].startswith("Welch")
    out3 = recommend_tests(family="parametric", n_groups=3, paired=False, equal_variance=False)
    assert "Welch ANOVA" in out3["primary"] and "Games-Howell" in out3["primary"]


def test_advise_bioequivalence_overrides_rank_pick_for_exposures():
    # Arrange: heavy-tailed Cmax in a 2-period crossover flagged as BE
    base = _lognormal(16, cv=0.2)
    heavy = base + [float(v) for v in np.array(base[:5]) * 40.0]
    exposures = {"Cmax": {"R": heavy[:10], "T": heavy[10:]},
                 "Tmax": {"R": [1.0, 2.0, 1.5] * 3 + [1.0], "T": [1.0, 2.0, 2.0] * 3 + [1.5]}}
    design = {"n_subjects": 21, "group_var": "TRT", "groups": ["R", "T"],
              "n_per_group": {"R": 10, "T": 11}, "paired": True, "design_label": "crossover"}

    # Act
    res = advise(exposures=exposures, design=design, context="bioequivalence")

    # Assert: regulatory log-ANOVA primary for Cmax, rank-based kept as sensitivity;
    # Tmax stays rank-based; the BE recommendation is present.
    cmax = res["metrics"]["Cmax"]
    assert cmax["family"] == "parametric" and "Overridden for bioequivalence" in cmax["rationale"]
    assert "Wilcoxon" in cmax["sensitivity_test"]
    assert res["metrics"]["Tmax"]["family"] == "non-parametric"
    assert any(r["topic"] == "bioequivalence" for r in res["recommendations"])
    assert res["design"]["n_groups"] == 2


def test_advise_covariates_follow_family():
    exposures = {"Cmax": {"all": _lognormal(20)}}
    design = {"n_subjects": 20, "group_var": None, "groups": ["all"],
              "n_per_group": {"all": 20}, "paired": False, "design_label": "single-group"}
    covs = [{"name": "WT", "kind": "continuous", "n_levels": None},
            {"name": "SEX", "kind": "categorical", "n_levels": 2}]
    res = advise(exposures=exposures, design=design, covariates=covs)
    by_name = {c["name"]: c for c in res["covariates"]}
    assert "Pearson" in by_name["WT"]["recommendation"]
    assert "Fisher" in by_name["SEX"]["recommendation"]
    assert res["caveats"][-1].startswith("These are recommendations")


# ── agent wiring ──────────────────────────────────────────────────────────

def test_statistician_registered_everywhere():
    assert "statistician" in AGENTS and "statistician" in DESCRIPTIONS and "statistician" in KEYWORDS
    assert AGENT_WRITE_FIELDS["statistician"] == {"stats_advice", "widgets"}


def test_supervisor_routes_statistics_questions():
    sup = Supervisor(MockLLM())
    assert sup.route("which test should I use, parametric or non-parametric?")[0] == "statistician"
    assert sup.route("how should I analyze this data")[0] == "statistician"
    # existing routes are not hijacked
    assert sup.route("compute NCA AUC and Cmax")[0] == "nca"
    assert sup.route("run dose proportionality power model")[0] == "dose_prop"


def test_only_statistician_may_write_stats_advice():
    st = PharmState()
    with pytest.raises(PharmStateError):
        apply_writes(st, "nca", {"stats_advice": {"status": "ok"}})
    out = apply_writes(st, "statistician", {"stats_advice": {"status": "ok"}})
    assert out.stats_advice == {"status": "ok"}


def test_chat_produces_advice_from_nca_dose_groups():
    orch = _orch()
    sid = orch.create_session().id
    orch.chat(sid, f"load dataset {SAMPLES / 'oral_pk.csv'}")
    orch.chat(sid, "compute nca")
    out = orch.chat(sid, "is this parametric or non-parametric, what statistical test?")

    assert out["agent"] == "statistician"
    assert any(m.startswith("Statistical advice") for m in out["messages"])
    adv = out["state"]["stats_advice"]
    assert adv["status"] == "ok" and adv["design"]["source"] == "nca"
    assert adv["design"]["group_var"] == "dose" and adv["design"]["n_groups"] == 2
    assert adv["metrics"]["Tmax"]["family"] == "non-parametric"
    assert adv["metrics"]["Cmax"]["scale"] == "log"
    assert any(r["topic"] == "dose proportionality" for r in adv["recommendations"])
    # derived statistics only — never per-subject rows
    assert "subjects" not in adv and all(not isinstance(v, list) or all(
        not isinstance(x, dict) or "subject" not in x for x in v) for v in adv.values())
    # second ask is idempotent under the mock (advice present -> no tool call)
    again = orch.chat(sid, "which statistical test?")
    assert again["tool_calls"] == []


def test_chat_detects_crossover_and_covariates():
    orch = _orch()
    sid = orch.create_session().id
    orch.chat(sid, f"load dataset {SAMPLES / 'be_crossover.csv'}")
    be = orch.chat(sid, "how should I analyse this, parametric or nonparametric?")["state"]["stats_advice"]
    assert be["design"]["design_label"] == "crossover" and be["design"]["paired"] is True
    assert be["design"]["groups"] == ["R", "T"]
    assert any(r["topic"] == "bioequivalence" for r in be["recommendations"])
    assert be["metrics"]["Cmax"]["family"] == "parametric"          # regulatory override

    sid2 = orch.create_session().id
    orch.chat(sid2, f"load dataset {SAMPLES / 'cov_pk.csv'}")
    cov = orch.chat(sid2, "statistical analysis plan please")["state"]["stats_advice"]
    assert cov["design"]["design_label"] == "single-group"
    assert {c["name"] for c in cov["covariates"]} == {"CRCL", "AGE"}


def test_advice_without_dataset_is_reported_not_raised():
    orch = _orch()
    sid = orch.create_session().id
    out = orch.chat(sid, "which statistical test, parametric or non-parametric?")
    assert out["agent"] == "statistician"
    assert any(m.startswith("Could not run recommend_statistics") for m in out["messages"])
    assert out["state"]["stats_advice"] is None
