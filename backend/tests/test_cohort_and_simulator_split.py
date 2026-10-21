"""Architectural boundary tests for ADR-002:
1. `app.compute.cohort` is pure domain compute (no imports from `app.tools`, `app.core`, `app.agents`).
2. `app.engines.runner` does not import upward from `app.tools`.
3. `pkmodel_tools.TOOLS` owns only `modeler` tools; `simulator_tools.TOOLS` owns only `simulator` tools.
4. `default_registry()` registers both tool sets and `pkmodel_tools` preserves backward-compatible re-exports.
"""
from __future__ import annotations

import ast
from pathlib import Path

from app.compute import cohort
from app.engines import runner as engine_runner
from app.tools import pkmodel_tools, simulator_tools
from app.tools.builtins import default_registry


def _imported_modules(py_file: Path) -> set[str]:
    tree = ast.parse(py_file.read_text(encoding="utf-8"))
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mods.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module)
    return mods


def test_compute_cohort_has_no_upward_imports():
    mods = _imported_modules(Path(cohort.__file__))
    forbidden = [m for m in mods if m.startswith(("app.tools", "app.core", "app.agents", "app.engines"))]
    assert forbidden == [], f"app.compute.cohort must not import from upper layers: {forbidden}"


def test_engines_runner_does_not_import_tools():
    mods = _imported_modules(Path(engine_runner.__file__))
    forbidden = [m for m in mods if m.startswith("app.tools")]
    assert forbidden == [], f"app.engines.runner must not import from app.tools: {forbidden}"


def test_pkmodel_and_simulator_tools_partitioned_by_agent():
    modeler_agents = {t.agent for t in pkmodel_tools.TOOLS}
    simulator_agents = {t.agent for t in simulator_tools.TOOLS}
    assert modeler_agents == {"modeler"}
    assert simulator_agents == {"simulator"}

    expected_modeler = {
        "fit_pk_model",
        "list_pk_models",
        "run_vpc",
        "run_diagnostics",
        "run_covariate_forest",
        "run_nlme",
        "run_prior_check",
        "run_scm",
        "forecast_map",
    }
    expected_simulator = {
        "simulate_pk_profile",
        "run_dose_sweep",
        "run_clinsim",
        "run_exposure_forest",
        "run_special_population",
        "run_individual_exposures",
        "run_pediatric_simulation",
    }
    assert {t.name for t in pkmodel_tools.TOOLS} == expected_modeler
    assert {t.name for t in simulator_tools.TOOLS} == expected_simulator


def test_default_registry_and_backward_compat_reexports():
    reg = default_registry()
    for tool in pkmodel_tools.TOOLS + simulator_tools.TOOLS:
        registered = reg.get(tool.name)
        assert registered is not None
        assert registered.agent == tool.agent

    assert pkmodel_tools._build_subjects is cohort.build_subjects
    assert pkmodel_tools._dataset_lloq is cohort.dataset_lloq
    assert pkmodel_tools._cov_eval_points is cohort.cov_eval_points
    assert pkmodel_tools.simulate_pk_profile is simulator_tools.simulate_pk_profile
    assert pkmodel_tools.run_dose_sweep is simulator_tools.run_dose_sweep
    assert pkmodel_tools.run_clinsim is simulator_tools.run_clinsim
    assert pkmodel_tools.run_exposure_forest is simulator_tools.run_exposure_forest
    assert pkmodel_tools.run_special_population is simulator_tools.run_special_population
    assert pkmodel_tools.run_individual_exposures is simulator_tools.run_individual_exposures
    assert pkmodel_tools.run_pediatric_simulation is simulator_tools.run_pediatric_simulation
