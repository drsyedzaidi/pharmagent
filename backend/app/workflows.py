"""Workflow templates — ordered steps with agent/tool assignment and review gates.

When a template runs, the Supervisor routes by the task plan rather than
classifying intent. Gates pause execution for human review at scientific
decision points.

Two template kinds:
* step templates (the default) — a fixed, ordered ``steps`` list the
  orchestrator executes verbatim, pausing at ``gate`` steps;
* ``"kind": "goal"`` templates — a ``goal`` plus an ``allowed_tools`` allow-list.
  An LLM planner proposes each next tool from the diagnostics in state and a
  deterministic policy (app.core.policy) admits it, parks it for human
  approval, or rejects it. Long-running fits and scientific model choices are
  always gated; the budget bounds the loop.
"""
from __future__ import annotations

from typing import Any

WORKFLOWS: dict[str, dict[str, Any]] = {
    "nca_full": {
        "name": "nca_full",
        "description": "End-to-end NCA: load → profile → validate → NCA → adversarial review → QC → report.",
        "steps": [
            {"agent": "data_manager", "tool": "load_dataset", "label": "Load dataset"},
            {"agent": "data_manager", "tool": "profile_pk_dataset", "label": "Profile data"},
            {"agent": "data_manager", "tool": "validate_cdisc", "label": "Validate structure"},
            {"agent": "data_manager", "tool": "generate_spaghetti_plot", "label": "Spaghetti plot"},
            {"agent": "nca", "tool": "compute_nca", "label": "Compute NCA"},
            {"agent": "reviewer", "tool": "adversarial_review", "label": "Adversarial review"},
            {"agent": "qc", "tool": "run_qc", "label": "QC review", "gate": True},
            {"agent": "report", "tool": "generate_report", "label": "Generate report"},
        ],
    },
    "poppk_full": {
        "name": "poppk_full",
        "description": (
            "End-to-end population PK: load → profile → validate → spaghetti → "
            "structural model comparison (human gate) → NLME fit → SCM covariate "
            "build → residual diagnostics → covariate forest → VPC → adversarial "
            "review (human gate) → report. The first gate sits before the NLME "
            "leg so the structural model is confirmed before the expensive fits "
            "commit. The NCA QC checklist (run_qc) is excluded — it scores an NCA "
            "analysis, not a mixed-effects fit. run_engine_comparison is left to "
            "poppk_modeling and run_simest is excluded: it needs a study `design` "
            "the template cannot know and costs several extra real NLME fits."),
        "steps": [
            {"agent": "data_manager", "tool": "load_dataset", "label": "Load dataset"},
            {"agent": "data_manager", "tool": "profile_pk_dataset", "label": "Profile data"},
            {"agent": "data_manager", "tool": "validate_cdisc", "label": "Validate structure"},
            {"agent": "data_manager", "tool": "generate_spaghetti_plot", "label": "Spaghetti plot"},
            # Gate: confirm the structural model before the expensive NLME leg runs.
            {"agent": "modeler", "tool": "fit_pk_model", "label": "Compare structural models",
             "args": {"compare": True}, "gate": True},
            # method is left at the tool default (focei); "auto" is job-backed/opt-in.
            {"agent": "modeler", "tool": "run_nlme", "label": "Population (NLME) fit"},
            {"agent": "modeler", "tool": "run_scm", "label": "Covariate model (SCM)"},
            {"agent": "modeler", "tool": "run_diagnostics", "label": "Residual diagnostics"},
            {"agent": "modeler", "tool": "run_covariate_forest", "label": "Covariate forest"},
            {"agent": "modeler", "tool": "run_vpc", "label": "VPC / goodness-of-fit"},
            {"agent": "reviewer", "tool": "adversarial_review",
             "label": "Adversarial review", "gate": True},
            {"agent": "report", "tool": "generate_report", "label": "Generate report"},
        ],
    },
    "poppk_modeling": {
        "name": "poppk_modeling",
        "description": ("Structural model selection then cross-engine confirmation: "
                        "load → profile → fit (compare) → cross-engine comparison "
                        "→ adversarial review (human gate). Ends at the review gate; "
                        "the NCA-specific QC checklist and report are intentionally "
                        "excluded as they do not assess a structural/cross-engine fit."),
        "steps": [
            {"agent": "data_manager", "tool": "load_dataset", "label": "Load dataset"},
            {"agent": "data_manager", "tool": "profile_pk_dataset", "label": "Profile data"},
            {"agent": "modeler", "tool": "fit_pk_model", "label": "Fit structural models",
             "args": {"compare": True}},
            {"agent": "modeler", "tool": "run_engine_comparison",
             "label": "Cross-engine comparison",
             "args": {"engines": ["pharmagent_focei", "nlmixr2", "monolix"]}},
            {"agent": "reviewer", "tool": "adversarial_review",
             "label": "Adversarial review", "gate": True},
        ],
    },
    "poppk_goal": {
        "name": "poppk_goal",
        "kind": "goal",
        "description": (
            "Goal-driven population PK: a planner proposes each next step from the "
            "diagnostics in state and a deterministic policy admits, gates or rejects "
            "it. Same tool set as poppk_full. Autonomous: load, profile, validate, "
            "spaghetti, structural comparison (no model pinned), residual diagnostics, "
            "covariate forest, VPC, adversarial review. Gated (human approve/reject "
            "via state.pending_tool): run_nlme, run_scm, any fit that pins a method, "
            "model, error or covariate model, and the report. Under the keyless mock "
            "provider the planner walks the poppk_full order."),
        "goal": (
            "Build a defensible population PK model for the loaded dataset: confirm the "
            "structural model, fit it as a mixed-effects model, test covariates, check "
            "residual diagnostics and the VPC, have the result adversarially reviewed, and "
            "deliver the report. Stop when the goal is met or no admissible step remains."),
        "allowed_tools": [
            "load_dataset", "profile_pk_dataset", "validate_cdisc", "generate_spaghetti_plot",
            "fit_pk_model", "run_nlme", "run_scm", "run_diagnostics", "run_covariate_forest",
            "run_vpc", "adversarial_review", "generate_report",
        ],
        "budget": {"max_steps": 20, "max_gated": 6},
    },
    "protocol_simulation": {
        "name": "protocol_simulation",
        "description": (
            "Governed protocol simulation and dose selection from an approved "
            "population PK fit: load approved fit → propose simulation plan "
            "(human gate) → dose sweep → trial PTA across covariate bands and "
            "regimens → special-population / pediatric simulations (human gate "
            "before expensive uncertainty job) → outer-loop parameter uncertainty "
            "propagation → adversarial review (human gate) → regulatory report."
        ),
        # Documentary only — nothing reads this key.  The real precondition
        # is enforced by load_approved_fit, which raises if nlme_results or
        # review_results are absent/invalid.
        "requires": ["nlme_results.status == 'ok'"],
        "steps": [
            {"agent": "modeler", "tool": "load_approved_fit",
             "label": "Load approved PopPK fit"},
            {"agent": "simulator", "tool": "propose_simulation_plan",
             "label": "Propose simulation plan (human review gate)", "gate": True},
            {"agent": "simulator", "tool": "run_dose_sweep",
             "label": "Dose sweep across regimens"},
            {"agent": "simulator", "tool": "simulate_trial_pta",
             "label": "Trial simulation & PTA by band"},
            {"agent": "simulator", "tool": "simulate_special_populations",
             "label": "Special-population / pediatric exposures", "gate": True},
            {"agent": "simulator", "tool": "propagate_parameter_uncertainty",
             "label": "Parameter uncertainty in PTA (job)"},
            {"agent": "reviewer", "tool": "adversarial_review",
             "label": "Adversarial review of simulation claims", "gate": True},
            {"agent": "report", "tool": "generate_report",
             "label": "Protocol simulation report section"},
        ],
    },
}


def get_workflow(name: str) -> dict[str, Any]:
    if name not in WORKFLOWS:
        raise KeyError(f"unknown workflow: {name}")
    return WORKFLOWS[name]
