"""Workflow templates — ordered steps with agent/tool assignment and review gates.

When a template runs, the Supervisor routes by the task plan rather than
classifying intent. Gates pause execution for human review at scientific
decision points.
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
             "args": {"engines": ["pharmagent_focei", "nlmixr2"]}},
            {"agent": "reviewer", "tool": "adversarial_review",
             "label": "Adversarial review", "gate": True},
        ],
    },
}


def get_workflow(name: str) -> dict[str, Any]:
    if name not in WORKFLOWS:
        raise KeyError(f"unknown workflow: {name}")
    return WORKFLOWS[name]
