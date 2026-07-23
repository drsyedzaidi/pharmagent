"""PharmState — the typed communication bus.

Agents never call each other. They read and write a single shared state object,
and each agent may only write the fields it owns (enforced here). This prevents
one agent from clobbering another's results and makes the data flow auditable.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class StudyInfo(BaseModel):
    """Compound/study metadata for CTD regulatory reports.

    Populated by the caller via POST /sessions/{id}/report/272 body or chat.
    Every field defaults to empty string so partial population is safe.
    """
    drug_name: str = ""
    sponsor: str = ""
    study_id: str = ""
    route: str = "oral"
    indication: str = ""
    pop_description: str = "healthy adult volunteers"
    dose_range: str = ""
    matrix: str = "plasma"
    assay_lloq: str = ""


class PharmStateError(Exception):
    """Raised when an agent writes to a field it does not own."""


class PharmState(BaseModel):
    """Shared analysis state. Extend with new fields as agents are added."""

    # --- session / routing -------------------------------------------------
    session_id: str = ""
    last_agent: str | None = None
    workflow_name: str | None = None
    current_step: int = 0

    # --- data (Data Manager) ----------------------------------------------
    dataset_id: str | None = None
    dataset_path: str | None = None
    dataset_metadata: dict[str, Any] | None = None  # schema-only, never raw rows
    data_quality: dict[str, Any] | None = None

    # --- NCA (NCA Agent) ---------------------------------------------------
    nca_parameters: list[dict[str, Any]] | None = None  # per-subject
    nca_summary: dict[str, Any] | None = None           # dose-group summary

    # --- QC (QC Agent) -----------------------------------------------------
    qc_verdict: str | None = None        # PASS | CONDITIONAL PASS | FAIL
    qc_issues: list[dict[str, Any]] | None = None
    qc_checklist: list[dict[str, Any]] | None = None

    # --- bioequivalence (BE Agent) ----------------------------------------
    be_results: dict[str, Any] | None = None

    # --- dose proportionality (Dose-Prop Agent) ---------------------------
    dose_prop_results: dict[str, Any] | None = None

    # --- compartmental modeling (Compartmental Agent) ---------------------
    compartmental_results: dict[str, Any] | None = None

    # --- population PK two-stage (PopPK Agent) ----------------------------
    poppk_results: dict[str, Any] | None = None

    # --- structural PK model library (Modeler Agent) ----------------------
    pk_model_results: dict[str, Any] | None = None
    nlme_results: dict[str, Any] | None = None         # FOCE-I / SAEM mixed-effects fit
    scm_results: dict[str, Any] | None = None          # stepwise covariate modeling
    forecast_results: dict[str, Any] | None = None     # MAP/TDM Bayesian forecast
    vpc_results: dict[str, Any] | None = None          # GOF / VPC diagnostics
    diagnostics_results: dict[str, Any] | None = None  # IWRES / CWRES / npd residual diagnostics
    forest_results: dict[str, Any] | None = None       # covariate GMR forest plot
    engine_comparison_results: dict[str, Any] | None = None  # cross-engine model comparison
    dose_sweep_results: dict[str, Any] | None = None   # dose-comparison simulation
    clinsim_results: dict[str, Any] | None = None      # clinical trial simulation / PTA
    exposure_forest_results: dict[str, Any] | None = None  # simulated exposure covariate forest
    simest_results: dict[str, Any] | None = None
    bootstrap_results: dict[str, Any] | None = None       # non-parametric bootstrap CIs
    sir_results: dict[str, Any] | None = None             # sampling importance resampling CIs
    profile_results: dict[str, Any] | None = None         # log-likelihood profile CIs

    # --- reporting (Report Agent) -----------------------------------------
    report_path: str | None = None
    report_sections: dict[str, Any] | None = None

    # --- adversarial review (Reviewer Agent) -----------------------------------
    review_results: dict[str, Any] | None = None   # findings, goal, goal_met, counts

    # --- regulatory (Regulatory Agent) ----------------------------------------
    study_info: StudyInfo | None = None
    regulatory_report_path: str | None = None
    regulatory_refs: list[dict[str, Any]] | None = None

    # --- placeholders for later phases (PKPD / E-R / sim) -------------------
    model_results: dict[str, Any] | None = None
    covariate_results: dict[str, Any] | None = None
    simulation_results: dict[str, Any] | None = None

    # --- visualization data (pre- and post-NCA plots) ---------------------
    spaghetti_data: dict[str, Any] | None = None
    nca_plot_data: dict[str, Any] | None = None
    flexplot_data: dict[str, Any] | None = None   # exploratory flexplot geometry

    # --- free-form artifacts shown in the UI ------------------------------
    widgets: list[dict[str, Any]] = Field(default_factory=list)


# Per-agent write-access whitelist. The orchestrator enforces this on every
# state mutation; an agent attempting to write outside its set raises.
AGENT_WRITE_FIELDS: dict[str, set[str]] = {
    "supervisor": {"last_agent", "workflow_name", "current_step", "session_id"},
    "data_manager": {"dataset_id", "dataset_path", "dataset_metadata", "data_quality",
                     "widgets", "spaghetti_data", "flexplot_data"},
    "nca": {"nca_parameters", "nca_summary", "widgets", "nca_plot_data"},
    "be": {"be_results", "widgets"},
    "dose_prop": {"dose_prop_results", "widgets"},
    "compartmental": {"compartmental_results", "widgets"},
    "poppk": {"poppk_results", "covariate_results", "widgets"},
    "modeler": {"pk_model_results", "nlme_results", "scm_results", "forecast_results",
                "vpc_results", "diagnostics_results", "engine_comparison_results",
                "forest_results", "widgets"},
    "qc": {"qc_verdict", "qc_issues", "qc_checklist"},
    "reviewer": {"review_results"},
    "report": {"report_path", "report_sections"},
    "simulator": {"simulation_results", "dose_sweep_results", "simest_results",
                  "clinsim_results", "exposure_forest_results",
                  "bootstrap_results", "sir_results", "profile_results", "widgets"},
    "regulatory": {"study_info", "regulatory_report_path", "regulatory_refs"},
}


def apply_writes(state: PharmState, agent: str, writes: dict[str, Any]) -> PharmState:
    """Apply ``writes`` to ``state`` if ``agent`` owns every targeted field.

    Returns a NEW state (immutable update). Raises PharmStateError on violation.
    """
    allowed = AGENT_WRITE_FIELDS.get(agent, set())
    illegal = set(writes) - allowed
    if illegal:
        raise PharmStateError(
            f"Agent '{agent}' may not write {sorted(illegal)}; allowed: {sorted(allowed)}"
        )
    return state.model_copy(update=writes)
