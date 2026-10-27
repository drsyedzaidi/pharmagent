"""The seven scientific stages, the human-review gates, and the allowed stage transitions."""
from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import model_validator

from .base import Label, Message, WorkflowContract


class WorkflowStage(StrEnum):
    DATA_CURATION = "data_curation"
    EDA = "eda"
    NCA = "nca"
    MODEL_DEVELOPMENT = "model_development"
    MODEL_EVALUATION = "model_evaluation"
    SIMULATION = "simulation"
    REPORTING = "reporting"


class ReviewGate(StrEnum):
    CLEANED_DATASET = "cleaned_dataset"
    NCA_EXCLUSIONS_BLQ = "nca_exclusions_blq"
    BASE_STRUCTURAL_MODEL = "base_structural_model"
    COVARIATE_MODEL = "covariate_model"
    FINAL_MODEL = "final_model"
    SIMULATION_SCENARIO = "simulation_scenario"
    FINAL_REPORT = "final_report"


S = WorkflowStage
GATE_STAGE: dict[ReviewGate, WorkflowStage] = {
    ReviewGate.CLEANED_DATASET: S.DATA_CURATION,
    ReviewGate.NCA_EXCLUSIONS_BLQ: S.NCA,
    ReviewGate.BASE_STRUCTURAL_MODEL: S.MODEL_DEVELOPMENT,
    ReviewGate.COVARIATE_MODEL: S.MODEL_DEVELOPMENT,
    ReviewGate.FINAL_MODEL: S.MODEL_EVALUATION,
    ReviewGate.SIMULATION_SCENARIO: S.SIMULATION,
    ReviewGate.FINAL_REPORT: S.REPORTING,
}

#: Forward moves only. A population analysis may skip NCA; an NCA-only analysis may go straight to reporting.
#: Going back (to repair or refit) is a rollback and a repeat of the same stage is a rerun — both need a reason.
ALLOWED_ADVANCES: dict[WorkflowStage | None, frozenset[WorkflowStage]] = {
    None: frozenset({S.DATA_CURATION}),
    S.DATA_CURATION: frozenset({S.EDA}),
    S.EDA: frozenset({S.NCA, S.MODEL_DEVELOPMENT}),
    S.NCA: frozenset({S.MODEL_DEVELOPMENT, S.REPORTING}),
    S.MODEL_DEVELOPMENT: frozenset({S.MODEL_EVALUATION}),
    S.MODEL_EVALUATION: frozenset({S.SIMULATION, S.REPORTING}),
    S.SIMULATION: frozenset({S.REPORTING}),
    S.REPORTING: frozenset(),
}
_ORDER = list(WorkflowStage)


def _stage(value: object) -> WorkflowStage | None:
    """A WorkflowStage for an enum member or a plain ``str`` value; anything else (including ``str`` subclasses, whose
    comparison methods could lie) is refused."""
    if isinstance(value, WorkflowStage):
        return value
    if type(value) is str:
        try:
            return WorkflowStage(value)
        except ValueError:
            return None
    return None


def is_allowed(frm: WorkflowStage | str | None, to: WorkflowStage | str, kind: str) -> bool:
    if kind not in ("advance", "rerun", "rollback"):
        return False
    target = _stage(to)
    source = None if frm is None else _stage(frm)
    if target is None or (frm is not None and source is None):         # an unrecognised stage is never allowed
        return False
    if kind == "advance":
        return target in ALLOWED_ADVANCES.get(source, frozenset())
    if source is None:
        return False
    if kind == "rerun":
        return source == target
    return _ORDER.index(target) < _ORDER.index(source)


class StageTransition(WorkflowContract):
    from_stage: WorkflowStage | None
    to_stage: WorkflowStage
    kind: Literal["advance", "rerun", "rollback"]
    actor: Label
    reason: Message | None = None

    @model_validator(mode="after")
    def _check_transition(self):
        if not is_allowed(self.from_stage, self.to_stage, self.kind):
            frm = self.from_stage.value if self.from_stage else "start"
            raise ValueError(f"{self.kind} {frm} -> {self.to_stage.value} is not an allowed transition")
        if self.kind != "advance" and not self.reason:
            raise ValueError(f"a {self.kind} needs a reason")
        return self
