"""Typed schema for the conventions catalog (read-only data; see docs/design/context-engineering-ideas.md)."""
from __future__ import annotations

import string
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

_ID = r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$"
_SEMVER = r"^\d+\.\d+\.\d+$"
Scalar = float | int | str | bool


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Source(_Strict):
    """Where a convention comes from. `implementation_convention` means: how the code behaves, not independently
    sourced; it must never be presented as guidance."""
    kind: Literal["regulatory_guidance", "literature", "owner_decision", "implementation_convention"]
    reference: str = Field(min_length=1)


class Approval(_Strict):
    status: Literal["pending_owner_review", "approved"]
    by: str | None = None
    date: str | None = None

    @model_validator(mode="after")
    def _approved_needs_who_and_when(self) -> Approval:
        if self.status == "approved" and not (self.by and self.date):
            raise ValueError("an approved convention needs `by` and `date`")
        return self


class BranchPoint(_Strict):
    """Set only where a plausible alternative can change a conclusion or materially move an estimate."""
    alternatives: list[str] = Field(min_length=2)
    changes_conclusion_when: str = Field(min_length=1)


class Convention(_Strict):
    id: str = Field(pattern=_ID)
    title: str = Field(min_length=1)
    domain: Literal["nca", "be", "modeling", "uncertainty"]
    applies_to: list[str] = Field(min_length=1)
    statement: str = Field(min_length=1)                 # may contain {name} placeholders filled from `parameters`
    parameters: dict[str, Scalar] = Field(default_factory=dict)
    version: str = Field(pattern=_SEMVER)
    source: Source
    approval: Approval
    tests: list[str] = Field(min_length=1)               # "tests/<file>.py::<test_function>"
    known_gaps: list[str] = Field(default_factory=list)
    branch_point: BranchPoint | None = None

    @model_validator(mode="after")
    def _text_and_parameters_agree(self) -> Convention:
        used = {f for _, f, _, _ in string.Formatter().parse(self.statement) if f}
        if used != set(self.parameters):
            raise ValueError(f"placeholders {sorted(used)} and parameters {sorted(self.parameters)} must match exactly")
        return self


class Catalog(_Strict):
    catalog_version: str = Field(pattern=_SEMVER)
    conventions: list[Convention] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_ids(self) -> Catalog:
        ids = [c.id for c in self.conventions]
        dup = sorted({i for i in ids if ids.count(i) > 1})
        if dup:
            raise ValueError(f"duplicate convention ids: {dup}")
        return self
