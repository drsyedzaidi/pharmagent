"""Typed schema for the mikebrain rule catalog.

Mirrors the conventions catalog (``app.conventions.schema``): strict, frozen models,
an ``Approval`` that is ``pending_owner_review`` until the owner records who approved
it and when, and a canonical-JSON digest of the whole file for provenance.

A rule's predicate is a closed vocabulary, not an expression: a fixed allow-list of
diagnostic paths, six comparison operators, and an ``any`` / ``all`` quantifier for
per-parameter maps. There is no ``eval``, no free text and no way for a model to
smuggle logic into the catalog.
"""
from __future__ import annotations

import math
import re
from datetime import date as _date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.conventions.schema import Approval

_ID = r"^mb\.[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$"
_SEMVER = r"^\d+\.\d+\.\d+$"
_SHA256 = r"^[0-9a-f]{64}$"
_DATE = r"^\d{4}-\d{2}-\d{2}$"
Scalar = float | int | str | bool

#: diagnostic path -> value kind. ``num`` scalar number, ``bool``, ``str``, ``map_num`` a
#: {parameter: number} map (needs a quantifier). Every path is something
#: ``app.compute.adversarial`` already reads, so a rule never reaches raw data.
ALLOWED_PATHS: dict[str, str] = {
    "nlme_results.converged": "bool",
    "nlme_results.condition_number": "num",
    "nlme_results.shrinkage_pct": "map_num",
    "nlme_results.theta_rse_pct": "map_num",
    "nlme_results.omega_cv_pct": "map_num",
    "pk_model_results.mode": "str",
}
#: results block -> status a block must report before any rule reads it (the same gate
#: the deterministic refutations use).
ROOT_GATES: dict[str, tuple[str, str]] = {
    "nlme_results": ("status", "ok"),
    "pk_model_results": ("status", "ok"),
}
ORDERED_OPS: tuple[str, ...] = ("gt", "ge", "lt", "le")
OPS: tuple[str, ...] = (*ORDERED_OPS, "eq", "ne")
#: a rule-derived finding may never be CRITICAL or HIGH: raising this is an owner
#: decision (docs/agentic_planner_design.md, open question 3), not a catalog edit.
RULE_SEVERITIES: tuple[str, ...] = ("MEDIUM", "LOW")
MAX_CONDITIONS = 4
MAX_EVIDENCE = 8
#: identities that must never appear as an approver. A speed bump, not a security control:
#: approval is the owner's act, reviewed in the pull request that records it.
AUTOMATED_APPROVER_TOKENS: frozenset[str] = frozenset(
    {"claude", "anthropic", "gpt", "llm", "bot", "agent", "mikebrain", "pharmagent", "automation", "loop",
     "assistant", "copilot"})


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _check_iso_date(value: str) -> str:
    if not re.fullmatch(_DATE, value):
        raise ValueError("date must be YYYY-MM-DD")
    try:
        _date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"not a real calendar date: {value}") from exc
    return value


class Condition(_Strict):
    path: str
    op: Literal["gt", "ge", "lt", "le", "eq", "ne"]
    value: Scalar
    quantifier: Literal["any", "all"] | None = None   # required for map paths, forbidden otherwise

    @model_validator(mode="after")
    def _consistent(self) -> Condition:
        kind = ALLOWED_PATHS.get(self.path)
        if kind is None:
            raise ValueError(f"path {self.path!r} is not an allowed diagnostic path; allowed: {sorted(ALLOWED_PATHS)}")
        if (kind == "map_num") != (self.quantifier is not None):
            raise ValueError(f"{self.path}: a quantifier (any|all) is required for per-parameter maps, "
                             "and only for them")
        if kind in ("num", "map_num"):
            v = self.value
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError(f"{self.path}: value must be a finite number")
            if kind == "map_num" and self.op not in ORDERED_OPS:
                raise ValueError(f"{self.path}: per-parameter maps support only {ORDERED_OPS}")
        elif kind == "bool":
            if not isinstance(self.value, bool) or self.op not in ("eq", "ne"):
                raise ValueError(f"{self.path}: needs a boolean value and op eq|ne")
        elif not isinstance(self.value, str) or self.op not in ("eq", "ne"):
            raise ValueError(f"{self.path}: needs a string value and op eq|ne")
        return self


class EvidenceRef(_Strict):
    """A pointer into the owner's private corpus: document hash plus the exact character span and the
    SHA-256 of that span's text. No passage text is stored, so the catalog carries no copyrighted material."""
    doc_sha: str = Field(pattern=_SHA256)
    document: str = Field(min_length=1, max_length=300)      # bibliographic title
    year: int | None = Field(default=None, ge=1800, le=2200)
    source_type: Literal["book", "paper", "guideline"]
    section: str = Field(default="", max_length=160)
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    text_sha256: str = Field(pattern=_SHA256)

    @model_validator(mode="after")
    def _span(self) -> EvidenceRef:
        if self.end <= self.start:
            raise ValueError("evidence span must have end > start")
        return self


class Rule(_Strict):
    id: str = Field(pattern=_ID)
    version: str = Field(pattern=_SEMVER)
    title: str = Field(min_length=1)
    when: list[Condition] = Field(min_length=1, max_length=MAX_CONDITIONS)
    severity: Literal["MEDIUM", "LOW"]
    target: str = Field(min_length=1)
    claim: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    suggested_action: str = Field(min_length=1)
    evidence: list[EvidenceRef] = Field(min_length=1, max_length=MAX_EVIDENCE)
    known_limits: list[str] = Field(default_factory=list)
    proposed_by: str = Field(min_length=1)
    proposed_on: str
    approval: Approval

    @model_validator(mode="after")
    def _provenance(self) -> Rule:
        _check_iso_date(self.proposed_on)
        by = self.approval.by
        if by is not None:
            tokens = set(re.findall(r"[a-z]+", by.lower()))
            hit = sorted(tokens & AUTOMATED_APPROVER_TOKENS)
            if hit:
                raise ValueError(f"approval.by {by!r} names an automated identity ({hit}); "
                                 "only the owner approves a rule")
        return self


class RuleCatalog(_Strict):
    catalog_version: str = Field(pattern=_SEMVER)
    rules: list[Rule] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_ids(self) -> RuleCatalog:
        ids = [r.id for r in self.rules]
        dup = sorted({i for i in ids if ids.count(i) > 1})
        if dup:
            raise ValueError(f"duplicate rule ids: {dup}")
        return self
