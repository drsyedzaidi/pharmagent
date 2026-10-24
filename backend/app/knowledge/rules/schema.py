"""Typed schema for the mikebrain rule catalog.

Follows the conventions catalog's shape (strict, frozen models, an ``Approval`` that is
``pending_owner_review`` until the owner records who approved it and when, a canonical-JSON
digest of the whole file) without importing it: the conventions package may be imported only
by startup, the NCA tool and run provenance (a conventions test pins that), so the small pieces
needed here are defined locally.

A rule's predicate is a closed vocabulary, not an expression: a fixed allow-list of
diagnostic paths, six comparison operators, and an ``any`` / ``all`` quantifier for
per-parameter maps. There is no ``eval``, no free text and no way for a model to
smuggle logic into the catalog.
"""
from __future__ import annotations

import math
import re
import unicodedata
from datetime import date as _date
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

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
#: decision (open question 3 in the planner design doc), not a catalog edit.
RULE_SEVERITIES: tuple[str, ...] = ("MEDIUM", "LOW")
MAX_CONDITIONS = 4
MAX_EVIDENCE = 8
#: identities that must never appear as an approver. A speed bump, not a security control:
#: approval is the owner's act, reviewed in the pull request that records it.
AUTOMATED_APPROVER_TOKENS: frozenset[str] = frozenset(
    {"claude", "anthropic", "gpt", "llm", "bot", "agent", "mikebrain", "pharmagent", "automation", "loop",
     "assistant", "copilot", "chatgpt", "dependabot", "gemini"})


#: size caps on the owner's free text. They stop an accidentally pasted passage from becoming a catalog entry;
#: `rules add` additionally refuses text that repeats a long run of the cited passage verbatim.
MAX_TITLE, MAX_CLAIM, MAX_RATIONALE, MAX_ACTION, MAX_LIMIT = 200, 400, 800, 400, 300


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def finite_float(x: Any) -> float | None:
    """``x`` as a finite float, else None. A bool is not a number, and an int too large for a float
    (``10**400``) is not finite here rather than an OverflowError."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    try:
        f = float(x)
    except OverflowError:
        return None
    return f if math.isfinite(f) else None


class Approval(_Strict):
    """``approved`` means the owner reviewed the rule and its evidence in the change that records it. It is
    not a cryptographic signature."""
    status: Literal["pending_owner_review", "approved"]
    by: str | None = None
    date: str | None = None

    @model_validator(mode="after")
    def _who_and_when_only_when_approved(self) -> Approval:
        if self.status == "pending_owner_review":
            if self.by is not None or self.date is not None:
                raise ValueError("a pending_owner_review rule cannot name an approver or a date")
            return self
        if not (self.by and self.date):
            raise ValueError("an approved rule needs `by` and `date`")
        _check_iso_date(self.date)
        return self


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
            if finite_float(self.value) is None:
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
    SHA-256 of that span's text. Only the bibliographic title and year are stored from the source; no passage
    text and no section heading, so the catalog carries no copyrighted material."""
    doc_sha: str = Field(pattern=_SHA256)
    document: str = Field(min_length=1, max_length=300)      # bibliographic title
    year: int | None = Field(default=None, ge=1800, le=2200)
    source_type: Literal["book", "paper", "guideline"]
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    text_sha256: str = Field(pattern=_SHA256)

    @model_validator(mode="after")
    def _span(self) -> EvidenceRef:
        if self.end <= self.start:
            raise ValueError("evidence span must have end > start")
        return self


def _approver_tokens(by: str) -> set[str]:
    """Lowercase alphabetic words of ``by`` after NFKC folding and dropping invisible format characters, so
    full-width letters and zero-width joiners do not hide a token."""
    folded = unicodedata.normalize("NFKC", by)
    visible = "".join(c for c in folded if unicodedata.category(c) != "Cf")
    return set(re.findall(r"[a-z]+", visible.lower()))


class Rule(_Strict):
    # Tuples, not lists: the process-wide cached catalog cannot be mutated in place.
    id: str = Field(pattern=_ID)
    version: str = Field(pattern=_SEMVER)
    title: str = Field(min_length=1, max_length=MAX_TITLE)
    when: tuple[Condition, ...] = Field(min_length=1, max_length=MAX_CONDITIONS)
    severity: Literal["MEDIUM", "LOW"]
    target: str = Field(min_length=1, max_length=MAX_TITLE)
    claim: str = Field(min_length=1, max_length=MAX_CLAIM)
    rationale: str = Field(min_length=1, max_length=MAX_RATIONALE)
    suggested_action: str = Field(min_length=1, max_length=MAX_ACTION)
    evidence: tuple[EvidenceRef, ...] = Field(min_length=1, max_length=MAX_EVIDENCE)
    known_limits: tuple[Annotated[str, Field(min_length=1, max_length=MAX_LIMIT)], ...] = ()
    proposed_by: str = Field(min_length=1)
    proposed_on: str
    approval: Approval

    @model_validator(mode="after")
    def _provenance(self) -> Rule:
        _check_iso_date(self.proposed_on)
        by = self.approval.by
        if by is not None:
            if not any(c.isalpha() for c in by):
                raise ValueError("approval.by must name a person")
            hit = sorted(_approver_tokens(by) & AUTOMATED_APPROVER_TOKENS)
            if hit:
                raise ValueError(f"approval.by {by!r} names an automated identity ({hit}); "
                                 "only the owner approves a rule")
        if self.approval.date is not None and self.approval.date < self.proposed_on:   # ISO dates sort as text
            raise ValueError("a rule cannot be approved before it was proposed")
        return self


class RuleCatalog(_Strict):
    catalog_version: str = Field(pattern=_SEMVER)
    rules: tuple[Rule, ...] = ()

    @model_validator(mode="after")
    def _unique_ids(self) -> RuleCatalog:
        ids = [r.id for r in self.rules]
        dup = sorted({i for i in ids if ids.count(i) > 1})
        if dup:
            raise ValueError(f"duplicate rule ids: {dup}")
        return self
