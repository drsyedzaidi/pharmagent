"""Evidence data model: frozen records, status vocabularies, reason/flag codes, errors.

store.py's SQLite tables are the ledger; these dataclasses are the typed view the
tools and routes work with. Every value is deterministic output or a human-approved
proposal carrying a source locator.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

from app.core.store import _json as canonical_json

PARSE_STATUSES, VERDICTS = ("parsed", "no_text", "failed"), ("include", "exclude", "unsure")
DECISION_STATUSES, DECISIONS = ("proposed", "approved", "rejected"), ("approved", "rejected")
SCREEN_METHODS, EXTRACTORS, VERIFY_STATUSES = ("llm", "keyword"), ("llm", "table", "manual"), ("pass", "fail")
CONFIG_KINDS, DECIMAL_STYLES, LOCATOR_KINDS = ("criteria", "template"), (".", ","), ("text", "table")

# Verifier reason codes (verify_reason column). Empty string = verified.
REASON_OK, REASON_PAGE_OUT_OF_RANGE, REASON_SNIPPET_NOT_FOUND = "", "page_out_of_range", "snippet_not_found"
REASON_VALUE_NOT_IN_SNIPPET, REASON_CELL_MISMATCH = "value_not_in_snippet", "cell_mismatch"
REASON_NO_TEXT, REASON_BAD_NUMBER, REASON_SOURCE_CHANGED = "no_text", "bad_number", "source_changed"
# QC flag codes (flags_json entries: {code, severity, message, field, group, origin}).
FLAG_DUPLICATE, FLAG_TABLE_LOW_CONFIDENCE, FLAG_OUT_OF_RANGE = "duplicate", "table_low_confidence", "out_of_range"
FLAG_UNIT_MISSING, FLAG_UNIT_MISMATCH, FLAG_REQUIRED_MISSING = "unit_missing", "unit_mismatch", "required_missing"
FLAG_NON_NUMERIC, FLAG_DECIMAL_AMBIGUOUS = "non_numeric", "decimal_ambiguous"


class EvidenceStateError(RuntimeError):
    """A state transition that is no longer allowed (e.g. deciding twice)."""


class IngestError(ValueError):
    """An uploaded file was rejected before anything was written."""


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def body_hash(body: Any) -> str:
    """sha256 of canonical JSON (key order and whitespace independent)."""
    return sha256_text(canonical_json(body))


def record_hash(*, source_sha256: str, template_hash: str, field: str, group: str, value: str,
                unit: str | None, locator: dict[str, Any], extractor: str) -> str:
    """Claim identity: excludes status/flags/verify and the verifier-set span/literal."""
    loc = {k: v for k, v in locator.items() if k not in ("span", "literal")}
    return body_hash({"source_sha256": source_sha256, "template_hash": template_hash, "field": field,
                      "group": group, "value": value, "unit": unit, "locator": loc, "extractor": extractor})


@dataclass(frozen=True)
class Locator:
    page: int
    kind: str = "text"
    snippet: str = ""
    span: list[int] | None = None
    literal: str | None = None
    cell: dict[str, Any] | None = None
    column: int | None = None
    decimal_style: str = "."

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Locator:
        keys = cls.__dataclass_fields__
        return cls(**{k: v for k, v in data.items() if k in keys})


@dataclass(frozen=True)
class VerifyResult:
    status: str                     # 'pass' | 'fail'
    reason: str = REASON_OK
    locator: dict[str, Any] | None = None
    literal: str | None = None
    span: list[int] | None = None


@dataclass(frozen=True)
class Source:
    id: str
    session_id: str
    sha256: str
    filename: str
    pdf_path: str
    parse_path: str
    pages: int
    title: str | None
    metadata: dict[str, str]
    text_chars: int
    parse_status: str
    decimal_style: str
    licence_note: str | None
    uploaded_by: str
    uploaded_at: str


@dataclass(frozen=True)
class Screen:
    id: str
    session_id: str
    source_id: str
    criteria_hash: str
    verdict: str
    score: float
    reasoning: str
    criteria: list[dict[str, Any]]
    method: str
    status: str
    decided_by: str | None
    decided_at: str | None
    decision_reason: str | None
    created_at: str


@dataclass(frozen=True)
class Record:
    id: str
    session_id: str
    source_id: str
    template_hash: str
    field: str
    group_label: str
    value: str
    value_num: float | None
    unit: str | None
    locator: dict[str, Any]
    extractor: str
    verify_status: str
    verify_reason: str
    flags: list[dict[str, Any]]
    status: str
    decided_by: str | None
    decided_at: str | None
    decision_reason: str | None
    record_hash: str
    created_at: str


@dataclass(frozen=True)
class Dataset:
    id: str
    session_id: str
    template_hash: str
    long_path: str
    wide_path: str
    long_sha256: str
    wide_sha256: str
    record_hashes: list[str]
    n_records: int
    n_sources: int
    exported_by: str
    exported_at: str
