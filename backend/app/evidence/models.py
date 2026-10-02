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

# Verifier reason codes (verify_reason column). REASON_OK (empty string) = verified.
REASON_OK = ""
REASON_EMPTY_SNIPPET, REASON_SNIPPET_TOO_LONG = "empty_snippet", "snippet_too_long"
REASON_PAGE_OUT_OF_RANGE, REASON_NON_ASCII_DIGITS = "page_out_of_range", "non_ascii_digits"
REASON_NO_NUMBER, REASON_NUMBER_MISMATCH = "no_number", "number_mismatch"
REASON_INTEGER_EXPECTED, REASON_SNIPPET_NOT_ON_PAGE = "integer_expected", "snippet_not_on_page"
REASON_TABLE_CELL_MISMATCH, REASON_SOURCE_CHANGED = "table_cell_mismatch", "source_changed"
REASON_TEXT_NOT_IN_SNIPPET = "text_not_in_snippet"        # text field: value absent from its own snippet
REASON_CODES = (REASON_OK, REASON_EMPTY_SNIPPET, REASON_SNIPPET_TOO_LONG, REASON_PAGE_OUT_OF_RANGE,
                REASON_NON_ASCII_DIGITS, REASON_NO_NUMBER, REASON_NUMBER_MISMATCH, REASON_INTEGER_EXPECTED,
                REASON_SNIPPET_NOT_ON_PAGE, REASON_TABLE_CELL_MISMATCH, REASON_SOURCE_CHANGED,
                REASON_TEXT_NOT_IN_SNIPPET)
# QC flag codes (flags_json entries: {code, severity warn|error, message, field, group, origin script|llm}).
FLAG_REQUIRED_MISSING, FLAG_OUT_OF_RANGE, FLAG_NON_INTEGER_N = "required_missing", "out_of_range", "non_integer_n"
FLAG_DUPLICATE, FLAG_UNIT_MISMATCH, FLAG_UNIT_MISSING = "duplicate", "unit_mismatch", "unit_missing"
FLAG_SE_NONPOSITIVE, FLAG_CI_INVERTED, FLAG_COMPOSITE_CELL = "se_nonpositive", "ci_inverted", "composite_cell"
FLAG_TABLE_LOW_CONFIDENCE, FLAG_VERIFY_FAIL, FLAG_NON_NUMERIC = "table_low_confidence", "verify_fail", "non_numeric"
FLAG_SUSPICIOUS_INSTRUCTION = "suspicious_instruction_text"   # value/snippet reads like an instruction
FLAG_CODES = (FLAG_REQUIRED_MISSING, FLAG_OUT_OF_RANGE, FLAG_NON_INTEGER_N, FLAG_DUPLICATE, FLAG_UNIT_MISMATCH,
              FLAG_UNIT_MISSING, FLAG_SE_NONPOSITIVE, FLAG_CI_INVERTED, FLAG_COMPOSITE_CELL,
              FLAG_TABLE_LOW_CONFIDENCE, FLAG_VERIFY_FAIL, FLAG_NON_NUMERIC, FLAG_SUSPICIOUS_INSTRUCTION)
FLAG_SEVERITIES, FLAG_ORIGINS = ("warn", "error"), ("script", "llm")


class EvidenceStateError(RuntimeError):
    """A state transition that is no longer allowed (e.g. deciding twice)."""


class IngestError(ValueError):
    """An uploaded file was rejected before anything was written."""


def as_mapping(record_like: Any) -> dict[str, Any]:
    """Read-only dict view of a proposal dict or a frozen dataclass (Record, Source, ...)."""
    if isinstance(record_like, dict):
        return record_like
    if hasattr(record_like, "__dataclass_fields__"):
        return asdict(record_like)
    raise TypeError(f"unsupported record type {type(record_like).__name__}")


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
