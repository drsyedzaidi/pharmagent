"""Base types shared by every versioned workflow record.

Every persisted record is a frozen Pydantic model that forbids unknown fields and carries a schema version, a stable
identifier, a creation time (always UTC), a producer, parent references, and a content hash over everything else. The
hash is computed on construction when omitted and verified when supplied, so an altered record is refused when it is
loaded again. ``verify_integrity()`` and ``validated_copy()`` re-check a record that is already in memory.
"""
from __future__ import annotations

import copy
import re
import unicodedata
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Annotated, Any, Literal

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    ValidationInfo,
    WrapSerializer,
    field_validator,
    model_validator,
)

from .errors import ContractError
from .hashing import canonical_json, sha256_hex

SCHEMA_VERSION = 1
SUPPORTED_SCHEMA_VERSIONS = frozenset({1})



#: controls, format characters (bidi overrides, zero-width), line and paragraph separators. (Lone surrogates in plain
#: string fields are already refused by pydantic itself; JSON content is checked by ``_json_utf8_ok``.)
_FORBIDDEN_CATEGORIES = frozenset({"Cc", "Cf", "Zl", "Zp"})
#: characters that render as nothing (or as blank space) yet are not whitespace or format characters: Hangul fillers,
#: braille blank, Khmer inherent vowels, combining grapheme joiner
_BLANK_LOOKING = frozenset("\u115f\u1160\u3164\uffa0\u2800\u17b4\u17b5\u034f")


def _bad_character(c: str, *, allow_layout: bool = False) -> bool:
    if allow_layout and c in "\n\t":
        return False
    return unicodedata.category(c) in _FORBIDDEN_CATEGORIES or c in _BLANK_LOOKING


def _utf8(value: str) -> str:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("string contains a lone surrogate and is not valid UTF-8 text") from exc
    return value


def _has_visible_character(value: str) -> bool:
    return any(unicodedata.category(c)[0] not in "ZC" for c in value)


def _label(value: str) -> str:
    if value != value.strip():
        raise ValueError("must not have leading or trailing whitespace")
    if any(_bad_character(c) for c in value):
        raise ValueError("must not contain control, format, blank-looking or surrogate characters")
    return value          # non-empty, no edge whitespace and none of the invisible characters: a visible one remains


def _message(value: str) -> str:
    if any(_bad_character(c, allow_layout=True) for c in value):
        raise ValueError("must not contain control, format, blank-looking or surrogate characters")
    if not _has_visible_character(value):
        raise ValueError("must contain a visible character")
    return value


MAX_JSON_BYTES = 64 * 1024


MAX_NESTING = 64
#: Private key for the validation context that skips advisory lints during structural re-validation. A module-level
#: object, not a string, so it cannot be guessed; it is not a security boundary (importing it is possible).
STRUCTURAL = object()
#: Private key marking a LOAD of a stored record (see ``WorkflowContract.load``).
REQUIRE_HASH = object()
#: Hashes are computed over {domain, record_type, schema_version, fields}, so records of different types or schema
#: versions can never share a hash even if their fields happened to coincide.
HASH_DOMAIN = "pharmagent.workflow.record"

_ISO_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,6})?(Z|[+-]\d{2}:\d{2})$")


def _strict_timestamp(value: Any) -> Any:
    """A ``datetime``, or an ISO-8601 string with an explicit offset and at most microsecond precision. Epoch numbers,
    bare dates, naive strings, bytes and over-precise fractions are refused rather than coerced or truncated."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and _ISO_TIMESTAMP.fullmatch(value):
        return value
    raise ValueError("a timestamp must be a datetime or an ISO-8601 string with an offset (at most microseconds)")


Timestamp = Annotated[AwareDatetime, BeforeValidator(_strict_timestamp)]


def to_utc(value: datetime) -> datetime:
    """The same instant as a plain ``datetime`` in UTC (any ``datetime`` subclass is rebuilt as the built-in type)."""
    try:
        u = value.astimezone(UTC)
    except OverflowError as exc:                       # the instant is outside the representable UTC range
        raise ValueError("datetime is outside the representable UTC range") from exc
    return datetime(u.year, u.month, u.day, u.hour, u.minute, u.second, u.microsecond, tzinfo=UTC)


class FrozenMap(Mapping):
    """A read-only, serializable mapping with string keys. Values are deep-frozen on construction (mappings become
    ``FrozenMap``, lists become tuples), so neither the stored data nor a caller's original input can change it. It
    holds only JSON-like values (and frozen models): sets and bytes are refused, so every map hashes and serializes
    deterministically. Nesting deeper than 64 levels is refused, however the nesting was assembled."""

    __slots__ = ("_data", "_height")

    def __init__(self, data: Mapping[str, Any] | None = None, _depth: int = 0):
        if hasattr(self, "_data"):
            raise TypeError("FrozenMap is already initialized and cannot be re-initialized")
        items = dict(data or {})
        if not all(isinstance(k, str) for k in items):
            raise TypeError("FrozenMap keys must be strings")
        frozen = {k: freeze(items[k], _depth + 1) for k in sorted(items)}
        height = 1 + max((_height_of(v) for v in frozen.values()), default=0)
        _too_deep(height)
        object.__setattr__(self, "_data", MappingProxyType(frozen))
        object.__setattr__(self, "_height", height)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("FrozenMap is immutable")

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __eq__(self, other: object) -> bool:
        return thaw(self) == thaw(other) if isinstance(other, Mapping) else NotImplemented

    def __hash__(self) -> int:
        return hash(tuple(self._data.items()))

    def __repr__(self) -> str:
        return f"FrozenMap({thaw(self)!r})"

    def __reduce__(self):
        return (FrozenMap, (thaw(self),))

    def to_plain(self) -> dict[str, Any]:
        """A fresh, mutable copy as plain dicts and lists (for JSON); changing it never affects this map."""
        return thaw(self)


def _too_deep(depth: int) -> None:
    if depth > MAX_NESTING:
        raise ValueError(f"value is nested too deeply (limit {MAX_NESTING} levels)")


def _height_of(value: Any) -> int:
    if type(value) is FrozenMap:
        return value._height
    if isinstance(value, tuple):
        return 1 + max((_height_of(v) for v in value), default=0)
    return 0


def freeze(value: Any, _depth: int = 0) -> Any:
    _too_deep(_depth)
    if type(value) is FrozenMap:                 # a subclass is rebuilt, so it cannot smuggle in its own accessors
        return value
    if isinstance(value, set | frozenset | bytes | bytearray):
        raise TypeError("a FrozenMap holds JSON-like values only: sets or bytes are not allowed")
    if isinstance(value, Mapping):
        return FrozenMap(value, _depth)
    if isinstance(value, list | tuple):
        return tuple(freeze(v, _depth + 1) for v in value)
    return value


def thaw(value: Any, _depth: int = 0) -> Any:
    _too_deep(_depth)
    if isinstance(value, Mapping):
        return {k: thaw(v, _depth + 1) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [thaw(v, _depth + 1) for v in value]
    return value


def _all_utf8(value: Any) -> None:
    if isinstance(value, str):
        _utf8(value)
    elif isinstance(value, Mapping):
        for k, v in value.items():
            _utf8(k)
            _all_utf8(v)
    elif isinstance(value, list | tuple):
        for v in value:
            _all_utf8(v)


def _json_utf8_ok(value: dict[str, Any]) -> dict[str, Any]:
    _all_utf8(value)
    return value


def _json_size_ok(value: dict[str, Any]) -> dict[str, Any]:
    if len(canonical_json(value)) > MAX_JSON_BYTES:         # also raises on NaN / Infinity / non-JSON values
        raise ValueError(f"must serialize to at most {MAX_JSON_BYTES} bytes")
    return value


def _thaw_for_serialization(value: Any, handler: Any) -> Any:
    return handler(thaw(value))


def frozen_map_of(key_type: Any, value_type: Any) -> Any:
    """A validated, immutable mapping field: validated as a plain dict, then frozen; it serializes as a dict."""
    return Annotated[dict[key_type, value_type], Field(max_length=1000), BeforeValidator(thaw),
                     AfterValidator(freeze), WrapSerializer(_thaw_for_serialization)]


ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
Identifier = Annotated[str, StringConstraints(pattern=ID_PATTERN)]
Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
def _reject_bool(value: Any) -> Any:
    if isinstance(value, bool):
        raise ValueError("a boolean is not a number")
    return value


NoBool = BeforeValidator(_reject_bool)
Label = Annotated[str, StringConstraints(min_length=1, max_length=200), AfterValidator(_label)]
ColumnName = Annotated[str, StringConstraints(min_length=1, max_length=128), AfterValidator(_label)]
Message = Annotated[str, StringConstraints(min_length=1, max_length=2000), AfterValidator(_message)]
NonNegInt = Annotated[int, NoBool, Field(ge=0)]
FiniteFloat = Annotated[float, NoBool, Field(allow_inf_nan=False)]
JsonObject = Annotated[dict[str, JsonValue], BeforeValidator(thaw), AfterValidator(_json_utf8_ok),
                       AfterValidator(_json_size_ok), AfterValidator(freeze), WrapSerializer(_thaw_for_serialization)]
IdTuple = Annotated[tuple[Identifier, ...], Field(max_length=1000)]


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")


class Unknown(Frozen):
    """A value the source never recorded. Always explicit: a missing fact is never filled in."""
    status: Literal["unknown", "unavailable"]
    reason: Annotated[str, StringConstraints(min_length=1, max_length=500)]


def unrecorded() -> Unknown:
    return Unknown(status="unknown", reason="not recorded")


class MigrationInfo(Frozen):
    from_schema_version: Annotated[int, Field(ge=0)]       # 0 = produced before these contracts existed
    note: Annotated[str, StringConstraints(min_length=1, max_length=500)]


class WorkflowContract(Frozen):
    schema_version: int = SCHEMA_VERSION
    id: Identifier
    created_at: Timestamp
    producer: Label
    parents: IdTuple = ()
    hash_algorithm: Literal["sha256"] = "sha256"
    content_hash: str = ""
    migration: MigrationInfo | None = None

    @field_validator("schema_version", mode="before")
    @classmethod
    def _supported_version(cls, v: Any) -> int:
        if type(v) is not int or v not in SUPPORTED_SCHEMA_VERSIONS:      # exact int: not 1.0, "1" or True
            raise ValueError(f"unsupported schema_version {v}; supported: {sorted(SUPPORTED_SCHEMA_VERSIONS)}")
        return v

    @field_validator("created_at")
    @classmethod
    def _utc(cls, v):
        return to_utc(v)

    @field_validator("parents")
    @classmethod
    def _unique_parents(cls, v: IdTuple) -> IdTuple:
        if len(set(v)) != len(v):
            raise ValueError("parents must be unique")
        return v

    @model_validator(mode="after")
    def _seal(self, info: ValidationInfo):
        if self.id in self.parents:
            raise ValueError("a record cannot be its own parent: id appears in parents (itself)")
        digest = self.compute_hash()
        if self.content_hash == "":
            if info.context and info.context.get(REQUIRE_HASH):
                raise ValueError("a stored record must carry its content_hash; a missing or blank hash is refused")
            object.__setattr__(self, "content_hash", digest)          # fresh construction: seal now
        elif self.content_hash != digest:
            raise ValueError("content_hash does not match the record content")
        return self

    @classmethod
    def load(cls, data):
        """The entry point for STORED records (JSON text or bytes, or a mapping). Unlike fresh construction it refuses
        a missing or blank ``content_hash`` instead of sealing the content it was given, so tampering plus stripping the
        hash cannot launder a record. Anything read back from storage must come through here."""
        context = {REQUIRE_HASH: True}
        if isinstance(data, str | bytes | bytearray):
            return cls.model_validate_json(data, context=context)
        return cls.model_validate(data, context=context)

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False):
        """A copy is validated from scratch, so ``update`` cannot smuggle in a record that skips its own rules or keeps
        a stale hash: a change must also reset ``content_hash`` to ``""`` so the copy is sealed afresh. Unknown
        field names are refused."""
        fields = type(self).model_fields
        unknown = set(update or {}) - set(fields)
        if unknown:
            raise ContractError(f"unknown fields in update: {sorted(unknown)}")
        data = {name: getattr(self, name) for name in fields}
        data.update(update or {})
        return type(self).model_validate(copy.deepcopy(data) if deep else data)

    def validated_copy(self):
        """A freshly validated copy (content hash and every structural rule re-checked), or ``ContractError``.
        Code that makes decisions must read this copy, never the live object. The copy is rebuilt from the record's JSON
        form, so every field comes back as an exact built-in type (a ``str``, ``int`` or ``datetime`` subclass with a
        lying comparison, smuggled in through ``model_construct``, cannot reach a decision). A record with no stored
        content hash is refused. Policy lints (see ``ReviewRecord``) are skipped, so tightening one later does not void
        old records."""
        try:
            context = {STRUCTURAL: True, REQUIRE_HASH: True}           # REQUIRE_HASH refuses a blank stored hash
            return type(self).model_validate_json(self.model_dump_json(), context=context)
        except Exception as exc:
            raise ContractError(f"integrity check failed: {type(self).__name__} failed re-validation") from exc

    def compute_hash(self) -> str:
        payload = {"domain": HASH_DOMAIN, "record_type": type(self).__name__, "schema_version": self.schema_version,
                   "fields": self.model_dump(mode="json", exclude={"content_hash"})}
        return sha256_hex(canonical_json(payload))

    def verify_integrity(self) -> bool:
        """True if the live object still matches its content hash. The seal is checked when a record is constructed
        or loaded; a forced attribute rewrite (``object.__setattr__``) or ``model_construct`` is caught by this call."""
        try:
            return self.content_hash == self.compute_hash()
        except Exception:                 # content that cannot even be serialized is not intact
            return False

    def canonical_bytes(self) -> bytes:
        return canonical_json(self.model_dump(mode="json"))


def require_unique(values: tuple[Any, ...], what: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{what} must be unique")
