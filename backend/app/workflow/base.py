"""Base types shared by every versioned workflow record.

Every persisted record is a frozen Pydantic model that forbids unknown fields and carries a schema version, a stable
identifier, a creation time (always UTC), a producer, parent references, and a content hash over everything else. The
hash is created only by ``create()`` / ``replace()``; every other validation path (the constructor, ``model_validate*``,
``load``, ``model_copy``) requires an existing, correct hash, so an altered or hash-stripped record is refused. The hash
is an unkeyed integrity check, not proof of authenticity. ``verify_integrity()`` and ``validated_copy()`` re-check a
record that is already in memory.
"""
from __future__ import annotations

import copy
import re
import unicodedata
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Annotated, Any, ClassVar, Literal

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



#: controls, format characters (bidi overrides, zero-width), line and paragraph separators, private-use and unassigned
#: code points. (Lone surrogates in plain string fields are already refused by pydantic itself; JSON content is
#: checked by ``_json_utf8_ok``.)
_FORBIDDEN_CATEGORIES = frozenset({"Cc", "Cf", "Zl", "Zp", "Co", "Cn"})
#: Unicode Default_Ignorable_Code_Point ranges (rendered as nothing). Written as integers so this file stays ASCII.
_DEFAULT_IGNORABLE = (
    (0x00AD, 0x00AD), (0x034F, 0x034F), (0x061C, 0x061C), (0x115F, 0x1160), (0x17B4, 0x17B5), (0x180B, 0x180F),
    (0x200B, 0x200F), (0x202A, 0x202E), (0x2060, 0x206F), (0x3164, 0x3164), (0xFE00, 0xFE0F), (0xFEFF, 0xFEFF),
    (0xFFA0, 0xFFA0), (0xFFF0, 0xFFF8), (0x16FE4, 0x16FE4), (0x1BCA0, 0x1BCA3), (0x1D173, 0x1D17A),
    (0xE0000, 0xE0FFF),
)
#: characters that render as nothing (or as blank space) yet are not whitespace or format characters: Hangul fillers,
#: braille blank, Khmer inherent vowels, combining grapheme joiner
_BLANK_LOOKING = frozenset("\u115f\u1160\u3164\uffa0\u2800\u17b4\u17b5\u034f")


def _default_ignorable(c: str) -> bool:
    cp = ord(c)
    return any(lo <= cp <= hi for lo, hi in _DEFAULT_IGNORABLE)


def _bad_character(c: str, *, allow_layout: bool = False) -> bool:
    if allow_layout and c in "\n\t":
        return False
    return unicodedata.category(c) in _FORBIDDEN_CATEGORIES or c in _BLANK_LOOKING or _default_ignorable(c)


def _utf8(value: str) -> str:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("string contains a lone surrogate and is not valid UTF-8 text") from exc
    return value


def _has_visible_character(value: str) -> bool:
    """At least one letter, number, punctuation or symbol (combining marks and separators alone do not count)."""
    return any(unicodedata.category(c)[0] in "LNPS" for c in value)


def _label(value: str) -> str:
    if value != value.strip():
        raise ValueError("must not have leading or trailing whitespace")
    if any(_bad_character(c) for c in value):
        raise ValueError("must not contain control, format, blank-looking or surrogate characters")
    if not _has_visible_character(value):
        raise ValueError("must contain a letter, number, punctuation or symbol")
    return value


def _message(value: str) -> str:
    if any(_bad_character(c, allow_layout=True) for c in value):
        raise ValueError("must not contain control, format, blank-looking or surrogate characters")
    if not _has_visible_character(value):
        raise ValueError("must contain a visible character")
    return value


MAX_JSON_BYTES = 64 * 1024


MAX_NESTING = 64
#: Private validation-context key that marks the one place a hash is created: ``create()`` and ``replace()``. Every
#: other validation path (constructor, ``model_validate*``, ``load``, ``model_copy``) requires an existing, correct
#: hash. A module-level object so it cannot be guessed; importing it is possible, so it is not a security boundary.
SEAL = object()
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
        u = datetime.astimezone(value, UTC)               # the real method, not a subclass override
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
def _numbers_only(value: Any) -> Any:
    """int or float, not bool and not a numeric string: stored bytes stay canonical (one spelling per value)."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("a number is required (not a boolean or a string)")
    return value


NonNegInt = Annotated[int, Field(ge=0, strict=True)]            # strict: not "3", 3.0 or True
FiniteFloat = Annotated[float, BeforeValidator(_numbers_only), Field(allow_inf_nan=False)]
JsonObject = Annotated[dict[str, JsonValue], BeforeValidator(thaw), AfterValidator(_json_utf8_ok),
                       AfterValidator(_json_size_ok), AfterValidator(freeze), WrapSerializer(_thaw_for_serialization)]
IdTuple = Annotated[tuple[Identifier, ...], Field(max_length=1000)]


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always", hide_input_in_errors=True)


class Unknown(Frozen):
    """A value the source never recorded. Always explicit: a missing fact is never filled in."""
    status: Literal["unknown", "unavailable"]
    reason: Annotated[str, StringConstraints(min_length=1, max_length=500), AfterValidator(_message)]


def unrecorded() -> Unknown:
    return Unknown(status="unknown", reason="not recorded")


class MigrationInfo(Frozen):
    from_schema_version: Annotated[int, Field(ge=0, strict=True)]       # 0 = produced before these contracts existed
    note: Annotated[str, StringConstraints(min_length=1, max_length=500), AfterValidator(_message)]


_RECORD_TYPES: dict[str, tuple[str, str]] = {}      # constant -> (module, qualname) that claimed it


class WorkflowContract(Frozen):
    """Common header of every persisted record. Each concrete record class declares an explicit, unique, stable
    ``RECORD_TYPE`` string that is part of its hash; renaming a Python class therefore never changes stored hashes."""

    RECORD_TYPE: ClassVar[str]

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        constant = vars(cls).get("RECORD_TYPE")
        if not isinstance(constant, str) or not constant.strip():
            raise TypeError(f"{cls.__name__} must declare its own non-empty RECORD_TYPE string")
        owner = _RECORD_TYPES.get(constant)
        here = (cls.__module__, cls.__qualname__)
        if owner is not None and owner != here:
            raise TypeError(f"RECORD_TYPE {constant!r} is already used by {owner[0]}.{owner[1]}")
        _RECORD_TYPES[constant] = here

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
            if not (info.context and info.context.get(SEAL)):
                raise ValueError(
                    "content_hash is required: build a NEW record with create() or replace(), and read a stored "
                    "one with load(); a missing or blank hash is never sealed implicitly")
            object.__setattr__(self, "content_hash", digest)
        elif self.content_hash != digest:
            raise ValueError("content_hash does not match the record content")
        return self

    # -- the only ways to obtain a record -------------------------------------------------------------------------
    @classmethod
    def create(cls, **fields: Any):
        """Build a NEW record and seal it. This is the one named creation API: the constructor, ``model_validate*``,
        ``load`` and ``model_copy`` all require an existing, correct ``content_hash`` and never seal on their own."""
        if "content_hash" in fields:
            raise TypeError("create() seals the record itself; do not pass content_hash")
        if "RECORD_TYPE" not in vars(cls):
            raise TypeError(f"{cls.__name__} is abstract: it declares no RECORD_TYPE")
        return cls.model_validate({**fields, "content_hash": ""}, context={SEAL: True})

    def replace(self, **changes: Any):
        """A NEW sealed record equal to this one except for ``changes`` (the named replacement API). The original is
        untouched. Unknown fields are refused."""
        fields = type(self).model_fields
        unknown = set(changes) - set(fields)
        if unknown:
            raise ContractError(f"unknown fields: {sorted(unknown)}")
        data = {name: getattr(self, name) for name in fields if name != "content_hash"}
        data.update(changes)
        return type(self).create(**data)

    @classmethod
    def load(cls, data):
        """Read a STORED record (JSON text or bytes, or a mapping). The hash must be present and correct; there is no
        sealing here, so tampering plus stripping the hash cannot launder a record. ``model_validate`` and
        ``model_validate_json`` behave the same way: they are all verification paths."""
        if isinstance(data, str | bytes | bytearray):
            return cls.model_validate_json(data)
        return cls.model_validate(data)

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False):
        """An exact, re-validated copy. ``update`` cannot be used to change content: a changed record keeps a stale
        hash and is refused. Use ``replace()`` to make a changed record. Unknown field names are refused."""
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
        content hash is refused, and the advisory lints (see ``ReviewRecord``) are not applied because nothing is
        being created."""
        try:
            return type(self).model_validate_json(self.model_dump_json())
        except Exception as exc:
            raise ContractError(f"integrity check failed: {type(self).__name__} failed re-validation") from exc

    def compute_hash(self) -> str:
        payload = {"domain": HASH_DOMAIN, "record_type": self.RECORD_TYPE, "schema_version": self.schema_version,
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
