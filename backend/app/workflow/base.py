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
import functools
import json
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
    create_model,
    field_validator,
    model_validator,
)

from .errors import ContractError, safe_errors
from .hashing import canonical_json, sha256_hex

SCHEMA_VERSION = 1
SUPPORTED_SCHEMA_VERSIONS = frozenset({1})



#: Always refused: controls, format characters (bidi overrides, zero-width), line and paragraph separators. Lone
#: surrogates in plain string fields are already refused by pydantic itself; JSON content is checked by
#: ``_json_utf8_ok``. Two policies exist. LOAD semantics (every validation path except ``create()``/``replace()``)
#: refuse only what can never be acceptable, and accept ZWNJ, ZWJ and VS16 anywhere. The CREATION policy (new or
#: changed values) additionally refuses private-use (Co) and unassigned (Cn) code points, joiners outside the supported
#: contexts below, and over-long mark and joiner runs. Those extra rules depend on the interpreter's Unicode database or
#: are judgement calls, so a stored record written on a newer Python must stay loadable on an older one.
_ALWAYS_BAD_CATEGORIES = frozenset({"Cc", "Cf", "Zl", "Zp"})
_CREATE_BAD_CATEGORIES = frozenset({"Co", "Cn"})
#: Unicode Default_Ignorable_Code_Point ranges (rendered as nothing), plus U+16FE4 (Khitan small script filler, which is
#: not Default_Ignorable but is a blank-rendering filler). Written as integers so this file stays ASCII.
_DEFAULT_IGNORABLE = (
    (0x00AD, 0x00AD), (0x034F, 0x034F), (0x061C, 0x061C), (0x115F, 0x1160), (0x17B4, 0x17B5), (0x180B, 0x180F),
    (0x200B, 0x200F), (0x202A, 0x202E), (0x2060, 0x206F), (0x3164, 0x3164), (0xFE00, 0xFE0F), (0xFEFF, 0xFEFF),
    (0xFFA0, 0xFFA0), (0xFFF0, 0xFFF8), (0x16FE4, 0x16FE4), (0x1BCA0, 0x1BCA3), (0x1D173, 0x1D17A),
    (0xE0000, 0xE0FFF),
)
_IGNORABLE = frozenset(cp for lo, hi in _DEFAULT_IGNORABLE for cp in range(lo, hi + 1))
#: characters that render as nothing (or as blank space) yet are not whitespace or format characters: Hangul fillers,
#: braille blank, Khmer inherent vowels, combining grapheme joiner
_BLANK_LOOKING = frozenset(map(chr, (0x115F, 0x1160, 0x3164, 0xFFA0, 0x2800, 0x17B4, 0x17B5, 0x034F)))

# --- the supported contextual-joiner subset (creation policy). NOT IDNA CONTEXTJ: that rule needs the full Unicode
# Joining_Type table, which this layer does not carry. Everything below is a small, pinned, documented subset.
_ZWNJ, _ZWJ, _VS16, _KEYCAP = 0x200C, 0x200D, 0xFE0F, 0x20E3
_JOINERS = (_ZWNJ, _ZWJ, _VS16)
_VIRAMA = 9                                   # canonical combining class of Indic viramas
MAX_MARK_RUN = 5                              # consecutive combining marks (letter + nukta + virama + vowel signs fit)
MAX_ZWJ_LINKS = 7                             # emoji ZWJ links in one sequence (a family has 3)
_KEYCAP_BASES = frozenset("#*0123456789")
#: blocks whose Symbol-other characters are emoji: Miscellaneous Symbols and Dingbats, and the emoji planes
_EMOJI_BLOCKS = ((0x2600, 0x27BF), (0x1F000, 0x1FAFF))
_SKIN_TONES = (0x1F3FB, 0x1F3FF)
#: Persian/Arabic letters by joining behaviour, pinned from the Unicode ArabicShaping data (a SUBSET of the script):
#: dual-joining letters join on both sides, right-joining letters join only to the letter before them.
_DUAL_JOINING = frozenset(
    {0x626, 0x628, 0x649, 0x64A, 0x67E, 0x686, 0x6A9, 0x6AF, 0x6CC}
    | set(range(0x62A, 0x62F)) | set(range(0x633, 0x63B)) | set(range(0x641, 0x648)))
_RIGHT_JOINING = frozenset({0x627, 0x629, 0x648, 0x698} | set(range(0x622, 0x626)) | set(range(0x62F, 0x633)))
_ASCII_STRICT = re.compile(r"[\x00-\x1f\x7f]")
_ASCII_LAYOUT = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")           # newline and tab allowed
_ASCII_VISIBLE = re.compile(r"[!-~]")                                # every printable ASCII non-space is L, N, P or S
_TEXT_REFUSED = "must not contain control, format, blank-looking, private-use or surrogate characters"


def _pictograph(c: str) -> bool:
    cp = ord(c)
    return unicodedata.category(c) == "So" and any(lo <= cp <= hi for lo, hi in _EMOJI_BLOCKS)


def _emoji_element(c: str) -> bool:
    """A character that can sit inside an emoji sequence: a pictograph, a skin-tone modifier, or the emoji selector."""
    return bool(c) and (_pictograph(c) or _SKIN_TONES[0] <= ord(c) <= _SKIN_TONES[1] or ord(c) == _VS16)


def _script(c: str) -> str:
    return unicodedata.name(c, "").partition(" ")[0]


def _after_script_virama(value: str, i: int) -> bool:
    """``value[i]`` directly follows a virama (combining class 9) that follows a letter of the same script, with only
    other combining marks (a nukta) between letter and virama. The mark-run cap bounds how many there can be."""
    j = i - 1                      # j == -1 reads the last character, but then k < 0 below and the answer is False
    if unicodedata.combining(value[j]) != _VIRAMA:
        return False
    k = j - 1
    while k >= 0 and unicodedata.category(value[k]) == "Mn" and unicodedata.combining(value[k]) != _VIRAMA:
        k -= 1
    return k >= 0 and unicodedata.category(value[k]).startswith("L") and _script(value[k]) == _script(value[j])


def _zwnj_between_persian_letters(prev: str, nxt: str) -> bool:
    return bool(prev and nxt) and ord(prev) in _DUAL_JOINING and ord(nxt) in _DUAL_JOINING | _RIGHT_JOINING


def check_text(value: str, *, allow_layout: bool, creating: bool, labels: bool = False) -> None:
    """Refuse invisible, control and private-use characters. Plain ASCII takes a single compiled-regex pass; other text
    is scanned per character. ``creating`` selects the creation policy (see above) instead of load semantics. ``labels``
    (identifier-like text) additionally refuses every Unicode space separator except the ASCII space, under both
    policies."""
    if value.isascii():
        if (_ASCII_LAYOUT if allow_layout else _ASCII_STRICT).search(value):
            raise ValueError(_TEXT_REFUSED)
        return
    prev = ""
    marks = links = 0
    last = len(value) - 1
    for i, c in enumerate(value):
        cp = ord(c)
        if allow_layout and c in "\n\t":
            prev, marks, links = c, 0, 0
            continue
        if cp in _JOINERS:
            if creating:
                nxt = value[i + 1] if i < last else ""
                if cp == _VS16:
                    allowed = (prev in _KEYCAP_BASES and nxt == chr(_KEYCAP)) or (bool(prev) and _pictograph(prev))
                elif _after_script_virama(value, i):
                    allowed = True
                elif cp == _ZWNJ:
                    allowed = _zwnj_between_persian_letters(prev, nxt)
                else:
                    allowed = _emoji_element(prev) and bool(nxt) and _pictograph(nxt)
                    links += 1
                    allowed = allowed and links <= MAX_ZWJ_LINKS
                if not allowed:
                    raise ValueError(_TEXT_REFUSED)
            prev, marks = c, 0
            continue
        category = unicodedata.category(c)
        if category in _ALWAYS_BAD_CATEGORIES or c in _BLANK_LOOKING or cp in _IGNORABLE:
            raise ValueError(_TEXT_REFUSED)
        if labels and category == "Zs" and c != " ":
            raise ValueError(_TEXT_REFUSED)
        if creating:
            if category in _CREATE_BAD_CATEGORIES:
                raise ValueError(_TEXT_REFUSED)
            marks = marks + 1 if category[0] == "M" else 0
            if marks > MAX_MARK_RUN:
                raise ValueError(_TEXT_REFUSED)
            if links and (not _emoji_element(c) or (_pictograph(c) and prev != chr(_ZWJ))):
                links = 0                                   # a pictograph not joined by ZWJ starts a new sequence
        prev = c


def _utf8(value: str) -> str:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        valid = False
    else:
        valid = True
    if not valid:                                    # raised outside the except block: nothing is chained
        raise ValueError("string contains a lone surrogate and is not valid UTF-8 text")
    return value


def _has_visible_character(value: str) -> bool:
    """At least one letter, number, punctuation or symbol (combining marks and separators alone do not count)."""
    if value.isascii():
        return bool(_ASCII_VISIBLE.search(value))
    return any(unicodedata.category(c)[0] in "LNPS" for c in value)


def sealing_class(info: ValidationInfo) -> Any:
    """The validation context's seal marker (the record class being sealed by ``create()``/``replace()``), else
    ``None`` for any other context. A record seals a blank hash only when this IS its own class, so a junk marker, a
    superclass or a subclass never seals anything."""
    context = info.context
    return context.get(SEAL) if isinstance(context, Mapping) else None


def creation_policy(info: ValidationInfo) -> bool:
    """True when the value is being validated under the CREATION policy: a new record (``create()``) or a changed value
    (``replace()``). Every other path validates with load semantics. Anything but a mapping holding the policy marker
    counts as load semantics."""
    context = info.context
    return isinstance(context, Mapping) and context.get(CREATE_POLICY) is True


_NOT_NFC = "must already be in Unicode normalization form NFC (it is never rewritten)"


def _label(value: str, info: ValidationInfo) -> str:
    if value != value.strip():
        raise ValueError("must not have leading or trailing whitespace")
    if not value.isascii() and not unicodedata.is_normalized("NFC", value):
        raise ValueError(_NOT_NFC)
    check_text(value, allow_layout=False, creating=creation_policy(info), labels=True)
    if not _has_visible_character(value):
        raise ValueError("must contain a letter, number, punctuation or symbol")
    return value


def _message(value: str, info: ValidationInfo) -> str:
    check_text(value, allow_layout=True, creating=creation_policy(info))
    if not _has_visible_character(value):
        raise ValueError("must contain a visible character")
    return value


MAX_JSON_BYTES = 64 * 1024


MAX_NESTING = 64
#: Private validation-context key that marks the one place a hash is created: ``create()`` and ``replace()``. Every
#: other validation path (constructor, ``model_validate*``, ``load``, ``model_copy``) requires an existing, correct
#: hash. A module-level object so it cannot be guessed; importing it is possible, so it is not a security boundary.
SEAL = object()
#: Validation-context key that selects the CREATION text policy. It is separate from ``SEAL`` on purpose: it only makes
#: text rules stricter and can never create a hash, so helpers that merely want strict text (the legacy projection) pass
#: ``CREATION_CONTEXT`` and cannot seal anything.
CREATE_POLICY = object()
CREATION_CONTEXT = MappingProxyType({CREATE_POLICY: True})
#: Upper bound, in UTF-8 bytes, on the JSON ``load()`` will parse. Generous for any real record (each JSON object field
#: is capped at 64 KiB); it exists so a hostile document is refused before it is parsed.
MAX_RECORD_BYTES = 4 * 1024 * 1024
#: Deepest container nesting ``load()`` accepts: the longest legal JSON content (64 levels) plus the record around it.
MAX_LOAD_DEPTH = 72
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
    except OverflowError:                              # the instant is outside the representable UTC range
        u = None
    if u is None:
        raise ValueError("datetime is outside the representable UTC range")
    def field(name: str) -> int:                                 # read the C-level value, never a subclass override
        return getattr(datetime, name).__get__(u)

    return datetime(field("year"), field("month"), field("day"), field("hour"), field("minute"), field("second"),
                    field("microsecond"), tzinfo=UTC)


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
        if not isinstance(other, Mapping):
            return NotImplemented
        try:
            return thaw(self) == thaw(other)
        except ValueError:                                      # a foreign mapping nested deeper than the limit
            return False

    def __hash__(self) -> int:
        return hash(tuple(self._data.items()))

    def __repr__(self) -> str:
        return f"FrozenMap(<{len(self._data)} entries>)"            # contents may be PHI: never printed by default

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
        items = list(value.items())                                    # one read: a stateful mapping is snapshotted
        if not all(isinstance(k, str) for k, _ in items):
            raise ValueError("mapping keys must be strings")          # a static message: the key is never echoed
        return {k: thaw(v, _depth + 1) for k, v in items}
    if isinstance(value, list | tuple):
        return [thaw(v, _depth + 1) for v in value]
    return value


def _json_text_ok(value: dict[str, Any], info: ValidationInfo) -> dict[str, Any]:
    """Every key and string value, at any depth, is valid UTF-8 and passes the text rules (keys strictly; values
    may hold newline and tab). The creation policy applies only when the value is new or changed."""
    creating = creation_policy(info)
    stack: list[Any] = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            check_text(_utf8(item), allow_layout=True, creating=creating)
        elif isinstance(item, Mapping):
            for key, child in item.items():
                _utf8(key)
                if not key.isascii() and not unicodedata.is_normalized("NFC", key):
                    raise ValueError(_NOT_NFC)
                check_text(key, allow_layout=False, creating=creating, labels=True)
                stack.append(child)
        elif isinstance(item, list | tuple):
            stack.extend(item)
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
Label = Annotated[str, StringConstraints(min_length=1, max_length=200), AfterValidator(_label)]
ColumnName = Annotated[str, StringConstraints(min_length=1, max_length=128), AfterValidator(_label)]
Message = Annotated[str, StringConstraints(min_length=1, max_length=2000), AfterValidator(_message)]
def numbers_only(value: Any) -> Any:
    """int or float, not bool and not a numeric string: stored bytes stay canonical (one spelling per value)."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("a number is required (not a boolean or a string)")
    return value


NonNegInt = Annotated[int, Field(ge=0, strict=True)]            # strict: not "3", 3.0 or True
FiniteFloat = Annotated[float, BeforeValidator(numbers_only), Field(allow_inf_nan=False)]
JsonObject = Annotated[dict[str, JsonValue], BeforeValidator(thaw), AfterValidator(_json_text_ok),
                       AfterValidator(_json_size_ok), AfterValidator(freeze), WrapSerializer(_thaw_for_serialization)]
IdTuple = Annotated[tuple[Identifier, ...], Field(max_length=1000)]


class Frozen(BaseModel):
    """Base of every nested value model. ``repr`` and ``str`` are redacted: fields can hold free text and identifiers
    that are PHI in a real study. Use ``model_dump()`` deliberately when the content is needed."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(<redacted>)"

    __str__ = __repr__

    def __repr_args__(self):
        return iter(())                                  # the source of repr, Rich and pretty-printing: no values

    def __pretty__(self, fmt, **kwargs):
        yield repr(self)

    def __rich_repr__(self):
        return iter(())

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


class _DuplicateKey(Exception):
    pass


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [key for key, _ in pairs]
    if len(set(keys)) != len(keys):
        raise _DuplicateKey
    return dict(pairs)


def _too_nested(value: Any) -> bool:
    """True if mappings, lists and tuples inside ``value`` nest deeper than ``MAX_LOAD_DEPTH`` (a cycle counts as
    infinitely deep). Iterative, and it stops at the first violation."""
    stack = [(value, 1)]
    while stack:
        item, depth = stack.pop()
        if isinstance(item, Mapping):
            children = list(item.values())
        elif isinstance(item, list | tuple):
            children = item
        else:
            continue
        if depth > MAX_LOAD_DEPTH:
            return True
        stack.extend((child, depth + 1) for child in children)
    return False


def _json_problem(raw: bytes) -> str | None:
    """``"duplicate"`` if any JSON object repeats a key, ``"deep"`` if the document nests too deeply (or is too deep
    for the parser), else ``None``. Malformed JSON is left for pydantic to report (it gives a position, never the
    content)."""
    try:
        parsed = json.loads(raw, object_pairs_hook=_no_duplicate_keys)
    except _DuplicateKey:
        return "duplicate"
    except RecursionError:
        return "deep"
    except ValueError:
        return None
    return "deep" if _too_nested(parsed) else None


@functools.lru_cache(maxsize=256)
def _field_model(cls: type, name: str) -> type[BaseModel]:
    """A one-field model built from ``cls``'s field (its annotation and metadata), so ``replace()`` can validate a
    changed value alone under the creation policy. It hides the offending input in errors and keeps the static field
    name in ``loc``. Bounded cache: at most 256 (class, field) pairs are kept alive."""
    info = cls.model_fields[name]
    return create_model("ReplaceField", __config__=ConfigDict(hide_input_in_errors=True, extra="forbid"),
                        **{name: (info.annotation, info)})


def _refuse_self_nesting(record: WorkflowContract) -> None:
    """A contract may not contain a record of its own class: ``create()`` seals a missing hash only when the record's
    class is the one being created, so a nested instance of that class (direct or through another record) would be
    sealed implicitly. Only the top-level record is ever sealed. The walk covers pydantic models of any kind, mappings,
    lists, tuples, sets and frozensets, and is cycle-safe."""
    cls = type(record)
    stack: list[Any] = [record]
    seen = {id(record)}
    while stack:
        item = stack.pop()
        if isinstance(item, BaseModel):
            if item is not record and type(item) is cls:
                raise ContractError("a contract cannot nest a record of its own class")
            children = [getattr(item, name, None) for name in type(item).model_fields]
        elif isinstance(item, Mapping):
            children = list(item.values())
        else:                                  # only containers are pushed: list, tuple, set or frozenset
            children = list(item)
        for child in children:
            if isinstance(child, BaseModel | Mapping | list | tuple | set | frozenset) and id(child) not in seen:
                seen.add(id(child))
                stack.append(child)


_RECORD_TYPE_FORM = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
_RECORD_TYPES: dict[str, tuple[str, str]] = {}      # constant -> (module, qualname) that claimed it


class WorkflowContract(Frozen):
    """Common header of every persisted record. Each concrete record class declares an explicit, unique, stable
    ``RECORD_TYPE`` string that is part of its hash; renaming a Python class therefore never changes stored hashes."""

    RECORD_TYPE: ClassVar[str]

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        constant = vars(cls).get("RECORD_TYPE")
        if not isinstance(constant, str) or not _RECORD_TYPE_FORM.fullmatch(constant):
            raise TypeError(f"{cls.__name__} must declare its own RECORD_TYPE: a plain ASCII identifier string")
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
            raise ValueError(f"unsupported schema_version (got a {type(v).__name__}); "
                             f"supported: {sorted(SUPPORTED_SCHEMA_VERSIONS)}")
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
            if sealing_class(info) is not type(self):            # nested records and every other path must bring a hash
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
        """Build a NEW record and seal it, under the creation text policy. This is the one named creation API: the
        constructor, ``model_validate*``, ``load`` and ``model_copy`` all require an existing, correct ``content_hash``
        and never seal on their own. A record that nests another record of its own class is refused."""
        if "content_hash" in fields:
            raise TypeError("create() seals the record itself; do not pass content_hash")
        record = cls.model_validate({**fields, "content_hash": ""}, context={SEAL: cls, CREATE_POLICY: True})
        _refuse_self_nesting(record)
        return record

    def replace(self, **changes: Any):
        """A NEW sealed record equal to this one except for ``changes`` (the named replacement API). The original is
        untouched. The source record must still match its own hash (it is read through ``validated_copy``), so a
        tampered record cannot be turned into a valid-looking one. Unknown fields are refused with a fixed code (the
        names are never echoed). Each CHANGED value is validated once, on its own, under the creation policy, and the
        validated value (never the caller's object) is what goes into the new record, so iterators and stateful
        mappings are read exactly once. Unchanged values are re-validated under load semantics: stored text that was
        acceptable when it was written, including a stored reviewer label, never blocks an unrelated change."""
        cls = type(self)
        fields = cls.model_fields
        unknown = set(changes) - set(fields)
        if unknown:
            raise ContractError("unknown fields in replace()", detail={"code": "unknown_field", "count": len(unknown)})
        if "content_hash" in changes:
            raise TypeError("replace() seals the record itself; do not pass content_hash")
        source = self.validated_copy()
        data = {name: getattr(source, name) for name in fields if name != "content_hash"}
        for name, value in changes.items():
            data[name] = getattr(_field_model(cls, name).model_validate({name: value}, context=CREATION_CONTEXT), name)
        record = cls.model_validate({**data, "content_hash": ""}, context={SEAL: cls})
        _refuse_self_nesting(record)
        return record

    @classmethod
    def load(cls, data):
        """Read a STORED record: JSON as an exact ``str``, ``bytes`` or ``bytearray`` (a subclass is refused: its
        ``len`` and ``decode`` could lie), or a mapping. The hash must be present and correct; there is no sealing here,
        so tampering plus stripping the hash cannot launder a record. ``model_validate`` and ``model_validate_json``
        behave the same way: they are all verification paths. JSON is snapshotted once into immutable bytes, and
        refused before validation if it is larger than ``MAX_RECORD_BYTES`` UTF-8 bytes, repeats an object key, or
        nests deeper than ``MAX_LOAD_DEPTH`` (a mapping is checked for depth the same way). Every excessive-nesting case
        is one ``ContractError``."""
        if type(data) in (str, bytes, bytearray):
            if type(data) is str and len(data) > MAX_RECORD_BYTES:
                raise ContractError("stored record is too large to read")
            raw = data.encode("utf-8", "surrogatepass") if type(data) is str else bytes(data)
            if len(raw) > MAX_RECORD_BYTES:
                raise ContractError("stored record is too large to read")
            problem = _json_problem(raw)
            if problem == "duplicate":
                raise ContractError("stored record JSON has duplicate keys")
            if problem == "deep":
                raise ContractError("stored record is nested too deeply")
            return cls.model_validate_json(raw)
        if isinstance(data, str | bytes | bytearray):
            raise ContractError("stored record JSON must be an exact str, bytes or bytearray")
        if _too_nested(data):
            raise ContractError("stored record is nested too deeply")
        return cls.model_validate(data)

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False):
        """An exact, re-validated copy. ``update`` cannot be used to change content: a changed record keeps a stale
        hash and is refused. Use ``replace()`` to make a changed record. Unknown field names are refused."""
        fields = type(self).model_fields
        unknown = set(update or {}) - set(fields)
        if unknown:
            raise ContractError("unknown fields in model_copy() update",
                                detail={"code": "unknown_field", "count": len(unknown)})
        data = {name: getattr(self, name) for name in fields}
        data.update(update or {})
        return type(self).model_validate(copy.deepcopy(data) if deep else data)

    def validated_copy(self):
        """A freshly validated copy (content hash and every structural rule re-checked), or ``ContractError``.
        Code that makes decisions must read this copy, never the live object. The copy is rebuilt from the record's JSON
        form, so every field comes back as an exact built-in type (a ``str``, ``int`` or ``datetime`` subclass with a
        lying comparison, smuggled in through ``model_construct``, cannot reach a decision). A record with no stored
        content hash is refused, and the advisory lints (see ``ReviewRecord``) are not applied because nothing is
        being created. The ``ContractError`` keeps no raw exception; its ``detail`` is the sanitized error list."""
        try:
            return type(self).model_validate_json(self.model_dump_json(warnings="error"))
        except Exception as exc:
            problem = safe_errors(exc)
        raise ContractError(f"integrity check failed: {type(self).__name__} failed re-validation", detail=problem)

    def __repr__(self) -> str:
        record_type = getattr(type(self), "RECORD_TYPE", type(self).__name__)
        return f"{record_type}(hash={self.content_hash[:12] or 'unsealed'})"

    __str__ = __repr__

    def compute_hash(self) -> str:
        record_type = getattr(type(self), "RECORD_TYPE", None)
        if not isinstance(record_type, str):
            raise TypeError(f"{type(self).__name__} declares no RECORD_TYPE")
        payload = {"domain": HASH_DOMAIN, "record_type": record_type, "schema_version": self.schema_version,
                   "fields": self._dump(exclude={"content_hash"})}
        return sha256_hex(canonical_json(payload))

    def verify_integrity(self) -> bool:
        """True if the live object still matches its content hash. It checks the HASH ONLY: a forced attribute rewrite
        (``object.__setattr__``) is caught, but a ``model_construct`` record whose hash was recomputed passes. Use
        ``validated_copy()`` for the structural check."""
        try:
            return self.content_hash == self.compute_hash()
        except Exception:                 # content that cannot even be serialized is not intact
            return False

    def canonical_bytes(self) -> bytes:
        return canonical_json(self._dump())

    def _dump(self, **options: Any) -> dict[str, Any]:
        """The record as plain JSON-like data. Content that does not serialize cleanly (a tampered field of the wrong
        type) is an error rather than a warning, so nothing about it is printed or logged."""
        try:
            return self.model_dump(mode="json", warnings="error", **options)
        except Exception:
            pass                                                     # raised below, outside the handler: no chain
        raise ContractError("record content does not serialize cleanly")


def require_unique(values: tuple[Any, ...], what: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{what} must be unique")
