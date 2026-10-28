"""Errors raised by the workflow-contract layer, and the one safe way to turn a validation error into something a
caller may log or return.

``ContractError`` messages are fixed text, and a ``ContractError`` never keeps the exception that caused it: it is
raised after the ``except`` block has finished, so neither ``__cause__`` nor ``__context__`` holds a raw pydantic error
(whose text can contain the offending input or a caller's mapping key). When a validation failure is the reason,
``.detail`` holds the already-sanitized ``safe_errors`` list; otherwise it holds ids for an authorised debugger.

``safe_errors`` is the PHI-safe boundary for pydantic ``ValidationError``: it returns only the error type, a location
scrubbed to static schema names and genuine list indices, and a length-capped message. It never includes the input
value, the documentation URL, or the error context. Any location element that is not a static schema name (for example
a key the caller chose) becomes ``"<key>"``; the element right after a mapping field is always a caller's key and is
redacted even when it equals a schema name; and an integer survives only directly after a field that holds a sequence
(one level), so an integer key anywhere else is redacted. Pydantic's own ``errors()``, ``json()`` and ``str()`` carry
the offending input or the caller's keys in ``loc``; do not log or return those directly. Anything that is not a
``ValidationError`` or ``ContractError`` yields a fixed "unsupported" entry.
"""
from __future__ import annotations

import types
from collections.abc import Mapping
from typing import Annotated, Any, Union, get_args, get_origin

from pydantic import ValidationError

#: location tags pydantic adds around dict keys and unions that carry no caller data
_STATIC_TAGS = frozenset({"[key]", "str", "int", "float", "bool", "list", "dict", "tuple", "none", "literal",
                          "json-or-python", "dict[str,json-or-python]", "list[json-or-python]"})
_MAX_MESSAGE = 200
#: pydantic error types whose message repeats the offending input (a discriminated-union tag)
_ECHOING_TYPES = frozenset({"union_tag_invalid"})
_ECHOING_MESSAGE = "Input tag is not valid"
_UNSUPPORTED = {"type": "unsupported_exception", "loc": (),
                "msg": "error details are not available for this kind of exception"}


class ContractError(ValueError):
    """A workflow record is malformed, inconsistent, or cannot be read. ``str()`` is safe to log."""

    def __init__(self, message: str, *, detail: Any = None):
        super().__init__(message)
        self.detail = detail


def _kinds(annotation: Any) -> set[str]:
    """Whether a field annotation is a sequence, a mapping, or something else (union arms are all considered)."""
    kinds: set[str] = set()
    stack = [annotation]
    while stack:
        item = stack.pop()
        origin = get_origin(item)
        if origin is Annotated:
            stack.append(get_args(item)[0])
        elif origin is Union or origin is types.UnionType:
            stack.extend(get_args(item))
        elif origin in (tuple, list):
            kinds.add("sequence")
        elif origin is dict or origin is Mapping:
            kinds.add("mapping")
        else:
            kinds.add("other")
    return kinds


def _schema() -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    """(every static schema name, the field names that hold a sequence and never a mapping, the field names that may
    hold a mapping)."""
    from .base import Frozen  # local: base imports this module
    names: set[str] = set(_STATIC_TAGS)
    sequences: set[str] = set()
    mappings: set[str] = set()
    stack = [Frozen]
    while stack:
        cls = stack.pop()
        names.add(cls.__name__)
        for name, field in cls.model_fields.items():
            names.add(name)
            kinds = _kinds(field.annotation)
            if "sequence" in kinds:
                sequences.add(name)
            if "mapping" in kinds:
                mappings.add(name)
        stack.extend(cls.__subclasses__())
    return frozenset(names), frozenset(sequences - mappings), frozenset(mappings)


def _clean_loc(loc: tuple[Any, ...], names: frozenset[str], sequences: frozenset[str],
               mappings: frozenset[str]) -> tuple[Any, ...]:
    cleaned: list[Any] = []
    expects = ""                                              # "index" after a sequence field, "key" after a mapping
    for part in loc:
        if expects == "key":
            cleaned.append("<key>")                           # whatever follows a mapping field is a caller's key
            expects = ""
        elif expects == "index" and isinstance(part, int) and not isinstance(part, bool):
            cleaned.append(part)                              # a genuine list index (only one level is trusted)
            expects = ""
        elif isinstance(part, str) and part in names:
            cleaned.append(part)
            expects = "index" if part in sequences else "key" if part in mappings else ""
        else:
            cleaned.append("<key>")
            expects = ""
    return tuple(cleaned)


def safe_errors(exc: Any) -> list[dict[str, Any]]:
    """The PHI-safe form of a pydantic ``ValidationError``: a list of ``{"type", "loc", "msg"}``."""
    if isinstance(exc, ContractError):
        return [{"type": "contract_error", "loc": (), "msg": str(exc)[:_MAX_MESSAGE]}]
    if not isinstance(exc, ValidationError):
        return [dict(_UNSUPPORTED)]
    names, sequences, mappings = _schema()
    cleaned = []
    for err in exc.errors(include_input=False, include_url=False, include_context=False):
        message = _ECHOING_MESSAGE if err["type"] in _ECHOING_TYPES else str(err["msg"])[:_MAX_MESSAGE]
        cleaned.append({"type": err["type"], "loc": _clean_loc(err["loc"], names, sequences, mappings), "msg": message})
    return cleaned
