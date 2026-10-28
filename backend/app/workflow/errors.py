"""Errors raised by the workflow-contract layer, and the one safe way to turn a validation error into something a
caller may log or return.

``ContractError`` messages are fixed text. Anything that identifies a record (ids, field names) is kept on
``.detail`` for an authorised debugger and is never part of ``str(error)``.

``safe_errors`` is the PHI-safe boundary for pydantic ``ValidationError``: it returns only the error type, a location
scrubbed to static schema names and list indices, and a length-capped message. It never includes the input value, the
documentation URL, or the error context, and it replaces any location element that is not a known schema name (for
example a key the caller chose) with ``"<key>"``. Pydantic's own ``errors()``, ``json()`` and ``str()`` carry the
offending input or the caller's keys in ``loc``; do not log or return those directly.
"""
from __future__ import annotations

from typing import Any

#: location tags pydantic adds around dict keys and unions that carry no caller data
_STATIC_TAGS = frozenset({"[key]", "str", "int", "float", "bool", "list", "dict", "tuple", "none", "literal",
                          "json-or-python", "dict[str,json-or-python]", "list[json-or-python]"})
_MAX_MESSAGE = 200


class ContractError(ValueError):
    """A workflow record is malformed, inconsistent, or cannot be read. ``str()`` is safe to log; ``detail`` is not."""

    def __init__(self, message: str, *, detail: Any = None):
        super().__init__(message)
        self.detail = detail


def _static_names() -> frozenset[str]:
    from .base import Frozen  # local: base imports this module
    names: set[str] = set(_STATIC_TAGS)
    stack = [Frozen]
    while stack:
        cls = stack.pop()
        names.add(cls.__name__)
        names.update(cls.model_fields)
        stack.extend(cls.__subclasses__())
    return frozenset(names)


def safe_errors(exc: Any) -> list[dict[str, Any]]:
    """The PHI-safe form of a pydantic ``ValidationError``: a list of ``{"type", "loc", "msg"}``."""
    names = _static_names()
    cleaned = []
    for err in exc.errors(include_input=False, include_url=False, include_context=False):
        loc = tuple(part if isinstance(part, int) else (part if part in names else "<key>") for part in err["loc"])
        cleaned.append({"type": err["type"], "loc": loc, "msg": str(err["msg"])[:_MAX_MESSAGE]})
    return cleaned
