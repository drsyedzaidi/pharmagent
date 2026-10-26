"""Validate tool-call arguments against the tool's declared JSON ``input_schema``.

One function, two callers: ``policy.decide`` (a planner proposal becomes a rejected
decision) and ``ToolRegistry.execute`` (defence in depth, raises ``ValueError``).
The ``x-normalized`` schema flag marks an enum the tool itself normalizes (case,
spelling), so the enum is advertised but not enforced. Checked: unknown top-level keys,
JSON types (``bool`` is not an integer or number), ``enum``, ``minimum`` / ``maximum``
(+ exclusive variants), array ``items``, nested
object property types, and ``ci_lower < ci_upper`` wherever both appear. ``required``
is not enforced here: tools report a missing input in their own words. ``None`` is
treated as "not supplied", as the tools do.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

_MAX_DEPTH = 6

# Accepted by ToolRegistry.execute but deliberately NOT advertised in the schema (a
# strict opt-in the chat LLM must not see). A planner proposal naming one is rejected.
UNADVERTISED_ARGS: dict[str, frozenset[str]] = {
    "adversarial_review": frozenset({"consult_mikebrain"}),
}


def _type_ok(kind: str, value: Any) -> bool:
    if kind == "string":
        return isinstance(value, str)
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "number":
        return (isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value))
    if kind == "array":
        return isinstance(value, (list, tuple))
    if kind == "object":
        return isinstance(value, Mapping)
    return True


def _check(path: str, schema: Mapping[str, Any], value: Any, depth: int, errors: list[str]) -> None:
    if value is None:
        return
    kind = schema.get("type")
    if isinstance(kind, str) and not _type_ok(kind, value):
        errors.append(f"{path}: expected {kind}, got {type(value).__name__} ({value!r:.40})")
        return
    if "enum" in schema and not schema.get("x-normalized") and value not in schema["enum"]:
        errors.append(f"{path}: {value!r:.40} is not one of {list(schema['enum'])}")
        return
    if kind in ("integer", "number"):
        for key, ok, sym in (("minimum", lambda v, b: v >= b, ">="),
                             ("maximum", lambda v, b: v <= b, "<="),
                             ("exclusiveMinimum", lambda v, b: v > b, ">"),
                             ("exclusiveMaximum", lambda v, b: v < b, "<")):
            if key in schema and not ok(value, schema[key]):
                errors.append(f"{path}: {value!r} must be {sym} {schema[key]}")
    elif kind == "array" and depth < _MAX_DEPTH and isinstance(schema.get("items"), Mapping):
        for i, item in enumerate(value):
            _check(f"{path}[{i}]", schema["items"], item, depth + 1, errors)
    elif kind == "object" and depth < _MAX_DEPTH:
        _check_object(path, schema, value, depth, errors, strict=False)


def _check_object(path: str, schema: Mapping[str, Any], value: Mapping[str, Any], depth: int,
                  errors: list[str], *, strict: bool, allow_extra: frozenset[str] = frozenset()) -> None:
    props = schema.get("properties")
    props = props if isinstance(props, Mapping) else {}
    if strict:
        for key in value:
            if key not in props and key not in allow_extra:
                errors.append(f"{path + '.' if path else ''}{key}: unknown argument "
                              f"(accepted: {', '.join(sorted(props)) or 'none'})")
    for key, sub in props.items():
        if key in value and isinstance(sub, Mapping):
            _check(f"{path + '.' if path else ''}{key}", sub, value[key], depth + 1, errors)
    lo, hi = value.get("ci_lower"), value.get("ci_upper")
    if all(_type_ok("number", v) for v in (lo, hi)) and not lo < hi:
        errors.append(f"{path + '.' if path else ''}ci_lower ({lo}) must be below ci_upper ({hi})")


def validate_args(schema: Mapping[str, Any], args: Any, *,
                  allow_extra: frozenset[str] = frozenset()) -> str | None:
    """A refusal reason, or None when ``args`` fits ``schema``."""
    if args is None:
        return None
    if not isinstance(args, Mapping):
        return f"arguments must be an object, got {type(args).__name__}"
    errors: list[str] = []
    _check_object("", schema, args, 0, errors, strict=True, allow_extra=allow_extra)
    return "; ".join(errors) if errors else None
