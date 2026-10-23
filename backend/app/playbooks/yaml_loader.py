"""Strict YAML frontmatter loader with duplicate-key, alias/anchor/merge, custom-tag, and depth guards."""
from __future__ import annotations

import re
from typing import Any

import yaml

from .digest import _normalize_newlines
from .schema import PlaybookValidationIssue

_FRONTMATTER_RE = re.compile(r"\A---[ \t]*\n(?P<yaml>.*?)\n---[ \t]*(?:\n(?P<body>.*))?\Z", re.DOTALL)
_ZERO_WIDTH_RE = re.compile(r"[\u00ad\u200b-\u200f\u202a-\u202e\u2060-\u206f]")
MAX_YAML_DEPTH = 16
_SAFE_YAML_TAGS = frozenset({
    "tag:yaml.org,2002:str",
    "tag:yaml.org,2002:int",
    "tag:yaml.org,2002:float",
    "tag:yaml.org,2002:bool",
    "tag:yaml.org,2002:null",
    "tag:yaml.org,2002:seq",
    "tag:yaml.org,2002:map",
})


class _DuplicateYamlKeyError(ValueError):
    """Raised when YAML frontmatter contains duplicate mapping keys."""


class _UnsafeYamlError(ValueError):
    """Raised when YAML frontmatter uses unsafe tags, constructors, anchors, aliases, or excessive depth."""


class _StrictSafeLoader(yaml.SafeLoader):
    """YAML SafeLoader subclass that forbids duplicate keys, custom tags, anchors, aliases, and implicit dates."""

    # Copy yaml_implicit_resolvers into _StrictSafeLoader's own class dict without mutating yaml.SafeLoader
    yaml_implicit_resolvers = {
        ch: [(tag, regexp) for tag, regexp in resolvers if tag != "tag:yaml.org,2002:timestamp"]
        for ch, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
    }


def _strict_compose_node(self: _StrictSafeLoader, parent: Any, index: Any) -> Any:
    depth = getattr(self, "_yaml_depth", 0) + 1
    if depth > MAX_YAML_DEPTH:
        raise _UnsafeYamlError(f"YAML nesting depth exceeds maximum allowed ({MAX_YAML_DEPTH})")
    self._yaml_depth = depth  # type: ignore[attr-defined]
    try:
        if self.check_event(yaml.AliasEvent):
            raise _UnsafeYamlError("YAML aliases and anchors are forbidden in playbook frontmatter")
        event = self.peek_event()
        if getattr(event, "anchor", None) is not None:
            raise _UnsafeYamlError("YAML anchors are forbidden in playbook frontmatter")
        if getattr(event, "tag", None) not in (None, "!", "?") and event.tag not in _SAFE_YAML_TAGS:
            raise _UnsafeYamlError(f"Unsafe or unsupported YAML tag: {event.tag!r}")
        node = yaml.SafeLoader.compose_node(self, parent, index)
        if node.tag not in _SAFE_YAML_TAGS:
            raise _UnsafeYamlError(f"Unsafe or unsupported YAML tag: {node.tag!r}")
        return node
    finally:
        self._yaml_depth = depth - 1  # type: ignore[attr-defined]


def _strict_construct_mapping(self: _StrictSafeLoader, node: yaml.MappingNode, deep: bool = False) -> dict[str, Any]:
    if not isinstance(node, yaml.MappingNode):
        raise _UnsafeYamlError("Expected a YAML mapping node")
    mapping: dict[str, Any] = {}
    for key_node, value_node in node.value:
        if key_node.tag != "tag:yaml.org,2002:str":
            raise _UnsafeYamlError(f"YAML mapping keys must be strings, got tag {key_node.tag!r}")
        key = self.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise _UnsafeYamlError("YAML mapping keys must be strings")
        if key in mapping:
            raise _DuplicateYamlKeyError(f"Duplicate YAML key {key!r} in frontmatter")
        mapping[key] = self.construct_object(value_node, deep=deep)
    return mapping


_StrictSafeLoader.compose_node = _strict_compose_node  # type: ignore[assignment]
_StrictSafeLoader.add_constructor("tag:yaml.org,2002:map", _strict_construct_mapping)


def _parse_frontmatter(raw_text: str) -> tuple[dict[str, Any] | None, str, list[PlaybookValidationIssue]]:
    """Split and strictly parse YAML frontmatter and Markdown body from `SKILL.md` text."""
    normalized = _normalize_newlines(raw_text)
    m = _FRONTMATTER_RE.match(normalized)
    if not m:
        return None, "", [
            PlaybookValidationIssue(
                code="invalid_frontmatter",
                field_or_path="SKILL.md",
                message="SKILL.md must begin with YAML frontmatter delimited by `---` lines",
                severity="error",
            )
        ]
    yaml_block = m.group("yaml")
    body = (m.group("body") or "").strip()
    try:
        parsed = yaml.load(yaml_block, Loader=_StrictSafeLoader)
    except _DuplicateYamlKeyError as exc:
        return None, body, [
            PlaybookValidationIssue(
                code="duplicate_yaml_key",
                field_or_path="frontmatter",
                message=str(exc),
                severity="security",
            )
        ]
    except (_UnsafeYamlError, RecursionError) as exc:
        msg = str(exc) if isinstance(exc, _UnsafeYamlError) else "YAML nesting depth exceeded recursion limit"
        return None, body, [
            PlaybookValidationIssue(
                code="unsafe_yaml",
                field_or_path="frontmatter",
                message=msg,
                severity="security",
            )
        ]
    except yaml.constructor.ConstructorError as exc:
        return None, body, [
            PlaybookValidationIssue(
                code="unsafe_yaml",
                field_or_path="frontmatter",
                message=f"Unsafe YAML constructor rejected: {exc}",
                severity="security",
            )
        ]
    except yaml.YAMLError as exc:
        return None, body, [
            PlaybookValidationIssue(
                code="invalid_frontmatter",
                field_or_path="frontmatter",
                message=f"Invalid YAML syntax in frontmatter: {exc}",
                severity="error",
            )
        ]

    if not isinstance(parsed, dict):
        return None, body, [
            PlaybookValidationIssue(
                code="invalid_frontmatter",
                field_or_path="frontmatter",
                message="YAML frontmatter must be a key-value mapping",
                severity="error",
            )
        ]
    return parsed, body, []
