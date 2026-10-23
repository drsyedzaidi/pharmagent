"""Isolated, unregistered read-only tool adapter for Scientific Playbooks.

Touches no guarded files (`builtins.py`, `base.py`, `orchestrator.py`, `pharmstate.py`).
Every call returns `writes={}`, enforces strict argument and path confinement, caps output
size to `MAX_ADAPTER_OUTPUT_BYTES`, and includes full playbook identity and digest metadata.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from app.core.pharmstate import PharmState
from app.tools.base import Tool, ToolContext, ToolRegistry, ToolResult

from .catalog import (
    list_playbooks,
    load_approved_playbook,
    read_playbook_resource,
    validate_all_playbooks,
)
from .schema import (
    MAX_ADAPTER_OUTPUT_BYTES,
    PlaybookApprovalError,
    PlaybookDisabledError,
    PlaybookError,
    PlaybookLookupError,
    PlaybookResource,
    PlaybookSecurityError,
    PlaybookSummary,
    PlaybookValidationError,
    PlaybookVersionError,
    ScientificPlaybook,
)

_ALLOWED_OPERATIONS = ("list_playbooks", "load_playbook", "read_resource", "validate_catalog")
_ALLOWED_ARG_KEYS = frozenset({"operation", "playbook_id", "version", "resource", "include_disabled"})


def _identity_dict(pb: PlaybookSummary | ScientificPlaybook | PlaybookResource) -> dict[str, Any]:
    return {
        "playbook_id": pb.playbook_id,
        "name": pb.name,
        "version": pb.version,
        "canonical_digest": pb.canonical_digest,
        "approval_status": pb.approval_status,
        "source": pb.source,
        "license": pb.license,
        "allowed_tools": list(pb.allowed_tools),
        "required_inputs": list(pb.required_inputs),
        "limitations": list(pb.limitations),
    }


def _error_result(code: str, message: str, *, action: str = "playbook_adapter(error)", **extra: Any) -> ToolResult:
    payload: dict[str, Any] = {
        "status": "error",
        "error_code": code,
        "message": message,
        **extra,
    }
    return ToolResult(
        summary=f"Scientific Playbook adapter error ({code}): {message}",
        writes={},
        result=payload,
        action=action,
    )


def _json_size(payload: Mapping[str, Any]) -> int:
    """Byte size of `payload` as a consumer serializes it (default `json.dumps`, ASCII-escaped)."""
    return len(json.dumps(payload).encode("utf-8"))


def _fit_text(build: Callable[[str, bool], dict[str, Any]], text: str) -> dict[str, Any] | None:
    """Return build(text, False) if it fits the cap; else the longest prefix that fits, marked truncated.

    Sizes are measured after JSON serialization, so escaping of quotes, backslashes,
    newlines and non-ASCII characters can never push the result over the cap.
    """
    full = build(text, False)
    if _json_size(full) <= MAX_ADAPTER_OUTPUT_BYTES:
        return full
    lo, hi, best = 0, len(text) - 1, None
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = build(text[:mid], True)
        if _json_size(candidate) <= MAX_ADAPTER_OUTPUT_BYTES:
            best, lo = candidate, mid + 1
        else:
            hi = mid - 1
    return best


def _fit_items(build: Callable[[list[Any], bool], dict[str, Any]], items: Sequence[Any]) -> dict[str, Any] | None:
    """Like `_fit_text`, but keeps the longest leading run of list items that fits the cap."""
    full = build(list(items), False)
    if _json_size(full) <= MAX_ADAPTER_OUTPUT_BYTES:
        return full
    lo, hi, best = 0, len(items) - 1, None
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = build(list(items[:mid]), True)
        if _json_size(candidate) <= MAX_ADAPTER_OUTPUT_BYTES:
            best, lo = candidate, mid + 1
        else:
            hi = mid - 1
    return best


def _too_large(op: str) -> ToolResult:
    return _error_result(
        "output_too_large",
        f"{op} metadata alone exceeds the {MAX_ADAPTER_OUTPUT_BYTES}-byte adapter output cap",
    )


def run_playbook_adapter(
    state: PharmState,
    ctx: ToolContext,
    args: Mapping[str, Any] | None = None,
    *,
    skills_root: Path | str | None = None,
    registry: ToolRegistry | None = None,
) -> ToolResult:
    """Execute a read-only Scientific Playbook operation in complete isolation (`writes={}`)."""
    _ = (state, ctx)  # explicitly read-only; never mutates PharmState or ToolContext
    if args is not None and not isinstance(args, Mapping):
        return _error_result("invalid_arguments", "`args` must be a mapping")
    call_args = dict(args or {})

    illegal_keys = sorted(set(call_args) - _ALLOWED_ARG_KEYS)
    if illegal_keys:
        return _error_result(
            "forbidden_argument",
            f"Unsupported or unsafe adapter argument(s): {illegal_keys}; allowed: {sorted(_ALLOWED_ARG_KEYS)}",
        )

    op = call_args.get("operation")
    if op not in _ALLOWED_OPERATIONS:
        return _error_result(
            "invalid_operation",
            f"Invalid operation {op!r}; allowed: {list(_ALLOWED_OPERATIONS)}",
        )

    try:
        if op == "list_playbooks":
            raw_inc = call_args.get("include_disabled", True)
            if not isinstance(raw_inc, bool):
                return _error_result("invalid_arguments", "`include_disabled` must be a boolean")
            include_disabled = raw_inc
            summaries = list_playbooks(
                skills_root=skills_root,
                registry=registry,
                include_disabled=include_disabled,
            )
            items = [
                {
                    **_identity_dict(s),
                    "domain": s.domain,
                    "description": s.description,
                    "status": s.status,
                    "produced_artifacts": list(s.produced_artifacts),
                    "requires_human_review": s.requires_human_review,
                    "runtime_available": s.runtime_available,
                    "resource_paths": list(s.resource_paths),
                }
                for s in summaries
            ]
            listed = _fit_items(
                lambda kept, cut: {"status": "ok", "operation": op, "playbooks": kept, "truncated": cut}, items
            )
            if listed is None:
                return _too_large(op)
            return ToolResult(
                summary=f"Listed {len(listed['playbooks'])} of {len(items)} Scientific Playbook(s).",
                writes={},
                result=listed,
                action="playbook_adapter(list_playbooks)",
            )

        if op == "load_playbook":
            playbook_id = call_args.get("playbook_id")
            if not isinstance(playbook_id, str) or not playbook_id.strip():
                return _error_result("missing_playbook_id", "`playbook_id` is required for load_playbook")
            version = call_args.get("version")
            pb = load_approved_playbook(
                playbook_id,
                version=version,
                skills_root=skills_root,
                registry=registry,
            )
            pb_meta = {
                **_identity_dict(pb),
                "domain": pb.domain,
                "description": pb.description,
                "status": pb.status,
                "produced_artifacts": list(pb.produced_artifacts),
                "requires_human_review": pb.requires_human_review,
                "runtime_available": pb.runtime_available,
                "resource_paths": list(pb.resource_paths),
                "trust_notice": pb.trust_notice,
                "untrusted_data": True,
            }
            loaded = _fit_text(
                lambda text, cut: {
                    "status": "ok",
                    "operation": op,
                    "playbook": {**pb_meta, "instructions": text, "truncated": cut},
                },
                pb.instructions,
            )
            if loaded is None:
                return _too_large(op)
            return ToolResult(
                summary=f"Loaded Scientific Playbook {pb.playbook_id} v{pb.version} ({pb.canonical_digest[:12]}).",
                writes={},
                result=loaded,
                action=f"playbook_adapter(load_playbook:{pb.playbook_id}@{pb.version})",
            )

        if op == "read_resource":
            playbook_id = call_args.get("playbook_id")
            resource_rel = call_args.get("resource")
            version = call_args.get("version")
            if not isinstance(playbook_id, str) or not playbook_id.strip():
                return _error_result("missing_playbook_id", "`playbook_id` is required for read_resource")
            if not isinstance(resource_rel, str) or not resource_rel.strip():
                return _error_result("missing_resource", "`resource` is required for read_resource")
            res = read_playbook_resource(
                playbook_id,
                resource_rel,
                version=version,
                skills_root=skills_root,
                registry=registry,
            )
            res_meta = {
                **_identity_dict(res),
                "resource_path": res.resource_path,
                "sha256": res.sha256,
                "size_bytes": res.size_bytes,
                "trust_notice": res.trust_notice,
                "untrusted_data": True,
            }
            read = _fit_text(
                lambda text, cut: {
                    "status": "ok",
                    "operation": op,
                    "resource": {**res_meta, "content": text, "truncated": cut},
                },
                res.content,
            )
            if read is None:
                return _too_large(op)
            return ToolResult(
                summary=f"Read resource {res.resource_path} from {res.playbook_id} v{res.version}.",
                writes={},
                result=read,
                action=f"playbook_adapter(read_resource:{res.playbook_id}:{res.resource_path})",
            )

        # op == "validate_catalog"
        report = validate_all_playbooks(skills_root=skills_root, registry=registry)
        issues = [i.model_dump() for r in report.results for i in r.issues] + [
            i.model_dump() for i in report.catalog_issues
        ]
        playbooks_meta = [
            {
                "playbook_id": r.playbook_id,
                "name": r.name,
                "version": r.version,
                "canonical_digest": r.canonical_digest,
                "status": r.status,
                "approval_status": r.approval_status,
                "valid": r.valid,
                "playbook": _identity_dict(r.playbook) if r.playbook is not None else None,
            }
            for r in report.results
        ]
        result_payload = {
            "status": "ok" if report.valid else "invalid",
            "operation": op,
            "valid": report.valid,
            "total_playbooks": report.total_playbooks,
            "approved_count": report.approved_count,
            "pending_approval_count": report.pending_approval_count,
            "draft_count": report.draft_count,
            "reviewed_count": report.reviewed_count,
            "deprecated_count": report.deprecated_count,
            "missing_tool_references": list(report.missing_tool_references),
            "duplicate_identities": list(report.duplicate_identities),
            "security_violations_count": len(report.security_violations),
            "issues": issues,
            "playbooks": playbooks_meta,
        }
        validated = _fit_items(lambda kept, cut: {**result_payload, "issues": kept, "truncated": cut}, issues)
        if validated is None:
            validated = _fit_items(
                lambda kept, cut: {**result_payload, "issues": [], "playbooks": kept, "truncated": True},
                playbooks_meta,
            )
        if validated is None:
            return _too_large(op)
        return ToolResult(
            summary=(f"Validated {report.total_playbooks} playbook(s): valid={report.valid}, issues={len(issues)}."),
            writes={},
            result=validated,
            action="playbook_adapter(validate_catalog)",
        )
    except PlaybookApprovalError as exc:
        return _error_result("playbook_not_approved", str(exc))
    except PlaybookDisabledError as exc:
        return _error_result("playbook_disabled", str(exc))
    except PlaybookSecurityError as exc:
        return _error_result("security_violation", str(exc))
    except PlaybookVersionError as exc:
        return _error_result("version_mismatch", str(exc))
    except PlaybookLookupError as exc:
        return _error_result("not_found", str(exc))
    except PlaybookValidationError as exc:
        return _error_result("validation_error", str(exc))
    except PlaybookError as exc:
        return _error_result("playbook_error", str(exc))


def build_playbook_tools(
    *,
    skills_root: Path | str | None = None,
    registry: ToolRegistry | None = None,
) -> tuple[Tool, ...]:
    """Construct isolated Tool definitions for Scientific Playbooks (unregistered in MVP)."""

    def _runner(state: PharmState, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        return run_playbook_adapter(state, ctx, args, skills_root=skills_root, registry=registry)

    tool = Tool(
        name="inspect_scientific_playbook",
        description=(
            "Read-only progressive-disclosure access to authored Scientific Playbooks: "
            "list playbooks, load a validated playbook's instructions, read a confined "
            "playbook resource, or validate the playbook catalog. Never executes scripts "
            "or mutates analysis state."
        ),
        agent="supervisor",
        input_schema={
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": list(_ALLOWED_OPERATIONS),
                },
                "playbook_id": {"type": "string"},
                "version": {"type": "string"},
                "resource": {"type": "string"},
                "include_disabled": {"type": "boolean"},
            },
            "required": ["operation"],
            "additionalProperties": False,
        },
        run=_runner,
        expensive=False,
        proposable=False,
    )
    return (tool,)
