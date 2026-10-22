"""Isolated, unregistered read-only tool adapter for Scientific Playbooks.

Touches no guarded files (`builtins.py`, `base.py`, `orchestrator.py`, `pharmstate.py`).
Every call returns `writes={}`, enforces strict argument and path confinement, caps output
size to `MAX_ADAPTER_OUTPUT_BYTES`, and includes full playbook identity and digest metadata.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from app.core.pharmstate import PharmState
from app.tools.base import Tool, ToolContext, ToolRegistry, ToolResult

from .catalog import (
    list_playbooks,
    load_playbook,
    read_playbook_resource,
    validate_all_playbooks,
)
from .schema import (
    MAX_ADAPTER_OUTPUT_BYTES,
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
_METADATA_RESERVE_BYTES = 4096


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


def _cap_text(text: str, max_bytes: int = MAX_ADAPTER_OUTPUT_BYTES) -> tuple[str, bool]:
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text, False
    truncated = encoded[:max_bytes].decode("utf-8", errors="ignore")
    return truncated, True


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
            return ToolResult(
                summary=f"Listed {len(items)} Scientific Playbook(s).",
                writes={},
                result={"status": "ok", "operation": op, "playbooks": items},
                action="playbook_adapter(list_playbooks)",
            )

        if op == "load_playbook":
            playbook_id = call_args.get("playbook_id")
            if not isinstance(playbook_id, str) or not playbook_id.strip():
                return _error_result("missing_playbook_id", "`playbook_id` is required for load_playbook")
            version = call_args.get("version")
            pb = load_playbook(
                playbook_id,
                version=version,
                skills_root=skills_root,
                registry=registry,
                allow_disabled=False,
            )
            instructions, truncated = _cap_text(
                pb.instructions, MAX_ADAPTER_OUTPUT_BYTES - _METADATA_RESERVE_BYTES
            )
            pb_dict = {
                **_identity_dict(pb),
                "domain": pb.domain,
                "description": pb.description,
                "status": pb.status,
                "produced_artifacts": list(pb.produced_artifacts),
                "requires_human_review": pb.requires_human_review,
                "runtime_available": pb.runtime_available,
                "resource_paths": list(pb.resource_paths),
                "instructions": instructions,
                "truncated": truncated,
                "trust_notice": pb.trust_notice,
                "untrusted_data": True,
            }
            return ToolResult(
                summary=f"Loaded Scientific Playbook {pb.playbook_id} v{pb.version} ({pb.canonical_digest[:12]}).",
                writes={},
                result={"status": "ok", "operation": op, "playbook": pb_dict},
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
                allow_disabled=False,
            )
            content, truncated = _cap_text(
                res.content, MAX_ADAPTER_OUTPUT_BYTES - _METADATA_RESERVE_BYTES
            )
            res_dict = {
                **_identity_dict(res),
                "resource_path": res.resource_path,
                "content": content,
                "sha256": res.sha256,
                "size_bytes": res.size_bytes,
                "truncated": truncated,
                "trust_notice": res.trust_notice,
                "untrusted_data": True,
            }
            return ToolResult(
                summary=f"Read resource {res.resource_path} from {res.playbook_id} v{res.version}.",
                writes={},
                result={"status": "ok", "operation": op, "resource": res_dict},
                action=f"playbook_adapter(read_resource:{res.playbook_id}:{res.resource_path})",
            )

        # op == "validate_catalog"
        report = validate_all_playbooks(skills_root=skills_root, registry=registry)
        issues = [
            i.model_dump()
            for r in report.results
            for i in r.issues
        ] + [i.model_dump() for i in report.catalog_issues]
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
        raw_json = json.dumps(result_payload, sort_keys=True)
        if len(raw_json.encode("utf-8")) > MAX_ADAPTER_OUTPUT_BYTES:
            result_payload["issues"] = issues[:25]
            result_payload["truncated"] = True
        return ToolResult(
            summary=(
                f"Validated {report.total_playbooks} playbook(s): "
                f"valid={report.valid}, issues={len(issues)}."
            ),
            writes={},
            result=result_payload,
            action="playbook_adapter(validate_catalog)",
        )
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
