"""CLI: `python -m app.playbooks [--check] [--format text|json] [--skills-root DIR]`."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .catalog import DEFAULT_SKILLS_ROOT, validate_all_playbooks
from .schema import CatalogValidationReport


def render_check_text(report: CatalogValidationReport) -> str:
    all_issues = [
        (r.playbook_id or r.name or "unknown", i)
        for r in report.results
        for i in r.issues
    ] + [("catalog", i) for i in report.catalog_issues]

    lines = [
        f"Playbooks: {report.total_playbooks}",
        f"Approved: {report.approved_count}",
        f"Pending owner review: {report.pending_approval_count}",
        f"Draft: {report.draft_count}",
        f"Reviewed: {report.reviewed_count}",
        f"Deprecated: {report.deprecated_count}",
        f"Runtime available: {report.runtime_available_count}",
        f"Disabled specifications: {report.disabled_count}",
        f"Validation failures: {len(all_issues)}",
        f"Missing tool references: {len(report.missing_tool_references)}",
        f"Duplicate identities: {len(report.duplicate_identities)}",
        f"Security violations: {len(report.security_violations)}",
    ]
    for r in report.results:
        digest_short = (r.canonical_digest or "invalid")[:12]
        lines.append(
            f"  - {r.playbook_id or r.name}: v{r.version or '?'} "
            f"[{r.status or '?'} / {r.approval_status or '?'}] "
            f"runtime={r.runtime_available} digest={digest_short} valid={r.valid}"
        )
    for owner, issue in all_issues:
        lines.append(
            f"  ! [{issue.severity}] {owner} ({issue.code} @ {issue.field_or_path}): {issue.message}"
        )
    return "\n".join(lines) + "\n"


def render_check_json(report: CatalogValidationReport) -> str:
    all_issues = [
        {"playbook": r.playbook_id or r.name, **i.model_dump()}
        for r in report.results
        for i in r.issues
    ] + [{"playbook": "catalog", **i.model_dump()} for i in report.catalog_issues]

    payload = {
        "valid": report.valid,
        "skills_root": report.skills_root,
        "total_playbooks": report.total_playbooks,
        "approved_count": report.approved_count,
        "pending_approval_count": report.pending_approval_count,
        "draft_count": report.draft_count,
        "reviewed_count": report.reviewed_count,
        "deprecated_count": report.deprecated_count,
        "runtime_available_count": report.runtime_available_count,
        "disabled_count": report.disabled_count,
        "validation_failures_count": len(all_issues),
        "missing_tool_references": list(report.missing_tool_references),
        "duplicate_identities": list(report.duplicate_identities),
        "security_violations_count": len(report.security_violations),
        "issues": all_issues,
        "playbooks": [
            {
                "playbook_id": r.playbook_id,
                "name": r.name,
                "version": r.version,
                "canonical_digest": r.canonical_digest,
                "status": r.status,
                "approval_status": r.approval_status,
                "runtime_available": r.runtime_available,
                "valid": r.valid,
            }
            for r in report.results
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m app.playbooks", description=__doc__)
    ap.add_argument("--check", action="store_true", help="Validate all playbooks and exit 1 on any issue")
    ap.add_argument("--format", choices=["text", "json"], default="text")
    ap.add_argument(
        "--skills-root",
        default=str(DEFAULT_SKILLS_ROOT),
        help="Path to the canonical skills/ directory",
    )
    args = ap.parse_args(argv)
    report = validate_all_playbooks(skills_root=Path(args.skills_root))
    out = render_check_json(report) if args.format == "json" else render_check_text(report)
    sys.stdout.write(out)
    return 0 if report.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
