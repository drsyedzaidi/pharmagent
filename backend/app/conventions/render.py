"""Generated rule text and the exportable report. Rule text is derived from the catalog, never stored in it."""
from __future__ import annotations

import json

from .coverage import CoverageProblem
from .schema import Catalog, Convention

APPROVAL_MEANING = ("Approved means the owner reviewed an entry as an accurate description of PharmAgent behaviour. "
                    "For an implementation_convention it is not regulatory validation.")


def rule_text(conv: Convention) -> str:
    return conv.statement.format(**conv.parameters)


def render_report_json(catalog: Catalog, problems: list[CoverageProblem]) -> str:
    return json.dumps({
        "catalog_version": catalog.catalog_version,
        "approval_meaning": APPROVAL_MEANING,
        "coverage_problems": [{"convention_id": p.convention_id, "ref": p.ref, "reason": p.reason} for p in problems],
        "conventions": [{**c.model_dump(), "rule_text": rule_text(c)} for c in catalog.conventions],
    }, indent=2, sort_keys=True) + "\n"


def render_report_markdown(catalog: Catalog, problems: list[CoverageProblem]) -> str:
    pending = sum(c.approval.status == "pending_owner_review" for c in catalog.conventions)
    lines = [
        f"# PharmAgent conventions report (catalog {catalog.catalog_version})", "",
        f"Conventions: {len(catalog.conventions)}. Pending owner review: {pending}. "
        f"Coverage problems: {len(problems)}.", "",
        "Rule text below is generated from the catalog data; it is not hand-written.", "",
        APPROVAL_MEANING, "",
        "| Id | Version | Approval | Source | Tests |", "|---|---|---|---|---|",
    ]
    for c in catalog.conventions:
        lines.append(f"| `{c.id}` | {c.version} | {c.approval.status} | {c.source.kind} | {len(c.tests)} |")
    for c in catalog.conventions:
        lines += ["", f"## `{c.id}`: {c.title}", "", rule_text(c), "",
                  f"- Version: {c.version}. Approval: {c.approval.status}"
                  + (f" ({c.approval.by}, {c.approval.date})" if c.approval.by else "") + ".",
                  f"- Source ({c.source.kind}): {c.source.reference}",
                  f"- Applies to: {', '.join(c.applies_to)}",
                  "- Tests:"] + [f"  - `{t}`" for t in c.tests]
        lines.append("- Known gaps:" + ("" if c.known_gaps else " none"))
        lines += [f"  - {g}" for g in c.known_gaps]
        if c.branch_point:
            bp = c.branch_point
            alts = "; ".join(bp.alternatives)
            lines.append(f"- Branch point: alternatives {alts}. Surface when: {bp.changes_conclusion_when}")
    if problems:
        lines += ["", "## Coverage problems", ""] + [f"- `{p.convention_id}`: {p.ref}: {p.reason}" for p in problems]
    return "\n".join(lines) + "\n"
