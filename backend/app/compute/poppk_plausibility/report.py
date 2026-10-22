"""Immutable report builder, summary counter, SHA-256 digest, and Markdown renderer.

Never uses 'VALID' or 'VALIDATED' terminology; always embeds the mandatory
physiological plausibility screening disclaimer.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from app.compute.poppk_plausibility.context import ResolvedPhysiologyContext
from app.compute.poppk_plausibility.schema import (
    NormalizedParameter,
    PlausibilityFinding,
    PopPKContext,
    ScreeningStatus,
    _FrozenModel,
)

MANDATORY_DISCLAIMER = (
    "Deterministic physiological plausibility screening only. This report does not "
    "constitute regulatory validation, clinical approval, or proof of model validity."
)


class PlausibilityReport(_FrozenModel):
    report_version: str = "1.0.0"
    catalog_version: str
    model_id: str
    screening_status: ScreeningStatus
    disclaimer: str = MANDATORY_DISCLAIMER
    context_summary: dict[str, Any]
    normalized_parameters: tuple[NormalizedParameter, ...]
    findings: tuple[PlausibilityFinding, ...]
    summary_counts: dict[str, int]
    report_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def to_markdown(self) -> str:
        lines: list[str] = [
            f"# PopPK Physiological Plausibility Screening Report: `{self.model_id}`",
            "",
            f"> **Disclaimer:** {self.disclaimer}",
            "",
            f"- **Screening Status:** `{self.screening_status}`",
            f"- **Catalog Version:** `{self.catalog_version}`",
            f"- **Report SHA-256:** `{self.report_sha256}`",
            "",
            "## Summary Counts",
            f"- PASS: {self.summary_counts.get('PASS', 0)}",
            f"- FLAGGED: {self.summary_counts.get('FLAGGED', 0)}",
            f"- UNRESOLVED: {self.summary_counts.get('UNRESOLVED', 0)}",
            f"- NOT_APPLICABLE: {self.summary_counts.get('NOT_APPLICABLE', 0)}",
            f"- Justification Required: {self.summary_counts.get('justification_required', 0)}",
            f"- Accepted With Justification: {self.summary_counts.get('accepted_with_justification', 0)}",
            f"- Override Rejected: {self.summary_counts.get('rejected', 0)}",
            "",
            "## Findings",
            "| Rule | Parameter | Status | Severity | Observed | Normalized | Expected | Override Status |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for f in self.findings:
            obs = f"{f.observed_value} {f.observed_unit or ''}".strip() if f.observed_value is not None else "-"
            norm = (
                f"{f.normalized_value:.4g} {f.normalized_unit or ''}".strip()
                if f.normalized_value is not None
                else "-"
            )
            lines.append(
                f"| `{f.rule_id}` | `{f.parameter}` | `{f.status}` | `{f.severity}` "
                f"| {obs} | {norm} | {f.expected_range or '-'} | `{f.override_status}` |"
            )
        justified = [f for f in self.findings if f.justification is not None]
        if justified:
            lines.extend(["", "## Documented Human Justifications"])
            for f in justified:
                j = f.justification
                if j is not None:
                    lines.append(
                        f"- **`{f.rule_id}` (`{f.parameter}`)** — *{f.override_status}* "
                        f"by {j.reviewer} ({j.timestamp}): {j.rationale} "
                        f"[Basis: {j.scientific_basis_or_citation}]"
                    )
        return "\n".join(lines)


def _compute_screening_status(findings: tuple[PlausibilityFinding, ...]) -> ScreeningStatus:
    for f in findings:
        is_crit = f.severity == "CRITICAL" or not f.override_allowed or f.override_status == "rejected"
        if f.status == "FLAGGED" and is_crit:
            return "CRITICAL_IMPLAUSIBILITY"
    if any(f.override_status == "justification_required" for f in findings):
        return "REQUIRES_JUSTIFICATION"
    if any(f.status == "UNRESOLVED" for f in findings):
        return "INCOMPLETE_CONTEXT"
    if any(f.override_status == "accepted_with_justification" for f in findings):
        return "REVIEW_ACCEPTED_WITH_JUSTIFICATION"
    return "SCREENING_PASS"


def build_plausibility_report(
    *,
    catalog_version: str,
    model_id: str,
    context: PopPKContext,
    phys: ResolvedPhysiologyContext,
    normalized_parameters: tuple[NormalizedParameter, ...],
    findings: tuple[PlausibilityFinding, ...],
) -> PlausibilityReport:
    sorted_params = tuple(sorted(normalized_parameters, key=lambda p: (p.canonical_name, p.name)))
    sorted_findings = tuple(sorted(findings, key=lambda f: (f.rule_id, f.parameter)))

    counts: dict[str, int] = {
        "PASS": 0,
        "FLAGGED": 0,
        "UNRESOLVED": 0,
        "NOT_APPLICABLE": 0,
        "INFO": 0,
        "WARN": 0,
        "HIGH": 0,
        "CRITICAL": 0,
        "justification_required": 0,
        "accepted_with_justification": 0,
        "rejected": 0,
    }
    for f in sorted_findings:
        counts[f.status] = counts.get(f.status, 0) + 1
        counts[f.severity] = counts.get(f.severity, 0) + 1
        if f.override_status in counts:
            counts[f.override_status] += 1

    screening_status = _compute_screening_status(sorted_findings)
    context_summary = {
        "species": context.species,
        "population": context.population,
        "reference_weight_kg": context.reference_weight_kg,
        "route": context.route,
        "modality": context.modality,
        "physiology_resolvable": phys.is_resolvable,
        "unresolved_reason": phys.unresolved_reason,
        "scaled_hepatic_blood_flow_l_h": phys.hepatic_blood_flow_l_h,
        "scaled_cardiac_output_l_h": phys.cardiac_output_l_h,
        "scaled_gfr_l_h": phys.gfr_l_h,
        "scaled_renal_plasma_flow_l_h": phys.renal_plasma_flow_l_h,
    }

    canonical_payload = {
        "report_version": "1.0.0",
        "catalog_version": catalog_version,
        "model_id": model_id,
        "screening_status": screening_status,
        "disclaimer": MANDATORY_DISCLAIMER,
        "context_summary": context_summary,
        "normalized_parameters": [p.model_dump(mode="json") for p in sorted_params],
        "findings": [f.model_dump(mode="json") for f in sorted_findings],
        "summary_counts": counts,
    }
    digest = hashlib.sha256(
        json.dumps(canonical_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    return PlausibilityReport(
        report_version="1.0.0",
        catalog_version=catalog_version,
        model_id=model_id,
        screening_status=screening_status,
        disclaimer=MANDATORY_DISCLAIMER,
        context_summary=context_summary,
        normalized_parameters=sorted_params,
        findings=sorted_findings,
        summary_counts=counts,
        report_sha256=digest,
    )
