"""Pure evaluation pipeline and non-destructive human justification engine."""
from __future__ import annotations

from typing import Any

from app.compute.poppk_plausibility.context import resolve_physiology_context
from app.compute.poppk_plausibility.report import PlausibilityReport, build_plausibility_report
from app.compute.poppk_plausibility.rules import evaluate_all_rules
from app.compute.poppk_plausibility.schema import (
    JustificationRecord,
    PlausibilityFinding,
    PopPKModelInput,
    load_rule_catalog,
)
from app.compute.poppk_plausibility.units import classify_parameter, normalize_parameter_spec


def apply_justifications(
    findings: tuple[PlausibilityFinding, ...],
    justifications: tuple[JustificationRecord, ...],
) -> tuple[PlausibilityFinding, ...]:
    """Attach documented human justifications to findings without deleting or hiding any finding."""
    just_map: dict[tuple[str, str], JustificationRecord] = {}
    for j in justifications:
        canon_p, _, _ = classify_parameter(j.parameter)
        just_map[(j.rule_id, j.parameter.strip().lower())] = j
        just_map[(j.rule_id, canon_p.lower())] = j

    updated: list[PlausibilityFinding] = []
    for f in findings:
        if f.status != "FLAGGED":
            updated.append(f.model_copy(update={"override_status": "none"}))
            continue

        canon_f, _, _ = classify_parameter(f.parameter)
        j = (
            just_map.get((f.rule_id, f.parameter.strip().lower()))
            or just_map.get((f.rule_id, canon_f.lower()))
        )
        if j is not None:
            if not f.override_allowed or f.severity == "CRITICAL":
                updated.append(
                    f.model_copy(
                        update={
                            "override_status": "rejected",
                            "justification": j,
                            "override_note": (
                                "Hard mathematical or physical impossibility cannot be overridden."
                            ),
                        }
                    )
                )
            else:
                updated.append(
                    f.model_copy(
                        update={
                            "override_status": "accepted_with_justification",
                            "justification": j,
                            "override_note": (
                                "Human scientific justification recorded; original finding preserved."
                            ),
                        }
                    )
                )
        else:
            status = "justification_required" if f.override_allowed else "none"
            updated.append(f.model_copy(update={"override_status": status}))

    return tuple(updated)


def evaluate_poppk_plausibility(
    model_input: PopPKModelInput | dict[str, Any],
) -> PlausibilityReport:
    """Evaluate a PopPK model specification against all 20 deterministic plausibility rules.

    Pure function: never mutates `model_input`, performs no network/LLM calls,
    and returns an immutable `PlausibilityReport`.
    """
    model = (
        model_input
        if isinstance(model_input, PopPKModelInput)
        else PopPKModelInput.model_validate(model_input)
    )
    catalog = load_rule_catalog()

    normalized_params = tuple(
        sorted(
            (normalize_parameter_spec(p, route=model.context.route) for p in model.parameters),
            key=lambda np_item: (np_item.canonical_name, np_item.name),
        )
    )
    phys = resolve_physiology_context(model.context, catalog.reference_physiology)
    raw_findings = evaluate_all_rules(catalog, model, normalized_params, phys)
    final_findings = apply_justifications(raw_findings, model.justifications)

    return build_plausibility_report(
        catalog_version=catalog.catalog_version,
        model_id=model.model_id,
        context=model.context,
        phys=phys,
        normalized_parameters=normalized_params,
        findings=final_findings,
    )
