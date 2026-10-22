"""Pure evaluation pipeline and non-destructive human justification engine."""
from __future__ import annotations

from collections import Counter
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


def _base_param_key(name: str) -> str:
    return name.removesuffix(" (derived)").strip().lower()


def apply_justifications(
    findings: tuple[PlausibilityFinding, ...],
    justifications: tuple[JustificationRecord, ...],
) -> tuple[PlausibilityFinding, ...]:
    """Attach documented human justifications to findings without deleting or hiding any finding."""
    sorted_justs = sorted(
        justifications,
        key=lambda x: (x.rule_id, x.parameter.strip().lower(), x.reviewer, x.timestamp, x.rationale),
    )
    exact_map: dict[tuple[str, str], JustificationRecord] = {}
    canon_map: dict[tuple[str, str], JustificationRecord] = {}
    for j in sorted_justs:
        raw_k = _base_param_key(j.parameter)
        canon_p, _, _ = classify_parameter(raw_k)
        exact_map[(j.rule_id, raw_k)] = j
        canon_map[(j.rule_id, canon_p.lower())] = j

    flagged_canon_counts: Counter[tuple[str, str]] = Counter()
    for f in findings:
        if f.status == "FLAGGED":
            cf, _, _ = classify_parameter(_base_param_key(f.parameter))
            flagged_canon_counts[(f.rule_id, cf.lower())] += 1

    updated: list[PlausibilityFinding] = []
    for f in findings:
        if f.status != "FLAGGED":
            updated.append(f.model_copy(update={"override_status": "none"}))
            continue

        raw_f = f.parameter.strip().lower()
        base_f = _base_param_key(f.parameter)
        canon_f, _, _ = classify_parameter(base_f)
        j = exact_map.get((f.rule_id, raw_f)) or exact_map.get((f.rule_id, base_f))
        if j is None and flagged_canon_counts[(f.rule_id, canon_f.lower())] == 1:
            j = canon_map.get((f.rule_id, canon_f.lower()))

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
    has_explicit_f = any(
        classify_parameter(p.name)[0] == "F"
        and isinstance(p.value, (int, float))
        and not isinstance(p.value, bool)
        and p.value > 0
        for p in model.parameters
    )

    normalized_params = tuple(
        sorted(
            (
                normalize_parameter_spec(p, route=model.context.route, has_explicit_f=has_explicit_f)
                for p in model.parameters
            ),
            key=lambda np_item: (
                np_item.canonical_name,
                np_item.name,
                str(np_item.observed_value),
                str(np_item.observed_unit),
            ),
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
