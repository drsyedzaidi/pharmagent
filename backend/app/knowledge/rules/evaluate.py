"""Deterministic evaluation of approved rules over the analysis diagnostics.

Pure functions of ``(state_dump, catalog)``: no corpus, no network, no LLM, no clock. The
result of a review therefore does not depend on whether the owner's private documents are
present on the machine, and it reproduces on CI. A rule's evidence is a recorded pointer into
the corpus (see ``evidence.py``); it is checked by ``rules verify``, never at review time.

Only ``approved`` rules are evaluated. A pending rule is counted and ignored. A condition
whose value is missing, ``None`` or the wrong type simply does not hold, so absence of data
never produces a finding. Every produced finding is MEDIUM or LOW (enforced by the schema and
again by ``adversarial.review``) and therefore can never change ``goal_met``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from app.knowledge.rules.catalog_io import RuleCatalogRecord, cached_rule_catalog
from app.knowledge.rules.schema import ALLOWED_PATHS, ROOT_GATES, Condition, Rule, finite_float

log = logging.getLogger("pharmagent")

ORIGIN = "mikebrain_rule"
_OPS = {"gt": ">", "ge": ">=", "lt": "<", "le": "<=", "eq": "==", "ne": "!="}
_COMPARE = {
    "gt": lambda a, b: a > b, "ge": lambda a, b: a >= b, "lt": lambda a, b: a < b,
    "le": lambda a, b: a <= b, "eq": lambda a, b: a == b, "ne": lambda a, b: a != b,
}


@dataclass(frozen=True)
class RuleReport:
    findings: tuple[dict[str, Any], ...]
    info: dict[str, Any] | None      # None while no rule is approved: the review result is then untouched


def _lookup(state: dict[str, Any], path: str) -> Any:
    cur: Any = state
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _observe(cond: Condition, state: dict[str, Any]) -> str | None:
    """A short description of what held, or None when the condition does not hold."""
    kind, sym, value = ALLOWED_PATHS[cond.path], _OPS[cond.op], cond.value
    got = _lookup(state, cond.path)
    cmp = _COMPARE[cond.op]
    if kind == "num":
        n = finite_float(got)
        return f"{cond.path} = {n:.12g} ({sym} {float(value):.12g})" if n is not None and cmp(n, float(value)) else None
    if kind == "bool":
        return f"{cond.path} = {got}" if isinstance(got, bool) and cmp(got, value) else None
    if kind == "str":
        return f"{cond.path} = {got}" if isinstance(got, str) and cmp(got, value) else None
    if not isinstance(got, dict) or not got:     # map_num
        return None
    items = sorted(((str(k), finite_float(v)) for k, v in got.items()), key=lambda kv: kv[0])
    hits = [(k, n) for k, n in items if n is not None and cmp(n, float(value))]
    ok = bool(hits) if cond.quantifier == "any" else len(hits) == len(items)
    if not ok:
        return None
    shown = ", ".join(f"{k}={n:.12g}" for k, n in hits)
    return f"{cond.path} [{cond.quantifier}]: {shown} ({sym} {float(value):.12g})"


def _gates_open(rule: Rule, state: dict[str, Any]) -> bool:
    roots = {c.path.split(".", 1)[0] for c in rule.when}
    return all(_lookup(state, f"{root}.{ROOT_GATES[root][0]}") == ROOT_GATES[root][1] for root in roots)


def _finding(rule: Rule, observed: list[str], record: RuleCatalogRecord) -> dict[str, Any]:
    return {
        "id": rule.id, "severity": rule.severity, "target": rule.target, "claim": rule.claim,
        "evidence": f"{'; '.join(observed)} — {rule.rationale}",
        "suggested_action": rule.suggested_action, "resolved": False,
        "origin": ORIGIN, "rule_id": rule.id, "rule_version": rule.version,
        "catalog_version": record.catalog.catalog_version, "catalog_sha256": record.catalog_sha256,
        "approved_by": rule.approval.by, "approved_on": rule.approval.date,
        "literature": [{"document": e.document, "year": e.year, "doc_sha": e.doc_sha,
                        "text_sha256": e.text_sha256} for e in rule.evidence],
    }


def evaluate(state_dump: dict[str, Any], record: RuleCatalogRecord) -> RuleReport:
    """Findings for every approved rule whose conditions all hold, in catalog order."""
    rules = record.catalog.rules
    approved = [r for r in rules if r.approval.status == "approved"]
    if not approved:
        return RuleReport((), None)
    findings: list[dict[str, Any]] = []
    for rule in approved:
        if not _gates_open(rule, state_dump):
            continue
        observed = [_observe(c, state_dump) for c in rule.when]
        if all(o is not None for o in observed):
            findings.append(_finding(rule, [o for o in observed if o is not None], record))
    info = {
        "status": "ok", "catalog_version": record.catalog.catalog_version, "catalog_sha256": record.catalog_sha256,
        "n_approved": len(approved), "n_pending_ignored": len(rules) - len(approved),
        "n_fired": len(findings), "fired": [f["id"] for f in findings],
    }
    return RuleReport(tuple(findings), info)


def apply_default_catalog(state_dump: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Findings + info block from the process-wide catalog.

    Nothing about the catalog may take a review down, and nothing may pass silently: whatever goes wrong
    (a missing or unparseable file, a schema violation, an unexpected error while evaluating), rules are
    skipped and the info block says so. The client-facing ``error`` is generic; the detail, which can
    contain a filesystem path, goes to the server log only.
    """
    try:
        record = cached_rule_catalog()
    except Exception as exc:  # noqa: BLE001 - includes OverflowError/RecursionError from hostile files
        log.warning("mikebrain_rule_catalog_invalid", extra={"error": f"{type(exc).__name__}: {str(exc)[:300]}"})
        return [], {"status": "invalid", "error": f"rule catalog could not be loaded ({type(exc).__name__}); "
                                                  "run `python -m app.knowledge rules list` for details"}
    try:
        report = evaluate(state_dump, record)
    except Exception as exc:  # noqa: BLE001 - rules are supplementary; the deterministic review must still run
        log.warning("mikebrain_rule_evaluation_failed", extra={"error": f"{type(exc).__name__}: {str(exc)[:300]}"})
        return [], {"status": "error", "error": f"rule evaluation failed ({type(exc).__name__}); rules skipped",
                    "catalog_version": record.catalog.catalog_version, "catalog_sha256": record.catalog_sha256}
    return list(report.findings), report.info
