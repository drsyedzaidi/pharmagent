"""Owner CLI for the rule catalog (``python -m app.knowledge rules ...``).

    rules add --id mb.nlme.high_shrinkage_ebe_screening --title "..." --severity MEDIUM \\
        --when "nlme_results.shrinkage_pct any gt 30" --target "..." --claim "..." --rationale "..." \\
        --action "..." --chunk 7cefffafb03f:2 [--when ...] [--chunk ...] [--limit "..."]
    rules list
    rules verify

``add`` is the only writer. It always writes ``pending_owner_review`` with no approver and
refuses to do anything else; approving a rule is the owner editing ``approval`` in a reviewed
change. The predicate is built from ``--when`` flags against a closed vocabulary — never
generated from free text.
"""
from __future__ import annotations

import argparse
import getpass
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.knowledge.manifest import ManifestError
from app.knowledge.rules import catalog_io as cio
from app.knowledge.rules.evidence import PRESENT, EvidenceError, evidence_for_chunks, verify_ref
from app.knowledge.rules.schema import ALLOWED_PATHS, RULE_SEVERITIES, Condition, Rule

RULE_VERSION = "1.0.0"


def parse_when(text: str) -> Condition:
    """``"PATH [any|all] OP VALUE"`` -> Condition (the schema rejects anything outside the vocabulary)."""
    parts = text.split()
    if len(parts) == 4:
        path, quantifier, op, raw = parts
    elif len(parts) == 3:
        (path, op, raw), quantifier = parts, None
    else:
        raise ValueError(f"--when {text!r}: expected 'PATH [any|all] OP VALUE', e.g. "
                         "'nlme_results.shrinkage_pct any gt 30'")
    try:
        value: Any = json.loads(raw)
    except ValueError:
        value = raw
    return Condition(path=path, op=op, value=value, quantifier=quantifier)


def register(sub: argparse._SubParsersAction) -> None:
    r = sub.add_parser("rules", help="owner-approved rule catalog (add is always pending_owner_review)")
    r.add_argument("--catalog", type=Path, default=None, help="catalog file (default: the shipped catalog)")
    rs = r.add_subparsers(dest="rules_cmd", required=True)
    a = rs.add_parser("add", help="propose a rule: pending_owner_review, evidence bound by hash")
    a.add_argument("--id", required=True)
    a.add_argument("--title", required=True)
    a.add_argument("--severity", choices=RULE_SEVERITIES, default="MEDIUM")
    a.add_argument("--when", action="append", required=True, metavar="'PATH [any|all] OP VALUE'",
                   help=f"condition; paths: {', '.join(sorted(ALLOWED_PATHS))}")
    a.add_argument("--target", required=True)
    a.add_argument("--claim", required=True)
    a.add_argument("--rationale", required=True)
    a.add_argument("--action", required=True, dest="suggested_action")
    a.add_argument("--chunk", action="append", required=True, help="chunk id from `search` (repeatable)")
    a.add_argument("--limit", action="append", default=[], dest="known_limits", help="known limit (repeatable)")
    a.add_argument("--by", dest="proposed_by", default=getpass.getuser())
    rs.add_parser("list", help="rules, approval status and evidence counts")
    rs.add_parser("verify", help="re-hash every evidence span against the stored corpus; exit 1 on any problem")


def _cmd_add(args: argparse.Namespace, root: Path) -> int:
    rule = Rule(
        id=args.id, version=RULE_VERSION, title=args.title, when=[parse_when(w) for w in args.when],
        severity=args.severity, target=args.target, claim=args.claim, rationale=args.rationale,
        suggested_action=args.suggested_action,
        evidence=evidence_for_chunks(root, args.chunk, own_words={
            "title": args.title, "target": args.target, "claim": args.claim, "rationale": args.rationale,
            "action": args.suggested_action, **{f"limit[{i}]": t for i, t in enumerate(args.known_limits)}}),
        known_limits=args.known_limits, proposed_by=args.proposed_by,
        proposed_on=datetime.now(UTC).date().isoformat(), approval={"status": "pending_owner_review"})
    record = cio.append_pending_rule(rule, args.catalog)
    print(json.dumps({"added": rule.id, "approval": "pending_owner_review",
                      "catalog_version": record.catalog.catalog_version, "catalog_sha256": record.catalog_sha256,
                      "next": "review the entry, then record approval.by and approval.date in a reviewed change"},
                     indent=1))
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    rec = cio.load_rule_catalog(args.catalog)
    print(json.dumps({"catalog_version": rec.catalog.catalog_version, "catalog_sha256": rec.catalog_sha256,
                      "rules": [{"id": r.id, "version": r.version, "severity": r.severity,
                                 "approval": r.approval.status, "approved_by": r.approval.by,
                                 "approved_on": r.approval.date, "n_conditions": len(r.when),
                                 "n_evidence": len(r.evidence)} for r in rec.catalog.rules]}, indent=1))
    return 0


def _cmd_verify(args: argparse.Namespace, root: Path) -> int:
    rec = cio.load_rule_catalog(args.catalog)
    report = [{"id": r.id, "evidence": [{"document": e.document, "start": e.start, "end": e.end,
                                         "status": verify_ref(root, e)} for e in r.evidence]}
              for r in rec.catalog.rules]
    problems = sum(e["status"] != PRESENT for r in report for e in r["evidence"])
    print(json.dumps({"catalog_version": rec.catalog.catalog_version, "problems": problems, "rules": report}, indent=1))
    return 1 if problems else 0


def run(args: argparse.Namespace, root: Path) -> int:
    try:
        if args.rules_cmd == "add":
            return _cmd_add(args, root)
        if args.rules_cmd == "list":
            return _cmd_list(args)
        return _cmd_verify(args, root)
    except (ValueError, ValidationError, EvidenceError, ManifestError, OSError, OverflowError, RecursionError) as exc:
        print(f"error: {str(exc)[:600]}", file=sys.stderr)
        return 2
