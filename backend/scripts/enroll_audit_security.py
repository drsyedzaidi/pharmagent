#!/usr/bin/env python3
"""Explicitly enroll legacy audit rows into keyed external sealing.

Run from the repository root with ``PYTHONPATH=backend``. The default is a
read-only inventory. ``--apply --operator ... --yes`` is required to attest the
current legacy snapshots.
"""
from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime

from app.config import settings
from app.core.audit import AuditChain
from app.core.audit_seal import (
    AuditSecurityError,
    audit_security_from_settings,
    enroll_legacy_session,
)
from app.core.store import SessionStore


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inventory or explicitly seal legacy PharmAgent audit rows"
    )
    parser.add_argument("--session", action="append", default=[],
                        help="session id to process (repeatable; default: all)")
    parser.add_argument("--apply", action="store_true",
                        help="write seals and advance the external anchor")
    parser.add_argument("--operator", help="operator identity recorded in migration entry")
    parser.add_argument("--yes", action="store_true",
                        help="confirm the current legacy snapshots are accepted baselines")
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    try:
        security = audit_security_from_settings(settings)
    except AuditSecurityError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    if security is None:
        print("PHARMAGENT_AUDIT_MODE must be enforce", file=sys.stderr)
        return 2
    if args.apply and (not args.operator or not args.yes):
        print("--apply requires --operator and --yes", file=sys.stderr)
        return 2

    store = SessionStore(str(settings.db_path))
    rows = store.load_all()
    selected = set(args.session)
    if selected:
        missing = selected - {row["id"] for row in rows}
        if missing:
            print(f"unknown sessions: {', '.join(sorted(missing))}", file=sys.stderr)
            return 2
        rows = [row for row in rows if row["id"] in selected]

    failures = 0
    for row in rows:
        chain_status = AuditChain.from_list(row["audit"]).verify_status()
        if row["audit_seal"] is not None:
            label = "already_sealed"
        elif not chain_status["ok"]:
            label = "corrupt_unenrollable"
            failures += 1
        else:
            label = "legacy_ready"
        print(
            f"{row['id']} {label} entries={len(row['audit'])} "
            f"legacy_v1={chain_status['legacy_entries']}"
        )
        if args.apply and label == "legacy_ready":
            try:
                status = enroll_legacy_session(
                    store,
                    security,
                    row["id"],
                    operator=args.operator,
                    timestamp=datetime.now(UTC).isoformat(),
                )
            except AuditSecurityError as exc:
                print(f"{row['id']} enrollment failed: {exc}", file=sys.stderr)
                failures += 1
            else:
                print(
                    f"{row['id']} enrolled generation={status['generation']} "
                    f"trusted_since_index={status['trusted_since_index']}"
                )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
