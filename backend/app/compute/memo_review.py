"""Reviewer check for the briefing memo (called from app.compute.adversarial.review).

The memo stored in ``state.memo_results`` is re-checked against a value table
re-derived from the CURRENT state, two ways:

* ``memo-untraced-numbers``: a number in the memo text that no current state value
  produces (an edited or invented figure) -- the same membership check the build
  tool ran, now against fresh state;
* ``memo-stale``: a (state path, rendered value) pair the memo was built from that
  the current state no longer produces (an analysis re-run after the memo). Pure
  membership cannot see this when the old figure coincides with another value, so
  the stored manifest is compared pair by pair.
* ``memo-dataset-changed``: the memo was built while another dataset was loaded
  (its results describe that dataset, not the current one).

The fresh table is built without an audit trail on purpose: value text does not
depend on tags, and the reviewer has no business trusting a stored table for the
membership check.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from typing import Any

from app.compute.memo import AuditTrace, memo_untraced_numbers
from app.compute.memo_build import build_memo
from app.core.pharmstate import PharmState

MAX_SHOWN = 8
FindingFactory = Callable[..., dict[str, Any]]


def memo_present(state_dump: dict[str, Any]) -> bool:
    memo = state_dump.get("memo_results")
    return isinstance(memo, dict) and memo.get("status") == "ok" and isinstance(memo.get("memo_text"), str)


def memo_findings(state_dump: dict[str, Any], make_finding: FindingFactory) -> list[dict[str, Any]]:
    """Findings for ``state.memo_results``; empty when there is no memo or it is fully traced."""
    if not memo_present(state_dump):
        return []
    try:
        table = build_memo(PharmState.model_validate(state_dump), AuditTrace()).values
    except Exception:   # fail closed: any render failure IS the finding
        return [make_finding(
            "memo-unverifiable", "HIGH", "briefing memo",
            "a briefing memo is present but its numbers could not be re-derived from state",
            "the current analysis state could not be re-rendered through the memo template",
            "rebuild the memo (build_briefing_memo) from the current results")]
    memo = state_dump["memo_results"]
    findings: list[dict[str, Any]] = []
    if "dataset_id" in memo and memo.get("dataset_id") != state_dump.get("dataset_id"):
        findings.append(make_finding(
            "memo-dataset-changed", "HIGH", "briefing memo",
            "the briefing memo was built on a different dataset than the one now loaded",
            f"memo dataset {memo.get('dataset_id')!r}, current dataset {state_dump.get('dataset_id')!r}",
            "re-run the analyses on the current dataset and rebuild the memo with build_briefing_memo"))
    untraced = memo_untraced_numbers(memo["memo_text"], table)
    if untraced:
        findings.append(make_finding(
            "memo-untraced-numbers", "HIGH", "briefing memo",
            f"the briefing memo prints {len(untraced)} number(s) that no current state value produces",
            f"not traceable to analysis state: {_shown(untraced)}",
            "the memo was edited, or an analysis was re-run after it was built; "
            "rebuild it with build_briefing_memo and never edit its numbers by hand"))
    drifted = _drifted(memo.get("value_table"), table)
    if drifted:
        findings.append(make_finding(
            "memo-stale", "HIGH", "briefing memo",
            f"{len(drifted)} value(s) in the briefing memo no longer match the analysis state",
            "built from, but no longer produced by, current state: "
            + _shown([f"{key}={text}" for key, text in drifted]),
            "an analysis was re-run after the memo was built; re-run build_briefing_memo"))
    return findings


def _shown(items: list[str]) -> str:
    return ", ".join(items[:MAX_SHOWN]) + (" ..." if len(items) > MAX_SHOWN else "")


def _drifted(stored: Any, current: Any) -> list[tuple[str, str]]:
    """Stored (key, text) pairs missing from the current table, counted as a multiset."""
    if not isinstance(stored, list):
        return []
    available = Counter((v.key, v.text) for v in current)
    missing: list[tuple[str, str]] = []
    for entry in stored:
        pair = (entry.get("key"), entry.get("text")) if isinstance(entry, dict) else None
        if pair is None:
            continue
        if available[pair] > 0:
            available[pair] -= 1
        else:
            missing.append(pair)
    return missing
