# Decision: gate enforcement for unresolved adversarial findings

## Current behaviour

`resume_workflow` in `backend/app/core/orchestrator.py` (line 755) appends a
signed `human_review` audit entry with `actor` and `reason`, then advances the
workflow.  It does **not** inspect `review_results.findings` before approving.
An unresolved HIGH or CRITICAL adversarial finding does not prevent the gate
from being approved.

The report now renders these findings with the actual gate decision from the
audit trail (actor, timestamp, reason) rather than falsely claiming they
"block" the gate.

## Options

### Option A: refuse approval while unresolved HIGH/CRITICAL findings exist

Modify `resume_workflow` to check `review_results.findings` before appending
the approval audit entry.  If any HIGH or CRITICAL finding has neither
`resolved: true` nor a `disposition`, reject the approval unless the caller
provides an explicit `override_reason`.  Record the override in the audit
entry.

**Pro:** Findings have enforceable teeth.
**Con:** Requires changing gate behaviour in `orchestrator.py`, which is a
guarded file (CLAUDE.md line 16).  Must be done by the owner.

### Option B: keep current behaviour, rely on the report disclosure

The gate remains a human decision point.  The report section now transparently
shows "Unresolved; gate approved by <actor> on <timestamp>: <reason>" so
reviewers and regulators see what happened.  No code change to the
orchestrator.

**Pro:** No risk of breaking the gate contract.
**Con:** Enforcement is advisory only; a human can approve without addressing
findings and the system will not stop them.

## Recommendation

This is the owner's decision.  Neither option is implemented in this PR.
