"""mikebrain M2 — the owner-approved rule catalog.

A rule turns a piece of literature the owner has read into a DETERMINISTIC check
over the analysis diagnostics. Three things are never allowed to happen here:

* an LLM verdict becoming a rule (rules are authored by the owner through
  ``python -m app.knowledge rules add`` and enter as ``pending_owner_review``);
* a pending rule having any effect (only ``approved`` rules are evaluated);
* a rule-derived finding blocking a gate (severity is capped below HIGH).

See docs/mikebrain.md.
"""
