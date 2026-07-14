# Loop Run Log — PharmAgent

Append one JSON line per run. Prune entries older than 30 days.

## Format
```json
{
  "run_id": "2026-07-13T23:32:00Z",
  "pattern": "security-triage",
  "duration_s": 0,
  "items_found": 8,
  "actions_taken": 0,
  "escalations": 0,
  "tokens_estimate": 0,
  "outcome": "report-only | fix-proposed | escalated | budget-exceeded | no-op"
}
```

## Recent Runs
<!-- Loop appends below this line -->
{"run_id":"2026-07-13T23:32:00Z","pattern":"security-triage","duration_s":0,"items_found":8,"actions_taken":0,"escalations":0,"tokens_estimate":0,"outcome":"report-only","note":"seeded from scan b8fc4338 (3 high, 5 medium)"}

{"run_id":"2026-07-14T00:24:00Z","pattern":"security-triage","duration_s":0,"items_found":8,"actions_taken":8,"escalations":0,"tokens_estimate":0,"outcome":"security-fixes-pushed","note":"scan b8fc4338 findings resolved in c9d2658"}
